import torch
import random
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import copy
import os
from sklearn.model_selection import StratifiedShuffleSplit
from config import NODE_ID, DIRICHLET_ALPHA, DATASET_TYPE_MAP

try:
    from config import AKIDA_CIFAR_RATIO
except ImportError:
    AKIDA_CIFAR_RATIO = 0.5


# -------------------------------------------------------------------------
# Dataset caching: keep raw datasets in memory and apply transforms dynamically
# -------------------------------------------------------------------------
_cached_raw = {}  # e.g. {'CIFAR10_train_RAW': ds, 'CIFAR10_test_RAW': ds}


# -------------------------------------------------------------------------
# DataLoader worker seeding: ensures deterministic randomness across workers
# -------------------------------------------------------------------------
def _worker_init_fn(wid):
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + wid)
    random.seed(base_seed + wid)


# -------------------------------------------------------------------------
# Training transforms: augmentation + normalization per dataset
# -------------------------------------------------------------------------
def get_train_transform(dataset_type):
    if dataset_type == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_type == "CIFAR10":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# -------------------------------------------------------------------------
# Evaluation transforms: normalization only (no augmentation)
# -------------------------------------------------------------------------
def get_eval_transform(dataset_type):
    if dataset_type == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_type == "CIFAR10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# -------------------------------------------------------------------------
# Dataset loader with transform swapping via cached raw datasets
# -------------------------------------------------------------------------
def load_dataset(dataset_type, train=True, transform=None):
    root = './data'
    key = f"{dataset_type}_{'train' if train else 'test'}_RAW"

    if key not in _cached_raw:
        if dataset_type == "MNIST":
            raw = datasets.MNIST(root, train=train, download=True, transform=None)
        elif dataset_type == "CIFAR10":
            raw = datasets.CIFAR10(root, train=train, download=True, transform=None)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_type}")
        _cached_raw[key] = raw

    ds = copy.copy(_cached_raw[key])
    ds.transform = transform
    return ds


# -------------------------------------------------------------------------
# Validation loader: fixed-size reproducible validation subset
# -------------------------------------------------------------------------
def get_validation_loader(batch_size=64, device=torch.device('cpu')):
    target_node = NODE_ID
    dataset_type = DATASET_TYPE_MAP[target_node]

    # Node 7 input convention is grayscale 28x28. If it ever validates on CIFAR10,
    # this transform converts RGB -> grayscale and resizes to match the model input.
    if NODE_ID == 7 and dataset_type == "CIFAR10":
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        t = get_eval_transform(dataset_type)

    ds = load_dataset(dataset_type, train=False, transform=t)

    # Deterministic selection of validation subset per node
    np.random.seed(1337 + target_node)

    if hasattr(ds, 'targets'):
        targets = np.array(ds.targets)
    elif hasattr(ds, 'labels'):
        targets = np.array(ds.labels)
    else:
        try:
            targets = np.array([ds[i][1] for i in range(len(ds))])
        except Exception:
            targets = np.arange(len(ds))

    # Prefer stratified sampling; fall back to random sampling if stratification fails
    if len(np.unique(targets)) == 1 and len(targets) > 0:
        idx = np.random.permutation(len(ds))[:min(2000, len(ds))]
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=min(2000, len(targets)),
            random_state=1337 + target_node
        )
        try:
            _, idx = next(sss.split(np.zeros(len(targets)), targets))
        except ValueError:
            idx = np.random.permutation(len(ds))[:min(2000, len(ds))]

    subset = Subset(ds, idx)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )


# -------------------------------------------------------------------------
# Warm-up loader: MNIST training loader used for Node 7 pretraining
# -------------------------------------------------------------------------
def get_mnist_loader_for_warmup(batch_size=128, device=torch.device('cpu')):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = load_dataset("MNIST", train=True, transform=t)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn
    )


# -------------------------------------------------------------------------
# Dirichlet non-IID split + minimum-per-client rebalancing
# -------------------------------------------------------------------------
def dirichlet_split_noniid(dataset, n_clients, alpha, min_per_client=400, rng_seed=42):
    """
    Splits samples across clients using a per-class Dirichlet distribution.
    After splitting, rebalances by moving samples from large clients to smaller
    clients until every client has at least min_per_client samples.
    """
    n_classes = len(dataset.classes)
    labels = np.array(dataset.targets)
    idx_by_class = {c: np.where(labels == c)[0] for c in range(n_classes)}
    rng = np.random.RandomState(rng_seed)

    # Step 1: Dirichlet per-class allocation
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        class_idx = idx_by_class[c].copy()
        rng.shuffle(class_idx)

        p = rng.dirichlet(np.repeat(alpha, n_clients))
        expected = p * len(class_idx)

        counts = np.floor(expected).astype(int)
        remainder = len(class_idx) - counts.sum()
        if remainder > 0:
            frac = expected - counts
            for j in np.argsort(-frac)[:remainder]:
                counts[j] += 1

        ptr = 0
        for i in range(n_clients):
            k = counts[i]
            if k > 0:
                client_indices[i].extend(class_idx[ptr:ptr + k].tolist())
                ptr += k

    # Step 2: Post-hoc rebalance to enforce min_per_client
    sizes = np.array([len(lst) for lst in client_indices], dtype=int)
    needers = [i for i in range(n_clients) if sizes[i] < min_per_client]
    donors = [i for i in range(n_clients) if sizes[i] > min_per_client]
    donors.sort(key=lambda i: sizes[i], reverse=True)

    for i in needers:
        deficit = int(min_per_client - sizes[i])
        if deficit <= 0:
            continue

        d_ptr = 0
        while deficit > 0 and d_ptr < len(donors):
            j = donors[d_ptr]
            spare = int(sizes[j] - min_per_client)

            if spare > 0:
                give = min(deficit, spare)
                donor_arr = np.array(client_indices[j], dtype=int)

                sel = rng.choice(len(donor_arr), size=give, replace=False)
                move = donor_arr[sel]

                keep_mask = np.ones(len(donor_arr), dtype=bool)
                keep_mask[sel] = False

                client_indices[j] = donor_arr[keep_mask].tolist()
                client_indices[i].extend(move.tolist())

                sizes[j] -= give
                sizes[i] += give
                deficit -= give
            else:
                d_ptr += 1

        donors = [d for d in donors if sizes[d] > min_per_client]
        donors.sort(key=lambda k: sizes[k], reverse=True)

    return {i: np.array(lst, dtype=int) for i, lst in enumerate(client_indices)}


# -------------------------------------------------------------------------
# Training data loader: Node 7 hybrid dataset, others CIFAR10 Dirichlet split
# -------------------------------------------------------------------------
def get_data(batch_size=128, seed=None, device=torch.device('cpu')):
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)

    # Node 7: hybrid dataset (MNIST + grayscale CIFAR subset resized to 28x28)
    if NODE_ID == 7:
        batch_size = min(batch_size, 64)

        mnist_t = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        cifar_t = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        mnist = load_dataset("MNIST", train=True, transform=mnist_t)
        cifar = load_dataset("CIFAR10", train=True, transform=cifar_t)

        total_cifar = min(int(AKIDA_CIFAR_RATIO * 1.2 * len(cifar)), len(cifar))

        # Deterministic selection of CIFAR subset for Node 7
        np.random.seed(NODE_ID)
        cifar_idx = np.random.choice(len(cifar), size=total_cifar, replace=False)

        combo = ConcatDataset([mnist, Subset(cifar, cifar_idx)])
        print(f"[Node 7] Loaded {len(mnist)} MNIST + {total_cifar} CIFAR10 grayscale samples.", flush=True)

        return DataLoader(
            combo,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            worker_init_fn=_worker_init_fn,
            generator=gen
        )

    # CIFAR nodes: CIFAR10 + deterministic non-IID split persisted to disk
    dataset_type = "CIFAR10"
    t = get_train_transform(dataset_type)
    ds = load_dataset(dataset_type, train=True, transform=t)

    cifar_nodes = sorted([nid for nid, dtype in DATASET_TYPE_MAP.items() if dtype == "CIFAR10"])
    if NODE_ID not in cifar_nodes:
        raise ValueError(f"Node {NODE_ID} is not part of CIFAR10 client list.")

    nid2idx = {nid: i for i, nid in enumerate(cifar_nodes)}
    local_id = nid2idx[NODE_ID]
    n_clients = len(cifar_nodes)

    # Minimum-per-client is derived from dataset size and number of clients
    avg_share = len(ds) // max(1, n_clients)
    min_per = max(6000, int(0.6 * avg_share))

    # Deterministic split persistence ensures all nodes reproduce the same partition
    os.makedirs("splits", exist_ok=True)
    nodes_tag = "-".join(map(str, cifar_nodes))
    split_tag = f"CIFAR10_dirichlet_a{DIRICHLET_ALPHA}_clients{n_clients}_nodes{nodes_tag}_seed42.npy"
    split_path = os.path.join("splits", split_tag)

    if os.path.exists(split_path):
        client_map = np.load(split_path, allow_pickle=True).item()
    else:
        client_map = dirichlet_split_noniid(
            ds,
            n_clients=n_clients,
            alpha=DIRICHLET_ALPHA,
            min_per_client=min_per,
            rng_seed=42
        )
        np.save(split_path, client_map)

    indices = client_map.get(local_id, [])
    if len(indices) == 0:
        raise ValueError(
            f"No samples for Node {NODE_ID} (client {local_id}). "
            f"Lower alpha or min_per, or fewer clients."
        )

    print(f"[Node {NODE_ID}] Loaded {len(indices)} CIFAR10 samples.", flush=True)

    subset = Subset(ds, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn,
        generator=gen
    )
