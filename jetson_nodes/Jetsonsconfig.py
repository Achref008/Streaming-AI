# -------------------------------------------------------------------------
# Node identity and deployment size
# -------------------------------------------------------------------------
# NODE_ID selects which device this config is for.
# TOTAL_NODES reflects the size of the experiment (4 Jetsons + 1 Akida).
NODE_ID = 1
TOTAL_NODES = 5


# -------------------------------------------------------------------------
# Network topology: neighbor lists for decentralized communication
# -------------------------------------------------------------------------
# Each node exchanges updates with the nodes listed in its NEIGHBORS entry.
# This topology is fully connected between Jetsons (1–4) and includes Akida (7)
# as a neighbor of every Jetson, and vice-versa.
NEIGHBORS = {
    1: [2, 3, 4, 7],
    2: [1, 3, 4, 7],
    3: [1, 2, 4, 7],
    4: [1, 2, 3, 7],
    7: [1, 2, 3, 4]
}


# -------------------------------------------------------------------------
# Cross-node validation policy
# -------------------------------------------------------------------------
# PEER_VALIDATION_MAP defines which node's dataset is used for peer validation.
# For example, NODE_ID=1 validates on node 2's dataset split.
PEER_VALIDATION_MAP = {
    1: 2,
    2: 3,
    3: 1,
    4: 2,
    7: 1
}


# -------------------------------------------------------------------------
# Static IP assignment for each node in the testbed
# -------------------------------------------------------------------------
# Used for socket connections (comms.py) when sending/receiving deltas.
IP_MAP = {
    1: "10.0.1.2",
    2: "10.0.2.2",
    3: "10.0.3.2",
    4: "10.0.4.2",
    7: "10.0.7.2"
}


# -------------------------------------------------------------------------
# Dataset mapping by node
# -------------------------------------------------------------------------
# Jetson nodes (1–4) train on CIFAR-10.
# Akida node (7) trains on MNIST (and may optionally include grayscale CIFAR
# via AKIDA_CIFAR_RATIO depending on the data pipeline implementation).
DATASET_TYPE_MAP = {
    1: "CIFAR10",   # Jetson 1
    2: "CIFAR10",   # Jetson 2
    3: "CIFAR10",   # Jetson 3
    4: "CIFAR10",   # Jetson 4
    7: "MNIST"      # Akida
}


# -------------------------------------------------------------------------
# Non-IID control for CIFAR-10 client splits
# -------------------------------------------------------------------------
# DIRICHLET_ALPHA controls how non-IID the Dirichlet partition is:
# lower alpha -> more skewed label distribution per node.
DIRICHLET_ALPHA = 1.0


# -------------------------------------------------------------------------
# Central node settings (optional): used to collect results/logs
# -------------------------------------------------------------------------
# These parameters are only relevant if you use a central machine to aggregate
# logs, plots, or checkpoints from the nodes.
CENTRAL_NODE_HOST = "10.1.1.1"
CENTRAL_NODE_USER = "sai"
CENTRAL_RESULTS_DIR = "/home/sai/Desktop/achref/DFLtest1/central_node"


# -------------------------------------------------------------------------
# Socket communication base port
# -------------------------------------------------------------------------
# Each node listens on (PORT_BASE + NODE_ID).
PORT_BASE = 5000


# -------------------------------------------------------------------------
# Akida hybrid training configuration
# -------------------------------------------------------------------------
# AKIDA_CIFAR_RATIO controls how much CIFAR-10 (converted to grayscale and resized)
# is mixed into Node 7 training, if your data_utils implements the hybrid dataset.
AKIDA_CIFAR_RATIO = 0.9


# -------------------------------------------------------------------------
# Training schedule and optimizer settings
# -------------------------------------------------------------------------
# TAU1: number of local update steps/epochs per round.
# ROUNDS: total number of communication rounds.
# LEARNING_RATE: applied when updating parameters after aggregation/momentum.
TAU1 = 20
ROUNDS = 100
LEARNING_RATE = 1e-3
OPTIMIZER_CHOICE = "adam"


# -------------------------------------------------------------------------
# Runtime behavior
# -------------------------------------------------------------------------
# GRACEFUL_ON_SIGINT controls whether the node attempts to log/exit cleanly
# when interrupted (Ctrl+C).
GRACEFUL_ON_SIGINT = True
