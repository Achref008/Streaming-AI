import torch
import torch.nn as nn
import numpy as np
from config import NODE_ID


class MyModel(nn.Module):
    """
    CNN model used by heterogeneous DFL nodes.
    The architecture is dynamically adapted depending on the node type.
    NODE_ID == 7 represents the Akida neuromorphic node (MNIST / 1-channel input).
    All other nodes represent Jetson CNN nodes (CIFAR-10 / 3-channel input).
    """

    def __init__(self, input_channels=1):
        super(MyModel, self).__init__()
        self.input_channels = input_channels

        # Dataset-dependent input adaptation
        if NODE_ID == 7:
            image_size = 28
            self.adapter = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=1),
                nn.ReLU()
            )
            feature_channels = 4
        else:
            image_size = 32
            self.adapter = nn.Identity()
            feature_channels = input_channels

        # Shared convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(feature_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Automatic computation of classifier input dimension
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, image_size, image_size)
            dummy = self.adapter(dummy)
            dummy = self.feature_extractor(dummy)
            flattened_size = dummy.view(1, -1).size(1)

        # Dataset-specific classifier heads
        if NODE_ID == 7:
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

    def forward(self, x):
        """
        Normalizes channel count across heterogeneous datasets before inference.
        """
        if x.size(1) != self.input_channels:
            if self.input_channels == 1:
                x = x.mean(dim=1, keepdim=True)
            elif self.input_channels == 3:
                x = x.repeat(1, 3, 1, 1)

        x = self.adapter(x)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Heterogeneous delta adaptation logic (DFL cross-architecture exchange)
    # ------------------------------------------------------------------

    def convert_weights(self, incoming_deltas):
        """
        Adapts received parameter deltas from heterogeneous peers to the local
        model structure by projecting mismatched tensors into valid shapes.
        """

        adapted = {}
        state = self.state_dict()
        device = next(self.parameters()).device

        direct, projected = 0, 0

        for name, param in state.items():
            if name not in incoming_deltas:
                adapted[name] = torch.zeros_like(param, device=device)
                continue

            delta = incoming_deltas[name].to(device)
            target_shape = param.shape

            if delta.shape == target_shape:
                adapted[name] = delta
                direct += 1

            elif delta.dim() == param.dim() and delta.dtype.is_floating_point:
                projected_tensor = MyModel._project_tensor_general(delta, target_shape)
                if projected_tensor is not None:
                    adapted[name] = projected_tensor
                    projected += 1
                else:
                    adapted[name] = torch.zeros_like(param, device=device)
            else:
                adapted[name] = torch.zeros_like(param, device=device)

        print(f"[Akida] Delta adaptation completed: {direct} direct, {projected} projected")
        return adapted

    # ------------------------------------------------------------------

    @staticmethod
    def _project_tensor_general(tensor, target_shape):
        """
        Projects an arbitrary tensor into a compatible parameter shape by
        cropping or zero-padding while preserving existing values.
        """
        try:
            out = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)

            if len(tensor.shape) == 4:  # Conv2d
                oc = min(tensor.shape[0], target_shape[0])
                ic = min(tensor.shape[1], target_shape[1])
                h = min(tensor.shape[2], target_shape[2])
                w = min(tensor.shape[3], target_shape[3])
                out[:oc, :ic, :h, :w] = tensor[:oc, :ic, :h, :w]

            elif len(tensor.shape) == 2:  # Linear
                o = min(tensor.shape[0], target_shape[0])
                i = min(tensor.shape[1], target_shape[1])
                out[:o, :i] = tensor[:o, :i]

            else:  # Generic tensors (bias, BN, etc.)
                flat = tensor.view(-1)
                total = np.prod(target_shape)
                if flat.numel() >= total:
                    out.view(-1)[:] = flat[:total]
                else:
                    out.view(-1)[:flat.numel()] = flat

            return torch.nan_to_num(out, nan=0.0, posinf=1e5, neginf=-1e5)

        except Exception:
            return None
