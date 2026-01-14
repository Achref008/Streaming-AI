# Decentralized Federated Learning on Heterogeneous Edge Devices (Jetson + Akida)

This repository contains a **peer-to-peer decentralized federated learning (DFL)** framework designed for **heterogeneous edge AI** setups:
- **NVIDIA Jetson** nodes (GPU CNN training, CIFAR-10)
- **BrainChip Akida** node (neuromorphic / low-power client, MNIST + grayscale CIFAR-10)

Unlike classic Federated Learning, this system runs **without a central server**. Nodes communicate directly and exchange **differential model updates (deltas)**. Aggregation is performed using **Metropolis-Hastings consensus** with stability mechanisms (e.g., momentum-style smoothing).

## Key Features
- Fully decentralized training (peer-to-peer)
- Robust to intermittent connectivity / node dropouts
- Non-IID data support (Dirichlet split)
- Cross-architecture update adaptation (projection for mismatched shapes)
- Lightweight communication: compressed delta exchange
- Experiment logging and validation workflows

## Repository Structure
- `jetson_nodes/` : Jetson client-side scripts (communication + data utilities)
- `akida_node/`   : Akida client scripts (model + config + training loop)
- `docs/`         : network/IP notes and device applicability documentation

## Quick Start (high level)
1. Configure node IDs, neighbors, IP map, dataset mapping inside the config.
2. Start each node on its device (Jetson nodes + Akida node).
3. Each node trains locally and exchanges deltas with its neighbors.

See device-specific instructions:
- `jetson_nodes/README.md`
- `akida_node/README.md`

## Notes
- This is a research/PhD-oriented prototype for decentralized learning on heterogeneous hardware.
- Some scripts assume a local network with fixed IP addresses.
