# Akida Node (Neuromorphic / Low-Power Client)

The Akida node participates in decentralized learning as a heterogeneous client:
- Uses MNIST and/or grayscale CIFAR-10 paths
- Can receive deltas from Jetson CNN nodes
- Adapts incoming deltas via **projection** when shapes mismatch

## Files
- `akidamain.py`     : training loop + communication + aggregation orchestration
- `Akidamodel.py`    : model definition + delta projection / conversion utilities
- `Akidaconfig.py`   : node configuration (neighbors, IP map, datasets, hyperparams)

## Typical Devices
- BrainChip Akida (AKD1000) Dev Kits
- Raspberry Pi + Akida (PCIe / add-on)
- Low-power embedded gateways that host Akida runtime

## Run
```bash
python3 akidamain.py
