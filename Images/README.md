# Experimental Results

## 1. Loss under Disconnections (5 Heterogeneous Nodes)

![Loss](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/5nodes%20loss%20average%20under%20Disconnections.PNG)  
This figure shows the average training loss across a decentralized network under intermittent connectivity.
Orange dashed lines indicate node disconnections, while green dashed lines indicate reconnections.

**Parameters:**
- Dataset: CIFAR-10 (Jetsons) and MNIST (Akida) 
- Nodes: 4 NVIDIA Jetson (CNN) nodes + 1 BrainChip Akida (SNN/CNN) node
- Momentum buffer: β = 0.9
- Learning rate schedule: Cosine annealing
- Gradient clipping: 5
- Local training steps: TAU1 = 20 batches per round

This experiment demonstrates that momentum buffering stabilizes convergence despite network volatility and packet loss.

---

## 2. CIFAR-10 Validation Accuracy vs Momentum β


![Accuracy](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/accuracy.PNG) 

This figure compares validation accuracy on unseen CIFAR-10 data for different momentum coefficients β.

**Parameters:**
- Dataset: CIFAR-10
- Nodes: 4 NVIDIA Jetson (CNN) nodes + 1 BrainChip Akida (SNN/CNN) node
- Decentralized gossip-based learning
- Same learning rate, batch size, and topology of los.png image for all runs.

Lower β converges faster initially but shows higher variance, while higher β (β = 0.9) provides smoother and more stable convergence.

---

## 3. CIFAR10–MNIST Training Loss (Non-IID)

![Loss NonIID](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/loss.PNG) 

The raining loss evolution under heterogeneous non-IID data distribution across CNN and neuromorphic nodes is shown in this figure.

**Parameters:**
- Dataset: CIFAR-10
- Nodes: 4 NVIDIA Jetson (CNN) nodes + 1 BrainChip Akida (SNN/CNN) node
- Learning rate: 3e-4  
- Batch size: 128  
- Dirichlet non-IID factor: α = 0.1  
- Local steps per round: TAU1 = 50  
- Optimizer: Adam

Higher momentum significantly improves stability and reduces oscillations in cross-architecture decentralized learning.

---

## 4. Live Heterogeneous Communication & Cross-Architecture Adaptation

![Live Heterogeneous](https://github.com/Achref008/Edgesync-Decentralized-federated-learning-for-Heterogeneous-Edge-Devices/blob/main/Images/SuccessfulConverterHetergenousCommVar.PNG)

This screenshot shows real-time execution logs from all decentralized peers, including Jetson CNN nodes and a BrainChip Akida neuromorphic node, during collaborative training. Each terminal represents one device performing peer discovery, weight/logit exchange, gossip aggregation, and cross-architecture parameter adaptation.

**What happens in real time**
- Peer discovery and connection setup
- Weight/logit exchange between neighbors
- Gossip-based aggregation rounds
- Temporary connection failures and automatic recovery
- Cross-architecture weight adaptation between CNN ↔ SNN models

**Key behaviors visible in the logs**
- Received weights / Sent logits : Decentralized peer-to-peer communication (no server)
- Distillation with peers        : Knowledge transfer between heterogeneous models
- Shape mismatch ... resizing    : Automatic parameter projection when converting CNN weights to Akida-compatible formats
- Connection refused             : Simulated network instability
- Waiting for at least k weights : Robust aggregation despite missing peers
- Saved plot / metrics_log       : Per-node monitoring and reproducibility

The logs highlight automatic weight conversion (CNN ↔ SNN), robustness to temporary connection failures, and continued learning despite missing peers. This confirms stable, fully decentralized training across heterogeneous edge hardware without any central coordinator.
