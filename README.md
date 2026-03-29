# Reconfigurable Digital RRAM Logic Enables In-situ Pruning and Learning for Edge AI

The human brain simultaneously optimizes synaptic weights and network topology by growing, pruning, and strengthening synapses while performing all computation entirely in memory. In contrast, modern artificial intelligence systems separate weight optimization from topology optimization and depend on energy-intensive von Neumann architectures.

Here, we present a **software–hardware co-design** that bridges this gap.

## Overview

### Algorithmic side

We introduce a **real-time dynamic weight-pruning strategy** that monitors weight similarity during training and removes redundancies on the fly.

- On **MNIST**, the method reduces operations by **26.80%** while preserving **91.44%** accuracy.
- On **ModelNet10**, the method reduces operations by **59.94%** while preserving **77.75%** accuracy.
- Across **CIFAR-10, CIFAR-100, and ImageNet** models, the method prunes **27–68.79%** of kernels with only **1–1.5%** accuracy drop.

### Hardware side

We fabricate a **reconfigurable, fully digital compute-in-memory (CIM) chip** based on **180 nm one-transistor-one-resistor (1T1R) RRAM arrays**.

- The chip integrates **reconfigurable Boolean logic** directly in memory, supporting **NAND, AND, XOR, and OR**.
- The same in-memory hardware supports both **forward computation** and **weight-similarity evaluation**.
- The fully digital design eliminates **ADC/DAC overhead** and avoids analogue-noise-induced accuracy degradation.

## Key Results

Under a **22 nm node-normalized reference**, the proposed digital RRAM CIM design achieves:

- **CIM weight density:** **4.37 Mb/mm²**
- **Energy efficiency:** **14.97 TOPS/W (8b/8b)** and **222.72 TOPS/W (1b/1b)**
- **SWaP:** **973.1 TOPS/W·Mb/mm²**

Here, **SWaP** denotes the product of **energy efficiency** and **CIM weight density**, capturing the joint benefit of compute efficiency and on-chip storage density under a fixed area budget.

Compared with an **NVIDIA RTX 4090**, the digital RRAM system reduces energy consumption by:

- **99.27% on MNIST**
- **96.07% on ModelNet10**

The chip also supports multi-bit storage, with measured write bit error rates of:

- **6.1×10⁻⁶** for **binary storage**
- **2.4×10⁻⁴** for **2-bit storage**

Together, this work establishes a **scalable brain-inspired paradigm for adaptive, energy-efficient edge intelligence**.
---

# Deployment

## Clone the repository

```bash
git clone https://github.com/wangsongq/Dynamic-Kernel-Pruning.git
cd Dynamic-Kernel-Pruning
pip install -r requirements.txt
```
## Script Overview: `cimcam_mnist_cnn.py`

This script implements a **VGG16-like CNN** with:

**Binary convolution layers** (weights binarized to +1/-1)  
**Filter pruning based on Hamming distance similarity**  
**Training and testing on MNIST dataset**

### **Configurable Training Parameters**

| Parameter | Description | Default |
| --- | --- | --- |
| `learning` | Learning rate for Adam optimizer | `0.001` |
| `epoch` | Number of training epochs | `50` |
| `batch_size` | Training batch size (in `DataLoader`) | `64` |
| `threshold_c` | Pruning frequency threshold in `binary_prune_model` | `0.15` |
| `threshold_b` | Variance scaling factor for pruning criterion | `0.146` |
| `device` | GPU or CPU (auto-detected) | `"cuda:1"` if available |

## Script Overview: `pointnet_train.py`

This script implements **PointNet++ with quantization-based pruning** for 3D point cloud classification using PyTorch.

### **Configurable Training Parameters**

| Parameter     | Description                                   | Default   |
|---------------|-----------------------------------------------|-----------|
| `epoch`       | Number of training epochs                    | `50`     |
| `learning`    | Learning rate for Adam optimizer             | `0.001`   |
| `threshold_c` | Pruning frequency threshold (filter pruning) | `0.45`    |
| `threshold_b` | Pruning variance scaling threshold           | `0.01`    |
| `batch_size`  | Training batch size                          | `32`      |
| `device`      | CUDA device selection                        | `"cuda:1"` if available |

If you want to test the performance of a trained PointNet++ model, you can use pointnet_test.py. Remember to update the weights path in the code before running.

