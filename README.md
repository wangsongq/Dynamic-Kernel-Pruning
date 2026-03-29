# Reconfigurable Digital RRAM Logic Enables In-situ Pruning and Learning for Edge AI

The human brain simultaneously optimizes synaptic weights and network topology by growing, pruning, and strengthening synapses while performing all computation entirely in memory. In contrast, modern artificial intelligence systems separate weight optimization from topology optimization and depend on energy-intensive von Neumann architectures.

Here, we present a **software–hardware co-design** that bridges this gap:

- **Algorithmic side:** We introduce a real-time dynamic weight-pruning strategy that monitors weight similarity during training and removes redundancies on the fly, reducing operations by **26.80% on MNIST** and **59.94% on ModelNet10** while preserving accuracy (**91.44%** and **77.75%**, respectively). Across CIFAR-10/100 and ImageNet models, our method prunes **27–68.79%** of kernels with only **1–1.5%** accuracy drop.

- **Hardware side:** We fabricate a **reconfigurable, fully digital compute-in-memory (CIM) chip** based on **180 nm one-transistor-one-resistor (1T1R) RRAM arrays**. Each array embeds flexible Boolean logic (**NAND, AND, XOR, OR**), enabling both convolution and similarity evaluation inside memory and eliminating ADC/DAC overhead.

Under a **22 nm node-normalized reference**, our digital CIM design achieves:

- **CIM weight density:** **4.37 Mb/mm²**
- **Energy efficiency:** **14.97 TOPS/W (8b/8b)** and **222.72 TOPS/W (1b/1b)**
- **SWaP:** **973.1 TOPS/W·Mb/mm²**

Compared with an **NVIDIA RTX 4090**, the digital RRAM system lowers energy consumption by:

- **99.27% on MNIST**
- **96.07% on ModelNet10**

The chip supports multi-bit storage, with measured write bit error rates of **6.1×10⁻⁶** for binary storage and **2.4×10⁻⁴** for 2-bit storage.

Together, our co-design establishes a **scalable brain-inspired paradigm for adaptive, energy-efficient edge intelligence**.

Please refer to the accompanying manuscript for detailed experimental results, methodology, and chip design principles.

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

