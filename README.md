# Reconfigurable Digital RRAM Logic Enables In-situ Pruning and Learning for Edge AI

The human brain simultaneously optimizes synaptic weights and network topology by growing, pruning, and strengthening synapses while performing all computation entirely in memory. In contrast, modern artificial intelligence systems separate weight optimization from topology optimization and depend on energy-intensive von Neumann architectures.

Here, we present a **software–hardware co-design** that bridges this gap:

- **Algorithmic side:** We introduce a real-time dynamic weight pruning strategy that monitors weight similarity during training and removes redundancies on the fly, reducing operations by **26.8% on MNIST** and **59.9% on ModelNet10** without sacrificing accuracy (**91.4%** and **77.8%**, respectively).

- **Hardware side:** We fabricate a **reconfigurable, fully digital compute-in-memory (CIM) chip** based on 180 nm one-transistor-one-resistor (1T1R) RRAM arrays. Each array embeds flexible Boolean logic (NAND, AND, XOR, OR), enabling both convolution and similarity evaluation inside memory and eliminating all ADC/DAC overhead.

Our **digital CIM design** achieves:

- **Zero bit-error digital computing**
- **72.3% reduction in silicon area** and **57.3% reduction in overall energy** compared to analogue RRAM CIM
- **75.6% and 86.5% lower energy consumption** on MNIST and ModelNet10, respectively, compared to an NVIDIA RTX 4090 GPU.

Together, our co-design establishes a **scalable brain-inspired paradigm for adaptive, energy-efficient edge intelligence**.

Please refer to our [paper](https://arxiv.org/abs/2506.13151) for detailed experimental results, methodology, and chip design principles.

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
| `epoch` | Number of training epochs | `2` |
| `batch_size` | Training batch size (in `DataLoader`) | `64` |
| `threshold_c` | Pruning frequency threshold in `binary_prune_model` | `0.15` |
| `threshold_b` | Variance scaling factor for pruning criterion | `0.146` |
| `device` | GPU or CPU (auto-detected) | `"cuda:1"` if available |

## Script Overview: pointnet_train.py

This script implements **PointNet++ with quantization-based pruning** for 3D point cloud classification using PyTorch.

### **Configurable Training Parameters**

| Parameter     | Description                                   | Default   |
|---------------|-----------------------------------------------|-----------|
| `epoch`       | Number of training epochs                    | `100`     |
| `learning`    | Learning rate for Adam optimizer             | `0.001`   |
| `threshold_c` | Pruning frequency threshold (filter pruning) | `0.45`    |
| `threshold_b` | Pruning variance scaling threshold           | `0.01`    |
| `batch_size`  | Training batch size                          | `32`      |
| `device`      | CUDA device selection                        | `"cuda:1"` if available |

If you want to test the performance of a trained PointNet++ model, you can use pointnet_test.py. Remember to update the weights path in the code before running.

