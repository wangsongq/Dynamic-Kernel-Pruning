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
