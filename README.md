# Neural Networks from Scratch (PyTorch Internals)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-No_nn.Module-orange)](https://pytorch.org/)

**[中文版](README_CN.md) | [English](README.md)**

> **"What I cannot create, I do not understand."** — Richard Feynman

This project implements classic neural network architectures (MLP, CNN) **without using `torch.nn.Module`**.

The goal is to demystify the "black box" of Deep Learning. By rebuilding layers like `Linear`, `Conv2d`, and `MaxPool2d` using only raw tensors and Autograd, we can visualize the math behind the magic.

## 🚀 Why this repo?

Most tutorials start with:
```python
# The "Magic" Way
import torch.nn as nn
model = nn.Linear(10, 1)  # What actually happens here?

```

This repo does:

```python
# The "From Scratch" Way (What we build)
import torch
class MyLinear:
    def __init__(self, in_features, out_features):
        # We manually initialize weights and bias
        self.weights = torch.randn(out_features, in_features, requires_grad=True)
        self.bias = torch.randn(out_features, requires_grad=True)
        
    def forward(self, x):
        # We implement the raw math: y = xA^T + b
        return x @ self.weights.t() + self.bias

```

---

## 📂 Project Structure & Progress

| Module | Status | Key Concepts | Dataset & Accuracy |
| :--- | :--- | :--- | :--- |
| **01_Linear_Regression** | ✅ Done | Numpy Manual Gradients vs Autograd | Housing Prices ($R^2 \approx 0.6$) |
| **02_Logistic_Regression** | ✅ Done | Sigmoid, Cross-Entropy Loss | Breast Cancer ($97\%$) |
| **03_MLP** | ✅ Done | Hidden Layers, ReLU, Chain Rule | Iris ($100\%$), Housing ($R^2 \approx 0.7$) |
| **04_CNN** | ✅ Done | **Convolution, Pooling, Flatten** | **MNIST ($98.3\%$), CIFAR-10 ($81.4\%$)** |
| *05_RNN* | 🚧 Planned | Temporal Unrolling, Hidden States | - |
| *06_Attention* | 🚧 Planned | QKV, Masking, Transformer Block | - |

> **Note:** The directory structure reflects the learning path. Modules 05 and 06 are currently under active development.

```text
nn-from-scratch/
├── 01_linear_regression/   # Numpy & PyTorch implementations
├── 02_logistic_regression/ # Binary classification
├── 03_mlp/                 # Deep Feed-forward networks
├── 04_cnn/                 # Computer Vision from scratch
├── utils/                  # Evaluation & Plotting tools
└── requirements.txt

```

---

## 🛠️ Environment Setup

Designed for **Python 3.8+**.

### 1. Create Virtual Environment

```bash
# Create and activate
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

```

### 2. Install PyTorch

Choose the version matching your hardware:

* **Standard (CPU / Older GPU):**
```bash
pip install torch torchvision

```


* **NVIDIA RTX 40xx / 30xx (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

```


* **NVIDIA RTX 50xx (Blackwell / CUDA 12.8):** ⚡
```bash
# Nightly build required for RTX 5080/5090 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

```


* **Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision
# Automatically supports MPS (Metal Performance Shaders)

```



### 3. Install Utilities

```bash
pip install matplotlib numpy jupyter notebook

```

### 4. Verification

Run this quick snippet to verify Autograd is working:

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.sum() ** 2
y.backward()
print(f"Gradient: {x.grad}")  # Should be [6., 6.]

```

---

## 📚 Learning Path

1. **Linear Regression:** Mastering Tensor operations and `requires_grad`.
2. **Logistic Regression:** Understanding probability outputs and classification loss.
3. **MLP:** Implementing the universal approximation theorem.
4. **CNN:** Learning how kernels extract spatial features (Edge detection -> Shapes -> Objects).
5. **Next Steps:** RNNs (Memory) and Attention (Context).

## 🤝 Contributing

This is an educational project. If you find a bug in the math or code, feel free to open an issue or PR!
