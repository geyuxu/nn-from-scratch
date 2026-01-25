# Neural Networks from Scratch

[中文版](README_CN.md)

Implement classic neural networks from scratch without using `nn.Module`, to deeply understand PyTorch and model principles.

## Project Structure

```
nn-from-scratch/
├── 01_linear_regression/       # Linear Regression
├── 02_logistic_regression/     # Logistic Regression
├── 03_mlp/                     # Multi-Layer Perceptron
├── 04_cnn/                     # Convolutional Neural Network
├── 05_rnn/                     # Recurrent Neural Network
├── 06_attention/               # Attention & Transformer
├── utils/                      # Utility functions
└── README.md
```

## Completed Modules

| Module | Description | Dataset | Docs |
|--------|-------------|---------|------|
| [01_linear_regression](01_linear_regression/) | Linear Regression - Numpy manual gradient & PyTorch autograd | Synthetic + California Housing (R²≈0.6) | [README](01_linear_regression/README.md) |
| [02_logistic_regression](02_logistic_regression/) | Logistic Regression - Sigmoid & Cross-Entropy | Synthetic + Breast Cancer (97% acc) | [README](02_logistic_regression/README.md) |
| [03_mlp](03_mlp/) | Multi-Layer Perceptron - Hidden layers & ReLU/Softmax activation | Breast Cancer (99% acc) + Iris (100% acc) + California Housing (R²≈0.7) | [README](03_mlp/README.md) |
| [04_cnn](04_cnn/) | Convolutional Neural Network - Conv, Pooling & spatial features | MNIST (98.3% acc) + CIFAR-10 VGG16 (81.4% acc) | [README](04_cnn/README.md) |

## Environment Setup

### 1. Create Virtual Environment

```bash
# Create project directory
mkdir nn-from-scratch
cd nn-from-scratch

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate
```

### 2. Install PyTorch

**CPU Version:**

```bash
pip install torch torchvision
```

**NVIDIA GPU Version:**

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.8 for rtx 50xx
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Apple Silicon (M1/M2/M3):**

```bash
pip install torch torchvision  # Automatically supports MPS acceleration
```

### 3. Install Helper Tools

```bash
pip install matplotlib numpy jupyter
```

### 4. Verify Installation

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Simple test
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.sum() ** 2
y.backward()
print(f"Gradient test: x.grad = {x.grad}")  # Should be [6., 6.]
```

### 5. Save Dependencies

```bash
pip freeze > requirements.txt
```

## Common Commands

```bash
# Deactivate virtual environment
deactivate

# Reactivate
source venv/bin/activate

# List installed packages
pip list
```

## Learning Path

| Stage | Model | Key Concepts |
|-------|-------|--------------|
| 1 | Linear Regression | Tensor operations, autograd, gradient descent |
| 2 | Logistic Regression | Sigmoid, cross-entropy, classification |
| 3 | MLP | Multi-layer, activation functions, chain rule |
| 4 | CNN | Convolution, feature extraction |
| 5 | RNN/LSTM | Temporal unrolling, hidden state, gating |
| 6 | Attention | QKV, softmax, mask |
| 7 | Transformer | Multi-head attention, positional encoding |
