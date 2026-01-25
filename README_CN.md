# 从零实现经典神经网络

[English](README.md)

从底层实现经典神经网络，不使用 `nn.Module`，深入理解 PyTorch 等框架和模型原理。

## 项目结构

```
nn-from-scratch/
├── 01_linear_regression/       # 线性回归
├── 02_logistic_regression/     # 逻辑回归
├── 03_mlp/                     # 多层感知机
├── 04_cnn/                     # 卷积神经网络
├── 05_rnn/                     # 循环神经网络
├── 06_attention/               # 注意力机制 & Transformer
├── utils/                      # 工具函数
└── README.md
```

## 已完成模块

| 模块 | 说明 | 数据集 | 文档 |
|------|------|--------|------|
| [01_linear_regression](01_linear_regression/) | 线性回归 - Numpy手动梯度 & PyTorch自动微分 | 合成数据 + California Housing (R²≈0.6) | [README](01_linear_regression/README_CN.md) |
| [02_logistic_regression](02_logistic_regression/) | 逻辑回归 - Sigmoid & 交叉熵 | 合成数据 + Breast Cancer (97% acc) | [README](02_logistic_regression/README_CN.md) |
| [03_mlp](03_mlp/) | 多层感知机 - 隐藏层 & ReLU/Softmax激活 | Breast Cancer (99% acc) + Iris (100% acc) + California Housing (R²≈0.7) | [README](03_mlp/README_CN.md) |
| [04_cnn](04_cnn/) | 卷积神经网络 - 卷积、池化 & 空间特征 | MNIST (98.3% acc) + CIFAR-10 VGG16 (81.4% acc) | [README](04_cnn/README_CN.md) |

## 环境搭建

### 1. 创建虚拟环境

```bash
# 创建项目目录
mkdir nn-from-scratch
cd nn-from-scratch

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate
```

### 2. 安装 PyTorch

**CPU 版本：**

```bash
pip install torch torchvision
```

**NVIDIA GPU 版本：**

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.8 for rtx 50xx
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Apple Silicon (M1/M2/M3)：**

```bash
pip install torch torchvision  # 自动支持 MPS 加速
```

### 3. 安装辅助工具

```bash
pip install matplotlib numpy jupyter
```

### 4. 验证安装

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 简单测试
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.sum() ** 2
y.backward()
print(f"梯度测试: x.grad = {x.grad}")  # 应该是 [6., 6.]
```

### 5. 保存依赖

```bash
pip freeze > requirements.txt
```

## 常用命令

```bash
# 退出虚拟环境
deactivate

# 重新进入
source venv/bin/activate

# 查看已安装包
pip list
```

## 学习路线

| 阶段 | 模型 | 关键概念 |
|------|------|----------|
| 1 | 线性回归 | 张量运算、autograd、梯度下降 |
| 2 | 逻辑回归 | sigmoid、交叉熵、分类 |
| 3 | MLP | 多层、激活函数、链式法则 |
| 4 | CNN | 卷积运算、特征提取 |
| 5 | RNN/LSTM | 时序展开、隐状态、门控 |
| 6 | Attention | QKV、softmax、mask |
| 7 | Transformer | 多头注意力、位置编码 |
