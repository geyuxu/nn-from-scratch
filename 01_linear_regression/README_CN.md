# 01 - 线性回归 (Linear Regression)

[English](README.md)

## 目标

学习线性函数 `y = Wx + b` 的参数，其中：
- `W` (权重) = 3
- `b` (偏置) = 2

## 核心原理

### 1. 模型

线性回归是最简单的神经网络，只有一层线性变换：

```
y_pred = X @ W + b
```

### 2. 损失函数 - 均方误差 (MSE)

衡量预测值与真实值的差距：

```
L = mean((y_pred - y_true)²)
```

### 3. 梯度下降

通过计算损失对参数的梯度，沿梯度反方向更新参数：

```
W = W - lr * ∂L/∂W
b = b - lr * ∂L/∂b
```

### 4. 梯度推导 (Numpy版本手动实现)

对于 MSE 损失 `L = (1/n) * Σ(y_pred - y_true)²`：

```python
dL/dy_pred = 2 * (y_pred - y_true) / n   # 损失对预测值的梯度
dL/dW = X.T @ dL/dy_pred                  # 链式法则
dL/db = sum(dL/dy_pred)                   # 链式法则
```

## 两种实现

| 文件 | 方式 | 特点 |
|------|------|------|
| `linear_regression_numpy.py` | 手动计算梯度 | 理解底层原理 |
| `linear_regression_torch.py` | PyTorch 自动微分 | 使用 `loss.backward()` 自动计算 |

## 训练结果

经过训练后，模型能够学到接近真实值的参数：
- 学到的 W ≈ 3.0
- 学到的 b ≈ 2.0

### Numpy 版本
![numpy训练结果](Figure_linear_regression_numpy.png)

### PyTorch 版本
![torch训练结果](Figure_linear_regression_torch.png)

## 关键代码对比

### Numpy - 手动梯度
```python
# 前向传播
y_pred = X @ W + b
loss = ((y_pred - y_true) ** 2).mean()

# 手动计算梯度
dL_dy = 2 * (y_pred - y_true) / n
dL_dW = X.T @ dL_dy
dL_db = dL_dy.sum()

# 更新参数
W -= lr * dL_dW
b -= lr * dL_db
```

### PyTorch - 自动微分
```python
# 前向传播
y_pred = X @ W + b
loss = ((y_pred - y_true)**2).mean()

# 自动计算梯度
loss.backward()

# 更新参数
with torch.no_grad():
    W -= lr * W.grad
    b -= lr * b.grad
```
