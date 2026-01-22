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

## 实现版本

| 文件 | 方式 | 特点 |
|------|------|------|
| `linear_regression_numpy.py` | 手动计算梯度 | 理解底层原理 |
| `linear_regression_torch.py` | PyTorch 自动微分 | 使用 `loss.backward()` 自动计算 |
| `linear_regression_california_housing.py` | 真实数据集 | 多特征回归 + 训练/测试集划分 |

## 训练结果 - 合成数据

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

---

## 真实数据集 - California Housing

使用加州房价数据集进行多特征线性回归。

### 数据集信息

- **样本数**: 20,640
- **特征数**: 8 个 (收入中位数、房龄、平均房间数、平均卧室数、人口、平均住户数、纬度、经度)
- **目标**: 房价中位数 (单位: $100,000)

### 与合成数据的区别

| 方面 | 合成数据 | California Housing |
|------|----------|-------------------|
| 特征数 | 1 | 8 |
| 数据划分 | 无 | 80% 训练 / 20% 测试 |
| 预处理 | 无 | StandardScaler 标准化 |
| 评估指标 | 仅 Loss | MSE, RMSE, MAE, R² |

### 评估指标说明

| 指标 | 说明 |
|------|------|
| **MSE** | 均方误差 - 预测误差平方的平均值 |
| **RMSE** | 均方根误差 - MSE 的平方根，与目标同单位 |
| **MAE** | 平均绝对误差 - 预测误差绝对值的平均 |
| **R²** | 决定系数 - 模型解释的方差比例 (1.0 = 完美拟合) |

### 训练结果

![California Housing 结果](Figure_linear_regression_california_housing.png)

- **左图**: 训练损失曲线
- **中图**: 训练集预测效果 (展示收入中位数特征)
- **右图**: 测试集预测效果及 R² 分数

### 结果分析

**R² 对比**

| 模型 | Train R² | Test R² |
|------|----------|---------|
| 单特征 (MedInc) | 0.601 | 0.563 |
| 8特征 | 0.608 | 0.576 |

**特征权重解读**

```
正相关（越高房价越贵）：
  MedInc:    0.8616  ← 收入最重要
  AveBedrms: 0.2884
  HouseAge:  0.1499

负相关（越高房价越低）：
  Latitude: -0.6822  ← 越往北越便宜
  Longitude:-0.6537  ← 越往东越便宜（加州内陆）
  AveRooms: -0.2606  ← 可能与 AveBedrms 存在多重共线性
```

**符合直觉**：加州沿海（西、南）房价贵，收入高的地方房价贵。

### 结论

线性回归在这个数据集上的天花板大约是 **R² ≈ 0.6**。想要更高，需要：

1. **非线性模型** - MLP、决策树、XGBoost
2. **特征工程** - 交叉特征、多项式特征

这正是下一步 MLP 要解决的问题！
