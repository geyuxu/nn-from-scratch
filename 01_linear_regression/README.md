# 01 - Linear Regression

[中文版](README_CN.md)

## Objective

Learn the parameters of linear function `y = Wx + b`, where:
- `W` (weight) = 3
- `b` (bias) = 2

## Core Principles

### 1. Model

Linear regression is the simplest neural network with only one linear transformation:

```
y_pred = X @ W + b
```

### 2. Loss Function - Mean Squared Error (MSE)

Measures the difference between predicted and true values:

```
L = mean((y_pred - y_true)²)
```

### 3. Gradient Descent

Update parameters in the opposite direction of the gradient:

```
W = W - lr * ∂L/∂W
b = b - lr * ∂L/∂b
```

### 4. Gradient Derivation (Manual Implementation in Numpy)

For MSE loss `L = (1/n) * Σ(y_pred - y_true)²`:

```python
dL/dy_pred = 2 * (y_pred - y_true) / n   # gradient of loss w.r.t. prediction
dL/dW = X.T @ dL/dy_pred                  # chain rule
dL/db = sum(dL/dy_pred)                   # chain rule
```

## Two Implementations

| File | Method | Features |
|------|--------|----------|
| `linear_regression_numpy.py` | Manual gradient | Understand underlying principles |
| `linear_regression_torch.py` | PyTorch autograd | Use `loss.backward()` for automatic computation |

## Training Results

After training, the model learns parameters close to true values:
- Learned W ≈ 3.0
- Learned b ≈ 2.0

### Numpy Version
![numpy training result](Figure_linear_regression_numpy.png)

### PyTorch Version
![torch training result](Figure_linear_regression_torch.png)

## Key Code Comparison

### Numpy - Manual Gradient
```python
# Forward pass
y_pred = X @ W + b
loss = ((y_pred - y_true) ** 2).mean()

# Manual gradient computation
dL_dy = 2 * (y_pred - y_true) / n
dL_dW = X.T @ dL_dy
dL_db = dL_dy.sum()

# Update parameters
W -= lr * dL_dW
b -= lr * dL_db
```

### PyTorch - Automatic Differentiation
```python
# Forward pass
y_pred = X @ W + b
loss = ((y_pred - y_true)**2).mean()

# Automatic gradient computation
loss.backward()

# Update parameters
with torch.no_grad():
    W -= lr * W.grad
    b -= lr * b.grad
```
