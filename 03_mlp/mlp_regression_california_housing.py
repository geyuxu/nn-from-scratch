import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def relu(t):
    return torch.maximum(t, torch.tensor(0.0))

# 1 import datas
data = fetch_california_housing()
X, y = data.data, data.target

#X = X[:,[0]] # MedInc（收入中位数）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（用训练集fit，transform训练集和测试集）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转为tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# 2 init paramaters
hidden_size = 32
W1 = torch.randn(X_train.shape[1], hidden_size) * 0.1
W1.requires_grad = True
b1 = torch.zeros(hidden_size, requires_grad=True)
W2 = torch.randn(hidden_size,1,requires_grad=True)
b2 = torch.zeros(1,requires_grad=True)

# 3 超参数
lr = 0.01
epoch_count = 1000

# train loop
losses = []
for epoch in range(epoch_count):
    # forward
    h = relu(X_train @ W1 + b1)
    y_pred = h @ W2 + b2

    # loss
    loss = ((y_pred - y_train) ** 2).mean()
    losses.append(loss.item())

    # backward
    loss.backward()

    # update
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()

    if epoch % 100 == 0:
        with torch.no_grad():
            h_test = relu(X_test @ W1 + b1)
            y_test_pred = h_test @ W2 + b2
            test_loss = ((y_test_pred - y_test) ** 2).mean().item()
        print(f"Epoch {epoch}: train_loss={loss.item():.4f}, test_loss={test_loss:.4f}")

print(f"\n网络结构：{X_train.shape[1]} → {hidden_size} → 1")

# ========== 评估指标 ==========
def compute_metrics(y_true, y_pred):
    """计算回归评估指标"""
    mse = ((y_pred - y_true) ** 2).mean().item()
    rmse = mse ** 0.5
    mae = (y_pred - y_true).abs().mean().item()
    # R² = 1 - SS_res / SS_tot
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

with torch.no_grad():
    # MLP forward pass
    h_train = relu(X_train @ W1 + b1)
    y_train_pred = h_train @ W2 + b2

    h_test = relu(X_test @ W1 + b1)
    y_test_pred = h_test @ W2 + b2

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

print("\n" + "=" * 40)
print("评估指标 (Evaluation Metrics)")
print("=" * 40)
print(f"{'指标':<8} {'训练集':>12} {'测试集':>12}")
print("-" * 40)
for key in train_metrics:
    print(f"{key:<8} {train_metrics[key]:>12.4f} {test_metrics[key]:>12.4f}")
print("=" * 40)

# 注：MLP无法像线性回归那样直接解释特征权重
# 因为输入特征经过隐藏层的非线性变换后才影响输出

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 4))

# 1. loss曲线
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 2. 训练集拟合效果（选择MedInc特征）
plt.subplot(1, 3, 2)
feature_idx = 0  # MedInc
X_feature = X_train[:, feature_idx].numpy()
plt.scatter(X_feature, y_train.numpy(), alpha=0.3, s=5, label='true')
plt.scatter(X_feature, y_train_pred.numpy(), alpha=0.3, s=5, c='red', label='pred')
plt.xlabel('MedInc (standardized)')
plt.ylabel('House Price')
plt.title(f'Train Set (R²={train_metrics["R²"]:.3f})')
plt.legend()

# 3. 测试集拟合效果
plt.subplot(1, 3, 3)
X_feature_test = X_test[:, feature_idx].numpy()
plt.scatter(X_feature_test, y_test.numpy(), alpha=0.3, s=5, label='true')
plt.scatter(X_feature_test, y_test_pred.numpy(), alpha=0.3, s=5, c='red', label='pred')
plt.xlabel('MedInc (standardized)')
plt.ylabel('House Price')
plt.title(f'Test Set (R²={test_metrics["R²"]:.3f})')
plt.legend()

plt.tight_layout()
plt.show()


