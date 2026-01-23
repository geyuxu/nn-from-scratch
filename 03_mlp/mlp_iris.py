import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def sigmoid(t):
    return 1 / (1 + torch.exp(-t))

def relu(t):
    return torch.maximum(t, torch.tensor(0.0))

def softmax(t):
    exp_t = torch.exp(t - t.max(dim=1, keepdim=True).values)  # 数值稳定性
    return exp_t / exp_t.sum(dim=1, keepdim=True)

def forward(X):
    h = relu(X @ W1 + b1)
    return softmax(h @ W2 + b2)

def one_hot(y, num_classes):
    return (y.unsqueeze(1) == torch.arange(num_classes)).float()


# 导入数据
data = load_iris()
X, y = data.data, data.target

#  划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（用训练集fit，防止数据泄露）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"类别分布: setosa={(y_train == 0).sum().item()}, "
      f"versicolor={(y_train == 1).sum().item()}, "
      f"virginica={(y_train == 2).sum().item()}")


# 2 init paramaters
num_classes = 3
hidden_size = 32

W1 = (torch.randn(X_train.shape[1], hidden_size) * 0.01).requires_grad_(True)
b1 = torch.zeros(hidden_size, requires_grad=True)
W2 = (torch.randn(hidden_size, num_classes) * 0.01).requires_grad_(True)
b2 = torch.zeros(num_classes, requires_grad=True)

print("--------")

# 3 超参数
lr = 0.1
epoch_count = 3000

# train loop
losses = []

for epoch in range(epoch_count):
    # forward
    y_pred = forward(X_train)

    # loss 多分类交叉熵
    y_train_onehot = one_hot(y_train, num_classes)
    loss = -(y_train_onehot * torch.log(y_pred + 1e-8)).sum(dim=1).mean()
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
        acc = (y_pred.argmax(dim=1) == y_train).float().mean()
        print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")


# ========== 评估指标 ==========

with torch.no_grad():
    # 训练集预测
    y_train_pred = forward(X_train)
    y_train_pred_class = y_train_pred.argmax(dim=1)
    train_cm = confusion_matrix(y_train.numpy(), y_train_pred_class.numpy())
    train_acc = (y_train_pred_class == y_train).float().mean().item()

    # 测试集预测
    y_test_pred = forward(X_test)
    y_test_pred_class = y_test_pred.argmax(dim=1)
    test_cm = confusion_matrix(y_test.numpy(), y_test_pred_class.numpy())
    test_acc = (y_test_pred_class == y_test).float().mean().item()

print("\n" + "=" * 50)
print("评估指标 (Classification Metrics)")
print("=" * 50)
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print("=" * 50)

# ========== 可视化 ==========
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Loss曲线
axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Cross-Entropy)')

# 2. 混淆矩阵 - 测试集 (3x3)
class_names = ['setosa', 'versicolor', 'virginica']
im = axes[1].imshow(test_cm, cmap='Blues')
axes[1].set_xticks([0, 1, 2])
axes[1].set_yticks([0, 1, 2])
axes[1].set_xticklabels(class_names)
axes[1].set_yticklabels(class_names)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Confusion Matrix (Test Acc={test_acc:.3f})')
# 添加数值标注
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, test_cm[i, j], ha='center', va='center', fontsize=14,
                    color='white' if test_cm[i, j] > test_cm.max()/2 else 'black')

# 3. 用PCA降维后可视化分类效果
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test.numpy())

y_pred_class = y_test_pred_class.numpy()
y_true_class = y_test.numpy()

# 正确分类和错误分类
correct = y_pred_class == y_true_class

# 三个类别用不同颜色
colors = ['red', 'green', 'blue']

for i in range(3):
    mask = correct & (y_true_class == i)
    axes[2].scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   c=colors[i], marker='o', alpha=0.6, label=f'{class_names[i]} (correct)')

# 错误分类的点
axes[2].scatter(X_test_pca[~correct, 0], X_test_pca[~correct, 1],
               c='black', marker='x', s=100, label='Misclassified')

axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].set_title('Test Set (PCA Projection)')
axes[2].legend()

plt.tight_layout()
plt.show()
