import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def sigmoid(t):
    return 1 / (1 + torch.exp(-t))

def relu(t):
    return torch.maximum(t, torch.tensor(0.0))

def forward(X):
    h = relu(X @ W1 + b1)
    return sigmoid(h @ W2 + b2)

# 导入数据
data = load_breast_cancer()
X, y = data.data, data.target

#  划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 标准化（用训练集fit，防止数据泄露）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 注意：用 transform 而非 fit_transform

# 转tensor
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test,dtype=torch.float32).reshape(-1, 1)

print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"类别分布: 良性={y_train.sum().int().item()}, 恶性={len(y_train) - y_train.sum().int().item()}")



# 2 init paramaters
hidden_size = 32

W1 = (torch.randn(X_train.shape[1],hidden_size) * 0.01).requires_grad_(True)
b1 = torch.zeros(hidden_size,requires_grad=True)
W2 = (torch.randn(hidden_size,1) * 0.01).requires_grad_(True)
b2 = torch.zeros(1,requires_grad=True)

print("--------")

# 3 超参数
lr = 0.01
epoch_count = 3000

# train loop
losses = []

for epoch in range(epoch_count):
    # forwared
    y_pred = forward(X_train)

    # loss 交叉熵
    loss = -(y_train * torch.log(y_pred + 1e-8) + (1 - y_train) * torch.log((1 - y_pred) + 1e-8)).mean()
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
        acc = ((y_pred > 0.5) == y_train).float().mean()
        print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")


# ========== 评估指标 ==========
def compute_classification_metrics(y_true, y_pred_prob, threshold=0.5):
    """计算分类评估指标"""
    y_pred = (y_pred_prob > threshold).float().squeeze()
    y_true = y_true.squeeze()

    # 准确率
    acc = (y_pred == y_true).float().mean().item()

    # TP, TN, FP, FN
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }

with torch.no_grad():
    # 训练集预测 (MLP两层)
    h_train = relu(X_train @ W1 + b1)
    y_train_pred = sigmoid(h_train @ W2 + b2)
    train_metrics = compute_classification_metrics(y_train, y_train_pred)

    # 测试集预测 (MLP两层)
    y_test_pred = forward(X_test)
    test_metrics = compute_classification_metrics(y_test, y_test_pred)

print("\n" + "=" * 50)
print("评估指标 (Classification Metrics)")
print("=" * 50)
print(f"{'指标':<12} {'训练集':>12} {'测试集':>12}")
print("-" * 50)
for key in ['Accuracy', 'Precision', 'Recall', 'F1']:
    print(f"{key:<12} {train_metrics[key]:>12.4f} {test_metrics[key]:>12.4f}")
print("=" * 50)

# ========== 可视化 ==========
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Loss曲线
axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Binary Cross-Entropy)')

# 2. 混淆矩阵 - 测试集
cm = np.array([[test_metrics['TN'], test_metrics['FP']],
               [test_metrics['FN'], test_metrics['TP']]])
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Malignant (0)', 'Benign (1)'])
axes[1].set_yticklabels(['Malignant (0)', 'Benign (1)'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Confusion Matrix (Test Acc={test_metrics["Accuracy"]:.3f})')
# 添加数值标注
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')

# 3. 用PCA降维后可视化分类效果
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test.numpy())

y_pred_class = (y_test_pred > 0.5).squeeze().numpy()
y_true_class = y_test.squeeze().numpy()

# 正确分类和错误分类
correct = y_pred_class == y_true_class
axes[2].scatter(X_test_pca[correct & (y_true_class == 1), 0],
               X_test_pca[correct & (y_true_class == 1), 1],
               c='blue', marker='o', alpha=0.6, label='Benign (correct)')
axes[2].scatter(X_test_pca[correct & (y_true_class == 0), 0],
               X_test_pca[correct & (y_true_class == 0), 1],
               c='red', marker='o', alpha=0.6, label='Malignant (correct)')
axes[2].scatter(X_test_pca[~correct, 0], X_test_pca[~correct, 1],
               c='black', marker='x', s=100, label='Misclassified')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].set_title('Test Set (PCA Projection)')
axes[2].legend()

plt.tight_layout()
plt.show()