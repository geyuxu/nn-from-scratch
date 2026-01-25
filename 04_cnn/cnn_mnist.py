from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

# 设置设备：优先使用 MPS (Apple Silicon GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())

print("=== 原始数据 ===")
print(f"类型: {type(train_data)}")
print(f"图像 shape: {train_data.data.shape}")      # (60000, 28, 28)
print(f"标签 shape: {train_data.targets.shape}")   # (60000,)
print(f"图像 dtype: {train_data.data.dtype}")      # uint8
print(f"像素范围: {train_data.data.min()} ~ {train_data.data.max()}")  # 0 ~ 255


# 数据移到设备上
X_train = (train_data.data.float() / 255.0).to(device)
y_train = train_data.targets.to(device)
X_test = (test_data.data.float() / 255.0).to(device)
y_test = test_data.targets.to(device)

print("\n=== 转换后 ===")
print(f"X dtype: {X_train.dtype}")                 # float32
print(f"像素范围: {X_train.min()} ~ {X_train.max()}")  # 0 ~ 1

# import matplotlib.pyplot as plt
# idx = 99
# plt.imshow(X_train[idx], cmap='gray')
# plt.show()
# print(f"target={y_train[idx]}")

print("=========")

def conv2d(X, W, b):
    """
    向量化卷积（无 for 循环，GPU 并行）
    X: (batch, H, W)
    W: (out_c, kH, kW)
    b: (out_c,)
    return: (batch, out_c, out_H, out_W)
    """
    batch, H, W_in = X.shape
    out_c, kH, kW = W.shape
    out_H, out_W = H - kH + 1, W_in - kW + 1

    # 1. unfold 提取所有窗口: (batch, out_H, out_W, kH, kW)
    patches = X.unfold(1, kH, 1).unfold(2, kW, 1)

    # 2. 重塑为矩阵乘法形式
    patches = patches.reshape(batch, out_H * out_W, kH * kW)  # (batch, out_H*out_W, kH*kW)
    W_flat = W.reshape(out_c, kH * kW)                        # (out_c, kH*kW)

    # 3. 矩阵乘法: (batch, out_H*out_W, kH*kW) @ (kH*kW, out_c) → (batch, out_H*out_W, out_c)
    out = patches @ W_flat.T + b

    # 4. 重塑回 (batch, out_c, out_H, out_W)
    return out.reshape(batch, out_H, out_W, out_c).permute(0, 3, 1, 2)


def conv2d_multichannel(X, W, b):
    """
    向量化多通道卷积（无 for 循环，GPU 并行）
    X: (batch, in_c, H, W)
    W: (out_c, in_c, kH, kW)
    b: (out_c,)
    return: (batch, out_c, out_H, out_W)
    """
    batch, in_c, H, W_in = X.shape
    out_c, _, kH, kW = W.shape
    out_H, out_W = H - kH + 1, W_in - kW + 1

    # 1. unfold 提取所有窗口: (batch, in_c, out_H, out_W, kH, kW)
    patches = X.unfold(2, kH, 1).unfold(3, kW, 1)

    # 2. 重塑: (batch, out_H*out_W, in_c*kH*kW)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch, out_H * out_W, in_c * kH * kW)
    W_flat = W.reshape(out_c, in_c * kH * kW)

    # 3. 矩阵乘法
    out = patches @ W_flat.T + b

    return out.reshape(batch, out_H, out_W, out_c).permute(0, 3, 1, 2)


def maxpool2d(X, size=2):
    """
    向量化池化（无 for 循环，GPU 并行）
    X: (batch, c, H, W)
    return: (batch, c, H//size, W//size)
    """
    batch, c, H, W = X.shape
    out_H, out_W = H // size, W // size

    # 截断到能被 size 整除的尺寸（处理奇数尺寸）
    X = X[:, :, :out_H * size, :out_W * size]

    # 重塑为 (batch, c, out_H, size, out_W, size)，然后取 max
    X = X.reshape(batch, c, out_H, size, out_W, size)
    return X.max(dim=3).values.max(dim=4).values

def relu(x):
    return torch.maximum(x, torch.tensor(0.0, device=x.device))

def dropout(x, p=0.25, training=True):
    """Dropout: 训练时随机置零，测试时不变"""
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)  # scale to maintain expected value

def softmax(x):
    # x: (batch, 10)
    # 减去最大值防止 exp 溢出
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return exp_x / exp_x.sum(dim=1, keepdim=True)


# 参数移到设备上
# Conv1: 32 个 3x3 滤波器
W1 = (torch.randn(32, 3, 3, device=device) * 0.1).requires_grad_(True)
b1 = torch.zeros(32, device=device, requires_grad=True)

# Conv2: 64 个 3x3 滤波器，输入 32 通道
W2 = (torch.randn(64, 32, 3, 3, device=device) * 0.1).requires_grad_(True)
b2 = torch.zeros(64, device=device, requires_grad=True)

# Dense1: 9216 → 128 (64 * 12 * 12 = 9216)
W3 = (torch.randn(9216, 128, device=device) * 0.1).requires_grad_(True)
b3 = torch.zeros(128, device=device, requires_grad=True)

# Dense2: 128 → 10
W4 = (torch.randn(128, 10, device=device) * 0.1).requires_grad_(True)
b4 = torch.zeros(10, device=device, requires_grad=True)

batch_size = 256
epochs = 10
lr = 0.1

# 记录训练过程
train_losses = []
train_accs = []
test_accs = []

def cross_entropy(y_pred, y_true,batch):
    return -torch.log(y_pred[range(batch), y_true] + 1e-8).mean()

def evaluate(X, y):
    """在数据集上评估模型（测试时不使用 dropout）"""
    with torch.no_grad():
        # Conv1 → ReLU
        h1 = conv2d(X, W1, b1)
        h1 = relu(h1)

        # Conv2 → ReLU → MaxPool
        h2 = conv2d_multichannel(h1, W2, b2)
        h2 = relu(h2)
        h2 = maxpool2d(h2)
        # 测试时不用 dropout

        # Flatten → Dense 128 → ReLU
        h2_flat = h2.view(h2.shape[0], -1)
        h3 = relu(h2_flat @ W3 + b3)
        # 测试时不用 dropout

        # Dense 10 → Softmax
        out = softmax(h3 @ W4 + b4)

        pred = out.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        return acc, pred

for epoch in range(epochs):
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    for i in range(0,len(X_train),batch_size):
        X = X_train[i:i+batch_size]
        y = y_train[i:i+batch_size]

        # 前向
        # Conv1 → ReLU
        h1 = conv2d(X, W1, b1)
        h1 = relu(h1)

        # Conv2 → ReLU → MaxPool → Dropout
        h2 = conv2d_multichannel(h1, W2, b2)
        h2 = relu(h2)
        h2 = maxpool2d(h2)
        h2 = dropout(h2, p=0.25, training=True)

        # Flatten → Dense 128 → ReLU → Dropout
        h2_flat = h2.view(h2.shape[0], -1)
        h3 = relu(h2_flat @ W3 + b3)
        h3 = dropout(h3, p=0.25, training=True)

        # Dense 10 → Softmax
        out = softmax(h3 @ W4 + b4)

        #print(out.shape)
        pred = out.argmax(dim=1)

        # 损失
        loss = cross_entropy(out, y, out.shape[0])

        # 累积统计
        epoch_loss += loss.item() * len(y)
        epoch_correct += (pred == y).sum().item()
        epoch_total += len(y)

        # 反向 + 更新
        loss.backward()
        with torch.no_grad():
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad
            W3 -= lr * W3.grad
            b3 -= lr * b3.grad
            W4 -= lr * W4.grad
            b4 -= lr * b4.grad
            W1.grad.zero_()
            b1.grad.zero_()
            W2.grad.zero_()
            b2.grad.zero_()
            W3.grad.zero_()
            b3.grad.zero_()
            W4.grad.zero_()
            b4.grad.zero_()

    # Epoch 结束后记录指标
    train_loss = epoch_loss / epoch_total
    train_acc = epoch_correct / epoch_total
    test_acc, _ = evaluate(X_test, y_test)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# ========== 最终评估 ==========
print("\n=== 最终评估 ===")
final_test_acc, test_pred = evaluate(X_test, y_test)
print(f"测试集准确率: {final_test_acc:.4f}")

# ========== 绘制报告 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 损失曲线
axes[0, 0].plot(train_losses, 'b-', label='Train Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. 准确率曲线
axes[0, 1].plot(train_accs, 'b-', label='Train Acc')
axes[0, 1].plot(test_accs, 'r-', label='Test Acc')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. 混淆矩阵
confusion = torch.zeros(10, 10, dtype=torch.int64)
for t, p in zip(y_test.cpu(), test_pred.cpu()):
    confusion[t, p] += 1

im = axes[1, 0].imshow(confusion.numpy(), cmap='Blues')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xticks(range(10))
axes[1, 0].set_yticks(range(10))
plt.colorbar(im, ax=axes[1, 0])

# 4. 样本预测展示
axes[1, 1].axis('off')
axes[1, 1].set_title('Sample Predictions')

# 选取一些样本展示
sample_indices = [0, 1, 2, 3, 4, 5, 6, 7]
for idx, sample_idx in enumerate(sample_indices):
    ax_sub = fig.add_axes([0.55 + (idx % 4) * 0.1, 0.35 - (idx // 4) * 0.15, 0.08, 0.12])
    ax_sub.imshow(X_test[sample_idx].cpu().numpy(), cmap='gray')
    ax_sub.set_title(f'P:{test_pred[sample_idx].item()}\nT:{y_test[sample_idx].item()}', fontsize=8)
    ax_sub.axis('off')

plt.tight_layout()
plt.savefig('04_cnn/Figure_cnn_mnist.png', dpi=150)
plt.show()

print("\n报告已保存到 04_cnn/Figure_cnn_mnist.png")


