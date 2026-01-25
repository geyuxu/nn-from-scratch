"""
VGG16 for CIFAR-10 v2 (类结构，不使用 nn.Module)

改进版本（解决过拟合）：
- 数据增强：随机水平翻转、随机裁剪
- 减小 FC 层：4096 -> 512
- L2 正则化（权重衰减）
- 日志输出到文件

VGG16 架构 (16 weight layers):
- Block 1: Conv64 -> Conv64 -> MaxPool
- Block 2: Conv128 -> Conv128 -> MaxPool
- Block 3: Conv256 -> Conv256 -> Conv256 -> MaxPool
- Block 4: Conv512 -> Conv512 -> Conv512 -> MaxPool
- Block 5: Conv512 -> Conv512 -> Conv512 -> MaxPool
- FC: 512 -> 512 -> 512 -> 10 (减小)
"""
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import sys
from datetime import datetime


# ==================== 日志系统 ====================

class Logger:
    """同时输出到控制台和文件的日志类"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')
        self.log(f"日志开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)

    def log(self, message):
        """同时打印到控制台和写入文件"""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log("=" * 60)
        self.log(f"日志结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_file.close()


# 创建日志文件
log_filename = f"04_cnn/vgg16_cifar10_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = Logger(log_filename)


# ==================== 设备设置 ====================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.log("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.log("Using CUDA GPU")
else:
    device = torch.device("cpu")
    logger.log("Using CPU")


# ==================== 基础函数 ====================

def conv2d_multichannel(X, W, b, padding=0):
    """
    向量化多通道卷积（支持 padding）
    X: (batch, in_c, H, W)
    W: (out_c, in_c, kH, kW)
    b: (out_c,)
    padding: int, 在边缘填充 0 的层数
    return: (batch, out_c, out_H, out_W)
    """
    if padding > 0:
        X = torch.nn.functional.pad(X, (padding, padding, padding, padding))

    batch, in_c, H, W_in = X.shape
    out_c, _, kH, kW = W.shape
    out_H, out_W = H - kH + 1, W_in - kW + 1

    patches = X.unfold(2, kH, 1).unfold(3, kW, 1)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch, out_H * out_W, in_c * kH * kW)
    W_flat = W.reshape(out_c, in_c * kH * kW)
    out = patches @ W_flat.T + b

    return out.reshape(batch, out_H, out_W, out_c).permute(0, 3, 1, 2)


def maxpool2d(X, size=2):
    """向量化池化"""
    batch, c, H, W = X.shape
    out_H, out_W = H // size, W // size
    X = X[:, :, :out_H * size, :out_W * size]
    X = X.reshape(batch, c, out_H, size, out_W, size)
    return X.max(dim=3).values.max(dim=4).values


def relu(x):
    return torch.maximum(x, torch.tensor(0.0, device=x.device))


def dropout(x, p=0.5, training=True):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)


def softmax(x):
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return exp_x / exp_x.sum(dim=1, keepdim=True)


def cross_entropy(y_pred, y_true):
    batch = y_pred.shape[0]
    return -torch.log(y_pred[range(batch), y_true] + 1e-8).mean()


# ==================== 数据增强 ====================

def random_horizontal_flip(X, p=0.5):
    """随机水平翻转"""
    mask = torch.rand(X.shape[0], device=X.device) < p
    X[mask] = X[mask].flip(dims=[3])  # 翻转 W 维度
    return X


def random_crop(X, padding=4):
    """随机裁剪（先 padding 再裁剪回原尺寸）"""
    batch, c, h, w = X.shape
    # Padding
    X_padded = torch.nn.functional.pad(X, (padding, padding, padding, padding))
    # 随机裁剪位置
    crops = []
    for i in range(batch):
        top = torch.randint(0, 2 * padding + 1, (1,)).item()
        left = torch.randint(0, 2 * padding + 1, (1,)).item()
        crops.append(X_padded[i:i+1, :, top:top+h, left:left+w])
    return torch.cat(crops, dim=0)


def augment_batch(X):
    """应用数据增强"""
    X = random_horizontal_flip(X.clone(), p=0.5)
    X = random_crop(X, padding=4)
    return X


# ==================== 层类 ====================

class Conv2D:
    """卷积层（支持 padding）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, device='cpu'):
        # He 初始化
        scale = (2.0 / (in_channels * kernel_size * kernel_size)) ** 0.5
        self.W = (torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device) * scale).requires_grad_(True)
        self.b = torch.zeros(out_channels, device=device, requires_grad=True)
        self.padding = padding

    def forward(self, X):
        return conv2d_multichannel(X, self.W, self.b, padding=self.padding)

    def parameters(self):
        return [self.W, self.b]


class Dense:
    """全连接层"""
    def __init__(self, in_features, out_features, device='cpu'):
        scale = (2.0 / in_features) ** 0.5
        self.W = (torch.randn(in_features, out_features, device=device) * scale).requires_grad_(True)
        self.b = torch.zeros(out_features, device=device, requires_grad=True)

    def forward(self, X):
        return X @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


# ==================== VGG16 网络 ====================

class VGG16:
    """
    VGG16 网络 for CIFAR-10

    架构 (16 weight layers):
    Block 1: Conv64 -> ReLU -> Conv64 -> ReLU -> MaxPool (32->16)
    Block 2: Conv128 -> ReLU -> Conv128 -> ReLU -> MaxPool (16->8)
    Block 3: Conv256 -> ReLU -> Conv256 -> ReLU -> Conv256 -> ReLU -> MaxPool (8->4)
    Block 4: Conv512 -> ReLU -> Conv512 -> ReLU -> Conv512 -> ReLU -> MaxPool (4->2)
    Block 5: Conv512 -> ReLU -> Conv512 -> ReLU -> Conv512 -> ReLU -> MaxPool (2->1)
    FC: 512 -> 512 -> ReLU -> Dropout -> 512 -> ReLU -> Dropout -> 10 -> Softmax (减小FC防过拟合)
    """
    def __init__(self, device='cpu'):
        self.device = device

        # Block 1: 3 -> 64
        self.conv1_1 = Conv2D(3, 64, 3, padding=1, device=device)
        self.conv1_2 = Conv2D(64, 64, 3, padding=1, device=device)

        # Block 2: 64 -> 128
        self.conv2_1 = Conv2D(64, 128, 3, padding=1, device=device)
        self.conv2_2 = Conv2D(128, 128, 3, padding=1, device=device)

        # Block 3: 128 -> 256
        self.conv3_1 = Conv2D(128, 256, 3, padding=1, device=device)
        self.conv3_2 = Conv2D(256, 256, 3, padding=1, device=device)
        self.conv3_3 = Conv2D(256, 256, 3, padding=1, device=device)

        # Block 4: 256 -> 512
        self.conv4_1 = Conv2D(256, 512, 3, padding=1, device=device)
        self.conv4_2 = Conv2D(512, 512, 3, padding=1, device=device)
        self.conv4_3 = Conv2D(512, 512, 3, padding=1, device=device)

        # Block 5: 512 -> 512
        self.conv5_1 = Conv2D(512, 512, 3, padding=1, device=device)
        self.conv5_2 = Conv2D(512, 512, 3, padding=1, device=device)
        self.conv5_3 = Conv2D(512, 512, 3, padding=1, device=device)

        # FC layers (减小以防止过拟合)
        # 32 -> pool -> 16 -> pool -> 8 -> pool -> 4 -> pool -> 2 -> pool -> 1
        # FC 输入: 512 * 1 * 1 = 512
        self.fc1 = Dense(512, 512, device)  # 减小: 4096 -> 512
        self.fc2 = Dense(512, 512, device)  # 减小: 4096 -> 512
        self.fc3 = Dense(512, 10, device)

        self.training = True

    def forward(self, X):
        # Block 1
        h = relu(self.conv1_1.forward(X))
        h = relu(self.conv1_2.forward(h))
        h = maxpool2d(h)  # 32 -> 16

        # Block 2
        h = relu(self.conv2_1.forward(h))
        h = relu(self.conv2_2.forward(h))
        h = maxpool2d(h)  # 16 -> 8

        # Block 3
        h = relu(self.conv3_1.forward(h))
        h = relu(self.conv3_2.forward(h))
        h = relu(self.conv3_3.forward(h))
        h = maxpool2d(h)  # 8 -> 4

        # Block 4
        h = relu(self.conv4_1.forward(h))
        h = relu(self.conv4_2.forward(h))
        h = relu(self.conv4_3.forward(h))
        h = maxpool2d(h)  # 4 -> 2

        # Block 5
        h = relu(self.conv5_1.forward(h))
        h = relu(self.conv5_2.forward(h))
        h = relu(self.conv5_3.forward(h))
        h = maxpool2d(h)  # 2 -> 1

        # Flatten
        h = h.view(h.shape[0], -1)  # (batch, 512)

        # FC layers
        h = relu(self.fc1.forward(h))
        h = dropout(h, p=0.5, training=self.training)
        h = relu(self.fc2.forward(h))
        h = dropout(h, p=0.5, training=self.training)
        h = self.fc3.forward(h)

        return softmax(h)

    def parameters(self):
        params = []
        for layer in [self.conv1_1, self.conv1_2,
                      self.conv2_1, self.conv2_2,
                      self.conv3_1, self.conv3_2, self.conv3_3,
                      self.conv4_1, self.conv4_2, self.conv4_3,
                      self.conv5_1, self.conv5_2, self.conv5_3,
                      self.fc1, self.fc2, self.fc3]:
            params.extend(layer.parameters())
        return params

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# ==================== 数据加载 ====================

logger.log("加载 CIFAR-10 数据集...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10("./data", train=False, transform=transform)

# 转换为张量
logger.log("转换数据到张量...")
X_train = torch.stack([train_data[i][0] for i in range(len(train_data))]).to(device)
y_train = torch.tensor([train_data[i][1] for i in range(len(train_data))]).to(device)
X_test = torch.stack([test_data[i][0] for i in range(len(test_data))]).to(device)
y_test = torch.tensor([test_data[i][1] for i in range(len(test_data))]).to(device)

logger.log(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
logger.log(f"类别: {train_data.classes}")


# ==================== 训练 ====================

logger.log("\n初始化 VGG16...")
model = VGG16(device=device)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
logger.log(f"总参数量: {total_params:,} ({total_params/1e6:.1f}M)")

batch_size = 64  # VGG16 较大，用小 batch
epochs = 30  # 增加 epoch，因为数据增强需要更多训练
lr = 0.01
weight_decay = 5e-4  # L2 正则化

train_losses = []
train_accs = []
test_accs = []
best_test_acc = 0

logger.log("\n开始训练...")
logger.log(f"超参数: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}")
logger.log("改进: 数据增强(翻转+裁剪) + 小FC(512) + L2正则化")
logger.log("-" * 60)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    # 打乱数据
    perm = torch.randperm(len(X_train))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    for i in range(0, len(X_train), batch_size):
        X = X_train_shuffled[i:i+batch_size]
        y = y_train_shuffled[i:i+batch_size]

        # 数据增强
        X = augment_batch(X)

        # 前向
        out = model.forward(X)
        pred = out.argmax(dim=1)

        # 损失
        loss = cross_entropy(out, y)

        # 统计
        epoch_loss += loss.item() * len(y)
        epoch_correct += (pred == y).sum().item()
        epoch_total += len(y)

        # 反向
        loss.backward()

        # 更新参数 (带 L2 正则化)
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * (param.grad + weight_decay * param)  # L2 正则化
                param.grad.zero_()

        # 打印进度
        if (i // batch_size) % 100 == 0:
            logger.log(f"  Batch {i//batch_size}/{len(X_train)//batch_size}, loss={loss.item():.4f}")

    # Epoch 统计
    train_loss = epoch_loss / epoch_total
    train_acc = epoch_correct / epoch_total

    # 测试集评估（分批处理，避免内存溢出）
    model.eval()
    test_correct = 0
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            out = model.forward(X_batch)
            pred = out.argmax(dim=1)
            test_correct += (pred == y_batch).sum().item()
            test_preds.append(pred)
    test_acc = test_correct / len(X_test)
    test_pred = torch.cat(test_preds)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # 记录最佳
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        logger.log(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f} [NEW BEST]")
    else:
        logger.log(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    # 学习率衰减
    if (epoch + 1) % 15 == 0:
        lr *= 0.1
        logger.log(f"  学习率衰减到 {lr}")


# ==================== 最终评估 ====================

logger.log("\n" + "=" * 60)
logger.log("=== 最终评估 ===")
logger.log(f"最终测试准确率: {test_accs[-1]:.4f}")
logger.log(f"最佳测试准确率: {best_test_acc:.4f} (Epoch {best_epoch})")
logger.log(f"最终训练准确率: {train_accs[-1]:.4f}")
logger.log(f"过拟合差距: {train_accs[-1] - test_accs[-1]:.4f} (越小越好)")


# ==================== 绘制报告 ====================

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

classes = train_data.classes
sample_indices = [0, 1, 2, 3, 4, 5, 6, 7]
for idx, sample_idx in enumerate(sample_indices):
    ax_sub = fig.add_axes([0.55 + (idx % 4) * 0.1, 0.35 - (idx // 4) * 0.15, 0.08, 0.12])
    # 反归一化显示
    img = X_test[sample_idx].cpu()
    img = img * torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    img = img.permute(1, 2, 0).clamp(0, 1)
    ax_sub.imshow(img.numpy())
    ax_sub.set_title(f'P:{classes[test_pred[sample_idx]]}\nT:{classes[y_test[sample_idx]]}', fontsize=6)
    ax_sub.axis('off')

plt.tight_layout()
plt.savefig('04_cnn/Figure_vgg16_cifar10_v2.png', dpi=150)
plt.show()

logger.log("\n报告已保存到 04_cnn/Figure_vgg16_cifar10_v2.png")
logger.log(f"日志已保存到 {log_filename}")
logger.close()
