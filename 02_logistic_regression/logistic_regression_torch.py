import torch
from viz_utils import plot_training_classification
# 1 make datas
torch.manual_seed(42)
X = torch.randn(100,1)
y_true = (X>0).float()

print("--------")

# 2 init paramaters
W = torch.randn(1,1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

print("--------")

# 3 超参数
lr = 0.01
epoch_count = 1000

# train loop
losses = []
for epoch in range(epoch_count):
    # forwared
    t = X @ W + b
    y_pred = 1 / (1 + torch.exp(-t)) # sigmoid

    # loss 交叉熵
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()
    losses.append(loss.item())

    # backward
    loss.backward()

    # update
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()

    if epoch % 20 == 0:
        acc = ((y_pred > 0.5) == y_true).float().mean()  # 准确率
        print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")

print(f"\n学到的参数：W={W.item():.4f} (真实=3),b={b.item():.4f} (真实=2)")

# 可视化
plot_training_classification(losses, X, y_true, W, b)