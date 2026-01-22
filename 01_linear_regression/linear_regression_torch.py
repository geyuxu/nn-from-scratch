import torch
from viz_utils import plot_training
# 1 make datas
torch.manual_seed(42)
X = torch.randn(100,1)
y_true = 3 * X + 2 + torch.randn(100,1) * 0.3   # y = 3x + 2 + noise

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
    y_pred = X @ W + b

    # loss
    loss = ((y_pred - y_true)**2).mean()
    losses.append(loss.item())

    # backward
    loss.backward()

    # update
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, W={W.item():.4f}, b={b.item():.4f}")

print(f"\n学到的参数：W={W.item():.4f} (真实=3),b={b.item():.4f} (真实=2)")

# 可视化
plot_training(losses,X,y_true,X@W+b)