import numpy as np
from viz_utils import plot_training

np.random.seed(42)
X = np.random.randn(100,1)
y_true = 3 * X + 2 + np.random.randn(100,1) * 0.3

W = np.random.randn(1,1)
b = np.zeros(1)

lr = 0.1
epoch_count = 100

losses = []
for epoch in range(epoch_count):
    y_pred = X @ W + b

    loss = ((y_pred - y_true) ** 2).mean()
    #print(loss)
    losses.append(loss)

    n = len(X)
    dL_dy = 2 * (y_pred - y_true) / n
    dL_dW = X.T @ dL_dy
    dL_db = dL_dy.sum()

    W -= lr * dL_dW
    b -= lr * dL_db

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, W={W[0,0]:.4f}, b={b[0]:.4f}")

print(f"\n学到的参数：W={W[0,0]:.4f} (真实=3), b={b[0]:.4f} (真实=2)")

# 可视化
plot_training(losses,X,y_true,X@W+b)