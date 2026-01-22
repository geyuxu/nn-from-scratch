import matplotlib.pyplot as plt
import numpy as np
import torch


def to_numpy(x):
    """将tensor或numpy数组统一转为numpy"""
    if hasattr(x, 'detach'):
        return x.detach().numpy()
    return np.asarray(x)


def plot_loss(losses):
    """绘制loss曲线"""
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')


def plot_fit(X, y_true, y_pred):
    """绘制拟合效果（回归）"""
    plt.scatter(to_numpy(X), to_numpy(y_true), label='data')
    plt.plot(to_numpy(X), to_numpy(y_pred), 'r', label='fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Fitting Result')
    plt.legend()


def plot_classification(X, y_true, W, b):
    """绘制分类效果（逻辑回归）"""
    X_np = to_numpy(X).flatten()
    y_np = to_numpy(y_true).flatten()
    
    # 散点
    plt.scatter(X_np[y_np == 0], y_np[y_np == 0], label='class 0', c='blue')
    plt.scatter(X_np[y_np == 1], y_np[y_np == 1], label='class 1', c='red')
    
    # sigmoid 曲线
    X_range = torch.linspace(X_np.min() - 0.5, X_np.max() + 0.5, 100).reshape(-1, 1)
    with torch.no_grad():
        y_prob = 1 / (1 + torch.exp(-(X_range @ W + b)))
    plt.plot(to_numpy(X_range), to_numpy(y_prob), 'g-', label='sigmoid')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='threshold')
    
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.title('Classification Result')
    plt.legend()


def plot_training(losses, X, y_true, y_pred):
    """绘制完整训练结果：loss曲线 + 拟合效果（回归）"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plot_loss(losses)

    plt.subplot(1, 2, 2)
    plot_fit(X, y_true, y_pred)

    plt.tight_layout()
    plt.show()


def plot_training_classification(losses, X, y_true, W, b):
    """绘制完整训练结果：loss曲线 + 分类效果（逻辑回归）"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plot_loss(losses)

    plt.subplot(1, 2, 2)
    plot_classification(X, y_true, W, b)

    plt.tight_layout()
    plt.show()