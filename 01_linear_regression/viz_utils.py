import matplotlib.pyplot as plt
import numpy as np


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
    """绘制拟合效果"""
    plt.scatter(to_numpy(X), to_numpy(y_true), label='data')
    plt.plot(to_numpy(X), to_numpy(y_pred), 'r', label='fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Fitting Result')
    plt.legend()


def plot_training(losses, X, y_true, y_pred):
    """绘制完整训练结果：loss曲线 + 拟合效果"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plot_loss(losses)

    plt.subplot(1, 2, 2)
    plot_fit(X, y_true, y_pred)

    plt.tight_layout()
    plt.show()
