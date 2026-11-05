"""将真实训练记录及预测结果保存为图片。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_samples(images, labels, path, predictions=None):
    fig, axes = plt.subplots(4, 5, figsize=(9, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        title = f"Label: {labels[i]}"
        color = "black"
        if predictions is not None:
            title = f"True: {labels[i]}  Pred: {predictions[i]}"
            color = "green" if labels[i] == predictions[i] else "red"
        ax.set_title(title, fontsize=10, color=color)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_curves(history, path):
    epochs = [row["epoch"] + 1 for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train (sample mean)")
    axes[0].plot(epochs, [r["test_loss_pdf"] for r in history], label="Test (PDF batch mean)")
    axes[0].set(xlabel="Epoch (1-30)", ylabel="Cross entropy", title="Loss")
    axes[1].plot(epochs, [r["train_accuracy"] for r in history], label="Train")
    axes[1].plot(epochs, [r["test_accuracy"] for r in history], label="Test")
    axes[1].set(xlabel="Epoch (1-30)", ylabel="Accuracy", title="Accuracy", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion(matrix, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=8,
                color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit",
           ylabel="True digit", title="MNIST test set (10,000 images)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
