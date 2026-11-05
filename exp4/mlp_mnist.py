"""实验四：按参考PDF训练MNIST多层感知机并保存完整结果。"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torchvision
from torch import optim

from mnist_model import MLP
from plotting import save_confusion, save_curves, save_samples
from training import evaluate, make_loaders, predict_samples, train_epoch


def run(base_dir, epochs=30, batch_size=2048, lr=0.01, seed=42):
    started = datetime.now().astimezone().isoformat()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = base_dir / "results"
    output.mkdir(exist_ok=True)
    log_lines = []

    def log(message):
        print(message, flush=True)
        log_lines.append(message)
        (output / "run.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(f"Python {platform.python_version()} | PyTorch {torch.__version__} | torchvision {torchvision.__version__}")
    log(f"Device: {device} | Seed: {seed} | Batch size: {batch_size} | SGD lr: {lr} | Epochs: {epochs}")
    train_loader, test_loader = make_loaders(base_dir / "data", batch_size)
    log(f"MNIST train: {len(train_loader.dataset)} | test: {len(test_loader.dataset)} | input: 28x28")
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    log(str(model))
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    sample_indices = list(range(20))
    # 展示测试集最前20张，不挑选识别正确的样本。
    sample_images = test_loader.dataset.data[:20].numpy() / 255.0
    sample_labels = test_loader.dataset.targets[:20].tolist()
    save_samples(sample_images, sample_labels, output / "01_inputs.png")
    history = []
    timer = time.perf_counter()
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        metrics, confusion = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_acc, **metrics})
        log(f"epoch:{epoch}, loss:{metrics['test_loss_pdf']:.4f}, acc:{metrics['test_accuracy']:.4f}")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - timer
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    images, labels, predictions = predict_samples(model, test_loader.dataset, sample_indices, device)
    save_curves(history, output / "02_learning_curves.png")
    save_samples(images, labels, output / "03_predictions.png", predictions)
    save_confusion(confusion, output / "04_confusion_matrix.png")
    np.savetxt(output / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    torch.save({"state_dict": model.state_dict(), "architecture": [784, 128, 10],
                "epochs": epochs, "seed": seed}, output / "model_final.pt")
    info = {"started_at": started, "finished_at": datetime.now().astimezone().isoformat(),
            "python": platform.python_version(), "python_executable": sys.executable,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "system": platform.platform(), "device": str(device),
            "parameters": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                           "seed": seed, "optimizer": "SGD", "threads": 4,
                           "architecture": [784, 128, 10], "train_shuffle": True,
                           "test_shuffle": True, "transform": "ToTensor"},
            "train_count": len(train_loader.dataset), "test_count": len(test_loader.dataset),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_seconds": elapsed, "final": history[-1],
            "prediction_indices": sample_indices, "labels": labels, "predictions": predictions,
            "confusion_matrix": confusion.tolist(),
            "loss_note": "test_loss_pdf is an unweighted mean of batch means, as in the reference; test_loss_sample_mean is weighted by batch size."}
    (output / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Final: {history[-1]['correct']}/10000 | Accuracy: {history[-1]['test_accuracy']:.2%}")
    log(f"Training/evaluation time: {elapsed:.2f}s | Saved results and model_final.pt")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs、batch-size和lr必须为正数")
    run(Path(__file__).resolve().parent, args.epochs, args.batch_size, args.lr, args.seed)
