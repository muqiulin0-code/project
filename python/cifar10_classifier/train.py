"""Train and evaluate the compact CIFAR-10 classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from .data import build_dataloaders
from .model import Cifar10ResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cifar10"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--validation-size", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-accuracy", type=float, default=0.70)
    parser.add_argument("--limit-train", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--limit-val", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    return device


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    use_amp = device.type == "cuda"

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size

    return running_loss / seen, correct / seen


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size

    return running_loss / seen, correct / seen


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_curves(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, [row["train_loss"] for row in history], label="train")
    loss_axis.plot(epochs, [row["val_loss"] for row in history], label="validation")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("CIFAR-10 loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    accuracy_axis.plot(epochs, [row["val_accuracy"] * 100 for row in history], label="validation")
    accuracy_axis.axhline(70, color="tab:red", linestyle="--", label="70% target")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_title("CIFAR-10 accuracy")
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 < args.target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_size=args.validation_size,
        seed=args.seed,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        download=not args.no_download,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "the training dataloader is empty; reduce batch_size or increase data limits"
        )

    model = Cifar10ResNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(
        f"device={device} train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    history: list[dict[str, float]] = []
    best_val_accuracy = 0.0
    best_epoch = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}/{args.epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.2%} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.2%} seconds={row['seconds']:.1f}",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "model": {"num_classes": 10, "width": 32, "dropout": 0.2},
                    "normalization": {
                        "mean": [0.4914, 0.4822, 0.4465],
                        "std": [0.2470, 0.2435, 0.2616],
                    },
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    total_seconds = time.perf_counter() - started_at
    passed = test_accuracy > args.target_accuracy

    metrics: dict[str, Any] = {
        "exercise": "PyTorch CIFAR-10 classifier",
        "acceptance": f"test accuracy > {args.target_accuracy:.0%}",
        "passed": passed,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "training_seconds": total_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": checkpoint_path.name,
    }

    save_history(history, args.output_dir)
    save_curves(history, args.output_dir)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(f"Acceptance failed: test accuracy {test_accuracy:.2%} is not above 70%")


if __name__ == "__main__":
    main()
