"""Evaluate a saved CIFAR-10 checkpoint on the official test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data import build_dataloaders
from .model import Cifar10ResNet
from .train import evaluate, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = Cifar10ResNet(**checkpoint.get("model", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )
    loss, accuracy = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    result = {"checkpoint": str(args.checkpoint), "test_loss": loss, "test_accuracy": accuracy}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
