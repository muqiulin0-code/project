"""CIFAR-10 dataset and dataloader construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


class NumpyCifar10Dataset(Dataset):
    """CIFAR-10 arrays produced by ``scripts/prepare_hf_cifar10.py``."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose,
    ) -> None:
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError(f"unexpected image array shape: {images.shape}")
        if labels.shape != (len(images),):
            raise ValueError(f"unexpected label array shape: {labels.shape}")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(np.asarray(self.images[index], dtype=np.uint8), mode="RGB")
        return self.transform(image), int(self.labels[index])


def build_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    validation_size: int = 5_000,
    seed: int = 42,
    limit_train: int | None = None,
    limit_val: int | None = None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create augmented train, clean validation, and clean test dataloaders."""

    data_path = Path(data_dir)
    numpy_dir = data_path / "cifar10_numpy"
    numpy_files = {
        "train_images": numpy_dir / "train_images.npy",
        "train_labels": numpy_dir / "train_labels.npy",
        "test_images": numpy_dir / "test_images.npy",
        "test_labels": numpy_dir / "test_labels.npy",
    }
    if all(path.is_file() for path in numpy_files.values()):
        print(f"using NumPy CIFAR-10 mirror: {numpy_dir}")
        train_images = np.load(numpy_files["train_images"], mmap_mode="r")
        train_labels = np.load(numpy_files["train_labels"], mmap_mode="r")
        test_images = np.load(numpy_files["test_images"], mmap_mode="r")
        test_labels = np.load(numpy_files["test_labels"], mmap_mode="r")
        train_source = NumpyCifar10Dataset(
            train_images, train_labels, build_train_transform()
        )
        clean_source = NumpyCifar10Dataset(
            train_images, train_labels, build_test_transform()
        )
        test_dataset = NumpyCifar10Dataset(
            test_images, test_labels, build_test_transform()
        )
    else:
        train_source = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=download,
            transform=build_train_transform(),
        )
        clean_source = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=download,
            transform=build_test_transform(),
        )
        test_dataset = datasets.CIFAR10(
            root=data_path,
            train=False,
            download=download,
            transform=build_test_transform(),
        )

    if not 0 < validation_size < len(train_source):
        raise ValueError("validation_size must be between 1 and the training set size")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    if limit_train is not None:
        if limit_train <= 0:
            raise ValueError("limit_train must be positive")
        train_indices = train_indices[:limit_train]
    if limit_val is not None:
        if limit_val <= 0:
            raise ValueError("limit_val must be positive")
        val_indices = val_indices[:limit_val]

    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_source, train_indices),
        shuffle=True,
        drop_last=True,
        **common_options,
    )
    val_loader = DataLoader(
        Subset(clean_source, val_indices),
        shuffle=False,
        **common_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_options,
    )
    return train_loader, val_loader, test_loader
