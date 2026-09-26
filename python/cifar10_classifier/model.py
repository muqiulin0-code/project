"""Small residual CNN designed for fast CIFAR-10 experiments."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = False) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.insert(0, nn.MaxPool2d(kernel_size=2, stride=2))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """Two-convolution residual block for equal-resolution feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(inputs + self.block(inputs))


class Cifar10ResNet(nn.Module):
    """A compact ResNet variant that reaches strong CIFAR-10 accuracy quickly.

    The model has about 1.9M parameters at the default width. It is intentionally
    simpler than torchvision's ResNet-18 so the full training workflow remains
    practical on a modest GPU.
    """

    def __init__(self, num_classes: int = 10, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        if width < 8:
            raise ValueError("width must be at least 8")

        self.stem = ConvBlock(3, width)
        self.stage1 = nn.Sequential(
            ConvBlock(width, width * 2, pool=True),
            ResidualBlock(width * 2),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(width * 2, width * 4, pool=True),
            ResidualBlock(width * 4),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(width * 4, width * 8, pool=True),
            ResidualBlock(width * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        return self.head(features)
