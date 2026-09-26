"""CIFAR-10 image classification exercise."""

from .data import NumpyCifar10Dataset
from .model import Cifar10ResNet

__all__ = ["Cifar10ResNet", "NumpyCifar10Dataset"]
