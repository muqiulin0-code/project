import pytest
import torch
from cifar10_classifier import Cifar10ResNet


def test_model_output_shape_and_backpropagation() -> None:
    model = Cifar10ResNet(width=8)
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.tensor([0, 9])
    logits = model(inputs)
    assert logits.shape == (2, 10)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_invalid_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="width"):
        Cifar10ResNet(width=4)
