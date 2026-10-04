#!/usr/bin/env python3
"""Validate that the active interpreter is the shared conda PyTorch environment."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

EXPECTED_ENV = os.environ.get("CONDA_ENV", "pytorch")


def main() -> None:
    failures: list[str] = []
    prefix = Path(sys.prefix)
    # Use the interpreter prefix itself, not CONDA_DEFAULT_ENV. Some shell
    # hooks leave CONDA_DEFAULT_ENV=pytorch while conda run actually selects
    # the base interpreter, which previously made this check pass incorrectly.
    environment_name = prefix.name
    if environment_name != EXPECTED_ENV:
        failures.append(
            f"active conda environment is '{environment_name}', expected '{EXPECTED_ENV}'"
        )

    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Environment: {environment_name} ({prefix})")

    modules = ("torch", "torchvision", "numpy", "cv2", "matplotlib", "pytest")
    imported: dict[str, object] = {}
    for module_name in modules:
        try:
            imported[module_name] = __import__(module_name)
        except ImportError as error:
            failures.append(f"missing module '{module_name}': {error}")

    if "torch" in imported:
        torch = imported["torch"]
        cuda_available = bool(torch.cuda.is_available())
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    if "torchvision" in imported:
        print(f"torchvision: {imported['torchvision'].__version__}")
    if "numpy" in imported:
        print(f"NumPy: {imported['numpy'].__version__}")
    if "cv2" in imported:
        version = imported["cv2"].__version__
        print(f"OpenCV: {version}")
        if not version.startswith("4."):
            failures.append(f"OpenCV 4.x is required, found {version}")
    if "matplotlib" in imported:
        print(f"Matplotlib: {imported['matplotlib'].__version__}")

    try:
        print(f"pytest: {importlib.metadata.version('pytest')}")
        print(f"project package: {importlib.metadata.version('embodied-ai-stage0')}")
    except importlib.metadata.PackageNotFoundError as error:
        failures.append(f"missing package metadata: {error}")

    if failures:
        print("\nEnvironment check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nActivate the shared environment with:\n"
            f"  conda activate {EXPECTED_ENV}\n"
            "Then run:\n"
            "  python -m pip install -e . --no-deps",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
