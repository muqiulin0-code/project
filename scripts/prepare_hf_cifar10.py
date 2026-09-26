#!/usr/bin/env python3
"""Convert the Hugging Face CIFAR-10 parquet mirror to memory-mappable NumPy files.

This optional fallback is useful when the original Toronto server is slow.
It requires ``pyarrow`` only during conversion:

    python -m pip install pyarrow
    python scripts/prepare_hf_cifar10.py

Training automatically detects ``data/cifar10_numpy``. The standard
``torchvision.CIFAR10`` path remains the default when that directory is absent.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_URL = (
    "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/"
    "plain_text/train-00000-of-00001.parquet"
)
TEST_URL = (
    "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/"
    "plain_text/test-00000-of-00001.parquet"
)


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading: {url}\n         to: {destination}")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def convert(parquet_path: Path, images_path: Path, labels_path: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["img", "label"])
    encoded_images = table.column("img").to_pylist()
    labels = np.asarray(table.column("label").to_numpy(), dtype=np.int64)
    images = np.empty((len(encoded_images), 32, 32, 3), dtype=np.uint8)

    for index, encoded in enumerate(encoded_images):
        with Image.open(io.BytesIO(encoded["bytes"])) as image:
            rgb = image.convert("RGB")
            if rgb.size != (32, 32):
                raise ValueError(f"unexpected image shape at row {index}: {rgb.size}")
            images[index] = np.asarray(rgb, dtype=np.uint8)

    np.save(images_path, images)
    np.save(labels_path, labels)
    return {"samples": len(labels), "height": 32, "width": 32, "channels": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--keep-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_dir = args.data_dir / "hf-cifar10"
    output_dir = args.data_dir / "cifar10_numpy"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_parquet = parquet_dir / "train.parquet"
    test_parquet = parquet_dir / "test.parquet"

    download(TRAIN_URL, train_parquet)
    download(TEST_URL, test_parquet)

    metadata = {
        "source": "https://huggingface.co/datasets/uoft-cs/cifar10",
        "mirror": "https://hf-mirror.com/datasets/uoft-cs/cifar10",
        "train": convert(
            train_parquet, output_dir / "train_images.npy", output_dir / "train_labels.npy"
        ),
        "test": convert(
            test_parquet, output_dir / "test_images.npy", output_dir / "test_labels.npy"
        ),
        "color_order": "RGB",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    if not args.keep_parquet:
        for path in (train_parquet, test_parquet):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
