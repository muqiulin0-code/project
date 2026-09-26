"""Download the MobileNet-SSD files used for object detection."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.error import URLError

PROTOTXT_URLS = (
    "https://ghproxy.net/https://raw.githubusercontent.com/chuanqi305/"
    "MobileNet-SSD/master/deploy.prototxt",
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
)
WEIGHTS_URLS = (
    "https://ghproxy.net/https://raw.githubusercontent.com/chuanqi305/"
    "MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
    "mobilenet_iter_73000.caffemodel",
)


def download(urls: tuple[str, ...], destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    errors: list[str] = []
    for url in urls:
        print(f"downloading: {url}\n         to: {destination}")
        try:
            curl = shutil.which("curl")
            if curl:
                subprocess.run(
                    [
                        curl,
                        "-4",
                        "-L",
                        "--fail",
                        "--retry",
                        "3",
                        "--retry-delay",
                        "2",
                        "--output",
                        str(temporary),
                        url,
                    ],
                    check=True,
                )
            else:
                with urllib.request.urlopen(url, timeout=60) as source:
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(destination)
            return
        except (OSError, subprocess.CalledProcessError, URLError) as error:
            errors.append(f"{url}: {error}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError("all model download mirrors failed:\n" + "\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(PROTOTXT_URLS, args.model_dir / "MobileNetSSD_deploy.prototxt")
    download(WEIGHTS_URLS, args.model_dir / "MobileNetSSD_deploy.caffemodel")


if __name__ == "__main__":
    main()
