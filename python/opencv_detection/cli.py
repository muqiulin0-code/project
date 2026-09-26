"""Command-line interface for OpenCV face/object detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detector import FaceDetector, ObjectDetector, process_video


def parse_source(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--mode", choices=("face", "object"), default="face")
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--prototxt",
        type=Path,
        default=Path("models/MobileNetSSD_deploy.prototxt"),
    )
    parser.add_argument(
        "--caffemodel",
        type=Path,
        default=Path("models/MobileNetSSD_deploy.caffemodel"),
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the video window; use --no-show for headless runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = (
        FaceDetector()
        if args.mode == "face"
        else ObjectDetector(args.prototxt, args.caffemodel, args.confidence)
    )
    stats = process_video(
        source=parse_source(args.source),
        detector=detector,
        output_path=args.output,
        show=args.show,
        max_frames=args.max_frames,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
