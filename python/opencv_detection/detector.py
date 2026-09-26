"""OpenCV detectors and a reusable video-processing pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detection in image coordinates."""

    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class FaceDetector:
    """Viola-Jones face detector shipped with OpenCV."""

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (40, 40),
    ) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"Could not load OpenCV face cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [
            Detection("face", 1.0, int(x), int(y), int(width), int(height))
            for x, y, width, height in faces
        ]


class ObjectDetector:
    """MobileNet-SSD object detector using OpenCV's DNN module."""

    CLASSES = (
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(
        self,
        prototxt: str | Path,
        weights: str | Path,
        confidence_threshold: float = 0.5,
    ) -> None:
        prototxt_path = Path(prototxt)
        weights_path = Path(weights)
        if not prototxt_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "MobileNet-SSD files are missing. Run "
                "`python -m opencv_detection.download_models` first."
            )
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(weights_path))
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self.net.setInput(blob)
        predictions = self.net.forward()
        detections: list[Detection] = []

        for index in range(predictions.shape[2]):
            confidence = float(predictions[0, 0, index, 2])
            if confidence < self.confidence_threshold:
                continue
            class_id = int(predictions[0, 0, index, 1])
            if not 0 < class_id < len(self.CLASSES):
                continue
            box = predictions[0, 0, index, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    self.CLASSES[class_id],
                    confidence,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                )
            )
        return detections


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw labeled boxes on a copy of the frame."""

    annotated = frame.copy()
    for detection in detections:
        x, y, width, height = detection.box
        color = (0, 200, 0) if detection.label == "face" else (0, 140, 255)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        confidence = "" if detection.label == "face" else f" {detection.confidence:.0%}"
        cv2.putText(
            annotated,
            f"{detection.label}{confidence}",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def process_video(
    source: int | str,
    detector: Detector,
    output_path: str | Path | None = None,
    show: bool = True,
    max_frames: int | None = None,
    camera_fps: float = 30.0,
) -> dict[str, float | int]:
    """Process a camera/video stream and optionally preview or save it."""

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or camera_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer: cv2.VideoWriter | None = None
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), codec, source_fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output}")

    frame_count = 0
    detection_count = 0
    started_at = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            annotated = draw_detections(frame, detections)
            frame_count += 1
            detection_count += len(detections)

            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("OpenCV detection - press q or Esc to quit", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    return {
        "frames": frame_count,
        "detections": detection_count,
        "elapsed_seconds": elapsed,
        "processing_fps": frame_count / elapsed,
        "width": width,
        "height": height,
        "source_fps": source_fps,
    }
