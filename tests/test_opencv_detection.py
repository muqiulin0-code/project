import numpy as np
import pytest
from opencv_detection.cli import parse_source
from opencv_detection.detector import Detection, FaceDetector, ObjectDetector, draw_detections


def test_camera_index_and_video_path_parsing() -> None:
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source("sample.mp4") == "sample.mp4"


def test_face_detector_runs_on_blank_frame() -> None:
    detector = FaceDetector()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_draw_detections_does_not_mutate_input() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    annotated = draw_detections(frame, [Detection("person", 0.9, 10, 10, 30, 40)])
    np.testing.assert_array_equal(frame, original)
    assert np.count_nonzero(annotated) > 0
    assert annotated.shape == frame.shape


def test_object_detector_reports_missing_model(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download_models"):
        ObjectDetector(tmp_path / "missing.prototxt", tmp_path / "missing.caffemodel")
