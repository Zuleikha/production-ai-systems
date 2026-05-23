"""Behavioral tests for ObjectDetector — all images are synthetic numpy arrays."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.detector import DetectionResult, ObjectDetector

_COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@pytest.fixture(scope="module")
def detector() -> ObjectDetector:
    return ObjectDetector(
        model_path="models/best_model.pth",
        class_names=_COCO_CLASSES,
        confidence_threshold=0.5,
        nms_threshold=0.2,
    )


def test_detect_returns_detection_result(detector: ObjectDetector) -> None:
    # Verifies the public return type contract — callers depend on DetectionResult attributes.
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect_single_image(image)
    assert isinstance(result, DetectionResult)


def test_processing_time_is_positive(detector: ObjectDetector) -> None:
    # Ensures timing is wired up; a zero time means the measurement path is broken.
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect_single_image(image)
    assert result.processing_time > 0


def test_confidence_filter_respected(detector: ObjectDetector) -> None:
    # No detection score should leak below the threshold — confidence filtering is a correctness guarantee.
    detector.confidence_threshold = 0.99
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect_single_image(image)
    assert all(s >= 0.99 for s in result.scores), "Score below confidence threshold leaked through"
    detector.confidence_threshold = 0.5


def test_output_arrays_have_equal_length(detector: ObjectDetector) -> None:
    # boxes, scores, and classes must stay in sync — mismatched lengths cause index errors downstream.
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect_single_image(image)
    assert len(result.boxes) == len(result.scores) == len(result.classes)


def test_inference_times_accumulate(detector: ObjectDetector) -> None:
    # Each call must append to inference_times — benchmark.py and /stats depend on this history.
    before = len(detector.inference_times)
    detector.detect_single_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert len(detector.inference_times) == before + 1
