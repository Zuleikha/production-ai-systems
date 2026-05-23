"""Behavioral tests for benchmark.py helper functions — no real files loaded."""
import sys
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import create_synthetic_images


def test_creates_correct_number_of_images() -> None:
    # create_synthetic_images must produce exactly the requested count, no more, no fewer.
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        create_synthetic_images(folder, count=3)
        assert len(list(folder.glob("*.jpg"))) == 3


def test_created_images_are_readable_by_opencv() -> None:
    # Each generated image must be a valid JPEG that OpenCV can decode — invalid files would
    # cause silent skips in run_benchmark() and produce misleading empty results.
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        create_synthetic_images(folder, count=2)
        for path in folder.glob("*.jpg"):
            img = cv2.imread(str(path))
            assert img is not None, f"{path.name} could not be decoded"


def test_created_images_have_expected_shape() -> None:
    # Images must be 480x640x3 BGR — detector.preprocess_image() assumes a 3-channel input.
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        create_synthetic_images(folder, count=1)
        img = cv2.imread(str(next(folder.glob("*.jpg"))))
        assert img.shape == (480, 640, 3)


def test_creates_folder_if_missing() -> None:
    # create_synthetic_images must create the target directory when it does not exist.
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "new_subfolder"
        assert not folder.exists()
        create_synthetic_images(folder, count=1)
        assert folder.exists()
