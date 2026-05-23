"""Behavioral tests for the FastAPI detection endpoints — images are synthetic numpy arrays."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


def _encode_jpg(arr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", arr)
    return buf.tobytes()


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client: TestClient) -> None:
    # Health must always return 200 — it is polled by container liveness probes.
    resp = client.get("/health")
    assert resp.status_code == 200


def test_detect_valid_image_returns_success(client: TestClient) -> None:
    # A well-formed image upload must produce success=True, confirming the full pipeline ran.
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    resp = client.post("/detect", files={"file": ("test.jpg", _encode_jpg(img), "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_detect_corrupt_bytes_returns_400(client: TestClient) -> None:
    # Non-image bytes must be rejected with 400, not 500 — bad input is a client error.
    resp = client.post("/detect", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 400


def test_detect_response_has_required_fields(client: TestClient) -> None:
    # DetectionResponse schema must include detections, processing_time, and image_shape every time.
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    resp = client.post("/detect", files={"file": ("blank.jpg", _encode_jpg(img), "image/jpeg")})
    body = resp.json()
    for field in ("detections", "processing_time", "image_shape"):
        assert field in body, f"Missing field: {field}"


def test_stats_exposes_device(client: TestClient) -> None:
    # /stats must surface the compute device so operators can confirm GPU is in use.
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert "device" in resp.json()
