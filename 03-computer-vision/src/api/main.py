"""
FastAPI backend for the object detection service.
Accepts image uploads and returns bounding boxes, class labels, and confidence scores.
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
import sys

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(Path(__file__).parent.parent / "inference"))

from detector import ObjectDetector, DetectionResult  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO class names — index 0 is background, N/A entries are reserved IDs.
COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
]

app_state: Dict = {"detector": None, "start_time": time.time(), "request_count": 0}


class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict]
    processing_time: float
    image_shape: List[int]
    message: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising object detection service...")
    try:
        app_state["detector"] = ObjectDetector(
            model_path="models/best_model.pth",
            class_names=COCO_CLASSES,
            confidence_threshold=0.5,
            nms_threshold=0.2,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialise detector: {e}")
    yield


app = FastAPI(title="Object Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_detections(result: DetectionResult) -> List[Dict]:
    """Convert a DetectionResult into a JSON-serialisable list, dropping reserved classes."""
    detector = app_state["detector"]
    detections = []
    for i in range(len(result.boxes)):
        class_id = int(result.classes[i])
        class_name = (
            detector.class_names[class_id]
            if class_id < len(detector.class_names)
            else "unknown"
        )
        if class_name not in ("background", "N/A"):
            detections.append({
                "bbox": result.boxes[i].tolist(),
                "confidence": float(result.scores[i]),
                "class_name": class_name,
                "class_id": class_id,
            })
    return detections


@app.get("/")
async def root():
    return {"message": "Object Detection API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": app_state["detector"] is not None}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    if not app_state["detector"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        app_state["detector"].confidence_threshold = confidence_threshold
        result = app_state["detector"].detect_single_image(image)
        detections = format_detections(result)
        app_state["request_count"] += 1

        return DetectionResponse(
            success=True,
            detections=detections,
            processing_time=result.processing_time,
            image_shape=list(result.image_shape),
            message=f"Detected {len(detections)} objects",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    if app_state["detector"] is None:
        return {"error": "Detector not loaded"}
    return {
        "total_requests": app_state["request_count"],
        "uptime_seconds": time.time() - app_state["start_time"],
        "device": str(app_state["detector"].device),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
