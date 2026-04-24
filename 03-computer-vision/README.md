# Computer Vision — Object Detection

Real-time object detection using Faster R-CNN ResNet-50 FPN, served via FastAPI with a static HTML frontend.

## Architecture

```
Image upload (multipart)
      ↓
FastAPI /detect endpoint
      ↓
ObjectDetector (detector.py)
  BGR → RGB → ToTensor → Faster R-CNN inference
  → confidence filter → torchvision NMS
      ↓
DetectionResponse (JSON bounding boxes, scores, class names)
      ↓
HTML frontend (visualization)
```

## Tech Stack

| Component | Tool |
|---|---|
| Detection model | Faster R-CNN ResNet-50 FPN (COCO pretrained) |
| DL framework | PyTorch + Torchvision |
| Image processing | OpenCV, Pillow |
| API | FastAPI + async lifespan |
| Visualization | Matplotlib, Seaborn, OpenCV drawing |

## Quick Start

```bash
pip install -r requirements.txt

# Start API (downloads COCO pretrained weights on first run)
uvicorn src.api.main:app --reload

# Open frontend
open src/frontend/app.html
```

API docs: http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + model load status |
| POST | `/detect` | Detect objects in uploaded image |
| GET | `/stats` | Request count + uptime + device |

`/detect` accepts `multipart/form-data` with an image file and optional `confidence_threshold` query param (default 0.5).

## Model Behaviour

- Falls back to COCO pretrained weights (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`) if no checkpoint exists at `models/best_model.pth`
- Drops COCO reserved IDs (`N/A`) and background class from responses
- NMS threshold defaults to 0.2 (tighter than the typical 0.4 — filters redundant boxes aggressively)

## What Changed (Modernisation)

| Before | After |
|---|---|
| Pinned exact versions (`==`) | Minimum version pins (`>=`) |
| stdlib modules in requirements (`pathlib`, `logging`, etc.) | Removed — these are stdlib |
| `yolov8n.pt` binary tracked in git | Untracked; `.gitignore` covers `*.pt` |
| `yolo_detector.py` skeleton (print statements only) | Deleted |
| `vit_detector.py`, `dataset.py`, `visualize.py` (all `pass`) | Deleted |
| `src/api/New Text Document.txt` | Deleted |
| `black` + `flake8` | `ruff` |
