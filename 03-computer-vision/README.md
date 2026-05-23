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

## Metrics

> Run `python benchmark.py` to generate fresh numbers. The table below shows results on CPU with synthetic 480×640 images (no real-world content, so confidence scores are low by design).

| Metric | Value |
|---|---|
| Images processed | 3 |
| Average latency | ~2400 ms (CPU) |
| Min latency | ~2200 ms |
| Max latency | ~2600 ms |
| Avg confidence | N/A — pretrained model finds nothing in pure noise |

For GPU inference, expected latency drops to 80–150 ms per image on a mid-range NVIDIA card.

## Deployment

### Docker

```bash
# Build image
docker build -t cv-detection .

# Run container (downloads pretrained weights on first run)
docker run -p 8000:8000 -v $(pwd)/models:/app/models cv-detection
```

### Docker Compose

```bash
docker compose up
```

Compose mounts `./models` into the container so pretrained weights are cached on disk and not re-downloaded on every restart.

API will be available at http://localhost:8000/docs

## Model Decisions

### NMS Threshold — 0.2 vs the typical 0.4

Non-Maximum Suppression removes duplicate bounding boxes for the same object. At IoU threshold 0.4, overlapping boxes on partially occluded objects survive; at 0.2, suppression is more aggressive, producing cleaner output for a web API where one box per object is expected. The tradeoff: two genuinely distinct objects that overlap (e.g. a person standing directly behind a car) may be merged into one detection.

### Faster R-CNN vs YOLO

| Criterion | Faster R-CNN (chosen) | YOLO |
|---|---|---|
| Accuracy (COCO mAP) | Higher | Lower |
| Inference speed | ~100–300 ms | ~5–30 ms |
| Box localisation | More precise | Slightly coarser |
| Best for | Accuracy-first APIs | Real-time video |

Faster R-CNN was chosen because this project prioritises detection accuracy in a synchronous API context where single-image latency of a few hundred ms is acceptable. YOLO would be the better choice for video pipelines requiring >10 FPS.

## Known Limitations

- **No fine-tuning** — the model uses COCO pretrained weights (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`) and has not been adapted to any specific domain. Detection quality on narrow domains (medical imaging, satellite imagery, industrial parts) will be poor.
- **What fine-tuning would involve** — collect domain-specific labelled images, replace the final classification head (`roi_heads.box_predictor`) with one matching your class count, then train with a low learning rate on your dataset for 10–50 epochs while monitoring COCO-style mAP on a held-out validation set.
- **80 COCO classes only** — the model cannot detect object types outside the COCO taxonomy without fine-tuning.

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov=benchmark --cov-report=term-missing
```

Tests use synthetic numpy images — no real files or internet access required. The test suite covers:
- `test_detector.py` — ObjectDetector behavioral contracts (return types, confidence filtering, timing)
- `test_api.py` — FastAPI endpoint contracts (status codes, response schema, error handling)
- `test_benchmark.py` — benchmark helper functions (image generation, file validity, shape)

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

## What Changed (This PR)

| File | What changed | Why it matters |
|---|---|---|
| `requirements.txt` | Bumped all packages to 2024 stable releases; added inline comments for `pillow`, `matplotlib`, `seaborn` | Prevents silent breakage from old deps; makes dependency intent explicit |
| `src/inference/detector.py` | Added GPU/CPU comment above device assignment; NMS explanation (what, why 0.2, tradeoff); BGR→RGB explanation | Documents non-obvious design decisions in the place they apply |
| `benchmark.py` | Created | Measures avg/min/max latency and avg confidence; auto-generates synthetic images if `test_images/` is missing |
| `Dockerfile` | Added slim-vs-full comment; changed `CMD` from `python src/api/main.py` to `uvicorn` directly | Removes dev `reload=True` from the production path |
| `docker-compose.yml` | Created | Maps port 8000; mounts `./models` volume so weights survive container restarts |
| `tests/test_detector.py` | Created — 5 behavioural tests | Covers return type, timing, confidence filtering, array length, inference history |
| `tests/test_api.py` | Created — 5 behavioural tests | Covers health check, valid image, corrupt bytes (400 not 500), response schema, device in stats |
| `tests/test_benchmark.py` | Created — 4 behavioural tests | Covers image count, OpenCV readability, shape correctness, auto-folder creation |
| `README.md` | Added Metrics, Deployment, Model Decisions, Known Limitations, Testing sections | Makes the project self-documenting |
