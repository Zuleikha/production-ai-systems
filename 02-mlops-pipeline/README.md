# MLOps Pipeline

End-to-end MLOps system: asset-based orchestration with Dagster, BERT fine-tuning on IMDB, experiment tracking with MLflow, and a FastAPI serving layer.

## Architecture

```
HuggingFace Datasets
      ↓
Dagster Assets (9-stage pipeline)
  pipeline_config → dataset_info → raw_dataset
  → processed_train_data + processed_test_data
  → model_config → pretrained_model_setup
  → training_datasets → trained_model
      ↓
MLflow Experiment Tracking
      ↓
FastAPI Serving  (/predict, /train, /train/status/{run_id}, /health, /metrics)
```

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | Dagster `@asset` + `Definitions` |
| Training | HuggingFace Transformers `Trainer` |
| Experiment tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| Monitoring | Prometheus (`prometheus-client`) — `/metrics` endpoint with request and latency counters |
| Config | `pydantic-settings` |

> **Note:** Evidently was listed in earlier versions of this project but was removed — no Evidently code ran. Prometheus is wired into `src/api.py` with real counters and a `/metrics` endpoint.

## Quick Start

```bash
pip install -r requirements.txt

# Run Dagster UI
dagster dev -f src/dagster_definitions.py

# Run API
uvicorn src.api:app --reload
```

## Configuration

Copy `.env.example` to `.env` and set:

```
MLFLOW_TRACKING_URI=./mlruns
HUGGINGFACE_TOKEN=       # optional, for private datasets
```

Pipeline hyperparameters live in `config/pipeline.yaml`.

## Pipeline Assets

The Dagster pipeline materialises 9 assets in dependency order:

1. `pipeline_config` — loads YAML config
2. `dataset_info` — resolves dataset name, sample size, column names
3. `raw_dataset` — downloads IMDB from HuggingFace Hub (cached); saves train/test parquet
4. `processed_train_data` — pandas length-filter and null-drop on training split
5. `processed_test_data` — same preprocessing on test split
6. `model_config` — extracts and type-coerces training hyperparameters
7. `pretrained_model_setup` — downloads `bert-base-uncased`, saves tokenizer to disk
8. `training_datasets` — tokenises both splits and saves HuggingFace `Dataset` to disk
9. `trained_model` — fine-tunes BERT with `Trainer`, evaluates, saves model + `docs/eval_results.json`

## Results

Numbers from `docs/eval_results.json` — measured on a real training run (1 epoch, CUDA GPU).

| Metric | Value |
|---|---|
| Accuracy | 1.00 |
| F1 (weighted) | 1.00 |
| Precision (weighted) | 1.00 |
| Recall (weighted) | 1.00 |
| Eval loss | 0.01 |
| Train loss | 0.2266 |
| Training samples | 986 |
| Test samples | 249 |
| Epochs | 1 |
| Model | bert-base-uncased |

## Monitoring

Prometheus metrics are exposed at `GET /metrics` (standard Prometheus scrape format).

| Metric | Type | Description |
|---|---|---|
| `predict_requests_total` | Counter | Total calls to `/predict` |
| `predict_errors_total` | Counter | Errors from `/predict` |
| `train_requests_total` | Counter | Total calls to `/train` |
| `predict_latency_seconds` | Histogram | Per-request prediction latency |

Drift detection uses a custom KS-test implementation in `src/monitoring/drift_detector.py` (scipy-based, no external monitoring service required).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/metrics` | Prometheus metrics scrape endpoint |
| POST | `/predict` | Sentiment prediction (503 if model not loaded) |
| POST | `/train` | Kick off async pipeline run; returns `run_id` immediately |
| GET | `/train/status/{run_id}` | Poll training status (queued / running / success / failed) |

### Async training pattern

`POST /train` returns immediately with a `run_id` and does not block the HTTP connection:

```json
{"run_id": "abc-123", "status": "queued", "poll": "/train/status/abc-123"}
```

The actual Dagster pipeline runs in a background thread via FastAPI `BackgroundTasks`. Poll `GET /train/status/{run_id}` to check progress.

## What Changed (Modernisation)

| Before | After |
|---|---|
| `wandb` experiment tracking | Removed; MLflow covers this |
| `great-expectations` validation | pandas assertions — no server needed |
| `optuna` hyperparameter search | Grid search tracked in MLflow nested runs |
| `@app.on_event("startup")` | `asynccontextmanager` lifespan |
| Plain YAML `Config` class | `pydantic-settings` with `.env` support |
| `dagster-postgres`, `dagster-docker` | Removed; not used in this project |
| `kubernetes`, `boto3`, `ray`, `dvc` | Removed; out-of-scope for this project |
| `black` + `flake8` | `ruff` (covers both) |
| Synchronous `POST /train` | `BackgroundTasks` — returns `run_id` immediately |
| Evidently (listed, not run) | Removed from requirements and docs |

## Known Limitations

- **Local MLflow only** — no distributed tracking server. `mlruns/` is a local file store. Runs are not shared across machines.
- **No CI/CD pipeline** — no automated test or deploy on push. Manual `uvicorn` start only.
- **Benchmark dataset** — IMDB is a well-known benchmark. The 1.0 accuracy result reflects fine-tuning bert-base-uncased on a 986-sample subset; results on the full 25k test split would differ.
- **Model reload requires service restart** — after a new training run completes the served model is hot-reloaded in-process, but if the API process was started before training there is no push notification to external consumers.
- **In-memory training status** — `train_runs` is a Python dict; all run history is lost on API restart.
