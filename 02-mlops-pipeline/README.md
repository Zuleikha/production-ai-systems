# MLOps Pipeline

End-to-end MLOps system: asset-based orchestration with Dagster, BERT fine-tuning on IMDB, experiment tracking with MLflow, and a FastAPI serving layer.

## Architecture

```
HuggingFace Datasets
      ↓
Dagster Assets (9-stage pipeline)
  pipeline_config → dataset_info → raw_dataset → validated_dataset
  → preprocessed_dataset → trained_model → evaluated_model
  → model_registry → deployment_config
      ↓
MLflow Experiment Tracking
      ↓
FastAPI Serving  (/predict, /train, /health)
```

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | Dagster `@asset` + `Definitions` |
| Training | HuggingFace Transformers `Trainer` |
| Experiment tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| Monitoring | Evidently, Prometheus |
| Config | pydantic-settings |

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
2. `dataset_info` — resolves dataset metadata
3. `raw_dataset` — downloads IMDB from HuggingFace Hub
4. `validated_dataset` — pandas-based schema checks
5. `preprocessed_dataset` — tokenisation, label encoding
6. `trained_model` — BERT fine-tuning with MLflow tracking
7. `evaluated_model` — accuracy, F1, confusion matrix
8. `model_registry` — saves artefacts to MLflow registry
9. `deployment_config` — writes serving config

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Sentiment prediction |
| POST | `/train` | Trigger full pipeline run |

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
