# MLOps Pipeline — Usage Guide

## Quick Start

### 1. Environment Setup

```bash
git clone <repository-url>
cd 02-mlops-pipeline

python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

```
MLFLOW_TRACKING_URI=./mlruns
HUGGINGFACE_TOKEN=            # optional — only needed for private datasets
```

Pipeline hyperparameters (epochs, batch size, learning rate) live in `config/pipeline.yaml` — these are not secrets and belong in version control.

---

### 2. Run the Dagster Pipeline

Start the Dagster UI:

```bash
dagster dev -f src/dagster_definitions.py
```

Open `http://localhost:3000` in your browser. You will see the 9-asset pipeline graph.

**To run the full pipeline:**
1. Click **Materialize All** in the Dagster UI
2. Watch each asset materialise in order: `pipeline_config` → `dataset_info` → `raw_dataset` → `validated_dataset` → `preprocessed_dataset` → `trained_model` → `evaluated_model` → `model_registry` → `deployment_config`

**To re-run a single stage** (e.g. retrain without re-downloading data):
1. Click on the `trained_model` asset
2. Click **Materialize Selected**
3. Dagster will only re-run `trained_model` and its dependents

---

### 3. View Experiment Tracking in MLflow

```bash
mlflow ui
```

Open `http://localhost:5000`. You will see:
- All training runs with parameters logged (model name, batch size, learning rate, epochs)
- Metric curves (train/val loss per epoch)
- Saved model artefacts
- Model registry with versioned models

---

### 4. Start the Serving API

```bash
uvicorn src.api:app --reload
```

API available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{ "status": "healthy", "model_loaded": true }
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This film was absolutely brilliant."}'
```

Expected response:
```json
{ "label": "positive", "score": 0.98, "model_version": "1" }
```

### Trigger Full Training Run

```bash
curl -X POST http://localhost:8000/train
```

This triggers the full Dagster pipeline synchronously. **Note:** this is a long-running call (~10 minutes for BERT fine-tuning on IMDB). Not suitable for production — use a job queue in a real deployment.

---

## Pipeline Assets — What Each One Does

| Asset | What it produces |
|---|---|
| `pipeline_config` | Loads and validates `config/pipeline.yaml` |
| `dataset_info` | Resolves dataset name, split sizes, label names |
| `raw_dataset` | Downloads IMDB from HuggingFace Hub (25k train, 25k test) |
| `validated_dataset` | Runs pandas assertions: no nulls, correct columns, text length bounds |
| `preprocessed_dataset` | Tokenises with `bert-base-uncased` tokenizer, encodes labels |
| `trained_model` | Fine-tunes BERT with HuggingFace `Trainer`, logs to MLflow |
| `evaluated_model` | Accuracy, F1, confusion matrix on the test set |
| `model_registry` | Saves model artefacts and metadata to MLflow registry |
| `deployment_config` | Writes serving config for the FastAPI layer |

---

## Configuration

Edit `config/pipeline.yaml` to change training hyperparameters:

```yaml
model:
  name: bert-base-uncased
  num_labels: 2

training:
  epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  evaluation_strategy: epoch
  load_best_model_at_end: true

dataset:
  name: imdb
  max_length: 512
```

---

## Common Issues

**Issue: HuggingFace download is slow or times out**
```bash
# Set a HuggingFace token to get higher rate limits
export HUGGINGFACE_TOKEN=hf_yourtoken
```

**Issue: Out of memory during training**
- Reduce `batch_size` in `pipeline.yaml` to 8
- Reduce `max_length` to 256

**Issue: MLflow UI shows no runs**
- Ensure `MLFLOW_TRACKING_URI` in `.env` points to `./mlruns`
- Check that `mlruns/` directory exists — it's created on first training run

**Issue: `dagster dev` fails to start**
```bash
# Check Python version (requires 3.9+)
python --version

# Reinstall in a clean venv
pip install --upgrade dagster dagster-webserver
```

**Issue: Model checkpoint not found when starting API**
- Run the full pipeline at least once through Dagster before starting the API
- The API falls back gracefully with an error message if no checkpoint exists
