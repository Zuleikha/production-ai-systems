# CLAUDE.md — Fraud Detection Project

This file is read automatically by Claude Code at session start. It defines the project context, conventions, and constraints that apply to all work in this directory.

---

## Project Purpose

End-to-end machine learning system for real-time credit card fraud detection. Trained on the Kaggle Credit Card Fraud dataset (284,807 transactions, 575:1 class imbalance). Deployed as a FastAPI REST API on Render.

The primary model is an XGBoost classifier tuned with Optuna. The production threshold is 0.28, loaded from the model bundle at startup.

---

## ML Pipeline Stages (in order)

| Stage | Notebook | Output |
|---|---|---|
| 1. EDA | `01_eda.ipynb` | Figures: class balance, feature distributions, fraud by hour |
| 2. Baseline | `02_baseline_model.ipynb` | `baseline_lr.joblib`, `baseline_scaler.joblib` — LR baseline only |
| 3. Imbalance handling | `03_imbalance_handling.ipynb` | SMOTE / class weight comparison |
| 4. XGBoost + Optuna | `04_xgboost_tuning.ipynb` | `best_xgb.pkl` — production model |
| 5. SHAP explainability | `05_shap_explainability.ipynb` | SHAP figures |
| 6. Threshold tuning | `06_threshold_tuning.ipynb` | Threshold analysis (0.03 vs 0.28) |
| 7. API deployment | `07_api_and_deployment.ipynb` | Deployment walkthrough |
| 8. Monitoring | `08_monitoring_and_drift.ipynb` | Drift figures |

---

## Folder Structure

```
06-fraud-detection/
├── app/main.py              # PRODUCTION API — FastAPI app served on Render
├── api/main.py              # Alternative API endpoint
├── src/
│   ├── features.py          # Feature engineering (time features, velocity, preprocessor)
│   ├── train.py             # Training pipeline (reads config/config.yaml)
│   └── predict.py           # Inference helpers (loads model + threshold from bundle)
├── notebooks/               # Jupyter notebooks (01–08, run in order)
├── outputs/
│   ├── models/
│   │   ├── best_xgb.pkl     # Production XGBoost model
│   │   ├── baseline_lr.joblib      # LR baseline only — do not use for inference
│   │   ├── baseline_scaler.joblib  # LR scaler only — do not apply to XGBoost
│   │   └── bundle_v1/       # Versioned model bundle (see BUNDLE_SPEC.md)
│   │       ├── model.pkl
│   │       ├── feature_list.json
│   │       ├── threshold.json
│   │       └── metadata.json
│   └── figures/             # All evaluation and SHAP plots
├── config/config.yaml       # All configurable values — threshold, paths, split params
├── tests/                   # pytest suite (test_api.py, test_predict.py, test_features.py)
├── docs/                    # Markdown study notes matching each notebook
├── BUNDLE_SPEC.md           # Bundle file reference — what each file contains and why
├── DECISIONS.md             # Technical decisions with full reasoning
└── README.md                # Public-facing project overview
```

---

## Bundle Location and Loading Pattern

The model bundle lives at `outputs/models/bundle_v1/`.

**Threshold loading (app/main.py, api/main.py, src/predict.py):**
```python
import json
from pathlib import Path
THRESHOLD = json.loads(Path("outputs/models/bundle_v1/threshold.json").read_text())["threshold"]
```

**Model loading:**
```python
import joblib
MODEL = joblib.load("outputs/models/best_xgb.pkl")
# or from bundle:
MODEL = joblib.load("outputs/models/bundle_v1/model.pkl")
```

Never hardcode the threshold value `0.28` directly in source code. Always load from the bundle or config.

---

## Coding Standards

All code in this project must follow these standards:

- **Type hints** on every function signature — parameters and return types
- **Docstrings** on every function — one-line minimum, explain what it returns
- **No hardcoded values** — threshold, model paths, feature lists, and split parameters must come from `config/config.yaml` or `outputs/models/bundle_v1/threshold.json`
- **Config source of truth:** `config/config.yaml` for training pipeline values; `bundle_v1/threshold.json` for inference threshold
- **No inline magic numbers** — if a numeric constant is not self-evident (e.g. `3600` for seconds-per-hour is acceptable with a comment), it belongs in config
- **Tests required** for any new function in `src/` — place in `tests/` matching the module name

---

## Production File Constraints

`app/main.py` and `api/main.py` are **live production files** served on Render.

- Do **not** refactor, restructure, or rename anything in these files without explicit instruction
- Do **not** change the `Transaction` Pydantic model field names or types — these are the public API contract
- Do **not** change the response schema keys (`fraud_probability`, `is_fraud`, `threshold_used`) — downstream callers depend on them
- Do **not** add middleware, authentication, or new endpoints without explicit instruction
- Threshold loading and model loading may be updated if the bundle structure changes, but the `THRESHOLD` variable name must remain the same — it is referenced at line 35 (`app/main.py`) and equivalent line in `api/main.py`

---

## Key Facts

| Item | Value |
|---|---|
| Production model | XGBClassifier, Optuna-tuned, 150 trials |
| Training data | Kaggle Credit Card Fraud (creditcard.csv) |
| Features | 30: V1–V28, log_amount, hour_of_day |
| No scaler | XGBoost is scale-invariant — baseline_scaler.joblib is for LR only |
| No imputer | XGBoost handles NaN natively |
| Threshold | 0.28 — loaded from bundle_v1/threshold.json |
| PR-AUC | 0.8828 | ROC-AUC | 0.9807 |
| Live API | https://fraud-detection-api-5gno.onrender.com |
