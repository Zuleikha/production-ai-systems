# Model Bundle Specification — bundle_v1

Location: `outputs/models/bundle_v1/`

This document describes every file in the bundle, what it contains, why it exists, and what breaks if it is missing.

---

## Files Present

### model.pkl

**Contains:** Serialised `XGBClassifier` object (joblib format). Direct copy of `outputs/models/best_xgb.pkl`.

**Why it exists:** The model is the core inference artefact. All predictions flow through `model.predict_proba()`. Without it no scoring is possible.

**What breaks if missing:** The API cannot start. Both `app/main.py` and `api/main.py` call `joblib.load()` at startup and raise `RuntimeError` immediately if the file is absent.

---

### feature_list.json

**Contains:** Ordered list of the 30 feature names the model was trained on, plus a count field.

```json
{
  "features": ["V1", "V2", ..., "V28", "log_amount", "hour_of_day"],
  "count": 30
}
```

**Why it exists:** XGBoost enforces that inference inputs match training feature names and order exactly. If a caller sends features in the wrong order or uses wrong names the model raises a feature mismatch error. This file is the authoritative reference for building valid request payloads.

**What breaks if missing:** Integration tests, client SDKs, and any tooling that dynamically constructs payloads lose their schema reference. The model itself will still run if the correct features are passed in the correct order, but there is no machine-readable source of truth for what "correct" means.

---

### threshold.json

**Contains:** The classification threshold value, its source, justification, and evaluated performance at both 0.28 and 0.03.

```json
{
  "threshold": 0.28,
  "source": "...",
  "justification": "...",
  "performance_at_0_28": { ... },
  "performance_at_0_03": { ... }
}
```

**Why it exists:** The threshold is a business decision separate from the model. Storing it here decouples it from the model artefact and from the API source code. Both `app/main.py` and `api/main.py` load `threshold` from this file at startup. `src/predict.py` reads it at inference time.

**What breaks if missing:** Both API files fail to load `THRESHOLD` at startup. `src/predict.py` raises a `FileNotFoundError` on the first call to `predict()`. The API returns 500 on every request.

---

### metadata.json

**Contains:** Full provenance record — model type, Optuna hyperparameters, training dataset details, train/test split, PR-AUC, ROC-AUC, performance at threshold 0.28, and artifact inventory.

**Why it exists:** Provides a single machine-readable record of how this bundle was produced. Required for audit trails, model registries, and any downstream tooling that needs to know model version, training conditions, or expected performance without running the model.

**What breaks if missing:** No inference breaks. The API still runs. What is lost: reproducibility documentation, version tracking, and any tool that reads metadata to validate bundle integrity before deployment.

---

## Full Feature List (in training order)

These 30 features must be passed to the model in this exact order:

| Position | Feature | Description |
|---|---|---|
| 1–28 | V1, V2, V3 … V28 | PCA components from the Kaggle Credit Card Fraud dataset. Original features anonymised. |
| 29 | log_amount | `log1p(Amount)` — log-scaled transaction amount. Reduces right skew. |
| 30 | hour_of_day | `(Time // 3600) % 24` — hour of day derived from the raw Time field (seconds from first transaction). |

The raw `Time` and `Amount` columns are dropped before inference. The model was never trained on them directly.

---

## Intentionally Absent Files

### scaler.pkl — NOT PRESENT

**Reason:** The production model (`best_xgb.pkl`) is an `XGBClassifier`. XGBoost uses decision tree splits which are based on rank order, not absolute feature magnitude. It is mathematically scale-invariant. No `StandardScaler` was fitted or applied during XGBoost training.

`baseline_scaler.joblib` does exist in `outputs/models/` but it belongs exclusively to the Logistic Regression baseline built in `notebooks/02_baseline_model.ipynb`. Logistic Regression is a distance-based linear model and requires scaling. That scaler must never be applied to XGBoost inputs — the features it was fitted on are the same columns, but applying it would change values without improving accuracy and would create a training/serving skew.

**What would happen if someone added a scaler incorrectly:** Predictions would change subtly, creating a discrepancy between notebook evaluation results and API results. Model performance metrics would no longer match what was benchmarked.

---

### label_encoders.pkl — NOT PRESENT

**Reason:** There are no categorical features in this model. All 30 input features are continuous numeric values (V1–V28 are PCA-derived floats, `log_amount` and `hour_of_day` are numeric transforms). No encoding step was needed or applied.

---

### imputer.pkl — NOT PRESENT

**Reason:** XGBoost handles missing values natively via the `missing=nan` parameter (default behaviour). During training, XGBoost learns the optimal direction to route `NaN` values at each split. No explicit imputation was applied to the training data and none should be applied at inference.

Applying an imputer that was not used during training would alter the feature values and break the training/serving contract.
