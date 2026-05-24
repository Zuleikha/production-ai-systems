# Fraud Detection ML System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189EBF)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)](https://fraud-detection-api-5gno.onrender.com)

End-to-end machine learning system for real-time credit card fraud detection. Trained on 284,807 transactions with a 575:1 class imbalance. Deployed as a REST API.

---

## What It Does

Scores a payment transaction for fraud risk using an XGBoost classifier. Returns a fraud probability, a binary prediction, and the threshold used. The full pipeline covers exploratory analysis, imbalance handling, Optuna hyperparameter tuning, SHAP explainability, threshold analysis, and production deployment.

**Pipeline:** `EDA → Baseline → Imbalance Handling → XGBoost + Optuna → SHAP → Threshold Analysis → API Deployment`

---

## Model Performance

Model: XGBoost, tuned with Optuna (150 trials), evaluated at threshold **0.28** on a held-out test set of 56,962 transactions.

| Metric | Value |
|---|---|
| PR-AUC | 0.8828 |
| ROC-AUC | 0.9807 |
| Precision | 81.6% |
| Recall | 85.7% |
| F1 Score | 83.6% |
| Fraud caught | 84 / 98 (85.7%) |
| False alarms | 19 |
| Transactions flagged | 0.18% |

For threshold selection rationale and full model decisions, see [DECISIONS.md](DECISIONS.md).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & features | pandas, NumPy, scikit-learn |
| Modelling | XGBoost, Optuna, imbalanced-learn |
| Explainability | SHAP |
| Experiment tracking | MLflow + DagsHub |
| API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest, httpx |
| Deployment | Render |

---

## Live API

**Interactive docs:** [https://fraud-detection-api-5gno.onrender.com/docs](https://fraud-detection-api-5gno.onrender.com/docs)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Returns fraud probability and binary prediction |

**Input data format**

> **Important:** This API expects data in the same format as the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The 28 features V1–V28 are PCA components — they are not raw transaction fields such as merchant, card number, or location. Raw bank transaction data cannot be sent directly. It must first be transformed using the same PCA fitted on the original dataset to produce the V1–V28 components. The two additional fields are derived as follows:
> - `log_amount` — `log1p(transaction_amount)`
> - `hour_of_day` — hour extracted from the transaction timestamp (0–23)

**Example request:**

```bash
curl -s -X POST https://fraud-detection-api-5gno.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782, "V5": -0.3383,
    "V6": 0.4624, "V7": 0.2396, "V8": 0.0987, "V9": 0.3638, "V10": 0.0908,
    "V11": -0.5516, "V12": -0.6178, "V13": -0.9914, "V14": -0.3112, "V15": 1.4682,
    "V16": -0.4704, "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669, "V25": 0.1285,
    "V26": -0.1892, "V27": 0.1336, "V28": -0.0211,
    "log_amount": 3.64,
    "hour_of_day": 14.0
  }'
```

**Response:**

```json
{
  "fraud_probability": 0.0312,
  "is_fraud": false,
  "threshold_used": 0.28
}
```

> Render free-tier instances spin down after inactivity. The first request may take up to 30 seconds.

---

## Run Locally

**1. Clone and install**

```bash
git clone https://github.com/Zuleikha/fraud-detection-ML-project.git
cd fraud-detection-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Download the dataset**

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

**3. Run the notebooks in order** (01 → 07) to reproduce training and generate model artefacts.

```bash
jupyter notebook notebooks/
```

**4. Start the API**

```bash
uvicorn app.main:app --reload
```

Docs available at `http://localhost:8000/docs`.

---

## Run Tests

```bash
pytest tests/ -v
```

Covers API endpoints, threshold boundary logic, feature engineering, and input validation.

---

## Experiment Tracking

All Optuna trials logged to MLflow via DagsHub:
[https://dagshub.com/Zuleikha/fraud-detection-ML-project.mlflow](https://dagshub.com/Zuleikha/fraud-detection-ML-project.mlflow)
