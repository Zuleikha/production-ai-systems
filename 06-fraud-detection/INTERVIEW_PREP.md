# Fraud Detection — Interview Prep

---

## 30-Second Pitch

> "I built an end-to-end credit card fraud detection system — from raw data through to a live REST API.
> The dataset has 284,000 transactions with extreme class imbalance (0.17% fraud).
> I compared four imbalance strategies, tuned XGBoost with Optuna across 150 trials logged in MLflow,
> explained predictions with SHAP, and optimised the decision threshold using a business cost matrix.
> The model is deployed on Render and serves real-time predictions via FastAPI."

---

## What the Project Does — Step by Step

| Stage | Notebook | What happens | Output |
|-------|----------|-------------|--------|
| **EDA** | `01` | Understand data — class imbalance, feature distributions, correlations | Key features identified |
| **Baseline** | `02` | Logistic Regression + StandardScaler — sets the floor to beat | PR-AUC 0.73 |
| **Imbalance** | `03` | Compare 4 resampling strategies empirically | PR-AUC 0.76 |
| **Tuning** | `04` | XGBoost + Optuna (150 trials) logged to MLflow | PR-AUC 0.88 |
| **Explainability** | `05` | SHAP waterfall + summary plots | Feature importance rankings |
| **Threshold** | `06` | Cost-matrix sweep — minimise € lost, not just error rate | Threshold = 0.28 |
| **Deployment** | `07` | FastAPI + Render — live REST API with health check | Live API |

---

## Tech Stack

| Layer | Tool | Why this tool |
|-------|------|--------------|
| **Core ML** | XGBoost | Handles tabular data well, fast, built-in `scale_pos_weight` for imbalance |
| **Imbalance** | imbalanced-learn (SMOTE) | Synthesises minority samples without distorting the majority distribution |
| **Hyperparameter tuning** | Optuna | Bayesian optimisation — smarter than grid search, scales to many params |
| **Experiment tracking** | MLflow + DagsHub | Every trial logged — reproducible, comparable, sharable |
| **Explainability** | SHAP | Model-agnostic, works on individual predictions, regulatorily useful |
| **Feature engineering** | pandas, NumPy | Rolling windows, log transforms, time extraction |
| **API** | FastAPI + Pydantic | Auto-validates request schema, auto-generates `/docs`, async-ready |
| **Deployment** | Render | Zero-config deployment from `render.yaml`, free tier |
| **Testing** | pytest + httpx | Unit tests (mocked model), feature tests, API integration tests |

---

## Results

| Model | PR-AUC | Notes |
|-------|--------|-------|
| Logistic Regression (baseline) | 0.73 | `class_weight='balanced'` only |
| LR + SMOTE | 0.76 | Best imbalance strategy |
| XGBoost (default params) | 0.82 | Immediate jump from tree model |
| **XGBoost + Optuna (150 trials)** | **0.88** | Production model |

**Threshold:** 0.28 (not the default 0.5) — tuned using business cost matrix.

---

## Key Decisions — and Why

### 1. Why PR-AUC, not Accuracy?

- Dataset is 99.83% legitimate — a model that predicts "all legit" gets 99.83% accuracy but catches **zero fraud**
- PR-AUC measures performance on the minority class specifically
- Accuracy is meaningless when classes are this imbalanced

> **Say in interview:** "Accuracy is a vanity metric here. PR-AUC tells me how well the model ranks fraudulent transactions above legitimate ones — which is the actual task."

---

### 2. Why SMOTE inside a Pipeline?

- SMOTE generates **synthetic** minority samples by interpolating between real examples
- If applied before train/test split, synthetic samples from test data leak into training → **data leakage**
- Wrapping it in `imblearn.Pipeline` ensures SMOTE only runs on training folds during cross-validation

```
imblearn.Pipeline([
    ("preprocessor", ColumnTransformer(...)),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   XGBClassifier(...)),
])
```

> **Say in interview:** "SMOTE must be inside the pipeline or you contaminate your validation set with synthetic data — your CV scores will be optimistic and your model will underperform in production."

---

### 3. Why XGBoost over Random Forest?

| | XGBoost | Random Forest |
|--|---------|--------------|
| Learning | Boosting — each tree corrects the previous | Bagging — trees vote independently |
| Imbalance | `scale_pos_weight` handles it natively | `class_weight='balanced'` only |
| Speed | Faster on large datasets | Slower (more trees needed) |
| Tuning | More hyperparameters but bigger performance ceiling | Fewer hyperparameters, easier to tune |

> **Say in interview:** "XGBoost's `scale_pos_weight` parameter directly addresses class imbalance in the loss function. Combined with boosting's sequential correction, it's a strong choice for fraud data."

---

### 4. Why Optuna over GridSearchCV?

- Grid search tests every combination — exponential cost as params grow
- Optuna uses **Bayesian optimisation** — it learns which parameter regions are promising and focuses there
- 150 trials with Optuna explores the space more effectively than a 150-point grid

> **Say in interview:** "With 9 hyperparameters, a modest grid of 3 values each is 3⁹ = 19,683 evaluations. Optuna finds a near-optimal solution in 150 trials because it's not random — it builds a probabilistic model of the search space."

---

### 5. Why threshold = 0.28, not 0.5?

**Cost matrix:**
- Missing fraud (False Negative) = **€200** average loss
- Blocking a legitimate transaction (False Positive) = **€5** friction cost

```
Total cost = FN × 200 + FP × 5
```

- Sweep thresholds 0.01 → 0.99
- At each threshold: compute cost on held-out test set
- Pick the threshold where total cost is lowest → **0.28**

| Threshold | Fraud caught | False alarms | Business cost |
|-----------|:------------:|:------------:|:-------------:|
| 0.50 | 80% | Low | High (missed fraud expensive) |
| **0.28** | **93%** | **Medium** | **Lowest overall** |
| 0.10 | 97% | High | High (too many false alarms) |

> **Say in interview:** "The default 0.5 threshold is a technical convenience, not a business decision. I frame it as: what costs more — a missed fraud or an annoyed customer? That ratio sets the threshold."

---

### 6. Why SHAP for explainability?

- XGBoost is a black box — stakeholders and regulators need to know **why** a transaction was flagged
- SHAP assigns each feature a contribution value for each individual prediction
- Waterfall plot shows: "this transaction was flagged because V14 pushed the score up by +0.3 and the amount was unusually large"

> **Say in interview:** "GDPR and financial regulations often require explainability. SHAP gives me a per-transaction explanation I can show to a compliance team — not just global feature importances."

---

### 7. Why FastAPI over Flask?

- FastAPI uses Pydantic models — **request schema is validated automatically**, wrong types get a 422 before the model even runs
- Auto-generated interactive docs at `/docs` — great for demos
- Async-ready (handles concurrent requests without blocking)
- The `Transaction` Pydantic model documents all 30 required fields as part of the code

---

## Architecture — How a Prediction is Made

```
HTTP POST /predict
    │
    ▼
Pydantic validates 30 fields (V1–V28, log_amount, hour_of_day)
    │  wrong type or missing → 422 immediately
    ▼
pandas DataFrame (1 row, 30 columns)
    │
    ▼
model.predict_proba(df)   ← XGBoost loaded once at startup
    │
    ▼
prob = output[:, 1][0]    ← fraud probability
    │
    ▼
is_fraud = prob >= 0.28   ← business threshold applied
    │
    ▼
JSON response: { fraud_probability, is_fraud, threshold_used }
```

**Model loads once at startup** — not on every request. Saves ~200ms per call.

---

## Common Interview Questions — Short Answers

### "Walk me through the project."
> Start with the business problem → data challenge → pipeline stages → result → deployment. Use the 30-second pitch above, then offer to go deep on any stage.

### "What is class imbalance and how did you handle it?"
> 0.17% of transactions are fraud. Three approaches: (1) penalise misclassifying the minority class more (`class_weight`), (2) oversample the minority class (SMOTE), (3) undersample the majority class. I compared all empirically — SMOTE + XGBoost won on PR-AUC.

### "What is PR-AUC and why use it?"
> Precision-Recall AUC measures the area under the precision-recall curve. It focuses entirely on the minority class — precision = of what we flagged, how many were real fraud; recall = of all real fraud, how many did we catch. ROC-AUC can look good even on imbalanced data because it includes true negatives. PR-AUC doesn't — it forces you to care about the rare class.

### "What is SMOTE?"
> Synthetic Minority Over-sampling Technique. For each minority sample, it finds its k nearest neighbours and creates new synthetic samples by interpolating between them. It diversifies the minority class rather than just duplicating rows. The risk is creating unrealistic samples near the decision boundary.

### "What is data leakage and how did you prevent it?"
> Data leakage is when information from the test set influences model training — giving falsely optimistic metrics. I prevented it by: (1) fitting StandardScaler and SMOTE only on training folds inside a pipeline, (2) using `stratify=y` on the same `random_state=42` split throughout.

### "Why XGBoost?"
> It's a gradient boosting algorithm that builds trees sequentially, each correcting the errors of the last. It has native support for class imbalance (`scale_pos_weight`), handles missing values, and is regularised (L1/L2) to avoid overfitting. It consistently outperforms random forests on structured tabular data.

### "How does Optuna work?"
> Optuna uses a Tree-structured Parzen Estimator (TPE) — a form of Bayesian optimisation. It builds a probabilistic model of the objective function and uses it to choose the next hyperparameter set to try, focusing on regions that have performed well so far. Much more efficient than grid or random search.

### "How would you monitor this model in production?"
> Watch for data drift (distribution of V1–V28 shifting over time), label drift (fraud rate changing), and model performance degradation (PR-AUC dropping on a rolling window). I'd also track threshold effectiveness — if business costs are rising, the threshold may need re-tuning.

### "What would you do differently at scale?"
> Velocity features (transactions in last 1h/24h) currently need a full table scan — at scale these would come from a feature store like Redis or Feast, pre-computed in real time. The model would run in a separate inference service with a load balancer, and the threshold would be managed via a config service rather than a YAML file.

### "How did you handle the threshold in deployment?"
> The threshold lives in `config.yaml` and is read at startup by `src/predict.py`. Changing it requires only a config update and restart — no retraining, no code change. This is intentional: the model captures learned patterns; the threshold captures a business decision. They should be independently adjustable.

---

## Numbers to Remember

| Metric | Value |
|--------|-------|
| Dataset size | 284,807 transactions |
| Fraud rate | 0.172% (575:1 imbalance) |
| Best PR-AUC | **0.88** |
| Baseline PR-AUC | 0.73 |
| Production threshold | **0.28** |
| Optuna trials | 150 |
| Features | 30 (V1–V28 PCA + log_amount + hour_of_day) |
| Live API | https://fraud-detection-api-5gno.onrender.com |
| MLflow UI | https://dagshub.com/Zuleikha/fraud-detection-ML-project.mlflow |

---

## Things to Proactively Mention

- **You evaluated four imbalance strategies** — not just picked one blindly
- **Threshold tuning with a cost matrix** — shows you think in business terms, not just metrics
- **SMOTE inside a pipeline** — shows you understand data leakage
- **MLflow for experiment tracking** — 150 trials logged, reproducible
- **SHAP for explainability** — shows awareness of regulatory requirements
- **Pydantic validation** — shows production engineering mindset (fail fast at the boundary)
- **Model loaded once at startup** — shows you think about latency
- **Test suite** — unit tests for features, prediction logic, and all API endpoints
