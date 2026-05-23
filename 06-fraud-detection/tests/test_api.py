"""
Integration tests for the FastAPI app in api/main.py.

Uses the real XGBoost model loaded from outputs/models/best_xgb.pkl.
Tests verify response shape, types, and Pydantic validation — not exact probabilities.
Run: pytest tests/test_api.py
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day"]
SAMPLE_TRANSACTION = {name: 0.0 for name in FEATURE_NAMES}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ─── /health ──────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_body_is_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_content_type_is_json(self, client):
        assert "application/json" in client.get("/health").headers["content-type"]


# ─── /predict — happy path ────────────────────────────────────────────────────

class TestPredictHappyPath:
    def test_returns_200_for_valid_transaction(self, client):
        assert client.post("/predict", json=SAMPLE_TRANSACTION).status_code == 200

    def test_response_contains_fraud_probability(self, client):
        assert "fraud_probability" in client.post("/predict", json=SAMPLE_TRANSACTION).json()

    def test_response_contains_is_fraud(self, client):
        assert "is_fraud" in client.post("/predict", json=SAMPLE_TRANSACTION).json()

    def test_response_contains_threshold_used(self, client):
        assert "threshold_used" in client.post("/predict", json=SAMPLE_TRANSACTION).json()

    def test_fraud_probability_is_float(self, client):
        assert isinstance(
            client.post("/predict", json=SAMPLE_TRANSACTION).json()["fraud_probability"], float
        )

    def test_is_fraud_is_bool(self, client):
        assert isinstance(
            client.post("/predict", json=SAMPLE_TRANSACTION).json()["is_fraud"], bool
        )

    def test_threshold_is_028(self, client):
        data = client.post("/predict", json=SAMPLE_TRANSACTION).json()
        assert data["threshold_used"] == pytest.approx(0.28)

    def test_fraud_probability_in_unit_interval(self, client):
        p = client.post("/predict", json=SAMPLE_TRANSACTION).json()["fraud_probability"]
        assert 0.0 <= p <= 1.0

    def test_is_fraud_consistent_with_probability_and_threshold(self, client):
        data = client.post("/predict", json=SAMPLE_TRANSACTION).json()
        expected = data["fraud_probability"] >= data["threshold_used"]
        assert data["is_fraud"] == expected

    def test_integer_feature_values_accepted(self, client):
        int_txn = {name: 0 for name in FEATURE_NAMES}
        assert client.post("/predict", json=int_txn).status_code == 200

    def test_negative_feature_values_accepted(self, client):
        neg_txn = {name: -1.5 for name in FEATURE_NAMES}
        assert client.post("/predict", json=neg_txn).status_code == 200

    def test_extra_fields_are_ignored(self, client):
        extra = {**SAMPLE_TRANSACTION, "unexpected_field": 99.9}
        assert client.post("/predict", json=extra).status_code == 200


# ─── /predict — input validation (Pydantic) ───────────────────────────────────

class TestInputValidation:
    def test_empty_body_returns_422(self, client):
        assert client.post("/predict", json={}).status_code == 422

    def test_missing_v1_returns_422(self, client):
        bad = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "V1"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_missing_log_amount_returns_422(self, client):
        bad = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "log_amount"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_missing_hour_of_day_returns_422(self, client):
        bad = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "hour_of_day"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_non_numeric_v1_returns_422(self, client):
        bad = {**SAMPLE_TRANSACTION, "V1": "not_a_float"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_null_v1_returns_422(self, client):
        bad = {**SAMPLE_TRANSACTION, "V1": None}
        assert client.post("/predict", json=bad).status_code == 422

    @pytest.mark.parametrize("field", FEATURE_NAMES)
    def test_each_feature_field_is_required(self, client, field):
        bad = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != field}
        assert client.post("/predict", json=bad).status_code == 422
