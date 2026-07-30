import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE_PAYLOAD = {
    "customer_id": "9237-HQITU",
    "tenure_months": 2,
    "monthly_charges": 70.7,
    "total_charges": 151.65,
    "contract_type": "month_to_month",
    "payment_method": "electronic_check",
    "internet_service": "fiber_optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "senior_citizen": 0,
    "partner": 0,
    "dependents": 0,
    "phone_service": 1,
    "multiple_lines": 0,
    "paperless_billing": 1,
}


@pytest.fixture()
def client():
    # Using the context-manager form triggers FastAPI's lifespan startup
    # handler, which loads the trained model artifact.
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_predict_endpoint_returns_valid_response(client):
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == "9237-HQITU"
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert "model_version" in body


def test_predict_endpoint_rejects_invalid_contract_type(client):
    bad_payload = dict(SAMPLE_PAYLOAD)
    bad_payload["contract_type"] = "not_a_real_contract"
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
