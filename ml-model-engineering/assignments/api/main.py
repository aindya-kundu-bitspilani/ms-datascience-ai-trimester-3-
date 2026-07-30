"""
api/main.py
-----------
Minimal online inference service (request-response pattern).

Why online request-response and not batch (documented further in the
design doc):
  - A human (retention/support agent, or a real-time app flow) is
    typically waiting on this score right when a customer interacts
    with support or billing -- e.g. to decide whether to offer a
    retention incentive during that same session.
  - Input is a single customer's current attributes, not a large
    periodic sweep, so per-request scoring is a natural fit and keeps
    latency low (see scripts/load_test.py for measured latency).
  - A nightly batch score is also mentioned as a valid complementary
    pattern in the design doc for populating dashboards, but the
    interactive retention-offer use case needs an online endpoint.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.features import to_model_matrix

MODEL_PATH = Path("models/production_model.pkl")
VERSION_PATH = Path("models/model_version.json")

_model = None
_model_version = "unknown"
_model_type = "unknown"


def _load_model():
    global _model, _model_version, _model_type
    if not MODEL_PATH.exists():
        # The app can still start (useful for tests / CI), but /predict
        # will fail loudly (503) until a model has been trained.
        return
    _model = joblib.load(MODEL_PATH)
    if VERSION_PATH.exists():
        meta = json.loads(VERSION_PATH.read_text())
        _model_version = meta.get("model_version", "unknown")
        _model_type = meta.get("promoted_model_type", "unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Churn Risk Scoring API",
    description="Mini production ML system - online churn risk scoring endpoint.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    start = time.perf_counter()

    raw_df = pd.DataFrame([request.model_dump()])
    # This is the exact same feature engineering function used at
    # training time -- the single-source-of-truth that prevents
    # training/serving skew.
    X = to_model_matrix(raw_df)

    proba = float(_model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    elapsed_ms = (time.perf_counter() - start) * 1000
    app.state.last_latency_ms = elapsed_ms  # exposed for ad-hoc inspection only

    return PredictResponse(
        customer_id=request.customer_id,
        churn_prediction=pred,
        churn_probability=round(proba, 4),
        model_version=_model_version,
        model_type=_model_type,
    )
