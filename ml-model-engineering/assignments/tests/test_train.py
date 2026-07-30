import os
import pandas as pd

from src.train import (
    build_baseline_pipeline,
    build_candidate_pipeline,
    decide_promotion,
    evaluate_model,
)
from src.features import to_model_matrix
from src.load_dataset import main as build_dataset


def _load_sample():
    if not os.path.exists("data/raw/churn_initial.csv"):
        build_dataset(argv=[])
    df = pd.read_csv("data/raw/churn_initial.csv").sample(n=300, random_state=1)
    X = to_model_matrix(df)
    y = df["churn"]
    return X, y


def test_baseline_and_candidate_train_and_evaluate():
    X, y = _load_sample()

    baseline = build_baseline_pipeline(seed=1)
    baseline.fit(X, y)
    metrics = evaluate_model(baseline, X, y)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["n_samples"] == len(y)

    candidate = build_candidate_pipeline(seed=1)
    candidate.fit(X, y)
    cmetrics = evaluate_model(candidate, X, y)
    assert 0.0 <= cmetrics["roc_auc"] <= 1.0


def test_decide_promotion_promotes_strong_candidate():
    cfg = {"promotion_guardrail": {"min_roc_auc": 0.8, "max_allowed_regression_vs_baseline": 0.01}}
    baseline_val = {"roc_auc": 0.75}
    candidate_val = {"roc_auc": 0.85}
    promote, reason = decide_promotion(baseline_val, candidate_val, cfg)
    assert promote is True


def test_decide_promotion_rejects_weak_candidate():
    cfg = {"promotion_guardrail": {"min_roc_auc": 0.8, "max_allowed_regression_vs_baseline": 0.01}}
    baseline_val = {"roc_auc": 0.75}
    candidate_val = {"roc_auc": 0.60}
    promote, reason = decide_promotion(baseline_val, candidate_val, cfg)
    assert promote is False
