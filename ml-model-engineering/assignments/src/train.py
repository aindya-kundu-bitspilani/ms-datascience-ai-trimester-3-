"""
train.py
--------
Repeatable training pipeline:

    load data -> split train/val/test -> engineer features ->
    train baseline -> train candidate -> evaluate both ->
    apply promotion guardrail -> save artifacts + eval report

Run:
    python -m src.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import MODEL_FEATURE_COLUMNS, to_model_matrix

CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_and_split(cfg: dict):
    df = pd.read_csv(cfg["paths"]["processed_data"])
    target_col = cfg["project"]["target_column"]

    X = to_model_matrix(df)
    y = df[target_col]

    test_size = cfg["training"]["test_size"]
    val_size = cfg["training"]["val_size"]
    seed = cfg["training"]["random_state"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # val_size is expressed as a fraction of the *original* dataset; convert
    # to a fraction of the remaining train_full set.
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=relative_val_size, random_state=seed, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_baseline_pipeline(seed: int) -> Pipeline:
    """Logistic Regression: simple, fast, interpretable -> our baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def build_candidate_pipeline(seed: int) -> Pipeline:
    """Random Forest: captures non-linear interactions -> our candidate."""
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )),
    ])


def evaluate_model(model, X, y) -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y, preds), 4),
        "roc_auc": round(roc_auc_score(y, probs), 4),
        "n_samples": int(len(y)),
    }


def decide_promotion(baseline_val: dict, candidate_val: dict, cfg: dict, split_label: str = "val") -> tuple[bool, str]:
    """
    Guardrail rule (as suggested in the assignment):
      Only promote the candidate if:
        candidate ROC AUC >= min_roc_auc, AND
        candidate is not worse than baseline by more than max_allowed_regression_vs_baseline

    split_label only affects the wording of the returned reason string
    (e.g. src/retrain.py reuses this function to compare TEST metrics
    against the current production model, not the usual VAL comparison
    train.py does between baseline and candidate).
    """
    min_auc = cfg["promotion_guardrail"]["min_roc_auc"]
    max_regression = cfg["promotion_guardrail"]["max_allowed_regression_vs_baseline"]

    meets_floor = candidate_val["roc_auc"] >= min_auc
    not_worse_than_baseline = (baseline_val["roc_auc"] - candidate_val["roc_auc"]) <= max_regression

    if meets_floor and not_worse_than_baseline:
        return True, (
            f"Promoted candidate: {split_label} ROC AUC {candidate_val['roc_auc']} >= {min_auc} "
            f"and within {max_regression} of baseline ({baseline_val['roc_auc']})."
        )
    reasons = []
    if not meets_floor:
        reasons.append(f"candidate {split_label} ROC AUC {candidate_val['roc_auc']} < floor {min_auc}")
    if not not_worse_than_baseline:
        reasons.append(
            f"candidate {split_label} ROC AUC {candidate_val['roc_auc']} regresses more than "
            f"{max_regression} vs baseline {baseline_val['roc_auc']}"
        )
    return False, "Kept baseline: " + "; ".join(reasons)


def main():
    cfg = load_config()
    seed = cfg["training"]["random_state"]

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(cfg)
    print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # --- Baseline ---
    baseline = build_baseline_pipeline(seed)
    baseline.fit(X_train, y_train)
    baseline_val_metrics = evaluate_model(baseline, X_val, y_val)
    baseline_test_metrics = evaluate_model(baseline, X_test, y_test)
    print("Baseline (LogisticRegression) val:", baseline_val_metrics)

    # --- Candidate ---
    candidate = build_candidate_pipeline(seed)
    candidate.fit(X_train, y_train)
    candidate_val_metrics = evaluate_model(candidate, X_val, y_val)
    candidate_test_metrics = evaluate_model(candidate, X_test, y_test)
    print("Candidate (RandomForest) val:", candidate_val_metrics)

    # --- Promotion decision ---
    promote, reason = decide_promotion(baseline_val_metrics, candidate_val_metrics, cfg)
    print(reason)

    promoted_model = candidate if promote else baseline
    promoted_name = cfg["training"]["candidate_model"] if promote else cfg["training"]["baseline_model"]
    promoted_test_metrics = candidate_test_metrics if promote else baseline_test_metrics

    # --- Save artifacts ---
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, models_dir / "baseline_model.pkl")
    joblib.dump(candidate, models_dir / "candidate_model.pkl")
    joblib.dump(promoted_model, models_dir / "production_model.pkl")

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    version_meta = {
        "model_version": version,
        "promoted_model_type": promoted_name,
        "promoted": promote,
        "reason": reason,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "promoted_test_metrics": promoted_test_metrics,
    }
    with open(cfg["paths"]["model_version_file"], "w") as f:
        json.dump(version_meta, f, indent=2)

    # --- Save evaluation report ---
    eval_dir = Path(cfg["paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_version": version,
        "trained_at": version_meta["trained_at"],
        "baseline": {"val": baseline_val_metrics, "test": baseline_test_metrics},
        "candidate": {"val": candidate_val_metrics, "test": candidate_test_metrics},
        "decision": {"promoted": promote, "reason": reason, "promoted_model": promoted_name},
    }
    report_path = eval_dir / f"eval_report_{version}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    # Also keep a stable "latest" pointer for the monitoring/serving code to read.
    with open(eval_dir / "eval_report_latest.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved models to {models_dir}/, version file to {cfg['paths']['model_version_file']}")
    print(f"Saved eval report to {report_path}")


if __name__ == "__main__":
    main()
