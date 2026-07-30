"""
monitoring.py
-------------
Lightweight data quality + drift checks, meant to be run periodically
(e.g. daily, right after ingestion) against a "recent batch" compared
to the training data distribution.

This intentionally does NOT try to be a production-grade monitoring
stack (no Prometheus/Grafana wiring) -- per the assignment, the goal is
to *design* the monitoring plan (see README.md) and *implement one
lightweight working check*, which is what this module does:

  1. Missing-value / out-of-range check on the recent batch.
  2. Mean/std drift check: for each configured feature, compare the
     recent batch's mean to the training set's mean, expressed in
     standard-deviation units (a simple z-score-style drift signal).
     This stands in for a more rigorous test (e.g. population stability
     index or a KS-test) while being easy to read and reason about.

Run:
    python -m src.monitoring
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def check_data_quality(df: pd.DataFrame, cfg: dict) -> dict:
    max_missing = cfg["monitoring"]["max_missing_fraction"]
    issues = []

    missing_fracs = df.isna().mean()
    for col, frac in missing_fracs.items():
        if frac > max_missing:
            issues.append(f"Column '{col}' has {frac:.2%} missing values (> {max_missing:.0%} threshold)")

    numeric_cols = ["tenure_months", "monthly_charges", "total_charges"]
    for col in numeric_cols:
        if col in df.columns and (df[col] < 0).any():
            n_bad = int((df[col] < 0).sum())
            issues.append(f"Column '{col}' has {n_bad} out-of-range (negative) values")

    return {"n_rows_checked": len(df), "issues": issues, "passed": len(issues) == 0}


def check_drift(train_df: pd.DataFrame, recent_df: pd.DataFrame, cfg: dict) -> dict:
    features = cfg["monitoring"]["drift_features"]
    threshold = cfg["monitoring"]["drift_zscore_threshold"]

    results = {}
    max_score = 0.0
    for feat in features:
        train_mean, train_std = train_df[feat].mean(), train_df[feat].std()
        recent_mean = recent_df[feat].mean()
        if train_std == 0:
            score = 0.0
        else:
            score = abs(recent_mean - train_mean) / train_std
        max_score = max(max_score, score)
        results[feat] = {
            "train_mean": round(float(train_mean), 3),
            "recent_mean": round(float(recent_mean), 3),
            "drift_zscore": round(float(score), 3),
            "flagged": bool(score > threshold),
        }

    return {
        "feature_drift": results,
        "max_drift_zscore": round(float(max_score), 3),
        "drift_detected": any(r["flagged"] for r in results.values()),
        "threshold": threshold,
    }


def run_monitoring_report(cfg: dict) -> dict:
    processed_path = Path(cfg["paths"]["processed_data"])
    train_df = pd.read_csv(processed_path)

    # "Recent batch" = the most recently ingested rows. In this mini
    # system we approximate that as the last 400 rows of the processed
    # table (a real system would use an ingestion-date column instead).
    recent_df = train_df.tail(400)

    quality = check_data_quality(recent_df, cfg)
    drift = check_drift(train_df, recent_df, cfg)

    report = {"data_quality": quality, "drift": drift}

    if not quality["passed"]:
        print("DATA QUALITY WARNING:")
        for issue in quality["issues"]:
            print(f"  - {issue}")
    else:
        print("Data quality check passed.")

    if drift["drift_detected"]:
        print("DRIFT WARNING: the following features exceeded the drift threshold:")
        for feat, r in drift["feature_drift"].items():
            if r["flagged"]:
                print(f"  - {feat}: train_mean={r['train_mean']} recent_mean={r['recent_mean']} "
                      f"z={r['drift_zscore']} (threshold {drift['threshold']})")
    else:
        print(f"No drift detected (max z-score {drift['max_drift_zscore']} <= {drift['threshold']}).")

    return report


def main():
    cfg = load_config()
    report = run_monitoring_report(cfg)
    out_path = Path(cfg["paths"]["eval_dir"]) / "monitoring_report_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved monitoring report to {out_path}")


if __name__ == "__main__":
    main()
