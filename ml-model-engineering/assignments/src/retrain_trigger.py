"""
retrain_trigger.py
-------------------
Defines the retraining trigger logic as real (but simple) functions,
combining the three signals called out in the assignment:

  1. Enough new data has accumulated since the last training run.
  2. Recent labeled-feedback AUC has dropped meaningfully vs. the
     model currently in production.
  3. Feature drift score (from monitoring.py) exceeds a threshold.

This is intentionally NOT wired to a real scheduler (e.g. Airflow/cron)
-- per the assignment, pseudo-code / a callable function is enough. In
a real deployment, `should_retrain()` would be invoked by a scheduled
job that has access to a feedback/labels table and the monitoring
output.

Run (uses the values already computed by train.py / monitoring.py):
    python -m src.retrain_trigger
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

CONFIG_PATH = Path("configs/config.yaml")


@dataclass
class RetrainSignals:
    days_since_last_training: int
    recent_labeled_auc: float | None   # None if not enough labeled feedback yet
    production_auc: float
    drift_zscore: float


@dataclass
class RetrainDecision:
    should_retrain: bool
    triggered_by: list[str]


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def should_retrain(signals: RetrainSignals, cfg: dict) -> RetrainDecision:
    triggers = cfg["retraining_triggers"]
    reasons: list[str] = []

    # Signal 1: enough new data has accumulated
    if signals.days_since_last_training >= triggers["min_new_days_of_data"]:
        reasons.append(
            f"{signals.days_since_last_training} days of new data "
            f">= minimum {triggers['min_new_days_of_data']}"
        )

    # Signal 2: measured performance drop on recent labeled feedback
    if signals.recent_labeled_auc is not None:
        auc_drop = signals.production_auc - signals.recent_labeled_auc
        if auc_drop >= triggers["auc_drop_threshold"]:
            reasons.append(
                f"recent labeled AUC dropped by {auc_drop:.3f} "
                f">= threshold {triggers['auc_drop_threshold']}"
            )

    # Signal 3: feature drift
    if signals.drift_zscore >= triggers["drift_score_threshold"]:
        reasons.append(
            f"drift z-score {signals.drift_zscore:.2f} >= threshold {triggers['drift_score_threshold']}"
        )

    return RetrainDecision(should_retrain=len(reasons) > 0, triggered_by=reasons)


def _load_latest_drift_zscore(cfg: dict) -> float:
    path = Path(cfg["paths"]["eval_dir"]) / "monitoring_report_latest.json"
    if not path.exists():
        return 0.0
    report = json.loads(path.read_text())
    return report.get("drift", {}).get("max_drift_zscore", 0.0)


def _load_production_auc(cfg: dict) -> float:
    path = Path(cfg["paths"]["eval_dir"]) / "eval_report_latest.json"
    if not path.exists():
        return 0.0
    report = json.loads(path.read_text())
    decision = report.get("decision", {})
    promoted = "candidate" if decision.get("promoted") else "baseline"
    return report.get(promoted, {}).get("test", {}).get("roc_auc", 0.0)


def main():
    cfg = load_config()

    # Example wiring: pull the real drift score and production AUC we just
    # computed from train.py / monitoring.py; days_since_last_training and
    # recent_labeled_auc would come from a scheduler + a labeled-feedback
    # table in a real system, so here they're illustrative example values.
    signals = RetrainSignals(
        days_since_last_training=8,
        recent_labeled_auc=None,   # no fresh labels available yet in this demo
        production_auc=_load_production_auc(cfg),
        drift_zscore=_load_latest_drift_zscore(cfg),
    )

    decision = should_retrain(signals, cfg)
    print(f"Signals: {signals}")
    if decision.should_retrain:
        print("RETRAIN TRIGGERED:")
        for reason in decision.triggered_by:
            print(f"  - {reason}")
    else:
        print("No retrain trigger fired.")

    out = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals.__dict__,
        "should_retrain": decision.should_retrain,
        "triggered_by": decision.triggered_by,
    }
    out_path = Path(cfg["paths"]["eval_dir"]) / "retrain_decision_latest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved retrain decision to {out_path}")


if __name__ == "__main__":
    main()
