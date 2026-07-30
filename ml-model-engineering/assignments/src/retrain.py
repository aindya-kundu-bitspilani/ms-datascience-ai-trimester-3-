"""
retrain.py
----------
Executes an actual retraining cycle and decides whether to promote the
result to production. This is distinct from src/retrain_trigger.py,
which only decides WHETHER a retrain should happen (data age / AUC drop
/ drift signals) -- this module is what a scheduler would call once that
trigger fires.

    load current production metrics -> retrain baseline + candidate on
    the CURRENT processed training table (which src/data_ingestion.py
    has already been growing with every new batch) -> pick the better
    of the two exactly as src/train.py does -> compare that retrained
    model's test ROC AUC against the model currently in production,
    using the SAME promotion guardrail function (src/train.py::decide_promotion)
    -> if it wins, archive the old production artifacts and deploy the
    new ones; if not, leave production untouched and save the retrain
    attempt for audit.

Why the training table itself is not reassembled here (unlike a design
that merges "new labels" files at retrain time): src/data_ingestion.py
already appends every new batch into data/processed/training_data.csv
as it arrives, so "the current training table" already IS the expanded
pool -- retraining on it is just calling the same training routine again
on more rows. This keeps retrain.py simple and guarantees it can never
drift out of sync with what ingestion has actually collected.

Run:
    python -m src.retrain
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

from src.train import (
    build_baseline_pipeline,
    build_candidate_pipeline,
    decide_promotion,
    evaluate_model,
    load_and_split,
)
from src.features import MODEL_FEATURE_COLUMNS

CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_current_production(cfg: dict) -> dict | None:
    """Read what's currently deployed, so the retrain has something to beat."""
    version_path = Path(cfg["paths"]["model_version_file"])
    if not version_path.exists():
        return None
    return json.loads(version_path.read_text())


def archive_current_production(cfg: dict, current_meta: dict) -> None:
    """Move the outgoing production model + its eval report into an archive
    folder before overwriting them, so every past production model stays
    inspectable."""
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir = Path(cfg["paths"]["eval_dir"])
    old_version = current_meta["model_version"]

    archive_dir = models_dir / "archive" / old_version
    archive_dir.mkdir(parents=True, exist_ok=True)

    old_model_path = models_dir / "production_model.pkl"
    if old_model_path.exists():
        shutil.copy2(old_model_path, archive_dir / "production_model.pkl")
    (archive_dir / "model_version.json").write_text(json.dumps(current_meta, indent=2))

    old_report = eval_dir / "eval_report_latest.json"
    if old_report.exists():
        eval_archive_dir = eval_dir / "archive"
        eval_archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_report, eval_archive_dir / f"eval_report_{old_version}.json")


def run_retraining(cfg: dict) -> dict:
    seed = cfg["training"]["random_state"]
    current_meta = load_current_production(cfg)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(cfg)
    print(f"Retraining on current processed table: "
          f"train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)} rows")

    # --- Retrain both candidates on the (possibly larger) current pool ---
    baseline = build_baseline_pipeline(seed)
    baseline.fit(X_train, y_train)
    baseline_val = evaluate_model(baseline, X_val, y_val)
    baseline_test = evaluate_model(baseline, X_test, y_test)

    candidate = build_candidate_pipeline(seed)
    candidate.fit(X_train, y_train)
    candidate_val = evaluate_model(candidate, X_val, y_val)
    candidate_test = evaluate_model(candidate, X_test, y_test)

    print(f"  retrained baseline (LogisticRegression) val: {baseline_val}")
    print(f"  retrained candidate (RandomForest)      val: {candidate_val}")

    # Pick the better of the two retrained models, same rule train.py uses.
    inner_promote, inner_reason = decide_promotion(baseline_val, candidate_val, cfg)
    new_model = candidate if inner_promote else baseline
    new_name = cfg["training"]["candidate_model"] if inner_promote else cfg["training"]["baseline_model"]
    new_val = candidate_val if inner_promote else baseline_val
    new_test = candidate_test if inner_promote else baseline_test
    print(f"  best of the two retrained models: {new_name} ({inner_reason})")

    # --- Compare the new best model against what's currently in production ---
    if current_meta is None:
        # Nothing deployed yet -- this retrain simply becomes production.
        promote_to_prod, prod_reason = True, "No model currently in production; deploying retrained model."
    else:
        current_test_auc = current_meta["promoted_test_metrics"]["roc_auc"]
        promote_to_prod, prod_reason = decide_promotion(
            baseline_val={"roc_auc": current_test_auc},
            candidate_val={"roc_auc": new_test["roc_auc"]},
            cfg=cfg,
            split_label="test",
        )
        prod_reason = (
            prod_reason
            .replace("Promoted candidate", "Promoted retrained model")
            .replace("Kept baseline", "Kept current production model")
            .replace("candidate test", "retrained model's test")
            .replace("baseline (", "current production model (")
            .replace("vs baseline", "vs current production model")
        )
    print(f"  vs. current production: {prod_reason}")

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    retrain_report = {
        "retrain_version": version,
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(X_train) + len(X_val) + len(X_test),
        "retrained_baseline": {"val": baseline_val, "test": baseline_test},
        "retrained_candidate": {"val": candidate_val, "test": candidate_test},
        "best_of_retrain": {"model_type": new_name, "val": new_val, "test": new_test},
        "current_production": current_meta,
        "promoted_to_production": promote_to_prod,
        "promotion_reason": prod_reason,
    }

    eval_dir = Path(cfg["paths"]["eval_dir"])

    if promote_to_prod:
        if current_meta is not None:
            archive_current_production(cfg, current_meta)

        models_dir = Path(cfg["paths"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_model, models_dir / "production_model.pkl")

        version_meta = {
            "model_version": version,
            "promoted_model_type": new_name,
            "promoted": True,
            "reason": f"[retrain] {prod_reason}",
            "trained_at": retrain_report["retrained_at"],
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "promoted_test_metrics": new_test,
        }
        with open(cfg["paths"]["model_version_file"], "w") as f:
            json.dump(version_meta, f, indent=2)

        full_report = {
            "model_version": version,
            "trained_at": retrain_report["retrained_at"],
            "baseline": {"val": baseline_val, "test": baseline_test},
            "candidate": {"val": candidate_val, "test": candidate_test},
            "decision": {"promoted": inner_promote, "reason": inner_reason, "promoted_model": new_name},
            "retrain_context": retrain_report,
        }
        with open(eval_dir / f"eval_report_{version}.json", "w") as f:
            json.dump(full_report, f, indent=2)
        with open(eval_dir / "eval_report_latest.json", "w") as f:
            json.dump(full_report, f, indent=2)

        print(f"PROMOTED: new production model is {new_name} ({version}). "
              f"Old model archived under models/archive/.")
    else:
        candidates_dir = eval_dir / "retrain_candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidates_dir / f"retrain_candidate_{version}.json"
        with open(candidate_path, "w") as f:
            json.dump(retrain_report, f, indent=2)
        print(f"NOT PROMOTED: production model unchanged. "
              f"Retrain attempt saved to {candidate_path} for audit.")

    return retrain_report


def main():
    cfg = load_config()
    run_retraining(cfg)


if __name__ == "__main__":
    main()
