import json
import os

import pandas as pd
import pytest

import src.retrain as retrain_mod
from src.load_dataset import main as build_dataset


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """A config pointing entirely at tmp_path, so this test never touches
    the real models/ or artifacts/eval/ directories."""
    if not os.path.exists("data/raw/churn_initial.csv"):
        build_dataset(argv=[])

    models_dir = tmp_path / "models"
    eval_dir = tmp_path / "artifacts" / "eval"
    models_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    # Use the real processed table if present (fast path); otherwise fall
    # back to the raw initial file so this test works on a fresh clone.
    processed = "data/processed/training_data.csv"
    if not os.path.exists(processed):
        processed = "data/raw/churn_initial.csv"

    return {
        "project": {"target_column": "churn"},
        "paths": {
            "processed_data": processed,
            "models_dir": str(models_dir),
            "eval_dir": str(eval_dir),
            "model_version_file": str(models_dir / "model_version.json"),
        },
        "training": {
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 1,
            "baseline_model": "logistic_regression",
            "candidate_model": "random_forest",
        },
        "promotion_guardrail": {"min_roc_auc": 0.0, "max_allowed_regression_vs_baseline": 0.01},
    }


def test_retrain_promotes_when_no_current_production(cfg):
    # No model_version.json exists yet in this tmp models_dir -> first
    # retrain should always become production.
    report = retrain_mod.run_retraining(cfg)
    assert report["promoted_to_production"] is True

    version_file = os.path.join(cfg["paths"]["models_dir"], "model_version.json")
    assert os.path.exists(version_file)
    meta = json.loads(open(version_file).read())
    assert meta["promoted"] is True


def test_retrain_rejects_when_current_production_is_much_stronger(cfg):
    # Forge a "current production" that's essentially unbeatable (AUC 0.999),
    # so the guardrail must refuse to promote the freshly retrained model.
    version_file = os.path.join(cfg["paths"]["models_dir"], "model_version.json")
    fake_current = {
        "model_version": "v_fake_strong",
        "promoted_model_type": "random_forest",
        "promoted": True,
        "reason": "test fixture",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "feature_columns": [],
        "promoted_test_metrics": {"accuracy": 0.99, "roc_auc": 0.999, "n_samples": 100},
    }
    with open(version_file, "w") as f:
        json.dump(fake_current, f)

    report = retrain_mod.run_retraining(cfg)
    assert report["promoted_to_production"] is False

    # Production file must be untouched -- still the fake metadata we wrote.
    meta = json.loads(open(version_file).read())
    assert meta["model_version"] == "v_fake_strong"

    # ...and the rejected attempt should be saved for audit.
    candidates_dir = os.path.join(cfg["paths"]["eval_dir"], "retrain_candidates")
    assert os.path.isdir(candidates_dir)
    assert len(os.listdir(candidates_dir)) == 1


def test_retrain_archives_old_model_on_promotion(cfg):
    # Seed a "current production" that a freshly retrained model should
    # comfortably beat (AUC 0.01), then confirm the old one gets archived.
    models_dir = cfg["paths"]["models_dir"]
    version_file = os.path.join(models_dir, "model_version.json")
    fake_current = {
        "model_version": "v_fake_weak",
        "promoted_model_type": "logistic_regression",
        "promoted": True,
        "reason": "test fixture",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "feature_columns": [],
        "promoted_test_metrics": {"accuracy": 0.5, "roc_auc": 0.01, "n_samples": 100},
    }
    with open(version_file, "w") as f:
        json.dump(fake_current, f)
    # Also drop a dummy "old model file" so archiving has something to copy.
    with open(os.path.join(models_dir, "production_model.pkl"), "wb") as f:
        f.write(b"dummy")

    report = retrain_mod.run_retraining(cfg)
    assert report["promoted_to_production"] is True

    archive_dir = os.path.join(models_dir, "archive", "v_fake_weak")
    assert os.path.isdir(archive_dir)
    assert os.path.exists(os.path.join(archive_dir, "production_model.pkl"))
    assert os.path.exists(os.path.join(archive_dir, "model_version.json"))
