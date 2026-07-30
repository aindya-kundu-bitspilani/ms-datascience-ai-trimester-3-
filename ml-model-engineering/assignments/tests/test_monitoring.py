import pandas as pd
import pytest

from src.monitoring import check_data_quality, check_drift, run_monitoring_report

BASE_CFG = {
    "monitoring": {
        "drift_features": ["tenure_months", "monthly_charges"],
        "drift_zscore_threshold": 2.0,
        "max_missing_fraction": 0.05,
    }
}


def _df(n, tenure_mean=30.0, charges_mean=65.0, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tenure_months": rng.normal(tenure_mean, 15, size=n).clip(0),
        "monthly_charges": rng.normal(charges_mean, 20, size=n).clip(15),
        "total_charges": rng.normal(2000, 1500, size=n).clip(0),
    })


def test_data_quality_passes_on_clean_batch():
    df = _df(200)
    result = check_data_quality(df, BASE_CFG)
    assert result["passed"] is True
    assert result["issues"] == []
    assert result["n_rows_checked"] == 200


def test_data_quality_flags_missing_values_above_threshold():
    df = _df(200)
    df.loc[: int(0.20 * len(df)), "monthly_charges"] = None  # 20% missing >> 5% threshold
    result = check_data_quality(df, BASE_CFG)
    assert result["passed"] is False
    assert any("monthly_charges" in issue for issue in result["issues"])


def test_data_quality_flags_negative_values():
    df = _df(200)
    df.loc[0, "tenure_months"] = -5
    result = check_data_quality(df, BASE_CFG)
    assert result["passed"] is False
    assert any("tenure_months" in issue and "negative" in issue for issue in result["issues"])


def test_check_drift_no_drift_when_distributions_match():
    train_df = _df(2000, tenure_mean=30, charges_mean=65, seed=1)
    recent_df = _df(400, tenure_mean=30, charges_mean=65, seed=2)  # same params, different sample
    result = check_drift(train_df, recent_df, BASE_CFG)
    assert result["drift_detected"] is False
    assert all(not r["flagged"] for r in result["feature_drift"].values())


def test_check_drift_flags_shifted_feature():
    train_df = _df(2000, tenure_mean=30, charges_mean=65, seed=1)
    # monthly_charges shifted by several standard deviations -> should trip the z-score check
    recent_df = _df(400, tenure_mean=30, charges_mean=200, seed=2)
    result = check_drift(train_df, recent_df, BASE_CFG)
    assert result["drift_detected"] is True
    assert result["feature_drift"]["monthly_charges"]["flagged"] is True
    assert result["feature_drift"]["tenure_months"]["flagged"] is False


def test_check_drift_zero_variance_feature_does_not_crash():
    train_df = pd.DataFrame({"tenure_months": [5] * 100, "monthly_charges": [50] * 100})
    recent_df = pd.DataFrame({"tenure_months": [5] * 50, "monthly_charges": [50] * 50})
    result = check_drift(train_df, recent_df, BASE_CFG)
    # std == 0 is handled explicitly (score forced to 0.0) rather than dividing by zero
    assert result["feature_drift"]["tenure_months"]["drift_zscore"] == 0.0


def test_run_monitoring_report_is_json_serializable(tmp_path):
    # Regression-style check: every value in the report must survive
    # json.dumps (numpy bool_/float64 types have bitten this kind of
    # code before -- see src/monitoring.py's explicit float()/bool() casts).
    import json

    processed = tmp_path / "training_data.csv"
    _df(500).assign(churn=[0, 1] * 250).to_csv(processed, index=False)

    cfg = {
        "paths": {"processed_data": str(processed), "eval_dir": str(tmp_path)},
        "monitoring": BASE_CFG["monitoring"],
    }
    report = run_monitoring_report(cfg)
    json.dumps(report)  # must not raise
