from src.retrain_trigger import RetrainSignals, should_retrain

CFG = {
    "retraining_triggers": {
        "min_new_days_of_data": 7,
        "auc_drop_threshold": 0.03,
        "drift_score_threshold": 2.0,
    }
}


def test_no_trigger_when_all_signals_quiet():
    signals = RetrainSignals(
        days_since_last_training=2, recent_labeled_auc=0.85, production_auc=0.85, drift_zscore=0.1
    )
    decision = should_retrain(signals, CFG)
    assert decision.should_retrain is False
    assert decision.triggered_by == []


def test_triggers_on_data_age():
    signals = RetrainSignals(
        days_since_last_training=8, recent_labeled_auc=None, production_auc=0.85, drift_zscore=0.1
    )
    decision = should_retrain(signals, CFG)
    assert decision.should_retrain is True
    assert any("days of new data" in r for r in decision.triggered_by)


def test_triggers_on_auc_drop():
    signals = RetrainSignals(
        days_since_last_training=1, recent_labeled_auc=0.80, production_auc=0.85, drift_zscore=0.1
    )
    decision = should_retrain(signals, CFG)
    assert decision.should_retrain is True
    assert any("AUC dropped" in r for r in decision.triggered_by)


def test_does_not_trigger_on_small_auc_drop_below_threshold():
    # 0.85 -> 0.83 is only a 0.02 drop, below the 0.03 threshold
    signals = RetrainSignals(
        days_since_last_training=1, recent_labeled_auc=0.83, production_auc=0.85, drift_zscore=0.1
    )
    decision = should_retrain(signals, CFG)
    assert decision.should_retrain is False


def test_triggers_on_drift():
    signals = RetrainSignals(
        days_since_last_training=1, recent_labeled_auc=None, production_auc=0.85, drift_zscore=2.5
    )
    decision = should_retrain(signals, CFG)
    assert decision.should_retrain is True
    assert any("drift z-score" in r for r in decision.triggered_by)


def test_multiple_signals_all_reported():
    signals = RetrainSignals(
        days_since_last_training=10, recent_labeled_auc=0.70, production_auc=0.85, drift_zscore=3.0
    )
    decision = should_retrain(signals, CFG)
    assert decision.should_retrain is True
    assert len(decision.triggered_by) == 3


def test_recent_labeled_auc_none_skips_that_signal_cleanly():
    # No labeled feedback yet -> signal 2 must not fire or error
    signals = RetrainSignals(
        days_since_last_training=1, recent_labeled_auc=None, production_auc=0.85, drift_zscore=0.1
    )
    decision = should_retrain(signals, CFG)
    assert decision.should_retrain is False
