from __future__ import annotations

from datetime import datetime, timedelta

from feature_drift_monitor import FeatureDriftMonitor
from market_data import Candle
from market_regime_detector import MarketRegimeDetector
from prediction_drift_monitor import PredictionDriftMonitor


def test_feature_drift_monitor_flags_shifted_feature():
    monitor = FeatureDriftMonitor(psi_threshold=0.1, kl_threshold=0.05, mean_shift_threshold=0.3, variance_shift_threshold=1.2)
    training = {"x": [0.1, 0.2, 0.15, 0.18, 0.22, 0.19], "y": [1, 1, 1, 1, 1, 1]}
    live = {"x": [0.8, 0.85, 0.9, 0.95, 1.0, 1.05], "y": [1, 1, 1, 1, 1, 1]}
    report = monitor.evaluate(training, live)
    alert_features = {row["feature"] for row in report["alerts"]}
    assert "x" in alert_features


def test_prediction_drift_monitor_detects_confidence_shift():
    monitor = PredictionDriftMonitor(confidence_mean_threshold=0.05, class_distribution_threshold=0.2, calibration_threshold=0.05)
    rows = []
    for idx in range(40):
        rows.append(
            {
                "probability": 0.52 if idx < 20 else 0.80,
                "confidence": 0.52 if idx < 20 else 0.80,
                "prediction": 1.0,
                "pnl": 100.0 if idx % 2 == 0 else -50.0,
            }
        )
    report = monitor.evaluate(rows)
    assert report["status"] == "alert"
    assert any("confidence_drift" in alert for alert in report["alerts"])


def test_prediction_drift_monitor_flags_feature_drift_and_kill_switch():
    class DummyCfg:
        enable_ml_signals = True

    class DummyStrategy:
        def __init__(self):
            self.cfg = DummyCfg()
            self._last_ml_bet_multiplier = 0.75

    baseline_monitor = PredictionDriftMonitor(
        psi_threshold=0.10,
        ks_pvalue_threshold=0.05,
        baseline_brier_threshold=0.25,
        strategy_ref=DummyStrategy(),
    )
    baseline_monitor._feature_baseline = {
        "rolling_range_position_20": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
        "realized_vol_30": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015],
    }
    for _ in range(100):
        report = baseline_monitor.ingest_live_observation(
            feature_snapshot={
                "rolling_range_position_20": 0.95,
                "realized_vol_30": 0.09,
            }
        )
    assert report is not None
    assert any("CRITICAL_FEATURE_DRIFT_WARNING" in alert for alert in report["alerts"])
    assert getattr(baseline_monitor.strategy_ref, "_ml_drift_kill_switch_active", False) is True
    assert baseline_monitor.strategy_ref.cfg.enable_ml_signals is False


def test_prediction_drift_monitor_flags_calibration_collapse():
    class DummyCfg:
        enable_ml_signals = True

    class DummyStrategy:
        def __init__(self):
            self.cfg = DummyCfg()
            self._last_ml_bet_multiplier = 1.0

    monitor = PredictionDriftMonitor(strategy_ref=DummyStrategy(), baseline_brier_threshold=0.25)
    for idx in range(50):
        report = monitor.ingest_resolved_outcome(
            probability=0.99 if idx % 2 == 0 else 0.01,
            realized_label=0 if idx % 2 == 0 else 1,
            prediction_id=f"pred_{idx}",
        )
    assert report is not None
    assert any("CALIBRATION_COLLAPSE_ALERT" in alert for alert in report["alerts"])
    assert monitor.strategy_ref.cfg.enable_ml_signals is False


def test_market_regime_detector_classifies_high_volatility():
    detector = MarketRegimeDetector()
    start = datetime(2026, 1, 1, 9, 15)
    candles = []
    price = 100.0
    for idx in range(60):
        price += 2.5 if idx % 2 == 0 else -2.0
        candles.append(
            Candle(
                time=start + timedelta(minutes=5 * idx),
                open=price - 1.0,
                high=price + 2.0,
                low=price - 2.5,
                close=price,
                volume=1000 + idx,
            )
        )
    snapshot = detector.classify_candles(candles, lookback=30)
    assert snapshot.regime in {"high_volatility", "trending"}
