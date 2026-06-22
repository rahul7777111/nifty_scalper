from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from market_data import Candle
from retraining_validation import (
    classification_metrics,
    compute_final_day_completeness,
    generate_purged_splits,
    generate_purged_embargoed_cv_splits,
    optimize_threshold_for_economic_edge,
    prune_redundant_features,
)


def _session_candles(day: datetime, count: int = 375) -> list[Candle]:
    base = day.replace(hour=9, minute=15, second=0, microsecond=0)
    candles = []
    for idx in range(count):
        ts = base + timedelta(minutes=idx)
        candles.append(Candle(time=ts, open=100.0, high=101.0, low=99.0, close=100.5, volume=0.0))
    return candles


def test_compute_final_day_completeness_complete_session() -> None:
    candles = _session_candles(datetime(2026, 6, 2))
    report = compute_final_day_completeness(candles)
    assert report.is_complete is True
    assert report.is_provisional is False
    assert report.candle_count == 375


def test_generate_purged_splits_respects_gap() -> None:
    splits = generate_purged_splits(1000, n_splits=5, gap=5, min_train_samples=250, min_test_samples=50)
    assert splits
    for split in splits:
        assert split["train_end"] <= split["validation_start"] - 5


def test_classification_metrics_basic() -> None:
    metrics = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] >= 0.99


def test_generate_purged_embargoed_cv_splits_blocks_overlap() -> None:
    index = pd.date_range("2026-06-01 09:15:00", periods=60, freq="min", tz="Asia/Kolkata")
    df = pd.DataFrame({"feature": range(60)}, index=index)
    splits = generate_purged_embargoed_cv_splits(df, label_horizon_bars=5, embargo_pct=0.20, n_splits=3)
    assert splits
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert max(train_idx) < min(val_idx)
        purge_gap = min(val_idx) - max(train_idx)
        assert purge_gap >= 5


def test_optimize_threshold_for_economic_edge_respects_trade_floor() -> None:
    y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    y_prob = [0.95, 0.90, 0.88, 0.70, 0.55, 0.45, 0.80, 0.30, 0.65, 0.25]
    result = optimize_threshold_for_economic_edge(y_true, y_prob, min_trades=3)
    assert 0.01 <= result["threshold"] <= 0.99
    assert result["trades_count"] >= 3
    assert result["f1"] > 0.0


def test_prune_redundant_features_drops_known_and_dynamic_columns() -> None:
    df = pd.DataFrame(
        {
            "close_vs_open_pct": [0.1, 0.2, 0.3],
            "ctx_adx": [10.0, 11.0, 12.0],
            "alpha": [1.0, 2.0, 3.0],
            "beta": [2.0, 4.0, 6.0],
            "gamma": [3.0, 1.0, 2.0],
        }
    )
    pruned = prune_redundant_features(df, threshold=0.98)
    assert "close_vs_open_pct" not in pruned.columns
    assert "ctx_adx" not in pruned.columns
    assert not {"alpha", "beta"}.issubset(set(pruned.columns))
