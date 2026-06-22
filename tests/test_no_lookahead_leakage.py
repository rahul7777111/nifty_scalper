from __future__ import annotations

from datetime import datetime, timedelta

from market_data import Candle
from ml_pipeline import build_market_feature_vector, build_supervised_dataset_v2, verify_no_lookahead_leakage


def make_candles(count: int = 120):
    start = datetime(2026, 1, 1, 9, 15)
    candles = []
    price = 100.0
    for idx in range(count):
        price += 0.2 if idx % 3 else -0.1
        candles.append(
            Candle(
                time=start + timedelta(minutes=idx),
                open=price - 0.2,
                high=price + 0.4,
                low=price - 0.5,
                close=price,
                volume=1000.0 + idx,
            )
        )
    return candles


def test_verify_no_lookahead_leakage_passes_for_v2_builder():
    result = verify_no_lookahead_leakage(make_candles(), build_supervised_dataset_v2, lookback=20, horizon=5)
    assert result["passed"] is True
    assert result["changed_rows"] == 0
    assert result["changed_features"] == []


def test_verify_no_lookahead_leakage_detects_future_dependent_builder():
    def leaky_builder(candles, **kwargs):
        X, y, feature_names = build_supervised_dataset_v2(candles, **kwargs)
        closes = [c.close for c in candles]
        leaky_X = []
        for row_idx, row in enumerate(X):
            candle_idx = kwargs.get("lookback", 20) + row_idx
            leaked = closes[candle_idx + 1] / closes[candle_idx] if candle_idx + 1 < len(closes) and closes[candle_idx] else 0.0
            leaky_X.append(list(row) + [leaked])
        return leaky_X, y, list(feature_names) + ["leaked_next_close_ratio"]

    result = verify_no_lookahead_leakage(make_candles(), leaky_builder, lookback=20, horizon=5)
    assert result["passed"] is False
    assert result["changed_rows"] > 0
    assert "leaked_next_close_ratio" in result["changed_features"]


def test_closed_history_features_ignore_current_bar_in_rolling_windows():
    candles = make_candles(60)
    base_values, feature_names = build_market_feature_vector(candles)
    mutated = list(candles)
    mutated[-1] = Candle(
        time=mutated[-1].time,
        open=mutated[-1].open,
        high=mutated[-1].high + 50.0,
        low=mutated[-1].low - 50.0,
        close=mutated[-1].close,
        volume=mutated[-1].volume,
    )
    mutated_values, mutated_names = build_market_feature_vector(mutated)
    assert feature_names == mutated_names

    base = dict(zip(feature_names, base_values))
    changed = dict(zip(mutated_names, mutated_values))
    assert base["rolling_range_position_20"] == changed["rolling_range_position_20"]
    assert base["dist_to_rolling_high_20"] == changed["dist_to_rolling_high_20"]
    assert base["realized_vol_30"] == changed["realized_vol_30"]
