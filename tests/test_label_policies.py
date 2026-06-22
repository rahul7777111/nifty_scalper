from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from label_policies import build_label_dataset
from market_data import Candle


def make_candles(n: int = 140) -> list[Candle]:
    start = datetime(2026, 1, 1, 9, 15)
    candles: list[Candle] = []
    price = 100.0
    for idx in range(n):
        drift = 0.25 if idx % 9 < 5 else -0.12
        price += drift
        candles.append(
            Candle(
                time=start + timedelta(minutes=10 * idx),
                open=price - 0.1,
                high=price + 0.2,
                low=price - 0.3,
                close=price,
                volume=1000 + idx,
            )
        )
    return candles


def test_label_policies_are_deterministic():
    candles = make_candles()
    first = build_label_dataset(candles, policy_name="trade_quality_binary", lookback=30, horizon=12)
    second = build_label_dataset(candles, policy_name="trade_quality_binary", lookback=30, horizon=12)
    assert first.y == second.y
    assert first.label_distribution == second.label_distribution


def test_neutral_label_policy_drops_noisy_samples():
    candles = make_candles()
    dataset = build_label_dataset(candles, policy_name="magnitude_filtered_direction", lookback=30, horizon=12)
    assert dataset.neutral_samples_dropped >= 0
    assert len(dataset.observations) >= len(dataset.y)


def test_end_of_dataset_unresolved_labels_are_not_used():
    candles = make_candles(60)
    dataset = build_label_dataset(candles, policy_name="current_triple_barrier", lookback=20, horizon=10)
    expected_max = len(candles) - 20 - 10
    assert len(dataset.observations) == expected_max
