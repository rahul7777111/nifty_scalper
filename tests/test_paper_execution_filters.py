from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_execution_filters_reduce_trade_count_and_enforce_cooldown() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-06-01T09:25:00+05:30",
            "2026-06-01T09:30:00+05:30",
            "2026-06-01T09:35:00+05:30",
        ]),
        "ltp": [10.0, 10.0, 10.0],
        "bid_ask_spread_pct": [0.01, 0.01, 0.01],
        "option_type_ce": [1, 1, 1],
        "option_type_pe": [0, 0, 0],
    })
    probs = np.array([0.9, 0.9, 0.9])
    returns = np.array([0.1, 0.1, 0.1])
    report = retrain._paper_execution_filter_report(
        frame,
        probs,
        0.5,
        returns,
        max_trades_per_day=5,
        cooldown_minutes=15,
        allow_ce=True,
        allow_pe=True,
        min_option_price=5.0,
        max_bid_ask_spread_pct=5.0,
        avoid_opening_minutes=5,
        avoid_closing_minutes=5,
    )
    assert report["before"]["trade_count"] == 3
    assert report["after"]["trade_count"] < 3


def test_max_trades_per_day_is_enforced() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([f"2026-06-01T10:{minute:02d}:00+05:30" for minute in range(10)]),
        "ltp": [10.0] * 10,
        "bid_ask_spread_pct": [0.01] * 10,
    })
    probs = np.ones(10)
    returns = np.ones(10) * 0.1
    report = retrain._paper_execution_filter_report(
        frame,
        probs,
        0.5,
        returns,
        max_trades_per_day=5,
        cooldown_minutes=0,
        allow_ce=True,
        allow_pe=True,
        min_option_price=5.0,
        max_bid_ask_spread_pct=5.0,
        avoid_opening_minutes=5,
        avoid_closing_minutes=5,
    )
    assert report["after"]["trade_count"] == 5
