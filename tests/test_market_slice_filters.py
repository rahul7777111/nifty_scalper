from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:15", periods=6, freq="5min"),
            "option_type": ["CE", "PE", "CE", "PE", "CE", "PE"],
            "moneyness": [0.98, 1.00, 1.03, 1.08, 1.15, 0.90],
            "dte_days": [1, 1, 0, 2, 3, 4],
            "ltp": [10, 11, 12, 13, 14, 15],
            "bid_ask_spread_pct": [0.01, 0.02, 0.01, 0.03, 0.01, 0.01],
            "profitable_trade_label": [1, 0, 1, 0, 1, 0],
            "net_forward_return": [0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
        }
    )


def test_ce_only_filter_keeps_ce_rows_only() -> None:
    res = retrain._apply_market_slice_filters(_frame(), {"ce_only": True})
    assert set(res["filtered_df"]["option_type"]) == {"CE"}


def test_pe_only_filter_keeps_pe_rows_only() -> None:
    res = retrain._apply_market_slice_filters(_frame(), {"pe_only": True})
    assert set(res["filtered_df"]["option_type"]) == {"PE"}


def test_itm_only_filter_keeps_itm_rows_only() -> None:
    res = retrain._apply_market_slice_filters(_frame(), {"itm_only": True})
    assert all(retrain._moneyness_bucket(value) == "ITM" for value in res["filtered_df"]["moneyness"])


def test_filters_are_applied_before_chronological_split() -> None:
    res = retrain._apply_market_slice_filters(_frame().sample(frac=1.0, random_state=42), {"exclude_0dte": True})
    assert pd.Index(res["filtered_df"]["timestamp"]).is_monotonic_increasing
    assert (res["filtered_df"]["dte_days"] > 0).all()
