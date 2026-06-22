from __future__ import annotations

import os
import sys
from datetime import date, datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from options_archive.feature_builder import build_point_in_time_features  # noqa: E402
from options_archive.symbols import archive_config_for_symbol, resolve_symbol_spec  # noqa: E402


def test_point_in_time_feature_builder_joins_snapshot_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "snapshot_id": "s1",
                "as_of_ts": datetime(2026, 5, 31, 9, 15),
                "session_date": date(2026, 5, 31),
                "underlying": "NIFTY",
                "spot": 23000.0,
                "strike": 23000.0,
                "expiry": date(2026, 6, 4),
                "option_type": "CE",
                "symbol": "NIFTY25JUN23000CE",
                "ltp": 120.0,
                "bid": 119.5,
                "ask": 120.5,
                "volume": 1000,
                "oi": 5000,
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.12,
                "theta": -0.08,
                "iv": 0.16,
            },
            {
                "snapshot_id": "s1",
                "as_of_ts": datetime(2026, 5, 31, 9, 15),
                "session_date": date(2026, 5, 31),
                "underlying": "NIFTY",
                "spot": 23000.0,
                "strike": 23000.0,
                "expiry": date(2026, 6, 4),
                "option_type": "PE",
                "symbol": "NIFTY25JUN23000PE",
                "ltp": 118.0,
                "bid": 117.5,
                "ask": 118.5,
                "volume": 900,
                "oi": 7000,
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.12,
                "theta": -0.08,
                "iv": 0.17,
            },
        ]
    )

    out = build_point_in_time_features(frame)

    assert len(out) == 2
    assert round(float(out["pcr_oi"].iloc[0]), 4) == 1.4
    assert out["feature_schema_version"].iloc[0] == "2026-05-31"
    assert out["vol_risk_premium"].notna().all()


def test_banknifty_symbol_registry_and_metadata() -> None:
    spec = resolve_symbol_spec("BANKNIFTY")
    cfg = archive_config_for_symbol("BANKNIFTY")

    assert spec.symbol_id == 2
    assert spec.index_family == "BANKNIFTY"
    assert cfg.symbol_key == "BANKNIFTY"
    assert cfg.symbol_id == 2
    assert cfg.spot_symbol == "NSE:BANKNIFTY"

    frame = pd.DataFrame(
        [
            {
                "snapshot_id": "b1",
                "as_of_ts": datetime(2026, 5, 31, 9, 15),
                "session_date": date(2026, 5, 31),
                "symbol_key": "BANKNIFTY",
                "symbol_id": 2,
                "index_family": "BANKNIFTY",
                "underlying": "BANKNIFTY",
                "spot": 50000.0,
                "strike": 50000.0,
                "expiry": date(2026, 6, 4),
                "option_type": "CE",
                "symbol": "BANKNIFTY25JUN50000CE",
                "ltp": 210.0,
                "bid": 209.5,
                "ask": 210.5,
                "volume": 1500,
                "oi": 8000,
                "delta": 0.55,
                "gamma": 0.012,
                "vega": 0.18,
                "theta": -0.12,
                "iv": 0.20,
            },
            {
                "snapshot_id": "b1",
                "as_of_ts": datetime(2026, 5, 31, 9, 15),
                "session_date": date(2026, 5, 31),
                "symbol_key": "BANKNIFTY",
                "symbol_id": 2,
                "index_family": "BANKNIFTY",
                "underlying": "BANKNIFTY",
                "spot": 50000.0,
                "strike": 50000.0,
                "expiry": date(2026, 6, 4),
                "option_type": "PE",
                "symbol": "BANKNIFTY25JUN50000PE",
                "ltp": 208.0,
                "bid": 207.5,
                "ask": 208.5,
                "volume": 1400,
                "oi": 9000,
                "delta": -0.45,
                "gamma": 0.011,
                "vega": 0.18,
                "theta": -0.12,
                "iv": 0.21,
            },
        ]
    )

    out = build_point_in_time_features(frame)
    assert out["symbol_key"].iloc[0] == "BANKNIFTY"
    assert int(out["symbol_id"].iloc[0]) == 2
    assert out["index_family"].iloc[0] == "BANKNIFTY"
    assert out["liquidity_bucket"].iloc[0] in {"low", "medium", "high"}
