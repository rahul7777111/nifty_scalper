from __future__ import annotations

import math

import numpy as np

from src.ml.ensemble_auto_router import build_ensemble_feature_frame


def _snapshot():
    return {
        "spot": 24500.0,
        "timestamp": "2026-06-18T10:00:00",
        "candles": [{"open": 24480.0, "high": 24520.0, "low": 24460.0, "close": 24500.0}],
        "option_chain": [
            {
                "option_type": "CE",
                "strike_price": 24500,
                "expiry_date": "2026-06-25",
                "last_price": 120.0,
                "bid": 119.5,
                "ask": 120.5,
                "open_interest": 10000,
                "volume": 500,
                "trading_symbol": "NIFTY24500CE",
            },
            {
                "option_type": "PE",
                "strike_price": 24500,
                "expiry_date": "2026-06-25",
                "last_price": 110.0,
                "bid": 109.5,
                "ask": 110.5,
                "open_interest": 9000,
                "volume": 450,
                "trading_symbol": "NIFTY24500PE",
            },
        ],
    }


def test_mapper_fills_required_ohlc_option_fields_from_snapshot():
    snap = dict(_snapshot()["option_chain"][0])
    snap.update({"candles": _snapshot()["candles"], "spot": 24500.0, "timestamp": "2026-06-18T10:00:00"})
    features = ["open", "high", "low", "close", "ltp", "volume", "oi", "dte_days", "is_weekly"]

    frame, diag = build_ensemble_feature_frame(snap, features, return_diagnostics=True)

    assert list(frame.columns) == features
    assert frame.shape == (1, len(features))
    assert diag["truly_missing_count"] == 0
    row = frame.iloc[0].to_dict()
    assert row["open"] == 24480.0
    assert row["high"] == 24520.0
    assert row["low"] == 24460.0
    assert row["ltp"] == 120.0
    assert row["volume"] == 500
    assert row["oi"] == 10000
    assert row["dte_days"] == 7.0


def test_aliases_last_price_open_interest_and_expiry_work():
    snap = {
        "last_price": 99.0,
        "open_interest": 1234,
        "strike_price": 24500,
        "option_type": "CE",
        "trading_symbol": "NIFTYCE",
        "expiry_date": "2026-06-25",
        "timestamp": "2026-06-18",
    }
    features = ["ltp", "oi", "strike", "ce_pe", "dte_days"]

    frame, diag = build_ensemble_feature_frame(snap, features, return_diagnostics=True)

    assert diag["truly_missing_count"] == 0
    assert diag["filled_by_alias_count"] >= 3
    assert frame.iloc[0]["ltp"] == 99.0
    assert frame.iloc[0]["oi"] == 1234
    assert frame.iloc[0]["strike"] == 24500
    assert frame.iloc[0]["ce_pe"] == 1.0


def test_safe_defaults_and_derived_percent_fields_are_filled():
    snap = {"ltp": 100.0, "bid": 99.0, "ask": 101.0, "open": 98.0, "high": 102.0, "low": 97.0, "close": 100.0}
    features = ["volume", "oi_change_pct", "iv_change_pct", "spread_pct", "range_pct", "oc_change_pct", "hl_change_pct"]

    frame, diag = build_ensemble_feature_frame(snap, features, return_diagnostics=True)

    assert diag["truly_missing_count"] == 0
    assert diag["filled_by_default_count"] >= 3
    row = frame.iloc[0]
    assert row["volume"] == 0.0
    assert row["oi_change_pct"] == 0.0
    assert row["iv_change_pct"] == 0.0
    assert row["spread_pct"] == 0.02
    assert math.isclose(row["range_pct"], 0.05)


def test_price_fields_never_become_zero_silently():
    frame, diag = build_ensemble_feature_frame({}, ["open", "high", "low", "close", "ltp"], return_diagnostics=True)

    assert diag["truly_missing_count"] == 5
    assert frame.isna().any(axis=None)


def test_final_frame_has_no_nan_or_inf_when_features_mapped():
    snap = {"ltp": 100.0, "bid": 99.0, "ask": 101.0, "open": 98.0, "high": 102.0, "low": 97.0, "close": 100.0}
    features = ["open", "high", "low", "close", "ltp", "volume", "spread_pct", "range_pct"]

    frame, diag = build_ensemble_feature_frame(snap, features, return_diagnostics=True)

    assert diag["truly_missing_count"] == 0
    arr = frame.to_numpy(dtype=float)
    assert arr.shape == (1, len(features))
    assert np.isfinite(arr).all()


def test_ce_and_pe_rows_are_built_separately():
    base = _snapshot()
    ce = dict(base["option_chain"][0])
    pe = dict(base["option_chain"][1])
    for row in (ce, pe):
        row["spot"] = base["spot"]
        row["candles"] = base["candles"]
        row["timestamp"] = base["timestamp"]
    features = ["ltp", "oi", "option_type_ce", "option_type_pe", "ce_pe"]

    ce_frame, ce_diag = build_ensemble_feature_frame(ce, features, return_diagnostics=True)
    pe_frame, pe_diag = build_ensemble_feature_frame(pe, features, return_diagnostics=True)

    assert ce_diag["truly_missing_count"] == 0
    assert pe_diag["truly_missing_count"] == 0
    assert ce_frame.iloc[0]["ltp"] == 120.0
    assert pe_frame.iloc[0]["ltp"] == 110.0
    assert ce_frame.iloc[0]["option_type_ce"] == 1.0
    assert pe_frame.iloc[0]["option_type_pe"] == 1.0
    assert ce_frame.iloc[0]["ce_pe"] == 1.0
    assert pe_frame.iloc[0]["ce_pe"] == -1.0
