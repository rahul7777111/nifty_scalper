"""Tests for Paper Forward readiness, cost quality, LTP fallback, and XGBoost dependency handling."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_forward_readiness import (
    XGBOOST_INSTALL_HINT,
    compute_cost_quality,
    format_readiness_log,
    is_stale_last_good_mark,
    last_good_mark_allowed,
    ltp_fallback_priority,
    summarize_readiness,
)
from paper_forward_engine import (
    _LAST_GOOD_MARKS,
    _LAST_GOOD_MARK_TTL_SEC,
    _LTP_FAIL_CACHE,
    _fetch_live_broker_option_ltp,
    _last_good_mark,
    _record_last_good_mark,
    format_paper_forward_confidence,
)
from paper_forward_predict import predict_confidence_from_artifact


def _broker_row() -> dict:
    return {
        "bid": 118.0,
        "ask": 122.0,
        "volume": 45000,
        "oi": 120000,
        "ltp": 120.0,
        "chain_source": "BROKER",
    }


def test_cost_quality_real_only_with_broker_inputs() -> None:
    cq, flags = compute_cost_quality(
        row=_broker_row(),
        features={},
        used_default_fee=False,
        used_default_spread=False,
        used_chain_pctile=False,
    )
    assert cq == "REAL"
    assert flags == []


def test_cost_quality_approx_with_default_fee() -> None:
    cq, flags = compute_cost_quality(
        row=_broker_row(),
        features={},
        used_default_fee=True,
        used_default_spread=False,
        used_chain_pctile=False,
    )
    assert cq == "APPROX"
    assert "default_fee_bps" in flags


def test_cost_quality_approx_without_broker_volume_oi() -> None:
    row = {"bid": 10.0, "ask": 11.0}
    cq, flags = compute_cost_quality(row=row, features={})
    assert cq == "APPROX"
    assert flags


def test_stale_last_good_mark_rejected() -> None:
    rec = {"price": 12.5, "timestamp": time.time() - (_LAST_GOOD_MARK_TTL_SEC + 5.0)}
    assert is_stale_last_good_mark(rec, _LAST_GOOD_MARK_TTL_SEC)
    assert not last_good_mark_allowed(rec, _LAST_GOOD_MARK_TTL_SEC)


def test_fresh_last_good_mark_allowed() -> None:
    _LAST_GOOD_MARKS.clear()
    _record_last_good_mark(
        symbol="NIFTY2661624000CE",
        token="50615",
        exchange="NFO",
        price=0.10,
        source="BROKER_LTP",
    )
    px, source, stale, rec = _last_good_mark("NIFTY2661624000CE", "50615", "NFO")
    assert px == 0.10
    assert source == "BROKER_LTP"
    assert stale is False
    assert last_good_mark_allowed(rec, _LAST_GOOD_MARK_TTL_SEC)


def test_ltp_fallback_priority_order() -> None:
    assert ltp_fallback_priority() == [
        "BROKER_LTP",
        "CHAIN_LTP",
        "MID_BID_ASK",
        "LAST_GOOD_MARK",
        "SYNTHETIC",
    ]


class _SymbolFallbackClient:
    def fetch_option_quote_for_paper(self, exchange, token, symbol):
        raise RuntimeError("token quote unavailable")

    def get_ltp(self, key):
        if key in ("NFO:50615", "50615"):
            raise RuntimeError("token get_ltp unavailable")
        if key == "NFO:NIFTY2661624000CE":
            return 0.10
        raise RuntimeError(f"unexpected key {key}")


def test_ltp_fallback_token_then_symbol() -> None:
    _LTP_FAIL_CACHE.clear()
    px, source = _fetch_live_broker_option_ltp(
        _SymbolFallbackClient(),
        symbol="NIFTY2661624000CE",
        token="50615",
        exchange="NFO",
        contract={"trading_symbol": "NIFTY2661624000CE", "token": "50615", "exchange": "NFO"},
    )
    assert px == 0.10
    assert source == "BROKER_LTP"


def test_readiness_summary_counts_candidates() -> None:
    results = [
        {"no_trade_reason": "FEATURE_VECTOR_INVALID", "predict_attempted": True, "confidence": None},
        {"no_trade_reason": "XGBOOST_NOT_INSTALLED", "predict_attempted": True, "confidence": None},
        {"no_trade_reason": "low_confidence_0.2000_lt_0.5000", "predict_attempted": True, "confidence": 0.2},
        {"final_signal": "BUY_CE", "no_trade_reason": "", "predict_attempted": True, "confidence": 0.8},
        {"no_trade_reason": "", "predict_attempted": True, "confidence": 0.55},
    ]
    summary = summarize_readiness(
        candidates_total=5,
        results=results,
        cost_qualities=["REAL", "APPROX", "APPROX", "REAL", "APPROX"],
        ltp_sources=["BROKER_LTP", "CHAIN_LTP", "LAST_GOOD_MARK"],
        data_quality="DATA_OK",
    )
    assert summary["candidates_total"] == 5
    assert summary["candidates_blocked_feature"] == 1
    assert summary["candidates_blocked_dependency"] == 1
    assert summary["candidates_low_confidence"] == 1
    assert summary["candidates_signal"] == 1
    assert summary["candidates_predictable"] == 3
    assert summary["cost_real"] == 2
    assert summary["cost_approx"] == 3
    assert summary["ltp_broker_ok"] == 1
    assert summary["ltp_fallback"] == 2
    assert summary["data_quality"] == "DATA_OK"
    assert "[PF-READINESS]" in format_readiness_log(summary)


def test_tiny_nonzero_confidence_display_is_not_rounded_to_zero() -> None:
    assert format_paper_forward_confidence(9.680551105535084e-08, predict_attempted=True) == "9.68e-08"


def _mk_xgb_artifact(tmp_path: Path, features: list[str]) -> Path:
    art = tmp_path / "xgb_cand"
    art.mkdir(parents=True, exist_ok=True)
    X = np.random.randn(30, len(features))
    y = (X[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=100)
    model.fit(scaler.transform(X), y)
    with (art / "model.pkl").open("wb") as fh:
        pickle.dump({"model": model, "scaler": scaler}, fh)
    (art / "feature_schema.json").write_text(
        __import__("json").dumps({"features": features}),
        encoding="utf-8",
    )
    return art


def test_xgboost_missing_gives_xgboost_not_installed(tmp_path, monkeypatch) -> None:
    features = ["f0", "f1", "f2"]
    art = _mk_xgb_artifact(tmp_path, features)
    monkeypatch.setattr("paper_forward_predict.xgboost_available", lambda: False)
    snap = {f: 1.0 for f in features}
    res = predict_confidence_from_artifact(
        str(art / "model.pkl"),
        snap,
        feature_order=features,
        artifact_dir=str(art),
        cand_meta={"candidate_id": "xgb_cand", "model_name": "xgboost", "model_family": "xgboost"},
    )
    assert res.get("error") == "XGBOOST_NOT_INSTALLED"
    assert res.get("confidence") is None
    assert res.get("install_hint") == XGBOOST_INSTALL_HINT
