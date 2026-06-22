"""Tests for signal marker creation from DB rows."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import SignalMarker


def _row_to_signal_marker(row, spot=None):
    """Local reimplementation of the panel's helper for testing."""
    from live_chart_snapshot import SignalMarker
    from datetime import datetime as dt
    try:
        ts_val = row.get("ts") or row.get("timestamp")
        if ts_val is None:
            return None
        ts_dt = dt.fromtimestamp(float(ts_val)) if isinstance(ts_val, (int, float)) else dt.now()
        prob = float(row.get("probability", 0))
        pred_class = int(row.get("predicted_class", -1))
        trade_candidate = str(row.get("trade_candidate", "")).upper()
        trade_taken = row.get("trade_taken") in (True, 1, "1", "true", "True")

        if pred_class == 1 or "LONG" in trade_candidate or "BUY" in trade_candidate:
            side, instrument = "BUY", "CE"
        elif pred_class == 0 or "SHORT" in trade_candidate or "SELL" in trade_candidate:
            side, instrument = "SELL", "PE"
        else:
            side, instrument = "EXIT", ""

        confidence = min(abs(prob - 0.5) * 2, 1.0)
        return SignalMarker(
            timestamp=ts_dt, price=spot or 0.0, side=side, instrument=instrument,
            strike=row.get("strike"), probability=prob, confidence=confidence,
            model_name=str(row.get("model_name", "N/A")), threshold=0.5,
            expected_edge=row.get("expected_edge"),
            reason=str(row.get("reason", "") or row.get("regime", "")),
            status="executed" if trade_taken else "shadow",
            trade_id=str(row.get("trade_id", "") or ""),
        )
    except Exception:
        return None


class TestSignalMarkerFromDBRow:
    def test_buy_signal_from_predicted_class_1(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 1, "probability": 0.72}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.side == "BUY"
        assert sm.probability == 0.72

    def test_sell_signal_from_predicted_class_0(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 0, "probability": 0.65}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.side == "SELL"

    def test_buy_from_trade_candidate_long(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": -1,
               "trade_candidate": "LONG", "probability": 0.68}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.side == "BUY"

    def test_sell_from_trade_candidate_short(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": -1,
               "trade_candidate": "SHORT", "probability": 0.60}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.side == "SELL"

    def test_exit_when_class_unknown(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": -1,
               "trade_candidate": "UNKNOWN", "probability": 0.50}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.side == "EXIT"

    def test_shadow_status_when_trade_not_taken(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 1,
               "probability": 0.72, "trade_taken": False}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.status == "shadow"

    def test_executed_status_when_trade_taken(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 1,
               "probability": 0.72, "trade_taken": True}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.status == "executed"

    def test_confidence_scaled_0_to_1(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 1, "probability": 0.90}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert 0.0 <= sm.confidence <= 1.0
        assert sm.confidence > 0.5  # prob 0.9 -> conf = min(0.8, 1.0) = 0.8

    def test_missing_ts_returns_none(self) -> None:
        row = {"predicted_class": 1, "probability": 0.72}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is None

    def test_instrument_ce_for_buy(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 1, "probability": 0.72}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.instrument == "CE"

    def test_instrument_pe_for_sell(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 0, "probability": 0.65}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.instrument == "PE"

    def test_model_name_preserved(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 1,
               "probability": 0.72, "model_name": "xgb_v3_edge"}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert sm.model_name == "xgb_v3_edge"

    def test_reason_preserved_from_regime(self) -> None:
        row = {"ts": datetime.now().timestamp(), "predicted_class": 1,
               "probability": 0.72, "regime": "TREND"}
        sm = _row_to_signal_marker(row, 24500.0)
        assert sm is not None
        assert "TREND" in sm.reason