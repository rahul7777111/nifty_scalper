"""Tests for snapshot builder — verifies LiveChartSnapshot assembly."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import build_live_chart_snapshot, SignalMarker


class TestSnapshotBuilderCandles:
    def test_candles_parsed_from_namespace_objects(self) -> None:
        now = datetime.now()
        candles = [
            SimpleNamespace(open=100, high=105, low=99, close=103, volume=5000, time=now.timestamp()),
            SimpleNamespace(open=103, high=107, low=102, close=105, volume=6000, time=(now.timestamp() + 60)),
        ]
        snap = build_live_chart_snapshot(candles=candles)
        assert len(snap.candles) == 2
        assert snap.candles[0].close == 103.0
        assert snap.candles[1].close == 105.0
        assert snap.volume == [5000.0, 6000.0]

    def test_empty_candles_gives_empty_list(self) -> None:
        snap = build_live_chart_snapshot(candles=[])
        assert snap.candles == []
        assert snap.volume == []

    def test_candles_with_missing_time_skipped(self) -> None:
        now = datetime.now()
        candles = [
            SimpleNamespace(open=100, high=105, low=99, close=103, volume=5000, time=now.timestamp()),
            SimpleNamespace(open=103, high=107, low=102, close=105, volume=6000, time=None),
        ]
        snap = build_live_chart_snapshot(candles=candles)
        assert len(snap.candles) == 1


class TestSnapshotBuilderIndicators:
    def test_ema_values_passed_through(self) -> None:
        snap = build_live_chart_snapshot(ema_9=24510.5, ema_21=24480.2, ema_50=24400.0, ema_200=24000.0)
        assert snap.ema_9 == 24510.5
        assert snap.ema_21 == 24480.2
        assert snap.ema_50 == 24400.0
        assert snap.ema_200 == 24000.0

    def test_vwap_passed_through(self) -> None:
        snap = build_live_chart_snapshot(vwap=24525.0)
        assert snap.vwap == 24525.0

    def test_session_and_prevday_levels(self) -> None:
        snap = build_live_chart_snapshot(
            session_high=24600.0, session_low=24400.0,
            previous_day_high=24700.0, previous_day_low=24300.0,
        )
        assert snap.session_high == 24600.0
        assert snap.session_low == 24400.0
        assert snap.previous_day_high == 24700.0
        assert snap.previous_day_low == 24300.0


class TestSnapshotBuilderLatestPrediction:
    def test_latest_prediction_sets_current_signal(self) -> None:
        now = datetime.now()
        pred = SignalMarker(timestamp=now, price=24500.0, side="SELL", instrument="PE",
                            probability=0.65, confidence=0.6)
        snap = build_live_chart_snapshot(latest_prediction=pred)
        assert snap.current_signal == "SELL"

    def test_no_prediction_gives_none_signal(self) -> None:
        snap = build_live_chart_snapshot()
        assert snap.current_signal is None


class TestSnapshotBuilderStatus:
    def test_broker_connection_status(self) -> None:
        snap = build_live_chart_snapshot(broker_connection_status="CONNECTED")
        assert snap.broker_connection_status == "CONNECTED"

    def test_model_status(self) -> None:
        snap = build_live_chart_snapshot(model_status="LOADED")
        assert snap.model_status == "LOADED"

    def test_shadow_mode_active_defaults_true(self) -> None:
        snap = build_live_chart_snapshot()
        assert snap.shadow_mode_active is True

    def test_regime_label(self) -> None:
        snap = build_live_chart_snapshot(regime_label="TREND")
        assert snap.regime_label == "TREND"


class TestSnapshotBuilderDataHealth:
    def test_data_health_filled(self) -> None:
        snap = build_live_chart_snapshot(
            last_candle_ts=datetime.now().timestamp() - 10,
            broker_connected=True,
            model_loaded=True,
            shadow_logger_ok=True,
            db_write_ok=True,
        )
        assert snap.data_health.broker_connected is True
        assert snap.data_health.model_loaded is True
        assert snap.data_health.status == "OK"


class TestSnapshotBuilderOptionChain:
    def test_option_chain_summary_from_payload(self) -> None:
        payload = {
            "chain": [
                {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 199.0, "ask": 201.0, "iv": 18.0, "oi": 50000},
                {"option_type": "PE", "strike": 24500.0, "ltp": 190.0, "bid": 189.0, "ask": 191.0, "iv": 17.5, "oi": 55000},
            ],
        }
        snap = build_live_chart_snapshot(spot_price=24500.0, option_chain_payload=payload)
        assert snap.option_chain_summary.atm_strike == 24500.0
        assert snap.option_chain_summary.ce_ltp == 200.0
        assert snap.option_chain_summary.pcr is not None


class TestSnapshotBuilderShadowMetrics:
    def test_shadow_metrics_computed_from_rows(self) -> None:
        now_ts = datetime.now().timestamp()
        pred_rows = [
            {"ts": now_ts, "probability": 0.72, "trade_taken": True},
            {"ts": now_ts, "probability": 0.42, "trade_taken": False},
        ]
        snap = build_live_chart_snapshot(shadow_rows=pred_rows)
        assert snap.shadow_metrics.predictions_today == 2
        assert snap.shadow_metrics.accepted_signals == 1
        assert snap.shadow_metrics.blocked_signals == 1