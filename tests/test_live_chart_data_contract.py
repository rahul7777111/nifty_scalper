"""Tests for live_chart_snapshot data contract — no broker needed."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import (
    Candle,
    SignalMarker,
    ShadowMetrics,
    OptionChainSummary,
    DataHealth,
    LiveChartSnapshot,
    build_live_chart_snapshot,
    option_chain_to_summary,
    compute_shadow_metrics,
    assess_data_health,
)


class TestCandleDataclass:
    def test_candle_requires_all_fields(self) -> None:
        now = datetime.now()
        c = Candle(timestamp=now, open=100.0, high=105.0, low=99.0, close=103.0, volume=5000.0)
        assert c.open == 100.0
        assert c.high == 105.0
        assert c.close == 103.0
        assert c.volume == 5000.0

    def test_candle_timestamp_is_datetime(self) -> None:
        now = datetime.now()
        c = Candle(timestamp=now, open=100.0, high=105.0, low=99.0, close=103.0, volume=5000.0)
        assert isinstance(c.timestamp, datetime)


class TestSignalMarkerDataclass:
    def test_signal_marker_defaults(self) -> None:
        now = datetime.now()
        sm = SignalMarker(timestamp=now, price=24500.0, side="BUY", instrument="CE")
        assert sm.side == "BUY"
        assert sm.instrument == "CE"
        assert sm.probability == 0.0
        assert sm.confidence == 0.0
        assert sm.status == "shadow"

    def test_signal_marker_full_fields(self) -> None:
        now = datetime.now()
        sm = SignalMarker(
            timestamp=now, price=24500.0, side="SELL", instrument="PE",
            strike=24450.0, probability=0.72, confidence=0.88,
            model_name="xgb_v3", threshold=0.5, expected_edge=0.05,
            reason="IV expansion", status="executed", trade_id="T123", pnl=150.0,
        )
        assert sm.probability == 0.72
        assert sm.expected_edge == 0.05
        assert sm.pnl == 150.0


class TestShadowMetricsDataclass:
    def test_shadow_metrics_defaults(self) -> None:
        m = ShadowMetrics()
        assert m.predictions_today == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert m.drift_warning is False

    def test_shadow_metrics_all_fields(self) -> None:
        m = ShadowMetrics(
            predictions_today=50, accepted_signals=12, blocked_signals=38,
            resolved_trades=10, pending_trades=2, win_rate=60.0,
            avg_win=200.0, avg_loss=100.0, profit_factor=2.0,
            expectancy=1.5, max_drawdown=-50.0, daily_pnl=1200.0,
            avg_holding_minutes=8.5, calibration_error=0.05,
            drift_warning=True, last_prediction_time=datetime.now(),
        )
        assert m.predictions_today == 50
        assert m.drift_warning is True
        assert m.avg_holding_minutes == 8.5


class TestOptionChainSummaryDataclass:
    def test_option_chain_summary_defaults(self) -> None:
        oc = OptionChainSummary()
        assert oc.atm_strike is None
        assert oc.liquidity_score == "N/A"
        assert oc.is_stale is False

    def test_option_chain_summary_full(self) -> None:
        oc = OptionChainSummary(
            atm_strike=24500.0, ce_ltp=250.0, pe_ltp=240.0,
            ce_iv=18.5, pe_iv=19.2, ce_oi=50000, pe_oi=55000,
            pcr=1.1, ce_bid_ask_spread=3.5, pe_bid_ask_spread=3.2,
            liquidity_score="WARNING", is_stale=False,
            wide_spread_warning=True, iv_spike_warning=False,
        )
        assert oc.pcr == 1.1
        assert oc.liquidity_score == "WARNING"
        assert oc.wide_spread_warning is True


class TestDataHealthDataclass:
    def test_data_health_defaults(self) -> None:
        dh = DataHealth()
        assert dh.stale_data_seconds == -1
        assert dh.status == "UNKNOWN"
        assert dh.broker_connected is False

    def test_data_health_full(self) -> None:
        dh = DataHealth(
            last_candle_time=datetime.now(),
            last_tick_time=datetime.now(),
            stale_data_seconds=15,
            missing_candles=0,
            broker_connected=True,
            model_loaded=True,
            shadow_logger_ok=True,
            db_write_ok=True,
            api_error_count_today=2,
            status="OK",
            status_reason="all systems nominal",
        )
        assert dh.status == "OK"
        assert dh.broker_connected is True


class TestOptionChainToSummary:
    def test_none_payload_returns_defaults(self) -> None:
        oc = option_chain_to_summary(None, None)
        assert oc.atm_strike is None
        assert oc.ce_ltp is None
        assert oc.liquidity_score == "N/A"

    def test_payload_with_spot_sets_atm(self) -> None:
        oc = option_chain_to_summary({}, spot=24517.0)
        assert oc.atm_strike == 24500.0

    def test_wide_spread_triggers_warning(self) -> None:
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 195.0, "ask": 205.0, "iv": 15.0, "oi": 10000},
            {"option_type": "PE", "strike": 24500.0, "ltp": 190.0, "bid": 185.0, "ask": 195.0, "iv": 15.0, "oi": 10000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.ce_bid_ask_spread == 10.0
        assert oc.wide_spread_warning is True
        assert oc.liquidity_score == "WARNING"

    def test_iv_spike_triggers_warning(self) -> None:
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 199.0, "ask": 201.0, "iv": 35.0, "oi": 10000},
            {"option_type": "PE", "strike": 24500.0, "ltp": 190.0, "bid": 189.0, "ask": 191.0, "iv": 35.0, "oi": 10000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.iv_spike_warning is True
        assert oc.ce_iv == 35.0


class TestComputeShadowMetrics:
    def test_empty_rows_returns_defaults(self) -> None:
        m = compute_shadow_metrics(None, None)
        assert m.predictions_today == 0
        assert m.win_rate == 0.0

    def test_predictions_today_counted(self) -> None:
        now_ts = datetime.now().timestamp()
        rows = [
            {"ts": now_ts, "probability": 0.6, "trade_taken": True},
            {"ts": now_ts, "probability": 0.4, "trade_taken": False},
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.predictions_today == 2
        assert m.accepted_signals == 1
        assert m.blocked_signals == 1


class TestAssessDataHealth:
    def test_connected_loaded_ok(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "OK"

    def test_disconnected_is_critical(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=False, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "CRITICAL"
        assert "disconnected" in dh.status_reason

    def test_stale_120s_is_critical(self) -> None:
        old_ts = datetime.now().timestamp() - 130
        dh = assess_data_health(
            last_candle_ts=old_ts,
            last_tick_ts=old_ts,
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "CRITICAL"


class TestBuildLiveChartSnapshot:
    def test_minimal_snapshot_builds(self) -> None:
        snap = build_live_chart_snapshot()
        assert snap.market_status == "CLOSED"
        assert snap.shadow_mode_active is True
        assert isinstance(snap.option_chain_summary, OptionChainSummary)
        assert isinstance(snap.shadow_metrics, ShadowMetrics)
        assert isinstance(snap.data_health, DataHealth)

    def test_spot_and_atm_set(self) -> None:
        snap = build_live_chart_snapshot(spot_price=24517.0)
        assert snap.spot_price == 24517.0

    def test_latest_prediction_set(self) -> None:
        now = datetime.now()
        pred = SignalMarker(timestamp=now, price=24500.0, side="BUY", instrument="CE",
                            probability=0.72, confidence=0.88, model_name="xgb")
        snap = build_live_chart_snapshot(latest_prediction=pred)
        assert snap.latest_prediction is not None
        assert snap.latest_prediction.side == "BUY"
        assert snap.current_signal == "BUY"

    def test_market_status_passthrough(self) -> None:
        snap = build_live_chart_snapshot(market_status="OPEN")
        assert snap.market_status == "OPEN"

    def test_shadow_mode_active_default_true(self) -> None:
        snap = build_live_chart_snapshot()
        assert snap.shadow_mode_active is True