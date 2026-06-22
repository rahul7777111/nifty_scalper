"""Tests for data health assessment — no broker needed."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import assess_data_health


class TestDataHealthStatusOK:
    def test_all_ok_gives_ok_status(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "OK"
        assert dh.stale_data_seconds < 30

    def test_connected_and_recent(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp() - 10,
            last_tick_ts=datetime.now().timestamp() - 5,
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "OK"


class TestDataHealthStatusWarning:
    def test_moderate_staleness_gives_warning(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp() - 60,
            last_tick_ts=datetime.now().timestamp() - 60,
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "WARNING"

    def test_missing_candles_gives_warning(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=5, api_errors_today=0,
        )
        assert dh.status == "WARNING"
        assert "missing candles" in dh.status_reason

    def test_api_errors_gives_warning(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=15,
        )
        assert dh.status == "WARNING"


class TestDataHealthStatusCritical:
    def test_disconnected_is_critical(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=False, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "CRITICAL"

    def test_model_not_loaded_is_critical(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=True, model_loaded=False,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "CRITICAL"

    def test_stale_120s_is_critical(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp() - 130,
            last_tick_ts=datetime.now().timestamp() - 130,
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "CRITICAL"

    def test_multiple_critical_factors(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp() - 200,
            last_tick_ts=datetime.now().timestamp() - 200,
            broker_connected=False, model_loaded=False,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=10, api_errors_today=20,
        )
        assert dh.status == "CRITICAL"


class TestDataHealthFields:
    def test_last_candle_time_set(self) -> None:
        now = datetime.now()
        dh = assess_data_health(
            last_candle_ts=now.timestamp(),
            last_tick_ts=now.timestamp(),
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.last_candle_time is not None
        assert dh.stale_data_seconds >= 0

    def test_missing_candles_count_passthrough(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=7, api_errors_today=0,
        )
        assert dh.missing_candles == 7

    def test_api_error_count_passthrough(self) -> None:
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=True, model_loaded=True,
            shadow_logger_ok=True, db_write_ok=True,
            missing_candles=0, api_errors_today=12,
        )
        assert dh.api_error_count_today == 12