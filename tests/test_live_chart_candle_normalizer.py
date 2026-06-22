"""Tests for normalize_live_chart_candles()."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import (
    Candle,
    normalize_live_chart_candles,
)


class TestNormalizeObjectWithTime:
    """Object with time= (market_data style)."""

    def test_single_market_data_style_candle(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        obj = SimpleNamespace(
            time=now,
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=5000.0,
        )
        result = normalize_live_chart_candles([obj])
        assert len(result) == 1
        assert result[0].timestamp == now
        assert result[0].open == 100.0
        assert result[0].high == 105.0
        assert result[0].low == 99.0
        assert result[0].close == 103.0
        assert result[0].volume == 5000.0

    def test_multiple_market_data_style_candles(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        candles = [
            SimpleNamespace(time=now, open=100.0, high=105.0, low=99.0, close=103.0, volume=5000.0),
            SimpleNamespace(time=now, open=103.0, high=107.0, low=102.0, close=105.0, volume=6000.0),
        ]
        result = normalize_live_chart_candles(candles)
        assert len(result) == 2
        assert result[0].close == 103.0
        assert result[1].close == 105.0


class TestNormalizeObjectWithTimestamp:
    """Object with timestamp= (snapshot style)."""

    def test_single_snapshot_style_candle(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        obj = SimpleNamespace(
            timestamp=now,
            open=200.0,
            high=210.0,
            low=195.0,
            close=205.0,
            volume=8000.0,
        )
        result = normalize_live_chart_candles([obj])
        assert len(result) == 1
        assert result[0].timestamp == now
        assert result[0].open == 200.0
        assert result[0].high == 210.0
        assert result[0].low == 195.0
        assert result[0].close == 205.0
        assert result[0].volume == 8000.0

    def test_timestamp_preferred_over_time_if_both_present(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        earlier = datetime(2020, 1, 1)
        obj = SimpleNamespace(
            time=earlier,
            timestamp=now,
            open=200.0,
            high=210.0,
            low=195.0,
            close=205.0,
            volume=8000.0,
        )
        result = normalize_live_chart_candles([obj])
        assert len(result) == 1
        assert result[0].timestamp == now  # timestamp preferred


class TestNormalizeDictWithTime:
    """Dict with time key."""

    def test_single_dict_with_time(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        d = {
            "time": now,
            "open": 300.0,
            "high": 310.0,
            "low": 295.0,
            "close": 305.0,
            "volume": 10000.0,
        }
        result = normalize_live_chart_candles([d])
        assert len(result) == 1
        assert result[0].timestamp == now
        assert result[0].close == 305.0
        assert result[0].volume == 10000.0


class TestNormalizeDictWithTimestamp:
    """Dict with timestamp key."""

    def test_single_dict_with_timestamp(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        d = {
            "timestamp": now,
            "open": 400.0,
            "high": 415.0,
            "low": 395.0,
            "close": 410.0,
            "volume": 12000.0,
        }
        result = normalize_live_chart_candles([d])
        assert len(result) == 1
        assert result[0].timestamp == now
        assert result[0].close == 410.0

    def test_dict_with_ts_key(self, caplog: pytest.LogCaptureFixture) -> None:
        ts = datetime.now().timestamp()
        d = {
            "ts": ts,
            "open": 500.0,
            "high": 510.0,
            "low": 490.0,
            "close": 505.0,
            "volume": 15000.0,
        }
        result = normalize_live_chart_candles([d])
        assert len(result) == 1
        assert abs(result[0].timestamp.timestamp() - ts) < 1.0
        assert result[0].close == 505.0

    def test_dict_with_date_key(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        d = {
            "date": now,
            "open": 600.0,
            "high": 620.0,
            "low": 590.0,
            "close": 610.0,
            "volume": 20000.0,
        }
        result = normalize_live_chart_candles([d])
        assert len(result) == 1
        assert result[0].timestamp == now


class TestNormalizeBrokerAPIRows:
    """List/tuple from broker API (6 elements: time, open, high, low, close, volume)."""

    def test_single_list_row(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        row = [now, 700.0, 720.0, 690.0, 710.0, 25000.0]
        result = normalize_live_chart_candles([row])
        assert len(result) == 1
        assert result[0].timestamp == now
        assert result[0].open == 700.0
        assert result[0].high == 720.0
        assert result[0].low == 690.0
        assert result[0].close == 710.0
        assert result[0].volume == 25000.0

    def test_tuple_row(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        row = (now, 800.0, 825.0, 795.0, 815.0, 30000.0)
        result = normalize_live_chart_candles([row])
        assert len(result) == 1
        assert result[0].close == 815.0

    def test_multiple_rows(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        rows = [
            [now, 900.0, 920.0, 890.0, 910.0, 35000.0],
            [now, 910.0, 930.0, 905.0, 925.0, 40000.0],
        ]
        result = normalize_live_chart_candles(rows)
        assert len(result) == 2
        assert result[0].close == 910.0
        assert result[1].close == 925.0

    def test_numeric_timestamp_in_list(self, caplog: pytest.LogCaptureFixture) -> None:
        ts = datetime.now().timestamp()
        row = [ts, 1000.0, 1020.0, 990.0, 1010.0, 50000.0]
        result = normalize_live_chart_candles([row])
        assert len(result) == 1
        assert abs(result[0].timestamp.timestamp() - ts) < 1.0


class TestNormalizeEmptyInput:
    """Empty input → returns []."""

    def test_empty_list(self, caplog: pytest.LogCaptureFixture) -> None:
        result = normalize_live_chart_candles([])
        assert result == []

    def test_none_input(self, caplog: pytest.LogCaptureFixture) -> None:
        result = normalize_live_chart_candles(None)
        assert result == []

    def test_single_none_in_list(self, caplog: pytest.LogCaptureFixture) -> None:
        result = normalize_live_chart_candles([None])
        assert result == []


class TestNormalizeLogging:
    """Logging of rejected rows."""

    def test_rejected_candles_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        now = datetime.now()
        rows = [
            {"time": now, "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 5000.0},
            {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 5000.0},  # missing time
        ]
        result = normalize_live_chart_candles(rows)
        assert len(result) == 1
        assert "rejected 1/2 rows" in caplog.text

    def test_no_warning_on_full_success(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        now = datetime.now()
        rows = [
            {"time": now, "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 5000.0},
            {"time": now, "open": 103.0, "high": 107.0, "low": 102.0, "close": 105.0, "volume": 6000.0},
        ]
        result = normalize_live_chart_candles(rows)
        assert len(result) == 2
        # No rejection warning should appear
        assert "rejected" not in caplog.text.lower()


class TestNormalizeDoesNotSilentlyReturnEmpty:
    """Does NOT silently return empty when input has data."""

    def test_returns_candles_not_empty_on_valid_input(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        rows = [
            {"time": now, "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 5000.0},
        ]
        result = normalize_live_chart_candles(rows)
        assert result != []
        assert len(result) == 1


class TestNormalizeMixedInputTypes:
    """Mixed input types in same list."""

    def test_mixed_object_and_dict(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        rows = [
            SimpleNamespace(time=now, open=100.0, high=105.0, low=99.0, close=103.0, volume=5000.0),
            {"time": now, "open": 200.0, "high": 210.0, "low": 195.0, "close": 205.0, "volume": 8000.0},
        ]
        result = normalize_live_chart_candles(rows)
        assert len(result) == 2
        assert result[0].open == 100.0
        assert result[1].open == 200.0


class TestNormalizeRealDataclasses:
    """Real Candle dataclasses from live_chart_snapshot and market_data."""

    def test_live_chart_snapshot_candle(self, caplog: pytest.LogCaptureFixture) -> None:
        now = datetime.now()
        candle = Candle(timestamp=now, open=100.0, high=105.0, low=99.0, close=103.0, volume=5000.0)
        result = normalize_live_chart_candles([candle])
        assert len(result) == 1
        assert result[0].timestamp == now
        assert result[0].close == 103.0
