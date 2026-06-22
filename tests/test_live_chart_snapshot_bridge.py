"""
Smoke tests for Live Chart snapshot building (bridge tests).

Verifies end-to-end snapshot assembly using mock data — no broker credentials needed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import build_live_chart_snapshot, LiveChartSnapshot, Candle


class TestBuildSnapshotWithFakeCandles:
    """Verify _build_snapshot handles fake Candle objects correctly."""

    def test_build_snapshot_with_fake_candles(self) -> None:
        """100 fake Candle objects pushed to plugin, _build_snapshot called,
        verify snapshot.candles >= 100."""
        now = datetime.now()
        candles = []
        for i in range(100):
            ts = now - timedelta(minutes=99 - i)
            candles.append(
                SimpleNamespace(
                    open=25000 + i * 0.5,
                    high=25001 + i * 0.5,
                    low=24999 + i * 0.5,
                    close=25000.5 + i * 0.5,
                    volume=5000 + i * 10,
                    time=ts,
                )
            )

        snapshot = build_live_chart_snapshot(candles=candles)
        assert len(snapshot.candles) >= 100, f"Expected >=100 candles, got {len(snapshot.candles)}"
        assert len(snapshot.candles) == 100
        # Verify volumes are extracted
        assert len(snapshot.volume) == 100

    def test_build_snapshot_spot_and_atm(self) -> None:
        """spot=25000, atm_strike=25000, verify snapshot has these values."""
        snapshot = build_live_chart_snapshot(spot_price=25000.0, atm_strike=25000.0)
        assert snapshot.spot_price == 25000.0
        assert snapshot.atm_strike == 25000.0

    def test_empty_candle_input(self) -> None:
        """Empty list → snapshot.candles = [] (not crash)."""
        snapshot = build_live_chart_snapshot(candles=[])
        assert snapshot.candles == []
        assert isinstance(snapshot.candles, list)

    def test_candle_normalization_mixed_types(self) -> None:
        """Mix of time= and timestamp= Candle objects."""
        now = datetime.now()
        # Some objects use time= (like market_data.Candle)
        candles_with_time = SimpleNamespace(
            open=25000, high=25010, low=24990, close=25005, volume=5000, time=now.timestamp()
        )
        # Others might use timestamp= (like live_chart_snapshot.Candle)
        candles_with_timestamp = SimpleNamespace(
            open=25005, high=25015, low=24995, close=25008, volume=6000,
            timestamp=(now + timedelta(minutes=1)).timestamp(),
        )

        # Test with time= object
        snap1 = build_live_chart_snapshot(candles=[candles_with_time])
        assert len(snap1.candles) == 1
        assert snap1.candles[0].close == 25005.0

        # Test with timestamp= object
        snap2 = build_live_chart_snapshot(candles=[candles_with_timestamp])
        assert len(snap2.candles) == 1
        assert snap2.candles[0].close == 25008.0

        # Test mixed in same call
        snap3 = build_live_chart_snapshot(candles=[candles_with_time, candles_with_timestamp])
        assert len(snap3.candles) == 2

    def test_candle_with_simple_namespace_input(self) -> None:
        """Verify SimpleNamespace-style candle input works (normal use case)."""
        now = datetime.now()
        candles = [
            SimpleNamespace(
                open=25000, high=25010, low=24990, close=25005, volume=5000, time=now.timestamp()
            )
        ]
        snapshot = build_live_chart_snapshot(candles=candles)
        assert len(snapshot.candles) == 1
        assert snapshot.candles[0].close == 25005.0

    def test_candle_with_real_candle_objects(self) -> None:
        """Verify real Candle objects from live_chart_snapshot work."""
        now = datetime.now()
        candles = [
            Candle(timestamp=now, open=25000, high=25010, low=24990, close=25005, volume=5000),
            Candle(
                timestamp=now + timedelta(minutes=1),
                open=25005,
                high=25015,
                low=25000,
                close=25010,
                volume=6000,
            ),
        ]
        snapshot = build_live_chart_snapshot(candles=candles)
        assert len(snapshot.candles) == 2
        assert snapshot.candles[0].close == 25005.0
        assert snapshot.candles[1].close == 25010.0

    def test_snapshot_metadata_fields_present(self) -> None:
        """Verify all key metadata fields are present in snapshot."""
        snapshot = build_live_chart_snapshot(
            spot_price=25000.0,
            atm_strike=25000.0,
            vwap=25005.0,
            ema_9=24980.0,
            ema_21=24970.0,
            ema_50=24950.0,
            ema_200=24800.0,
            previous_day_high=25100.0,
            previous_day_low=24900.0,
            session_high=25050.0,
            session_low=24950.0,
            opening_range_high=25020.0,
            opening_range_low=24980.0,
        )
        assert snapshot.spot_price == 25000.0
        assert snapshot.atm_strike == 25000.0
        assert snapshot.vwap == 25005.0
        assert snapshot.ema_9 == 24980.0
        assert snapshot.ema_21 == 24970.0
        assert snapshot.ema_50 == 24950.0
        assert snapshot.ema_200 == 24800.0
        assert snapshot.previous_day_high == 25100.0
        assert snapshot.previous_day_low == 24900.0
        assert snapshot.session_high == 25050.0
        assert snapshot.session_low == 24950.0
        assert snapshot.opening_range_high == 25020.0
        assert snapshot.opening_range_low == 24980.0