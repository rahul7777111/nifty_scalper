"""Tests for LiveChartPlugin rendering — no broker needed."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

import matplotlib
matplotlib.use("Agg")

# Ensure src/ is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Synthetic candle helper
# ---------------------------------------------------------------------------

def _synthetic_candles(n: int, base_price: float = 24500.0) -> list:
    """Return n candle-like SimpleNamespace objects with O/H/L/C and a unix timestamp."""
    import time
    now = int(time.time())
    candles = []
    price = float(base_price)
    for i in range(n):
        offset = (i - n // 2) * 60  # minutes centred around now
        open_ = price + (i * 5.0)
        close_ = open_ + ((i % 3) - 1) * 3.0
        high_ = max(open_, close_) + 2.0
        low_ = min(open_, close_) - 2.0
        candles.append(
            SimpleNamespace(
                open=open_,
                high=high_,
                low=low_,
                close=close_,
                time=now + offset,
            )
        )
        price = close_
    return candles


# ---------------------------------------------------------------------------
# Tkinter + LiveChartPlugin fixture
# ---------------------------------------------------------------------------

try:
    import tkinter as tk  # noqa: E402
    _TKINTER_AVAILABLE = True
except Exception:
    tk = None  # type: ignore[assignment]
    _TKINTER_AVAILABLE = False


def _make_chart():
    """Create a hidden Tk root + frame and return (root, frame).

    Skips the running test if tkinter is not available.
    """
    if not _TKINTER_AVAILABLE:
        pytest.skip("tkinter not available in this environment")
    try:
        root = tk.Tk()
    except Exception as exc:
        pytest.skip(f"tkinter runtime unavailable: {exc}")
    root.withdraw()
    frame = tk.Frame(root)
    return root, frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chart_initializes_with_empty_candles() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame, symbol="NIFTY", timeframe="1m")
        # Canvas must be attached
        assert plugin.canvas is not None
        # Empty candles -> _render() called silently with no crash
        plugin.push_candles([])
        plugin.push_candles(None)
    finally:
        root.destroy()


def test_chart_rejects_bad_nanosecond_overlay_timestamp() -> None:
    from chart import _normalize_epoch_seconds

    assert _normalize_epoch_seconds(725247989284439860618272602857668608) is None


def test_chart_push_candles_renders_bodies() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        candles = _synthetic_candles(10)
        plugin.push_candles(candles)
        # Wick collection must have segments (one per candle)
        assert len(plugin._wick_collection.get_segments()) >= 10
        # Body collection must have polygons
        assert len(plugin._body_collection.get_verts()) >= 10
    finally:
        root.destroy()


def test_chart_ema_lines_drawn() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        # 30 candles: enough for EMA(9) to produce values
        candles = _synthetic_candles(30)
        plugin.push_candles(candles)
        ema_x, ema_y = plugin._ema_fast_line.get_data()
        assert len(ema_x) > 0, "EMA fast line should have data after 30 candles"
        assert len(ema_y) > 0
        # Second EMA should also have data
        ema2_x, ema2_y = plugin._ema_slow_line.get_data()
        assert len(ema2_x) > 0, "EMA slow line should have data after 30 candles"
    finally:
        root.destroy()


def test_chart_supertrend_lines_drawn() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        candles = _synthetic_candles(20)  # period=10 needs 11+ candles
        plugin.push_candles(candles)
        st_x, st_y = plugin._supertrend_line.get_data()
        # Supertrend period=10, need at least 11 candles for valid values
        assert len(st_x) > 0, "Supertrend line should have data after 20 candles"
        assert len(st_y) > 0
    finally:
        root.destroy()


def test_chart_renders_without_crash_multiple_pushes() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        for _ in range(5):
            plugin.push_candles(_synthetic_candles(20))
    finally:
        root.destroy()


def test_lock_to_live_mode() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        assert plugin._lock_to_live is False
        plugin.set_lock_to_live(True)
        assert plugin._lock_to_live is True
        # Push new candles after locking — should not raise
        plugin.push_candles(_synthetic_candles(15))
        plugin.set_lock_to_live(False)
        plugin.push_candles(_synthetic_candles(15))
    finally:
        root.destroy()


def test_overlay_visibility_toggle() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        overlays = ["ema_fast", "ema_slow", "supertrend", "rsi"]
        for name in overlays:
            # Verify method exists
            assert hasattr(plugin, "set_overlay_visible"), "set_overlay_visible must exist"
            # Toggle off then on
            plugin.set_overlay_visible(name, False)
            plugin.set_overlay_visible(name, True)
            # Verify no error
        # Also verify the internal dict was updated
        assert plugin._overlay_visible.get("ema_fast") is True
        assert plugin._overlay_visible.get("supertrend") is True
    finally:
        root.destroy()


def test_export_png_creates_file(tmp_path) -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(5))
        out = tmp_path / "chart.png"
        plugin.export_png(str(out))
        assert out.exists(), "export_png should create a file"
        assert out.stat().st_size > 0, "exported PNG should be non-empty"
    finally:
        root.destroy()


def test_signal_filter_modes() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        for mode in ("all", "executed", "shadow", "blocked"):
            plugin.set_signal_filter(mode)
            assert plugin._signal_filter == mode
    finally:
        root.destroy()


def test_market_info_setter() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        plugin.set_market_info(symbol="BANKNIFTY", expiry="2026-06-26", lot_size=25)
        assert plugin._market_info["symbol"] == "BANKNIFTY"
        assert plugin._market_info["expiry"] == "2026-06-26"
        assert plugin._market_info["lot_size"] == 25
        # Extra kwargs preserved
        plugin.set_market_info(spot=45000.0)
        assert plugin._market_info["spot"] == 45000.0
    finally:
        root.destroy()


def test_prevday_session_levels() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        plugin.set_prevday_levels(high=24600.0, low=24400.0, open_price=24500.0)
        assert plugin._prevday_levels is not None
        assert plugin._prevday_levels["high"] == 24600.0
        assert plugin._prevday_levels["low"] == 24400.0
        assert plugin._prevday_levels["open"] == 24500.0

        plugin.set_session_levels(high=24550.0, low=24450.0, open_price=24500.0)
        assert plugin._session_levels is not None
        assert plugin._session_levels["high"] == 24550.0
    finally:
        root.destroy()


def test_atm_strike_marker() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        assert plugin._atm_strike is None
        plugin.set_atm_strike(24500.0)
        assert plugin._atm_strike == 24500.0
        # Push candles so render runs with ATM set
        plugin.push_candles(_synthetic_candles(5))
        plugin.set_atm_strike(None)
        assert plugin._atm_strike is None
    finally:
        root.destroy()


def test_load_prediction_rows_populates_markers() -> None:
    root, frame = _make_chart()
    try:
        import time
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "probability": 0.75}
            for i in range(3)
        ] + [
            {"timestamp": now + (i + 3) * 60, "predicted_class": 0, "probability": 0.65}
            for i in range(2)
        ]
        plugin.load_prediction_rows(rows)
        # Should not crash and _prediction_rows should be stored
        assert len(plugin._prediction_rows) == 5
    finally:
        root.destroy()


def test_focus_trade_by_id() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(20))

        trade_rows = [
            {"event": "OPEN", "ts": 1200.0, "trade_id": "T1", "name": "CE trade 1"},
            {"event": "CLOSE", "ts": 1300.0, "trade_id": "T1", "name": "CE trade 1"},
            {"event": "OPEN", "ts": 1400.0, "trade_id": "T2", "name": "PE trade 2"},
            {"event": "CLOSE", "ts": 1500.0, "trade_id": "T2", "name": "PE trade 2"},
            {"event": "OPEN", "ts": 1600.0, "trade_id": "T3", "name": "CE trade 3"},
        ]
        plugin.load_trade_event_rows(trade_rows)

        # Focus on T2 specifically — should not raise
        plugin.focus_trade("T2")
        assert plugin._focus_trade_id == "T2"

        # Focus on non-existent ID — should not raise
        plugin.focus_trade("NONEXISTENT")

        # Clear focus
        plugin.focus_trade(None)
        assert plugin._focus_trade_id is None
    finally:
        root.destroy()


def test_no_lookahead_in_ema() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin
        plugin = LiveChartPlugin(frame, ema_fast=9)

        # Push 20 candles
        candles = _synthetic_candles(20, base_price=24500.0)
        plugin.push_candles(candles)

        # Inspect the EMA indicator cache
        cache_key = "ema_9"
        assert cache_key in plugin._indicator_cache, "EMA cache should contain ema_9 key"

        cached = plugin._indicator_cache[cache_key]
        source = cached.get("source", [])
        series = cached.get("series", [])

        # Source list should have exactly 20 close values (no more, no less)
        assert len(source) == 20, f"EMA source should have exactly 20 values, got {len(source)}"

        # Series should have 20 entries (None for bars < period-1, values after)
        assert len(series) == 20, f"EMA series should have exactly 20 entries, got {len(series)}"

        # Verify no-bar look-ahead: all cached source values come from the closes of pushed candles
        closes_from_candles = [float(c.close) for c in candles]
        assert list(source) == closes_from_candles, "EMA cache source must exactly match candle closes"

        # Values before index 8 (period-1) should be None
        for i in range(8):
            assert series[i] is None, f"EMA at index {i} should be None (not enough history)"

        # Values from index 8 onwards should be real numbers
        for i in range(8, 20):
            assert series[i] is not None, f"EMA at index {i} should be a real value"
            assert isinstance(series[i], float), f"EMA at index {i} should be a float"
    finally:
        root.destroy()
