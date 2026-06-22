"""Tests for Live Chart fail-safe behavior — no broker needed."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

# Use Agg to avoid needing a display
import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    import tkinter as tk
    _TKINTER_AVAILABLE = True
except Exception:
    tk = None
    _TKINTER_AVAILABLE = False


_ROOT = None

def _make_chart():
    global _ROOT
    if not _TKINTER_AVAILABLE:
        pytest.skip("tkinter not available")
    if _ROOT is None:
        try:
            _ROOT = tk.Tk()
            _ROOT.withdraw()
        except Exception as e:
            pytest.skip(f"tkinter failed to initialize: {e}")
    frame = tk.Frame(_ROOT)
    return _ROOT, frame


class TestChartHandlesEmptyCandles:
    def test_push_empty_list_no_crash(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            plugin.push_candles([])
        finally:
            frame.destroy()

    def test_push_none_no_crash(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            plugin.push_candles(None)
        finally:
            frame.destroy()

    def test_render_with_empty_candles_clears_artists(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            plugin.push_candles([])
            # Should clear all artists without crash
            assert plugin._dirty is False
        finally:
            frame.destroy()


class TestChartHandlesMissingOptionChain:
    def test_build_snapshot_with_no_option_chain(self) -> None:
        from live_chart_snapshot import build_live_chart_snapshot
        snap = build_live_chart_snapshot(
            spot_price=24500.0,
            option_chain_payload=None,
        )
        assert snap.option_chain_summary.liquidity_score == "N/A"
        assert snap.option_chain_summary.ce_ltp is None

    def test_refresh_with_none_option_chain(self) -> None:
        from live_chart_snapshot import option_chain_to_summary
        oc = option_chain_to_summary(None, None)
        assert oc.liquidity_score == "N/A"


class TestChartHandlesMissingModel:
    def test_build_snapshot_model_not_loaded(self) -> None:
        from live_chart_snapshot import build_live_chart_snapshot
        snap = build_live_chart_snapshot(model_status="NOT_LOADED", broker_connection_status="DISCONNECTED")
        assert snap.model_status == "NOT_LOADED"
        assert snap.data_health.status == "CRITICAL"

    def test_data_health_model_not_loaded(self) -> None:
        from live_chart_snapshot import assess_data_health
        dh = assess_data_health(
            last_candle_ts=datetime.now().timestamp(),
            last_tick_ts=datetime.now().timestamp(),
            broker_connected=False, model_loaded=False,
            shadow_logger_ok=False, db_write_ok=False,
            missing_candles=0, api_errors_today=0,
        )
        assert dh.status == "CRITICAL"


class TestChartHandlesEmptyPredictionRows:
    def test_load_empty_prediction_rows(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            plugin.load_prediction_rows([])
            plugin.load_prediction_rows(None)
            assert len(plugin._prediction_rows) == 0
        finally:
            frame.destroy()


class TestChartHandlesMalformedCandles:
    def test_candle_with_missing_fields(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            # Candle with missing 'time'
            bad_candle = SimpleNamespace(open=100, high=105, low=99, close=103)
            plugin.push_candles([bad_candle])
            # Should not crash
        finally:
            frame.destroy()

    def test_candle_with_None_time(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            bad_candle = SimpleNamespace(open=100, high=105, low=99, close=103, time=None)
            plugin.push_candles([bad_candle])
        finally:
            frame.destroy()


class TestChartHandlesMissingTradeRows:
    def test_load_trade_event_rows_none(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            plugin.load_trade_event_rows(None)
        finally:
            frame.destroy()


class TestSnapshotBuilderDegradesGracefully:
    def test_no_crash_on_all_none(self) -> None:
        from live_chart_snapshot import build_live_chart_snapshot
        snap = build_live_chart_snapshot(
            spot_price=None,
            candles=None,
            option_chain_payload=None,
            shadow_rows=None,
            trade_events=None,
            last_candle_ts=None,
            broker_connected=False,
            model_loaded=False,
        )
        assert snap.spot_price is None
        assert snap.candles == []
        assert snap.shadow_metrics.predictions_today == 0

    def test_no_crash_with_malformed_candles(self) -> None:
        from live_chart_snapshot import build_live_chart_snapshot
        # Object with missing attributes
        bad_candle = SimpleNamespace()
        snap = build_live_chart_snapshot(candles=[bad_candle])
        # Should have skipped the bad candle
        assert len(snap.candles) == 0


class TestChartNoRealOrdersPlaced:
    """Verify LiveChartPlugin does not call any broker order methods."""

    def test_plugin_has_no_place_order_method(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            assert not hasattr(plugin, "place_order")
            assert not hasattr(plugin, "submit_order")
            assert not hasattr(plugin, "send_order")
        finally:
            frame.destroy()

    def test_plugin_no_broker_client_attribute(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            # Plugin should not hold a broker client reference
            assert not hasattr(plugin, "_client") or getattr(plugin, "_client", None) is None
        finally:
            frame.destroy()


class TestUIImportNoCrash:
    def test_ui_imports_without_broker(self) -> None:
        """Verify ui.py can be imported without broker credentials."""
        if not _TKINTER_AVAILABLE:
            pytest.skip("tkinter not available in this environment")
        import ui
        assert hasattr(ui, "ScalperUI")

    def test_live_chart_snapshot_import_no_broker(self) -> None:
        from live_chart_snapshot import build_live_chart_snapshot
        snap = build_live_chart_snapshot()
        assert snap is not None

    def test_ui_live_chart_panels_import_no_broker(self) -> None:
        if not _TKINTER_AVAILABLE:
            pytest.skip("tkinter not available in this environment")
        from ui_live_chart_panels import wire_live_chart_panels, _refresh_live_chart_tab
        assert callable(wire_live_chart_panels)
        assert callable(_refresh_live_chart_tab)