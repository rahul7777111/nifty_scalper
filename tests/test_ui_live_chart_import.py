"""Tests for UI Live Chart import and wiring — no broker needed."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Use Agg for matplotlib
import matplotlib
matplotlib.use("Agg")

# Check tkinter availability before any test tries to import it
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


class TestLiveChartSnapshotModule:
    def test_import(self) -> None:
        from live_chart_snapshot import (
            Candle, SignalMarker, ShadowMetrics,
            OptionChainSummary, DataHealth, LiveChartSnapshot,
            build_live_chart_snapshot,
            option_chain_to_summary,
            compute_shadow_metrics,
            assess_data_health,
        )
        assert True

    def test_all_exports_usable(self) -> None:
        from live_chart_snapshot import build_live_chart_snapshot
        snap = build_live_chart_snapshot(spot_price=24500.0)
        assert snap.spot_price == 24500.0


class TestChartPluginModule:
    def test_import_live_chart_plugin(self) -> None:
        # Importing chart.py requires tkinter at module level — skip if unavailable
        if not _TKINTER_AVAILABLE:
            pytest.skip("tkinter not available in this environment")
        from chart import LiveChartPlugin
        assert LiveChartPlugin is not None

    def test_live_chart_plugin_has_required_methods(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            required = [
                "push_candles", "load_prediction_rows", "load_trade_event_rows",
                "set_market_info", "set_session_levels", "set_prevday_levels",
                "set_atm_strike", "set_overlay_visible", "set_signal_filter",
                "export_png", "focus_trade",
            ]
            for m in required:
                assert hasattr(plugin, m), f"Missing method: {m}"
        finally:
            frame.destroy()


class TestUILiveChartPanelsModule:
    def test_import_wire_function(self) -> None:
        # ui_live_chart_panels.py imports tkinter at module level — skip if unavailable
        if not _TKINTER_AVAILABLE:
            pytest.skip("tkinter not available in this environment")
        from ui_live_chart_panels import wire_live_chart_panels
        assert callable(wire_live_chart_panels)

    def test_import_refresh_function(self) -> None:
        if not _TKINTER_AVAILABLE:
            pytest.skip("tkinter not available in this environment")
        from ui_live_chart_panels import _refresh_live_chart_tab
        assert callable(_refresh_live_chart_tab)

    def test_all_panel_build_functions_exist(self) -> None:
        if not _TKINTER_AVAILABLE:
            pytest.skip("tkinter not available in this environment")
        from ui_live_chart_panels import (
            _build_current_prediction_card,
            _build_shadow_metrics_card,
            _build_option_chain_card,
            _build_data_health_card,
            _build_alert_card,
            _build_event_timeline,
            _build_top_status_bar,
        )
        for fn in [
            _build_current_prediction_card,
            _build_shadow_metrics_card,
            _build_option_chain_card,
            _build_data_health_card,
            _build_alert_card,
            _build_event_timeline,
            _build_top_status_bar,
        ]:
            assert callable(fn)


class TestUIImportWithLiveChartPanels:
    def test_ui_scalperui_imports(self) -> None:
        # ui.py imports tkinter at module level — skip if unavailable
        if not _TKINTER_AVAILABLE:
            pytest.skip("tkinter not available in this environment")
        import ui
        assert hasattr(ui, "ScalperUI")

    def test_ui_import_includes_live_chart_panels_reference(self) -> None:
        # Verify the import statement was added to ui.py (source-level check, no tkinter needed)
        ui_source = Path(SRC_DIR) / "ui.py"
        content = ui_source.read_text(encoding="utf-8")
        assert "ui_live_chart_panels" in content
        assert "wire_live_chart_panels" in content


class TestChartPluginOverlayCompatibility:
    def test_overlay_visible_property(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            assert hasattr(plugin, "_overlay_visible")
            assert isinstance(plugin._overlay_visible, dict)
            assert plugin._overlay_visible.get("ema_fast") is True
        finally:
            frame.destroy()

    def test_prevday_levels_property(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            assert hasattr(plugin, "_prevday_levels")
            assert isinstance(plugin._prevday_levels, dict)
            assert plugin._prevday_levels["high"] is None
        finally:
            frame.destroy()

    def test_session_levels_property(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            assert hasattr(plugin, "_session_levels")
            assert isinstance(plugin._session_levels, dict)
        finally:
            frame.destroy()


class TestChartPluginVolumeOverlay:
    def test_volume_overlay_default_on(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            assert plugin._overlays.get("volume") is True
        finally:
            frame.destroy()

    def test_set_overlay_visible_toggles_volume(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin
            plugin = LiveChartPlugin(frame)
            plugin.set_overlay_visible("volume", False)
            assert plugin._overlays.get("volume") is False
            plugin.set_overlay_visible("volume", True)
            assert plugin._overlays.get("volume") is True
        finally:
            frame.destroy()