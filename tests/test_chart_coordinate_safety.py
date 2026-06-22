"""Coordinate safety tests for LiveChartPlugin — no broker needed."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import tkinter as tk

    _TKINTER_AVAILABLE = True
except Exception:
    tk = None
    _TKINTER_AVAILABLE = False


def _make_chart():
    if not _TKINTER_AVAILABLE:
        pytest.skip("tkinter not available")
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as exc:
        pytest.skip(f"tkinter unavailable: {exc}")
    frame = tk.Frame(root)
    return root, frame


def _synthetic_candles(n: int = 20, base_price: float = 24500.0) -> list:
    now = int(time.time())
    candles = []
    price = float(base_price)
    for i in range(n):
        offset = (i - n // 2) * 60
        open_ = price + (i * 2.0)
        close_ = open_ + ((i % 3) - 1) * 1.5
        high_ = max(open_, close_) + 1.0
        low_ = min(open_, close_) - 1.0
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


class TestCoordinateHelpers:
    def test_huge_timestamp_is_skipped(self) -> None:
        from chart import _normalize_epoch_seconds, _safe_mpl_date

        assert _normalize_epoch_seconds(84405173730175337682173952) is None
        assert _safe_mpl_date(1737000000) is not None  # unix converted to mpl date

    def test_huge_y_value_is_skipped(self) -> None:
        from chart import _valid_chart_y

        assert _valid_chart_y(84498318651573949998039040) is False
        assert _valid_chart_y(23989.15, near=24000.0) is True

    def test_nan_inf_values_are_skipped(self) -> None:
        from chart import _is_finite_number, _safe_float

        assert _is_finite_number(float("nan")) is False
        assert _is_finite_number(float("inf")) is False
        assert _safe_float("bad", default=None) is None


class TestAnnotationSafety:
    def test_bad_annotation_does_not_raise(self) -> None:
        from chart import _safe_annotate

        ax = mock.Mock()
        ax.annotate = mock.Mock(side_effect=TypeError("_set_transform bad"))
        result = _safe_annotate(ax, "bad", (1e30, 2e30), source="test")
        assert result is None

    def test_safe_ax_text_skips_invalid_data_coords(self) -> None:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin.__new__(LiveChartPlugin)
        ax = SimpleNamespace(text=lambda *a, **k: "ok")
        assert LiveChartPlugin.safe_ax_text(plugin, ax, 1e30, 2e30, "bad") is None


class TestRenderFallback:
    def test_render_fallback_does_not_crash(self) -> None:
        from chart import LiveChartPlugin

        class _FakeText:
            def __init__(self, x, y):
                self._pos = (x, y)
                self._visible = True

            def get_position(self):
                return self._pos

            def get_transform(self):
                return object()

            def get_text(self):
                return "bad"

            def set_visible(self, value):
                self._visible = value

        plugin = LiveChartPlugin.__new__(LiveChartPlugin)
        bad_text = _FakeText(1e30, 2)
        plugin.ax_price = SimpleNamespace(
            texts=[bad_text],
            annotations=[],
            transAxes=object(),
            get_xticklabels=lambda: [],
            get_yticklabels=lambda: [],
            get_legend=lambda: None,
        )
        plugin.ax_rsi = SimpleNamespace(
            texts=[],
            annotations=[],
            transAxes=object(),
            get_xticklabels=lambda: [],
            get_yticklabels=lambda: [],
            get_legend=lambda: None,
        )
        plugin.fig = SimpleNamespace(texts=[])
        plugin._last_render_xs = [20620.0, 20620.1]
        plugin._last_render_closes = [24500.0, 24501.0]
        calls = {"draw": 0}

        def draw():
            calls["draw"] += 1
            if calls["draw"] == 1:
                raise TypeError("_set_transform bad delta")

        plugin.canvas = SimpleNamespace(draw=draw)
        LiveChartPlugin._safe_draw_idle(plugin, fallback_basic=True)
        assert calls["draw"] >= 2

    def test_empty_candles_do_not_crash(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin

            plugin = LiveChartPlugin(frame)
            plugin.push_candles([])
            plugin._render()
        finally:
            frame.destroy()

    def test_huge_overlay_timestamp_skipped_on_render(self) -> None:
        root, frame = _make_chart()
        try:
            from chart import LiveChartPlugin

            plugin = LiveChartPlugin(frame)
            plugin.push_candles(_synthetic_candles(25))
            plugin.load_prediction_rows(
                [
                    {
                        "ts": 725247989284439860618272602857668608,
                        "probability": 0.9,
                        "predicted_class": 1,
                        "trade_candidate": "BUY_CE",
                        "trade_taken": False,
                    }
                ]
            )
            plugin._render()
        finally:
            frame.destroy()


class TestMalformedLiveSpot:
    def test_malformed_live_spot_json_does_not_crash_chart(self, tmp_path: Path) -> None:
        root, frame = _make_chart()
        spot_file = tmp_path / "live_spot.json"
        spot_file.write_text(
            json.dumps(
                {
                    "spot": "not-a-number",
                    "timestamp": 84405173730175337682173952,
                    "token": 50615,
                }
            ),
            encoding="utf-8",
        )
        try:
            from chart import LiveChartPlugin

            plugin = LiveChartPlugin(frame)
            plugin.push_candles(_synthetic_candles(15))
            # Simulate bad market overlay values that could come from malformed cache.
            plugin.set_market_info(
                spot=None,
                atm_strike=84405173730175337682173952,
                session_high=1e30,
                session_low=-5,
            )
            plugin.set_session_levels(1e30, 23900.0)
            plugin.set_prevday_levels(24050.0, 84498318651573949998039040)
            plugin._render()
            assert plugin._session_high is None or plugin._session_high < 1e9
        finally:
            frame.destroy()
            assert spot_file.exists()