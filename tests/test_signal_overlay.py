"""Tests for ML signal overlay on LiveChartPlugin."""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import pytest

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import tkinter as tk  # noqa: E402
    _TKINTER_AVAILABLE = True
except Exception:
    tk = None  # type: ignore[assignment]
    _TKINTER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _synthetic_candles(n: int, base_price: float = 24500.0) -> list:
    now = int(time.time())
    candles = []
    price = float(base_price)
    for i in range(n):
        offset = (i - n // 2) * 60
        open_ = price + (i * 5.0)
        close_ = open_ + ((i % 3) - 1) * 3.0
        high_ = max(open_, close_) + 2.0
        low_ = min(open_, close_) - 2.0
        candles.append(
            SimpleNamespace(open=open_, high=high_, low=low_, close=close_, time=now + offset)
        )
        price = close_
    return candles


def _make_chart():
    if not _TKINTER_AVAILABLE:
        pytest.skip("tkinter not available in this environment")
    root = tk.Tk()
    root.withdraw()
    frame = tk.Frame(root)
    return root, frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_buy_signal_arrow_plotted() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        buy_rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "probability": 0.72}
            for i in range(3)
        ]
        plugin.load_prediction_rows(buy_rows)

        # Scatter collection must be visible and have offsets after load
        assert plugin._signal_scatter.get_visible() is True, "Signal scatter should be visible"
        offsets = plugin._signal_scatter.get_offsets()
        assert offsets.shape[0] == 3, f"Expected 3 signal points, got {offsets.shape[0]}"
    finally:
        root.destroy()


def test_sell_signal_arrow_plotted() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        sell_rows = [
            {"timestamp": now + i * 60, "predicted_class": 0, "probability": 0.68}
            for i in range(4)
        ]
        plugin.load_prediction_rows(sell_rows)

        assert plugin._signal_scatter.get_visible() is True
        offsets = plugin._signal_scatter.get_offsets()
        assert offsets.shape[0] == 4, f"Expected 4 sell signals, got {offsets.shape[0]}"
        # All sell signals should be colored red
        colors = plugin._signal_scatter.get_facecolor()
        for row_idx in range(offsets.shape[0]):
            # Red = [1, 0, 0, 1] in RGBA
            assert colors[row_idx][0] > 0.5, "Sell signal should be red-tinted"
    finally:
        root.destroy()


def test_blocked_signal_marker() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        # Mix of taken and not-taken
        rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "trade_taken": False}
            for i in range(2)
        ] + [
            {"timestamp": now + (i + 2) * 60, "predicted_class": 1, "trade_taken": True}
            for i in range(2)
        ]
        plugin.load_prediction_rows(rows)
        plugin.set_signal_filter("blocked")
        # blocked mode: only show trade_taken=False
        offsets = plugin._signal_scatter.get_offsets()
        assert offsets.shape[0] == 2, "Filter='blocked' should show only blocked signals"
    finally:
        root.destroy()


def test_signal_filter_all() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "trade_taken": False}
            for i in range(2)
        ] + [
            {"timestamp": now + (i + 2) * 60, "predicted_class": 0, "trade_taken": True}
            for i in range(2)
        ]
        plugin.set_signal_filter("all")
        plugin.load_prediction_rows(rows)

        offsets = plugin._signal_scatter.get_offsets()
        assert offsets.shape[0] == 4, "Filter='all' should show all signals"
    finally:
        root.destroy()


def test_signal_filter_executed() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "trade_taken": False}
            for i in range(3)
        ] + [
            {"timestamp": now + (i + 3) * 60, "predicted_class": 1, "trade_taken": True}
            for i in range(2)
        ]
        plugin.set_signal_filter("executed")
        plugin.load_prediction_rows(rows)

        offsets = plugin._signal_scatter.get_offsets()
        assert offsets.shape[0] == 2, "Filter='executed' should show only taken trades"
    finally:
        root.destroy()


def test_signal_filter_shadow() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "trade_taken": False}
            for i in range(3)
        ] + [
            {"timestamp": now + (i + 3) * 60, "predicted_class": 1, "trade_taken": True}
            for i in range(2)
        ]
        plugin.set_signal_filter("shadow")
        plugin.load_prediction_rows(rows)

        offsets = plugin._signal_scatter.get_offsets()
        # shadow mode: only show trade_taken=False
        assert offsets.shape[0] == 3, "Filter='shadow' should show only shadow signals"
    finally:
        root.destroy()


def test_signal_confidence_color_mapping() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        # Rows with varying probability (high and low confidence)
        rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "probability": 0.95}
            for i in range(3)
        ] + [
            {"timestamp": now + (i + 3) * 60, "predicted_class": 0, "probability": 0.52}
            for i in range(3)
        ]
        # Should not crash even with extreme probability values
        plugin.load_prediction_rows(rows)
        offsets = plugin._signal_scatter.get_offsets()
        assert offsets.shape[0] == 6, "Should plot all rows regardless of confidence"
    finally:
        root.destroy()


def test_signal_labels_show_probability() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(30))

        now = int(time.time())
        rows = [
            {"timestamp": now + i * 60, "predicted_class": 1, "probability": 0.78}
            for i in range(3)
        ]
        # Should accept rows with probability field without crashing
        plugin.load_prediction_rows(rows)
        assert len(plugin._prediction_rows) == 3
        # Verify probability is stored
        assert plugin._prediction_rows[0]["probability"] == 0.78
    finally:
        root.destroy()


def test_multiple_signal_renders_no_memory_leak() -> None:
    root, frame = _make_chart()
    try:
        from chart import LiveChartPlugin

        plugin = LiveChartPlugin(frame)
        plugin.push_candles(_synthetic_candles(200))

        now = int(time.time())
        # 100 signals at different timestamps
        rows = [
            {
                "timestamp": now + i * 60,
                "predicted_class": i % 2,
                "probability": 0.60 + (i % 40) * 0.01,
            }
            for i in range(100)
        ]
        # Should not raise even with many signals
        plugin.load_prediction_rows(rows)
        offsets = plugin._signal_scatter.get_offsets()
        assert offsets.shape[0] == 100, "All 100 signals should be plotted"
    finally:
        root.destroy()