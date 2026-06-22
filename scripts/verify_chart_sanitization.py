#!/usr/bin/env python3
"""Verify chart candle sanitization drops bad rows without crashing render."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tkinter as tk

from chart import LiveChartPlugin, _normalize_epoch_seconds


def main() -> int:
    errors: list[str] = []

    bad_ts = 84405173730175337682173952
    if _normalize_epoch_seconds(bad_ts) is not None:
        errors.append("giant timestamp should be rejected")

    ok_ts = datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc).timestamp()
    if _normalize_epoch_seconds(ok_ts) is None:
        errors.append("valid timestamp should pass normalization")

    root = tk.Tk()
    root.withdraw()
    frame = tk.Frame(root)
    plugin = LiveChartPlugin(frame, symbol="NIFTY", timeframe="1m")

    now = datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc)
    candles = [
        SimpleNamespace(time=now, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000),
        SimpleNamespace(time=now, open=100.5, high=102.0, low=100.0, close=101.0, volume=1100),
        SimpleNamespace(time=bad_ts, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000),
        SimpleNamespace(time="not-a-date", open=100.0, high=101.0, low=99.0, close=100.5, volume=1000),
        SimpleNamespace(time=now.timestamp(), open=-1.0, high=1.0, low=0.5, close=0.0, volume=10),
    ]
    cleaned = plugin._sanitize_candles_for_render(candles)
    if len(cleaned) != 1:
        errors.append(f"expected 1 valid candle after sanitize, got {len(cleaned)}")

    try:
        series = []
        for i in range(5):
            t = datetime(2026, 6, 16, 10, i, tzinfo=timezone.utc)
            series.append(
                SimpleNamespace(
                    time=t, open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i, volume=1000 + i
                )
            )
        plugin.push_candles(series)
        plugin._render()
    except Exception as exc:
        errors.append(f"render crashed: {type(exc).__name__}: {exc}")

    try:
        root.destroy()
    except Exception:
        pass

    if errors:
        print("verify_chart_sanitization FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("verify_chart_sanitization PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())