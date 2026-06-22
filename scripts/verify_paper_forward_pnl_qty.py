#!/usr/bin/env python3
"""Verify Paper Forward option P&L uses rupee qty, not percent/per-unit diff."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import _compute_option_pnl, _resolve_paper_position_qty


def assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError(f"{name}: expected {expected:.2f}, got {actual:.2f}")


def main() -> int:
    os.environ.pop("NIFTY_LOT_SIZE", None)
    os.environ.pop("PAPER_FORWARD_DEFAULT_QTY", None)

    cases = [
        ("BUY CE unreal", _compute_option_pnl(15.95, 14.70, 65, side="BUY"), -81.25),
        ("BUY PE unreal", _compute_option_pnl(23.15, 23.65, 65, side="BUY"), 32.50),
        ("Winning exit", _compute_option_pnl(20.00, 25.00, 65, side="BUY"), 325.00),
        ("Losing exit", _compute_option_pnl(20.00, 18.00, 65, side="BUY"), -130.00),
    ]
    for name, actual, expected in cases:
        assert_close(name, actual, expected)

    qty, lot_size, lots, source = _resolve_paper_position_qty({}, lots=1)
    if (qty, lot_size, lots) != (65, 65, 1):
        raise AssertionError(f"missing lot_size default expected qty=65 lot_size=65 lots=1, got {(qty, lot_size, lots)}")
    if source != "default":
        raise AssertionError(f"missing lot_size source expected default, got {source}")

    bad_percent = (14.70 / 15.95) - 1.0
    bad_per_unit = 14.70 - 15.95
    actual = _compute_option_pnl(15.95, 14.70, 65, side="BUY")
    if math.isclose(actual, bad_percent, abs_tol=1e-9) or math.isclose(actual, bad_per_unit, abs_tol=1e-9):
        raise AssertionError("P&L is percent/per-unit, expected rupee qty P&L")

    print("paper_forward_pnl_qty: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
