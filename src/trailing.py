"""Trailing stop helpers (ATR-based)."""
from __future__ import annotations

from typing import Iterable, List


def atr_trailing_stop(prices: Iterable[float], atr_values: Iterable[float], atr_mult: float = 1.5) -> float:
    """Return trailing stop price given latest price and ATR history.

    This returns price - atr_mult * atr_last for long positions.
    """
    try:
        p = list(prices)
        a = list(atr_values)
        if not p or not a:
            return 0.0
        last_price = float(p[-1])
        last_atr = float(a[-1])
        return last_price - atr_mult * last_atr
    except Exception:
        return 0.0
