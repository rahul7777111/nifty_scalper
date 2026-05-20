"""Market regime detection and strategy selection.

This module provides a small, testable mapping from market regime to
recommended strategy templates. The detection functions are intentionally
simple and deterministic so they can be replaced by ML-based detectors
later without changing the strategy selection contract.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

def detect_regime(
    atr_values: Iterable[float], adx_values: Iterable[float], rsi_values: Iterable[float]
) -> str:
    """Return one of: 'trending', 'mean_reverting', 'volatile', 'quiet'.

    Heuristic rules:
    - High ADX (>25) + rising ATR -> 'trending'
    - High ATR (>recent median * 1.5) -> 'volatile'
    - Low ATR + RSI oscillating near 50 -> 'mean_reverting'
    - otherwise 'quiet'
    """
    try:
        atr = list(atr_values)
        adx = list(adx_values)
        rsi = list(rsi_values)
        if not atr or not adx:
            return "quiet"
        atr_recent = atr[-1]
        atr_med = sorted(atr)[max(0, len(atr) // 2 - 1)]
        adx_recent = adx[-1] if adx else 0.0
        rsi_recent = rsi[-1] if rsi else 50.0

        if adx_recent > 25 and atr_recent > atr_med:
            return "trending"
        if atr_recent > (atr_med * 1.5):
            return "volatile"
        if abs(rsi_recent - 50.0) < 8 and atr_recent <= atr_med:
            return "mean_reverting"
        return "quiet"
    except Exception:
        return "quiet"


_REGIME_TO_STRATEGY: Dict[str, str] = {
    "trending": "bull_call_spread",
    "volatile": "long_straddle",
    "mean_reverting": "iron_condor",
    "quiet": "short_strangle",
}


def select_strategy_for_regime(regime: str) -> str:
    """Map regime to an implemented strategy key used by the router."""
    return _REGIME_TO_STRATEGY.get(str(regime or "").lower(), "directional")
