"""Market regime detection and strategy selection.

This module provides a small, testable mapping from market regime to
recommended strategy templates. The detection functions are intentionally
simple and deterministic so they can be replaced by ML-based detectors
later without changing the strategy selection contract.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from capital_allocator import allocate_capital_across_sleeves, build_sleeve_scores

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
    "mean_reverting": "mean_reversion",
    "quiet": "stat_arb",
}

_REGIME_TUNING_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "trending": {
        "trend_mult": 1.20,
        "ml_threshold": 0.58,
        "preferred": ["bull_call_spread", "call_ratio_backspread", "long_call", "short_put"],
    },
    "volatile": {
        "trend_mult": 0.90,
        "ml_threshold": 0.62,
        "preferred": ["long_straddle", "long_strangle", "iron_condor"],
    },
    "mean_reverting": {
        "trend_mult": 0.75,
        "ml_threshold": 0.55,
        "preferred": ["mean_reversion", "iron_condor", "iron_fly", "short_straddle"],
    },
    "quiet": {
        "trend_mult": 0.65,
        "ml_threshold": 0.57,
        "preferred": ["stat_arb", "short_strangle", "iron_condor", "short_straddle"],
    },
}


def get_regime_tuning(regime: str, cfg: Optional[Any] = None) -> Dict[str, Any]:
    key = str(regime or "").strip().lower() or "quiet"
    tuning = dict(_REGIME_TUNING_DEFAULTS.get(key, _REGIME_TUNING_DEFAULTS["quiet"]))
    if cfg is None:
        tuning["regime"] = key
        return tuning

    attr_prefix = f"regime_{key}_"
    for attr, out_key in (("trend_mult", "trend_mult"), ("ml_threshold", "ml_threshold")):
        try:
            v = getattr(cfg, f"{attr_prefix}{attr}", None)
            if v is not None:
                tuning[out_key] = float(v)
        except Exception:
            pass
    try:
        pref = getattr(cfg, f"{attr_prefix}preferred_strategy", "")
        if str(pref or "").strip():
            tuning["preferred"] = [str(pref).strip()]
    except Exception:
        pass
    tuning["regime"] = key
    return tuning


def select_strategy_for_regime(regime: str, cfg: Optional[Any] = None) -> str:
    """Map regime to an implemented strategy key used by the router."""
    key = str(regime or "").lower()
    tuning = get_regime_tuning(key, cfg=cfg)
    preferred = list(tuning.get("preferred") or [])
    if preferred:
        return str(preferred[0])
    return _REGIME_TO_STRATEGY.get(key, "directional")


def allocate_sleeves_for_regime(
    regime: str,
    *,
    account_capital: float,
    trend_score: float,
    mean_reversion_score: float,
    stat_arb_score: float,
    max_single_sleeve_weight: float = 0.50,
    reserve_cash_weight: float = 0.10,
    drawdown_throttle: float = 1.0,
) -> Dict[str, Any]:
    scores = build_sleeve_scores(
        regime=str(regime or ""),
        trend_score=float(trend_score),
        mean_reversion_score=float(mean_reversion_score),
        stat_arb_score=float(stat_arb_score),
    )
    allocations = allocate_capital_across_sleeves(
        account_capital=float(account_capital),
        sleeve_scores=scores,
        max_single_sleeve_weight=float(max_single_sleeve_weight),
        reserve_cash_weight=float(reserve_cash_weight),
        drawdown_throttle=float(drawdown_throttle),
    )
    selected = "directional"
    if allocations:
        selected = max(allocations.values(), key=lambda row: float(getattr(row, "weight", 0.0))).sleeve
    strategy_map = {
        "trend": "directional",
        "mean_reversion": "mean_reversion",
        "stat_arb": "stat_arb",
    }
    return {
        "regime": str(regime or "").strip().lower() or "quiet",
        "scores": scores,
        "allocations": allocations,
        "selected_sleeve": selected,
        "selected_strategy": strategy_map.get(selected, select_strategy_for_regime(regime, None)),
    }
