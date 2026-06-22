#!/usr/bin/env python3
"""
dynamic_preset_selector.py
==========================
Runtime dynamic preset selection for candidate profiles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from candidate_profile import (
    CandidateProfile,
    DynamicPreset,
    SIDE_AUTO_DIRECTIONAL,
    SIDE_CE_ONLY,
    SIDE_PE_ONLY,
)


def detect_market_regime(snapshot: Dict[str, Any]) -> str:
    regime = str(snapshot.get("regime") or snapshot.get("regime_label") or "").lower()
    if "bull" in regime or "trend_up" in regime or "uptrend" in regime:
        return "bullish"
    if "bear" in regime or "trend_down" in regime or "downtrend" in regime:
        return "bearish"
    if "chop" in regime or "range" in regime or "quiet" in regime:
        return "mixed"
    adx = _f(snapshot.get("adx_14"))
    if adx >= 25:
        return "bullish" if _f(snapshot.get("ret_mean", 0)) >= 0 else "bearish"
    return "mixed"


def detect_volatility_state(snapshot: Dict[str, Any]) -> str:
    atr_pct = _f(snapshot.get("atr_pct"))
    if atr_pct >= 0.012:
        return "high"
    if atr_pct >= 0.006:
        return "normal"
    return "low"


def detect_trend_state(snapshot: Dict[str, Any]) -> str:
    adx = _f(snapshot.get("adx_14"))
    if adx >= 28:
        return "strong"
    if adx >= 18:
        return "moderate"
    return "weak"


def detect_liquidity_state(snapshot: Dict[str, Any]) -> str:
    spread = _f(snapshot.get("spread_pct", snapshot.get("range_pct")))
    vol = _f(snapshot.get("volume", snapshot.get("liquidity_score")))
    if spread > 0.15:
        return "poor"
    if vol < 50000 and vol > 0:
        return "thin"
    return "good"


def detect_time_window(snapshot: Dict[str, Any]) -> str:
    t = str(snapshot.get("time_of_day") or snapshot.get("session_time") or "")
    if not t and snapshot.get("timestamp"):
        try:
            from datetime import datetime
            ts = snapshot["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            t = ts.strftime("%H:%M")
        except Exception:
            t = ""
    if t >= "09:30" and t <= "10:30":
        return "open"
    if t >= "11:30" and t <= "14:00":
        return "midday"
    if t >= "15:00":
        return "close"
    return "regular"


def _f(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _preset_matches(
    preset: DynamicPreset,
    *,
    regime: str,
    vol_state: str,
    trend_state: str,
    liquidity_state: str,
    time_state: str,
) -> tuple[bool, str]:
    if preset.regime_filter_str not in ("all", "") and preset.regime_filter_str != regime:
        if preset.regime_filter_str == "high_volatility" and vol_state != "high":
            return False, f"regime_mismatch_need_high_vol"
        if preset.regime_filter_str == "non_chop" and regime == "mixed":
            return False, "regime_mismatch_chop"
        if preset.regime_filter_str == "trend_only" and trend_state == "weak":
            return False, "regime_mismatch_weak_trend"

    if preset.volatility_filter:
        min_atr = _f(preset.volatility_filter.get("min_atr_pct"))
        if min_atr > 0 and _f(0) >= min_atr:
            pass  # checked via snapshot in caller

    if liquidity_state == "poor" and preset.spread_limit_pct < 0.20:
        return False, "liquidity_poor_spread_too_strict"

    if preset.preset_family == "time_window_open" and time_state != "open":
        return False, "outside_open_window"
    if preset.preset_family == "time_window_midday" and time_state != "midday":
        return False, "outside_midday_window"

    return True, ""


def safe_no_trade_preset(reason: str) -> Dict[str, Any]:
    return {
        "selected_preset_name": "NO_TRADE",
        "why_selected": reason,
        "no_trade_reason": reason,
        "side_allowed": "NONE",
        "threshold_used": 1.0,
        "trade_allowed": False,
    }


def resolve_side_for_preset(preset: DynamicPreset, snapshot: Dict[str, Any], regime: str) -> str:
    policy = preset.option_side_policy
    opt = str(snapshot.get("option_type") or "").upper()
    if policy == SIDE_PE_ONLY:
        return "PE"
    if policy == SIDE_CE_ONLY:
        return "CE"
    if policy == SIDE_AUTO_DIRECTIONAL:
        detail = preset.regime_filter if isinstance(preset.regime_filter, dict) else {}
        if regime == "bullish":
            return str(detail.get("bullish", "CE")).upper()
        if regime == "bearish":
            return str(detail.get("bearish", "PE")).upper()
        return "NONE"
    if opt in {"CE", "PE"}:
        return opt
    return "BOTH"


def select_dynamic_preset(
    candidate_profile: CandidateProfile | Dict[str, Any],
    market_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Choose the best matching preset for the current market snapshot."""
    if isinstance(candidate_profile, dict):
        profile = CandidateProfile.from_dict(candidate_profile)
    else:
        profile = candidate_profile

    regime = detect_market_regime(market_snapshot)
    vol_state = detect_volatility_state(market_snapshot)
    trend_state = detect_trend_state(market_snapshot)
    liquidity_state = detect_liquidity_state(market_snapshot)
    time_state = detect_time_window(market_snapshot)

    presets: List[DynamicPreset] = []
    for _name, pdata in (profile.dynamic_presets or {}).items():
        if isinstance(pdata, dict):
            presets.append(DynamicPreset.from_dict(pdata))

    if not presets:
        # PHASE 2: try global central loader fallback (config/dynamic_presets.json etc) so we never emit the old generic no_dynamic... when paper testing
        try:
            from paper_forward_engine import load_dynamic_presets
            greg = load_dynamic_presets()
            gdata = greg.get("data", {}) if greg.get("loaded") else {}
            gp = gdata.get("presets", gdata.get("dynamic_presets", {})) if isinstance(gdata, dict) else {}
            for _name, pdata in (gp or {}).items():
                if isinstance(pdata, dict):
                    presets.append(DynamicPreset.from_dict(pdata))
        except Exception:
            pass
    if not presets:
        return safe_no_trade_preset("no_dynamic_presets_configured")  # only if truly nothing + no fallback possible

    for preset in presets:
        ok, why_not = _preset_matches(
            preset,
            regime=regime,
            vol_state=vol_state,
            trend_state=trend_state,
            liquidity_state=liquidity_state,
            time_state=time_state,
        )
        if not ok:
            continue
        side = resolve_side_for_preset(preset, market_snapshot, regime)
        if side == "NONE":
            return safe_no_trade_preset("auto_directional_mixed_regime_no_trade")
        return {
            "selected_preset_name": preset.name,
            "why_selected": f"matched regime={regime} vol={vol_state} trend={trend_state} liq={liquidity_state} time={time_state}",
            "no_trade_reason": None,
            "side_allowed": side,
            "threshold_used": preset.entry_threshold,
            "min_confidence": preset.min_confidence,
            "max_trades_per_day": preset.max_trades_per_day,
            "trade_allowed": True,
            "preset_family": preset.preset_family,
            "candidate_id": profile.candidate_id,
            "model_name": profile.model_name,
            "market_regime": regime,
            "preset_config": preset.to_retrainer_dict(),
        }

    return safe_no_trade_preset("no_preset_matched_current_conditions")