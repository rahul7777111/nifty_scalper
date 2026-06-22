"""
preset_selector.py
==================
Dynamic preset selection for shadow/paper router pipeline.

Loads preset policy from config/dynamic_presets.json and applies preset-based
threshold adjustments and trade restrictions based on market quality and mode.

Safety
------
- Presets are for shadow/paper modes ONLY.
- Cannot enable live trading or mutate production settings.
- All mutations are local to the decision dict — no global state changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)

# Module-level cache for loaded policy
_POLICY_CACHE: Optional[Dict[str, Any]] = None


def load_preset_policy(policy_path: str = "config/dynamic_presets.json") -> Dict[str, Any]:
    """Load preset policy JSON from disk."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Preset policy not found: {policy_path}")
    
    with path.open("rt", encoding="utf-8") as fh:
        policy = json.load(fh)
    
    _LOGGER.debug("Loaded preset policy version=%s from %s", 
                  policy.get("version", "unknown"), policy_path)
    return policy


def get_cached_policy(policy_path: str = "config/dynamic_presets.json") -> Dict[str, Any]:
    """Get preset policy with module-level caching."""
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        try:
            _POLICY_CACHE = load_preset_policy(policy_path)
        except Exception as exc:
            _LOGGER.warning("Failed to load preset policy: %s — using empty policy", exc)
            _POLICY_CACHE = {"presets": {}, "hard_caps": {}, "preset_order": []}
    return _POLICY_CACHE


def select_preset(
    router_result: Dict[str, Any],
    live_features: Dict[str, Any],
    market_quality: str = "acceptable",
    mode: str = "shadow",
) -> Dict[str, Any]:
    """Select the appropriate preset based on router output and market conditions."""
    policy = get_cached_policy()
    presets = policy.get("presets", {})
    
    router_action = router_result.get("router_action", "SKIP")
    if router_action == "SKIP":
        return _build_preset_response(
            presets.get("BLOCK", {}),
            "BLOCK",
            "router_returned_skip_no_eligible_candidates"
        )
    
    if mode == "live":
        _LOGGER.warning("Live mode detected — presets cannot enable trading")
        return _build_preset_response(
            presets.get("BLOCK", {}),
            "BLOCK",
            "live_mode_blocked_by_preset_safety"
        )
    
    aggressive_preset = presets.get("AGGRESSIVE_SHADOW_ONLY", {})
    allowed_modes = aggressive_preset.get("allowed_modes", [])
    if allowed_modes and mode not in allowed_modes:
        pass
    
    if market_quality == "poor":
        coverage = live_features.get("feature_coverage_pct", 
                                     router_result.get("feature_coverage", {}).get("coverage_pct", 0.0))
        if coverage < 95.0:
            return _build_preset_response(
                presets.get("OBSERVE_ONLY", {}),
                "OBSERVE_ONLY",
                f"poor_market_quality_coverage_{coverage:.1f}%"
            )
        return _build_preset_response(
            presets.get("CONSERVATIVE", {}),
            "CONSERVATIVE",
            "poor_market_quality_threshold_bump"
        )
    
    elif market_quality == "good":
        if mode == "shadow":
            aggressive = presets.get("AGGRESSIVE_SHADOW_ONLY", {})
            if aggressive:
                return _build_preset_response(
                    aggressive,
                    "AGGRESSIVE_SHADOW_ONLY",
                    "good_market_shadow_aggressive_enabled"
                )
        return _build_preset_response(
            presets.get("NORMAL", {}),
            "NORMAL",
            "good_market_quality_normal_mode"
        )
    
    else:
        return _build_preset_response(
            presets.get("NORMAL", {}),
            "NORMAL",
            "acceptable_market_quality_default"
        )


def apply_preset_to_decision(
    router_result: Dict[str, Any],
    selected_preset: Dict[str, Any],
    mode: str = "shadow",
) -> Dict[str, Any]:
    """Apply preset adjustments to router decision."""
    base_threshold = router_result.get("threshold", 0.5) or 0.5
    probability = router_result.get("probability", 0.0) or 0.0
    
    adjustment = selected_preset.get("effective_threshold_adjustment")
    if adjustment is None:
        adjustment = 0.0
    
    effective_threshold = base_threshold + adjustment
    
    if mode == "shadow":
        preset_trade_allowed = selected_preset.get("shadow_log_allowed", True)
    elif mode == "paper":
        preset_trade_allowed = selected_preset.get("paper_trade_allowed", False)
    else:
        preset_trade_allowed = selected_preset.get("trade_allowed", False)
    
    threshold_passed = probability > effective_threshold if probability > 0 else False
    
    if not preset_trade_allowed or not threshold_passed:
        final_action = "BLOCK"
    else:
        final_action = "TRADE"
    
    result = dict(router_result)
    result.update({
        "selected_preset": selected_preset.get("preset_name", "UNKNOWN"),
        "preset_reason": selected_preset.get("preset_reason", ""),
        "base_threshold": base_threshold,
        "effective_threshold": effective_threshold,
        "threshold_adjustment": adjustment,
        "preset_trade_allowed": preset_trade_allowed,
        "hard_safety_allowed": True,
        "final_action": final_action,
        "spread_limit_tier": selected_preset.get("spread_limit_tier"),
        "max_trades_per_day": selected_preset.get("max_trades_per_day"),
        "max_open_positions": selected_preset.get("max_open_positions"),
        "feature_coverage_minimum": selected_preset.get("feature_coverage_minimum"),
    })
    
    return result


def _build_preset_response(
    preset_config: Dict[str, Any],
    preset_name: str,
    preset_reason: str,
) -> Dict[str, Any]:
    """Helper to build a standardized preset response dict."""
    return {
        "preset_name": preset_name,
        "preset_reason": preset_reason,
        "trade_allowed": preset_config.get("trade_allowed", False),
        "paper_trade_allowed": preset_config.get("paper_trade_allowed", False),
        "shadow_log_allowed": preset_config.get("shadow_log_allowed", True),
        "effective_threshold_adjustment": preset_config.get("effective_threshold_adjustment"),
        "max_trades_per_day": preset_config.get("max_trades_per_day", 0),
        "max_open_positions": preset_config.get("max_open_positions", 0),
        "spread_limit_tier": preset_config.get("spread_limit_tier"),
        "feature_coverage_minimum": preset_config.get("feature_coverage_minimum"),
        "description": preset_config.get("description", ""),
    }


def invalidate_policy_cache() -> None:
    """Clear the cached preset policy."""
    global _POLICY_CACHE
    _POLICY_CACHE = None
    _LOGGER.debug("Preset policy cache cleared")
