"""
router_with_presets.py
======================
Router with dynamic preset selection.
Flow: live features -> candidate router -> preset selector -> risk firewall -> final decision
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src import candidate_router
from src.preset_selector import load_preset_policy, select_preset, apply_preset_to_decision

_LOGGER = logging.getLogger(__name__)

# Load preset policy once at module level
_PRESET_POLICY = None


def _get_preset_policy():
    """Get (or lazily load) the preset policy."""
    global _PRESET_POLICY
    if _PRESET_POLICY is None:
        try:
            _PRESET_POLICY = load_preset_policy("config/dynamic_presets.json")
        except Exception as exc:
            _LOGGER.warning("Failed to load preset policy: %s — using empty policy", exc)
            _PRESET_POLICY = {"presets": {}, "hard_caps": {}, "preset_order": []}
    return _PRESET_POLICY


def route_with_presets(
    snapshot: Dict[str, Any],
    candidate_manifests: List[Dict[str, Any]],
    live_features: Dict[str, Any],
    mode: str,
    market_quality: str = "acceptable",
) -> Dict[str, Any]:
    """
    Full pipeline: router -> preset selector -> merged decision.
    
    Returns dict with all router fields PLUS:
    - selected_preset: str
    - preset_reason: str  
    - base_threshold: float
    - effective_threshold: float
    - threshold_adjustment: float
    - preset_trade_allowed: bool
    - hard_safety_allowed: bool (set by firewall, initially True)
    - final_action: "TRADE" | "BLOCK"
    
    Parameters
    ----------
    snapshot : dict[str, Any]
        Live feature snapshot.
    candidate_manifests : list[dict[str, Any]]
        List of loaded and validated candidate manifest dicts.
    live_features : dict[str, Any]
        Current live features for preset selection context.
    mode : str
        Operating mode: "shadow", "paper", or "live".
    market_quality : str
        Market quality assessment: "good", "acceptable", or "poor".
    """
    # Step 1: Route candidates through the base router
    router_result = candidate_router.route_candidates(
        snapshot=snapshot,
        candidates=candidate_manifests,
        mode=mode,
    )
    
    # Step 2: Select appropriate preset based on router output and conditions
    selected_preset = select_preset(
        router_result=router_result,
        live_features=live_features,
        market_quality=market_quality,
        mode=mode,
    )
    
    # Step 3: Apply preset to decision (threshold adjustment, trade allowed check)
    merged_decision = apply_preset_to_decision(
        router_result=router_result,
        selected_preset=selected_preset,
        mode=mode,
    )
    
    # Log pipeline decision for audit trail
    _LOGGER.info(
        "route_with_presets: router_action=%s selected_preset=%s final_action=%s "
        "base_threshold=%.4f effective_threshold=%.4f adjustment=%.4f",
        router_result.get("router_action"),
        selected_preset.get("preset_name"),
        merged_decision.get("final_action"),
        merged_decision.get("base_threshold", 0.0),
        merged_decision.get("effective_threshold", 0.0),
        merged_decision.get("threshold_adjustment", 0.0),
    )
    
    return merged_decision


def route_with_presets_and_firewall(
    snapshot: Dict[str, Any],
    candidate_manifests: List[Dict[str, Any]],
    live_features: Dict[str, Any],
    mode: str,
    market_quality: str = "acceptable",
    hard_safety_allowed: bool = True,
) -> Dict[str, Any]:
    """
    Extended pipeline with explicit firewall override.
    
    Adds a hard_safety_allowed parameter that can force BLOCK regardless
    of preset settings.
    """
    result = route_with_presets(
        snapshot=snapshot,
        candidate_manifests=candidate_manifests,
        live_features=live_features,
        mode=mode,
        market_quality=market_quality,
    )
    
    # Firewall override - cannot be bypassed by presets
    result["hard_safety_allowed"] = hard_safety_allowed
    if not hard_safety_allowed:
        result["final_action"] = "BLOCK"
        _LOGGER.warning("FIREWALL OVERRIDE: hard_safety_allowed=False, forcing BLOCK")
    
    return result


def invalidate_preset_cache() -> None:
    """Clear the cached preset policy (for testing or hot reload)."""
    global _PRESET_POLICY
    _PRESET_POLICY = None
    _LOGGER.debug("Preset policy cache cleared in router_with_presets")
