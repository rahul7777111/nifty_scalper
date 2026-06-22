"""Tests for preset selector logic."""
import pytest
from src.preset_selector import (
    load_preset_policy, select_preset, apply_preset_to_decision,
    get_cached_policy, invalidate_policy_cache,
)

def test_select_preset_blocks_on_skip():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router_result = {"router_action": "SKIP", "probability": 0.6, "edge_score": 0.1,
                     "threshold": 0.5, "coverage_pct": 0.98}
    live_features = {"feature_coverage_pct": 0.98, "stale_quote": False}
    result = select_preset(router_result, live_features, "good", "paper")
    assert result["preset_name"] == "BLOCK"

def test_select_preset_blocks_on_live_mode():
    invalidate_policy_cache()
    router_result = {"router_action": "TRADE", "probability": 0.6, "edge_score": 0.1,
                     "threshold": 0.5, "coverage_pct": 0.98}
    live_features = {"feature_coverage_pct": 0.98, "stale_quote": False}
    result = select_preset(router_result, live_features, "good", "live")
    assert result["preset_name"] == "BLOCK"

def test_select_preset_normal_on_good_market():
    invalidate_policy_cache()
    router_result = {"router_action": "TRADE", "probability": 0.65, "edge_score": 0.15,
                     "threshold": 0.5, "coverage_pct": 0.97}
    live_features = {"feature_coverage_pct": 0.97, "stale_quote": False}
    result = select_preset(router_result, live_features, "good", "paper")
    assert result["preset_name"] == "NORMAL"

def test_select_preset_aggressive_shadow_only_in_shadow():
    invalidate_policy_cache()
    router_result = {"router_action": "TRADE", "probability": 0.70, "edge_score": 0.20,
                     "threshold": 0.5, "coverage_pct": 0.95}
    live_features = {"feature_coverage_pct": 0.95, "stale_quote": False}
    result = select_preset(router_result, live_features, "good", "shadow")
    assert result["preset_name"] == "AGGRESSIVE_SHADOW_ONLY"
    assert result["effective_threshold_adjustment"] == -0.03

def test_select_preset_aggressive_blocked_in_paper():
    invalidate_policy_cache()
    router_result = {"router_action": "TRADE", "probability": 0.70, "edge_score": 0.20,
                     "threshold": 0.5, "coverage_pct": 0.95}
    live_features = {"feature_coverage_pct": 0.95, "stale_quote": False}
    result = select_preset(router_result, live_features, "good", "paper")
    assert result["preset_name"] != "AGGRESSIVE_SHADOW_ONLY"

def test_select_preset_conservative_on_poor_quality():
    invalidate_policy_cache()
    router_result = {"router_action": "TRADE", "probability": 0.57, "edge_score": 0.07,
                     "threshold": 0.5, "coverage_pct": 0.97}
    live_features = {"feature_coverage_pct": 0.85, "stale_quote": False}
    result = select_preset(router_result, live_features, "poor", "paper")
    assert result["preset_name"] == "OBSERVE_ONLY"

def test_select_preset_default_accept():
    invalidate_policy_cache()
    router_result = {"router_action": "TRADE", "probability": 0.55, "edge_score": 0.08,
                     "threshold": 0.5, "coverage_pct": 0.96}
    live_features = {"feature_coverage_pct": 0.96, "stale_quote": False}
    result = select_preset(router_result, live_features, "acceptable", "paper")
    assert result["preset_name"] == "NORMAL"

def test_apply_preset_preserves_router_fields():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router = {"router_action": "TRADE", "probability": 0.6, "edge_score": 0.1,
              "threshold": 0.5, "coverage_pct": 0.98, "selected_candidate_id": "PE_elast"}
    conservative_preset = policy["presets"]["CONSERVATIVE"]
    selected = {
        "preset_name": "CONSERVATIVE",
        "trade_allowed": conservative_preset["trade_allowed"],
        "paper_trade_allowed": conservative_preset["paper_trade_allowed"],
        "effective_threshold_adjustment": conservative_preset["effective_threshold_adjustment"],
        "spread_limit_tier": conservative_preset["spread_limit_tier"],
        "max_trades_per_day": conservative_preset["max_trades_per_day"],
        "max_open_positions": conservative_preset["max_open_positions"],
        "feature_coverage_minimum": conservative_preset.get("feature_coverage_minimum"),
    }
    result = apply_preset_to_decision(router, selected, "paper")
    assert result["selected_candidate_id"] == "PE_elast"
    assert result["selected_preset"] == "CONSERVATIVE"
    assert result["base_threshold"] == 0.5
    assert result["effective_threshold"] == 0.55
    assert result["threshold_adjustment"] == 0.05

def test_preset_never_enables_real_trading():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    for name, preset in policy["presets"].items():
        assert preset["trade_allowed"] is False, f"Preset {name} has trade_allowed=True — unsafe"

def test_apply_preset_blocks_when_threshold_not_met():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router = {"router_action": "TRADE", "probability": 0.45, "edge_score": 0.1,
              "threshold": 0.5, "coverage_pct": 0.98, "selected_candidate_id": "CE_elast"}
    normal_preset = policy["presets"]["NORMAL"]
    selected = {
        "preset_name": "NORMAL",
        "trade_allowed": normal_preset["trade_allowed"],
        "paper_trade_allowed": normal_preset["paper_trade_allowed"],
        "effective_threshold_adjustment": normal_preset["effective_threshold_adjustment"],
        "spread_limit_tier": normal_preset["spread_limit_tier"],
        "max_trades_per_day": normal_preset["max_trades_per_day"],
        "max_open_positions": normal_preset["max_open_positions"],
        "feature_coverage_minimum": normal_preset.get("feature_coverage_minimum"),
    }
    result = apply_preset_to_decision(router, selected, "paper")
    assert result["final_action"] == "BLOCK", "Should block when probability <= threshold"

def test_aggressive_shadow_only_not_allowed_in_paper_apply():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router = {"router_action": "TRADE", "probability": 0.70, "edge_score": 0.2,
              "threshold": 0.5, "coverage_pct": 0.95, "selected_candidate_id": "CE_elast"}
    aggressive_preset = policy["presets"]["AGGRESSIVE_SHADOW_ONLY"]
    selected = {
        "preset_name": "AGGRESSIVE_SHADOW_ONLY",
        "trade_allowed": aggressive_preset["trade_allowed"],
        "paper_trade_allowed": aggressive_preset["paper_trade_allowed"],
        "effective_threshold_adjustment": aggressive_preset["effective_threshold_adjustment"],
        "spread_limit_tier": aggressive_preset["spread_limit_tier"],
        "max_trades_per_day": aggressive_preset["max_trades_per_day"],
        "max_open_positions": aggressive_preset["max_open_positions"],
        "feature_coverage_minimum": aggressive_preset.get("feature_coverage_minimum"),
    }
    result = apply_preset_to_decision(router, selected, "paper")
    assert result["preset_trade_allowed"] is False
    assert result["final_action"] == "BLOCK"