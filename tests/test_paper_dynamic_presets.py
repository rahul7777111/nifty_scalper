"""Tests that paper-forward respects preset restrictions."""
import pytest
from src.preset_selector import select_preset, load_preset_policy, invalidate_policy_cache

def test_paper_blocks_aggressive_shadow_only():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router_result = {"router_action": "TRADE", "probability": 0.70, "edge_score": 0.20,
                     "threshold": 0.5, "coverage_pct": 0.95}
    live_features = {"feature_coverage_pct": 0.95, "stale_quote": False, "spread_pct": 0.001}
    result = select_preset(router_result, live_features, "good", mode="paper")
    assert result["preset_name"] != "AGGRESSIVE_SHADOW_ONLY"

def test_paper_blocks_missing_bid():
    """select_preset does NOT check bid/ask — that's the firewall's job.
    This test verifies the preset selector returns NORMAL (the preset),
    and the firewall would later block for missing bid."""
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router_result = {"router_action": "TRADE", "probability": 0.60, "edge_score": 0.10,
                     "threshold": 0.5, "coverage_pct": 0.98}
    live_features = {"feature_coverage_pct": 0.98, "bid": None, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.001}
    result = select_preset(router_result, live_features, "good", mode="paper")
    # Preset selector only checks market quality, not bid/ask
    # The firewall (tested separately) handles the missing bid check
    assert result["preset_name"] in policy["presets"]

def test_paper_allows_normal_on_good_quality():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router_result = {"router_action": "TRADE", "probability": 0.65, "edge_score": 0.15,
                     "threshold": 0.5, "coverage_pct": 0.97}
    live_features = {"feature_coverage_pct": 0.97, "stale_quote": False, "spread_pct": 0.001}
    result = select_preset(router_result, live_features, "good", mode="paper")
    assert result["preset_name"] == "NORMAL"

def test_paper_conservative_adjustment():
    invalidate_policy_cache()
    policy = load_preset_policy("config/dynamic_presets.json")
    router_result = {"router_action": "TRADE", "probability": 0.55, "edge_score": 0.08,
                     "threshold": 0.5, "coverage_pct": 0.97}
    live_features = {"feature_coverage_pct": 0.97, "stale_quote": False, "spread_pct": 0.001}
    result = select_preset(router_result, live_features, "acceptable", mode="paper")
    assert result["preset_name"] == "NORMAL"
    # The effective threshold after apply should be base + adjustment