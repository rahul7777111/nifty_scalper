"""Tests for router_with_presets integration."""
import pytest
from unittest.mock import patch, MagicMock
from src.router_with_presets import route_with_presets, invalidate_preset_cache

def test_route_with_presets_includes_preset_fields():
    invalidate_preset_cache()
    snapshot = {}
    manifests = []
    live_features = {"feature_coverage_pct": 0.98, "stale_quote": False, "spread_pct": 0.001,
                     "bid": 100, "ask": 101}
    
    # Mock the candidate router to return a known result
    mock_router_result = {
        "router_action": "TRADE",
        "probability": 0.6,
        "edge_score": 0.1,
        "threshold": 0.5,
        "coverage_pct": 0.98,
        "selected_candidate_id": "PE_elast",
    }
    
    with patch('src.router_with_presets.candidate_router') as mock_candidate_router:
        mock_candidate_router.route_candidates.return_value = mock_router_result
        result = route_with_presets(snapshot, manifests, live_features, mode="paper")
    
    assert "selected_preset" in result or "preset_name" in result
    assert "effective_threshold" in result
    assert "base_threshold" in result
    assert "threshold_adjustment" in result
    # Check the key is present (API uses preset_name internally)
    preset_key = "selected_preset" if "selected_preset" in result else "preset_name"
    assert result.get(preset_key) is not None

def test_route_with_presets_shadow_mode():
    invalidate_preset_cache()
    snapshot = {}
    manifests = []
    live_features = {"feature_coverage_pct": 0.95, "stale_quote": False, "spread_pct": 0.001,
                     "bid": 100, "ask": 101}
    
    mock_router_result = {
        "router_action": "TRADE",
        "probability": 0.70,
        "edge_score": 0.20,
        "threshold": 0.5,
        "coverage_pct": 0.95,
        "selected_candidate_id": "CE_elast",
    }
    
    with patch('src.router_with_presets.candidate_router') as mock_candidate_router:
        mock_candidate_router.route_candidates.return_value = mock_router_result
        result = route_with_presets(snapshot, manifests, live_features, mode="shadow", market_quality="good")
    
    preset_key = "selected_preset" if "selected_preset" in result else "preset_name"
    assert result.get(preset_key) == "AGGRESSIVE_SHADOW_ONLY"

def test_route_with_presets_paper_mode_not_aggressive():
    invalidate_preset_cache()
    snapshot = {}
    manifests = []
    live_features = {"feature_coverage_pct": 0.95, "stale_quote": False, "spread_pct": 0.001,
                     "bid": 100, "ask": 101}
    
    mock_router_result = {
        "router_action": "TRADE",
        "probability": 0.70,
        "edge_score": 0.20,
        "threshold": 0.5,
        "coverage_pct": 0.95,
        "selected_candidate_id": "CE_elast",
    }
    
    with patch('src.router_with_presets.candidate_router') as mock_candidate_router:
        mock_candidate_router.route_candidates.return_value = mock_router_result
        result = route_with_presets(snapshot, manifests, live_features, mode="paper", market_quality="good")
    
    preset_key = "selected_preset" if "selected_preset" in result else "preset_name"
    assert result.get(preset_key) != "AGGRESSIVE_SHADOW_ONLY"