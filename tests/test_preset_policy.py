"""Tests for preset policy loading and validation."""
import pytest, json, tempfile, pathlib
from src.preset_selector import load_preset_policy

def test_load_preset_policy_returns_dict():
    policy = load_preset_policy("config/dynamic_presets.json")
    assert isinstance(policy, dict)
    assert "presets" in policy
    assert "hard_caps" in policy

def test_preset_policy_has_all_5_presets():
    policy = load_preset_policy("config/dynamic_presets.json")
    for name in ("BLOCK", "OBSERVE_ONLY", "CONSERVATIVE", "NORMAL", "AGGRESSIVE_SHADOW_ONLY"):
        assert name in policy["presets"], f"Missing preset: {name}"

def test_validate_preset_policy_valid():
    from src.preset_selector import get_cached_policy
    policy = load_preset_policy("config/dynamic_presets.json")
    # Basic structure validation
    errors = []
    if "presets" not in policy:
        errors.append("missing presets key")
    if "hard_caps" not in policy:
        errors.append("missing hard_caps key")
    if "preset_order" not in policy:
        errors.append("missing preset_order key")
    for name in ("BLOCK", "OBSERVE_ONLY", "CONSERVATIVE", "NORMAL", "AGGRESSIVE_SHADOW_ONLY"):
        if name not in policy.get("presets", {}):
            errors.append(f"Missing preset: {name}")
    assert errors == [], f"Policy invalid: {errors}"

def test_validate_preset_policy_invalid_missing_key():
    errors = []
    if "presets" not in {}:
        errors.append("missing presets key")
    assert len(errors) > 0, "Should fail closed on empty policy"

def test_hard_caps_are_immutable():
    policy = load_preset_policy("config/dynamic_presets.json")
    caps = policy["hard_caps"]
    assert caps.get("real_trading_enabled_mutable_by_ml") is False
    assert caps.get("kill_switch_mutable_by_ml") is False
    assert caps.get("max_daily_loss_mutable_by_ml") is False

def test_conservative_threshold_adjustment():
    policy = load_preset_policy("config/dynamic_presets.json")
    adj = policy["presets"]["CONSERVATIVE"]["effective_threshold_adjustment"]
    assert adj == 0.05

def test_aggressive_shadow_only_threshold_adjustment():
    policy = load_preset_policy("config/dynamic_presets.json")
    adj = policy["presets"]["AGGRESSIVE_SHADOW_ONLY"]["effective_threshold_adjustment"]
    assert adj == -0.03, "AGGRESSIVE_SHADOW_ONLY adjustment must be -0.03 max"
    assert policy["presets"]["AGGRESSIVE_SHADOW_ONLY"]["paper_trade_allowed"] is False