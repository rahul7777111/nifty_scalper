"""Tests for hard risk firewall."""
import pytest
from src.risk_firewall import HardRiskFirewall

def test_firewall_blocks_live_mode():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "NORMAL", "preset_trade_allowed": True,
                "effective_threshold": 0.5, "probability": 0.6, "max_trades_per_day": 3,
                "max_open_positions": 1, "spread_limit_tier": "normal"}
    snapshot = {"real_trading_enabled": False, "broker_place_order_allowed": False,
                "kill_switch_active": False, "daily_pnl": 0, "max_daily_loss_hard_cap": -100,
                "open_positions_count": 0, "trades_today": 0}
    live_features = {"feature_coverage": 0.98, "bid": 100, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.001}
    result = fw.check(decision, snapshot, live_features, mode="live")
    assert result["hard_risk_allowed"] is False
    assert "live mode" in result["hard_block_reason"].lower()

def test_firewall_blocks_real_trading_enabled():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "NORMAL", "preset_trade_allowed": True,
                "effective_threshold": 0.5, "probability": 0.6, "max_trades_per_day": 3,
                "max_open_positions": 1, "spread_limit_tier": "normal"}
    snapshot = {"real_trading_enabled": True}  # DANGEROUS
    result = fw.check(decision, snapshot, {}, mode="shadow")
    assert result["hard_risk_allowed"] is False

def test_firewall_blocks_kill_switch():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "NORMAL", "preset_trade_allowed": True,
                "effective_threshold": 0.5, "probability": 0.6, "max_trades_per_day": 3,
                "max_open_positions": 1, "spread_limit_tier": "normal"}
    snapshot = {"real_trading_enabled": False, "kill_switch_active": True}
    result = fw.check(decision, snapshot, {}, mode="shadow")
    assert result["hard_risk_allowed"] is False

def test_firewall_blocks_max_trades():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "CONSERVATIVE", "preset_trade_allowed": True,
                "effective_threshold": 0.5, "probability": 0.6, "max_trades_per_day": 1,
                "max_open_positions": 1, "spread_limit_tier": "tight"}
    snapshot = {"real_trading_enabled": False, "kill_switch_active": False,
                "trades_today": 1}  # already at limit
    result = fw.check(decision, snapshot, {}, mode="paper")
    assert result["hard_risk_allowed"] is False
    assert "max trades" in result["hard_block_reason"].lower()

def test_firewall_blocks_missing_bid_in_paper():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "NORMAL", "preset_trade_allowed": True,
                "effective_threshold": 0.5, "probability": 0.6, "max_trades_per_day": 3,
                "max_open_positions": 1, "spread_limit_tier": "normal"}
    snapshot = {"real_trading_enabled": False, "kill_switch_active": False}
    live_features = {"feature_coverage": 0.98, "bid": None, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.001}
    result = fw.check(decision, snapshot, live_features, mode="paper")
    assert result["hard_risk_allowed"] is False
    assert "bid" in result["hard_block_reason"].lower()

def test_firewall_allows_good_decision():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "NORMAL", "preset_trade_allowed": True,
                "effective_threshold": 0.5, "probability": 0.6, "max_trades_per_day": 3,
                "max_open_positions": 1, "spread_limit_tier": "normal",
                "feature_coverage_minimum": 0.95, "selected_candidate_id": "CE_elast"}
    snapshot = {"real_trading_enabled": False, "broker_place_order_allowed": False,
                "kill_switch_active": False, "daily_pnl": 0,
                "max_daily_loss_hard_cap": -100, "open_positions_count": 0,
                "trades_today": 0}
    live_features = {"feature_coverage": 0.98, "bid": 100, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.001}
    result = fw.check(decision, snapshot, live_features, mode="paper")
    assert result["hard_risk_allowed"] is True
    assert result["final_action"] == "TRADE"

def test_firewall_blocks_stale_quote():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "NORMAL", "preset_trade_allowed": True,
                "effective_threshold": 0.5, "probability": 0.6, "max_trades_per_day": 3,
                "max_open_positions": 1, "spread_limit_tier": "normal",
                "feature_coverage_minimum": 0.95}
    snapshot = {"real_trading_enabled": False, "broker_place_order_allowed": False,
                "kill_switch_active": False, "daily_pnl": 0}
    live_features = {"feature_coverage": 0.98, "bid": 100, "ask": 101,
                     "stale_quote": True, "spread_pct": 0.001}
    result = fw.check(decision, snapshot, live_features, mode="paper")
    assert result["hard_risk_allowed"] is False
    assert "stale" in result["hard_block_reason"].lower()

def test_firewall_blocks_low_coverage():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "CONSERVATIVE", "preset_trade_allowed": True,
                "effective_threshold": 0.55, "probability": 0.6, "max_trades_per_day": 1,
                "max_open_positions": 1, "spread_limit_tier": "tight",
                "feature_coverage_minimum": 0.97}
    snapshot = {"real_trading_enabled": False, "broker_place_order_allowed": False,
                "kill_switch_active": False, "daily_pnl": 0}
    live_features = {"feature_coverage": 0.80, "bid": 100, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.001}
    result = fw.check(decision, snapshot, live_features, mode="paper")
    assert result["hard_risk_allowed"] is False
    assert "coverage" in result["hard_block_reason"].lower()

def test_firewall_blocks_wide_spread_tight_tier():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "CONSERVATIVE", "preset_trade_allowed": True,
                "effective_threshold": 0.55, "probability": 0.6, "max_trades_per_day": 1,
                "max_open_positions": 1, "spread_limit_tier": "tight",
                "feature_coverage_minimum": 0.97}
    snapshot = {"real_trading_enabled": False, "broker_place_order_allowed": False,
                "kill_switch_active": False, "daily_pnl": 0}
    live_features = {"feature_coverage": 0.98, "bid": 100, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.005}  # wider than tight tier
    result = fw.check(decision, snapshot, live_features, mode="paper")
    assert result["hard_risk_allowed"] is False
    assert "spread" in result["hard_block_reason"].lower()

def test_firewall_blocks_aggressive_shadow_in_paper():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "AGGRESSIVE_SHADOW_ONLY", "preset_trade_allowed": True,
                "effective_threshold": 0.47, "probability": 0.70, "max_trades_per_day": 5,
                "max_open_positions": 2, "spread_limit_tier": "wide",
                "feature_coverage_minimum": 0.93, "selected_candidate_id": "CE_elast"}
    snapshot = {"real_trading_enabled": False, "broker_place_order_allowed": False,
                "kill_switch_active": False, "daily_pnl": 0}
    live_features = {"feature_coverage": 0.95, "bid": 100, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.001}
    result = fw.check(decision, snapshot, live_features, mode="paper")
    assert result["hard_risk_allowed"] is False
    assert "aggressive" in result["hard_block_reason"].lower()

def test_firewall_blocks_probability_below_threshold():
    fw = HardRiskFirewall()
    decision = {"selected_preset": "NORMAL", "preset_trade_allowed": True,
                "effective_threshold": 0.6, "probability": 0.5, "max_trades_per_day": 3,
                "max_open_positions": 1, "spread_limit_tier": "normal",
                "feature_coverage_minimum": 0.95}
    snapshot = {"real_trading_enabled": False, "broker_place_order_allowed": False,
                "kill_switch_active": False, "daily_pnl": 0}
    live_features = {"feature_coverage": 0.98, "bid": 100, "ask": 101,
                     "stale_quote": False, "spread_pct": 0.001}
    result = fw.check(decision, snapshot, live_features, mode="paper")
    assert result["hard_risk_allowed"] is False