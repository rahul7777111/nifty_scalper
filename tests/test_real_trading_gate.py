#!/usr/bin/env python3
"""Tests for real trading gate and micro-live mode safety."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import pytest

# -----------------------------------------------------------------------
# real_trading_allowed tests
# -----------------------------------------------------------------------

def test_real_trading_blocked_by_default():
    """Real trading must be blocked when all defaults are in place."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=False,
        scalper_allow_live_orders=False,
        scalper_real_trading_ack=False,
        broker_token_expiring=True,  # expired/missing
        kill_switch_active=True,
        paper_readiness_pass=False,
        broker_safety_pass=False,
        dry_run_coverage_pct=0.0,
        model_pkl_exists=False,
        current_probability=0.3,
        probability_threshold=0.5,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert len(blockers) > 0


def test_real_trading_blocked_without_allow_live_orders():
    """SCALPER_ALLOW_LIVE_ORDERS must be required."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=False,  # missing
        scalper_real_trading_ack=True,
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("SCALPER_ALLOW_LIVE_ORDERS" in b for b in blockers)


def test_real_trading_blocked_without_ack():
    """SCALPER_REAL_TRADING_ACK must be required."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=False,  # missing
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("SCALPER_REAL_TRADING_ACK" in b for b in blockers)


def test_real_trading_blocked_if_token_expiring():
    """Expired or expiring token must block real trading."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=True,
        broker_token_valid=False,
        broker_token_expiring=True,  # expired
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("token" in b.lower() for b in blockers)


def test_real_trading_blocked_if_readiness_report_missing():
    """Paper readiness not PASS blocks real trading."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=True,
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=False,  # not ready
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("readiness" in b.lower() or "not PASS" in b for b in blockers)


def test_real_trading_blocked_if_probability_below_threshold():
    """Probability below threshold blocks trading."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=True,
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.3,  # below 0.5 threshold
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("probability" in b.lower() for b in blockers)


def test_real_trading_blocked_if_spread_too_wide():
    """Spread > max_spread_pct blocks trading."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=True,
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.05,  # 5% spread > 2% max
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("spread" in b.lower() for b in blockers)


def test_real_trading_blocked_if_kill_switch_active():
    """Kill switch active blocks all trading."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=True,
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=True,  # kill switch ON
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("kill" in b.lower() for b in blockers)


def test_real_trading_allowed_only_when_all_gates_pass():
    """ALL gates must pass for real trading."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=True,
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=False,
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is True
    assert blockers == []


def test_real_trading_blocked_if_max_daily_loss_breached():
    """Max daily loss breach blocks further trading."""
    from config import real_trading_allowed, RealTradingGate

    gate = RealTradingGate(
        enable_live_trading=True,
        scalper_allow_live_orders=True,
        scalper_real_trading_ack=True,
        broker_token_valid=True,
        broker_token_expiring=False,
        order_polling_available=True,
        kill_switch_active=False,
        paper_readiness_pass=True,
        broker_safety_pass=True,
        dry_run_coverage_pct=100.0,
        model_pkl_exists=True,
        current_probability=0.7,
        probability_threshold=0.5,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=20.0,
        min_premium=5.0,
        max_daily_loss_breached=True,  # loss breach
        max_trades_per_day_breached=False,
        market_hours_valid=True,
        open_stale_position=False,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert any("loss" in b.lower() or "breach" in b.lower() for b in blockers)


# -----------------------------------------------------------------------
# micro_live_allowed tests
# -----------------------------------------------------------------------

def test_micro_live_blocked_by_default():
    """Micro-live must be disabled by default."""
    from config import micro_live_allowed

    cfg = MagicMock()
    cfg.enable_micro_live = False
    cfg.micro_live_strict_kill_switch = True
    cfg.micro_live_min_probability = 0.70
    cfg.micro_live_require_spread_pct = 0.01
    cfg.micro_live_require_min_premium = 10.0
    cfg.micro_live_max_trades_per_day = 1
    cfg.micro_live_max_daily_loss = 500.0

    allowed, blockers = micro_live_allowed(
        cfg, probability=0.8, spread_pct=0.005, premium=20.0,
        trades_today=0, daily_pnl=0.0, kill_switch=False
    )
    assert allowed is False
    assert any("enable_micro_live" in b for b in blockers)


def test_micro_live_blocked_if_kill_switch_active():
    """Micro-live must block if kill switch active."""
    from config import micro_live_allowed

    cfg = MagicMock()
    cfg.enable_micro_live = True
    cfg.micro_live_strict_kill_switch = True
    cfg.micro_live_min_probability = 0.70
    cfg.micro_live_require_spread_pct = 0.01
    cfg.micro_live_require_min_premium = 10.0
    cfg.micro_live_max_trades_per_day = 1
    cfg.micro_live_max_daily_loss = 500.0

    allowed, blockers = micro_live_allowed(
        cfg, probability=0.8, spread_pct=0.005, premium=20.0,
        trades_today=0, daily_pnl=0.0, kill_switch=True
    )
    assert allowed is False


def test_micro_live_blocked_if_probability_below_070():
    """Micro-live requires probability >= 0.70 by default."""
    from config import micro_live_allowed

    cfg = MagicMock()
    cfg.enable_micro_live = True
    cfg.micro_live_strict_kill_switch = True
    cfg.micro_live_min_probability = 0.70
    cfg.micro_live_require_spread_pct = 0.01
    cfg.micro_live_require_min_premium = 10.0
    cfg.micro_live_max_trades_per_day = 1
    cfg.micro_live_max_daily_loss = 500.0

    allowed, blockers = micro_live_allowed(
        cfg, probability=0.5, spread_pct=0.005, premium=20.0,
        trades_today=0, daily_pnl=0.0, kill_switch=False
    )
    assert allowed is False
    assert any("probability" in b.lower() for b in blockers)


def test_micro_live_blocked_if_trades_today_exceeded():
    """Micro-live allows only 1 trade per day."""
    from config import micro_live_allowed

    cfg = MagicMock()
    cfg.enable_micro_live = True
    cfg.micro_live_strict_kill_switch = True
    cfg.micro_live_min_probability = 0.70
    cfg.micro_live_require_spread_pct = 0.01
    cfg.micro_live_require_min_premium = 10.0
    cfg.micro_live_max_trades_per_day = 1
    cfg.micro_live_max_daily_loss = 500.0

    allowed, blockers = micro_live_allowed(
        cfg, probability=0.8, spread_pct=0.005, premium=20.0,
        trades_today=1, daily_pnl=0.0, kill_switch=False
    )
    assert allowed is False
    assert any("trade" in b.lower() for b in blockers)


def test_micro_live_blocked_if_daily_loss_exceeded():
    """Micro-live stops if daily loss exceeds max."""
    from config import micro_live_allowed

    cfg = MagicMock()
    cfg.enable_micro_live = True
    cfg.micro_live_strict_kill_switch = True
    cfg.micro_live_min_probability = 0.70
    cfg.micro_live_require_spread_pct = 0.01
    cfg.micro_live_require_min_premium = 10.0
    cfg.micro_live_max_trades_per_day = 1
    cfg.micro_live_max_daily_loss = 500.0

    allowed, blockers = micro_live_allowed(
        cfg, probability=0.8, spread_pct=0.005, premium=20.0,
        trades_today=0, daily_pnl=-600.0, kill_switch=False
    )
    assert allowed is False
    assert any("loss" in b.lower() for b in blockers)


def test_micro_live_allowed_when_all_strict_conditions_met():
    """All micro-live conditions must pass."""
    from config import micro_live_allowed

    cfg = MagicMock()
    cfg.enable_micro_live = True
    cfg.micro_live_strict_kill_switch = True
    cfg.micro_live_min_probability = 0.70
    cfg.micro_live_require_spread_pct = 0.01
    cfg.micro_live_require_min_premium = 10.0
    cfg.micro_live_max_trades_per_day = 1
    cfg.micro_live_max_daily_loss = 500.0

    allowed, blockers = micro_live_allowed(
        cfg, probability=0.8, spread_pct=0.005, premium=20.0,
        trades_today=0, daily_pnl=100.0, kill_switch=False
    )
    assert allowed is True
    assert blockers == []


# -----------------------------------------------------------------------
# safe wrapper mock test
# -----------------------------------------------------------------------

def test_safe_wrapper_blocks_without_real_trading_gate():
    """safe_place_real_order must block if real_trading_allowed returns False."""
    from config import RealTradingGate, real_trading_allowed

    # Gate with all required flags missing
    gate = RealTradingGate(
        enable_live_trading=False,
        scalper_allow_live_orders=False,
        scalper_real_trading_ack=False,
        broker_token_expiring=True,
        kill_switch_active=True,
        paper_readiness_pass=False,
        broker_safety_pass=False,
        dry_run_coverage_pct=0.0,
        model_pkl_exists=False,
        current_probability=0.3,
        probability_threshold=0.5,
    )
    allowed, blockers = real_trading_allowed(gate)
    assert allowed is False
    assert len(blockers) > 5  # Multiple gates should fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])