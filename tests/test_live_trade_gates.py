"""Tests for the full live trade gate matrix (the 8+ required checks)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.config import LiveTradeGate, live_trade_allowed


def _base_gate(**overrides):
    g = LiveTradeGate(
        live_mode=True,
        order_placement_enabled=True,
        live_order_dry_run=False,
        kill_switch_active=False,
        candidate_live_whitelisted=True,
        candidate_status="LIVE_1_LOT",
        broker_session_valid=True,
        option_chain_fresh=True,
        selected_expiry_valid=True,
        current_time_le_entry_cutoff=True,
        spread_pct=0.005,
        max_spread_pct=0.02,
        premium=45.0,
        min_premium=5.0,
        stop_loss_exists=True,
        exit_rule_exists=True,
        open_positions=0,
        max_open_positions=6,
        trades_today=0,
        max_trades_per_day=2,
        daily_loss=0.0,
        max_daily_loss=1000.0,
        duplicate_open_position=False,
        order_type_is_limit=True,
    )
    for k, v in overrides.items():
        setattr(g, k, v)
    return g


def test_live_order_blocked_when_LIVE_ORDER_DRY_RUN_true():
    g = _base_gate(live_order_dry_run=True)
    ok, blockers = live_trade_allowed(g)
    assert ok is False
    assert any("LIVE_ORDER_DRY_RUN" in b for b in blockers)


def test_live_order_blocked_when_ORDER_PLACEMENT_ENABLED_false():
    g = _base_gate(order_placement_enabled=False)
    ok, blockers = live_trade_allowed(g)
    assert ok is False
    assert any("ORDER_PLACEMENT_ENABLED" in b for b in blockers)


def test_live_order_blocked_when_kill_switch_is_active():
    g = _base_gate(kill_switch_active=True)
    ok, blockers = live_trade_allowed(g)
    assert ok is False
    assert any("kill_switch" in b.lower() for b in blockers)


def test_live_order_blocked_when_candidate_status_is_not_LIVE_1_LOT():
    g = _base_gate(candidate_status="SHADOW", candidate_live_whitelisted=True)
    ok, blockers = live_trade_allowed(g)
    assert ok is False
    assert any("LIVE_1_LOT" in b or "LIVE_SCALED" in b for b in blockers)


def test_live_order_blocked_when_order_type_is_MARKET_during_first_live():
    g = _base_gate(order_type_is_limit=False)
    ok, blockers = live_trade_allowed(g)
    assert ok is False
    assert any("LIMIT" in b for b in blockers)


def test_live_order_blocked_when_non_whitelisted():
    g = _base_gate(candidate_live_whitelisted=False)
    ok, blockers = live_trade_allowed(g)
    assert ok is False
    assert any("whitelisted" in b.lower() for b in blockers)
