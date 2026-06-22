"""Comprehensive tests for live_order_guard.validate_before_order hard blocks."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.live_order_guard import OrderIntent, validate_before_order
from src.candidate_lifecycle import Candidate, STATUS_SHADOW, STATUS_LIVE_1_LOT


def _intent(**kw):
    base = dict(candidate_id="test_cand", symbol="NIFTY24JUN24100PE", exchange="NFO", symbol_token="123",
                side="SELL", quantity=65, order_type="LIMIT", price=42.5, expiry="2024-06-24",
                option_type="PE", spread_pct=0.008, premium=42.5, ltp=42.5)
    base.update(kw)
    return OrderIntent(**base)


def test_order_guard_blocks_when_kill_switch_active():
    res = validate_before_order(_intent(), force_kill_switch=True)
    assert res.allowed is False
    assert any("KILL_SWITCH" in b for b in res.blockers)


def test_order_guard_blocks_when_live_dry_run_is_true():
    res = validate_before_order(_intent(), force_dry_run=True)
    assert res.allowed is False
    assert any("LIVE_ORDER_DRY_RUN" in b for b in res.blockers)


def test_order_guard_blocks_when_order_placement_false(monkeypatch):
    monkeypatch.setenv("ORDER_PLACEMENT_ENABLED", "false")
    res = validate_before_order(_intent())
    assert res.allowed is False
    assert any("ORDER_PLACEMENT" in b for b in res.blockers)


def test_order_guard_blocks_when_candidate_not_whitelisted():
    # empty whitelist + no candidate passed
    res = validate_before_order(_intent(candidate_id="nonexistent"))
    assert res.allowed is False
    assert any("whitelisted" in b.lower() for b in res.blockers)


def test_order_guard_blocks_when_candidate_status_is_SHADOW():
    c = Candidate(candidate_id="sh", status=STATUS_SHADOW, live_whitelisted=False, max_qty_lots=0)
    res = validate_before_order(_intent(), candidate=c)
    assert res.allowed is False
    assert any("LIVE_1_LOT" in b or "LIVE_SCALED" in b for b in res.blockers)


def test_order_guard_blocks_when_quantity_exceeds_1_lot():
    c = Candidate(candidate_id="l1", status=STATUS_LIVE_1_LOT, live_whitelisted=True, max_qty_lots=1)
    res = validate_before_order(_intent(quantity=130), candidate=c)
    assert res.allowed is False
    assert any("quantity" in b.lower() or "exceeds" in b.lower() for b in res.blockers)


def test_order_guard_blocks_MARKET_order_during_first_live():
    c = Candidate(candidate_id="l1", status=STATUS_LIVE_1_LOT, live_whitelisted=True, max_qty_lots=1)
    res = validate_before_order(_intent(order_type="MARKET"), candidate=c)
    assert res.allowed is False
    assert any("LIMIT" in b for b in res.blockers)


def test_order_guard_blocks_missing_stop_loss():
    c = Candidate(candidate_id="l1", status=STATUS_LIVE_1_LOT, live_whitelisted=True, max_qty_lots=1)
    res = validate_before_order(_intent(stop_loss=None), candidate=c, runtime_state={"has_sl": False})
    assert res.allowed is False
    assert any("stop loss" in b.lower() for b in res.blockers)


def test_order_guard_blocks_stale_option_chain():
    c = Candidate(candidate_id="l1", status=STATUS_LIVE_1_LOT, live_whitelisted=True, max_qty_lots=1)
    res = validate_before_order(_intent(), candidate=c, runtime_state={"option_chain_fresh": False})
    assert res.allowed is False
    assert any("stale" in b.lower() for b in res.blockers)


def test_order_guard_blocks_after_cutoff():
    c = Candidate(candidate_id="l1", status=STATUS_LIVE_1_LOT, live_whitelisted=True, max_qty_lots=1)
    res = validate_before_order(_intent(), candidate=c, runtime_state={"before_cutoff": False})
    assert res.allowed is False
    assert any("cutoff" in b.lower() for b in res.blockers)
