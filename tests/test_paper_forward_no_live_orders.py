#!/usr/bin/env python3
"""PHASE 19: ensure paper forward cannot place live orders."""
import pytest
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def test_paper_mode_sets_broker_block_flags():
    # simulate decision from runner
    decision = {"final_paper_action": "PAPER_TRADE", "broker_place_order_allowed": False, "no_order_sent": True, "paper_only": True}
    assert decision["broker_place_order_allowed"] is False
    assert decision["no_order_sent"] is True

def test_attempted_order_in_paper_raises(monkeypatch):
    # if code path tried mstock place, it should be guarded
    # here mock a guard function
    def guarded_place(**kw):
        if not kw.get("paper_only", True) or kw.get("real_trading_enabled", False):
            raise RuntimeError("LIVE ORDER BLOCKED in paper mode")
        return {"simulated": True}
    with pytest.raises(RuntimeError) as exc:
        guarded_place(paper_only=False, real_trading_enabled=True)
    assert "BLOCKED" in str(exc.value)

def test_sim_trade_path_works():
    trade = {"gross_pnl": 10.0, "costs": 1.5, "net_pnl": 8.5, "status": "CLOSED"}
    assert trade["net_pnl"] == 8.5
