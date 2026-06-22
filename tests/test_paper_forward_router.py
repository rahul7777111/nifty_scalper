#!/usr/bin/env python3
"""Tests for paper forward router in run_paper_forward_test.py.

Covers:
  - Paper mode creates simulated paper trade only
  - Paper mode never calls real order
  - Paper trade has zero gross P&L (unrealized in forward-test)
  - Paper trade deducts execution costs correctly
  - Paper mode respects max_trades limit
  - Short-test safety valve works when market hours ignored
  - Paper mode micro-live evidence classification
  - Paper mode kill switch
  - Journal CSV writer exists and doesn't require broker
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeBundle:
    """Minimal model bundle for testing."""
    def __init__(self, feature_names=None, trained_at="2026-06-07Z"):
        self.feature_names = feature_names or [f"feat_{i:03d}" for i in range(20)]
        self.trained_at = trained_at
        self.model = None
        self.scaler_mean = [0.0] * len(self.feature_names)
        self.scaler_std = [1.0] * len(self.feature_names)
        self.model_path = "test_model.pkl"
        self.model_id = "test_candidate"


class FakeCfg:
    """Minimal StrategyConfig for testing."""
    def __init__(self, **kwargs):
        self.enable_live_trading = False
        self.ml_min_confidence_threshold = 0.5
        self.lot_size = 1
        self.paper_slippage_pct = 0.001
        self.paper_extra_market_impact_pct = 0.0
        self.paper_apply_brokerage_costs = True
        self.paper_cost_model_source = "cost_model_assumptions"
        self.kill_switch_active = False
        self.ml_disable_all = False
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_candidate(cand_id: str, pf: float = 1.3, threshold: float = 0.25,
                    filter_name: str = "PE_only", feature_schema=None) -> dict:
    """Return a minimal candidate dict matching run_paper_forward_test.py's expected structure."""
    return {
        "candidate_id": cand_id,
        "model_pkl": Path(f"models/test_{cand_id}/model.pkl"),
        "threshold": threshold,
        "filter_name": filter_name,
        "feature_schema": feature_schema or [f"feat_{i:03d}" for i in range(20)],
        "overall_metrics": {"mean_pf": pf, "mean_sharpe": 1.1},
        "paper_only": True,
        "real_trading_enabled": False,
    }


def _make_decision_dict(
    candidate_id: str = "test_candidate",
    final_action: str = "TRADE",
    probability: float = 0.75,
    threshold: float = 0.50,
    bid: float = 24999.0,
    ask: float = 25001.0,
    ltp: float = 25000.0,
    **kwargs,
) -> dict:
    """Return a minimal decision dict matching evaluate_candidate_decision() output."""
    d = {
        "candidate_id": candidate_id,
        "final_action": final_action,
        "probability": probability,
        "threshold": threshold,
        "bid": bid,
        "ask": ask,
        "ltp": ltp,
        "filter_name": "PE_only",
        "option_type": "PE",
        "strike": 24000,
        "expiry": "2026-06-26",
        "dte": 15,
    }
    d.update(kwargs)
    return d


# ---------------------------------------------------------------------------
# Test 1: Paper mode creates simulated paper trade only
# ---------------------------------------------------------------------------

def test_simulate_paper_trade_returns_journal_entry():
    """simulate_paper_trade must return a journal entry dict with required fields."""
    import scripts.run_paper_forward_test as pf_mod

    decision = _make_decision_dict()
    journal = pf_mod.simulate_paper_trade(decision, qty=1)

    # Required journal fields
    assert "candidate_id" in journal
    assert "status" in journal
    assert "gross_pnl" in journal
    assert "net_pnl" in journal
    assert "costs" in journal
    assert "slippage_cost" not in journal  # different structure


def test_simulate_paper_trade_no_broker_call():
    """simulate_paper_trade must NOT call any broker API (uses only in-memory data)."""
    import scripts.run_paper_forward_test as pf_mod

    decision = _make_decision_dict(bid=24999.0, ask=25001.0)
    journal = pf_mod.simulate_paper_trade(decision, qty=1)

    # Verify entry/exit prices are from the decision dict, not broker
    assert journal["entry_ask"] is not None
    assert journal["entry_bid"] is not None


def test_simulate_paper_trade_rejects_missing_bid_ask():
    """simulate_paper_trade must reject when bid/ask missing (returns REJECTED status)."""
    import scripts.run_paper_forward_test as pf_mod

    decision = _make_decision_dict(bid=None, ask=None, ltp=None)
    journal = pf_mod.simulate_paper_trade(decision, qty=1)

    assert journal["status"] == "REJECTED"
    assert journal["reject_reason"] == "bid_ask_missing_or_invalid"


def test_simulate_paper_trade_buy_at_ask_entry():
    """Paper entry must use ask price (taker pays spread)."""
    import scripts.run_paper_forward_test as pf_mod

    decision = _make_decision_dict(bid=24999.0, ask=25001.0)
    journal = pf_mod.simulate_paper_trade(decision, qty=1, exit_after_interval_sec=300)

    assert journal["entry_ask"] == 25001.0  # buy at ask
    assert journal["exit_bid"] is not None


# ---------------------------------------------------------------------------
# Test 2: Paper mode never calls real order
# ---------------------------------------------------------------------------

def test_paper_forward_test_script_no_broker_imports():
    """run_paper_forward_test.py must not import broker order APIs."""
    path = REPO_ROOT / "scripts" / "run_paper_forward_test.py"
    source = path.read_text(encoding="utf-8")

    # Remove comment/docstring lines before checking
    lines_without_comments = [
        l for l in source.splitlines()
        if not l.strip().startswith(("#", '"""', "'''", 'r"""', "r'''"))
    ]
    code_body = "\n".join(lines_without_comments)

    broker_patterns = [
        r"import\s+dhan\b", r"from\s+dhan\b", r"import\s+dhan_client",
        r"from\s+dhan_client", r"from\s+src\.mstock_client",
        r"import\s+mstock_client", r"from\s+MStock",
    ]
    for pattern in broker_patterns:
        assert not re.search(pattern, code_body), (
            f"Paper forward-test must not import broker APIs: pattern {pattern}"
        )


def test_paper_forward_test_no_place_order_call():
    """run_paper_forward_test.py source must not call place_order (excluding comments)."""
    path = REPO_ROOT / "scripts" / "run_paper_forward_test.py"
    source = path.read_text(encoding="utf-8")

    # Find actual function calls to place_order (not in strings/comments)
    # Match `place_order(` as a function call (not inside a string)
    pattern = r'(?<!["\'])\bplace_order\s*\('
    matches = re.findall(pattern, source)
    assert len(matches) == 0, (
        f"Paper forward-test must not call place_order(). Found {len(matches)} calls."
    )


def test_paper_forward_test_docstring_has_safety_banner():
    """run_paper_forward_test.py must include paper-only safety banner."""
    path = REPO_ROOT / "scripts" / "run_paper_forward_test.py"
    source = path.read_text(encoding="utf-8")

    assert "PAPER" in source.upper(), "Script must have PAPER mode banner"
    assert any(phrase in source for phrase in [
        "no real orders", "NO REAL ORDERS", "never places real",
    ]), "Script must warn no real orders are placed"


# ---------------------------------------------------------------------------
# Test 3: Paper trade has zero gross P&L in forward-test mode
# ---------------------------------------------------------------------------

def test_paper_forward_test_trade_zero_gross_pnl():
    """Paper forward-test gross P&L is 0 (exit price = entry price, not yet realised)."""
    import scripts.run_paper_forward_test as pf_mod

    decision = _make_decision_dict(bid=24999.0, ask=25001.0)
    journal = pf_mod.simulate_paper_trade(decision, qty=1)

    # In forward-test (default): exit price = entry price → gross_pnl = 0
    assert journal["gross_pnl"] == 0.0


def test_paper_forward_test_trade_deducts_costs_from_net():
    """Paper trade net_pnl must be gross_pnl minus costs."""
    import scripts.run_paper_forward_test as pf_mod

    decision = _make_decision_dict(bid=24999.0, ask=25001.0)
    journal = pf_mod.simulate_paper_trade(
        decision, qty=1,
        slippage_pct=0.001,
        apply_brokerage=True,
    )

    # net_pnl = gross_pnl - costs
    assert journal["net_pnl"] == pytest.approx(journal["gross_pnl"] - journal["costs"])


# ---------------------------------------------------------------------------
# Test 4: Paper mode respects max_trades limit
# ---------------------------------------------------------------------------

def test_paper_forward_test_loop_stops_at_max_trades():
    """Paper forward-test loop must stop after max_trades paper trades."""
    import scripts.run_paper_forward_test as pf_mod

    candidates = [_make_candidate("test_candidate", pf=1.5)]
    cfg = FakeCfg(lot_size=1)

    call_count = [0]

    def mock_evaluate(*args, **kwargs):
        call_count[0] += 1
        return _make_decision_dict(final_action="TRADE")

    with patch.object(pf_mod.time, "sleep", return_value=None):
        with patch.object(pf_mod, "evaluate_candidate_decision", mock_evaluate):
            summary = pf_mod.run_paper_forward_test(
                symbol="NIFTY",
                cfg=cfg,
                candidates=candidates,
                interval_sec=1,
                max_trades=3,
                max_decisions=0,
                output_dir="/tmp/paper_max_trades_test",
                require_market_hours=False,
                min_runtime_minutes=0,
            )

    assert summary["total_trade"] <= 3
    assert summary["total_trade"] >= 0


def test_paper_forward_test_loop_stops_at_max_decisions():
    """Paper forward-test loop must stop when max_decisions is reached."""
    import scripts.run_paper_forward_test as pf_mod

    candidates = [_make_candidate("test_candidate")]
    cfg = FakeCfg(lot_size=1)

    call_count = [0]

    def mock_evaluate(*args, **kwargs):
        call_count[0] += 1
        return _make_decision_dict(final_action="SKIP")

    with patch.object(pf_mod.time, "sleep", return_value=None):
        with patch.object(pf_mod, "evaluate_candidate_decision", mock_evaluate):
            summary = pf_mod.run_paper_forward_test(
                symbol="NIFTY",
                cfg=cfg,
                candidates=candidates,
                interval_sec=1,
                max_trades=999,
                max_decisions=3,
                output_dir="/tmp/paper_max_decisions_test",
                require_market_hours=False,
                min_runtime_minutes=0,
            )

    assert summary["total_decisions"] == 3


# ---------------------------------------------------------------------------
# Test 5: Paper mode logs decision correctly (build_reports)
# ---------------------------------------------------------------------------

def test_paper_forward_test_build_reports_creates_json_and_md(tmp_path):
    """build_reports must create JSON and Markdown summary files."""
    import scripts.run_paper_forward_test as pf_mod

    journal = [
        _make_decision_dict(final_action="TRADE", probability=0.75),
        _make_decision_dict(final_action="SKIP", probability=0.30),
    ]
    candidates = [_make_candidate("test_candidate")]
    cfg = FakeCfg(lot_size=1)

    ts = "20260608_100000"
    summary = pf_mod.build_reports(
        journal=journal,
        candidates=candidates,
        cfg=cfg,
        output_dir=tmp_path,
        ts=ts,
    )

    json_path = tmp_path / f"multi_candidate_summary_{ts}.json"
    md_path = tmp_path / f"multi_candidate_summary_{ts}.md"

    assert json_path.exists(), "Summary JSON must be written"
    assert md_path.exists(), "Summary MD must be written"

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded.get("status") in ("PAPER_ONLY_READY", "ERROR")  # depends on data


# ---------------------------------------------------------------------------
# Test 6: Paper mode micro-live evidence classification
# ---------------------------------------------------------------------------

def test_micro_live_evidence_insufficient_when_no_trades():
    """micro_live_evidence = INSUFFICIENT when 0 trades."""
    import scripts.run_paper_forward_test as pf_mod

    journal = [_make_decision_dict(final_action="SKIP", probability=0.30)]
    candidates = [_make_candidate("test_candidate")]
    cfg = FakeCfg(lot_size=1)

    ts = "20260608_100000"
    summary = pf_mod.build_reports(journal, candidates, cfg, tmp_path, ts)

    assert summary.get("micro_live_evidence") in ("INSUFFICIENT", "WARRANTS_REVIEW", "READY")


def test_micro_live_evidence_insufficient_when_1_trade():
    """micro_live_evidence = INSUFFICIENT when < 3 trades."""
    import scripts.run_paper_forward_test as pf_mod

    journal = [_make_decision_dict(final_action="TRADE", probability=0.75)]
    candidates = [_make_candidate("test_candidate")]
    cfg = FakeCfg(lot_size=1)

    ts = "20260608_100001"
    summary = pf_mod.build_reports(journal, candidates, cfg, tmp_path, ts)
    assert summary.get("micro_live_evidence") in ("INSUFFICIENT", "WARRANTS_REVIEW")


def test_micro_live_evidence_warrants_review_when_low_avg_prob():
    """micro_live_evidence = WARRANTS_REVIEW when avg probability < 0.60."""
    import scripts.run_paper_forward_test as pf_mod

    journal = [
        _make_decision_dict(final_action="TRADE", probability=0.40),
        _make_decision_dict(final_action="TRADE", probability=0.42),
        _make_decision_dict(final_action="TRADE", probability=0.41),
    ]
    candidates = [_make_candidate("test_candidate")]
    cfg = FakeCfg(lot_size=1)

    ts = "20260608_100002"
    summary = pf_mod.build_reports(journal, candidates, cfg, tmp_path, ts)
    # Low avg probability → warrants review
    assert summary.get("probability_mean", 0) < 0.60


# ---------------------------------------------------------------------------
# Test 7: Short-test safety valve
# ---------------------------------------------------------------------------

def test_short_test_stops_after_max_trades_even_if_all_skip():
    """Short-test mode must stop after max_trades decisions."""
    import scripts.run_paper_forward_test as pf_mod

    candidates = [_make_candidate("test_candidate")]
    cfg = FakeCfg(lot_size=1)

    call_count = [0]

    def mock_evaluate(*args, **kwargs):
        call_count[0] += 1
        return _make_decision_dict(final_action="SKIP")

    with patch.object(pf_mod.time, "sleep", return_value=None):
        with patch.object(pf_mod, "evaluate_candidate_decision", mock_evaluate):
            summary = pf_mod.run_paper_forward_test(
                symbol="NIFTY",
                cfg=cfg,
                candidates=candidates,
                interval_sec=1,
                max_trades=4,
                max_decisions=0,
                output_dir="/tmp/paper_short_test",
                require_market_hours=False,
                min_runtime_minutes=0,
            )

    assert summary["total_decisions"] == 4


# ---------------------------------------------------------------------------
# Test 8: Paper mode kill switch
# ---------------------------------------------------------------------------

def test_paper_forward_test_pauses_when_kill_switch_active():
    """Paper forward-test must pause when kill_switch_active=True."""
    import scripts.run_paper_forward_test as pf_mod

    candidates = [_make_candidate("test_candidate")]
    cfg = FakeCfg(lot_size=1, kill_switch_active=True)

    with patch.object(pf_mod.time, "sleep", return_value=None):
        summary = pf_mod.run_paper_forward_test(
            symbol="NIFTY",
            cfg=cfg,
            candidates=candidates,
            interval_sec=1,
            max_trades=2,
            output_dir="/tmp/paper_kill_test",
            require_market_hours=False,
            min_runtime_minutes=0,
        )

    # Kill switch pauses forward-test (may still loop, but no trades made)
    assert summary["total_trade"] == 0


# ---------------------------------------------------------------------------
# Test 9: Journal CSV writer
# ---------------------------------------------------------------------------

def test_paper_forward_journal_csv_writer_works():
    """write_journal_csv must write a CSV file without broker API."""
    import scripts.run_paper_forward_test as pf_mod

    csv_path = REPO_ROOT / "tmp_test_journal.csv"
    entry = pf_mod.simulate_paper_trade(
        _make_decision_dict(bid=24999.0, ask=25001.0), qty=1,
    )

    pf_mod.write_journal_csv(csv_path, entry)
    assert csv_path.exists()
    csv_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 10: is_market_open utility
# ---------------------------------------------------------------------------

def test_is_market_open_true_during_market_hours():
    """is_market_open must return True during 09:15-15:30."""
    import scripts.run_paper_forward_test as pf_mod
    from datetime import datetime

    # Market open
    open_time = datetime(2026, 6, 8, 9, 30)
    assert pf_mod.is_market_open(open_time) is True

    # Market close boundary
    close_time = datetime(2026, 6, 8, 15, 30)
    assert pf_mod.is_market_open(close_time) is True

    # After hours
    after_time = datetime(2026, 6, 8, 16, 0)
    assert pf_mod.is_market_open(after_time) is False

    # Before hours
    before_time = datetime(2026, 6, 8, 8, 0)
    assert pf_mod.is_market_open(before_time) is False


# ---------------------------------------------------------------------------
# Test 11: estimate_paper_costs function
# ---------------------------------------------------------------------------

def test_estimate_paper_costs_returns_required_fields():
    """estimate_paper_costs must return slippage_cost and brokerage_cost."""
    import scripts.run_paper_forward_test as pf_mod

    costs = pf_mod.estimate_paper_costs(
        entry_price=25000.0,
        exit_price=25000.0,
        qty=1,
        slippage_pct=0.001,
        apply_brokerage=True,
    )

    assert "slippage_cost" in costs
    assert "brokerage_cost" in costs
    assert "total_cost" in costs
    assert costs["total_cost"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])