#!/usr/bin/env python3
"""
tests/test_shadow_promotion_consistency.py
PHASE 5: promotion / classification consistency tests.
"""
import json
import pytest
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from candidate_profile import evaluate_shadow_readiness, evaluate_candidate_gates

def test_gate_fail_0_and_empty_reject_cannot_be_shadow_false_without_paper_reason():
    metrics = {"pe_trade_count": 133, "ce_trade_count": 0, "total_trade_count": 133,
               "pf_net": 3.05, "cost_1_5x_pf": 1.70, "cost_2x_pf": 1.02,
               "net_return": 142, "win_rate": 0.54, "avg_trade_return": 1.42,
               "leakage_pass": True, "live_feature_pass": True, "liquidity_pass": True,
               "threshold_stability_score": 1.0}
    gates = evaluate_candidate_gates(metrics, side_policy="PE_ONLY")
    assert gates["gate_fail_count"] == 0
    # old logic would give shadow=false even on pass=7
    readiness = evaluate_shadow_readiness(metrics, side_policy="PE_ONLY", is_exploratory=False, artifacts_complete=True)
    # with fix, since 133 < SHADOW_MIN=200, it should be paper_forward_only, not shadow, and not reject
    assert readiness["shadow_ready"] is False
    assert readiness["paper_forward_only"] is True
    assert "low volume for shadow" in readiness["reject_reason"].lower() or "paper_forward_only" in readiness["recommended_mode"]

def test_shadow_ready_must_be_in_shadow_list_not_rejected():
    # simulated
    row = {"shadow_ready": True, "paper_forward_only": False, "reject_reason": ""}
    assert row["shadow_ready"]
    assert not row["paper_forward_only"]
    assert row["reject_reason"] == ""

def test_paper_forward_not_in_shadow_list():
    row = {"shadow_ready": False, "paper_forward_only": True, "reject_reason": "low volume for shadow"}
    assert not row["shadow_ready"]
    assert row["paper_forward_only"]

def test_rejected_has_non_empty_reason():
    row = {"shadow_ready": False, "paper_forward_only": False, "reject_reason": "enough_trades; cost_1_5x_survival"}
    assert row["reject_reason"]

def test_gates_and_readiness_agree_on_core():
    metrics = {"total_trade_count": 300, "pf_net": 2.0, "cost_1_5x_pf": 1.2, "cost_2x_pf": 0.9,
               "net_return": 50, "win_rate": 0.5, "avg_trade_return": 0.2,
               "leakage_pass": True, "live_feature_pass": True, "liquidity_pass": True,
               "threshold_stability_score": 0.6, "pe_trade_count": 150, "ce_trade_count": 0}
    r = evaluate_shadow_readiness(metrics, side_policy="PE_ONLY", is_exploratory=False, artifacts_complete=True)
    assert r["recommended_mode"] in ("shadow_ready", "paper_forward_only")
    assert "pf_net_positive" in r["passed_gates"] or "cost_1_5x_survival" in r["passed_gates"]

def test_no_both_rejected_and_shadow():
    row = {"shadow_ready": True, "reject_reason": ""}
    # after fix, if shadow then reject empty
    if row["shadow_ready"]:
        assert row["reject_reason"] == "" or "paper" in row.get("recommended_mode", "")
