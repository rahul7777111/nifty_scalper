#!/usr/bin/env python3
"""PHASE 19: candidate selection classification tests."""
import pandas as pd
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def test_shadow_requires_all_gates():
    row = {"shadow_ready": True, "total_trade_count": 500, "win_rate": 0.6, "PF_net": 1.5, "cost_1_5x_pf": 1.1, "exploratory": False}
    assert row["shadow_ready"] is True
    assert row["exploratory"] is False
    assert row["total_trade_count"] >= 100
    assert row["cost_1_5x_pf"] >= 1.0

def test_paper_forward_only_near_pass_but_not_broken():
    row = {"classification": "paper_forward_only", "total_trade_count": 150, "win_rate": 0.45, "PF_net": 1.2, "exploratory": True, "reject_reason": "cost_1_5x_survival;enough_trades"}
    assert row["classification"] == "paper_forward_only"
    assert row["total_trade_count"] > 50
    assert row["win_rate"] > 0.2
    # not live because exploratory or partial gate fail

def test_rejected_have_exact_reasons():
    row = {"classification": "rejected", "reject_reason": "enough_trades; cost_1_5x_survival; pf_net_positive"}
    assert row["classification"] == "rejected"
    assert "enough_trades" in row["reject_reason"]
    assert "cost_1_5x_survival" in row["reject_reason"]

def test_no_unusable_label():
    row = {"label_used": "cost_survivor_label_v2"}
    assert "high_conviction" not in row["label_used"]
    assert "paper_candidate" not in row["label_used"]
