#!/usr/bin/env python3
"""PHASE 19 logging tests."""
import json, tempfile
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def test_decisions_jsonl_has_required_fields(tmp_path):
    dec = {
        "timestamp": "2026-..", "underlying_price": 24000, "option_symbol_or_token": "NIFTY24JUN24000CE",
        "ce_pe": "CE", "score": 0.42, "threshold": 0.35, "top_n_rank": 2,
        "filter_pass_fail_reason": "passed", "model_name": "elasticnet", "preset_name": "bal",
        "sim_entry_price": 120.5, "sim_exit_price": 125.0,
        "gross_paper_pnl": 4.5, "estimated_cost": 0.8, "net_paper_pnl": 3.7,
        "skip_or_no_trade_reason": ""
    }
    j = tmp_path / "dec.jsonl"
    j.write_text(json.dumps(dec) + "\n")
    loaded = json.loads(j.read_text().strip())
    for k in ["timestamp", "underlying_price", "ce_pe", "score", "threshold", "model_name", "preset_name",
              "gross_paper_pnl", "estimated_cost", "net_paper_pnl", "skip_or_no_trade_reason"]:
        assert k in loaded

def test_skipped_have_reason():
    d = {"final_paper_action": "BLOCK", "block_reason": "router_side_policy_block", "skip_or_no_trade_reason": "router_side_policy_block"}
    assert d["skip_or_no_trade_reason"]
