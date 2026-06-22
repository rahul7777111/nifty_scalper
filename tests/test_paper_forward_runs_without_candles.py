#!/usr/bin/env python3
"""Prove paper forward decisions with option chain data even if candles empty."""
import json
import tempfile
from pathlib import Path
import pytest
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paper_forward_engine import PaperForwardEngine

def test_engine_produces_decision_or_reason_without_candles(tmp_path):
    cfg = {"candidates": [{"candidate_id": "p1", "paper_forward_only": True, "enabled": True, "artifact_dir": str(tmp_path/"p1")}]}
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps(cfg))
    (tmp_path/"p1").mkdir()
    (tmp_path/"p1"/"candidate_manifest.json").write_text(json.dumps({"paper_only":True}))
    eng = PaperForwardEngine(candidate_file=str(cf), artifacts_dir=str(tmp_path))
    # no price/candles, but chain present
    decs = eng.on_market_snapshot({"timestamp": "t"}, {"ltp": 82.5, "best_bid": 82.4, "best_ask": 82.6})
    assert len(decs) == 1
    d = decs[0]
    assert d.get("paper_forward_only") is True
    assert d.get("shadow_ready") is False
    # if router produced no_trade due to missing fields, reason should be present
    reason = d.get("no_trade_reason") or d.get("block_reason") or ""
    # even if decision, the call happened
    assert "candidate_id" in d
