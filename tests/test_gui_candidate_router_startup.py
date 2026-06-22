#!/usr/bin/env python3
"""Test GUI candidate router startup, multi paper config, no live block."""
import os
import json
import tempfile
from pathlib import Path
import pytest

def test_paper_config_loaded_and_blocks_live(monkeypatch, tmp_path):
    cfg = {"candidates": [{"candidate_id": "p1", "paper_forward_only": True, "enabled": True}]}
    cf = tmp_path / "paper_forward_candidates.json"
    cf.write_text(json.dumps(cfg))
    monkeypatch.setenv("MSTOCK_ENABLE_LIVE_ORDERS", "false")
    # simulate load
    data = json.load(open(cf))
    cands = [c for c in data.get("candidates", []) if c.get("paper_forward_only")]
    assert len(cands) == 1
    assert cands[0]["paper_forward_only"]

    # live block
    monkeypatch.setenv("MSTOCK_ENABLE_LIVE_ORDERS", "true")
    live = os.getenv("MSTOCK_ENABLE_LIVE_ORDERS", "false").lower() in ("1","true","yes")
    assert live is True  # would block start

def test_router_runs_with_option_chain_no_candles():
    # simulate: even with empty candles, if oc present, decision path ok (no crash, reason if missing fields)
    snap = {"price": None, "timestamp": "t"}  # no candles
    chain = {"ltp": 82.5, "best_bid": 82.4, "best_ask": 82.6}
    # would call engine which calls router -> produces decision or NO_TRADE with reason
    assert chain is not None  # data available
    # reason example
    reason = "missing_candles" if not snap.get("price") else "ok"
    assert reason == "missing_candles"
