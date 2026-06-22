#!/usr/bin/env python3
"""
test_shadow_candidate_preset_logging.py
Verifies that shadow JSONL / decision rows contain the required candidate/preset/side fields.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT))

from src.candidate_router import route_candidate_decision


def test_shadow_log_contains_required_candidate_preset_side_fields(tmp_path: Path):
    # Run a decision and inspect the returned dict (simulates what gets written to JSONL)
    dec = route_candidate_decision(
        market_snapshot={
            "symbol": "NIFTY",
            "spot": 24500.0,
            "regime": "bullish",
            "atr_pct": 0.008,
            "bid": 148.0,
            "ask": 149.5,
            "ltp": 148.7,
        },
        mode="shadow",
        active_candidate_id=None,
        candidate_dir="models/candidates",
        force_eval=True,  # allow even if nothing is shadow-ready in the test env
    )

    # The fields required by PHASE 4 spec must be present (even if values are conservative)
    required = [
        "timestamp", "candidate_id", "model_name", "preset_family", "selected_preset",
        "side_policy", "side_decision", "confidence", "threshold",
        "market_regime", "volatility_state", "trend_state", "liquidity_state",
        "allowed_by_model", "allowed_by_preset", "allowed_by_side_policy",
        "allowed_by_liquidity", "allowed_by_cost", "allowed_by_risk",
        "shadow_ready", "forced_eval", "final_signal", "no_trade_reason",
    ]
    for f in required:
        assert f in dec, f"Missing required shadow log field: {f}"

    # Simulated execution fields are added by the shadow script (we simulate here)
    dec["simulated_entry"] = dec.get("final_signal", "NO_TRADE") != "NO_TRADE"
    dec["simulated_exit"] = None
    dec["estimated_return"] = None
    dec["actual_return"] = None

    # Write a temp JSONL like the shadow script does and re-read to assert
    log_path = tmp_path / "shadow_test.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(dec, default=str) + "\n")

    with log_path.open("r", encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    assert "candidate_id" in row
    assert "selected_preset" in row
    assert "side_policy" in row
    assert "side_decision" in row
    assert "threshold" in row
    assert "final_signal" in row
    assert "no_trade_reason" in row
    assert "forced_eval" in row


def test_forced_eval_is_marked_when_candidate_not_shadow_ready():
    dec = route_candidate_decision(
        market_snapshot={"spot": 24500.0, "regime": "mixed"},
        mode="shadow",
        active_candidate_id="definitely_not_shadow_ready_zzz",
        candidate_dir=".",
        force_eval=True,
    )
    assert dec["forced_eval"] is True
    # It may still be NO_TRADE for other reasons, but the flag must be carried
    assert "forced_eval" in dec
