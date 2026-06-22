"""Shadow engine safety and logging tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.shadow_engine import ShadowEngine


def test_shadow_candidate_never_sends_real_broker_order(tmp_path, monkeypatch):
    # Force a tiny shadow list
    monkeypatch.setenv("SCALPER_KILL_SWITCH", "0")
    eng = ShadowEngine(log_dir=str(tmp_path))
    # Inject a synthetic candidate
    from src.candidate_lifecycle import Candidate, STATUS_SHADOW
    c = Candidate(candidate_id="fake_shadow_001", status=STATUS_SHADOW, enabled=True)
    eng.candidates = [c]
    eng._active[c.candidate_id] = {"open": False}

    rows = eng.on_market_snapshot({"spot": 24000}, [{"ltp": 120, "option_type": "PE", "strike": 23900}])
    # No place_order should have been called anywhere (we can only assert on the records)
    for r in rows:
        assert r.get("live_order_sent") is False
        assert "would_order_payload" in r
        assert r["would_order_payload"].get("dry_run") is True


def test_shadow_candidate_logs_would_order_payload(tmp_path):
    eng = ShadowEngine(log_dir=str(tmp_path))
    from src.candidate_lifecycle import Candidate, STATUS_SHADOW
    c = Candidate(candidate_id="fake_shadow_002", status=STATUS_SHADOW, enabled=True)
    eng.candidates = [c]
    eng._active[c.candidate_id] = {"open": False}

    rows = eng.on_market_snapshot({"spot": 24050}, [{"ltp": 95, "bid": 94, "ask": 96, "option_type": "CE"}])
    assert len(rows) >= 1
    p = rows[0].get("would_order_payload", {})
    assert p.get("live_order_sent") is False
    assert "symbol" in p
