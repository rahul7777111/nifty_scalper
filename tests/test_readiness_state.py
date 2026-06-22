"""tests/test_readiness_state.py — unit tests for build_readiness_state.py."""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import build_readiness_state as brs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_report(tmp_reports: Path, pattern: str, data: dict) -> Path:
    """Write a dummy report JSON into tmp_reports."""
    p = tmp_reports / pattern
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_find_latest_returns_none_when_empty(tmp_path):
    with patch.object(brs, "REPORTS", tmp_path):
        assert brs.find_latest("nonexistent_*.json") is None


def test_find_latest_returns_newest_match(tmp_path):
    with patch.object(brs, "REPORTS", tmp_path):
        (tmp_path / "old.json").write_text("{}", encoding="utf-8")
        new = tmp_path / "new.json"
        new.write_text("{}", encoding="utf-8")
        # Force mtime difference so ordering is deterministic
        os.utime(new, (new.stat().st_mtime + 10, new.stat().st_mtime + 10))
        assert brs.find_latest("*.json") == new


def test_load_json_returns_empty_for_missing():
    assert brs.load_json(None) == {}
    assert brs.load_json(Path("/nonexistent/path.json")) == {}


def test_is_stale_true_for_old_file(tmp_path):
    old = tmp_path / "old.json"
    old.write_text("{}", encoding="utf-8")
    os.utime(old, (old.stat().st_mtime - 100_000, old.stat().st_mtime - 100_000))
    assert brs.is_stale(old, max_age_hours=1) is True


def test_is_stale_false_for_fresh_file(tmp_path):
    fresh = tmp_path / "fresh.json"
    fresh.write_text("{}", encoding="utf-8")
    assert brs.is_stale(fresh, max_age_hours=48) is False


def test_missing_reports_block_readiness(tmp_path):
    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()
    # Without any reports, all gates should fail
    assert state["paper_engine_pass"] is False
    assert state["broker_safety_pass"] is False
    assert state["real_trading_allowed_now"] is False
    assert state["micro_live_allowed_now"] is False
    assert len(state["blockers"]) > 0


def test_valid_reports_produce_readiness_state(tmp_path):
    _write_report(tmp_path, "final_paper_engine_readiness_1.json", {"status": "PASS"})
    _write_report(tmp_path, "broker_safety_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "real_trading_gate_audit_1.json", {"all_gates_pass": True})
    _write_report(tmp_path, "risk_control_gap_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "live_decision_dry_run_1.json", {
        "coverage_pct": 98.0,
        "probability": 0.65,
        "threshold": 0.5,
        "model_pkl": "/some/model.pkl",
    })
    _write_report(tmp_path, "paper_forward_test_summary_1.json", {
        "total_decisions": 10,
        "total_trade": 5,
        "net_pnl_total": 120.0,
        "max_drawdown": 200.0,
    })
    _write_report(tmp_path, "model_signal_quality_audit_1.json", {"verdict": "PASS"})

    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()

    assert state["paper_engine_pass"] is True
    assert state["broker_safety_pass"] is True
    assert state["real_trading_gate_pass"] is True
    assert state["risk_control_gap_fix_pass"] is True
    assert state["model_signal_quality_verdict"] == "PASS"
    assert state["model_pkl_exists"] is True
    assert state["dry_run_coverage_pct"] == 98.0
    assert state["latest_probability"] == 0.65
    assert state["forward_test_trade_count"] == 5
    assert state["forward_test_enough_evidence"] is True
    assert state["real_trading_allowed_now"] is True
    assert state["micro_live_allowed_now"] is True
    assert len(state["blockers"]) == 0


def test_model_blocked_verdict_blocks_micro_live(tmp_path):
    _write_report(tmp_path, "final_paper_engine_readiness_1.json", {"status": "PASS"})
    _write_report(tmp_path, "broker_safety_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "real_trading_gate_audit_1.json", {"all_gates_pass": True})
    _write_report(tmp_path, "risk_control_gap_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "live_decision_dry_run_1.json", {
        "coverage_pct": 98.0, "probability": 0.65, "threshold": 0.5, "model_pkl": "/x.pkl",
    })
    _write_report(tmp_path, "paper_forward_test_summary_1.json", {
        "total_decisions": 10, "total_trade": 5, "net_pnl_total": 120.0, "max_drawdown": 200.0,
    })
    _write_report(tmp_path, "model_signal_quality_audit_1.json", {"verdict": "RESEARCH_ONLY"})

    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()

    assert state["real_trading_allowed_now"] is False
    assert state["micro_live_allowed_now"] is False
    assert "model signal quality verdict: RESEARCH_ONLY" in state["blockers"]


def test_low_probability_blocks_real_trading(tmp_path):
    _write_report(tmp_path, "final_paper_engine_readiness_1.json", {"status": "PASS"})
    _write_report(tmp_path, "broker_safety_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "real_trading_gate_audit_1.json", {"all_gates_pass": True})
    _write_report(tmp_path, "live_decision_dry_run_1.json", {
        "coverage_pct": 98.0, "probability": 0.3, "threshold": 0.5, "model_pkl": "/x.pkl",
    })

    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()

    assert state["latest_probability"] == 0.3
    assert state["real_trading_allowed_now"] is False
    assert state["micro_live_allowed_now"] is False


def test_insufficient_forward_test_trades_blocks_micro_live(tmp_path):
    _write_report(tmp_path, "final_paper_engine_readiness_1.json", {"status": "PASS"})
    _write_report(tmp_path, "broker_safety_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "real_trading_gate_audit_1.json", {"all_gates_pass": True})
    _write_report(tmp_path, "risk_control_gap_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "live_decision_dry_run_1.json", {
        "coverage_pct": 98.0, "probability": 0.65, "threshold": 0.5, "model_pkl": "/x.pkl",
    })
    _write_report(tmp_path, "paper_forward_test_summary_1.json", {
        "total_decisions": 2, "total_trade": 2, "net_pnl_total": 50.0, "max_drawdown": 100.0,
    })
    _write_report(tmp_path, "model_signal_quality_audit_1.json", {"verdict": "PASS"})

    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()

    assert state["forward_test_trade_count"] == 2
    assert state["forward_test_enough_evidence"] is False
    assert state["micro_live_allowed_now"] is False


def test_negative_net_pnl_blocks_micro_live(tmp_path):
    _write_report(tmp_path, "final_paper_engine_readiness_1.json", {"status": "PASS"})
    _write_report(tmp_path, "broker_safety_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "real_trading_gate_audit_1.json", {"all_gates_pass": True})
    _write_report(tmp_path, "risk_control_gap_fix_1.json", {"status": "PASS"})
    _write_report(tmp_path, "live_decision_dry_run_1.json", {
        "coverage_pct": 98.0, "probability": 0.65, "threshold": 0.5, "model_pkl": "/x.pkl",
    })
    _write_report(tmp_path, "paper_forward_test_summary_1.json", {
        "total_decisions": 10, "total_trade": 5, "net_pnl_total": -50.0, "max_drawdown": 100.0,
    })
    _write_report(tmp_path, "model_signal_quality_audit_1.json", {"verdict": "PASS"})

    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()

    assert state["real_trading_allowed_now"] is True  # real trading can still be allowed
    assert state["micro_live_allowed_now"] is False   # but micro-live blocked by negative PnL


def test_stale_report_blocks_readiness(tmp_path):
    _write_report(tmp_path, "final_paper_engine_readiness_old.json", {"status": "PASS"})
    old = tmp_path / "final_paper_engine_readiness_old.json"
    os.utime(old, (old.stat().st_mtime - 200_000, old.stat().st_mtime - 200_000))

    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()

    assert state["paper_engine_pass"] is True
    assert any("stale" in b.lower() for b in state["blockers"])
    assert state["real_trading_allowed_now"] is False


def test_outputs_created(tmp_path):
    with patch.object(brs, "REPORTS", tmp_path):
        state = brs.main()

    json_files = list(tmp_path.glob("readiness_state_*.json"))
    md_files = list(tmp_path.glob("readiness_state_*.md"))
    assert len(json_files) >= 1
    assert len(md_files) >= 1

    loaded = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert loaded["timestamp"] == state["timestamp"]
