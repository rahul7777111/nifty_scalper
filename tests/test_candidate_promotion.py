"""Tests for the candidate promotion lifecycle (paper->shadow, shadow->dry, dry->1lot)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.candidate_lifecycle import (
    Candidate,
    evaluate_paper_to_shadow,
    evaluate_shadow_to_live_dryrun,
    promote_candidate,
    STATUS_SHADOW,
    STATUS_LIVE_DRY_RUN,
    STATUS_LIVE_1_LOT,
)


def test_paper_candidate_with_stale_data_is_not_promoted():
    stats = {"trade_count": 30, "days_tested": 4, "stale_data_trade_count": 2}
    ok, reason = evaluate_paper_to_shadow(stats)
    assert ok is False
    assert "stale" in reason.lower()


def test_paper_candidate_with_invalid_expiry_is_not_promoted():
    stats = {"trade_count": 25, "days_tested": 3, "invalid_expiry_trade_count": 1}
    ok, reason = evaluate_paper_to_shadow(stats)
    assert ok is False
    assert "invalid_expiry" in reason.lower() or "expiry" in reason.lower()


def test_paper_candidate_with_missing_sl_is_not_promoted():
    stats = {"trade_count": 22, "days_tested": 3, "missing_sl_count": 1}
    ok, reason = evaluate_paper_to_shadow(stats)
    assert ok is False
    assert "missing_sl" in reason.lower()


def test_paper_candidate_with_enough_clean_trades_is_promoted_to_shadow():
    stats = {
        "trade_count": 25,
        "days_tested": 3,
        "stale_data_trade_count": 0,
        "duplicate_trade_count": 0,
        "invalid_expiry_trade_count": 0,
        "missing_sl_count": 0,
        "missing_exit_count": 0,
        "after_cutoff_trade_count": 0,
        "cost_adjusted_pnl": 120.0,
        "max_drawdown": 800.0,
    }
    ok, reason = evaluate_paper_to_shadow(stats, min_trades=20, min_days=2)
    assert ok is True
    c = Candidate(candidate_id="test_clean", status="PAPER_FORWARD")
    promote_candidate(c, STATUS_SHADOW, notes=reason)
    assert c.status == STATUS_SHADOW
    assert "shadow" in " ".join(c.allowed_modes).lower() or c.enabled is True
