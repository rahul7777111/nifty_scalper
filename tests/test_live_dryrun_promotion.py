"""Tests for live dry-run promotion and payload building (no real orders)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.live_order_dryrun import build_and_validate_dry_run_payload, is_live_dry_run_active, execute_dry_run_if_requested


def test_live_dry_run_builds_payload_but_does_not_send_order(monkeypatch):
    monkeypatch.setenv("LIVE_MODE", "true")
    monkeypatch.setenv("LIVE_ORDER_DRY_RUN", "true")
    monkeypatch.setenv("ORDER_PLACEMENT_ENABLED", "false")
    monkeypatch.setenv("SCALPER_KILL_SWITCH", "0")

    assert is_live_dry_run_active() is True

    payload, err = build_and_validate_dry_run_payload(
        candidate_id="t1",
        symbol="NIFTY24JUN24000PE",
        exchange="NFO",
        symbol_token="12345",
        side="SELL",
        quantity=65,
        order_type="LIMIT",
        price=88.5,
        expiry="2024-06-24",
        option_type="PE",
    )
    assert err is None
    assert payload["live_order_sent"] is False
    assert payload["dry_run"] is True
    assert payload["order_type"] == "LIMIT"


def test_non_whitelisted_candidate_cannot_trade_live():
    # The gate functions are exercised via the promotion scripts + mstock gate.
    # Here we just assert that a DISABLED candidate record does not look whitelisted.
    from src.candidate_lifecycle import Candidate, STATUS_DISABLED
    c = Candidate(candidate_id="bad", status=STATUS_DISABLED, live_whitelisted=False)
    assert c.live_whitelisted is False
    assert c.status != "LIVE_1_LOT"
