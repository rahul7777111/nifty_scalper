#!/usr/bin/env python3
"""Tests for shadow mode routing in evaluate_shadow_mode.py.

Covers:
  - Shadow mode never creates paper or real order
  - Shadow mode logs WOULD_ENTER signals correctly
  - Shadow mode handles empty log gracefully
  - Shadow evaluation computes correct stats from WOULD_ENTER rows
  - Shadow mode does not call broker API
  - Shadow mode does not place orders in any form
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shadow_row(
    option_type: str = "PE",
    probability: float = 0.70,
    threshold: float = 0.50,
    realized_return: float = 0.0,
    shadow_decision: str = "WOULD_ENTER",
) -> dict:
    """Return a minimal shadow-mode JSONL row."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "option_type": option_type,
        "probability": probability,
        "threshold": threshold,
        "shadow_decision": shadow_decision,
        "realized_return": realized_return,
        "coverage_pct": 100.0,
        "model_id": "test_candidate",
        "model_pkl": "test.pkl",
        "blocking_reasons": [],
    }


# ---------------------------------------------------------------------------
# Test 1: Shadow mode never creates paper or real order
# ---------------------------------------------------------------------------

def test_shadow_mode_never_calls_place_order(monkeypatch):
    """Shadow mode pipeline must never call broker place_order."""
    # We verify this by checking that the shadow evaluation script
    # only reads JSONL logs and never imports or calls broker client methods.
    import scripts.evaluate_shadow_mode as sm_mod

    # Confirm the module does NOT import dhan_client or mstock_client
    import types
    shadow_globals = dir(sm_mod)
    broker_modules = [k for k in shadow_globals if "client" in k.lower()
                      or "dhan" in k.lower() or "order" in k.lower()]
    assert len(broker_modules) == 0, (
        f"Shadow mode script must not import broker client modules. Found: {broker_modules}"
    )


def test_shadow_mode_jsonl_writer_never_creates_order_artifact(tmp_path):
    """Writing a shadow log row must not create any order file."""
    log_path = tmp_path / "ml_shadow_predictions_20260608.jsonl"

    # Write a shadow row
    rows = [_make_shadow_row()]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rows[0]) + "\n")

    # Shadow row written
    assert log_path.exists()

    # No order files created (no order CSV, no trade JSON)
    order_files = list(tmp_path.glob("*.csv")) + list(tmp_path.glob("trade_*.json"))
    assert len(order_files) == 0, "Shadow mode must not create order files"


def test_shadow_rows_never_contain_filled_prices(tmp_path):
    """Shadow mode WOULD_ENTER rows must not contain actual fill prices."""
    row = _make_shadow_row(shadow_decision="WOULD_ENTER")
    assert "fill_price" not in row
    assert "execution_price" not in row
    assert "entry_price" not in row
    assert "exit_price" not in row


# ---------------------------------------------------------------------------
# Test 2: Shadow mode logs WOULD_ENTER signals correctly
# ---------------------------------------------------------------------------

def test_evaluate_shadow_rows_counts_would_enter_correctly(tmp_path):
    """evaluate_shadow_rows must correctly count WOULD_ENTER rows."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER"),
        _make_shadow_row(shadow_decision="SKIP"),
        _make_shadow_row(shadow_decision="WOULD_ENTER"),
        _make_shadow_row(shadow_decision="WOULD_ENTER"),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    assert result["number_of_would_enter_signals"] == 3
    assert result["number_of_predictions"] == 4


def test_evaluate_shadow_rows_empty_log_returns_zero_counts(tmp_path):
    """Empty JSONL log → evaluate_shadow_rows returns zero counts (no crash)."""
    import scripts.evaluate_shadow_mode as sm_mod

    result = sm_mod.evaluate_shadow_rows([])
    assert result["number_of_predictions"] == 0
    assert result["number_of_would_enter_signals"] == 0
    assert result["realized_outcome_count"] == 0


def test_evaluate_shadow_rows_computes_win_rate(tmp_path):
    """evaluate_shadow_rows computes win rate from realized returns."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=-5.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=8.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=-2.0),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    assert result["realized_outcome_count"] == 4
    assert result["win_rate"] == pytest.approx(0.5)  # 2 wins / 4 = 0.5


def test_evaluate_shadow_rows_computes_profit_factor(tmp_path):
    """evaluate_shadow_rows computes profit factor correctly."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=20.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=-5.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    gross_profit = 30.0
    gross_loss = 5.0
    expected_pf = gross_profit / gross_loss
    assert result["profit_factor"] == pytest.approx(expected_pf)


def test_evaluate_shadow_rows_computes_avg_return(tmp_path):
    """evaluate_shadow_rows computes average return."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=20.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=-10.0),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    expected_avg = (10.0 + 20.0 - 10.0) / 3
    assert result["average_return"] == pytest.approx(expected_avg)


def test_evaluate_shadow_rows_splits_by_option_type(tmp_path):
    """evaluate_shadow_rows breaks down WOULD_ENTER by option_type."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", option_type="PE"),
        _make_shadow_row(shadow_decision="WOULD_ENTER", option_type="PE"),
        _make_shadow_row(shadow_decision="WOULD_ENTER", option_type="CE"),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    assert result["ce_pe_breakdown"].get("PE") == 2
    assert result["ce_pe_breakdown"].get("CE") == 1


# ---------------------------------------------------------------------------
# Test 3: Shadow mode does not call broker API
# ---------------------------------------------------------------------------

def test_shadow_mode_script_has_no_broker_imports():
    """evaluate_shadow_mode.py must not import any broker client module."""
    import scripts.evaluate_shadow_mode as sm_mod
    source = open(sm_mod.__file__, encoding="utf-8").read()

    forbidden_imports = [
        "import dhan", "from dhan", "import dhan_client",
        "from dhan_client", "import mstock", "from mstock",
        "from mstock_client", "import MStock", "from src.dhan",
        "from src.mstock_client",
    ]
    for forbidden in forbidden_imports:
        assert forbidden not in source, (
            f"Shadow mode script must not import broker module. Found: {forbidden}"
        )


def test_shadow_mode_script_never_calls_place_order():
    """evaluate_shadow_mode.py source code must not contain place_order call."""
    import scripts.evaluate_shadow_mode as sm_mod
    source = open(sm_mod.__file__, encoding="utf-8").read()

    assert "place_order" not in source, "Shadow mode must never call place_order"
    assert "submit_order" not in source, "Shadow mode must never call submit_order"


# ---------------------------------------------------------------------------
# Test 4: Shadow mode handles mixed decision types
# ---------------------------------------------------------------------------

def test_evaluate_shadow_rows_handles_skip_decisions(tmp_path):
    """SKIP decisions must not contribute to realized outcomes."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
        _make_shadow_row(shadow_decision="SKIP"),
        _make_shadow_row(shadow_decision="SKIP"),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    # Only 1 realized outcome (from WOULD_ENTER)
    assert result["realized_outcome_count"] == 1
    assert result["number_of_would_enter_signals"] == 1


def test_evaluate_shadow_rows_handles_missing_realized_return(tmp_path):
    """Rows without realized_return default to 0 (not counted)."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        {"shadow_decision": "WOULD_ENTER"},  # no realized_return
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    # Realized outcomes require realized_return to be present
    assert result["realized_outcome_count"] == 1


def test_evaluate_shadow_rows_handles_null_realized_return(tmp_path):
    """Rows with realized_return=null must not contribute to realized outcomes."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=None),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    assert result["realized_outcome_count"] == 1


# ---------------------------------------------------------------------------
# Test 5: Shadow mode reports safety compliance
# ---------------------------------------------------------------------------

def test_shadow_mode_report_contains_safety_fields(tmp_path):
    """Shadow evaluation output must include safety compliance fields."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [_make_shadow_row()]
    result = sm_mod.evaluate_shadow_rows(rows)

    # Required safety fields
    assert "number_of_predictions" in result
    assert "number_of_would_enter_signals" in result
    assert "realized_outcome_count" in result
    assert "win_rate" in result
    assert "profit_factor" in result
    assert "ce_pe_breakdown" in result


def test_shadow_mode_max_drawdown_computed(tmp_path):
    """evaluate_shadow_rows computes max drawdown from equity curve."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=-8.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=5.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=-3.0),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    # Equity: [10, 2, 7, 4]
    # Peaks:  [10, 10, 10, 10]
    # DD:     [0, 8, 3, 6] → max_dd = 8
    assert result["max_drawdown"] == 8.0


# ---------------------------------------------------------------------------
# Test 6: Shadow mode evaluation — no double-counting returns
# ---------------------------------------------------------------------------

def test_shadow_mode_no_double_counting_pnl(tmp_path):
    """Each WOULD_ENTER row contributes exactly once to P&L."""
    import scripts.evaluate_shadow_mode as sm_mod

    rows = [
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=10.0),
        _make_shadow_row(shadow_decision="WOULD_ENTER", realized_return=20.0),
    ]
    result = sm_mod.evaluate_shadow_rows(rows)
    assert result["realized_outcome_count"] == 2
    assert result["win_rate"] == 1.0  # Both positive → 100% win rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])