"""
tests/test_pe_only_filter.py
============================
Unit tests for apply_pe_only_filter().

Covers:
  1. PE row passes
  2. CE row rejects
  3. Missing option_type rejects safely (None, empty string, missing key)

Safety contract:
  - Filter does NOT read outcome, future return, PnL, or label columns.
  - Filter is live-computable (stateless, reads only current snapshot).
  - All test cases use synthetic snapshots — no broker interaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the scripts/ directory is on the path so the import below resolves
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from live_decision_dry_run import apply_pe_only_filter


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_pe_row_passes():
    """PE option_type should pass the filter and allow scoring."""
    snapshot = {"option_type": "PE", "spot": 24500.0}
    result = apply_pe_only_filter(snapshot)

    assert result["filter_name"] == "PE_only"
    assert result["filter_rule"] == "allow scoring only when option_type == 'PE'"
    assert result["filter_applied"] is True
    assert result["filter_passed"] is True
    assert result["filter_rejection_reason"] is None


def test_pe_row_passes_case_insensitive():
    """PE filter is case-insensitive: 'pe', 'Pe', 'PE' all pass."""
    for val in ("pe", "Pe", "PE", "  PE  "):
        snapshot = {"option_type": val}
        result = apply_pe_only_filter(snapshot)
        assert result["filter_passed"] is True, f"option_type={val!r} should pass"
        assert result["filter_rejection_reason"] is None


def test_ce_row_rejects():
    """CE option_type must be rejected with the correct reason."""
    snapshot = {"option_type": "CE", "spot": 24500.0}
    result = apply_pe_only_filter(snapshot)

    assert result["filter_name"] == "PE_only"
    assert result["filter_applied"] is True
    assert result["filter_passed"] is False
    assert result["filter_rejection_reason"] == "candidate_filter_rejected_option_type_not_PE"


def test_ce_row_case_insensitive():
    """CE rejection is case-insensitive."""
    for val in ("ce", "Ce", "CE", "  CE  "):
        snapshot = {"option_type": val}
        result = apply_pe_only_filter(snapshot)
        assert result["filter_passed"] is False, f"option_type={val!r} should reject"
        assert result["filter_rejection_reason"] == "candidate_filter_rejected_option_type_not_PE"


def test_missing_option_type_rejects_none():
    """option_type=None should reject safely."""
    snapshot = {"option_type": None, "spot": 24500.0}
    result = apply_pe_only_filter(snapshot)

    assert result["filter_passed"] is False
    assert result["filter_rejection_reason"] == "candidate_filter_rejected_option_type_not_PE"


def test_missing_option_type_rejects_empty_string():
    """option_type='' (empty string) should reject safely."""
    snapshot = {"option_type": "", "spot": 24500.0}
    result = apply_pe_only_filter(snapshot)

    assert result["filter_passed"] is False
    assert result["filter_rejection_reason"] == "candidate_filter_rejected_option_type_not_PE"


def test_missing_option_type_key_rejects():
    """Missing 'option_type' key should reject safely (defensive: cannot confirm PE)."""
    snapshot = {"spot": 24500.0, "rsi_14": 45.0}  # no option_type key at all
    result = apply_pe_only_filter(snapshot)

    assert result["filter_passed"] is False
    assert result["filter_rejection_reason"] == "candidate_filter_rejected_option_type_not_PE"


def test_unknown_option_type_rejects():
    """option_type with an unexpected value should reject safely."""
    for val in ("FUTURE", "CALL", "PUT", "VANILLA", "EXOTIC", "UNKNOWN"):
        snapshot = {"option_type": val}
        result = apply_pe_only_filter(snapshot)
        assert result["filter_passed"] is False, f"option_type={val!r} should reject"
        assert result["filter_rejection_reason"] == "candidate_filter_rejected_option_type_not_PE"


def test_filter_is_live_computable():
    """Filter must be stateless: calling twice with the same input gives the same output."""
    snapshot = {"option_type": "PE", "spot": 24500.0}
    r1 = apply_pe_only_filter(snapshot)
    r2 = apply_pe_only_filter(snapshot)
    assert r1 == r2


def test_filter_does_not_read_outcome_columns():
    """Verify filter only reads 'option_type' — no outcome/PnL/label/future-return cols."""
    snapshot_with_outcome_cols = {
        "option_type": "PE",
        # Outcome/label columns — must NOT be read by the filter
        "outcome": 1,
        "pnl": 150.0,
        "future_return": 0.02,
        "label": 1,
        "return_5m": 0.005,
        "win": True,
        "realized_pnl": 200.0,
    }
    result = apply_pe_only_filter(snapshot_with_outcome_cols)
    # Filter must pass PE and ignore all outcome columns
    assert result["filter_passed"] is True
    assert result["filter_rejection_reason"] is None


# ---------------------------------------------------------------------------
# Smoke test: run via pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest

    # Discover and run this file's tests
    exit(pytest.main([__file__, "-v", "--tb=short"]))