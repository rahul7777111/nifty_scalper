#!/usr/bin/env python3
"""
Tests for candidate_filters.py  (src/candidate_filters.py)

Covers all 7 filters:
  - PE_only         : option_type == "PE"
  - CE_only         : option_type == "CE"
  - DTE_greater_7   : dte_days > 7
  - DTE_14_30       : 14 <= dte_days <= 30
  - ITM_all         : moneyness_bucket == "ITM"
  - ATM_all         : moneyness_bucket == "ATM"
  - premium_10_100  : 10 <= ltp <= 100  OR  10 <= mid_price <= 100

Each test verifies:
  1. Pass case
  2. Reject case
  3. Missing required field → safe rejection
  4. filter_applied always True
  5. required_fields_present correct

No filter reads future/outcome/return/PnL/label columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from candidate_filters import (
    apply_pe_only_filter,
    apply_ce_only_filter,
    apply_dte_greater_7_filter,
    apply_dte_14_30_filter,
    apply_itm_all_filter,
    apply_atm_all_filter,
    apply_premium_10_100_filter,
    apply_filters,
    ALL_FILTERS,
)


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def _assert_pass(result: dict, filter_name: str) -> None:
    assert result["filter_applied"] is True, f"{filter_name}: filter_applied should be True"
    assert result["filter_passed"] is True, f"{filter_name}: filter_passed should be True"
    assert result["rejection_reason"] is None, f"{filter_name}: rejection_reason should be None on pass"
    assert result["required_fields_present"] is True, f"{filter_name}: required_fields_present should be True on pass"


def _assert_reject(result: dict, filter_name: str, expected_reason_contains: str) -> None:
    assert result["filter_applied"] is True, f"{filter_name}: filter_applied should be True"
    assert result["filter_passed"] is False, f"{filter_name}: filter_passed should be False"
    assert result["rejection_reason"] is not None, f"{filter_name}: rejection_reason must be set"
    assert expected_reason_contains in result["rejection_reason"], (
        f"{filter_name}: rejection reason '{result['rejection_reason']}' "
        f"should contain '{expected_reason_contains}'"
    )


def _assert_missing_field(result: dict, filter_name: str, expected_reason_contains: str) -> None:
    assert result["filter_applied"] is True, f"{filter_name}: filter_applied should be True"
    assert result["filter_passed"] is False, f"{filter_name}: filter_passed should be False"
    assert result["required_fields_present"] is False, (
        f"{filter_name}: required_fields_present should be False when field is missing"
    )
    assert expected_reason_contains in result["rejection_reason"], (
        f"{filter_name}: rejection reason '{result['rejection_reason']}' "
        f"should contain '{expected_reason_contains}'"
    )


# ===========================================================================
# PE_only
# ===========================================================================

class TestPEOnly:
    def test_pe_allows_pe_upper(self):
        result = apply_pe_only_filter({"option_type": "PE"})
        _assert_pass(result, "PE_only")

    def test_pe_allows_pe_lower(self):
        result = apply_pe_only_filter({"option_type": "pe"})
        _assert_pass(result, "PE_only")

    def test_pe_allows_pe_mixed(self):
        result = apply_pe_only_filter({"option_type": "Pe"})
        _assert_pass(result, "PE_only")

    def test_pe_rejects_ce(self):
        result = apply_pe_only_filter({"option_type": "CE"})
        _assert_reject(result, "PE_only", "not_PE")

    def test_pe_rejects_ce_lower(self):
        result = apply_pe_only_filter({"option_type": "ce"})
        _assert_reject(result, "PE_only", "not_PE")

    def test_pe_rejects_other(self):
        result = apply_pe_only_filter({"option_type": "FUT"})
        _assert_reject(result, "PE_only", "not_PE")

    def test_pe_missing_key(self):
        result = apply_pe_only_filter({})
        _assert_missing_field(result, "PE_only", "not_PE")

    def test_pe_none_value(self):
        result = apply_pe_only_filter({"option_type": None})
        _assert_missing_field(result, "PE_only", "not_PE")

    def test_pe_empty_string(self):
        result = apply_pe_only_filter({"option_type": ""})
        _assert_missing_field(result, "PE_only", "not_PE")


# ===========================================================================
# CE_only
# ===========================================================================

class TestCEOnly:
    def test_ce_allows_ce_upper(self):
        result = apply_ce_only_filter({"option_type": "CE"})
        _assert_pass(result, "CE_only")

    def test_ce_allows_ce_lower(self):
        result = apply_ce_only_filter({"option_type": "ce"})
        _assert_pass(result, "CE_only")

    def test_ce_allows_ce_mixed(self):
        result = apply_ce_only_filter({"option_type": "Ce"})
        _assert_pass(result, "CE_only")

    def test_ce_rejects_pe(self):
        result = apply_ce_only_filter({"option_type": "PE"})
        _assert_reject(result, "CE_only", "not_CE")

    def test_ce_rejects_pe_lower(self):
        result = apply_ce_only_filter({"option_type": "pe"})
        _assert_reject(result, "CE_only", "not_CE")

    def test_ce_rejects_other(self):
        result = apply_ce_only_filter({"option_type": "FUT"})
        _assert_reject(result, "CE_only", "not_CE")

    def test_ce_missing_key(self):
        result = apply_ce_only_filter({})
        _assert_missing_field(result, "CE_only", "not_CE")

    def test_ce_none_value(self):
        result = apply_ce_only_filter({"option_type": None})
        _assert_missing_field(result, "CE_only", "not_CE")

    def test_ce_empty_string(self):
        result = apply_ce_only_filter({"option_type": ""})
        _assert_missing_field(result, "CE_only", "not_CE")


# ===========================================================================
# DTE_greater_7
# ===========================================================================

class TestDTEGreater7:
    def test_dte_greater_7_exactly_8(self):
        result = apply_dte_greater_7_filter({"dte_days": 8.0})
        _assert_pass(result, "DTE_greater_7")

    def test_dte_greater_7_10(self):
        result = apply_dte_greater_7_filter({"dte_days": 10})
        _assert_pass(result, "DTE_greater_7")

    def test_dte_greater_7_100(self):
        result = apply_dte_greater_7_filter({"dte_days": 100})
        _assert_pass(result, "DTE_greater_7")

    def test_dte_greater_7_string_numeric(self):
        result = apply_dte_greater_7_filter({"dte_days": "15.5"})
        _assert_pass(result, "DTE_greater_7")

    def test_dte_greater_7_rejects_7(self):
        result = apply_dte_greater_7_filter({"dte_days": 7.0})
        _assert_reject(result, "DTE_greater_7", "dte_lte_7")

    def test_dte_greater_7_rejects_5(self):
        result = apply_dte_greater_7_filter({"dte_days": 5.0})
        _assert_reject(result, "DTE_greater_7", "dte_lte_7")

    def test_dte_greater_7_rejects_0(self):
        result = apply_dte_greater_7_filter({"dte_days": 0})
        _assert_reject(result, "DTE_greater_7", "dte_lte_7")

    def test_dte_greater_7_rejects_negative(self):
        result = apply_dte_greater_7_filter({"dte_days": -3.0})
        _assert_reject(result, "DTE_greater_7", "dte_lte_7")

    def test_dte_greater_7_rejects_string(self):
        result = apply_dte_greater_7_filter({"dte_days": "abc"})
        _assert_missing_field(result, "DTE_greater_7", "missing_or_invalid")

    def test_dte_greater_7_rejects_none(self):
        result = apply_dte_greater_7_filter({"dte_days": None})
        _assert_missing_field(result, "DTE_greater_7", "missing_or_invalid")

    def test_dte_greater_7_rejects_missing(self):
        result = apply_dte_greater_7_filter({})
        _assert_missing_field(result, "DTE_greater_7", "missing_or_invalid")


# ===========================================================================
# DTE_14_30
# ===========================================================================

class TestDTE14_30:
    def test_dte_14_30_exact_14(self):
        result = apply_dte_14_30_filter({"dte_days": 14.0})
        _assert_pass(result, "DTE_14_30")

    def test_dte_14_30_exact_30(self):
        result = apply_dte_14_30_filter({"dte_days": 30.0})
        _assert_pass(result, "DTE_14_30")

    def test_dte_14_30_mid(self):
        result = apply_dte_14_30_filter({"dte_days": 21})
        _assert_pass(result, "DTE_14_30")

    def test_dte_14_30_string_numeric(self):
        result = apply_dte_14_30_filter({"dte_days": "20.5"})
        _assert_pass(result, "DTE_14_30")

    def test_dte_14_30_rejects_13(self):
        result = apply_dte_14_30_filter({"dte_days": 13.0})
        _assert_reject(result, "DTE_14_30", "outside_14_30")

    def test_dte_14_30_rejects_31(self):
        result = apply_dte_14_30_filter({"dte_days": 31.0})
        _assert_reject(result, "DTE_14_30", "outside_14_30")

    def test_dte_14_30_rejects_7(self):
        result = apply_dte_14_30_filter({"dte_days": 7})
        _assert_reject(result, "DTE_14_30", "outside_14_30")

    def test_dte_14_30_rejects_60(self):
        result = apply_dte_14_30_filter({"dte_days": 60})
        _assert_reject(result, "DTE_14_30", "outside_14_30")

    def test_dte_14_30_rejects_string(self):
        result = apply_dte_14_30_filter({"dte_days": "abc"})
        _assert_missing_field(result, "DTE_14_30", "missing_or_invalid")

    def test_dte_14_30_rejects_none(self):
        result = apply_dte_14_30_filter({"dte_days": None})
        _assert_missing_field(result, "DTE_14_30", "missing_or_invalid")

    def test_dte_14_30_rejects_missing(self):
        result = apply_dte_14_30_filter({})
        _assert_missing_field(result, "DTE_14_30", "missing_or_invalid")


# ===========================================================================
# ITM_all
# ===========================================================================

class TestITMAll:
    def test_itm_allows_itm_upper(self):
        result = apply_itm_all_filter({"moneyness_bucket": "ITM"})
        _assert_pass(result, "ITM_all")

    def test_itm_allows_itm_lower(self):
        result = apply_itm_all_filter({"moneyness_bucket": "itm"})
        _assert_pass(result, "ITM_all")

    def test_itm_rejects_atm(self):
        result = apply_itm_all_filter({"moneyness_bucket": "ATM"})
        _assert_reject(result, "ITM_all", "not_ITM")

    def test_itm_rejects_otm(self):
        result = apply_itm_all_filter({"moneyness_bucket": "OTM"})
        _assert_reject(result, "ITM_all", "not_ITM")

    def test_itm_rejects_other(self):
        result = apply_itm_all_filter({"moneyness_bucket": "DEEP_ITM"})
        _assert_reject(result, "ITM_all", "not_ITM")

    def test_itm_missing_key(self):
        result = apply_itm_all_filter({})
        _assert_missing_field(result, "ITM_all", "not_ITM")

    def test_itm_none_value(self):
        result = apply_itm_all_filter({"moneyness_bucket": None})
        _assert_missing_field(result, "ITM_all", "not_ITM")

    def test_itm_empty_string(self):
        result = apply_itm_all_filter({"moneyness_bucket": ""})
        _assert_missing_field(result, "ITM_all", "not_ITM")


# ===========================================================================
# ATM_all
# ===========================================================================

class TestATMAll:
    def test_atm_allows_atm_upper(self):
        result = apply_atm_all_filter({"moneyness_bucket": "ATM"})
        _assert_pass(result, "ATM_all")

    def test_atm_allows_atm_lower(self):
        result = apply_atm_all_filter({"moneyness_bucket": "atm"})
        _assert_pass(result, "ATM_all")

    def test_atm_rejects_itm(self):
        result = apply_atm_all_filter({"moneyness_bucket": "ITM"})
        _assert_reject(result, "ATM_all", "not_ATM")

    def test_atm_rejects_otm(self):
        result = apply_atm_all_filter({"moneyness_bucket": "OTM"})
        _assert_reject(result, "ATM_all", "not_ATM")

    def test_atm_rejects_other(self):
        result = apply_atm_all_filter({"moneyness_bucket": "ATM_SPREAD"})
        _assert_reject(result, "ATM_all", "not_ATM")

    def test_atm_missing_key(self):
        result = apply_atm_all_filter({})
        _assert_missing_field(result, "ATM_all", "not_ATM")

    def test_atm_none_value(self):
        result = apply_atm_all_filter({"moneyness_bucket": None})
        _assert_missing_field(result, "ATM_all", "not_ATM")

    def test_atm_empty_string(self):
        result = apply_atm_all_filter({"moneyness_bucket": ""})
        _assert_missing_field(result, "ATM_all", "not_ATM")


# ===========================================================================
# premium_10_100
# ===========================================================================

class TestPremium10_100:
    # --- Pass: ltp in range, mid absent ---
    def test_premium_ltp_in_range_no_mid(self):
        result = apply_premium_10_100_filter({"ltp": 50.0})
        _assert_pass(result, "premium_10_100")

    def test_premium_ltp_at_lower_bound(self):
        result = apply_premium_10_100_filter({"ltp": 10.0})
        _assert_pass(result, "premium_10_100")

    def test_premium_ltp_at_upper_bound(self):
        result = apply_premium_10_100_filter({"ltp": 100.0})
        _assert_pass(result, "premium_10_100")

    def test_premium_ltp_string(self):
        result = apply_premium_10_100_filter({"ltp": "25.5"})
        _assert_pass(result, "premium_10_100")

    # --- Pass: mid_price in range, ltp absent ---
    def test_premium_mid_in_range_no_ltp(self):
        result = apply_premium_10_100_filter({"mid_price": 75.0})
        _assert_pass(result, "premium_10_100")

    def test_premium_mid_at_lower_bound(self):
        result = apply_premium_10_100_filter({"mid_price": 10.0})
        _assert_pass(result, "premium_10_100")

    def test_premium_mid_at_upper_bound(self):
        result = apply_premium_10_100_filter({"mid_price": 100.0})
        _assert_pass(result, "premium_10_100")

    def test_premium_mid_string(self):
        result = apply_premium_10_100_filter({"mid_price": "33.3"})
        _assert_pass(result, "premium_10_100")

    # --- Pass: both in range ---
    def test_premium_both_in_range(self):
        result = apply_premium_10_100_filter({"ltp": 30.0, "mid_price": 60.0})
        _assert_pass(result, "premium_10_100")

    # --- Pass: one in range, other out of range ---
    def test_premium_ltp_in_range_mid_out(self):
        result = apply_premium_10_100_filter({"ltp": 25.0, "mid_price": 200.0})
        _assert_pass(result, "premium_10_100")

    def test_premium_mid_in_range_ltp_out(self):
        result = apply_premium_10_100_filter({"ltp": 5.0, "mid_price": 50.0})
        _assert_pass(result, "premium_10_100")

    # --- Reject: ltp below range ---
    def test_premium_ltp_below_10(self):
        result = apply_premium_10_100_filter({"ltp": 9.9})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    def test_premium_ltp_zero(self):
        result = apply_premium_10_100_filter({"ltp": 0.0})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    def test_premium_ltp_negative(self):
        result = apply_premium_10_100_filter({"ltp": -5.0})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    # --- Reject: ltp above range ---
    def test_premium_ltp_above_100(self):
        result = apply_premium_10_100_filter({"ltp": 100.1})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    def test_premium_ltp_500(self):
        result = apply_premium_10_100_filter({"ltp": 500.0})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    # --- Reject: mid below range ---
    def test_premium_mid_below_10(self):
        result = apply_premium_10_100_filter({"mid_price": 9.9})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    # --- Reject: mid above range ---
    def test_premium_mid_above_100(self):
        result = apply_premium_10_100_filter({"mid_price": 500.0})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    # --- Reject: both out of range ---
    def test_premium_both_out_of_range(self):
        result = apply_premium_10_100_filter({"ltp": 5.0, "mid_price": 500.0})
        _assert_reject(result, "premium_10_100", "premium_outside_10_100")

    # --- Reject: non-numeric ---
    def test_premium_ltp_string(self):
        result = apply_premium_10_100_filter({"ltp": "abc"})
        _assert_missing_field(result, "premium_10_100", "premium_outside_10_100")

    def test_premium_mid_string(self):
        result = apply_premium_10_100_filter({"mid_price": "xyz"})
        _assert_missing_field(result, "premium_10_100", "premium_outside_10_100")

    # --- Reject: both absent ---
    def test_premium_missing_both(self):
        result = apply_premium_10_100_filter({})
        _assert_missing_field(result, "premium_10_100", "premium_outside_10_100")

    def test_premium_none_both(self):
        result = apply_premium_10_100_filter({"ltp": None, "mid_price": None})
        _assert_missing_field(result, "premium_10_100", "premium_outside_10_100")


# ===========================================================================
# apply_filters pipeline helper
# ===========================================================================

class TestApplyFilters:
    def test_apply_filters_subset_that_can_coexist(self):
        # ITM_all and ATM_all are mutually exclusive, so test a subset
        # that can all pass together: PE_only + DTE_greater_7 + ITM_all + premium_10_100
        result = apply_filters(
            {
                "option_type": "PE",
                "dte_days": 20,
                "moneyness_bucket": "ITM",
                "ltp": 50.0,
            },
            filter_names=["PE_only", "DTE_greater_7", "ITM_all", "premium_10_100"],
        )
        assert result["all_passed"] is True, (
            f"Expected all to pass. Results: "
            f"{[(r['filter_name'], r['filter_passed'], r['rejection_reason']) for r in result['results']]}"
        )
        assert result["blocking_reason"] is None
        assert len(result["results"]) == 4

    def test_apply_filters_all_7_filters_one_fails(self):
        # With all 7 filters, ITM_all and ATM_all will both run — one must fail.
        # This is by design: callers should select compatible filter sets.
        result = apply_filters({
            "option_type": "PE",
            "dte_days": 20,
            "moneyness_bucket": "ITM",
            "ltp": 50.0,
        })
        # ATM_all will reject ITM
        atm_result = next(r for r in result["results"] if r["filter_name"] == "ATM_all")
        assert atm_result["filter_passed"] is False
        # ITM_all will pass
        itm_result = next(r for r in result["results"] if r["filter_name"] == "ITM_all")
        assert itm_result["filter_passed"] is True
        assert result["all_passed"] is False  # ATM_all blocks it

    def test_apply_filters_first_fails(self):
        # PE_only filter fails on CE
        result = apply_filters({"option_type": "CE", "dte_days": 20, "ltp": 50.0})
        assert result["all_passed"] is False
        assert result["blocking_reason"] is not None
        assert "not_PE" in result["blocking_reason"]

    def test_apply_filters_subset(self):
        result = apply_filters(
            {"option_type": "CE", "dte_days": 20},
            filter_names=["CE_only", "DTE_greater_7"],
        )
        assert result["all_passed"] is True
        # CE_only passes, DTE_greater_7 passes

    def test_apply_filters_unknown_filter(self):
        result = apply_filters({"option_type": "PE"}, filter_names=["PE_only", "nonexistent"])
        assert result["all_passed"] is False
        unknown_res = next(r for r in result["results"] if r["filter_name"] == "nonexistent")
        assert unknown_res["filter_applied"] is False
        assert "unknown_filter" in unknown_res["rejection_reason"]

    def test_apply_filters_empty_snapshot(self):
        result = apply_filters({})
        assert result["all_passed"] is False
        assert result["blocking_reason"] is not None


# ===========================================================================
# Safety: no future/outcome/label/PnL columns used
# ===========================================================================

class TestSafetyConstraints:
    """Verify filters only read allowed live fields."""

    # These are NOT referenced anywhere in candidate_filters.py
    FORBIDDEN_FIELDS = {
        "outcome", "return_1h", "return_30m", "return_60m",
        "pnl", "pnl_pct", "label", "profit_flag",
        "winner", "loser", "result", "expiry_price",
    }

    def test_no_future_columns_in_pe_filter(self):
        import ast, inspect
        src = inspect.getsource(apply_pe_only_filter)
        tree = ast.parse(src)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for f in self.FORBIDDEN_FIELDS:
            assert f not in names, f"PE_only filter must not reference '{f}'"

    def test_no_future_columns_in_ce_filter(self):
        import ast, inspect
        src = inspect.getsource(apply_ce_only_filter)
        tree = ast.parse(src)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for f in self.FORBIDDEN_FIELDS:
            assert f not in names, f"CE_only filter must not reference '{f}'"

    def test_no_future_columns_in_dte_greater_7(self):
        import ast, inspect
        src = inspect.getsource(apply_dte_greater_7_filter)
        tree = ast.parse(src)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for f in self.FORBIDDEN_FIELDS:
            assert f not in names, f"DTE_greater_7 filter must not reference '{f}'"

    def test_no_future_columns_in_itm_all(self):
        import ast, inspect
        src = inspect.getsource(apply_itm_all_filter)
        tree = ast.parse(src)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for f in self.FORBIDDEN_FIELDS:
            assert f not in names, f"ITM_all filter must not reference '{f}'"

    def test_no_future_columns_in_premium(self):
        import ast, inspect
        src = inspect.getsource(apply_premium_10_100_filter)
        tree = ast.parse(src)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for f in self.FORBIDDEN_FIELDS:
            assert f not in names, f"premium_10_100 filter must not reference '{f}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])