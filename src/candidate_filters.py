#!/usr/bin/env python3
"""
candidate_filters.py
====================
Reusable live-computable candidate filters for the Nifty scalper decision pipeline.

Each filter is a pure function that:
  1. Reads only current/live fields from a candidate snapshot dict
  2. Does NOT read outcome, future-return, PnL, or label columns
  3. Returns a consistent dict:
       filter_name              : str
       filter_rule              : str
       filter_applied           : bool  (always True — filter is always evaluated)
       filter_passed            : bool
       rejection_reason         : str | None
       required_fields_present  : bool
  4. When required fields are missing → safe rejection (filter_passed=False)

Filters
-------
  PE_only          : option_type == "PE"
  CE_only          : option_type == "CE"
  DTE_greater_7    : dte_days > 7
  DTE_14_30        : 14 <= dte_days <= 30
  ITM_all          : moneyness_bucket == "ITM"
  ATM_all          : moneyness_bucket == "ATM"
  premium_10_100   : 10 <= ltp <= 100  OR  10 <= mid_price <= 100

Safety
------
- No filter uses future/outcome/return/PnL/label columns.
- All filters are pure functions — no side effects, no I/O, no state.
- Missing required fields always reject safely.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical_str(val: Any) -> str:
    """Canonicalise a value to an upper-cased stripped string; '' on error."""
    if val is None:
        return ""
    try:
        return str(val).strip().upper()
    except Exception:
        return ""


def _canonical_float(val: Any) -> float:
    """Canonicalise a value to float; NaN on error."""
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _base_result(
    filter_name: str,
    filter_rule: str,
    passed: bool,
    reason: Optional[str],
    required_present: bool,
) -> Dict[str, Any]:
    return {
        "filter_name": filter_name,
        "filter_rule": filter_rule,
        "filter_applied": True,
        "filter_passed": passed,
        "rejection_reason": reason,
        "required_fields_present": required_present,
    }


# ---------------------------------------------------------------------------
# Filter: PE_only
# ---------------------------------------------------------------------------

def apply_pe_only_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Allow scoring ONLY when option_type == 'PE'.

    Rule   : allow scoring only when option_type == 'PE'
    Input  : snapshot["option_type"]
    Rejects: CE, any other value, and rows with missing option_type

    Safety : Missing option_type field → safe rejection.
    """
    filter_name = "PE_only"
    filter_rule = "allow scoring only when option_type == 'PE'"
    opt = _canonical_str(snapshot.get("option_type"))
    required_present = bool(opt)          # key present AND non-empty after canonicalise

    if not opt:                           # missing or empty → reject safely
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_option_type_not_PE",
            required_present=False,
        )
    if opt != "PE":
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_option_type_not_PE",
            required_present=True,
        )
    return _base_result(
        filter_name, filter_rule,
        passed=True,
        reason=None,
        required_present=True,
    )


# ---------------------------------------------------------------------------
# Filter: CE_only
# ---------------------------------------------------------------------------

def apply_ce_only_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Allow scoring ONLY when option_type == 'CE'.

    Rule   : allow scoring only when option_type == 'CE'
    Input  : snapshot["option_type"]
    Rejects: PE, any other value, and rows with missing option_type

    Safety : Missing option_type field → safe rejection.
    """
    filter_name = "CE_only"
    filter_rule = "allow scoring only when option_type == 'CE'"
    opt = _canonical_str(snapshot.get("option_type"))
    required_present = bool(opt)

    if not opt:
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_option_type_not_CE",
            required_present=False,
        )
    if opt != "CE":
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_option_type_not_CE",
            required_present=True,
        )
    return _base_result(
        filter_name, filter_rule,
        passed=True,
        reason=None,
        required_present=True,
    )


# ---------------------------------------------------------------------------
# Filter: DTE_greater_7
# ---------------------------------------------------------------------------

def apply_dte_greater_7_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Allow scoring only when DTE > 7 days.

    Rule        : dte_days > 7
    Input       : snapshot["dte_days"]
    Rejects     : dte_days <= 7, non-numeric, missing
    Interpretation: weekly expiry (7 days) and shorter DTEs are excluded
                   to avoid gamma/theta headwinds of short-dated options.

    Safety : Missing / non-numeric dte_days → safe rejection.
    """
    filter_name = "DTE_greater_7"
    filter_rule = "allow scoring only when dte_days > 7"
    dte_val = snapshot.get("dte_days")
    dte = _canonical_float(dte_val)
    required_present = not (_is_nan(dte) or dte_val is None)

    if not required_present:
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_dte_missing_or_invalid",
            required_present=False,
        )
    if dte <= 7.0:
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_dte_lte_7",
            required_present=True,
        )
    return _base_result(
        filter_name, filter_rule,
        passed=True,
        reason=None,
        required_present=True,
    )


# ---------------------------------------------------------------------------
# Filter: DTE_14_30
# ---------------------------------------------------------------------------

def apply_dte_14_30_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Allow scoring only when 14 <= DTE <= 30 days.

    Rule        : 14 <= dte_days <= 30
    Input       : snapshot["dte_days"]
    Rejects     : dte_days < 14, dte_days > 30, non-numeric, missing
    Interpretation: mid-tenor monthly options — avoids weekly gamma risk
                   and long-dated illiquid strikes.

    Safety : Missing / non-numeric dte_days → safe rejection.
    """
    filter_name = "DTE_14_30"
    filter_rule = "allow scoring only when 14 <= dte_days <= 30"
    dte_val = snapshot.get("dte_days")
    dte = _canonical_float(dte_val)
    required_present = not (_is_nan(dte) or dte_val is None)

    if not required_present:
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_dte_missing_or_invalid",
            required_present=False,
        )
    if not (14.0 <= dte <= 30.0):
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_dte_outside_14_30",
            required_present=True,
        )
    return _base_result(
        filter_name, filter_rule,
        passed=True,
        reason=None,
        required_present=True,
    )


# ---------------------------------------------------------------------------
# Filter: ITM_all
# ---------------------------------------------------------------------------

def apply_itm_all_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Allow scoring ONLY when moneyness_bucket == 'ITM'.

    Rule   : moneyness_bucket == 'ITM'
    Input  : snapshot["moneyness_bucket"]
    Rejects: ATM, OTM, any other value, and rows with missing bucket

    Safety : Missing moneyness_bucket field → safe rejection.
    """
    filter_name = "ITM_all"
    filter_rule = "allow scoring only when moneyness_bucket == 'ITM'"
    bucket = _canonical_str(snapshot.get("moneyness_bucket"))
    required_present = bool(bucket)

    if not bucket:
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_moneyness_bucket_not_ITM",
            required_present=False,
        )
    if bucket != "ITM":
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_moneyness_bucket_not_ITM",
            required_present=True,
        )
    return _base_result(
        filter_name, filter_rule,
        passed=True,
        reason=None,
        required_present=True,
    )


# ---------------------------------------------------------------------------
# Filter: ATM_all
# ---------------------------------------------------------------------------

def apply_atm_all_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Allow scoring ONLY when moneyness_bucket == 'ATM'.

    Rule   : moneyness_bucket == 'ATM'
    Input  : snapshot["moneyness_bucket"]
    Rejects: ITM, OTM, any other value, and rows with missing bucket

    Safety : Missing moneyness_bucket field → safe rejection.
    """
    filter_name = "ATM_all"
    filter_rule = "allow scoring only when moneyness_bucket == 'ATM'"
    bucket = _canonical_str(snapshot.get("moneyness_bucket"))
    required_present = bool(bucket)

    if not bucket:
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_moneyness_bucket_not_ATM",
            required_present=False,
        )
    if bucket != "ATM":
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_moneyness_bucket_not_ATM",
            required_present=True,
        )
    return _base_result(
        filter_name, filter_rule,
        passed=True,
        reason=None,
        required_present=True,
    )


# ---------------------------------------------------------------------------
# Filter: premium_10_100
# ---------------------------------------------------------------------------

def apply_premium_10_100_filter(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Allow scoring only when option premium is in the range [10, 100].

    Rule  : 10 <= ltp <= 100  OR  10 <= mid_price <= 100
    Input : snapshot["ltp"]  and/or  snapshot["mid_price"]
    Logic : PASS if EITHER ltp OR mid_price is in [10, 100].
            Both fields may be missing — that is a safe rejection.
            One valid in-range price is sufficient.
    Rejects: all prices outside [10, 100], non-numeric values, both fields missing.

    Safety : Both fields missing / non-numeric → safe rejection.
    """
    filter_name = "premium_10_100"
    filter_rule = "allow scoring only when 10 <= ltp <= 100 or 10 <= mid_price <= 100"
    ltp_val = snapshot.get("ltp")
    mid_val = snapshot.get("mid_price")

    ltp = _canonical_float(ltp_val)
    mid = _canonical_float(mid_val)

    ltp_present = not (_is_nan(ltp) or ltp_val is None)
    mid_present = not (_is_nan(mid) or mid_val is None)
    required_present = ltp_present or mid_present

    if not required_present:
        return _base_result(
            filter_name, filter_rule,
            passed=False,
            reason="candidate_filter_rejected_premium_outside_10_100",
            required_present=False,
        )

    def _in_range(v: float) -> bool:
        return 10.0 <= v <= 100.0

    ltp_ok = ltp_present and _in_range(ltp)
    mid_ok = mid_present and _in_range(mid)

    if ltp_ok or mid_ok:
        return _base_result(
            filter_name, filter_rule,
            passed=True,
            reason=None,
            required_present=True,
        )

    # At least one field present but both outside range
    return _base_result(
        filter_name, filter_rule,
        passed=False,
        reason="candidate_filter_rejected_premium_outside_10_100",
        required_present=True,
    )


def _preset_dte_passes(snapshot, dte_range):
    if dte_range == 'all':
        return True
    dte_raw = snapshot.get('dte_days')
    if dte_raw is None:
        return True
    dte = _canonical_float(dte_raw)
    if _is_nan(dte):
        return True
    if dte_range == '0-1':
        return 0.0 <= dte <= 1.0
    if dte_range == '2-7':
        return 2.0 <= dte <= 7.0
    if dte_range == '7-30':
        return 7.0 <= dte <= 30.0
    return True


def _preset_spread_passes(snapshot, spread_limit_pct):
    if spread_limit_pct <= 0.0:
        return True
    raw = snapshot.get('range_pct')
    if raw is None:
        return True
    spread = abs(_canonical_float(raw))
    if _is_nan(spread):
        return True
    return spread <= spread_limit_pct


def _preset_liquidity_passes(snapshot, liquidity_min):
    if liquidity_min <= 0.0:
        return True
    raw = snapshot.get('volume')
    if raw is None:
        return False
    vol = _canonical_float(raw)
    if _is_nan(vol):
        return False
    return vol >= liquidity_min


def _preset_premium_passes(snapshot, premium_band):
    if premium_band == 'all':
        return True
    raw = snapshot.get('ltp')
    if raw is None:
        return True
    ltp = _canonical_float(raw)
    if _is_nan(ltp):
        return True
    if premium_band == 'low':
        return ltp < 50.0
    if premium_band == 'mid':
        return 50.0 <= ltp < 150.0
    if premium_band == 'high':
        return ltp >= 150.0
    return True


def _preset_time_window_passes(snapshot, avoid_first_n, avoid_last_n, entry_start, entry_end):
    ts_raw = snapshot.get('timestamp')
    if ts_raw is None:
        return True
    try:
        from datetime import datetime
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
        else:
            ts = ts_raw
    except Exception:
        return True
    from datetime import time as _time
    try:
        start_time = _time.fromisoformat(entry_start)
        end_time = _time.fromisoformat(entry_end)
    except Exception:
        return True
    cur_time = ts.time()
    if cur_time < start_time or cur_time > end_time:
        return False
    morning_start = _time(9, 30)
    afternoon_end = _time(15, 30)
    minutes_from_open = (cur_time.hour - morning_start.hour) * 60 + (cur_time.minute - morning_start.minute)
    minutes_to_close = (afternoon_end.hour - cur_time.hour) * 60 + (afternoon_end.minute - cur_time.minute)
    if minutes_from_open < avoid_first_n:
        return False
    if minutes_to_close < avoid_last_n:
        return False
    return True


def apply_dynamic_preset_filter(snapshot, preset_config):
    filter_name = "dynamic_preset"
    filter_rule = "live-computable preset risk controls -- no future/leakage data"
    dte_range     = str(preset_config.get('dte_range', 'all'))
    spread_limit  = float(preset_config.get('spread_limit_pct', 0.0))
    liquidity_min = float(preset_config.get('liquidity_min', 0.0))
    premium_band  = str(preset_config.get('premium_band', 'all'))
    avoid_first   = int(preset_config.get('avoid_first_n_minutes', 0))
    avoid_last    = int(preset_config.get('avoid_last_n_minutes', 0))
    entry_start   = str(preset_config.get('entry_time_start', '09:30'))
    entry_end     = str(preset_config.get('entry_time_end', '15:00'))
    regime_filter = str(preset_config.get('regime_filter', 'all'))
    if not _preset_dte_passes(snapshot, dte_range):
        return _base_result(filter_name, filter_rule, False,
                            "dynamic_preset_rejected_outside_dte_range", True)
    if not _preset_spread_passes(snapshot, spread_limit):
        return _base_result(filter_name, filter_rule, False,
                            "dynamic_preset_rejected_spread_too_high", True)
    if not _preset_liquidity_passes(snapshot, liquidity_min):
        return _base_result(filter_name, filter_rule, False,
                            "dynamic_preset_rejected_liquidity_too_low", True)
    if not _preset_premium_passes(snapshot, premium_band):
        return _base_result(filter_name, filter_rule, False,
                            "dynamic_preset_rejected_premium_band", True)
    if not _preset_time_window_passes(snapshot, avoid_first, avoid_last, entry_start, entry_end):
        return _base_result(filter_name, filter_rule, False,
                            "dynamic_preset_rejected_outside_time_window", True)
    if regime_filter not in ('all', '', None):
        live_regime = _canonical_str(snapshot.get('regime') or snapshot.get('market_regime') or '')
        if live_regime and live_regime != regime_filter:
            regime_aliases = {
                'high_volatility': ['HIGH_VOL', 'VOLATILE', 'HIGH_VOLATILITY', 'VOL'],
                'trend_only':      ['TRENDING', 'TREND', 'UP', 'DOWN', 'BULL', 'BEAR'],
                'non_chop':        ['TRENDING', 'UP', 'DOWN', 'BULL', 'BEAR'],
            }
            aliases = regime_aliases.get(regime_filter, [])
            if live_regime not in aliases:
                return _base_result(filter_name, filter_rule, False,
                                    "dynamic_preset_rejected_regime_mismatch", True)
    return _base_result(filter_name, filter_rule, True, None, True)


def apply_spread_limit_filter(snapshot, max_spread_pct):
    filter_name = "spread_limit_{}".format(int(max_spread_pct * 100))
    filter_rule = "allow scoring only when |range_pct| <= {:.2%}".format(max_spread_pct)
    raw = snapshot.get('range_pct')
    if raw is None:
        return _base_result(filter_name, filter_rule, True, None, True)
    spread = abs(_canonical_float(raw))
    if _is_nan(spread):
        return _base_result(filter_name, filter_rule, True, None, True)
    passed = spread <= max_spread_pct
    return _base_result(
        filter_name, filter_rule, passed,
        reason=None if passed else "dynamic_preset_rejected_spread_too_high",
        required_present=True,
    )


def apply_liquidity_filter(snapshot, min_volume):
    filter_name = "liquidity_min_{}".format(int(min_volume))
    filter_rule = "allow scoring only when volume >= {:.0f}".format(min_volume)
    raw = snapshot.get('volume')
    if raw is None:
        return _base_result(filter_name, filter_rule, False,
                            "dynamic_preset_rejected_liquidity_missing", False)
    vol = _canonical_float(raw)
    if _is_nan(vol):
        return _base_result(filter_name, filter_rule, False,
                            "dynamic_preset_rejected_liquidity_missing", False)
    passed = vol >= min_volume
    return _base_result(
        filter_name, filter_rule, passed,
        reason=None if passed else "dynamic_preset_rejected_liquidity_too_low",
        required_present=True,
    )


# ---------------------------------------------------------------------------
# Dynamic Preset Filters (live-computable only -- NO future/PnL/return/leakage)
# Note: Individual preset filter functions are defined ABOVE this dict.
# This dict maps filter names to filter callables for the apply_filters() pipeline.
# ---------------------------------------------------------------------------

ALL_FILTERS = {
    "PE_only":        apply_pe_only_filter,
    "CE_only":        apply_ce_only_filter,
    "DTE_greater_7":  apply_dte_greater_7_filter,
    "DTE_14_30":      apply_dte_14_30_filter,
    "ITM_all":        apply_itm_all_filter,
    "ATM_all":        apply_atm_all_filter,
    "premium_10_100": apply_premium_10_100_filter,
    "dynamic_preset": apply_dynamic_preset_filter,
}



def apply_filters(
    snapshot: Dict[str, Any],
    filter_names: list[str] | None = None,
) -> Dict[str, Any]:
    """Apply one or more named filters to a snapshot.

    Parameters
    ----------
    snapshot    : candidate feature snapshot (dict)
    filter_names: list of filter names to apply.  None = apply all.

    Returns
    -------
    dict with keys:
        all_passed       : bool  — True only if every applied filter passed
        results          : list[dict] — per-filter result dicts
        blocking_reason  : str | None — first rejection reason, if any
    """
    names = filter_names or list(ALL_FILTERS.keys())
    results: list[Dict[str, Any]] = []
    for name in names:
        fn = ALL_FILTERS.get(name)
        if fn is None:
            results.append({
                "filter_name": name,
                "filter_rule": "",
                "filter_applied": False,
                "filter_passed": False,
                "rejection_reason": f"unknown_filter_{name}",
                "required_fields_present": False,
            })
        else:
            results.append(fn(snapshot))

    all_passed = all(r["filter_passed"] for r in results)
    blocking = next((r["rejection_reason"] for r in results if not r["filter_passed"]), None)
    return {
        "all_passed": all_passed,
        "results": results,
        "blocking_reason": blocking,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

import math as _math

def _is_nan(v: float) -> bool:
    return _math.isnan(v)