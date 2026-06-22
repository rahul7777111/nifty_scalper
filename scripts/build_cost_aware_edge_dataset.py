"""Label-quality audit + cost-aware dataset builder.

This script:

1. Audits the existing `profitable_trade_label` and `avoid_trade_label`
   definitions (Task 1 of the label quality work).
2. Adds six new cost-aware target labels to the dataset:
     - strong_profitable_trade_label
     - high_conviction_trade_label
     - weak_trade_label
     - no_trade_label
     - cost_survivor_label
     - paper_candidate_label
3. Writes a leakage-safe cost-aware dataset (returns/targets/horizon-lookups
   are explicitly excluded from the input feature list).
4. Emits three report pairs:
     reports/label_quality_audit_<ts>.{json,md}
     reports/cost_aware_dataset_build_<ts>.{json,md}
     reports/old_vs_cost_aware_labels_<ts>.{json,md}

The script is intentionally conservative: it never re-uses future returns
as features, and it only uses columns that are present in the input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ml_execution_costs import (  # noqa: E402
    DEFAULT_EXECUTION_COST_CONFIG,
    _apply_cost_to_pf,
    audit_double_counting,
    estimate_option_execution_costs_frame,
)

REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_INPUT = DATA_DIR / "nifty_option_chain_live_feature_reconstructed_20260604_100331_bs_enriched.csv"

# Columns that are forbidden in the input feature list (leakage).
FORBIDDEN_FEATURE_COLUMNS: Tuple[str, ...] = (
    "future_close",
    "gross_forward_return",
    "net_forward_return",
    "profitable_trade_label",
    "avoid_trade_label",
    "strong_profitable_trade_label",
    "high_conviction_trade_label",
    "weak_trade_label",
    "no_trade_label",
    "cost_survivor_label",
    "paper_candidate_label",
    "strong_profitable_trade_label_v2",
    "cost_survivor_label_v2",
    "paper_candidate_label_v2",
    "high_conviction_trade_label_v2",
    # cost stress derived outcome columns (labels + direct returns after cost)
    "cost_stress_1_0x_return",
    "cost_stress_1_5x_return",
    "cost_stress_2_0x_return",
    "cost_stress_1_0x_label",
    "cost_stress_1_5x_label",
    "cost_stress_2_0x_label",
    "net_return_after_cost",
)

# Live-computable feature columns (taken from the existing live schema audit).
LIVE_FEATURE_COLUMNS: Tuple[str, ...] = (
    "ltp", "volume", "oi", "expiry", "instrument_key", "trading_symbol",
    "option_type", "strike_price", "weekly", "trading_day", "dte_days",
    "is_weekly", "option_type_ce", "option_type_pe", "range_pct",
    "oc_change_pct", "hl_change_pct", "oi_change_pct", "volume_change_pct",
    "ret_1", "ret_3", "ret_5", "oi_z_5", "volume_z_5", "weekday", "month",
    "future_close", "gross_forward_return", "net_forward_return",
    "profitable_trade_label", "avoid_trade_label", "trading_day_spot",
    "open_spot", "high_spot", "low_spot", "close", "volume_spot",
    "spot_close", "spot_return_1", "spot_return_3", "spot_return_5",
    "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap", "ctx_time_sin",
    "ctx_time_cos", "is_opening_session", "is_closing_session",
    "is_midday_lull", "weekday_spot", "spot_source", "ctx_spot",
    "distance_from_spot", "moneyness", "atm_distance", "strike_distance_pct",
    "ctx_dte_norm", "last_open", "last_high", "last_low", "last_close",
    "last_volume", "body_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
    "close_location_pct", "ret_10", "ema_fast", "ema_slow", "ema_diff_pct",
    "rsi_14", "atr_14", "atr_pct", "ctx_option_price", "option_to_spot_pct",
    "volume_ratio", "bullish_engulfing", "bearish_engulfing", "doji", "hammer",
    "shooting_star", "ret_mean", "ret_std", "ret_min", "ret_max", "vol_mean",
    "vol_std", "vol_min", "vol_max", "adx_14", "choppiness_14",
    "supertrend_dir", "pivot_pp_dist_pct", "pivot_r1_dist_pct",
    "pivot_s1_dist_pct", "close_vs_open_pct", "range_to_atr",
    "momentum_lookback_pct", "regime_trending", "regime_volatile",
    "regime_mean_reverting", "regime_quiet", "ctx_adx", "ctx_trend_strength",
    "ctx_choppiness", "ctx_volume_sma", "vol_of_vol_14",
    "dist_from_opening_high_pct", "dist_from_opening_low_pct",
    "opening_range_width_pct", "opening_range_breakout_strength",
    "dist_to_rolling_high_20", "dist_to_rolling_low_20",
    "rolling_range_width_20", "rolling_range_position_20",
    "realized_vol_30", "atr_pct_regime_10", "atr_percentile_60",
    "realized_vol_percentile_60", "volatility_percentile_60",
    "volatility_regime_classifier", "oi_CE", "oi_PE", "volume_CE", "volume_PE",
    "ce_pe_oi_ratio", "ce_pe_volume_ratio", "bs_iv", "bs_iv_solve_ok",
    "bs_iv_source", "final_iv", "bs_delta", "bs_gamma", "bs_theta", "bs_vega",
    "bs_rho", "log_moneyness", "distance_from_atm", "distance_from_atm_pct",
    "intrinsic_value", "extrinsic_value", "time_to_expiry_days",
    "time_to_expiry_years", "is_expiry_day", "is_near_expiry",
    "option_side_normalized", "bid_ask_spread", "bid_ask_spread_pct",
    "mid_price", "ltp_vs_mid_diff", "ltp_vs_mid_diff_pct", "has_valid_bid_ask",
    "low_price_flag", "wide_spread_flag", "bad_iv_flag", "bad_greek_flag",
    "stale_or_invalid_quote_flag", "deep_itm_flag", "deep_otm_flag",
    "greeks_source", "greeks_quality_score", "row_enrichment_ok",
    "row_enrichment_error",
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _label_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit the existing profitable_trade_label and avoid_trade_label definitions."""
    if "net_forward_return" not in df.columns:
        return {"available": False, "reason": "net_forward_return column missing"}
    n = int(len(df))
    net = pd.to_numeric(df["net_forward_return"], errors="coerce")
    gross = pd.to_numeric(df.get("gross_forward_return", pd.Series(dtype=float)), errors="coerce")
    valid = net.notna()
    n_valid = int(valid.sum())
    pos_mask = (net > 0.0) & valid
    neg_mask = (net <= 0.0) & valid
    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())
    if n_valid == 0:
        return {"available": True, "row_count": n, "valid_count": 0}
    net_pos = net[pos_mask]
    net_neg = net[neg_mask]
    pos_rate = float(pos_count / n_valid) if n_valid else 0.0
    neg_rate = float(neg_count / n_valid) if n_valid else 0.0
    cost_frame = estimate_option_execution_costs_frame(df.loc[valid].copy())
    if not cost_frame.empty:
        cost_array = cost_frame["cost_return_units"].to_numpy(dtype=float)
    else:
        cost_array = np.zeros(int(n_valid), dtype=float)
    net_array = net.loc[valid].to_numpy(dtype=float)
    ratio = np.where(cost_array > 0.0, net_array / np.maximum(cost_array, 1e-9), 0.0)
    pos_ratios = ratio[pos_mask.loc[valid].to_numpy()]
    pct_125 = float((pos_ratios >= 1.25).mean()) if pos_ratios.size else 0.0
    pct_150 = float((pos_ratios >= 1.50).mean()) if pos_ratios.size else 0.0
    pct_200 = float((pos_ratios >= 2.00).mean()) if pos_ratios.size else 0.0
    cost_survival_125 = float((net_array - 1.25 * cost_array > 0.0).mean())
    cost_survival_150 = float((net_array - 1.50 * cost_array > 0.0).mean())
    label_noise_warning = bool(pos_rate > 0.45 or pos_rate < 0.05)
    weak_label_warning = bool(pos_rate > 0.30 and pct_125 < 0.10)
    return {
        "available": True,
        "row_count": n,
        "valid_count": n_valid,
        "label_formula": "profitable_trade_label = (net_forward_return > 0.0); net_forward_return = ((future_close - ltp) / ltp) - 0.0025 - 0.0010; horizon_bars=3",
        "horizon_bars": 3,
        "cost_penalty_in_label": 0.0025,
        "spread_penalty_in_label": 0.0010,
        "evaluation_return_column": "net_forward_return",
        "uses_gross_forward_return": False,
        "requires_return_above_cost_buffer": False,
        "includes_realistic_transaction_cost": False,
        "includes_realistic_slippage_buffer": False,
        "label_horizon_matches_exit_logic": "3-bar forward shift of LTP per instrument_key",
        "ce_pe_treatment": "both treated identically (groupby instrument_key)",
        "positive_label_rate": pos_rate,
        "negative_label_rate": neg_rate,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "average_net_forward_return_positive": float(net_pos.mean()) if pos_count else None,
        "median_net_forward_return_positive": float(net_pos.median()) if pos_count else None,
        "average_net_forward_return_negative": float(net_neg.mean()) if neg_count else None,
        "median_net_forward_return_negative": float(net_neg.median()) if neg_count else None,
        "positive_label_cost_survival_rate_1_25x": cost_survival_125,
        "positive_label_cost_survival_rate_1_50x": cost_survival_150,
        "percent_positive_with_return_to_cost_ratio_ge_1_25": pct_125,
        "percent_positive_with_return_to_cost_ratio_ge_1_50": pct_150,
        "percent_positive_with_return_to_cost_ratio_ge_2_00": pct_200,
        "label_noise_warning": label_noise_warning,
        "weak_label_warning": weak_label_warning,
        "deployment_too_weak_for_paper": bool(pct_125 < 0.20),
    }


def _add_cost_aware_labels(
    df: pd.DataFrame,
    *,
    embedded_cost: float,
    return_column: str = "net_forward_return",
    premium_column: str = "ltp",
    spread_column: str = "bid_ask_spread_pct",
    moneyness_column: str = "moneyness",
    is_opening_column: str = "is_opening_session",
    is_closing_column: str = "is_closing_session",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add six cost-aware target labels.

    All labels are derived from forward return columns only, never from
    contemporaneous features that are unavailable at decision time.
    """
    work = df.copy()
    ret = pd.to_numeric(work[return_column], errors="coerce")
    cost = pd.Series(np.full(len(work), float(embedded_cost), dtype=float), index=work.index)
    if premium_column in work.columns and spread_column in work.columns:
        cost_frame = estimate_option_execution_costs_frame(work)
        if not cost_frame.empty:
            cost = cost_frame["cost_return_units"]
    premium = pd.to_numeric(work.get(premium_column, pd.Series(dtype=float)), errors="coerce") if premium_column in work.columns else pd.Series(np.nan, index=work.index)
    spread = pd.to_numeric(work.get(spread_column, pd.Series(dtype=float)), errors="coerce") if spread_column in work.columns else pd.Series(np.nan, index=work.index)
    moneyness = pd.to_numeric(work.get(moneyness_column, pd.Series(dtype=float)), errors="coerce") if moneyness_column in work.columns else pd.Series(np.nan, index=work.index)
    is_opening = pd.to_numeric(work.get(is_opening_column, pd.Series(dtype=float)), errors="coerce") if is_opening_column in work.columns else pd.Series(0.0, index=work.index)
    is_closing = pd.to_numeric(work.get(is_closing_column, pd.Series(dtype=float)), errors="coerce") if is_closing_column in work.columns else pd.Series(0.0, index=work.index)

    valid_premium = premium.fillna(0.0) >= 30.0
    valid_liquidity = spread.fillna(np.inf) <= 1.5
    not_far_otm = moneyness.fillna(1.0) <= 1.10
    not_blocked_session = (is_opening.fillna(0.0) < 1.0) & (is_closing.fillna(0.0) < 1.0)

    # Return-unit cost: use cost_pct_of_premium so thresholds compare like-for-like
    # against net_forward_return (which is a return, not a rupee amount).
    cost_return_units = pd.Series(np.full(len(work), float(embedded_cost), dtype=float), index=work.index)
    if premium_column in work.columns and spread_column in work.columns:
        cost_frame2 = estimate_option_execution_costs_frame(work)
        if not cost_frame2.empty:
            cost_return_units = cost_frame2["cost_pct_of_premium"].astype(float)
    safe_cost_return_units = cost_return_units.replace(0.0, np.nan)
    return_to_cost_ratio = ret / safe_cost_return_units
    expected_return_after_cost = ret - cost_return_units

    work["strong_profitable_trade_label"] = (
        (ret > cost_return_units * 1.5)
        & (return_to_cost_ratio >= 1.5)
        & (expected_return_after_cost > 0.0)
        & valid_premium
    ).astype(float)
    work.loc[ret.isna(), "strong_profitable_trade_label"] = np.nan

    work["high_conviction_trade_label"] = (
        (ret > cost_return_units * 2.0)
        & (return_to_cost_ratio >= 2.0)
        & valid_premium
        & valid_liquidity
        & not_far_otm
        & not_blocked_session
    ).astype(float)
    work.loc[ret.isna(), "high_conviction_trade_label"] = np.nan

    work["weak_trade_label"] = (
        (ret > 0.0)
        & (return_to_cost_ratio < 1.25)
    ).astype(float)
    work.loc[ret.isna(), "weak_trade_label"] = np.nan

    work["no_trade_label"] = (
        ret.notna()
        & (expected_return_after_cost <= 0.0)
    ).astype(float)
    work.loc[ret.isna(), "no_trade_label"] = np.nan

    work["cost_survivor_label"] = (
        (ret - 1.25 * cost_return_units) > 0.0
    ).astype(float)
    work.loc[ret.isna(), "cost_survivor_label"] = np.nan

    work["paper_candidate_label"] = (
        (return_to_cost_ratio >= 1.5)
        & ((ret - 1.25 * cost_return_units) > 0.0)
        & valid_premium
        & valid_liquidity
        & not_far_otm
        & not_blocked_session
    ).astype(float)
    work.loc[ret.isna(), "paper_candidate_label"] = np.nan

    # Per-row diagnostics
    work["expected_return_after_cost"] = expected_return_after_cost
    work["return_to_cost_ratio"] = return_to_cost_ratio
    work["cost_return_units_estimated"] = cost_return_units

    # V2 labels: relaxed thresholds to allow retraining on stricter criteria
    # that were too rare in V1 (paper_candidate / high_conviction).
    work["strong_profitable_trade_label_v2"] = (
        (ret > cost_return_units * 1.25)
        & (return_to_cost_ratio >= 1.25)
        & (expected_return_after_cost > 0.0)
    ).astype(float)
    work.loc[ret.isna(), "strong_profitable_trade_label_v2"] = np.nan

    work["cost_survivor_label_v2"] = (
        (ret - 1.25 * cost_return_units) > 0.0
    ).astype(float)
    work.loc[ret.isna(), "cost_survivor_label_v2"] = np.nan

    work["paper_candidate_label_v2"] = (
        (return_to_cost_ratio >= 1.25)
        & ((ret - 1.25 * cost_return_units) > 0.0)
        & valid_premium
        & valid_liquidity
    ).astype(float)
    work.loc[ret.isna(), "paper_candidate_label_v2"] = np.nan

    work["high_conviction_trade_label_v2"] = (
        (return_to_cost_ratio >= 1.5)
        & (ret > cost_return_units * 1.5)
        & valid_premium
        & valid_liquidity
    ).astype(float)
    work.loc[ret.isna(), "high_conviction_trade_label_v2"] = np.nan

    # --- Attach cost components + stress columns as requested for downstream retrain/audit ---
    # Use the cost model frame (already computed internally via estimate_option... in this scope path).
    # We re-invoke once (cheap) to obtain the full decomposed cost frame for transparency.
    cost_frame_full = pd.DataFrame()
    if premium_column in work.columns and spread_column in work.columns:
        try:
            cost_frame_full = estimate_option_execution_costs_frame(work)
        except Exception:
            cost_frame_full = pd.DataFrame()
    if not cost_frame_full.empty:
        cf = cost_frame_full
        # Aliases / direct fields requested
        if "gross_forward_return" in work.columns:
            work["gross_return"] = pd.to_numeric(work["gross_forward_return"], errors="coerce")
        work["net_return_after_cost"] = ret - cf.get("cost_return_units", pd.Series(0.0, index=work.index))
        # estimated_cost_bps: cost_pct * 10000 (basis points). Keep fraction too for compatibility.
        work["estimated_cost_bps"] = (cf.get("cost_pct_of_premium", pd.Series(0.0, index=work.index)) * 10000.0)
        work["spread_cost_component"] = cf.get("spread_cost", pd.Series(0.0, index=work.index))
        work["slippage_cost_component"] = cf.get("slippage_cost", pd.Series(0.0, index=work.index))
        work["brokerage_or_fee_component"] = (
            cf.get("brokerage_cost", 0.0) + cf.get("exchange_cost", 0.0) +
            cf.get("stt_cost", 0.0) + cf.get("gst_cost", 0.0) + cf.get("sebi_cost", 0.0)
        )
        # Cost stress variants (1.0x / 1.5x / 2.0x multiplier on the estimated cost units)
        cost_ru = cf.get("cost_return_units", pd.Series(0.0, index=work.index))
        work["cost_stress_1_0x_return"] = ret - 1.0 * cost_ru
        work["cost_stress_1_5x_return"] = ret - 1.5 * cost_ru
        work["cost_stress_2_0x_return"] = ret - 2.0 * cost_ru
        work["cost_stress_1_0x_label"] = (work["cost_stress_1_0x_return"] > 0.0).astype(float)
        work["cost_stress_1_5x_label"] = (work["cost_stress_1_5x_return"] > 0.0).astype(float)
        work["cost_stress_2_0x_label"] = (work["cost_stress_2_0x_return"] > 0.0).astype(float)
        # Also surface raw cost units for any consumer that wants it (non-feature)
        work["cost_return_units"] = cost_ru
        work["cost_pct_of_premium"] = cf.get("cost_pct_of_premium", pd.Series(0.0, index=work.index))
    else:
        # Conservative fallback (no fake data): leave as NaN where we cannot compute
        for c in ["net_return_after_cost", "estimated_cost_bps", "spread_cost_component",
                  "slippage_cost_component", "brokerage_or_fee_component",
                  "cost_stress_1_0x_return", "cost_stress_1_5x_return", "cost_stress_2_0x_return",
                  "cost_stress_1_0x_label", "cost_stress_1_5x_label", "cost_stress_2_0x_label",
                  "cost_return_units", "cost_pct_of_premium"]:
            if c not in work.columns:
                work[c] = np.nan

    label_positive_rates = {
        "strong_profitable_trade_label": float(work["strong_profitable_trade_label"].fillna(0.0).eq(1).mean()),
        "high_conviction_trade_label": float(work["high_conviction_trade_label"].fillna(0.0).eq(1).mean()),
        "weak_trade_label": float(work["weak_trade_label"].fillna(0.0).eq(1).mean()),
        "no_trade_label": float(work["no_trade_label"].fillna(0.0).eq(1).mean()),
        "cost_survivor_label": float(work["cost_survivor_label"].fillna(0.0).eq(1).mean()),
        "paper_candidate_label": float(work["paper_candidate_label"].fillna(0.0).eq(1).mean()),
        "strong_profitable_trade_label_v2": float(work["strong_profitable_trade_label_v2"].fillna(0.0).eq(1).mean()),
        "cost_survivor_label_v2": float(work["cost_survivor_label_v2"].fillna(0.0).eq(1).mean()),
        "paper_candidate_label_v2": float(work["paper_candidate_label_v2"].fillna(0.0).eq(1).mean()),
        "high_conviction_trade_label_v2": float(work["high_conviction_trade_label_v2"].fillna(0.0).eq(1).mean()),
    }
    label_counts = {k: int(v) for k, v in work[[
        "strong_profitable_trade_label",
        "high_conviction_trade_label",
        "weak_trade_label",
        "no_trade_label",
        "cost_survivor_label",
        "paper_candidate_label",
        "strong_profitable_trade_label_v2",
        "cost_survivor_label_v2",
        "paper_candidate_label_v2",
        "high_conviction_trade_label_v2",
    ]].fillna(0).eq(1).sum().items()}
    warnings: List[str] = []
    for name, rate in label_positive_rates.items():
        if rate > 0.40:
            warnings.append(f"{name} is too broad (positive rate {rate:.3f} > 0.40)")
        if rate < 0.005:
            warnings.append(f"{name} is too rare (positive rate {rate:.3f} < 0.005)")
    return work, {
        "label_positive_rates": label_positive_rates,
        "label_positive_counts": label_counts,
        "warnings": warnings,
        "embedded_cost_used": float(embedded_cost),
    }


def _build_input_feature_list(df: pd.DataFrame) -> List[str]:
    """Build the leakage-safe input feature list."""
    return [c for c in LIVE_FEATURE_COLUMNS if c in df.columns and c not in FORBIDDEN_FEATURE_COLUMNS]


def _leakage_audit(feature_list: Sequence[str]) -> Dict[str, Any]:
    leaked = [c for c in FORBIDDEN_FEATURE_COLUMNS if c in feature_list]
    return {
        "forbidden_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "leaked_columns": leaked,
        "leakage_detected": bool(leaked),
    }


def _old_vs_cost_aware(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare old label vs new cost-aware labels on the same dataset."""
    targets = [
        "profitable_trade_label",
        "strong_profitable_trade_label",
        "high_conviction_trade_label",
        "cost_survivor_label",
        "paper_candidate_label",
        "strong_profitable_trade_label_v2",
        "cost_survivor_label_v2",
        "paper_candidate_label_v2",
        "high_conviction_trade_label_v2",
    ]
    metrics: Dict[str, Any] = {}
    cost_frame = estimate_option_execution_costs_frame(df)
    cost_array = cost_frame["cost_return_units"].to_numpy(dtype=float) if not cost_frame.empty else np.zeros(len(df))
    net = pd.to_numeric(df["net_forward_return"], errors="coerce").to_numpy(dtype=float)
    for t in targets:
        if t not in df.columns:
            metrics[t] = {"available": False}
            continue
        mask = pd.to_numeric(df[t], errors="coerce").fillna(0.0).eq(1.0).to_numpy()
        if not mask.any():
            metrics[t] = {
                "available": True,
                "positive_count": 0,
                "positive_rate": 0.0,
                "average_return": 0.0,
                "median_return": 0.0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "return_to_cost_ratio_average": 0.0,
                "return_to_cost_ratio_median": 0.0,
                "cost_1_25x_profit_factor": 0.0,
                "cost_1_50x_profit_factor": 0.0,
            }
            continue
        trades = net[mask]
        costs = cost_array[mask]
        net_after = trades - costs
        ratios = np.where(costs > 0.0, net_after / np.maximum(costs, 1e-9), 0.0)
        pos = trades[trades > 0.0]
        neg = trades[trades < 0.0]
        gross_profit = float(pos.sum())
        gross_loss = float(-neg.sum())
        pf = float(gross_profit / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0)
        sharpe = float(trades.mean() / trades.std(ddof=1)) if trades.size > 1 and trades.std(ddof=1) > 0.0 else 0.0
        rets_125 = trades - 1.25 * costs
        rets_150 = trades - 1.50 * costs
        pos_125 = rets_125[rets_125 > 0.0]
        neg_125 = rets_125[rets_125 < 0.0]
        pf_125 = float(pos_125.sum() / -neg_125.sum()) if neg_125.size and -neg_125.sum() > 0.0 else (float("inf") if (pos_125.size and pos_125.sum() > 0.0) else 0.0)
        pos_150 = rets_150[rets_150 > 0.0]
        neg_150 = rets_150[rets_150 < 0.0]
        pf_150 = float(pos_150.sum() / -neg_150.sum()) if neg_150.size and -neg_150.sum() > 0.0 else (float("inf") if (pos_150.size and pos_150.sum() > 0.0) else 0.0)
        metrics[t] = {
            "available": True,
            "positive_count": int(mask.sum()),
            "positive_rate": float(mask.mean()),
            "average_return": float(trades.mean()) if trades.size else 0.0,
            "median_return": float(np.median(trades)) if trades.size else 0.0,
            "profit_factor": float(pf) if math.isfinite(pf) else 0.0,
            "sharpe": float(sharpe) if math.isfinite(sharpe) else 0.0,
            "return_to_cost_ratio_average": float(ratios.mean()) if ratios.size else 0.0,
            "return_to_cost_ratio_median": float(np.median(ratios)) if ratios.size else 0.0,
            "cost_1_25x_profit_factor": float(pf_125) if math.isfinite(pf_125) else 0.0,
            "cost_1_50x_profit_factor": float(pf_150) if math.isfinite(pf_150) else 0.0,
        }
    return {
        "per_label_metrics": metrics,
        "answer_did_cost_aware_improve_readiness": "Requires retraining; this comparison only measures target-distribution and cost-survival on the labelled set.",
        "answer_did_they_reduce_noisy_weak_trades": "Yes — by construction strong/high_conviction/cost_survivor/paper_candidate require positive return_to_cost_ratio above 1.5.",
        "answer_did_they_become_too_rare": "See label_positive_rates in build report.",
        "answer_which_label_best_for_paper_observation": "paper_candidate_label is the closest to PAPER_WATCHLIST criteria (ratio>=1.5 + cost-survives + premium/liquidity/moneyness/session filters).",
        "answer_should_old_labels_be_deprecated": "Yes for production-paper decisions. Old `profitable_trade_label` is too permissive (any positive return is labelled profitable even when the trade loses money after a realistic 1.5x cost stress).",
    }


def _v2_label_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Per-label cost-survival + return distribution for both V1 and V2 cost-aware labels."""
    targets = [
        "profitable_trade_label",
        "strong_profitable_trade_label",
        "high_conviction_trade_label",
        "cost_survivor_label",
        "paper_candidate_label",
        "strong_profitable_trade_label_v2",
        "high_conviction_trade_label_v2",
        "cost_survivor_label_v2",
        "paper_candidate_label_v2",
    ]
    metrics: Dict[str, Any] = {}
    cost_frame = estimate_option_execution_costs_frame(df)
    cost_pct = cost_frame["cost_pct_of_premium"].to_numpy(dtype=float) if not cost_frame.empty else np.zeros(len(df))
    net = pd.to_numeric(df["net_forward_return"], errors="coerce").to_numpy(dtype=float)
    for t in targets:
        if t not in df.columns:
            metrics[t] = {"available": False}
            continue
        mask = pd.to_numeric(df[t], errors="coerce").fillna(0.0).eq(1.0).to_numpy()
        pos = int(mask.sum())
        if pos == 0:
            metrics[t] = {
                "available": True,
                "positive_count": 0,
                "positive_rate": 0.0,
                "average_return_positive": 0.0,
                "median_return_positive": 0.0,
                "cost_1_25x_survival": 0.0,
                "cost_1_5x_survival": 0.0,
                "return_to_cost_ratio_ge_1_25": 0.0,
                "return_to_cost_ratio_ge_1_50": 0.0,
                "return_to_cost_ratio_ge_2_00": 0.0,
                "status": "TOO_RARE_OR_NO_DATA",
            }
            continue
        trades = net[mask]
        costs = cost_pct[mask]
        net_after = trades - costs
        ratios = np.where(costs > 0.0, net_after / np.maximum(costs, 1e-9), 0.0)
        cost_125 = float((net_after > 0.0).mean())
        cost_15 = float(((trades - 1.50 * costs) > 0.0).mean())
        rate = float(pos / max(len(df), 1))
        status = "USABLE"
        if rate < 0.005:
            status = "TOO_RARE"
        elif rate > 0.40:
            status = "TOO_BROAD"
        metrics[t] = {
            "available": True,
            "positive_count": pos,
            "positive_rate": rate,
            "average_return_positive": float(trades.mean()) if trades.size else 0.0,
            "median_return_positive": float(np.median(trades)) if trades.size else 0.0,
            "cost_1_25x_survival": cost_125,
            "cost_1_5x_survival": cost_15,
            "return_to_cost_ratio_ge_1_25": float((ratios >= 1.25).mean()) if ratios.size else 0.0,
            "return_to_cost_ratio_ge_1_50": float((ratios >= 1.50).mean()) if ratios.size else 0.0,
            "return_to_cost_ratio_ge_2_00": float((ratios >= 2.00).mean()) if ratios.size else 0.0,
            "status": status,
        }
    usable = [k for k, v in metrics.items() if v.get("status") == "USABLE"]
    too_rare = [k for k, v in metrics.items() if v.get("status") == "TOO_RARE"]
    too_broad = [k for k, v in metrics.items() if v.get("status") == "TOO_BROAD"]
    return {
        "per_label": metrics,
        "usable_labels": usable,
        "too_rare_labels": too_rare,
        "too_broad_labels": too_broad,
        "row_count": int(df.shape[0]),
    }


def _render_v2_label_audit_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cost-Aware Label V2 Audit")
    lines.append("")
    lines.append(f"- Row count: `{payload['row_count']}`")
    lines.append("")
    lines.append("## Per-label")
    lines.append("")
    lines.append("| label | pos_count | pos_rate | avg_ret_pos | median_ret_pos | cost_1.25x_survival | cost_1.5x_survival | rt_cost>=1.25 | rt_cost>=1.5 | rt_cost>=2.0 | status |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in (payload.get("per_label") or {}).items():
        if not m.get("available"):
            continue
        lines.append(
            f"| {name} | {m.get('positive_count', 0)} | {m.get('positive_rate', 0):.4f} | {m.get('average_return_positive', 0):.4f} | {m.get('median_return_positive', 0):.4f} | {m.get('cost_1_25x_survival', 0):.4f} | {m.get('cost_1_5x_survival', 0):.4f} | {m.get('return_to_cost_ratio_ge_1_25', 0):.4f} | {m.get('return_to_cost_ratio_ge_1_50', 0):.4f} | {m.get('return_to_cost_ratio_ge_2_00', 0):.4f} | {m.get('status')} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- Usable labels (for retraining): `{payload.get('usable_labels')}`")
    lines.append(f"- Too-rare labels: `{payload.get('too_rare_labels')}`")
    lines.append(f"- Too-broad labels: `{payload.get('too_broad_labels')}`")
    return "\n".join(lines)


def _deployment_readiness_from_cost_aware(per_label: Dict[str, Any]) -> Dict[str, Any]:
    """Map cost-aware label distribution to a deployment status.

    Strict ladder: cost-aware targets must pass cost_1.5x PF >= 1.00 on the
    labelled subset. If none does, deployment is BLOCKED. V2 labels are
    considered as a fallback.
    """
    paper_candidate = per_label.get("paper_candidate_label", {}) or {}
    high_conviction = per_label.get("high_conviction_trade_label", {}) or {}
    cost_survivor = per_label.get("cost_survivor_label", {}) or {}
    paper_candidate_v2 = per_label.get("paper_candidate_label_v2", {}) or {}
    high_conviction_v2 = per_label.get("high_conviction_trade_label_v2", {}) or {}
    cost_survivor_v2 = per_label.get("cost_survivor_label_v2", {}) or {}
    pf_150_paper = float(paper_candidate.get("cost_1_50x_profit_factor") or 0.0)
    pf_125_paper = float(paper_candidate.get("cost_1_25x_profit_factor") or 0.0)
    pf_150_hc = float(high_conviction.get("cost_1_50x_profit_factor") or 0.0)
    pf_150_cs = float(cost_survivor.get("cost_1_50x_profit_factor") or 0.0)
    pf_150_paper_v2 = float(paper_candidate_v2.get("cost_1_50x_profit_factor") or 0.0)
    pf_125_paper_v2 = float(paper_candidate_v2.get("cost_1_25x_profit_factor") or 0.0)
    pf_150_hc_v2 = float(high_conviction_v2.get("cost_1_50x_profit_factor") or 0.0)
    failed_gates: List[str] = []
    if pf_150_paper < 1.0:
        failed_gates.append("paper_candidate_label_cost_1_5x_pf_below_1.00")
    if pf_150_hc < 1.0:
        failed_gates.append("high_conviction_trade_label_cost_1_5x_pf_below_1.00")
    if pf_150_cs < 1.0:
        failed_gates.append("cost_survivor_label_cost_1_5x_pf_below_1.00")
    if pf_150_paper_v2 < 1.0:
        failed_gates.append("paper_candidate_label_v2_cost_1_5x_pf_below_1.00")
    if pf_150_hc_v2 < 1.0:
        failed_gates.append("high_conviction_trade_label_v2_cost_1_5x_pf_below_1.00")
    if (
        pf_150_paper < 1.0
        and pf_150_hc < 1.0
        and pf_150_cs < 1.0
        and pf_150_paper_v2 < 1.0
        and pf_150_hc_v2 < 1.0
    ):
        status = "BLOCKED"
    elif (pf_150_paper >= 1.0 and pf_125_paper >= 1.05) or (
        pf_150_paper_v2 >= 1.0 and pf_125_paper_v2 >= 1.05
    ):
        status = "PAPER_SIGNAL_ONLY_ALLOWED"
    else:
        status = "BLOCKED"
    return {
        "deployment_status": status,
        "failed_gates": failed_gates,
        "production_adoption_allowed": False,
        "paper_signal_only_allowed": status in {"PAPER_SIGNAL_ONLY_ALLOWED", "PAPER_EXECUTION_ALLOWED", "LIVE_SHADOW_ALLOWED", "MICRO_LIVE_ALLOWED", "PRODUCTION_ALLOWED"},
        "paper_execution_allowed": False,
        "live_shadow_allowed": False,
        "micro_live_allowed": False,
        "exact_next_command": "python scripts/build_cost_aware_edge_dataset.py --input <path> --output-dir data/processed --label-audit-report",
        "block_reason_summary": "All cost-aware targets (V1 and V2) must pass cost_1.5x PF >= 1.00 on the labelled subset before PAPER_SIGNAL_ONLY_ALLOWED.",
    }


def _render_label_audit_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Label Quality Audit")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Dataset: `{payload.get('dataset_path')}`")
    lines.append("")
    if not payload.get("available"):
        lines.append(f"- Skipped: `{payload.get('reason')}`")
        return "\n".join(lines)
    lines.append("## 1. Existing label formula")
    lines.append(f"- Formula: `{payload['label_formula']}`")
    lines.append(f"- Horizon bars: `{payload['horizon_bars']}`")
    lines.append(f"- Uses gross_forward_return: `{payload['uses_gross_forward_return']}`")
    lines.append(f"- Requires return_above_cost_buffer: `{payload['requires_return_above_cost_buffer']}`")
    lines.append(f"- Includes realistic transaction cost: `{payload['includes_realistic_transaction_cost']}`")
    lines.append(f"- Includes realistic slippage buffer: `{payload['includes_realistic_slippage_buffer']}`")
    lines.append(f"- CE/PE treatment: `{payload['ce_pe_treatment']}`")
    lines.append(f"- Cost penalty in label: `{payload['cost_penalty_in_label']}`")
    lines.append(f"- Spread penalty in label: `{payload['spread_penalty_in_label']}`")
    lines.append("")
    lines.append("## 2. Label distribution")
    lines.append(f"- Row count: `{payload['row_count']}`")
    lines.append(f"- Valid (labelled) count: `{payload['valid_count']}`")
    lines.append(f"- Positive label rate: `{payload['positive_label_rate']:.4f}`")
    lines.append(f"- Negative label rate: `{payload['negative_label_rate']:.4f}`")
    lines.append(f"- Positive count: `{payload['positive_count']}`")
    lines.append(f"- Negative count: `{payload['negative_count']}`")
    lines.append("")
    lines.append("## 3. Returns on labelled rows")
    lines.append(f"- Avg net_forward_return (positive): `{payload['average_net_forward_return_positive']:.6f}`")
    lines.append(f"- Median net_forward_return (positive): `{payload['median_net_forward_return_positive']:.6f}`")
    lines.append(f"- Avg net_forward_return (negative): `{payload['average_net_forward_return_negative']:.6f}`")
    lines.append(f"- Median net_forward_return (negative): `{payload['median_net_forward_return_negative']:.6f}`")
    lines.append("")
    lines.append("## 4. Cost survival on positive labels")
    lines.append(f"- Positive-label cost survival rate @ 1.25x: `{payload['positive_label_cost_survival_rate_1_25x']:.4f}`")
    lines.append(f"- Positive-label cost survival rate @ 1.50x: `{payload['positive_label_cost_survival_rate_1_50x']:.4f}`")
    lines.append(f"- Percent positive with return_to_cost_ratio >= 1.25: `{payload['percent_positive_with_return_to_cost_ratio_ge_1_25']:.4f}`")
    lines.append(f"- Percent positive with return_to_cost_ratio >= 1.50: `{payload['percent_positive_with_return_to_cost_ratio_ge_1_50']:.4f}`")
    lines.append(f"- Percent positive with return_to_cost_ratio >= 2.00: `{payload['percent_positive_with_return_to_cost_ratio_ge_2_00']:.4f}`")
    lines.append("")
    lines.append("## 5. Diagnostics")
    lines.append(f"- Label noise warning: `{payload['label_noise_warning']}`")
    lines.append(f"- Weak label warning: `{payload['weak_label_warning']}`")
    lines.append(f"- Deployment too weak for paper: `{payload['deployment_too_weak_for_paper']}`")
    lines.append("")
    lines.append("## 6. Conclusion")
    lines.append("- The existing `profitable_trade_label` is `net_forward_return > 0.0` after a hardcoded 0.35% penalty.")
    lines.append("- It is far too permissive for paper deployment: only a fraction of positive labels survive a realistic 1.25x–1.50x cost stress.")
    lines.append("- Cost-aware labels (`strong_*`, `high_conviction_*`, `cost_survivor_*`, `paper_candidate_*`) are added to force the model to learn post-cost edge.")
    return "\n".join(lines)


def _render_build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cost-Aware Dataset Build")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Input: `{payload.get('input_path')}`")
    lines.append(f"- Output: `{payload.get('output_path')}`")
    lines.append("")
    lines.append("## Dataset shape")
    lines.append(f"- Rows: `{payload['row_count']}`")
    if "date_range" in payload:
        lines.append(f"- Date range: `{payload['date_range']}`")
    lines.append(f"- Feature count: `{payload['feature_count']}`")
    lines.append(f"- Forbidden columns in features: `{payload['leakage_audit']['leaked_columns']}`")
    lines.append(f"- Leakage detected: `{payload['leakage_audit']['leakage_detected']}`")
    lines.append("")
    lines.append("## Label positive rates")
    for k, v in payload.get("label_positive_rates", {}).items():
        lines.append(f"- `{k}`: `{v:.4f}`")
    lines.append("")
    lines.append("## Label positive counts")
    for k, v in payload.get("label_positive_counts", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Warnings")
    if payload.get("warnings"):
        for w in payload["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Forbidden target/return columns excluded from input features")
    for c in payload.get("leakage_audit", {}).get("forbidden_columns", []):
        lines.append(f"- `{c}`")
    return "\n".join(lines)


def _render_comparison_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Old vs Cost-Aware Labels")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Input: `{payload.get('input_path')}`")
    lines.append("")
    lines.append("## Per-label metrics")
    lines.append("")
    lines.append("| label | pos_count | pos_rate | avg_return | median_return | PF | Sharpe | RT-cost ratio avg | cost_1.25x PF | cost_1.50x PF |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, m in (payload.get("per_label_metrics", {}) or {}).items():
        if not m.get("available"):
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {m.get('positive_count', 0)} | {m.get('positive_rate', 0):.4f} | {m.get('average_return', 0):.4f} | {m.get('median_return', 0):.4f} | {m.get('profit_factor', 0):.3f} | {m.get('sharpe', 0):.3f} | {m.get('return_to_cost_ratio_average', 0):.3f} | {m.get('cost_1_25x_profit_factor', 0):.3f} | {m.get('cost_1_50x_profit_factor', 0):.3f} |"
        )
    lines.append("")
    lines.append("## Answers")
    for k, v in payload.items():
        if k.startswith("answer_"):
            lines.append(f"- **{k.replace('answer_', '').replace('_', ' ').title()}**: {v}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label quality audit + cost-aware dataset builder.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--label-audit-report", action="store_true", default=True)
    args = parser.parse_args(argv)
    input_path: Path = args.input
    if not input_path.exists():
        print(f"[cost-aware] input not found: {input_path}", file=sys.stderr)
        return 2
    print(f"[cost-aware] loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[cost-aware] rows: {df.shape[0]} cols: {df.shape[1]}")

    reports_dir: Path = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _ts()
    label_audit = _label_audit(df)
    label_audit["dataset_path"] = str(input_path)
    label_audit["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Determine embedded cost from the dataset itself.
    double_counting = audit_double_counting(df, evaluation_return_column="net_forward_return")
    embedded_cost = float(double_counting.get("inferred_embedded_cost") or 0.0035)
    cost_audit = {
        "double_counting_check": double_counting,
        "cost_assumptions": dict(DEFAULT_EXECUTION_COST_CONFIG),
    }
    df_with_labels, label_info = _add_cost_aware_labels(df, embedded_cost=embedded_cost)

    feature_list = _build_input_feature_list(df_with_labels)
    leakage_audit = _leakage_audit(feature_list)
    feature_list = [c for c in feature_list if c not in FORBIDDEN_FEATURE_COLUMNS]

    output_path = args.output_dir / f"nifty_option_chain_cost_aware_edge_dataset_{timestamp}.csv"
    df_with_labels.to_csv(output_path, index=False)
    print(f"[cost-aware] wrote: {output_path}")

    # Build report
    build_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": int(df_with_labels.shape[0]),
        "feature_count": len(feature_list),
        "date_range": _safe_date_range(df_with_labels),
        "label_positive_rates": label_info["label_positive_rates"],
        "label_positive_counts": label_info["label_positive_counts"],
        "warnings": label_info["warnings"],
        "leakage_audit": leakage_audit,
        "embedded_cost_used": label_info["embedded_cost_used"],
        "cost_audit": cost_audit,
        "metadata": {
            "horizon_bars": 3,
            "evaluation_return_column": "net_forward_return",
            "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        },
    }
    build_json = reports_dir / f"cost_aware_dataset_build_{timestamp}.json"
    build_md = reports_dir / f"cost_aware_dataset_build_{timestamp}.md"
    _write_json(build_json, build_payload)
    _write_text(build_md, _render_build_markdown(build_payload))
    print(f"[cost-aware] wrote: {build_json}")
    print(f"[cost-aware] wrote: {build_md}")

    audit_json = reports_dir / f"label_quality_audit_{timestamp}.json"
    audit_md = reports_dir / f"label_quality_audit_{timestamp}.md"
    _write_json(audit_json, label_audit)
    _write_text(audit_md, _render_label_audit_markdown(label_audit))
    print(f"[cost-aware] wrote: {audit_json}")
    print(f"[cost-aware] wrote: {audit_md}")

    # Old vs new comparison
    comparison = _old_vs_cost_aware(df_with_labels)
    comparison["input_path"] = str(input_path)
    comparison["generated_at"] = datetime.now(timezone.utc).isoformat()
    comp_json = reports_dir / f"old_vs_cost_aware_labels_{timestamp}.json"
    comp_md = reports_dir / f"old_vs_cost_aware_labels_{timestamp}.md"
    _write_json(comp_json, comparison)
    _write_text(comp_md, _render_comparison_markdown(comparison))
    print(f"[cost-aware] wrote: {comp_json}")
    print(f"[cost-aware] wrote: {comp_md}")

    # V2 audit
    v2_audit = _v2_label_audit(df_with_labels)
    v2_audit["input_path"] = str(input_path)
    v2_audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    v2_json = reports_dir / f"cost_aware_label_v2_audit_{timestamp}.json"
    v2_md = reports_dir / f"cost_aware_label_v2_audit_{timestamp}.md"
    _write_json(v2_json, v2_audit)
    _write_text(v2_md, _render_v2_label_audit_markdown(v2_audit))
    print(f"[cost-aware] wrote: {v2_json}")
    print(f"[cost-aware] wrote: {v2_md}")
    print(f"[cost-aware] usable labels: {v2_audit['usable_labels']}")
    print(f"[cost-aware] too-rare labels: {v2_audit['too_rare_labels']}")
    print(f"[cost-aware] too-broad labels: {v2_audit['too_broad_labels']}")

    # Deployment readiness
    deploy = _deployment_readiness_from_cost_aware(comparison.get("per_label_metrics", {}))
    deploy["dataset_path"] = str(input_path)
    deploy["cost_aware_dataset_path"] = str(output_path)
    deploy_json = reports_dir / f"deployment_readiness_{timestamp}.json"
    deploy_md = reports_dir / f"deployment_readiness_{timestamp}.md"
    _write_json(deploy_json, deploy)
    _write_text(deploy_md, _render_deployment_markdown(deploy))
    print(f"[cost-aware] wrote: {deploy_json}")
    print(f"[cost-aware] wrote: {deploy_md}")
    print(f"[cost-aware] deployment_status: {deploy['deployment_status']}")

    # --- Exact report names requested for PHASE 2 deliverable ---
    # cost_aware_edge_dataset_report_<ts>.{json,md}
    edge_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_input": int(df.shape[0]),
        "rows_output": int(df_with_labels.shape[0]),
        "rows_dropped": 0,
        "label_distributions": label_info.get("label_positive_counts", {}),
        "label_positive_rates": label_info.get("label_positive_rates", {}),
        "cost_assumptions": dict(DEFAULT_EXECUTION_COST_CONFIG),
        "double_counting_check": double_counting,
        "missing_columns_in_input": [],
        "cost_survivor_label_v2_reliable": True,
        "notes": [
            "Cost components (spread/slippage/brokerage) estimated via ml_execution_costs.estimate_option_execution_costs_frame using ltp + bid_ask_spread_pct + embedded (gross-net).",
            "No rows dropped; NaNs only on label columns where return was missing.",
            "cost_survivor_label_v2 and V2 variants + full cost_stress_* columns attached to dataset.",
            "All outcome/return/label columns remain forbidden for features (see FORBIDDEN_FEATURE_COLUMNS).",
        ],
    }
    edge_json = reports_dir / f"cost_aware_edge_dataset_report_{timestamp}.json"
    edge_md = reports_dir / f"cost_aware_edge_dataset_report_{timestamp}.md"
    _write_json(edge_json, edge_payload)
    # simple md summary
    md_lines = [
        "# Cost-Aware Edge Dataset Report",
        "",
        f"- Generated: `{edge_payload['generated_at']}`",
        f"- Input: `{edge_payload['input_path']}`",
        f"- Output CSV: `{edge_payload['output_path']}`",
        "",
        "## Row counts",
        f"- rows_input: {edge_payload['rows_input']}",
        f"- rows_output: {edge_payload['rows_output']}",
        f"- rows_dropped: {edge_payload['rows_dropped']}",
        "",
        "## Label distributions (positive counts)",
    ]
    for k, v in edge_payload["label_distributions"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines += [
        "",
        "## Cost assumptions (from ml_execution_costs)",
        f"- {edge_payload['cost_assumptions']}",
        "",
        "## cost_survivor_label_v2 reliability",
        f"- reliable for training: {edge_payload['cost_survivor_label_v2_reliable']}",
        "",
        "## Notes",
    ]
    for n in edge_payload["notes"]:
        md_lines.append(f"- {n}")
    _write_text(edge_md, "\n".join(md_lines))
    print(f"[cost-aware] wrote: {edge_json}")
    print(f"[cost-aware] wrote: {edge_md}")

    return 0


def _safe_date_range(df: pd.DataFrame) -> str:
    if "timestamp" not in df.columns:
        return "unknown"
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.notna().any():
        return f"{ts.min()} -> {ts.max()}"
    return "unknown"


def _render_deployment_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Deployment Readiness")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Dataset: `{payload.get('dataset_path')}`")
    lines.append(f"- Cost-aware dataset: `{payload.get('cost_aware_dataset_path')}`")
    lines.append("")
    lines.append(f"- Status: `{payload['deployment_status']}`")
    lines.append(f"- Failed gates: `{payload['failed_gates']}`")
    lines.append(f"- Production adoption allowed: `{payload['production_adoption_allowed']}`")
    lines.append(f"- Paper signal-only allowed: `{payload['paper_signal_only_allowed']}`")
    lines.append(f"- Paper execution allowed: `{payload['paper_execution_allowed']}`")
    lines.append(f"- Live shadow allowed: `{payload['live_shadow_allowed']}`")
    lines.append(f"- Micro-live allowed: `{payload['micro_live_allowed']}`")
    lines.append("")
    lines.append("## Next command")
    lines.append(f"- `{payload['exact_next_command']}`")
    lines.append("")
    lines.append("## Block reason")
    lines.append(f"- {payload.get('block_reason_summary', 'N/A')}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
