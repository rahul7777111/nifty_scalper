from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


DEFAULT_EXECUTION_COST_CONFIG: Dict[str, Any] = {
    "brokerage_per_order": 0.02,
    "slippage_pct": 0.0025,
    "exchange_charges_pct": 0.00053,
    "stt_pct": 0.0005,
    "gst_pct": 0.18,
    "sebi_pct": 0.00001,
    "spread_cost_mode": "half_spread",
    "min_cost_pct": 0.0025,
    "max_cost_pct": 0.01,
    "roundtrip": True,
    "lot_size": 1.0,
    "premium_column": "ltp",
    "return_column": "net_forward_return",
    "gross_return_column": "gross_forward_return",
    "net_return_column": "net_forward_return",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def estimate_option_execution_cost(row: Any, config: Dict[str, Any] | None = None) -> Dict[str, float]:
    cfg = dict(DEFAULT_EXECUTION_COST_CONFIG)
    if config:
        cfg.update(config)
    premium = max(0.0, _float(getattr(row, cfg["premium_column"], None) if hasattr(row, cfg["premium_column"]) else row.get(cfg["premium_column"]), 0.0))
    gross_return = _float(row.get(cfg["gross_return_column"])) if isinstance(row, dict) else _float(getattr(row, cfg["gross_return_column"], 0.0))
    net_return = _float(row.get(cfg["net_return_column"])) if isinstance(row, dict) else _float(getattr(row, cfg["net_return_column"], 0.0))
    embedded_cost = max(0.0, gross_return - net_return)
    spread_pct = _float(row.get("bid_ask_spread_pct")) if isinstance(row, dict) else _float(getattr(row, "bid_ask_spread_pct", 0.0))
    spread_cost = premium * (spread_pct / 100.0) * (0.5 if str(cfg.get("spread_cost_mode")) == "half_spread" else 1.0)
    brokerage_cost = max(0.0, _float(cfg.get("brokerage_per_order"), 0.0)) * (2.0 if bool(cfg.get("roundtrip", True)) else 1.0)
    exchange_cost = premium * max(0.0, _float(cfg.get("exchange_charges_pct"), 0.0))
    stt_cost = premium * max(0.0, _float(cfg.get("stt_pct"), 0.0))
    sebi_cost = premium * max(0.0, _float(cfg.get("sebi_pct"), 0.0))
    slippage_cost = premium * max(0.0, _float(cfg.get("slippage_pct"), 0.0))
    taxable = brokerage_cost + exchange_cost
    gst_cost = taxable * max(0.0, _float(cfg.get("gst_pct"), 0.0))
    configured_cost = brokerage_cost + exchange_cost + stt_cost + sebi_cost + slippage_cost + spread_cost + gst_cost
    cost_pct = float(np.clip((configured_cost / premium) if premium > 0.0 else 0.0, _float(cfg.get("min_cost_pct"), 0.0), _float(cfg.get("max_cost_pct"), 1.0))) if premium > 0.0 else 0.0
    premium_capped_cost = premium * cost_pct
    total_cost = max(embedded_cost, premium_capped_cost, configured_cost)
    return {
        "premium": premium,
        "embedded_cost": embedded_cost,
        "brokerage_cost": brokerage_cost,
        "exchange_cost": exchange_cost,
        "stt_cost": stt_cost,
        "gst_cost": gst_cost,
        "sebi_cost": sebi_cost,
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
        "configured_cost": configured_cost,
        "premium_capped_cost": premium_capped_cost,
        "total_estimated_roundtrip_cost": total_cost,
        "cost_pct_of_premium": cost_pct,
        "cost_return_units": total_cost,
    }


def estimate_option_execution_costs_frame(frame: pd.DataFrame, config: Dict[str, Any] | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "premium",
            "embedded_cost",
            "brokerage_cost",
            "exchange_cost",
            "stt_cost",
            "gst_cost",
            "sebi_cost",
            "spread_cost",
            "slippage_cost",
            "configured_cost",
            "premium_capped_cost",
            "total_estimated_roundtrip_cost",
            "cost_pct_of_premium",
            "cost_return_units",
        ])
    records = [estimate_option_execution_cost(row, config=config) for row in frame.to_dict(orient="records")]
    return pd.DataFrame.from_records(records, index=frame.index)


# ---------------------------------------------------------------------------
# Premium-adjusted cost-avoidance helpers (audit-only, additive to existing
# cost-stress logic in retrain_all_edge_models.py).
# ---------------------------------------------------------------------------

PREMIUM_BANDS: tuple[float, ...] = (20.0, 30.0, 50.0, 75.0, 100.0)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index)


def audit_double_counting(frame: pd.DataFrame, *, evaluation_return_column: str = "net_forward_return") -> Dict[str, Any]:
    """Detect possible double-counting between gross and net returns.

    Returns a dict with gross/net averages, embedded cost, and a recommendation.
    """
    gross = _series(frame, "gross_forward_return")
    net = _series(frame, evaluation_return_column) if evaluation_return_column in frame.columns else _series(frame, "net_forward_return")
    diff = (gross - net).replace([np.inf, -np.inf], np.nan).dropna()
    positive_diff = diff[diff > 0.0]
    embedded_cost = float(positive_diff.median()) if not positive_diff.empty else 0.0
    already_cost_adjusted = bool(
        evaluation_return_column and (
            "net_" in str(evaluation_return_column).lower() or "cost_adjusted" in str(evaluation_return_column).lower()
        )
    )
    if not gross.notna().any() or not net.notna().any():
        recommendation = "BLOCKED_MISSING_GROSS_OR_NET"
    elif already_cost_adjusted and embedded_cost > 0.0:
        recommendation = "VALID_NET_RETURN_INCREMENTAL_STRESS_MODEL"
    elif already_cost_adjusted and embedded_cost <= 0.0:
        recommendation = "BLOCKED_UNCERTAIN_COST_PROVENANCE"
    else:
        recommendation = "VALID_GROSS_RETURN_COST_MODEL"
    return {
        "evaluation_return_column": evaluation_return_column,
        "gross_forward_return_available": bool(gross.notna().any()),
        "net_forward_return_available": bool(net.notna().any()),
        "gross_forward_return_average": float(gross.mean()) if gross.notna().any() else None,
        "net_forward_return_average": float(net.mean()) if net.notna().any() else None,
        "gross_minus_net_average": float(diff.mean()) if not diff.empty else None,
        "gross_minus_net_median": float(diff.median()) if not diff.empty else None,
        "inferred_embedded_cost": embedded_cost,
        "net_forward_return_already_cost_adjusted": already_cost_adjusted,
        "double_counting_prevented": bool(already_cost_adjusted and embedded_cost > 0.0),
        "recommendation": recommendation,
    }


def _cost_metrics_from_returns(returns: pd.Series) -> Dict[str, Any]:
    arr = np.asarray(returns.to_numpy(dtype=float), dtype=float) if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "trade_count": 0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "average_return_per_trade": 0.0,
            "median_return_per_trade": 0.0,
        }
    positives = arr[arr > 0.0]
    negatives = arr[arr < 0.0]
    gross_profit = float(positives.sum()) if positives.size else 0.0
    gross_loss = float(-negatives.sum()) if negatives.size else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0)
    average_return = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sharpe = float(average_return / std) if std > 0.0 else 0.0
    return {
        "trade_count": int(arr.size),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 0.0,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else 0.0,
        "average_return_per_trade": float(average_return),
        "median_return_per_trade": float(np.median(arr)),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
    }


def _apply_cost_to_pf(returns: pd.Series, cost_return_units: pd.Series) -> Dict[str, Any]:
    """Subtract per-trade cost from per-trade return and recompute trade metrics."""
    if returns.empty:
        return {
            "trade_count": 0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "average_return_per_trade": 0.0,
            "median_return_per_trade": 0.0,
            "cost_1_25x_profit_factor": 0.0,
            "cost_1_50x_profit_factor": 0.0,
            "return_to_cost_ratio": 0.0,
        }
    rets = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    costs = pd.to_numeric(cost_return_units, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    net_after_cost = rets - costs
    avg_cost = float(costs.mean())
    return_to_cost_ratio = float(net_after_cost.mean() / avg_cost) if avg_cost > 0.0 else 0.0
    base_metrics = _cost_metrics_from_returns(pd.Series(net_after_cost))
    stressed_125 = _cost_metrics_from_returns(pd.Series(net_after_cost - 0.25 * costs))
    stressed_150 = _cost_metrics_from_returns(pd.Series(net_after_cost - 0.50 * costs))
    base_metrics["cost_1_25x_profit_factor"] = float(stressed_125.get("profit_factor") or 0.0)
    base_metrics["cost_1_50x_profit_factor"] = float(stressed_150.get("profit_factor") or 0.0)
    base_metrics["return_to_cost_ratio"] = return_to_cost_ratio
    return base_metrics


def evaluate_premium_bands(
    frame: pd.DataFrame,
    *,
    bands: Sequence[float] = PREMIUM_BANDS,
    premium_column: str = "ltp",
    return_column: str = "net_forward_return",
    cost_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Per premium-band evaluation with realistic cost estimates.

    For each premium floor (ltp >= 20, 30, ...) compute:
    - trade_count, PF, Sharpe, average_return_per_trade
    - estimated cost pct and cost-to-return ratio
    - cost_1.25x / cost_1.5x PF
    """
    if frame.empty or premium_column not in frame.columns or return_column not in frame.columns:
        return {"available": False, "bands": []}
    work = frame[[premium_column, return_column]].copy()
    work[premium_column] = pd.to_numeric(work[premium_column], errors="coerce")
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    work = work.dropna(subset=[premium_column, return_column])
    if work.empty:
        return {"available": False, "bands": []}
    cost_frame = estimate_option_execution_costs_frame(work, config=cost_config)
    work = pd.concat([work, cost_frame], axis=1)
    sorted_bands = sorted({float(b) for b in bands})
    band_results: List[Dict[str, Any]] = []
    for band in sorted_bands:
        mask = work[premium_column] >= band
        subset = work.loc[mask]
        if subset.empty:
            band_results.append({
                "min_premium": band,
                "trade_count": 0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "average_return_per_trade": 0.0,
                "median_return_per_trade": 0.0,
                "estimated_cost_pct": 0.0,
                "return_to_cost_ratio": 0.0,
                "cost_1_25x_profit_factor": 0.0,
                "cost_1_50x_profit_factor": 0.0,
                "worst_fold_pf": 0.0,
                "daily_stability_available": False,
            })
            continue
        cost_metrics = _apply_cost_to_pf(subset[return_column], subset["cost_return_units"])
        avg_cost_pct = float(subset["cost_pct_of_premium"].mean())
        band_results.append({
            "min_premium": band,
            "trade_count": int(cost_metrics["trade_count"]),
            "profit_factor": float(cost_metrics["profit_factor"]),
            "sharpe": float(cost_metrics["sharpe"]),
            "average_return_per_trade": float(cost_metrics["average_return_per_trade"]),
            "median_return_per_trade": float(cost_metrics["median_return_per_trade"]),
            "estimated_cost_pct": avg_cost_pct,
            "return_to_cost_ratio": float(cost_metrics["return_to_cost_ratio"]),
            "cost_1_25x_profit_factor": float(cost_metrics["cost_1_25x_profit_factor"]),
            "cost_1_50x_profit_factor": float(cost_metrics["cost_1_50x_profit_factor"]),
            "worst_fold_pf": 0.0,
            "daily_stability_available": False,
        })
    return {
        "available": True,
        "premium_column": premium_column,
        "return_column": return_column,
        "bands": band_results,
    }


def evaluate_spread_filters(
    frame: pd.DataFrame,
    *,
    spread_pct_column: str = "bid_ask_spread_pct",
    return_column: str = "net_forward_return",
    cost_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply spread / slippage-aware filters when bid-ask columns exist.

    Skipped (and reported) when the column is missing.
    """
    if frame.empty or return_column not in frame.columns:
        return {"available": False, "filters": []}
    if spread_pct_column not in frame.columns:
        return {
            "available": False,
            "skipped": True,
            "skipped_reason": f"{spread_pct_column} column missing",
            "filters": [],
        }
    work = frame[[spread_pct_column, return_column]].copy()
    work[spread_pct_column] = pd.to_numeric(work[spread_pct_column], errors="coerce")
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    work = work.dropna(subset=[spread_pct_column, return_column])
    if work.empty:
        return {"available": False, "skipped": True, "filters": []}
    cost_frame = estimate_option_execution_costs_frame(work, config=cost_config)
    work = pd.concat([work, cost_frame], axis=1)
    abs_return = work[return_column].abs()
    spread = work[spread_pct_column]
    spread_to_expected_return = (spread / 100.0) / abs_return.replace(0.0, np.nan)
    filters: List[Dict[str, Any]] = []
    for label, condition in (
        ("bid_ask_spread_pct_le_0_5", spread <= 0.5),
        ("bid_ask_spread_pct_le_1_0", spread <= 1.0),
        ("bid_ask_spread_pct_le_1_5", spread <= 1.5),
        ("spread_to_expected_return_le_0_25", spread_to_expected_return.fillna(np.inf) <= 0.25),
        ("spread_to_expected_return_le_0_50", spread_to_expected_return.fillna(np.inf) <= 0.50),
    ):
        subset = work.loc[condition.fillna(False)]
        if subset.empty:
            filters.append({
                "filter_name": label,
                "trade_count": 0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "average_return_per_trade": 0.0,
                "return_to_cost_ratio": 0.0,
                "cost_1_25x_profit_factor": 0.0,
                "cost_1_50x_profit_factor": 0.0,
            })
            continue
        cost_metrics = _apply_cost_to_pf(subset[return_column], subset["cost_return_units"])
        filters.append({
            "filter_name": label,
            "trade_count": int(cost_metrics["trade_count"]),
            "profit_factor": float(cost_metrics["profit_factor"]),
            "sharpe": float(cost_metrics["sharpe"]),
            "average_return_per_trade": float(cost_metrics["average_return_per_trade"]),
            "return_to_cost_ratio": float(cost_metrics["return_to_cost_ratio"]),
            "cost_1_25x_profit_factor": float(cost_metrics["cost_1_25x_profit_factor"]),
            "cost_1_50x_profit_factor": float(cost_metrics["cost_1_50x_profit_factor"]),
        })
    return {"available": True, "spread_column": spread_pct_column, "filters": filters}


def evaluate_expected_move_filters(
    frame: pd.DataFrame,
    *,
    probability_column: str = "model_probability",
    return_column: str = "net_forward_return",
    cost_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Per-trade expected-move vs cost analysis.

    expected_move_proxy = probability_margin (model_prob - 0.5), or abs(return) fallback.
    """
    if frame.empty or return_column not in frame.columns:
        return {"available": False, "filters": []}
    work = frame[[return_column]].copy()
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    if probability_column in frame.columns:
        work[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
    else:
        work[probability_column] = np.nan
    work = work.dropna(subset=[return_column])
    if work.empty:
        return {"available": False, "filters": []}
    cost_frame = estimate_option_execution_costs_frame(work, config=cost_config)
    work = pd.concat([work, cost_frame], axis=1)
    if work[probability_column].notna().any():
        work["expected_move_proxy"] = (work[probability_column].fillna(0.5) - 0.5).abs()
    else:
        work["expected_move_proxy"] = work[return_column].abs()
    work["abs_expected_return"] = work[return_column].abs()
    work["expected_return_after_estimated_cost"] = work[return_column] - work["cost_return_units"]
    avg_cost = work["cost_return_units"].replace(0.0, np.nan)
    work["expected_return_to_cost_ratio"] = (work["expected_return_after_estimated_cost"].abs() / avg_cost).replace([np.inf, -np.inf], np.nan)
    summary = {
        "available": True,
        "probability_column_used": probability_column if work[probability_column].notna().any() else None,
        "median_return_after_estimated_cost": float(work["expected_return_after_estimated_cost"].median()) if not work["expected_return_after_estimated_cost"].empty else 0.0,
        "average_return_after_estimated_cost": float(work["expected_return_after_estimated_cost"].mean()) if not work["expected_return_after_estimated_cost"].empty else 0.0,
        "median_expected_return_to_cost_ratio": float(work["expected_return_to_cost_ratio"].median(skipna=True)) if work["expected_return_to_cost_ratio"].notna().any() else 0.0,
        "filters": [],
    }
    for label, condition in (
        ("expected_return_to_cost_ratio_ge_1_25", work["expected_return_to_cost_ratio"].fillna(0.0) >= 1.25),
        ("expected_return_to_cost_ratio_ge_1_50", work["expected_return_to_cost_ratio"].fillna(0.0) >= 1.50),
        ("expected_return_to_cost_ratio_ge_2_00", work["expected_return_to_cost_ratio"].fillna(0.0) >= 2.00),
        ("median_return_after_estimated_cost_gt_0", work["expected_return_after_estimated_cost"] > 0.0),
        ("average_return_after_estimated_cost_gt_0", work["expected_return_after_estimated_cost"] > 0.0),
    ):
        subset = work.loc[condition.fillna(False)]
        if subset.empty:
            summary["filters"].append({
                "filter_name": label,
                "trade_count": 0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "expected_return_after_estimated_cost": 0.0,
                "return_to_cost_ratio": 0.0,
            })
            continue
        cost_metrics = _apply_cost_to_pf(subset[return_column], subset["cost_return_units"])
        summary["filters"].append({
            "filter_name": label,
            "trade_count": int(cost_metrics["trade_count"]),
            "profit_factor": float(cost_metrics["profit_factor"]),
            "sharpe": float(cost_metrics["sharpe"]),
            "expected_return_after_estimated_cost": float(subset["expected_return_after_estimated_cost"].mean()),
            "return_to_cost_ratio": float(cost_metrics["return_to_cost_ratio"]),
        })
    return summary


def evaluate_cost_sensitive_bands(
    frame: pd.DataFrame,
    *,
    return_column: str = "net_forward_return",
    cost_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Per trade-count band reporting (100-250, 250-500, 500-1000, 1000-2000, 2000-3000).

    The frame must already be filtered to selected trades (one row per trade).
    """
    bands: tuple[tuple[int, int], ...] = (
        (100, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 3000),
    )
    if frame.empty or return_column not in frame.columns:
        return {"available": False, "bands": []}
    work = frame[[return_column]].copy()
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    work = work.dropna(subset=[return_column])
    cost_frame = estimate_option_execution_costs_frame(work, config=cost_config)
    work = pd.concat([work, cost_frame], axis=1)
    trade_count = int(len(work))
    band_results: List[Dict[str, Any]] = []
    for low, high in bands:
        if trade_count < low or trade_count > high:
            band_results.append({
                "band": f"{low}-{high}",
                "trade_count": trade_count,
                "in_band": False,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "return_to_cost_ratio": 0.0,
                "cost_1_25x_profit_factor": 0.0,
                "cost_1_50x_profit_factor": 0.0,
                "worst_fold_pf": 0.0,
                "cost_aware_stability_score": 0.0,
            })
            continue
        cost_metrics = _apply_cost_to_pf(work[return_column], work["cost_return_units"])
        cost_aware = (
            3.0 * min(cost_metrics["cost_1_25x_profit_factor"], 2.0)
            + 3.0 * min(cost_metrics["cost_1_50x_profit_factor"], 2.0)
            + 2.0 * min(cost_metrics["return_to_cost_ratio"], 3.0)
        )
        band_results.append({
            "band": f"{low}-{high}",
            "trade_count": int(cost_metrics["trade_count"]),
            "in_band": True,
            "profit_factor": float(cost_metrics["profit_factor"]),
            "sharpe": float(cost_metrics["sharpe"]),
            "return_to_cost_ratio": float(cost_metrics["return_to_cost_ratio"]),
            "cost_1_25x_profit_factor": float(cost_metrics["cost_1_25x_profit_factor"]),
            "cost_1_50x_profit_factor": float(cost_metrics["cost_1_50x_profit_factor"]),
            "worst_fold_pf": 0.0,
            "cost_aware_stability_score": float(cost_aware),
        })
    return {
        "available": True,
        "trade_count": trade_count,
        "bands": band_results,
    }


def cost_survival_score(candidate_metrics: Dict[str, Any]) -> float:
    """Cost-aware ranking score (per spec)."""
    cost_125 = float(candidate_metrics.get("cost_1_25x_profit_factor") or candidate_metrics.get("cost_stress_1_25x_pf") or 0.0)
    cost_150 = float(candidate_metrics.get("cost_1_50x_profit_factor") or candidate_metrics.get("cost_stress_1_50x_pf") or 0.0)
    return_to_cost_ratio = float(candidate_metrics.get("return_to_cost_ratio") or 0.0)
    worst_fold_pf = float(candidate_metrics.get("worst_fold_pf") or 0.0)
    median_fold_pf = float(candidate_metrics.get("median_fold_pf") or 0.0)
    profitable_days = float(candidate_metrics.get("profitable_days") or 0)
    top_2_days_profit_share = float(candidate_metrics.get("top_2_days_profit_share") or 0.0)
    best_fold_profit_share = float(candidate_metrics.get("best_fold_profit_share") or 0.0)
    max_drawdown = float(candidate_metrics.get("max_drawdown") or 0.0)
    drawdown_penalty = min(abs(max_drawdown) * 0.01, 5.0)
    return (
        3.0 * min(cost_125, 2.0)
        + 3.0 * min(cost_150, 2.0)
        + 2.0 * min(return_to_cost_ratio, 3.0)
        + 1.5 * min(worst_fold_pf, 2.0)
        + 1.0 * min(median_fold_pf, 2.0)
        + 1.0 * min(profitable_days / 50.0, 2.0)
        - 2.0 * top_2_days_profit_share
        - 2.0 * best_fold_profit_share
        - drawdown_penalty
    )
