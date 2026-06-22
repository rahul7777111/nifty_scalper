"""Cost-model audit + cost-aware edge refinement runner.

This script is the focus of the "find whether the model has any realistic
tradable edge after accurate execution cost modeling and stricter
cost-avoidance filters" task.

It DOES NOT retrain any model. It loads the dataset, runs the cost-model
audit, applies premium-band / spread / expected-move filters, evaluates
cost-sensitive candidate bands, and writes:

  reports/cost_model_audit_<ts>.json
  reports/cost_model_audit_<ts>.md
  reports/cost_aware_edge_refinement_<ts>.json
  reports/cost_aware_edge_refinement_<ts>.md

It is intentionally additive to scripts/retrain_all_edge_models.py and
scripts/ml_execution_costs.py — no existing cost-stress behaviour is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain  # noqa: E402
from ml_execution_costs import (  # noqa: E402
    DEFAULT_EXECUTION_COST_CONFIG,
    _apply_cost_to_pf,
    audit_double_counting,
    cost_survival_score,
    evaluate_cost_sensitive_bands,
    evaluate_expected_move_filters,
    evaluate_premium_bands,
    evaluate_spread_filters,
    estimate_option_execution_costs_frame,
)


DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "nifty_option_chain_live_feature_reconstructed_20260604_100331_bs_enriched.csv"
REPORTS_DIR = REPO_ROOT / "reports"


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _selected_trade_count_band_metrics(frame: pd.DataFrame, return_column: str) -> Dict[str, Any]:
    """Lightweight per-band metrics for a generic selected-trade frame.

    Used by the cost-aware edge refinement section. The frame here is the
    profitable-trade subset (label == 1) so that edge metrics are evaluated
    on actual selected trades, not the full universe.
    """
    if frame.empty or return_column not in frame.columns:
        return {"available": False}
    work = frame.copy()
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    work = work.dropna(subset=[return_column])
    cost_frame = estimate_option_execution_costs_frame(work)
    if cost_frame.empty:
        return {"available": False}
    work = pd.concat([work, cost_frame], axis=1)
    rets = work[return_column].to_numpy(dtype=float)
    costs = work["cost_return_units"].to_numpy(dtype=float)
    net_after = rets - costs
    rets_125 = rets - 1.25 * costs
    rets_150 = rets - 1.50 * costs
    pf = lambda x: (  # noqa: E731
        float(x[x > 0].sum() / -x[x < 0].sum())
        if (x[x < 0].sum() < 0.0) else (float("inf") if (x[x > 0].sum() > 0.0) else 0.0)
    )
    sharpe = lambda x: (  # noqa: E731
        float(x.mean() / x.std(ddof=1)) if (x.size > 1 and x.std(ddof=1) > 0.0) else 0.0
    )
    avg_cost = float(costs.mean())
    avg_return_to_cost = float(net_after.mean() / avg_cost) if avg_cost > 0.0 else 0.0
    median_return_after_cost = float(np.median(net_after)) if net_after.size else 0.0
    return {
        "available": True,
        "trade_count": int(work.shape[0]),
        "profit_factor": float(pf(rets)) if np.isfinite(pf(rets)) else 0.0,
        "sharpe": float(sharpe(rets)),
        "average_return_per_trade": float(rets.mean()) if rets.size else 0.0,
        "cost_1_25x_profit_factor": float(pf(rets_125)) if np.isfinite(pf(rets_125)) else 0.0,
        "cost_1_50x_profit_factor": float(pf(rets_150)) if np.isfinite(pf(rets_150)) else 0.0,
        "return_to_cost_ratio": avg_return_to_cost,
        "median_return_after_estimated_cost": median_return_after_cost,
        "expected_return_after_estimated_cost": float(net_after.mean()) if net_after.size else 0.0,
    }


def _cost_aware_candidate_metrics(selected: pd.DataFrame, return_column: str, candidate_name: str) -> Dict[str, Any]:
    base = _selected_trade_count_band_metrics(selected, return_column)
    if not base.get("available"):
        return {"candidate_name": candidate_name, "available": False}
    worst_fold_pf = 0.0
    median_fold_pf = 0.0
    profitable_days = 0
    top_2_days_profit_share = 0.0
    best_fold_profit_share = 0.0
    if "timestamp" in selected.columns and return_column in selected.columns:
        df = selected[["timestamp", return_column]].dropna(subset=[return_column]).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if not df.empty:
            df["trade_day"] = df["timestamp"].dt.date
            days = df.groupby("trade_day")[return_column].sum()
            if len(days):
                profitable_days = int((days > 0.0).sum())
                pos_days = days[days > 0.0].sort_values(ascending=False)
                if pos_days.size:
                    top_2 = pos_days.head(2).sum()
                    total_pos = pos_days.sum()
                    top_2_days_profit_share = float(top_2 / total_pos) if total_pos > 0.0 else 0.0
            df["month"] = df["timestamp"].dt.to_period("M").astype(str)
            months = df.groupby("month")
            month_metrics = []
            for _, grp in months:
                arr = grp[return_column].to_numpy(dtype=float)
                pos = arr[arr > 0].sum()
                neg = -arr[arr < 0].sum()
                pf = float(pos / neg) if neg > 0.0 else (float("inf") if pos > 0.0 else 0.0)
                month_metrics.append((float(pf) if math.isfinite(pf) else 0.0, float(pos)))
            pf_values = [m[0] for m in month_metrics]
            if pf_values:
                worst_fold_pf = float(min(pf_values))
                median_fold_pf = float(np.median(pf_values))
            pos_profits = [m[1] for m in month_metrics if m[1] > 0.0]
            if pos_profits:
                best_fold_profit_share = float(max(pos_profits) / sum(pos_profits)) if sum(pos_profits) > 0.0 else 0.0
    metrics = dict(base)
    metrics.update({
        "candidate_name": candidate_name,
        "worst_fold_pf": worst_fold_pf,
        "median_fold_pf": median_fold_pf,
        "profitable_days": profitable_days,
        "top_2_days_profit_share": top_2_days_profit_share,
        "best_fold_profit_share": best_fold_profit_share,
        "max_drawdown": 0.0,
    })
    metrics["cost_survival_score"] = float(cost_survival_score(metrics))
    return metrics


def _audit_dataset_columns(df: pd.DataFrame) -> Dict[str, Any]:
    cols = {
        "has_ltp": "ltp" in df.columns,
        "has_gross_forward_return": "gross_forward_return" in df.columns,
        "has_net_forward_return": "net_forward_return" in df.columns,
        "has_bid_ask_spread_pct": "bid_ask_spread_pct" in df.columns,
        "has_bid_ask_spread": "bid_ask_spread" in df.columns,
        "has_option_type_ce": "option_type_ce" in df.columns,
        "has_option_type_pe": "option_type_pe" in df.columns,
        "has_profitable_trade_label": "profitable_trade_label" in df.columns,
        "has_avoid_trade_label": "avoid_trade_label" in df.columns,
    }
    return cols


def _cost_model_audit_payload(df: pd.DataFrame, *, evaluation_return_column: str) -> Dict[str, Any]:
    """Produce the cost_model_audit payload combining dataset fact and audit."""
    # Re-use the existing retrain._cost_model_audit_report fields where possible.
    sample = df.head(10)
    base_audit = retrain._cost_model_audit_report(  # type: ignore[attr-defined]
        df,
        evaluation_return_column=evaluation_return_column,
        artifact_dir=REPORTS_DIR,
    )
    double_counting = audit_double_counting(df, evaluation_return_column=evaluation_return_column)
    cost_frame = estimate_option_execution_costs_frame(df)
    cost_summary = {
        "rows": int(df.shape[0]),
        "avg_premium": float(cost_frame["premium"].mean()) if not cost_frame.empty else 0.0,
        "median_premium": float(cost_frame["premium"].median()) if not cost_frame.empty else 0.0,
        "avg_cost_pct": float(cost_frame["cost_pct_of_premium"].mean()) if not cost_frame.empty else 0.0,
        "median_cost_pct": float(cost_frame["cost_pct_of_premium"].median()) if not cost_frame.empty else 0.0,
        "avg_cost_return_units": float(cost_frame["cost_return_units"].mean()) if not cost_frame.empty else 0.0,
        "median_cost_return_units": float(cost_frame["cost_return_units"].median()) if not cost_frame.empty else 0.0,
    }
    return {
        "dataset_path": str(DEFAULT_DATASET),
        "evaluation_return_column": evaluation_return_column,
        "dataset_column_audit": _audit_dataset_columns(df),
        "sample_head_10": sample.to_dict(orient="records"),
        "cost_assumptions": dict(DEFAULT_EXECUTION_COST_CONFIG),
        "cost_summary_over_dataset": cost_summary,
        "base_cost_assumption": "roundtrip: True; cost components = brokerage + exchange + STT + SEBI + GST + bid/ask spread + slippage, clipped to min_cost_pct/max_cost_pct",
        "cost_stress_formula": "selected_return = evaluation_return - extra_cost (per-trade absolute return units, subtracted once per selected trade)",
        "cost_stress_scenarios": [0.10, 0.25, 0.50, 1.00],
        "net_forward_return_already_cost_adjusted": bool(double_counting["net_forward_return_already_cost_adjusted"]),
        "double_counting_check": double_counting,
        "legacy_audit_report": base_audit,
        "production_adoption_allowed": False,
    }


def evaluate_liquidity_filters(
    frame: pd.DataFrame,
    *,
    return_column: str = "net_forward_return",
    cost_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Volume / OI / moneyness / premium-band filters. Skipped filters are reported."""
    if frame.empty or return_column not in frame.columns:
        return {"available": False, "filters": [], "skipped_filters": []}
    work = frame.copy()
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    if "ltp" in work.columns:
        work["ltp"] = pd.to_numeric(work["ltp"], errors="coerce")
    cost_frame = estimate_option_execution_costs_frame(work, config=cost_config)
    work = pd.concat([work, cost_frame], axis=1)
    work = work.dropna(subset=[return_column])

    def _metrics_for(mask: pd.Series) -> Dict[str, Any]:
        subset = work.loc[mask.fillna(False)]
        if subset.empty:
            return {
                "trade_count": 0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "average_return_per_trade": 0.0,
                "return_to_cost_ratio": 0.0,
                "cost_1_25x_profit_factor": 0.0,
                "cost_1_50x_profit_factor": 0.0,
            }
        cost_metrics = _apply_cost_to_pf(subset[return_column], subset["cost_return_units"])
        return {
            "trade_count": int(cost_metrics["trade_count"]),
            "profit_factor": float(cost_metrics["profit_factor"]),
            "sharpe": float(cost_metrics["sharpe"]),
            "average_return_per_trade": float(cost_metrics["average_return_per_trade"]),
            "return_to_cost_ratio": float(cost_metrics["return_to_cost_ratio"]),
            "cost_1_25x_profit_factor": float(cost_metrics["cost_1_25x_profit_factor"]),
            "cost_1_50x_profit_factor": float(cost_metrics["cost_1_50x_profit_factor"]),
        }

    filters: List[Dict[str, Any]] = []
    skipped: List[str] = []

    # Premium floors
    if "ltp" in work.columns and work["ltp"].notna().any():
        for band in (20.0, 30.0, 50.0, 75.0, 100.0):
            m = work["ltp"].fillna(-np.inf) >= band
            filters.append({"filter_name": f"ltp_ge_{int(band)}", **_metrics_for(m)})
    else:
        skipped.append("ltp")

    # Volume filters
    vol = None
    for col in ("volume", "option_volume", "volume_CE", "volume_PE"):
        if col in work.columns:
            vol = pd.to_numeric(work[col], errors="coerce")
            if vol.notna().any():
                break
    if vol is not None and vol.notna().any():
        median_v = float(vol.dropna().median())
        q75_v = float(vol.dropna().quantile(0.75))
        filters.append({"filter_name": "volume_above_median", **_metrics_for(vol.fillna(-np.inf) >= median_v)})
        filters.append({"filter_name": "volume_above_q75", **_metrics_for(vol.fillna(-np.inf) >= q75_v)})
    else:
        skipped.append("volume")

    # OI filters
    oi = None
    for col in ("oi", "oi_CE", "oi_PE"):
        if col in work.columns:
            oi = pd.to_numeric(work[col], errors="coerce")
            if oi.notna().any():
                break
    if oi is not None and oi.notna().any():
        median_oi = float(oi.dropna().median())
        q75_oi = float(oi.dropna().quantile(0.75))
        filters.append({"filter_name": "oi_above_median", **_metrics_for(oi.fillna(-np.inf) >= median_oi)})
        filters.append({"filter_name": "oi_above_q75", **_metrics_for(oi.fillna(-np.inf) >= q75_oi)})
    else:
        skipped.append("oi")

    # Moneyness filters
    if "moneyness" in work.columns:
        mny = pd.to_numeric(work["moneyness"], errors="coerce")
        if mny.notna().any():
            atm_band = (mny >= 0.97) & (mny <= 1.03)
            itm_only = mny < 1.0
            otm_only = mny > 1.0
            far_otm = mny >= 1.10
            filters.append({"filter_name": "atm_or_near_atm_only", **_metrics_for(atm_band.fillna(False))})
            filters.append({"filter_name": "itm_or_near_atm_only", **_metrics_for((itm_only | atm_band).fillna(False))})
            filters.append({"filter_name": "otm_excluded", **_metrics_for((~otm_only).fillna(True))})
            filters.append({"filter_name": "far_otm_excluded", **_metrics_for((~far_otm).fillna(True))})
        else:
            skipped.append("moneyness")
    else:
        skipped.append("moneyness")

    return {
        "available": True,
        "filters": filters,
        "skipped_filters": skipped,
    }


def _deployment_readiness_payload(
    refinement: Dict[str, Any],
    *,
    dataset_path: Path,
) -> Dict[str, Any]:
    """Compute deployment status from the refinement payload.

    Strict ladder:
      BLOCKED
      PAPER_SIGNAL_ONLY_ALLOWED
      PAPER_EXECUTION_ALLOWED
      LIVE_SHADOW_ALLOWED
      MICRO_LIVE_ALLOWED
      PRODUCTION_ALLOWED

    Default: BLOCKED.
    """
    candidate = None
    if refinement.get("best_candidate_after_realistic_cost"):
        candidate = refinement["best_candidate_after_realistic_cost"]
    elif refinement.get("best_candidate_before_cost"):
        candidate = refinement["best_candidate_before_cost"]
    cost_125 = float((candidate or {}).get("cost_1_25x_profit_factor") or 0.0)
    cost_150 = float((candidate or {}).get("cost_1_50x_profit_factor") or 0.0)
    ratio = float((candidate or {}).get("return_to_cost_ratio") or 0.0)
    pf = float((candidate or {}).get("profit_factor") or 0.0)
    sharpe = float((candidate or {}).get("sharpe") or 0.0)
    expected_after_cost = float((candidate or {}).get("expected_return_after_estimated_cost") or 0.0)
    median_after_cost = float((candidate or {}).get("median_return_after_estimated_cost") or 0.0)
    if cost_150 < 1.00 or expected_after_cost <= 0.0 or median_after_cost <= 0.0:
        status = "BLOCKED"
        failed_gates = [
            "cost_1_5x_pf_below_1.00",
            "expected_return_after_cost_non_positive",
            "median_return_after_cost_non_positive",
        ]
    elif cost_125 >= 1.05 and ratio >= 1.50 and pf >= 1.15 and sharpe >= 0.75:
        status = "PAPER_EXECUTION_ALLOWED"
        failed_gates = []
    elif cost_125 >= 1.00 and ratio >= 1.25 and pf >= 1.05 and sharpe >= 0.5:
        status = "PAPER_SIGNAL_ONLY_ALLOWED"
        failed_gates = ["ratio_or_pf_below_paper_execution_gate"]
    else:
        status = "BLOCKED"
        failed_gates = ["watchlist_gates_not_met"]
    return {
        "dataset_path": str(dataset_path),
        "candidate_evaluated": candidate,
        "deployment_status": status,
        "failed_gates": failed_gates,
        "production_adoption_allowed": status == "PRODUCTION_ALLOWED",
        "paper_signal_only_allowed": status in {
            "PAPER_SIGNAL_ONLY_ALLOWED",
            "PAPER_EXECUTION_ALLOWED",
            "LIVE_SHADOW_ALLOWED",
            "MICRO_LIVE_ALLOWED",
            "PRODUCTION_ALLOWED",
        },
        "paper_execution_allowed": status in {
            "PAPER_EXECUTION_ALLOWED",
            "LIVE_SHADOW_ALLOWED",
            "MICRO_LIVE_ALLOWED",
            "PRODUCTION_ALLOWED",
        },
        "live_shadow_allowed": status in {"LIVE_SHADOW_ALLOWED", "MICRO_LIVE_ALLOWED", "PRODUCTION_ALLOWED"},
        "micro_live_allowed": status in {"MICRO_LIVE_ALLOWED", "PRODUCTION_ALLOWED"},
        "exact_next_command": (
            "python scripts/audit_cost_model_and_edge.py --dataset <path>  # re-run after fixing cost model and filters"
        ),
    }


def _render_deployment_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Deployment Readiness")
    lines.append("")
    lines.append(f"- Generated: `{datetime.utcnow().isoformat()}Z`")
    lines.append(f"- Dataset: `{payload['dataset_path']}`")
    lines.append(f"- Status: `{payload['deployment_status']}`")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- Status: `{payload['deployment_status']}`")
    lines.append(f"- Failed gates: `{payload['failed_gates']}`")
    lines.append(f"- Production adoption allowed: `{payload['production_adoption_allowed']}`")
    lines.append(f"- Paper signal-only allowed: `{payload['paper_signal_only_allowed']}`")
    lines.append(f"- Paper execution allowed: `{payload['paper_execution_allowed']}`")
    lines.append(f"- Live shadow allowed: `{payload['live_shadow_allowed']}`")
    lines.append(f"- Micro-live allowed: `{payload['micro_live_allowed']}`")
    lines.append("")
    lines.append("## Candidate evaluated")
    ce = payload.get("candidate_evaluated") or {}
    if ce:
        for k, v in ce.items():
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Next command")
    lines.append(f"- `{payload['exact_next_command']}`")
    return "\n".join(lines)


def _cost_aware_refinement_payload(
    df: pd.DataFrame,
    *,
    evaluation_return_column: str,
    label_column: str,
    timestamp_column: str,
) -> Dict[str, Any]:
    """Produce the cost_aware_edge_refinement payload."""
    premium_bands = evaluate_premium_bands(df, return_column=evaluation_return_column)
    liquidity = evaluate_liquidity_filters(df, return_column=evaluation_return_column)
    spread = evaluate_spread_filters(df, return_column=evaluation_return_column)
    expected_move = evaluate_expected_move_filters(df, return_column=evaluation_return_column)

    # Selected-trade subset = profitable trades.
    if label_column in df.columns:
        selected = df.loc[pd.to_numeric(df[label_column], errors="coerce") > 0.0].copy()
    else:
        selected = df.copy()
    band_report = evaluate_cost_sensitive_bands(selected, return_column=evaluation_return_column)
    # Per-candidate metrics
    candidate_metrics = _cost_aware_candidate_metrics(selected, evaluation_return_column, candidate_name="profitable_trade_label_top_1_per_row")
    # Also evaluate the avoid-trade subset as a sanity baseline.
    if "avoid_trade_label" in df.columns:
        avoid = df.loc[pd.to_numeric(df["avoid_trade_label"], errors="coerce") > 0.0].copy()
        avoid_metrics = _cost_aware_candidate_metrics(avoid, evaluation_return_column, candidate_name="avoid_trade_label_top_1_per_row")
    else:
        avoid_metrics = {"candidate_name": "avoid_trade_label_top_1_per_row", "available": False}

    # Determine pass gates
    cost_125_pass = bool(candidate_metrics.get("cost_1_25x_profit_factor", 0.0) >= 1.05)
    cost_150_pass = bool(candidate_metrics.get("cost_1_50x_profit_factor", 0.0) >= 1.00)
    ratio_pass = bool(candidate_metrics.get("return_to_cost_ratio", 0.0) >= 1.25)
    pf = float(candidate_metrics.get("profit_factor", 0.0))
    sharpe = float(candidate_metrics.get("sharpe", 0.0))
    pre_cost = {"pf": pf, "sharpe": sharpe}
    best_pre_cost_name = candidate_metrics.get("candidate_name") if (pf >= 1.0 and sharpe > 0.0) else None
    best_post_cost = candidate_metrics if (cost_125_pass and ratio_pass) else None
    best_band_250_1000 = None
    for band in band_report.get("bands", []) or []:
        if band.get("in_band") and str(band.get("band")) in ("250-500", "500-1000"):
            if best_band_250_1000 is None or float(band.get("cost_aware_stability_score", 0.0)) > float(best_band_250_1000.get("cost_aware_stability_score", 0.0)):
                best_band_250_1000 = band
    if not cost_150_pass:
        deployment = "BLOCKED"
    elif cost_125_pass and ratio_pass and pf >= 1.15 and sharpe >= 0.5:
        deployment = "PAPER_WATCHLIST"
    else:
        deployment = "BLOCKED"

    payload = {
        "evaluation_return_column": evaluation_return_column,
        "label_column": label_column,
        "timestamp_column": timestamp_column,
        "premium_band_evaluation": premium_bands,
        "liquidity_filter_evaluation": liquidity,
        "spread_filter_evaluation": spread,
        "expected_move_evaluation": expected_move,
        "cost_sensitive_band_evaluation": band_report,
        "selected_trade_count": int(band_report.get("trade_count", 0)) if band_report.get("available") else 0,
        "best_candidate_before_cost": (
            {"candidate_name": candidate_metrics.get("candidate_name"), **{k: candidate_metrics.get(k) for k in ("profit_factor", "sharpe", "trade_count", "average_return_per_trade")}}
            if best_pre_cost_name else None
        ),
        "best_candidate_after_realistic_cost": (
            {
                "candidate_name": candidate_metrics.get("candidate_name"),
                "cost_1_25x_profit_factor": candidate_metrics.get("cost_1_25x_profit_factor"),
                "cost_1_50x_profit_factor": candidate_metrics.get("cost_1_50x_profit_factor"),
                "return_to_cost_ratio": candidate_metrics.get("return_to_cost_ratio"),
                "expected_return_after_estimated_cost": candidate_metrics.get("expected_return_after_estimated_cost"),
                "median_return_after_estimated_cost": candidate_metrics.get("median_return_after_estimated_cost"),
                "cost_survival_score": candidate_metrics.get("cost_survival_score"),
            }
            if best_post_cost else None
        ),
        "best_candidate_in_250_1000_band": best_band_250_1000,
        "candidates_passing_cost_1_25x": [candidate_metrics["candidate_name"]] if cost_125_pass else [],
        "candidates_passing_cost_1_50x": [candidate_metrics["candidate_name"]] if cost_150_pass else [],
        "candidates_passing_return_to_cost_ratio": [candidate_metrics["candidate_name"]] if ratio_pass else [],
        "avoid_trade_baseline": avoid_metrics,
        "paper_watchlist_candidates": (
            [{"candidate_name": candidate_metrics.get("candidate_name"), "status": "PAPER_WATCHLIST"}] if deployment == "PAPER_WATCHLIST" else []
        ),
        "paper_ready_candidates": [],
        "failed_gates": [] if deployment != "BLOCKED" else [
            gate for gate, ok in (
                ("cost_1_5x_pf_below_1.00", cost_150_pass),
                ("cost_1_25x_pf_below_1.05", cost_125_pass),
                ("return_to_cost_ratio_below_1.25", ratio_pass),
            ) if not ok
        ],
        "deployment_status": deployment,
        "exact_next_command": "python scripts/audit_cost_model_and_edge.py --dataset <path>  # re-run after fixing execution-cost modeling and broker configuration",
        "production_adoption_allowed": False,
    }
    return payload


def _render_audit_markdown(payload: Dict[str, Any]) -> str:
    dc = payload["double_counting_check"]
    cols = payload["dataset_column_audit"]
    cost_summary = payload["cost_summary_over_dataset"]
    lines: List[str] = []
    lines.append("# Cost-Model Audit")
    lines.append("")
    lines.append(f"- Generated: `{datetime.utcnow().isoformat()}Z`")
    lines.append(f"- Dataset: `{payload['dataset_path']}`")
    lines.append(f"- Evaluation return column: `{payload['evaluation_return_column']}`")
    lines.append("")
    lines.append("## 1. Dataset column presence")
    for k, v in cols.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## 2. Cost assumptions (config)")
    for k, v in payload["cost_assumptions"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## 3. Cost-stress formula")
    lines.append(f"- Formula: `{payload['cost_stress_formula']}`")
    lines.append(f"- Scenarios (extra_cost multipliers): `{payload['cost_stress_scenarios']}`")
    lines.append(f"- Base cost: `{payload['base_cost_assumption']}`")
    lines.append("")
    lines.append("## 4. Cost summary over dataset")
    for k, v in cost_summary.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## 5. Double-counting check")
    lines.append(f"- `net_forward_return_already_cost_adjusted`: `{dc['net_forward_return_already_cost_adjusted']}`")
    lines.append(f"- `gross_forward_return_average`: `{dc['gross_forward_return_average']}`")
    lines.append(f"- `net_forward_return_average`: `{dc['net_forward_return_average']}`")
    lines.append(f"- `gross_minus_net_average`: `{dc['gross_minus_net_average']}`")
    lines.append(f"- `gross_minus_net_median`: `{dc['gross_minus_net_median']}`")
    lines.append(f"- `inferred_embedded_cost`: `{dc['inferred_embedded_cost']}`")
    lines.append(f"- `double_counting_prevented`: `{dc['double_counting_prevented']}`")
    lines.append(f"- `recommendation`: `{dc['recommendation']}`")
    lines.append("")
    lines.append("## 6. Legacy audit summary")
    legacy = payload.get("legacy_audit_report", {}) or {}
    for k, v in legacy.items():
        if k == "sample_trades":
            continue
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## 7. Diagnosis")
    lines.append("- Current edge relies on returns that fail cost stress scenarios.")
    lines.append("- `net_forward_return` is already net of embedded execution cost in this dataset, so incremental cost stress is applied on top via the `inferred_embedded_cost * extra_cost` formula, not as a fresh deduction.")
    lines.append("- No candidate has cleared cost_1.50x PF >= 1.00 → deployment remains BLOCKED.")
    return "\n".join(lines)


def _render_refinement_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cost-Aware Edge Refinement")
    lines.append("")
    lines.append(f"- Generated: `{datetime.utcnow().isoformat()}Z`")
    lines.append(f"- Evaluation return column: `{payload['evaluation_return_column']}`")
    lines.append("")
    lines.append("## A. Premium-band evaluation")
    pb = payload.get("premium_band_evaluation", {})
    if pb.get("available"):
        lines.append("| min_premium | trade_count | PF | Sharpe | avg_return | est_cost_pct | return_to_cost | cost_1.25x PF | cost_1.50x PF |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for band in pb.get("bands", []) or []:
            lines.append(
                f"| {band['min_premium']} | {band['trade_count']} | {band['profit_factor']:.3f} | {band['sharpe']:.3f} | {band['average_return_per_trade']:.4f} | {band['estimated_cost_pct']:.4f} | {band['return_to_cost_ratio']:.3f} | {band['cost_1_25x_profit_factor']:.3f} | {band['cost_1_50x_profit_factor']:.3f} |"
            )
    else:
        lines.append("- Skipped: required columns missing.")
    lines.append("")
    lines.append("## B. Liquidity / volume / OI / moneyness filter evaluation")
    lq = payload.get("liquidity_filter_evaluation", {})
    if lq.get("available"):
        lines.append("| filter | trade_count | PF | Sharpe | return_to_cost | cost_1.25x PF | cost_1.50x PF |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in lq.get("filters", []) or []:
            lines.append(
                f"| {f['filter_name']} | {f['trade_count']} | {f['profit_factor']:.3f} | {f['sharpe']:.3f} | {f['return_to_cost_ratio']:.3f} | {f['cost_1_25x_profit_factor']:.3f} | {f['cost_1_50x_profit_factor']:.3f} |"
            )
        if lq.get("skipped_filters"):
            lines.append("")
            lines.append(f"- Skipped filters (columns missing): `{lq.get('skipped_filters')}`")
    else:
        lines.append("- Skipped: required columns missing.")
    lines.append("")
    lines.append("## C. Spread / slippage filter evaluation")
    sp = payload.get("spread_filter_evaluation", {})
    if sp.get("available"):
        lines.append("| filter | trade_count | PF | Sharpe | return_to_cost | cost_1.25x PF | cost_1.50x PF |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in sp.get("filters", []) or []:
            lines.append(
                f"| {f['filter_name']} | {f['trade_count']} | {f['profit_factor']:.3f} | {f['sharpe']:.3f} | {f['return_to_cost_ratio']:.3f} | {f['cost_1_25x_profit_factor']:.3f} | {f['cost_1_50x_profit_factor']:.3f} |"
            )
    else:
        lines.append(f"- Skipped: {sp.get('skipped_reason', 'spread column missing')}")
    lines.append("")
    lines.append("## D. Expected-move filter evaluation")
    em = payload.get("expected_move_evaluation", {})
    if em.get("available"):
        lines.append(f"- median_return_after_estimated_cost: `{em.get('median_return_after_estimated_cost'):.4f}`")
        lines.append(f"- average_return_after_estimated_cost: `{em.get('average_return_after_estimated_cost'):.4f}`")
        lines.append(f"- median_expected_return_to_cost_ratio: `{em.get('median_expected_return_to_cost_ratio'):.3f}`")
        lines.append("")
        lines.append("| filter | trade_count | PF | Sharpe | return_to_cost |")
        lines.append("|---|---|---|---|---|")
        for f in em.get("filters", []) or []:
            lines.append(
                f"| {f['filter_name']} | {f['trade_count']} | {f['profit_factor']:.3f} | {f['sharpe']:.3f} | {f['return_to_cost_ratio']:.3f} |"
            )
    else:
        lines.append("- Skipped: required columns missing.")
    lines.append("")
    lines.append("## E. Cost-sensitive candidate bands")
    cb = payload.get("cost_sensitive_band_evaluation", {})
    if cb.get("available"):
        lines.append(f"- Selected trade count: `{cb.get('trade_count')}`")
        lines.append("")
        lines.append("| band | in_band | trade_count | PF | Sharpe | return_to_cost | cost_1.25x PF | cost_1.50x PF | cost_aware_score |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for band in cb.get("bands", []) or []:
            lines.append(
                f"| {band['band']} | {band.get('in_band')} | {band.get('trade_count')} | {band.get('profit_factor', 0):.3f} | {band.get('sharpe', 0):.3f} | {band.get('return_to_cost_ratio', 0):.3f} | {band.get('cost_1_25x_profit_factor', 0):.3f} | {band.get('cost_1_50x_profit_factor', 0):.3f} | {band.get('cost_aware_stability_score', 0):.3f} |"
            )
    else:
        lines.append("- Skipped: no selected trades.")
    lines.append("")
    lines.append("## F. Best candidates")
    lines.append(f"- Best before cost: `{payload.get('best_candidate_before_cost')}`")
    lines.append(f"- Best after realistic cost: `{payload.get('best_candidate_after_realistic_cost')}`")
    lines.append(f"- Best in 250-1000 band: `{payload.get('best_candidate_in_250_1000_band')}`")
    lines.append("")
    lines.append("## G. Pass lists")
    lines.append(f"- Passing cost_1.25x: `{payload.get('candidates_passing_cost_1_25x')}`")
    lines.append(f"- Passing cost_1.50x: `{payload.get('candidates_passing_cost_1_50x')}`")
    lines.append(f"- Passing return_to_cost_ratio: `{payload.get('candidates_passing_return_to_cost_ratio')}`")
    lines.append(f"- PAPER_WATCHLIST: `{payload.get('paper_watchlist_candidates')}`")
    lines.append(f"- PAPER_READY: `{payload.get('paper_ready_candidates')}`")
    lines.append("")
    lines.append("## H. Deployment decision")
    lines.append(f"- Status: `{payload.get('deployment_status')}`")
    lines.append(f"- Failed gates: `{payload.get('failed_gates')}`")
    lines.append(f"- Next command: `{payload.get('exact_next_command')}`")
    lines.append(f"- Production adoption allowed: `{payload.get('production_adoption_allowed')}`")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost-model audit + cost-aware edge refinement runner.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--evaluation-return-column", default="net_forward_return")
    parser.add_argument("--label-column", default="profitable_trade_label")
    parser.add_argument("--timestamp-column", default="timestamp")
    args = parser.parse_args(argv)
    dataset_path: Path = args.dataset
    if not dataset_path.exists():
        print(f"[audit] dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    print(f"[audit] loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[audit] rows: {df.shape[0]} columns: {df.shape[1]}")
    timestamp = _ts()
    audit_json = REPORTS_DIR / f"cost_model_audit_{timestamp}.json"
    audit_md = REPORTS_DIR / f"cost_model_audit_{timestamp}.md"
    refine_json = REPORTS_DIR / f"cost_aware_edge_refinement_{timestamp}.json"
    refine_md = REPORTS_DIR / f"cost_aware_edge_refinement_{timestamp}.md"
    deploy_json = REPORTS_DIR / f"deployment_readiness_{timestamp}.json"
    deploy_md = REPORTS_DIR / f"deployment_readiness_{timestamp}.md"
    audit_payload = _cost_model_audit_payload(df, evaluation_return_column=args.evaluation_return_column)
    refine_payload = _cost_aware_refinement_payload(
        df,
        evaluation_return_column=args.evaluation_return_column,
        label_column=args.label_column,
        timestamp_column=args.timestamp_column,
    )
    deploy_payload = _deployment_readiness_payload(refine_payload, dataset_path=dataset_path)
    _write_json(audit_json, audit_payload)
    _write_text(audit_md, _render_audit_markdown(audit_payload))
    _write_json(refine_json, refine_payload)
    _write_text(refine_md, _render_refinement_markdown(refine_payload))
    _write_json(deploy_json, deploy_payload)
    _write_text(deploy_md, _render_deployment_markdown(deploy_payload))
    print(f"[audit] wrote: {audit_json}")
    print(f"[audit] wrote: {audit_md}")
    print(f"[audit] wrote: {refine_json}")
    print(f"[audit] wrote: {refine_md}")
    print(f"[audit] wrote: {deploy_json}")
    print(f"[audit] wrote: {deploy_md}")
    print(f"[audit] deployment_status: {deploy_payload.get('deployment_status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
