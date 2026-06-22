"""Model-diagnosis audits for the cost-aware retrain.

Tasks implemented (analysis-only, NO new model families, NO retraining):

  1. Label separability audit        -> reports/label_separability_audit_<ts>.{json,md}
  2. Probability decile audit        -> reports/probability_decile_audit_<ts>.{json,md}
  3. Feature ablation retrain        -> reports/feature_ablation_retrain_<ts>.{json,md}
  4. Label learnability report       -> reports/label_learnability_report_<ts>.{json,md}
  5. Two-stage profit/avoid overlay  -> reports/two_stage_profit_avoid_overlay_<ts>.{json,md}

The script reads:
- data/processed/nifty_option_chain_cost_aware_edge_dataset_<ts>.csv  (built by
  scripts/build_cost_aware_edge_dataset.py)
- models/core_retrain_<ts>/*.pkl + *_metrics.json (per-model OOF-sweep
  artefacts — note: per-row OOF probabilities are NOT persisted by the
  retrain pipeline, so decile-level and learnability analyses operate on
  threshold-sweep and metrics aggregates only).

Honest about the limitation: per-row OOF probability decile tables are
approximated from the available threshold sweep when OOF predictions are
missing. When a richer per-row OOF file is added to the retrain
artefacts, this script will pick it up automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Columns that are forbidden as features (target / future leakage).
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
    "expected_return_after_cost",
    "return_to_cost_ratio",
    "cost_return_units_estimated",
)

# Live-computable feature candidates (subset known to exist on the dataset).
LIVE_FEATURE_CANDIDATES: Tuple[str, ...] = (
    "ltp", "volume", "oi", "dte_days", "is_weekly",
    "option_type_ce", "option_type_pe", "range_pct",
    "oc_change_pct", "hl_change_pct", "oi_change_pct", "volume_change_pct",
    "ret_1", "ret_3", "ret_5", "ret_10",
    "oi_z_5", "volume_z_5", "weekday", "month",
    "moneyness", "atm_distance", "strike_distance_pct", "ctx_dte_norm",
    "body_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
    "close_location_pct", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct",
    "ctx_option_price", "option_to_spot_pct", "volume_ratio",
    "ret_mean", "ret_std", "ret_min", "ret_max", "vol_mean", "vol_std",
    "vol_min", "vol_max", "adx_14", "choppiness_14", "supertrend_dir",
    "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
    "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
    "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
    "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
    "vol_of_vol_14", "dist_from_opening_high_pct", "dist_from_opening_low_pct",
    "opening_range_width_pct", "opening_range_breakout_strength",
    "dist_to_rolling_high_20", "dist_to_rolling_low_20",
    "rolling_range_width_20", "rolling_range_position_20",
    "realized_vol_30", "atr_pct_regime_10", "atr_percentile_60",
    "realized_vol_percentile_60", "volatility_percentile_60",
    "volatility_regime_classifier",
    "oi_CE", "oi_PE", "volume_CE", "volume_PE",
    "ce_pe_oi_ratio", "ce_pe_volume_ratio",
    "bs_iv", "final_iv", "bs_delta", "bs_gamma", "bs_theta", "bs_vega", "bs_rho",
    "log_moneyness", "distance_from_atm", "distance_from_atm_pct",
    "intrinsic_value", "extrinsic_value",
    "time_to_expiry_days", "time_to_expiry_years",
    "is_expiry_day", "is_near_expiry",
    "bid_ask_spread", "bid_ask_spread_pct", "mid_price",
    "ltp_vs_mid_diff", "ltp_vs_mid_diff_pct",
    "has_valid_bid_ask", "low_price_flag", "wide_spread_flag",
    "bad_iv_flag", "bad_greek_flag", "stale_or_invalid_quote_flag",
    "deep_itm_flag", "deep_otm_flag", "greeks_quality_score",
    "is_opening_session", "is_closing_session", "is_midday_lull",
    "spot_return_1", "spot_return_3", "spot_return_5",
    "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap",
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Task 1 — Label separability audit
# ---------------------------------------------------------------------------

def _mutual_info_score(x: np.ndarray, y_bin: np.ndarray) -> float:
    """Discrete mutual information (nats) using empirical histograms."""
    if x.size == 0 or y_bin.size == 0:
        return 0.0
    # Discretise x into 16 quantile bins
    try:
        binned = pd.qcut(x, q=16, labels=False, duplicates="drop")
    except Exception:
        binned = np.zeros_like(x, dtype=int)
    binned = np.asarray(binned, dtype=int)
    y_bin = np.asarray(y_bin, dtype=int)
    mask = (binned >= 0) & (y_bin >= 0)
    if not mask.any():
        return 0.0
    binned = binned[mask]
    y_bin = y_bin[mask]
    p_y = np.bincount(y_bin) / max(len(y_bin), 1)
    h_y = float(-np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0] + 1e-12)))
    joint = pd.crosstab(pd.Series(binned), pd.Series(y_bin)).to_numpy(dtype=float)
    p_joint = joint / joint.sum() if joint.sum() > 0 else joint
    p_x = p_joint.sum(axis=1)
    p_y_arr = p_joint.sum(axis=0)
    mi = 0.0
    for i in range(p_joint.shape[0]):
        for j in range(p_joint.shape[1]):
            if p_joint[i, j] > 0 and p_x[i] > 0 and p_y_arr[j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_x[i] * p_y_arr[j]) + 1e-12)
    return float(mi / max(h_y, 1e-9))  # normalised MI in [0, 1]


def _auc_score(x: np.ndarray, y_bin: np.ndarray) -> float:
    """Univariate ROC-AUC (hand-rolled, no sklearn, memory-bounded).

    Uses sorting + cumulative counts instead of a full O(n_pos*n_neg)
    outer product. Acceptable for n up to 1e6.
    """
    if x.size == 0 or y_bin.size == 0 or y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return 0.5
    pos_mask = y_bin == 1
    n_pos = int(pos_mask.sum())
    n_neg = int(y_bin.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(x, kind="mergesort")
    y_sorted = y_bin[order]
    rank = np.arange(1, len(x) + 1, dtype=np.int64)
    pos_ranks = rank[(y_sorted == 1)]
    u = float(pos_ranks.sum()) - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


def label_separability_audit(df: pd.DataFrame, labels: Sequence[str], features: Sequence[str]) -> Dict[str, Any]:
    """For each label, per-feature positive vs negative comparison.

    Returns:
        {
            "labels": {label_name: per_feature_stats, ...},
            "ranked_features": [...top 30 + bottom 30 by absolute AUC...],
            "unstable_features": [...features with high month-to-month variance...],
            "leakage_candidates": [...features that look too good to be true (AUC > 0.85 and not in known safe list)...]
        }
    """
    work = df.copy()
    feature_cols = [c for c in features if c in work.columns]
    out_per_label: Dict[str, Any] = {}
    for label in labels:
        if label not in work.columns:
            out_per_label[label] = {"available": False, "reason": f"{label} column missing"}
            continue
        y = pd.to_numeric(work[label], errors="coerce")
        mask = y.notna()
        y_bin = y[mask].to_numpy(dtype=float)
        # convert {0,1,NaN} -> {0,1,NaN} kept as int
        y_bin_int = np.where(y_bin > 0.5, 1, np.where(y_bin < 0.5, 0, -1))
        valid = y_bin_int >= 0
        yv = y_bin_int[valid]
        pos = int((yv == 1).sum())
        neg = int((yv == 0).sum())
        per_feature: List[Dict[str, Any]] = []
        for f in feature_cols:
            arr = _safe_numeric(work.loc[mask, f])[valid]
            if arr.std() == 0.0:
                auc = 0.5
                mi = 0.0
            else:
                auc = _auc_score(arr, yv)
                mi = _mutual_info_score(arr, yv)
            pos_mean = float(arr[yv == 1].mean()) if pos > 0 else 0.0
            neg_mean = float(arr[yv == 0].mean()) if neg > 0 else 0.0
            pos_med = float(np.median(arr[yv == 1])) if pos > 0 else 0.0
            neg_med = float(np.median(arr[yv == 0])) if neg > 0 else 0.0
            per_feature.append({
                "feature": f,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "mean_diff": pos_mean - neg_mean,
                "pos_median": pos_med,
                "neg_median": neg_med,
                "median_diff": pos_med - neg_med,
                "abs_auc_diff_from_half": abs(auc - 0.5),
                "auc": auc,
                "mutual_info": mi,
                "missing_rate": float((work.loc[mask, f].isna().mean())),
            })
        per_feature.sort(key=lambda r: r["abs_auc_diff_from_half"], reverse=True)
        out_per_label[label] = {
            "available": True,
            "positive_count": pos,
            "negative_count": neg,
            "positive_rate": float(pos / max(pos + neg, 1)),
            "per_feature": per_feature,
            "top_30_useful": per_feature[:30],
        }

    # Cross-label ranking by mean abs AUC diff (stable signals only)
    all_features = feature_cols
    feature_score: Dict[str, float] = {}
    feature_mi: Dict[str, float] = {}
    for f in all_features:
        aucs = []
        mis = []
        for label, payload in out_per_label.items():
            if not payload.get("available"):
                continue
            row = next((p for p in payload["per_feature"] if p["feature"] == f), None)
            if row is None:
                continue
            aucs.append(row["abs_auc_diff_from_half"])
            mis.append(row["mutual_info"])
        if not aucs:
            continue
        feature_score[f] = float(np.mean(aucs))
        feature_mi[f] = float(np.mean(mis))
    ranked = sorted(feature_score.items(), key=lambda kv: kv[1], reverse=True)

    # Stability by year/month: compute per-feature month variance of mean diff
    unstable: List[Dict[str, Any]] = []
    if "timestamp" in work.columns and "month" not in work.columns:
        try:
            work["month"] = pd.to_datetime(work["timestamp"], errors="coerce").dt.to_period("M").astype(str)
        except Exception:
            work["month"] = "unknown"
    if "month" in work.columns and "profitable_trade_label" in work.columns:
        for f in all_features:
            rows = []
            for m, grp in work.groupby("month", sort=True):
                if len(grp) < 50 or grp["profitable_trade_label"].isna().all():
                    continue
                yv = grp["profitable_trade_label"].fillna(0.0).to_numpy()
                arr = _safe_numeric(grp[f])
                if yv.sum() == 0 or yv.sum() == len(yv):
                    continue
                auc = _auc_score(arr, yv)
                rows.append({"month": str(m), "auc": auc, "n": int(len(grp))})
            if len(rows) < 3:
                continue
            aucs = [r["auc"] for r in rows]
            if not aucs:
                continue
            aucs_arr = np.array(aucs)
            std_auc = float(aucs_arr.std())
            mean_auc = float(aucs_arr.mean())
            # Flag if either:
            # (a) std is very high (>0.10) AND |mean-0.5|>0.05 (signal flips AND has magnitude), OR
            # (b) std is extreme (>0.20) regardless of mean (signal flips between months)
            if (std_auc > 0.10 and abs(mean_auc - 0.5) > 0.05) or std_auc > 0.20:
                unstable.append({
                    "feature": f,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "month_count": len(rows),
                })
    unstable.sort(key=lambda r: r["std_auc"], reverse=True)

    # Leakage candidates: high AUC > 0.85 AND not in known-safe list
    safe_features = {"ltp", "volume", "oi", "moneyness", "dte_days", "ret_1", "ret_3", "ret_5", "ret_10",
                     "bid_ask_spread", "bid_ask_spread_pct", "oi_z_5", "volume_z_5", "regime_volatile",
                     "regime_trending", "regime_mean_reverting", "regime_quiet", "weekday", "month"}
    leakage_candidates = [f for f, s in feature_score.items() if s > 0.35 and f not in safe_features and f not in FORBIDDEN_FEATURE_COLUMNS]
    # Note: AUC diff > 0.35 corresponds to AUC > 0.85 or < 0.15

    return {
        "labels": list(out_per_label.keys()),
        "per_label": out_per_label,
        "top_30_useful_features": [{"feature": f, "mean_abs_auc_diff": s, "mean_mutual_info": feature_mi.get(f, 0.0)} for f, s in ranked[:30]],
        "bottom_30_useless_features": [{"feature": f, "mean_abs_auc_diff": s, "mean_mutual_info": feature_mi.get(f, 0.0)} for f, s in ranked[-30:]],
        "unstable_features": unstable[:30],
        "leakage_candidates": leakage_candidates,
    }


# ---------------------------------------------------------------------------
# Task 2 — Probability decile audit (uses actual OOF predictions when available)
# ---------------------------------------------------------------------------

def _analyze_oof_deciles(oof_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze OOF predictions into true probability deciles.

    Computes per-decile metrics using actual y_prob values, not threshold-sweep proxy.
    Returns decile-level statistics including profit_factor, sharpe, win_rate per decile.
    """
    if oof_df.empty:
        return {"available": False, "reason": "empty OOF DataFrame"}

    probs = oof_df["y_prob"].to_numpy(dtype=float)
    y_true = oof_df["y_true"].to_numpy(dtype=int) if "y_true" in oof_df.columns else np.zeros(len(oof_df), dtype=int)
    returns = oof_df["selected_return_column"].to_numpy(dtype=float) if "selected_return_column" in oof_df.columns else None

    if len(probs) < 100:
        return {"available": False, "reason": "insufficient OOF samples"}

    # Create 10 equal-frequency deciles based on y_prob
    try:
        oof_df["decile"] = pd.qcut(probs, q=10, labels=False, duplicates="drop")
    except Exception:
        # Fallback to 10 equal-width bins if qcut fails
        oof_df["decile"] = pd.cut(probs, bins=10, labels=False)

    decile_rows = []
    for decile_val, group in oof_df.groupby("decile"):
        decile_num = int(decile_val) if pd.notna(decile_val) else -1
        n = len(group)
        if n < 5:
            continue

        probs_decile = group["y_prob"].to_numpy(dtype=float)
        y_decile = group["y_true"].to_numpy(dtype=int)
        threshold = float(np.median(probs_decile))

        # Calculate metrics at decile median threshold
        predictions = (probs_decile >= threshold).astype(int)
        tp = int(np.sum((y_decile == 1) & (predictions == 1)))
        fp = int(np.sum((y_decile == 0) & (predictions == 1)))
        tn = int(np.sum((y_decile == 0) & (predictions == 0)))
        fn = int(np.sum((y_decile == 1) & (predictions == 0)))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        win_rate = float(np.mean(y_decile)) if len(y_decile) else 0.0

        # Trading metrics from returns if available
        trade_metrics = {
            "trade_count": n,
            "profit_factor": None,
            "sharpe": None,
            "average_return": None,
            "total_return": None,
            "win_rate": win_rate,
        }
        if returns is not None:
            decile_returns = group["selected_return_column"].to_numpy(dtype=float)
            selected_returns = decile_returns  # Already filtered to decile subset

            wins = selected_returns[selected_returns > 0]
            losses = selected_returns[selected_returns < 0]
            gross_profit = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(np.abs(losses.sum())) if len(losses) else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

            sharpe = 0.0
            if len(selected_returns) > 1:
                std = float(np.std(selected_returns, ddof=1))
                if std > 0:
                    sharpe = float((np.mean(selected_returns) / std) * math.sqrt(252.0))

            trade_metrics.update({
                "trade_count": n,
                "profit_factor": profit_factor,
                "sharpe": sharpe,
                "average_return": float(np.mean(selected_returns)),
                "total_return": float(np.sum(selected_returns)),
                "win_rate": win_rate,
            })

        decile_rows.append({
            "decile": decile_num,
            "decile_label": f"Q{decile_num + 1}",  # Q1 = lowest probabilities, Q10 = highest
            "sample_count": n,
            "prob_min": float(probs_decile.min()),
            "prob_max": float(probs_decile.max()),
            "prob_mean": float(probs_decile.mean()),
            "prob_median": float(np.median(probs_decile)),
            "actual_positive_rate": win_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **trade_metrics,
        })

    decile_rows.sort(key=lambda r: r["decile"])

    # Check monotonicity: each higher decile should have >= win_rate than lower
    win_rates = [r["actual_positive_rate"] for r in decile_rows]
    avg_returns = [r.get("average_return") for r in decile_rows]

    monotonic_win_rate = all(
        win_rates[i] <= win_rates[i + 1] + 1e-9
        for i in range(len(win_rates) - 1)
    )

    # Check profit factor increases with probability (if returns available)
    monotonic_profit = True
    if all(r is not None for r in avg_returns):
        monotonic_profit = all(
            (avg_returns[i] or 0) <= (avg_returns[i + 1] or 0) + 1e-9
            for i in range(len(avg_returns) - 1)
        )

    # Best vs worst decile comparison
    top_decile = decile_rows[-1] if decile_rows else {}
    bottom_decile = decile_rows[0] if decile_rows else {}

    return {
        "available": True,
        "n_deciles": len(decile_rows),
        "total_samples": sum(r["sample_count"] for r in decile_rows),
        "deciles": decile_rows,
        "top_decile": top_decile,
        "bottom_decile": bottom_decile,
        "monotonic_win_rate_across_deciles": monotonic_win_rate,
        "monotonic_profit_across_deciles": monotonic_profit,
        "monotonic_top_beats_bottom": bool(
            (top_decile.get("profit_factor") or 0) >= (bottom_decile.get("profit_factor") or 0)
        ) if top_decile and bottom_decile else False,
    }


def _find_oof_files(artifact_dir: Path, label_name: str, model_name: str) -> tuple[Path | None, Path | None]:
    """Find OOF parquet files for a given model/label combination.

    Returns (holdout_oof_path, walkforward_oof_path) or (None, None) if not found.
    """
    safe_label = str(label_name).replace("/", "_").replace("\\", "_")
    safe_model = str(model_name).replace("/", "_").replace("\\", "_")

    holdout_pattern = f"oof_predictions_{safe_label}_{safe_model}_*.parquet"
    wf_pattern = f"oof_predictions_{safe_label}_{safe_model}_walkforward_*.parquet"

    holdout_files = list(artifact_dir.glob(holdout_pattern))
    wf_files = list(artifact_dir.glob(wf_pattern))

    holdout_path = holdout_files[-1] if holdout_files else None
    wf_path = wf_files[-1] if wf_files else None

    return holdout_path, wf_path


def probability_decile_audit(artifact_dir: Path, labels: Sequence[str]) -> Dict[str, Any]:
    """For each model/label metrics file, compute probability decile audit.

    This function now uses actual OOF predictions when available:
    1. First checks for OOF parquet files (oof_predictions_<label>_<model>_<ts>.parquet)
    2. If OOF files exist, uses them for true per-row decile analysis
    3. Falls back to threshold-sweep proxy if OOF files are not available

    The OOF-based analysis provides:
    - True probability deciles (10 equal-frequency bins based on y_prob)
    - Per-decile metrics: profit_factor, sharpe, win_rate, sample_count
    - True monotonicity check across all deciles
    - Top vs bottom decile comparison
    """
    if not artifact_dir.exists():
        return {"available": False, "reason": f"artifact dir missing: {artifact_dir}"}
    results: Dict[str, Any] = {}
    oof_sources_used = 0
    sweep_proxy_used = 0

    for metrics_file in sorted(artifact_dir.glob("core_retrain_*_metrics.json")):
        try:
            m = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_name = m.get("model_name") or metrics_file.stem.replace("_metrics", "")
        label_name = m.get("label_name") or "unknown"
        key = f"{model_name}__{label_name}"

        # Try to find and use OOF predictions first
        holdout_path, wf_path = _find_oof_files(artifact_dir, label_name, model_name)
        oof_result: Dict[str, Any] = {"source": "none"}

        if wf_path and wf_path.exists():
            # Prefer walk-forward OOF as it covers more samples
            try:
                oof_df = pd.read_parquet(wf_path)
                oof_result = _analyze_oof_deciles(oof_df)
                if oof_result.get("available"):
                    oof_sources_used += 1
                    results[key] = {
                        "model_name": model_name,
                        "label_name": label_name,
                        "oof_source": str(wf_path),
                        "oof_type": "walk_forward",
                        **oof_result,
                    }
                    continue
            except Exception:
                pass

        if holdout_path and holdout_path.exists():
            # Fall back to holdout OOF
            try:
                oof_df = pd.read_parquet(holdout_path)
                oof_result = _analyze_oof_deciles(oof_df)
                if oof_result.get("available"):
                    oof_sources_used += 1
                    results[key] = {
                        "model_name": model_name,
                        "label_name": label_name,
                        "oof_source": str(holdout_path),
                        "oof_type": "holdout",
                        **oof_result,
                    }
                    continue
            except Exception:
                pass

        # Fallback to threshold-sweep proxy (original behavior)
        sweep = m.get("threshold_sweep") or []
        if not sweep:
            sweep_proxy_used += 1
            results[key] = {
                "model_name": model_name,
                "label_name": label_name,
                "oof_source": None,
                "oof_type": "none",
                "available": False,
                "reason": "no OOF file and no threshold sweep available",
            }
            continue

        sweep_sorted = sorted(sweep, key=lambda r: r.get("positive_prediction_rate", 0))
        n = len(sweep_sorted)
        if n >= 5:
            top_decile = sweep_sorted[0]
            bottom_decile = sweep_sorted[-1]
            top_5pct = sweep_sorted[0]
            top_1pct = sweep_sorted[0]
        else:
            top_decile = bottom_decile = top_5pct = top_1pct = sweep_sorted[0]

        monotonic_top_beats_bottom = bool(
            (top_decile.get("profit_factor", 0) or 0) >= (bottom_decile.get("profit_factor", 0) or 0)
        )
        sweep_proxy_used += 1
        per_decile = []
        for r in sweep_sorted:
            per_decile.append({
                "threshold": r.get("threshold"),
                "positive_prediction_rate": r.get("positive_prediction_rate"),
                "trade_count": r.get("number_of_trades"),
                "profit_factor": r.get("profit_factor"),
                "sharpe": r.get("sharpe"),
                "average_net_forward_return": r.get("average_net_forward_return"),
                "win_rate": r.get("win_rate"),
            })

        results[key] = {
            "model_name": model_name,
            "label_name": label_name,
            "oof_source": None,
            "oof_type": "none",
            "n_sweep_rows": n,
            "top_decile": top_decile,
            "bottom_decile": bottom_decile,
            "top_5pct": top_5pct,
            "top_1pct": top_1pct,
            "monotonic_top_beats_bottom": monotonic_top_beats_bottom,
            "per_threshold": per_decile,
        }

    n_models = len({v.get("model_name") for v in results.values()})
    n_labels = len({v.get("label_name") for v in results.values()})

    # Count monotonicity based on actual data type
    oof_based = [v for v in results.values() if v.get("oof_type") in {"walk_forward", "holdout"}]
    sweep_based = [v for v in results.values() if v.get("oof_type") == "none"]

    monotonic_oof = sum(1 for v in oof_based if v.get("monotonic_top_beats_bottom"))
    monotonic_sweep = sum(1 for v in sweep_based if v.get("monotonic_top_beats_bottom"))

    return {
        "available": True,
        "artifact_dir": str(artifact_dir),
        "per_model_label": results,
        "summary": {
            "n_models": n_models,
            "n_labels": n_labels,
            "oof_based_count": len(oof_based),
            "sweep_proxy_count": len(sweep_based),
            "monotonic_count": monotonic_oof + monotonic_sweep,
            "non_monotonic_count": (len(oof_based) + len(sweep_based)) - (monotonic_oof + monotonic_sweep),
        },
        "data_source": {
            "oof_predictions_used": oof_sources_used,
            "threshold_sweep_proxy_used": sweep_proxy_used,
        },
        "note": "OOF predictions are now persisted in core_retrain_* artifacts as oof_predictions_<label>_<model>_<ts>.parquet. The decile audit uses actual per-row probabilities for true monotonicity analysis when OOF files are available. Falls back to threshold-sweep proxy when OOF files are missing.",
    }


# ---------------------------------------------------------------------------
# Task 3 — Feature ablation (lightweight, no model retrain)
# ---------------------------------------------------------------------------

FEATURE_SETS: Dict[str, List[str]] = {
    "top_20_stable_features": [],   # populated at runtime
    "top_40_stable_features": [],
    "no_greeks_features": [],
    "price_volume_moneyness_only": [
        "ltp", "volume", "oi", "moneyness", "dte_days",
        "ret_1", "ret_3", "ret_5", "ret_10", "strike_distance_pct",
        "bid_ask_spread", "bid_ask_spread_pct", "oi_z_5", "volume_z_5",
    ],
    "regime_time_liquidity_only": [
        "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
        "weekday", "is_opening_session", "is_closing_session", "is_midday_lull",
        "bid_ask_spread_pct", "volume_ratio", "oi_change_pct", "volume_change_pct",
    ],
    "exclude_unstable_features": [],
    "exclude_premium_distorted_features": [],
}


def _fill_dynamic_feature_sets(separability: Dict[str, Any]) -> None:
    ranked = separability.get("top_30_useful_features", [])
    stable_only = [r for r in ranked if r.get("mean_abs_auc_diff", 0) > 0.01]
    top_20 = [r["feature"] for r in ranked[:20]]
    top_40 = [r["feature"] for r in ranked[:40]]
    unstable_set = {r["feature"] for r in separability.get("unstable_features", [])}
    leakage_set = set(separability.get("leakage_candidates", []))
    FEATURE_SETS["top_20_stable_features"] = [f for f in top_20 if f not in unstable_set and f not in leakage_set]
    FEATURE_SETS["top_40_stable_features"] = [f for f in top_40 if f not in unstable_set and f not in leakage_set]
    FEATURE_SETS["exclude_unstable_features"] = [
        r["feature"] for r in stable_only
        if r["feature"] not in unstable_set and r["feature"] not in leakage_set
    ]
    greek_features = {"bs_iv", "final_iv", "bs_delta", "bs_gamma", "bs_theta", "bs_vega", "bs_rho",
                       "log_moneyness", "distance_from_atm", "distance_from_atm_pct",
                       "intrinsic_value", "extrinsic_value", "greeks_quality_score",
                       "bad_iv_flag", "bad_greek_flag"}
    FEATURE_SETS["no_greeks_features"] = [
        r["feature"] for r in stable_only
        if r["feature"] not in greek_features and r["feature"] not in leakage_set
    ]
    premium_features = {"ltp", "volume", "oi", "ctx_option_price", "option_to_spot_pct"}
    FEATURE_SETS["exclude_premium_distorted_features"] = [
        r["feature"] for r in stable_only
        if r["feature"] not in premium_features and r["feature"] not in leakage_set
    ]


def feature_ablation_metrics(df: pd.DataFrame, label: str, feature_set: List[str]) -> Dict[str, Any]:
    """Compute univariate AUC + MI for the given feature subset on the given label.
    No model is retrained; this is a cheap separability comparison across feature sets."""
    if label not in df.columns or not feature_set:
        return {"available": False, "feature_count": len(feature_set)}
    y = pd.to_numeric(df[label], errors="coerce")
    mask = y.notna()
    yv = np.where(y[mask] > 0.5, 1, 0)
    if yv.sum() == 0 or yv.sum() == len(yv):
        return {"available": False, "reason": "label constant"}
    aucs = []
    mis = []
    for f in feature_set:
        if f not in df.columns:
            continue
        arr = _safe_numeric(df.loc[mask, f])
        if arr.std() == 0.0:
            continue
        aucs.append(_auc_score(arr, yv))
        mis.append(_mutual_info_score(arr, yv))
    if not aucs:
        return {"available": False, "reason": "no usable features"}
    return {
        "available": True,
        "feature_count": int(len(feature_set)),
        "mean_abs_auc_diff": float(np.mean([abs(a - 0.5) for a in aucs])),
        "max_abs_auc_diff": float(np.max([abs(a - 0.5) for a in aucs])),
        "mean_mutual_info": float(np.mean(mis)),
        "aucs": aucs,
    }


def feature_ablation_audit(df: pd.DataFrame, labels: Sequence[str], separability: Dict[str, Any]) -> Dict[str, Any]:
    _fill_dynamic_feature_sets(separability)
    out: Dict[str, Any] = {"per_feature_set": {}, "per_label": {}}
    for fs_name, fs_cols in FEATURE_SETS.items():
        per_label = {}
        for label in labels:
            if label not in df.columns:
                continue
            metrics = feature_ablation_metrics(df, label, fs_cols)
            per_label[label] = metrics
        out["per_feature_set"][fs_name] = {
            "feature_count": len(fs_cols),
            "per_label": per_label,
        }
    return out


# ---------------------------------------------------------------------------
# Task 4 — Label learnability (lightweight, no model retrain)
# ---------------------------------------------------------------------------

def label_learnability(df: pd.DataFrame, labels: Sequence[str], features: Sequence[str]) -> Dict[str, Any]:
    """Compare current-feature predictive power vs shuffled-label / date-shifted / random.

    We use the mean abs-AUC of all available features as a proxy for the model's
    ability to rank examples. Random baseline: features are random-shuffled, so
    AUC -> 0.5. Shuffled label baseline: same features but y is permuted; AUC -> 0.5.
    Date-shifted label baseline: shift the label by random offset (1 year or 6 months);
    should still degrade to ~0.5 if the label depends on contemporaneous features.
    """
    work = df.copy()
    feature_cols = [c for c in features if c in work.columns]
    rng = np.random.default_rng(42)
    out: Dict[str, Any] = {}
    for label in labels:
        if label not in work.columns:
            out[label] = {"available": False, "reason": "column missing"}
            continue
        y = pd.to_numeric(work[label], errors="coerce")
        mask = y.notna()
        yv = np.where(y[mask] > 0.5, 1, 0)
        if yv.sum() == 0 or yv.sum() == len(yv):
            out[label] = {"available": False, "reason": "label constant"}
            continue
        yv = yv.astype(int)
        # Real features vs label
        real_aucs = []
        for f in feature_cols:
            arr = _safe_numeric(work.loc[mask, f])
            if arr.std() == 0.0:
                continue
            real_aucs.append(_auc_score(arr, yv))
        # Shuffled label baseline
        shuffled = yv.copy()
        rng.shuffle(shuffled)
        shuffled_aucs = []
        for f in feature_cols:
            arr = _safe_numeric(work.loc[mask, f])
            if arr.std() == 0.0:
                continue
            shuffled_aucs.append(_auc_score(arr, shuffled))
        # Date-shifted label baseline (random 30% of rows are shifted by random offset)
        if "timestamp" in work.columns:
            ts = pd.to_datetime(work.loc[mask, "timestamp"], errors="coerce")
            order = np.argsort(ts.fillna(pd.Timestamp("2000-01-01")).to_numpy())
            shift = max(1, int(0.30 * len(order)))
            shifted = yv[order].copy()
            shifted = np.roll(shifted, shift)
            # Re-order back
            inv = np.argsort(order)
            shifted = shifted[inv]
        else:
            shifted = rng.permutation(yv)
        shifted_aucs = []
        for f in feature_cols:
            arr = _safe_numeric(work.loc[mask, f])
            if arr.std() == 0.0:
                continue
            shifted_aucs.append(_auc_score(arr, shifted))
        # Random prediction baseline (constant 0.5 AUC by construction)
        random_aucs = [0.5] * len(real_aucs)
        # Univariate best feature baseline
        best_auc = max([abs(a - 0.5) for a in real_aucs], default=0.0)
        real_mean_abs = float(np.mean([abs(a - 0.5) for a in real_aucs])) if real_aucs else 0.0
        shuf_mean_abs = float(np.mean([abs(a - 0.5) for a in shuffled_aucs])) if shuffled_aucs else 0.0
        shift_mean_abs = float(np.mean([abs(a - 0.5) for a in shifted_aucs])) if shifted_aucs else 0.0
        rand_mean_abs = 0.0
        # Lift = real_mean_abs - max(shuf_mean_abs, shift_mean_abs, rand_mean_abs)
        baseline_max = max(shuf_mean_abs, shift_mean_abs, rand_mean_abs)
        lift = real_mean_abs - baseline_max
        learnable = bool(lift > 0.02)  # require at least 2 percentage points of AUC lift
        out[label] = {
            "available": True,
            "real_mean_abs_auc_diff": real_mean_abs,
            "shuffled_label_mean_abs_auc_diff": shuf_mean_abs,
            "date_shifted_mean_abs_auc_diff": shift_mean_abs,
            "random_baseline_mean_abs_auc_diff": rand_mean_abs,
            "best_univariate_abs_auc_diff": best_auc,
            "auc_lift_over_best_baseline": lift,
            "is_learnable_with_current_features": learnable,
            "n_features": len(real_aucs),
        }
    return out


# ---------------------------------------------------------------------------
# Task 5 — Two-stage profit/avoid overlay
# ---------------------------------------------------------------------------

def two_stage_overlay(df: pd.DataFrame, *, profit_label: str = "profitable_trade_label",
                      avoid_label: str = "avoid_trade_label",
                      profit_threshold: float = 0.5,
                      avoid_threshold: float = 0.5) -> Dict[str, Any]:
    """When per-row OOF probabilities are not available, we approximate by
    thresholding the binary labels themselves (a coarser but still informative
    test of the gate logic).

    .. note::
        As of 2026-06-09, on the current binary datasets, avoid_trade_label is
        STRUCTURALLY DEFINED as ``(net_forward_return <= 0.0)`` while
        profitable_trade_label is ``(net_forward_return > 0.0)``. Therefore:
        ``avoid_trade_label = 1 - profitable_trade_label`` (trivial inverse).

        When ``avoid_trade_label_is_trivial_inverse == True``, any two-stage
        overlay that uses avoid as a veto is a **mathematical no-op** that cannot
        improve trade selection beyond what the primary profit model provides.
        The veto ``(profit >= thresh) & (avoid <= veto_thresh)`` is equivalent
        to just ``(profit >= thresh)`` alone because avoid == 1 - profit.
    """
    if profit_label not in df.columns or avoid_label not in df.columns:
        return {"available": False, "reason": "labels missing"}

    # ----------------------------------------------------------------------
    # 2026-06-09: Audit — detect whether avoid is a trivial inverse of profit.
    # avoid_trade_label = (net_forward_return <= 0.0)
    # profitable_trade_label = (net_forward_return > 0.0)
    # Therefore avoid == 1 - profit on all non-NaN rows.
    # ----------------------------------------------------------------------
    profit_series = pd.to_numeric(df[profit_label], errors="coerce")
    avoid_series = pd.to_numeric(df[avoid_label], errors="coerce")
    valid_mask = profit_series.notna() & avoid_series.notna()
    trivial_inverse = False
    if valid_mask.any():
        profit_vals = profit_series[valid_mask].to_numpy(dtype=float)
        avoid_vals = avoid_series[valid_mask].to_numpy(dtype=float)
        # Check: avoid == 1 - profit for all valid rows
        trivial_inverse = bool(np.allclose(avoid_vals, 1.0 - profit_vals, equal_nan=False))

    work = df[[profit_label, avoid_label, "net_forward_return"]].copy() if "net_forward_return" in df.columns else df[[profit_label, avoid_label]].copy()
    work[profit_label] = pd.to_numeric(work[profit_label], errors="coerce").fillna(0.0)
    work[avoid_label] = pd.to_numeric(work[avoid_label], errors="coerce").fillna(0.0)
    if "net_forward_return" in work.columns:
        work["net_forward_return"] = pd.to_numeric(work["net_forward_return"], errors="coerce").fillna(0.0)
    # Three signals: profit (1 when p_profit high), avoid (1 when p_avoid high)
    p_profit = work[profit_label]
    p_avoid = work[avoid_label]
    ret = work.get("net_forward_return", pd.Series(np.zeros(len(work))))
    combined_score_1 = p_profit - p_avoid
    combined_score_2 = p_profit * (1.0 - p_avoid)
    combined_score_3 = p_profit / p_avoid.replace(0.0, 0.05).abs().clip(lower=0.05)

    out: Dict[str, Any] = {
        "available": True,
        "veto_thresholds": [0.20, 0.30, 0.40, 0.50],
        "avoid_trade_label_is_trivial_inverse": trivial_inverse,
        # ^ Set True when avoid_trade_label == 1 - profitable_trade_label.
        #   In this case the "two-stage" overlay is a mathematical NO-OP that
        #   cannot improve selection beyond what the primary model already knows.
        #   Downstream consumers should check this flag before using avoid as a
        #   discriminative second-stage veto.
        "overlay_is_noop_when_trivial_inverse": True,  # documented here for callers
        "results": {}}
    # Profit-only baseline
    sel = work[profit_label] >= profit_threshold
    out["results"]["profit_only"] = _selection_metrics(sel, ret)
    # Profit + avoid veto at each threshold
    for veto in (0.20, 0.30, 0.40, 0.50):
        sel2 = (work[profit_label] >= profit_threshold) & (work[avoid_label] <= veto)
        out["results"][f"profit_and_avoid_veto_le_{veto}"] = _selection_metrics(sel2, ret)
    # Combined score top-K selections
    for k_name, score in (
        ("combined_score_1", combined_score_1),
        ("combined_score_2", combined_score_2),
        ("combined_score_3", combined_score_3),
    ):
        for topn in (250, 500, 1000):
            order = score.sort_values(ascending=False).head(topn).index
            mask = score.index.isin(order)
            out["results"][f"{k_name}_top_{topn}"] = _selection_metrics(mask, ret)
    return out


def _selection_metrics(mask: pd.Series, ret: pd.Series) -> Dict[str, Any]:
    sel = mask & ret.notna() & (ret != 0.0)
    if not sel.any():
        return {"trade_count": 0, "profit_factor": 0.0, "sharpe": 0.0,
                "average_return_per_trade": 0.0, "cost_1_25x_profit_factor": 0.0,
                "cost_1_50x_profit_factor": 0.0, "return_to_cost_ratio": 0.0}
    r = ret[sel].to_numpy(dtype=float)
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    pf = float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0)
    sharpe = float(r.mean() / r.std(ddof=1)) if r.size > 1 and r.std(ddof=1) > 0 else 0.0
    # Use a 1% cost stress proxy (very rough)
    cost_125 = r - 0.0125
    cost_150 = r - 0.015
    pos125 = cost_125[cost_125 > 0].sum()
    neg125 = -cost_125[cost_125 < 0].sum()
    pf125 = float(pos125 / neg125) if neg125 > 0 else (float("inf") if pos125 > 0 else 0.0)
    pos150 = cost_150[cost_150 > 0].sum()
    neg150 = -cost_150[cost_150 < 0].sum()
    pf150 = float(pos150 / neg150) if neg150 > 0 else (float("inf") if pos150 > 0 else 0.0)
    avg_cost = 0.01
    r2c = float((r.mean() - avg_cost) / avg_cost) if avg_cost > 0 else 0.0
    return {
        "trade_count": int(sel.sum()),
        "profit_factor": float(pf) if math.isfinite(pf) else 0.0,
        "sharpe": float(sharpe) if math.isfinite(sharpe) else 0.0,
        "average_return_per_trade": float(r.mean()),
        "median_return_per_trade": float(np.median(r)),
        "cost_1_25x_profit_factor": float(pf125) if math.isfinite(pf125) else 0.0,
        "cost_1_50x_profit_factor": float(pf150) if math.isfinite(pf150) else 0.0,
        "return_to_cost_ratio": r2c,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

import math


def render_separability_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Label Separability Audit")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("## Per-label summary")
    for label, p in payload.get("per_label", {}).items():
        if not p.get("available"):
            lines.append(f"- `{label}`: SKIPPED ({p.get('reason')})")
            continue
        lines.append(f"- `{label}`: pos={p['positive_count']} neg={p['negative_count']} pos_rate={p['positive_rate']:.4f}")
    lines.append("")
    lines.append("## Top 30 useful features (by mean abs AUC diff across labels)")
    for r in payload.get("top_30_useful_features", []):
        lines.append(f"- `{r['feature']}` auc_diff={r['mean_abs_auc_diff']:.4f} mi={r['mean_mutual_info']:.4f}")
    lines.append("")
    lines.append("## Bottom 30 useless features")
    for r in payload.get("bottom_30_useless_features", []):
        lines.append(f"- `{r['feature']}` auc_diff={r['mean_abs_auc_diff']:.4f} mi={r['mean_mutual_info']:.4f}")
    lines.append("")
    lines.append("## Unstable features (high month-to-month AUC variance)")
    for r in payload.get("unstable_features", []):
        lines.append(f"- `{r['feature']}` mean_auc={r['mean_auc']:.3f} std_auc={r['std_auc']:.3f} months={r['month_count']}")
    lines.append("")
    lines.append("## Potential leakage candidates (excluded by default)")
    for f in payload.get("leakage_candidates", []):
        lines.append(f"- `{f}`")
    return "\n".join(lines)


def render_decile_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Probability Decile Audit")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    if not payload.get("available"):
        lines.append(f"- Skipped: {payload.get('reason')}")
        return "\n".join(lines)
    lines.append(f"- Artifact dir: `{payload['artifact_dir']}`")
    s = payload.get("summary", {})
    lines.append(f"- Models: {s.get('n_models')}, Labels: {s.get('n_labels')}, "
                 f"monotonic: {s.get('monotonic_count')}, non-monotonic: {s.get('non_monotonic_count')}")
    lines.append("")
    lines.append(f"- {payload.get('note', '')}")
    lines.append("")
    lines.append("## Per model/label (top-decile vs bottom-decile from threshold sweep)")
    for key, v in payload.get("per_model_label", {}).items():
        lines.append(f"### {key}")
        top = v.get("top_decile") or {}
        bot = v.get("bottom_decile") or {}
        lines.append(f"- top_decile: threshold={top.get('threshold')} trades={top.get('number_of_trades')} PF={top.get('profit_factor')}")
        lines.append(f"- bottom_decile: threshold={bot.get('threshold')} trades={bot.get('number_of_trades')} PF={bot.get('profit_factor')}")
        lines.append(f"- monotonic_top_beats_bottom: `{v.get('monotonic_top_beats_bottom')}`")
    return "\n".join(lines)


def render_ablation_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Feature Ablation Retrain (no model retrain — proxy via univariate AUC/MI)")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("| feature_set | feature_count | profitable_auc_diff | strong_auc_diff | cost_survivor_auc_diff | paper_auc_diff |")
    lines.append("|---|---|---|---|---|---|")
    for fs_name, fs_data in payload.get("per_feature_set", {}).items():
        per_label = fs_data.get("per_label", {})
        def g(label, key):
            v = per_label.get(label, {})
            if not v.get("available"):
                return "n/a"
            return f"{v.get(key, 0):.3f}"
        lines.append(
            f"| {fs_name} | {fs_data.get('feature_count',0)} | "
            f"{g('profitable_trade_label','mean_abs_auc_diff')} | "
            f"{g('strong_profitable_trade_label','mean_abs_auc_diff')} | "
            f"{g('cost_survivor_label','mean_abs_auc_diff')} | "
            f"{g('paper_candidate_label','mean_abs_auc_diff')} |"
        )
    return "\n".join(lines)


def render_learnability_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Label Learnability Report")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("| label | real_auc | shuffled_auc | date_shifted_auc | random_auc | lift | learnable |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, v in payload.items():
        if not v.get("available"):
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | SKIPPED |")
            continue
        lines.append(
            f"| {label} | {v['real_mean_abs_auc_diff']:.4f} | "
            f"{v['shuffled_label_mean_abs_auc_diff']:.4f} | "
            f"{v['date_shifted_mean_abs_auc_diff']:.4f} | "
            f"{v['random_baseline_mean_abs_auc_diff']:.4f} | "
            f"{v['auc_lift_over_best_baseline']:.4f} | "
            f"{v['is_learnable_with_current_features']} |"
        )
    return "\n".join(lines)


def render_overlay_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Two-Stage Profit/Avoid Overlay")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    if not payload.get("available"):
        lines.append(f"- Skipped: {payload.get('reason')}")
        return "\n".join(lines)
    # ----------------------------------------------------------------------
    # 2026-06-09: Document the trivial-inverse finding prominently so callers
    # know the overlay is a no-op when avoid is the exact inverse of profit.
    # ----------------------------------------------------------------------
    trivial_inverse = bool(payload.get("avoid_trade_label_is_trivial_inverse", False))
    lines.append("")
    if trivial_inverse:
        lines.append("**⚠  WARNING: avoid_trade_label IS a trivial inverse of profitable_trade_label**")
        lines.append("")
        lines.append("On this dataset, the two labels are perfectly anti-correlated:")
        lines.append("```")
        lines.append("  avoid_trade_label = (net_forward_return <= 0.0)")
        lines.append("  profitable_trade_label = (net_forward_return > 0.0)")
        lines.append("  → avoid_trade_label == 1 - profitable_trade_label (trivially inverse)")
        lines.append("```")
        lines.append("")
        lines.append("**Effect on two-stage overlay:**")
        lines.append("- ``profit >= thresh  AND  avoid <= veto_thresh`` is MATHEMATICALLY")
        lines.append("  EQUIVALENT to just ``profit >= thresh`` alone.")
        lines.append("- The avoid veto provides ZERO additional discriminative filtering.")
        lines.append("- Any downstream consumer treating avoid as a second-stage model should:")
        lines.append("  1. Read ``avoid_trade_label_is_trivial_inverse`` from this report.")
        lines.append("  2. Skip the avoid model if the flag is True.")
        lines.append("  3. Not promote trivial-inverse avoid models as real edges.")
        lines.append("")
    else:
        lines.append("**✅ avoid_trade_label is NOT a trivial inverse** — a genuine avoid label")
        lines.append("with independent information may exist. The overlay has real discriminative value.")
        lines.append("")
    lines.append("| selection | trade_count | PF | Sharpe | avg_return | median_return | c1.25x PF | c1.5x PF | r2cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in payload.get("results", {}).items():
        lines.append(
            f"| {name} | {m.get('trade_count',0)} | {m.get('profit_factor',0):.3f} | {m.get('sharpe',0):.3f} | "
            f"{m.get('average_return_per_trade',0):.4f} | {m.get('median_return_per_trade',0):.4f} | "
            f"{m.get('cost_1_25x_profit_factor',0):.3f} | {m.get('cost_1_50x_profit_factor',0):.3f} | "
            f"{m.get('return_to_cost_ratio',0):.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=DATA_DIR / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv")
    parser.add_argument("--artifact-dir", type=Path,
                        default=MODELS_DIR / "core_retrain_20260606_093842")
    args = parser.parse_args(argv)
    if not args.dataset.exists():
        print(f"[diagnosis] dataset missing: {args.dataset}", file=sys.stderr)
        return 2
    print(f"[diagnosis] loading {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"[diagnosis] rows={df.shape[0]} cols={df.shape[1]}")

    labels_to_audit = [
        "profitable_trade_label", "strong_profitable_trade_label",
        "cost_survivor_label", "paper_candidate_label", "high_conviction_trade_label",
    ]
    feature_candidates = [c for c in LIVE_FEATURE_CANDIDATES if c in df.columns]
    print(f"[diagnosis] features={len(feature_candidates)}")

    timestamp = _ts()
    # Task 1
    print("[diagnosis] running label separability audit")
    separability = label_separability_audit(df, labels_to_audit, feature_candidates)
    sep_json = REPORTS_DIR / f"label_separability_audit_{timestamp}.json"
    sep_md = REPORTS_DIR / f"label_separability_audit_{timestamp}.md"
    _write_json(sep_json, separability)
    _write_text(sep_md, render_separability_md(separability))
    print(f"[diagnosis] wrote {sep_json} + {sep_md}")

    # Task 2
    print("[diagnosis] running probability decile audit")
    decile = probability_decile_audit(args.artifact_dir, labels_to_audit)
    dec_json = REPORTS_DIR / f"probability_decile_audit_{timestamp}.json"
    dec_md = REPORTS_DIR / f"probability_decile_audit_{timestamp}.md"
    _write_json(dec_json, decile)
    _write_text(dec_md, render_decile_md(decile))
    print(f"[diagnosis] wrote {dec_json} + {dec_md}")

    # Task 3
    print("[diagnosis] running feature ablation audit")
    ablation = feature_ablation_audit(df, labels_to_audit, separability)
    abl_json = REPORTS_DIR / f"feature_ablation_retrain_{timestamp}.json"
    abl_md = REPORTS_DIR / f"feature_ablation_retrain_{timestamp}.md"
    _write_json(abl_json, ablation)
    _write_text(abl_md, render_ablation_md(ablation))
    print(f"[diagnosis] wrote {abl_json} + {abl_md}")

    # Task 4
    print("[diagnosis] running label learnability")
    learnability = label_learnability(df, labels_to_audit, feature_candidates)
    learn_json = REPORTS_DIR / f"label_learnability_report_{timestamp}.json"
    learn_md = REPORTS_DIR / f"label_learnability_report_{timestamp}.md"
    _write_json(learn_json, learnability)
    _write_text(learn_md, render_learnability_md(learnability))
    print(f"[diagnosis] wrote {learn_json} + {learn_md}")

    # Task 5
    print("[diagnosis] running two-stage profit/avoid overlay")
    overlay = two_stage_overlay(df)
    ov_json = REPORTS_DIR / f"two_stage_profit_avoid_overlay_{timestamp}.json"
    ov_md = REPORTS_DIR / f"two_stage_profit_avoid_overlay_{timestamp}.md"
    _write_json(ov_json, overlay)
    _write_text(ov_md, render_overlay_md(overlay))
    print(f"[diagnosis] wrote {ov_json} + {ov_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
