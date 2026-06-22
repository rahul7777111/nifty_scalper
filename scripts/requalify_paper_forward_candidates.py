#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_ml_models_from_csv import (  # noqa: E402
    ArtifactLoadFailure,
    LoadedCandidateModel,
    _candidate_option_types,
    _candidate_threshold,
    _feature_alignment_diagnostics,
    _load_candidate_model,
    _load_candidate_config,
    _materialize_offline_model_features,
    _normalize_columns,
    _passes_filters,
    _planned_csv_columns,
    _preferred_data_path,
    _read_backtest_frame,
    _read_data_header,
    _resolve_path,
    _safe_float,
    _score_loaded_candidate_frame,
)


WATCHLIST_GATES = {
    "trade_count_min": 100,
    "trade_count_max": 5000,
    "profit_factor_min": 1.20,
    "sharpe_min": 1.00,
    "cost_1_25x_pf_min": 1.05,
    "cost_1_50x_pf_min": 1.00,
    "worst_fold_pf_min": 0.90,
    "median_fold_pf_min": 1.05,
    "return_to_cost_ratio_min": 1.25,
    "profitable_days_min": 30,
    "profitable_months_min": 3,
    "profitable_folds_min": 4,
    "top_1_day_profit_share_max": 0.35,
    "top_2_days_profit_share_max": 0.50,
    "top_month_profit_share_max": 0.50,
}

LABEL_COLS = [
    "profitable_trade_label",
    "paper_candidate_label",
    "cost_survivor_label",
    "profitable_trade_label_v2",
    "paper_candidate_label_v2",
    "cost_survivor_label_v2",
]
RETURN_COLS = [
    "net_forward_return",
    "expected_return_after_cost",
    "return_to_cost_ratio",
    "cost_return_units_estimated",
    "gross_forward_return",
]
TIME_COLS = ["timestamp", "trading_day", "month"]


@dataclass
class CandidateQualification:
    candidate_id: str
    enabled_before: bool
    passed: bool
    status: str
    primary_reason: str
    failed_requirements: List[str]
    threshold: float
    row_count: int = 0
    prediction_prefilter_rows: int = 0
    threshold_signal_count: int = 0
    qualified_signal_count: int = 0
    precision: float = 0.0
    mean_net_forward_return: float = 0.0
    median_net_forward_return: float = 0.0
    mean_expected_return_after_cost: float = 0.0
    median_expected_return_after_cost: float = 0.0
    return_to_cost_ratio: float = 0.0
    median_return_to_cost_ratio: float = 0.0
    profit_factor: float = 0.0
    cost_1_25x_profit_factor: float = 0.0
    cost_1_50x_profit_factor: float = 0.0
    sharpe: float = 0.0
    profitable_days: int = 0
    profitable_months: int = 0
    profitable_folds: int = 0
    worst_fold_pf: float = 0.0
    median_fold_pf: float = 0.0
    top_1_day_profit_share: float = 0.0
    top_2_days_profit_share: float = 0.0
    top_month_profit_share: float = 0.0
    required_feature_count: int = 0
    missing_feature_count: int = 0
    artifact_path: str = ""


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _coerce_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().upper() not in {"N", "NO", "FALSE", "0", "OFF", "DISABLED"}


def _profit_factor(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return 0.0
    wins = float(vals[vals > 0.0].sum())
    losses = abs(float(vals[vals < 0.0].sum()))
    if losses <= 1e-12:
        return 999.0 if wins > 0.0 else 0.0
    return wins / losses


def _sharpe(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size < 2:
        return 0.0
    sd = float(np.std(vals, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(vals) / sd * math.sqrt(vals.size))


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else 0.0


def _safe_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if len(vals) else 0.0


def _candidate_label_col(df: pd.DataFrame) -> Optional[str]:
    for col in LABEL_COLS:
        if col in df.columns:
            return col
    return None


def _candidate_return_col(df: pd.DataFrame) -> Optional[str]:
    if "net_forward_return" in df.columns:
        return "net_forward_return"
    if "expected_return_after_cost" in df.columns:
        return "expected_return_after_cost"
    if "gross_forward_return" in df.columns:
        return "gross_forward_return"
    return None


def _market_prefilter(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "ts" in df.columns:
        local = df["ts"].dt.tz_convert("Asia/Kolkata") if getattr(df["ts"].dt, "tz", None) is not None else df["ts"]
        minutes = local.dt.hour * 60 + local.dt.minute
        midnight_share = float((minutes == 0).mean()) if len(minutes) else 0.0
        if midnight_share < 0.80:
            mask &= (minutes >= (9 * 60 + 15)) & (minutes <= (15 * 60 + 30))
    if "option_type" in df.columns:
        mask &= df["option_type"].astype(str).str.upper().str[:2].isin(["CE", "PE"])
    if "ltp" in df.columns:
        price = pd.to_numeric(df["ltp"], errors="coerce")
        mask &= price.notna() & (price > 0)
    return mask.fillna(False)


def _option_type_mask(df: pd.DataFrame, option_types: Optional[List[str]]) -> pd.Series:
    if not option_types or "option_type" not in df.columns:
        return pd.Series(True, index=df.index)
    allowed = {str(item).upper()[:2] for item in option_types if str(item).strip()}
    if not allowed:
        return pd.Series(True, index=df.index)
    return df["option_type"].astype(str).str.upper().str[:2].isin(allowed).fillna(False)


def _filter_mask(df: pd.DataFrame, signal_mask: pd.Series, filters: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, int]]:
    if not filters:
        return signal_mask.copy(), {}
    out = pd.Series(False, index=df.index)
    reasons: Dict[str, int] = {}
    for idx, row in df.loc[signal_mask].iterrows():
        ok, bad = _passes_filters(row, filters)
        if ok:
            out.loc[idx] = True
            continue
        for reason in bad or ["filter_failed"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    return out, reasons


def _daily_monthly_stability(selected: pd.DataFrame, return_col: str) -> Dict[str, float]:
    if selected.empty:
        return {
            "profitable_days": 0,
            "profitable_months": 0,
            "top_1_day_profit_share": 0.0,
            "top_2_days_profit_share": 0.0,
            "top_month_profit_share": 0.0,
        }
    returns = pd.to_numeric(selected[return_col], errors="coerce").fillna(0.0)
    work = selected.copy()
    work["_selected_return"] = returns
    if "trading_day" in work.columns:
        day_key = work["trading_day"].astype(str)
    elif "ts" in work.columns:
        day_key = work["ts"].dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
    else:
        day_key = pd.Series("unknown", index=work.index)
    month_key = pd.to_datetime(day_key, errors="coerce").dt.to_period("M").astype(str)
    daily = work.groupby(day_key)["_selected_return"].sum()
    monthly = work.groupby(month_key)["_selected_return"].sum()
    positive_daily = daily[daily > 0.0].sort_values(ascending=False)
    positive_monthly = monthly[monthly > 0.0].sort_values(ascending=False)
    total_profit = float(positive_daily.sum())
    total_month_profit = float(positive_monthly.sum())
    top1 = float(positive_daily.iloc[0] / total_profit) if total_profit > 0 and len(positive_daily) else 0.0
    top2 = float(positive_daily.iloc[:2].sum() / total_profit) if total_profit > 0 and len(positive_daily) else 0.0
    top_month = float(positive_monthly.iloc[0] / total_month_profit) if total_month_profit > 0 and len(positive_monthly) else 0.0
    return {
        "profitable_days": int((daily > 0.0).sum()),
        "profitable_months": int((monthly > 0.0).sum()),
        "top_1_day_profit_share": top1,
        "top_2_days_profit_share": top2,
        "top_month_profit_share": top_month,
    }


def _fold_stability(selected: pd.DataFrame, return_col: str, folds: int = 5) -> Dict[str, float]:
    if selected.empty:
        return {"profitable_folds": 0, "worst_fold_pf": 0.0, "median_fold_pf": 0.0}
    ordered = selected.sort_values("ts") if "ts" in selected.columns else selected
    chunks = [ordered.iloc[idx] for idx in np.array_split(np.arange(len(ordered)), folds)]
    fold_pfs: List[float] = []
    profitable_folds = 0
    for chunk in chunks:
        if chunk.empty:
            fold_pfs.append(0.0)
            continue
        vals = pd.to_numeric(chunk[return_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pf = _profit_factor(vals)
        fold_pfs.append(float(pf))
        if float(vals.sum()) > 0.0:
            profitable_folds += 1
    return {
        "profitable_folds": int(profitable_folds),
        "worst_fold_pf": float(min(fold_pfs)) if fold_pfs else 0.0,
        "median_fold_pf": float(np.median(fold_pfs)) if fold_pfs else 0.0,
    }


def _gate_result(metrics: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    failed: List[str] = []
    if int(metrics.get("qualified_signal_count") or 0) < WATCHLIST_GATES["trade_count_min"]:
        failed.append("trade_count_below_watchlist_gate")
    if int(metrics.get("qualified_signal_count") or 0) > WATCHLIST_GATES["trade_count_max"]:
        failed.append("trade_count_above_watchlist_gate")
    if float(metrics.get("profit_factor") or 0.0) <= WATCHLIST_GATES["profit_factor_min"]:
        failed.append("profit_factor_below_watchlist_gate")
    if float(metrics.get("sharpe") or 0.0) <= WATCHLIST_GATES["sharpe_min"]:
        failed.append("sharpe_below_watchlist_gate")
    if float(metrics.get("cost_1_25x_profit_factor") or 0.0) <= WATCHLIST_GATES["cost_1_25x_pf_min"]:
        failed.append("cost_1_25x_pf_below_watchlist_gate")
    if float(metrics.get("cost_1_50x_profit_factor") or 0.0) < WATCHLIST_GATES["cost_1_50x_pf_min"]:
        failed.append("cost_1_50x_pf_below_watchlist_gate")
    if float(metrics.get("mean_expected_return_after_cost") or 0.0) <= 0.0:
        failed.append("expected_return_after_cost_non_positive")
    if float(metrics.get("median_expected_return_after_cost") or 0.0) <= 0.0:
        failed.append("median_return_after_cost_non_positive")
    if float(metrics.get("return_to_cost_ratio") or 0.0) < WATCHLIST_GATES["return_to_cost_ratio_min"]:
        failed.append("return_to_cost_ratio_below_watchlist_gate")
    if float(metrics.get("worst_fold_pf") or 0.0) <= WATCHLIST_GATES["worst_fold_pf_min"]:
        failed.append("worst_fold_pf_below_watchlist_gate")
    if float(metrics.get("median_fold_pf") or 0.0) <= WATCHLIST_GATES["median_fold_pf_min"]:
        failed.append("median_fold_pf_below_watchlist_gate")
    if int(metrics.get("profitable_days") or 0) < WATCHLIST_GATES["profitable_days_min"]:
        failed.append("profitable_days_below_30")
    if int(metrics.get("profitable_months") or 0) < WATCHLIST_GATES["profitable_months_min"]:
        failed.append("profitable_months_below_3")
    if int(metrics.get("profitable_folds") or 0) < WATCHLIST_GATES["profitable_folds_min"]:
        failed.append("profitable_folds_below_4_of_5")
    if float(metrics.get("top_1_day_profit_share") or 0.0) > WATCHLIST_GATES["top_1_day_profit_share_max"]:
        failed.append("top_1_day_profit_share_above_35pct")
    if float(metrics.get("top_2_days_profit_share") or 0.0) > WATCHLIST_GATES["top_2_days_profit_share_max"]:
        failed.append("top_2_days_profit_share_above_50pct")
    if float(metrics.get("top_month_profit_share") or 0.0) > WATCHLIST_GATES["top_month_profit_share_max"]:
        failed.append("best_month_profit_share_above_50pct")
    if not failed:
        return True, "PASS", []
    if "trade_count_below_watchlist_gate" in failed:
        status = "BLOCKED_SMALL_SAMPLE"
    elif "trade_count_above_watchlist_gate" in failed:
        status = "BLOCKED_TOO_MANY_TRADES"
    elif any(item in failed for item in ("cost_1_25x_pf_below_watchlist_gate", "cost_1_50x_pf_below_watchlist_gate")):
        status = "BLOCKED_COST_STRESS"
    elif any(item in failed for item in ("expected_return_after_cost_non_positive", "median_return_after_cost_non_positive", "return_to_cost_ratio_below_watchlist_gate")):
        status = "BLOCKED_THIN_EDGE"
    elif any(item in failed for item in ("worst_fold_pf_below_watchlist_gate", "median_fold_pf_below_watchlist_gate", "profitable_folds_below_4_of_5")):
        status = "BLOCKED_FOLD_INSTABILITY"
    elif any("profit_share" in item or "profitable_days" in item or "profitable_months" in item for item in failed):
        status = "BLOCKED_CONCENTRATED_PROFIT"
    else:
        status = "BLOCKED_LOW_EDGE"
    return False, status, failed


def _load_models(candidates: List[Dict[str, Any]], config_path: Path, fallback_threshold: float) -> Tuple[List[LoadedCandidateModel], List[ArtifactLoadFailure]]:
    loaded: List[LoadedCandidateModel] = []
    failures: List[ArtifactLoadFailure] = []
    config_dir = config_path.parent
    for cand in candidates:
        model, model_failures = _load_candidate_model(
            cand,
            root=REPO_ROOT,
            config_dir=config_dir,
            fallback_threshold=fallback_threshold,
            use_candidate_thresholds=True,
            strict_artifacts=True,
            allow_artifact_fallback=False,
            debug=False,
        )
        failures.extend(model_failures)
        if model is not None:
            loaded.append(model)
    return loaded, failures


def _read_dataset(dataset_path: Path, models: List[LoadedCandidateModel], max_rows: Optional[int]) -> pd.DataFrame:
    data_path = _preferred_data_path(dataset_path)
    header_cols = _read_data_header(data_path)
    wanted = _planned_csv_columns(header_cols, models, include_score_columns=False)
    wanted.extend(col for col in [*LABEL_COLS, *RETURN_COLS, *TIME_COLS] if col in header_cols)
    wanted = list(dict.fromkeys(wanted))
    df = _read_backtest_frame(data_path, usecols=wanted)
    if max_rows and max_rows > 0:
        df = df.head(int(max_rows)).copy()
    df = _normalize_columns(df)
    if "timestamp" not in df.columns:
        raise RuntimeError("Historical dataset has no timestamp-compatible column.")
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[df["ts"].notna()].sort_values("ts").reset_index(drop=True)
    if "trading_day" not in df.columns:
        df["trading_day"] = df["ts"].dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
    _materialize_offline_model_features(df, sorted({f for model in models for f in model.feature_cols}))
    return df


def _qualify_model(
    model: LoadedCandidateModel,
    df: pd.DataFrame,
    prefilter: pd.Series,
    enabled_before: bool,
) -> CandidateQualification:
    candidate_id = model.candidate_id
    diag = _feature_alignment_diagnostics(model, df)
    required = int(diag.get("required_feature_count") or 0)
    missing = int(diag.get("missing_feature_count") or 0)
    if required <= 0:
        return CandidateQualification(
            candidate_id=candidate_id,
            enabled_before=enabled_before,
            passed=False,
            status="BLOCKED_ARTIFACT",
            primary_reason="MODEL_FEATURES_MISSING",
            failed_requirements=["MODEL_FEATURES_MISSING"],
            threshold=float(model.threshold),
            row_count=len(df),
            prediction_prefilter_rows=int(prefilter.sum()),
            required_feature_count=required,
            missing_feature_count=missing,
            artifact_path=str(model.artifact_path),
        )
    if missing > 3 and missing / max(required, 1) > 0.05:
        reason = f"too_many_missing_features:{missing}/{required}"
        return CandidateQualification(
            candidate_id=candidate_id,
            enabled_before=enabled_before,
            passed=False,
            status="BLOCKED_FEATURE_ALIGNMENT",
            primary_reason=reason,
            failed_requirements=[reason],
            threshold=float(model.threshold),
            row_count=len(df),
            prediction_prefilter_rows=int(prefilter.sum()),
            required_feature_count=required,
            missing_feature_count=missing,
            artifact_path=str(model.artifact_path),
        )
    score_col = _score_loaded_candidate_frame(model, df, row_mask=prefilter)
    if not score_col or score_col not in df.columns:
        reason = model.rejected_reason or "score_failed"
        return CandidateQualification(
            candidate_id=candidate_id,
            enabled_before=enabled_before,
            passed=False,
            status="BLOCKED_SCORING",
            primary_reason=reason,
            failed_requirements=[reason],
            threshold=float(model.threshold),
            row_count=len(df),
            prediction_prefilter_rows=int(prefilter.sum()),
            required_feature_count=required,
            missing_feature_count=missing,
            artifact_path=str(model.artifact_path),
        )

    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    threshold_mask = prefilter & (scores >= float(model.threshold))
    option_mask = _option_type_mask(df, model.option_type_filter)
    signal_mask = threshold_mask & option_mask
    qualified_mask, _ = _filter_mask(df, signal_mask, model.filters or {})
    selected = df.loc[qualified_mask].copy()
    return_col = _candidate_return_col(df)
    label_col = _candidate_label_col(df)
    if not return_col:
        return CandidateQualification(
            candidate_id=candidate_id,
            enabled_before=enabled_before,
            passed=False,
            status="BLOCKED_DATASET",
            primary_reason="missing_forward_return_column",
            failed_requirements=["missing_forward_return_column"],
            threshold=float(model.threshold),
            row_count=len(df),
            prediction_prefilter_rows=int(prefilter.sum()),
            threshold_signal_count=int(threshold_mask.sum()),
            qualified_signal_count=int(qualified_mask.sum()),
            required_feature_count=required,
            missing_feature_count=missing,
            artifact_path=str(model.artifact_path),
        )

    returns = pd.to_numeric(selected[return_col], errors="coerce").fillna(0.0) if not selected.empty else pd.Series(dtype=float)
    expected_after = (
        pd.to_numeric(selected["expected_return_after_cost"], errors="coerce").fillna(0.0)
        if "expected_return_after_cost" in selected.columns
        else returns
    )
    cost_units = (
        pd.to_numeric(selected["cost_return_units_estimated"], errors="coerce").fillna(0.0)
        if "cost_return_units_estimated" in selected.columns
        else pd.Series(0.0, index=selected.index)
    )
    ratio = (
        pd.to_numeric(selected["return_to_cost_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "return_to_cost_ratio" in selected.columns
        else pd.Series(dtype=float)
    )
    precision = 0.0
    if label_col and not selected.empty:
        labels = pd.to_numeric(selected[label_col], errors="coerce").fillna(0.0)
        precision = float((labels > 0.0).mean()) if len(labels) else 0.0

    stability = _daily_monthly_stability(selected, return_col)
    folds = _fold_stability(selected, return_col)
    metrics: Dict[str, Any] = {
        "qualified_signal_count": int(qualified_mask.sum()),
        "profit_factor": _profit_factor(returns.to_numpy(dtype=float) if len(returns) else []),
        "cost_1_25x_profit_factor": _profit_factor((returns - 0.25 * cost_units).to_numpy(dtype=float) if len(returns) else []),
        "cost_1_50x_profit_factor": _profit_factor((returns - 0.50 * cost_units).to_numpy(dtype=float) if len(returns) else []),
        "sharpe": _sharpe(returns.to_numpy(dtype=float) if len(returns) else []),
        "mean_expected_return_after_cost": _safe_mean(expected_after),
        "median_expected_return_after_cost": _safe_median(expected_after),
        "return_to_cost_ratio": _safe_mean(ratio) if len(ratio) else 0.0,
        **stability,
        **folds,
    }
    passed, status, failed = _gate_result(metrics)
    primary = failed[0] if failed else "PASS"
    return CandidateQualification(
        candidate_id=candidate_id,
        enabled_before=enabled_before,
        passed=passed,
        status=status,
        primary_reason=primary,
        failed_requirements=failed,
        threshold=float(model.threshold),
        row_count=len(df),
        prediction_prefilter_rows=int(prefilter.sum()),
        threshold_signal_count=int(threshold_mask.sum()),
        qualified_signal_count=int(qualified_mask.sum()),
        precision=precision,
        mean_net_forward_return=_safe_mean(returns),
        median_net_forward_return=_safe_median(returns),
        mean_expected_return_after_cost=float(metrics["mean_expected_return_after_cost"]),
        median_expected_return_after_cost=float(metrics["median_expected_return_after_cost"]),
        return_to_cost_ratio=float(metrics["return_to_cost_ratio"]),
        median_return_to_cost_ratio=_safe_median(ratio) if len(ratio) else 0.0,
        profit_factor=float(metrics["profit_factor"]),
        cost_1_25x_profit_factor=float(metrics["cost_1_25x_profit_factor"]),
        cost_1_50x_profit_factor=float(metrics["cost_1_50x_profit_factor"]),
        sharpe=float(metrics["sharpe"]),
        profitable_days=int(stability["profitable_days"]),
        profitable_months=int(stability["profitable_months"]),
        profitable_folds=int(folds["profitable_folds"]),
        worst_fold_pf=float(folds["worst_fold_pf"]),
        median_fold_pf=float(folds["median_fold_pf"]),
        top_1_day_profit_share=float(stability["top_1_day_profit_share"]),
        top_2_days_profit_share=float(stability["top_2_days_profit_share"]),
        top_month_profit_share=float(stability["top_month_profit_share"]),
        required_feature_count=required,
        missing_feature_count=missing,
        artifact_path=str(model.artifact_path),
    )


def _failure_qualification(cand: Dict[str, Any], failure: ArtifactLoadFailure) -> CandidateQualification:
    cid = str(cand.get("candidate_id") or failure.candidate_id)
    return CandidateQualification(
        candidate_id=cid,
        enabled_before=_coerce_enabled(cand.get("enabled", True)),
        passed=False,
        status="BLOCKED_ARTIFACT",
        primary_reason=failure.reason or "artifact_load_failed",
        failed_requirements=[failure.reason or "artifact_load_failed"],
        threshold=_candidate_threshold(cand, 0.60),
        artifact_path=failure.path,
    )


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload["candidate_results"]
    lines = [
        "# Paper Forward Candidate Requalification",
        "",
        f"- Dataset: `{payload['dataset_path']}`",
        f"- Config: `{payload['config_path']}`",
        f"- Generated: `{payload['generated_at']}`",
        f"- Passed: `{payload['passed_count']}/{payload['candidate_count']}`",
        "",
        "## Minimum Requirements",
        "",
    ]
    for key, value in payload["minimum_requirements"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Candidate | Status | Enabled | Signals | PF | Sharpe | Mean after cost | Reason |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        enabled = "Y" if row["passed"] else "N"
        lines.append(
            "| {candidate_id} | {status} | {enabled} | {signals} | {pf:.3f} | {sharpe:.3f} | {edge:.5f} | {reason} |".format(
                candidate_id=row["candidate_id"],
                status=row["status"],
                enabled=enabled,
                signals=int(row.get("qualified_signal_count") or 0),
                pf=float(row.get("profit_factor") or 0.0),
                sharpe=float(row.get("sharpe") or 0.0),
                edge=float(row.get("mean_expected_return_after_cost") or 0.0),
                reason=row.get("primary_reason") or "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def requalify(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = _resolve_path(args.config, root=REPO_ROOT) or (REPO_ROOT / args.config)
    dataset_path = _resolve_path(args.dataset, root=REPO_ROOT) or Path(args.dataset)
    if not config_path.exists():
        raise FileNotFoundError(f"Candidate config not found: {config_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Historical dataset not found: {dataset_path}")

    cfg = _load_candidate_config(str(config_path))
    candidates = list(cfg.get("candidates") or [])
    candidate_by_id = {str(c.get("candidate_id") or ""): c for c in candidates}
    models, failures = _load_models(candidates, config_path, fallback_threshold=float(args.fallback_threshold))
    model_by_id = {m.candidate_id: m for m in models}
    results: List[CandidateQualification] = []
    for failure in failures:
        if failure.candidate_id not in model_by_id:
            cand = candidate_by_id.get(failure.candidate_id, {"candidate_id": failure.candidate_id})
            results.append(_failure_qualification(cand, failure))

    df = _read_dataset(dataset_path, models, int(args.max_rows or 0) or None)
    prefilter = _market_prefilter(df)
    for model in models:
        cand = candidate_by_id.get(model.candidate_id, {})
        results.append(_qualify_model(model, df, prefilter, _coerce_enabled(cand.get("enabled", True))))

    result_by_id = {row.candidate_id: row for row in results}
    generated_at = datetime.now(timezone.utc).isoformat()
    for cand in candidates:
        cid = str(cand.get("candidate_id") or "")
        row = result_by_id.get(cid)
        if row is None:
            cand["enabled"] = False
            cand["disabled_reason"] = "not_tested_no_loadable_model"
            continue
        cand["enabled"] = bool(row.passed)
        cand["paper_forward_only"] = True
        cand["paper_forward_eligibility"] = "PASS" if row.passed else "FAIL"
        cand["disabled_reason"] = "" if row.passed else row.primary_reason
        cand["qualification_status"] = row.status
        cand["qualification_dataset"] = str(dataset_path)
        cand["qualification_timestamp"] = generated_at
        cand["minimum_requirements_passed"] = bool(row.passed)
        cand["historical_qualification_metrics"] = asdict(row)

    passed_count = sum(1 for row in results if row.passed)
    cfg["paper_forward_candidate_requalification"] = {
        "timestamp": generated_at,
        "dataset": str(dataset_path),
        "candidate_count": len(candidates),
        "tested_count": len(results),
        "passed_count": passed_count,
        "enabled_count_after": passed_count,
        "minimum_requirements": WATCHLIST_GATES,
        "max_rows": int(args.max_rows or 0) or None,
    }
    payload = {
        "generated_at": generated_at,
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "candidate_count": len(candidates),
        "tested_count": len(results),
        "passed_count": passed_count,
        "minimum_requirements": WATCHLIST_GATES,
        "candidate_results": [asdict(row) for row in sorted(results, key=lambda r: (not r.passed, r.status, r.candidate_id))],
    }

    out_dir = _resolve_path(args.output_dir, root=REPO_ROOT) or (REPO_ROOT / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    report_json = out_dir / f"paper_forward_candidate_requalification_{stamp}.json"
    report_md = out_dir / f"paper_forward_candidate_requalification_{stamp}.md"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(report_md, payload)

    if not args.dry_run:
        backup_path = config_path.with_name(f"{config_path.stem}.backup_requalify_{stamp}{config_path.suffix}")
        shutil.copy2(config_path, backup_path)
        cfg["paper_forward_candidate_requalification"]["backup"] = str(backup_path)
        cfg["paper_forward_candidate_requalification"]["report_json"] = str(report_json)
        cfg["paper_forward_candidate_requalification"]["report_md"] = str(report_md)
        config_path.write_text(json.dumps(cfg, indent=2, default=_json_default), encoding="utf-8")
        payload["backup_path"] = str(backup_path)
    payload["report_json"] = str(report_json)
    payload["report_md"] = str(report_md)
    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-test paper-forward candidates against historical data and update eligibility.")
    parser.add_argument("--dataset", required=True, help="Historical option-chain CSV/Parquet dataset to test against.")
    parser.add_argument("--config", default="config/paper_forward_candidates.json", help="paper_forward_candidates.json path.")
    parser.add_argument("--output-dir", default="reports/candidate_requalification", help="Directory for audit reports.")
    parser.add_argument("--fallback-threshold", type=float, default=0.60, help="Fallback confidence threshold if a candidate has none.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional debug limit; 0 means full dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Write reports but do not update the candidate config.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    payload = requalify(args)
    print(
        "Requalified {passed}/{total} paper-forward candidates. report={report}".format(
            passed=payload["passed_count"],
            total=payload["candidate_count"],
            report=payload["report_json"],
        )
    )
    if payload.get("backup_path"):
        print(f"Backup: {payload['backup_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
