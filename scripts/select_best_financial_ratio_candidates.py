#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from candidate_artifact_resolver import validate_artifact_identity
from backtest_ml_models_from_csv import (
    DEFAULT_THRESHOLD,
    _candidate_option_types,
    _candidate_threshold,
    _entry_prefilter_mask,
    _materialize_offline_model_features,
    _normalize_columns,
    _normalize_loaded_artifact,
    _passes_filters,
)


DEFAULT_DATASET = ROOT / "data" / "processed" / "historical_unified_nifty_options_single" / "nifty_option_chain_historical_cost_aware_upstox_2024_2026_single.csv"
DEFAULT_OUTPUT_CONFIG = ROOT / "config" / "paper_forward_candidates_financial_best.json"
REPORTS_DIR = ROOT / "reports"
THRESHOLD_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
META_FILES = [
    "candidate_profile.json",
    "candidate_manifest.json",
    "paper_forward_manifest.json",
    "metadata.json",
    "model_card.json",
    "manifest.json",
    "deployment_manifest.json",
    "feature_schema.json",
    "feature_order.json",
]


@dataclass
class CandidateArtifact:
    candidate_id: str
    artifact_dir: Path
    model_name: str
    preset_family: str
    side_policy: str
    metadata: Dict[str, Any]
    model_path: Optional[Path] = None
    scaler_path: Optional[Path] = None
    feature_order_path: str = ""
    filters: Dict[str, Any] | None = None
    discovered_feature_order: List[str] | None = None
    threshold_value: Optional[float] = None
    threshold_source: str = "default"


@dataclass
class CandidateEvaluation:
    candidate_id: str
    artifact_dir: str
    model_name: str
    side_profile: str
    preset_family: str
    threshold_used: float
    threshold_source: str
    required_feature_count: int
    missing_feature_count: int
    missing_features_sample: List[str]
    prediction_attempted: bool
    prediction_error: str
    raw_prediction_min: Optional[float]
    raw_prediction_max: Optional[float]
    raw_prediction_mean: Optional[float]
    confidence_min: Optional[float]
    confidence_max: Optional[float]
    confidence_mean: Optional[float]
    trade_count: int
    gross_pnl: float
    total_cost: float
    net_pnl: float
    final_equity: float
    profit_factor: float
    win_rate: float
    loss_rate: float
    average_win: float
    average_loss: float
    payoff_ratio: float
    expectancy_per_trade: float
    expectancy_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    recovery_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    average_trade_pnl: float
    median_trade_pnl: float
    pnl_std: float
    consecutive_losses_max: int
    consecutive_wins_max: int
    downside_deviation: float
    risk_reward_ratio: float
    return_to_drawdown_ratio: float
    cost_to_gross_profit_ratio: float
    trades_per_day: float
    active_days: int
    activity_label: str
    local_threshold_score: float
    financial_score: float
    preferred_candidate: bool
    fallback_candidate: bool
    rank: int
    reason_selected: str
    technically_valid: bool
    reject_reason: str
    scaler_path: str
    model_path: str
    feature_order: List[str] = field(default_factory=list)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
    return {}


def _activity_label(trade_count: int) -> str:
    if trade_count == 0:
        return "NO_ACTIVITY"
    if trade_count >= 100:
        return "HIGH_ACTIVITY"
    if trade_count >= 25:
        return "MEDIUM_ACTIVITY"
    return "LOW_ACTIVITY"


def _side_matches_cli(candidate: CandidateArtifact, side_arg: str) -> bool:
    side_arg = (side_arg or "ALL").upper()
    if side_arg in {"ALL", "BOTH"}:
        return True
    allowed = _candidate_option_types(
        {
            "candidate_id": candidate.candidate_id,
            "side_policy": candidate.side_policy,
            "side": candidate.metadata.get("side"),
            "option_type": candidate.metadata.get("option_type"),
            "filters": candidate.filters or {},
        }
    )
    if not allowed:
        return side_arg == "ALL"
    return side_arg.replace("_ONLY", "") in allowed


def _base_row(candidate: CandidateArtifact, reject_reason: str = "") -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        artifact_dir=str(candidate.artifact_dir),
        model_name=candidate.model_name,
        side_profile=candidate.side_policy,
        preset_family=candidate.preset_family,
        threshold_used=float(candidate.threshold_value if candidate.threshold_value is not None else DEFAULT_THRESHOLD),
        threshold_source=candidate.threshold_source,
        required_feature_count=0,
        missing_feature_count=0,
        missing_features_sample=[],
        prediction_attempted=False,
        prediction_error="",
        raw_prediction_min=None,
        raw_prediction_max=None,
        raw_prediction_mean=None,
        confidence_min=None,
        confidence_max=None,
        confidence_mean=None,
        trade_count=0,
        gross_pnl=0.0,
        total_cost=0.0,
        net_pnl=0.0,
        final_equity=0.0,
        profit_factor=0.0,
        win_rate=0.0,
        loss_rate=0.0,
        average_win=0.0,
        average_loss=0.0,
        payoff_ratio=0.0,
        expectancy_per_trade=0.0,
        expectancy_percent=0.0,
        max_drawdown=0.0,
        max_drawdown_percent=0.0,
        recovery_factor=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        average_trade_pnl=0.0,
        median_trade_pnl=0.0,
        pnl_std=0.0,
        consecutive_losses_max=0,
        consecutive_wins_max=0,
        downside_deviation=0.0,
        risk_reward_ratio=0.0,
        return_to_drawdown_ratio=0.0,
        cost_to_gross_profit_ratio=0.0,
        trades_per_day=0.0,
        active_days=0,
        activity_label="NO_ACTIVITY",
        local_threshold_score=0.0,
        financial_score=0.0,
        preferred_candidate=False,
        fallback_candidate=False,
        rank=0,
        reason_selected="",
        technically_valid=False,
        reject_reason=reject_reason,
        scaler_path=str(candidate.scaler_path) if candidate.scaler_path else "",
        model_path=str(candidate.model_path) if candidate.model_path else "",
        feature_order=[],
    )


def _extract_prediction_arrays(estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(X), dtype=float)
        if probs.ndim != 2 or probs.shape[0] != len(X):
            raise RuntimeError("PREDICT_PROBA_SHAPE_INVALID")
        pos_idx = min(1, probs.shape[1] - 1)
        if hasattr(estimator, "classes_"):
            classes = list(np.asarray(getattr(estimator, "classes_")).tolist())
            if 1 in classes:
                pos_idx = classes.index(1)
        conf = probs[:, pos_idx]
        return conf.copy(), conf, "predict_proba"
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)
        conf = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
        return raw, conf, "decision_function"
    if hasattr(estimator, "predict"):
        raw = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
        if np.all((0.0 <= raw) & (raw <= 1.0)):
            conf = raw.copy()
        else:
            conf = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
        return raw, conf, "predict"
    raise RuntimeError("NO_SUPPORTED_PREDICT_METHOD")


def _profit_factor(values: np.ndarray) -> float:
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_loss > 0.0:
        return gross_profit / gross_loss
    if gross_profit > 0.0:
        return float("inf")
    return 0.0


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) <= 1e-12:
        return default
    return numerator / denominator


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def _streak(values: np.ndarray, positive: bool) -> int:
    best = 0
    current = 0
    for value in values:
        match = value > 0 if positive else value < 0
        if match:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _score_component(value: float, good_floor: float, good_cap: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if good_cap <= good_floor:
        return 0.0
    clipped = min(max(value, good_floor), good_cap)
    return (clipped - good_floor) / (good_cap - good_floor)


def _activity_confidence(trade_count: int, active_days: int) -> float:
    trade_score = 1.0 - math.exp(-trade_count / 40.0) if trade_count > 0 else 0.0
    day_score = 1.0 - math.exp(-active_days / 10.0) if active_days > 0 else 0.0
    return min(1.0, 0.75 * trade_score + 0.25 * day_score)


def _local_threshold_score(metrics: Dict[str, Any]) -> float:
    pf = metrics["profit_factor"]
    if math.isinf(pf):
        pf = 3.0
    profit_factor_score = _score_component(pf, 0.8, 2.5)
    net_score = _score_component(metrics["net_pnl"], -2.0, 5.0)
    expectancy_score = _score_component(metrics["expectancy_per_trade"], -0.10, 0.25)
    drawdown_penalty = _score_component(metrics["return_to_drawdown_ratio"], 0.0, 2.5)
    sharpe_sortino_score = _score_component((metrics["sharpe_ratio"] + metrics["sortino_ratio"]) / 2.0, -0.5, 2.5)
    win_payoff = (metrics["win_rate"] * 0.5) + (_score_component(metrics["payoff_ratio"], 0.4, 2.0) * 0.5)
    cost_eff = 1.0 - _score_component(metrics["cost_to_gross_profit_ratio"], 0.0, 1.0)
    activity_score = _activity_confidence(metrics["trade_count"], metrics["active_days"])
    return float(
        20.0 * profit_factor_score
        + 15.0 * net_score
        + 15.0 * expectancy_score
        + 15.0 * drawdown_penalty
        + 15.0 * sharpe_sortino_score
        + 10.0 * win_payoff
        + 5.0 * cost_eff
        + 5.0 * activity_score
    )


def _find_candidate_dirs(artifacts_dir: Path) -> Tuple[List[Path], int]:
    if not artifacts_dir.exists():
        return [], 0
    all_dirs = [p for p in artifacts_dir.rglob("*") if p.is_dir()]
    candidate_dirs: set[Path] = set()
    for path in artifacts_dir.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith((".pkl", ".joblib")) or lower in {name.lower() for name in META_FILES}:
            candidate_dirs.add(path.parent.resolve())
    return sorted(candidate_dirs), len(all_dirs)


def _discover_model_file(folder: Path) -> Optional[Path]:
    preferred = ["model.pkl", "model.joblib", "artifact.pkl", "artifact.joblib"]
    for name in preferred:
        path = folder / name
        if path.exists() and path.is_file():
            return path.resolve()
    candidates = sorted(
        [p.resolve() for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".pkl", ".joblib"} and "scaler" not in p.name.lower()]
    )
    return candidates[0] if candidates else None


def _discover_scaler_file(folder: Path) -> Optional[Path]:
    candidates = sorted(
        [p.resolve() for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".pkl", ".joblib"} and "scaler" in p.name.lower()]
    )
    return candidates[0] if candidates else None


def _discover_candidates(artifacts_dir: Path, fallback_threshold: float) -> Tuple[List[CandidateArtifact], int]:
    folders, scanned_dir_count = _find_candidate_dirs(artifacts_dir)
    rows: List[CandidateArtifact] = []
    for folder in folders:
        merged_meta: Dict[str, Any] = {}
        for name in META_FILES:
            payload = _load_json_file(folder / name)
            if payload:
                merged_meta.setdefault("_meta_sources", []).append(name)
                merged_meta.update(payload)
        model_path = _discover_model_file(folder)
        scaler_path = _discover_scaler_file(folder)
        if not model_path and not merged_meta:
            continue
        candidate_id = str(
            merged_meta.get("candidate_id")
            or merged_meta.get("artifact_id")
            or merged_meta.get("model_id")
            or folder.name
        ).strip() or folder.name
        model_name = str(merged_meta.get("model_name") or merged_meta.get("model_family") or folder.name).strip() or "unknown"
        preset_family = str(merged_meta.get("preset_family") or merged_meta.get("preset") or "unknown").strip() or "unknown"
        side_policy = str(merged_meta.get("side_policy") or merged_meta.get("side") or "BOTH").strip() or "BOTH"
        candidate_stub = {
            "candidate_id": candidate_id,
            "model_name": model_name,
            "preset_family": preset_family,
            "side_policy": side_policy,
            "artifact_dir": str(folder),
            "threshold": merged_meta.get("threshold"),
            "selected_threshold": merged_meta.get("selected_threshold"),
            "filters": merged_meta.get("filters") or merged_meta.get("entry_filters") or {},
        }
        threshold_value = _candidate_threshold(candidate_stub, fallback_threshold, use_candidate_thresholds=True)
        threshold_source = "default"
        if any(k in merged_meta for k in ("threshold", "selected_threshold", "decision_threshold", "min_probability", "min_confidence")):
            threshold_source = "metadata"
        feature_order_path = ""
        for name in ("feature_schema.json", "feature_order.json", "candidate_profile.json", "candidate_manifest.json", "paper_forward_manifest.json"):
            p = folder / name
            if p.exists():
                feature_order_path = str(p.resolve())
                break
        rows.append(
            CandidateArtifact(
                candidate_id=candidate_id,
                artifact_dir=folder.resolve(),
                model_name=model_name,
                preset_family=preset_family,
                side_policy=side_policy,
                metadata=merged_meta,
                model_path=model_path,
                scaler_path=scaler_path,
                feature_order_path=feature_order_path,
                filters=candidate_stub.get("filters") or {},
                discovered_feature_order=[],
                threshold_value=float(threshold_value) if threshold_value is not None else None,
                threshold_source=threshold_source,
            )
        )
    return rows, scanned_dir_count


def _extract_scaler_from_loaded(loaded: Any) -> Any:
    if isinstance(loaded, dict):
        for key in ("scaler", "feature_scaler", "x_scaler", "preprocessor"):
            if key in loaded and hasattr(loaded[key], "transform"):
                return loaded[key]
    if hasattr(loaded, "transform") and not any(hasattr(loaded, attr) for attr in ("predict_proba", "decision_function", "predict")):
        return loaded
    return None


def _candidate_feature_order(candidate: CandidateArtifact, fallback_threshold: float) -> List[str]:
    if not candidate.model_path or not candidate.model_path.exists():
        return []
    try:
        loaded = joblib.load(candidate.model_path)
        normalized = _normalize_loaded_artifact(
            {
                "candidate_id": candidate.candidate_id,
                "model_name": candidate.model_name,
                "preset_family": candidate.preset_family,
                "side_policy": candidate.side_policy,
                "artifact_dir": str(candidate.artifact_dir),
                "threshold": candidate.threshold_value,
                "selected_threshold": candidate.metadata.get("selected_threshold"),
                "filters": candidate.filters or {},
            },
            candidate.model_path,
            loaded,
            fallback_threshold=fallback_threshold,
            use_candidate_thresholds=True,
        )
        return [str(col) for col in (normalized.feature_cols or []) if str(col).strip()]
    except Exception:
        return []


def _required_feature_union(candidates: Sequence[CandidateArtifact], fallback_threshold: float) -> set[str]:
    union: set[str] = set()
    for candidate in candidates:
        features = candidate.discovered_feature_order or _candidate_feature_order(candidate, fallback_threshold)
        candidate.discovered_feature_order = features
        union.update(features)
    return union


def _select_outcome_columns(df: pd.DataFrame) -> Tuple[str, str]:
    net_candidates = ["net_forward_return", "expected_return_after_cost", "net_return_after_cost", "net_pnl", "realized_return"]
    gross_candidates = ["gross_forward_return", "gross_return", "gross_pnl", "raw_forward_return"]
    net_col = next((c for c in net_candidates if c in df.columns), "")
    gross_col = next((c for c in gross_candidates if c in df.columns), "")
    if not net_col:
        raise RuntimeError("NET_OUTCOME_COLUMN_MISSING")
    if not gross_col:
        gross_col = net_col
    return gross_col, net_col


def _row_passes_filters(row: pd.Series, filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    ok, _ = _passes_filters(row, filters, log_fn=None)
    return ok


def _prepare_dataset(dataset_path: Path, required_features: Iterable[str]) -> pd.DataFrame:
    print(f"[financial-selector] loading dataset {dataset_path}")
    df = _normalize_columns(pd.read_csv(dataset_path, low_memory=False))
    _materialize_offline_model_features(df, required_features, log_fn=lambda msg: print(f"[financial-selector] {msg}"))
    return df


def _compute_trade_metrics(selected: pd.DataFrame, gross_col: str, net_col: str) -> Dict[str, Any]:
    gross = pd.to_numeric(selected[gross_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    net = pd.to_numeric(selected[net_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    trade_count = int(net.size)
    gross_profit = float(net[net > 0].sum())
    gross_loss = abs(float(net[net < 0].sum()))
    average_win = float(net[net > 0].mean()) if np.any(net > 0) else 0.0
    average_loss = abs(float(net[net < 0].mean())) if np.any(net < 0) else 0.0
    payoff_ratio = _safe_div(average_win, average_loss, float("inf") if average_win > 0 and average_loss == 0 else 0.0)
    expectancy = float(net.mean()) if trade_count else 0.0
    pnl_std = float(np.std(net, ddof=0)) if trade_count else 0.0
    downside = np.minimum(net, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if trade_count else 0.0
    sharpe = _safe_div(expectancy, pnl_std, 0.0) * math.sqrt(max(trade_count, 1)) if pnl_std > 0 else 0.0
    sortino = _safe_div(expectancy, downside_deviation, 0.0) * math.sqrt(max(trade_count, 1)) if downside_deviation > 0 else 0.0
    final_equity = float(np.cumsum(net)[-1]) if trade_count else 0.0
    max_dd = _max_drawdown(net)
    active_days = 0
    if "timestamp" in selected.columns:
        ts = pd.to_datetime(selected["timestamp"], errors="coerce")
        active_days = int(ts.dt.date.nunique()) if not ts.empty else 0
    trades_per_day = _safe_div(trade_count, active_days, float(trade_count))
    gross_positive = float(np.maximum(gross, 0.0).sum())
    total_cost = float(np.sum(gross - net))
    return {
        "trade_count": trade_count,
        "gross_pnl": float(np.sum(gross)),
        "total_cost": total_cost,
        "net_pnl": float(np.sum(net)),
        "final_equity": final_equity,
        "profit_factor": _profit_factor(net),
        "win_rate": float(np.mean(net > 0)) if trade_count else 0.0,
        "loss_rate": float(np.mean(net < 0)) if trade_count else 0.0,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": float(payoff_ratio if math.isfinite(payoff_ratio) else 3.0),
        "expectancy_per_trade": expectancy,
        "expectancy_percent": _safe_div(expectancy, abs(float(np.mean(np.abs(gross)))) if trade_count else 0.0, 0.0),
        "max_drawdown": max_dd,
        "max_drawdown_percent": _safe_div(max_dd, max(abs(final_equity), 1e-9), 0.0),
        "recovery_factor": _safe_div(final_equity, max_dd, 0.0),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": _safe_div(final_equity, max_dd, 0.0),
        "average_trade_pnl": expectancy,
        "median_trade_pnl": float(np.median(net)) if trade_count else 0.0,
        "pnl_std": pnl_std,
        "consecutive_losses_max": _streak(net, positive=False),
        "consecutive_wins_max": _streak(net, positive=True),
        "downside_deviation": downside_deviation,
        "risk_reward_ratio": float(payoff_ratio if math.isfinite(payoff_ratio) else 3.0),
        "return_to_drawdown_ratio": _safe_div(final_equity, max_dd, 0.0),
        "cost_to_gross_profit_ratio": _safe_div(total_cost, gross_positive, 0.0),
        "trades_per_day": trades_per_day,
        "active_days": active_days,
        "activity_label": _activity_label(trade_count),
    }


def _preferred_candidate(metrics: Dict[str, Any], max_drawdown_limit: Optional[float], min_financial_score: float, financial_score: float) -> bool:
    if metrics["net_pnl"] <= 0.0:
        return False
    if metrics["profit_factor"] <= 1.0 and not math.isinf(metrics["profit_factor"]):
        return False
    if metrics["expectancy_per_trade"] <= 0.0:
        return False
    if max_drawdown_limit is not None and metrics["max_drawdown"] > max_drawdown_limit:
        return False
    return financial_score >= min_financial_score


def _evaluate_candidate(
    candidate: CandidateArtifact,
    *,
    dataset: pd.DataFrame,
    gross_col: str,
    net_col: str,
    threshold_grid: Sequence[float],
    fallback_threshold: float,
    max_drawdown_limit: Optional[float],
) -> CandidateEvaluation:
    row = _base_row(candidate)
    if not candidate.model_path or not candidate.model_path.exists():
        row.reject_reason = "ARTIFACT_NOT_FOUND"
        row.prediction_error = "model artifact missing"
        return row
    try:
        loaded = joblib.load(candidate.model_path)
    except Exception as exc:
        row.reject_reason = "MODEL_LOAD_FAILED"
        row.prediction_error = f"{type(exc).__name__}: {exc}"
        return row
    try:
        normalized = _normalize_loaded_artifact(
            {
                "candidate_id": candidate.candidate_id,
                "model_name": candidate.model_name,
                "preset_family": candidate.preset_family,
                "side_policy": candidate.side_policy,
                "artifact_dir": str(candidate.artifact_dir),
                "threshold": candidate.threshold_value,
                "selected_threshold": candidate.metadata.get("selected_threshold"),
                "filters": candidate.filters or {},
            },
            candidate.model_path,
            loaded,
            fallback_threshold=fallback_threshold,
            use_candidate_thresholds=True,
        )
    except Exception as exc:
        row.reject_reason = "MODEL_LOAD_FAILED"
        row.prediction_error = f"{type(exc).__name__}: {exc}"
        return row

    feature_cols = [str(col) for col in (normalized.feature_cols or []) if str(col).strip()]
    row.feature_order = feature_cols
    row.required_feature_count = len(feature_cols)
    if not feature_cols:
        row.reject_reason = "FEATURE_ORDER_MISSING"
        row.prediction_error = "feature order missing"
        return row
    ok_identity, mismatches = validate_artifact_identity(
        {
            "candidate_id": candidate.candidate_id,
            "model_name": candidate.model_name,
            "preset_family": candidate.preset_family,
            "side_policy": candidate.side_policy,
            "threshold": candidate.threshold_value,
        },
        candidate.artifact_dir,
    )
    if not ok_identity:
        row.reject_reason = "MODEL_LOAD_FAILED"
        row.prediction_error = ",".join(mismatches)
        return row

    missing = [col for col in feature_cols if col not in dataset.columns]
    row.missing_feature_count = len(missing)
    row.missing_features_sample = missing[:10]
    if missing:
        row.reject_reason = "FEATURES_MISSING"
        row.prediction_error = f"missing feature columns: {missing[:10]}"
        return row

    subset = dataset.loc[_entry_prefilter_mask(dataset)].copy()
    option_types = normalized.option_type_filter or _candidate_option_types(
        {
            "candidate_id": candidate.candidate_id,
            "side_policy": candidate.side_policy,
            "side": candidate.metadata.get("side"),
            "option_type": candidate.metadata.get("option_type"),
            "filters": candidate.filters or {},
        }
    )
    if option_types and "option_type" in subset.columns:
        subset = subset[subset["option_type"].astype(str).str.upper().str[:2].isin(option_types)]
    if subset.empty:
        row.reject_reason = "NO_TRADES"
        row.prediction_error = "no rows after side prefilter"
        return row

    X_df = subset[feature_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(X_df.to_numpy(dtype=float)).all(axis=1)
    subset = subset.loc[finite_mask].copy()
    if subset.empty:
        row.reject_reason = "FEATURE_VECTOR_INVALID"
        row.prediction_error = "no finite feature rows available"
        return row

    X = subset[feature_cols].to_numpy(dtype=float)
    scaler = _extract_scaler_from_loaded(loaded)
    if scaler is None and candidate.scaler_path and candidate.scaler_path.exists():
        try:
            scaler = joblib.load(candidate.scaler_path)
        except Exception:
            scaler = None
    if scaler is not None and hasattr(scaler, "transform"):
        try:
            X = np.asarray(scaler.transform(X), dtype=float)
        except Exception as exc:
            row.reject_reason = "PREDICTION_FAILED"
            row.prediction_error = f"scaler transform failed: {type(exc).__name__}: {exc}"
            return row
    try:
        raw_pred, confidence, predict_method = _extract_prediction_arrays(normalized.estimator, X)
    except Exception as exc:
        row.reject_reason = "PREDICTION_FAILED"
        row.prediction_error = f"{type(exc).__name__}: {exc}"
        return row
    row.prediction_attempted = True
    if not np.isfinite(raw_pred).all() or not np.isfinite(confidence).all():
        row.reject_reason = "NAN_INF_PREDICTION"
        row.prediction_error = "NaN/inf prediction values"
        return row
    if np.allclose(raw_pred, 0.0) and np.allclose(confidence, 0.0):
        row.reject_reason = "CONSTANT_ZERO_PREDICTION"
        row.prediction_error = predict_method
        return row
    row.raw_prediction_min = float(np.min(raw_pred))
    row.raw_prediction_max = float(np.max(raw_pred))
    row.raw_prediction_mean = float(np.mean(raw_pred))
    row.confidence_min = float(np.min(confidence))
    row.confidence_max = float(np.max(confidence))
    row.confidence_mean = float(np.mean(confidence))

    thresholds: List[float]
    if candidate.threshold_source == "metadata" and candidate.threshold_value is not None:
        thresholds = [float(candidate.threshold_value)]
    else:
        thresholds = [float(t) for t in threshold_grid]
    best_metrics: Optional[Dict[str, Any]] = None
    best_threshold = thresholds[0] if thresholds else float(candidate.threshold_value or fallback_threshold)
    best_source = candidate.threshold_source if candidate.threshold_source == "metadata" else "sweep"
    for threshold in thresholds:
        selected = subset.loc[confidence >= threshold].copy()
        if selected.empty:
            continue
        selected["_confidence"] = confidence[confidence >= threshold]
        if candidate.filters:
            selected = selected[selected.apply(lambda r: _row_passes_filters(r, candidate.filters or {}), axis=1)]
        if selected.empty:
            continue
        metrics = _compute_trade_metrics(selected, gross_col, net_col)
        metrics["local_threshold_score"] = _local_threshold_score(metrics)
        if max_drawdown_limit is not None and metrics["max_drawdown"] > max_drawdown_limit:
            metrics["reject_reason"] = "EXTREME_DRAWDOWN"
        if best_metrics is None or metrics["local_threshold_score"] > best_metrics["local_threshold_score"]:
            best_metrics = metrics
            best_threshold = threshold
            best_source = candidate.threshold_source if candidate.threshold_source == "metadata" else "sweep"
    if best_metrics is None:
        row.reject_reason = "NO_TRADES"
        return row
    if best_metrics.get("reject_reason") == "EXTREME_DRAWDOWN":
        row.reject_reason = "EXTREME_DRAWDOWN"
        row.threshold_used = best_threshold
        row.threshold_source = best_source
        return row
    row.threshold_used = float(best_threshold)
    row.threshold_source = best_source
    row.technically_valid = True
    row.reject_reason = ""
    for key, value in best_metrics.items():
        if hasattr(row, key):
            setattr(row, key, value)
    return row


def _normalize_series(values: Sequence[float], invert: bool = False) -> List[float]:
    arr = np.asarray([0.0 if not math.isfinite(float(v)) else float(v) for v in values], dtype=float)
    if arr.size == 0:
        return []
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        norm = np.full(arr.shape, 1.0 if hi > 0 else 0.0, dtype=float)
    else:
        norm = (arr - lo) / (hi - lo)
    if invert:
        norm = 1.0 - norm
    return [float(x) for x in norm]


def _assign_financial_scores(rows: List[CandidateEvaluation]) -> None:
    if not rows:
        return
    pf_vals = [3.0 if math.isinf(r.profit_factor) else r.profit_factor for r in rows]
    net_vals = [r.net_pnl for r in rows]
    exp_vals = [r.expectancy_per_trade for r in rows]
    dd_vals = [r.max_drawdown for r in rows]
    sharpe_vals = [max((r.sharpe_ratio + r.sortino_ratio) / 2.0, -5.0) for r in rows]
    balance_vals = [0.5 * r.win_rate + 0.5 * min(r.payoff_ratio / 2.5, 1.0) for r in rows]
    cost_vals = [r.cost_to_gross_profit_ratio for r in rows]
    activity_vals = [_activity_confidence(r.trade_count, r.active_days) for r in rows]
    pf_n = _normalize_series(pf_vals)
    net_n = _normalize_series(net_vals)
    exp_n = _normalize_series(exp_vals)
    dd_n = _normalize_series(dd_vals, invert=True)
    sharpe_n = _normalize_series(sharpe_vals)
    balance_n = _normalize_series(balance_vals)
    cost_n = _normalize_series(cost_vals, invert=True)
    activity_n = _normalize_series(activity_vals)
    for idx, row in enumerate(rows):
        row.financial_score = float(
            20.0 * pf_n[idx]
            + 15.0 * net_n[idx]
            + 15.0 * exp_n[idx]
            + 15.0 * dd_n[idx]
            + 15.0 * sharpe_n[idx]
            + 10.0 * balance_n[idx]
            + 5.0 * cost_n[idx]
            + 5.0 * activity_n[idx]
        )


def _sort_value(row: CandidateEvaluation, sort_by: str) -> float:
    mapping = {
        "financial_score": row.financial_score,
        "net_pnl": row.net_pnl,
        "profit_factor": 3.0 if math.isinf(row.profit_factor) else row.profit_factor,
        "sharpe": row.sharpe_ratio,
        "expectancy": row.expectancy_per_trade,
    }
    return float(mapping.get(sort_by, row.financial_score))


def _summary_payload(
    *,
    scanned_dirs: int,
    candidates: Sequence[CandidateArtifact],
    valid_model_candidates: int,
    evaluated_rows: Sequence[CandidateEvaluation],
    rejected_rows: Sequence[CandidateEvaluation],
    selected_rows: Sequence[CandidateEvaluation],
    sort_by: str,
) -> Dict[str, Any]:
    ranked = sorted(evaluated_rows, key=lambda row: row.financial_score, reverse=True)
    return {
        "total_artifact_folders_scanned": scanned_dirs,
        "candidate_directories_discovered": len(candidates),
        "candidates_with_valid_model_artifacts": valid_model_candidates,
        "candidates_evaluated_successfully": len(evaluated_rows),
        "candidates_technically_rejected": len(rejected_rows),
        "candidates_financially_ranked": len(evaluated_rows),
        "selected_candidates_count": len(selected_rows),
        "preferred_profitable_candidates_count": sum(1 for row in ranked if row.preferred_candidate),
        "fallback_candidates_count": sum(1 for row in selected_rows if row.fallback_candidate),
        "sorted_by": sort_by,
        "top_10_by_financial_score": [asdict(row) for row in ranked[:10]],
        "top_10_by_net_pnl": [asdict(row) for row in sorted(ranked, key=lambda row: row.net_pnl, reverse=True)[:10]],
        "top_10_by_profit_factor": [asdict(row) for row in sorted(ranked, key=lambda row: (3.0 if math.isinf(row.profit_factor) else row.profit_factor), reverse=True)[:10]],
        "worst_10_by_drawdown": [asdict(row) for row in sorted(ranked, key=lambda row: row.max_drawdown, reverse=True)[:10]],
        "reject_reasons": dict(pd.Series([row.reject_reason for row in rejected_rows]).value_counts()) if rejected_rows else {},
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[CandidateEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(row) for row in rows]).to_csv(path, index=False)


def _selected_reason(row: CandidateEvaluation, preferred_exists: bool) -> str:
    if row.fallback_candidate:
        return "fallback_observation_only"
    if preferred_exists:
        return "passed_preferred_financial_conditions"
    return "best_available_financial_score"


def _to_config_entry(row: CandidateEvaluation, candidate: CandidateArtifact) -> Dict[str, Any]:
    payload = dict(candidate.metadata)
    payload.update(
        {
            "enabled": True,
            "candidate_id": row.candidate_id,
            "artifact_dir": row.artifact_dir,
            "artifact_path": row.artifact_dir,
            "model_path": row.model_path,
            "scaler_path": row.scaler_path,
            "feature_order": row.feature_order,
            "feature_order_source": candidate.feature_order_path,
            "threshold_used": row.threshold_used,
            "threshold_source": row.threshold_source,
            "selected_threshold": row.threshold_used,
            "side_policy": row.side_profile,
            "preset_family": row.preset_family,
            "financial_score": row.financial_score,
            "rank": row.rank,
            "fallback_candidate": row.fallback_candidate,
            "activity_label": row.activity_label,
            "reason_selected": row.reason_selected,
            "paper_forward_only": True,
            "live_ready": False,
        }
    )
    return payload


def run_selection(
    *,
    artifacts_dir: Path,
    dataset_path: Path,
    output_config: Path,
    apply: bool,
    dry_run: bool,
    top_n: int,
    min_financial_score: float,
    side: str,
    include_low_activity: bool,
    no_min_trades: bool,
    allow_small_loss_if_best: bool,
    max_drawdown_limit: Optional[float],
    sort_by: str,
    fallback_threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    del include_low_activity
    del no_min_trades
    candidates, scanned_dirs = _discover_candidates(artifacts_dir, fallback_threshold)
    candidates = [row for row in candidates if _side_matches_cli(row, side)]
    valid_model_candidates = sum(1 for row in candidates if row.model_path and row.model_path.exists())
    required_features = _required_feature_union(candidates, fallback_threshold)
    dataset = _prepare_dataset(dataset_path, required_features)
    gross_col, net_col = _select_outcome_columns(dataset)
    evaluated: List[CandidateEvaluation] = []
    rejected: List[CandidateEvaluation] = []
    candidate_map = {row.candidate_id: row for row in candidates}
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        print(f"[financial-selector] evaluating {index}/{total} candidate_id={candidate.candidate_id}")
        row = _evaluate_candidate(
            candidate,
            dataset=dataset,
            gross_col=gross_col,
            net_col=net_col,
            threshold_grid=THRESHOLD_SWEEP,
            fallback_threshold=fallback_threshold,
            max_drawdown_limit=max_drawdown_limit,
        )
        if row.technically_valid:
            evaluated.append(row)
        else:
            rejected.append(row)
    _assign_financial_scores(evaluated)
    for row in evaluated:
        row.preferred_candidate = _preferred_candidate(
            {
                "net_pnl": row.net_pnl,
                "profit_factor": row.profit_factor,
                "expectancy_per_trade": row.expectancy_per_trade,
                "max_drawdown": row.max_drawdown,
            },
            max_drawdown_limit,
            min_financial_score,
            row.financial_score,
        )
    preferred = [row for row in evaluated if row.preferred_candidate]
    preferred_sorted = sorted(preferred, key=lambda row: _sort_value(row, sort_by), reverse=True)
    pool = preferred_sorted
    fallback_warning = ""
    if not evaluated:
        fallback_warning = "WARNING: no candidates could be evaluated successfully. Selected set is empty because all discovered artifacts were technically rejected on the provided dataset."
        pool = []
    elif not pool:
        fallback_source = sorted(evaluated, key=lambda row: _sort_value(row, sort_by), reverse=True)
        if not allow_small_loss_if_best:
            fallback_source = [row for row in fallback_source if row.net_pnl >= 0.0] or fallback_source
        pool = fallback_source
        fallback_warning = "WARNING: selected candidates are fallback paper-forward observation candidates only. No strictly profitable candidate passed preferred financial conditions."
        for row in pool:
            row.fallback_candidate = True
    selected = pool[: max(top_n, 0)] if top_n > 0 else list(pool)
    preferred_exists = bool(preferred_sorted)
    for rank, row in enumerate(sorted(evaluated, key=lambda item: _sort_value(item, sort_by), reverse=True), start=1):
        row.rank = rank
    selected_ids = {row.candidate_id for row in selected}
    for row in evaluated:
        if row.candidate_id in selected_ids:
            row.reason_selected = _selected_reason(row, preferred_exists)
    selected = sorted(selected, key=lambda row: row.rank)
    if selected and not fallback_warning:
        print("Selected candidates passed preferred financial-ratio conditions.")
    elif fallback_warning:
        print(fallback_warning)

    stamp = _now_stamp()
    audit_csv = REPORTS_DIR / f"financial_ratio_candidate_audit_{stamp}.csv"
    audit_json = REPORTS_DIR / f"financial_ratio_candidate_audit_{stamp}.json"
    ranking_csv = REPORTS_DIR / f"financial_ratio_candidate_ranking_{stamp}.csv"
    ranking_json = REPORTS_DIR / f"financial_ratio_candidate_ranking_{stamp}.json"
    selected_csv = REPORTS_DIR / f"financial_ratio_selected_candidates_{stamp}.csv"
    selected_json = REPORTS_DIR / f"financial_ratio_selected_candidates_{stamp}.json"

    audit_rows = list(evaluated) + list(rejected)
    ranking_rows = sorted(evaluated, key=lambda row: _sort_value(row, sort_by), reverse=True)
    summary = _summary_payload(
        scanned_dirs=scanned_dirs,
        candidates=candidates,
        valid_model_candidates=valid_model_candidates,
        evaluated_rows=evaluated,
        rejected_rows=rejected,
        selected_rows=selected,
        sort_by=sort_by,
    )
    _write_csv(audit_csv, audit_rows)
    _write_csv(ranking_csv, ranking_rows)
    _write_csv(selected_csv, selected)
    _write_json(audit_json, {"summary": summary, "rows": [asdict(row) for row in audit_rows]})
    _write_json(ranking_json, {"summary": summary, "rows": [asdict(row) for row in ranking_rows]})
    _write_json(selected_json, {"summary": summary, "rows": [asdict(row) for row in selected]})

    config_written = False
    if apply and not dry_run:
        config_payload = {
            "mode": "paper_forward_multi",
            "paper_only": True,
            "real_trading_enabled": False,
            "broker_orders_enabled": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset_path),
            "selection_warning": fallback_warning,
            "selection_status": "preferred" if preferred_exists else "fallback",
            "selection_message": "Selected candidates passed preferred financial-ratio conditions." if preferred_exists else fallback_warning,
            "source_reports": {
                "audit_csv": str(audit_csv),
                "audit_json": str(audit_json),
                "ranking_csv": str(ranking_csv),
                "ranking_json": str(ranking_json),
                "selected_csv": str(selected_csv),
                "selected_json": str(selected_json),
            },
            "candidates": [_to_config_entry(row, candidate_map[row.candidate_id]) for row in selected],
        }
        output_config.parent.mkdir(parents=True, exist_ok=True)
        output_config.write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
        config_written = True

    return {
        "total_artifact_folders_scanned": scanned_dirs,
        "candidate_directories_discovered": len(candidates),
        "valid_model_candidates_found": valid_model_candidates,
        "evaluated_candidates_count": len(evaluated),
        "technically_rejected_count": len(rejected),
        "preferred_profitable_candidates_count": len(preferred),
        "fallback_candidates_count": sum(1 for row in selected if row.fallback_candidate),
        "selected_candidates_count": len(selected),
        "selected_candidate_ids": [row.candidate_id for row in selected],
        "selected_candidate_financial_score": [row.financial_score for row in selected],
        "selected_candidate_net_pnl": [row.net_pnl for row in selected],
        "selected_candidate_profit_factor": [row.profit_factor for row in selected],
        "selected_candidate_max_drawdown": [row.max_drawdown for row in selected],
        "selected_candidate_trade_count": [row.trade_count for row in selected],
        "output_config_path": str(output_config),
        "output_report_paths": {
            "audit_csv": str(audit_csv),
            "audit_json": str(audit_json),
            "ranking_csv": str(ranking_csv),
            "ranking_json": str(ranking_json),
            "selected_csv": str(selected_csv),
            "selected_json": str(selected_json),
        },
        "config_written": config_written,
        "selected_rows": [asdict(row) for row in selected],
        "rejected_rows": [asdict(row) for row in rejected],
        "evaluated_rows": [asdict(row) for row in ranking_rows],
        "warning": fallback_warning,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best financial-ratio candidates for paper-forward observation.")
    parser.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-config", default=str(DEFAULT_OUTPUT_CONFIG))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-financial-score", type=float, default=0.0)
    parser.add_argument("--side", default="ALL", choices=["ALL", "CE_ONLY", "PE_ONLY", "BOTH"])
    parser.add_argument("--include-low-activity", action="store_true")
    parser.add_argument("--no-min-trades", action="store_true")
    parser.add_argument("--allow-small-loss-if-best", action="store_true")
    parser.add_argument("--max-drawdown-limit", type=float, default=None)
    parser.add_argument("--sort-by", default="financial_score", choices=["financial_score", "net_pnl", "profit_factor", "sharpe", "expectancy"])
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_selection(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        dataset_path=Path(args.dataset).resolve(),
        output_config=Path(args.output_config).resolve(),
        apply=bool(args.apply),
        dry_run=bool(args.dry_run),
        top_n=int(args.top_n),
        min_financial_score=float(args.min_financial_score),
        side=str(args.side),
        include_low_activity=bool(args.include_low_activity),
        no_min_trades=bool(args.no_min_trades),
        allow_small_loss_if_best=bool(args.allow_small_loss_if_best),
        max_drawdown_limit=args.max_drawdown_limit,
        sort_by=str(args.sort_by),
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
