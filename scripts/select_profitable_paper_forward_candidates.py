#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


PREFERRED_DATASET = ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
DEFAULT_OUTPUT_CONFIG = ROOT / "config" / "paper_forward_candidates_eligible.json"
REPORTS_DIR = ROOT / "reports"


@dataclass
class CandidateArtifact:
    candidate_id: str
    artifact_dir: Path
    model_name: str
    preset_family: str
    side_policy: str
    threshold_value: float
    threshold_source: str
    metadata: Dict[str, Any]
    model_path: Optional[Path] = None
    feature_order_path: str = ""
    filters: Dict[str, Any] | None = None
    discovered_feature_order: List[str] | None = None


@dataclass
class CandidateAuditRow:
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
    profit_factor: float
    win_rate: float
    max_drawdown: float
    final_equity: float
    eligible_paper_forward: bool
    activity_label: str
    reject_reason: str


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _find_dataset_path(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"dataset not found: {path}")
    if PREFERRED_DATASET.exists():
        return PREFERRED_DATASET.resolve()
    processed_dir = ROOT / "data" / "processed"
    candidates = sorted(
        processed_dir.glob("nifty_option_chain_cost_aware_edge_dataset_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no suitable dataset found under {processed_dir}")
    return candidates[0].resolve()


def _dataset_candidates() -> List[Path]:
    processed_dir = ROOT / "data" / "processed"
    paths = sorted(
        processed_dir.glob("nifty_option_chain_cost_aware_edge_dataset_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if PREFERRED_DATASET.exists():
        preferred = PREFERRED_DATASET.resolve()
        paths = [preferred] + [p.resolve() for p in paths if p.resolve() != preferred]
    else:
        paths = [p.resolve() for p in paths]
    return paths


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
    return {}


def _discover_candidate_artifacts(artifacts_dir: Path, fallback_threshold: float) -> List[CandidateArtifact]:
    rows: List[CandidateArtifact] = []
    if not artifacts_dir.exists():
        return rows
    for folder in sorted(artifacts_dir.iterdir()):
        if not folder.is_dir():
            continue
        merged_meta: Dict[str, Any] = {}
        meta_files = [
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
        for name in meta_files:
            payload = _load_json_file(folder / name)
            if payload:
                merged_meta.setdefault("_meta_sources", []).append(name)
                merged_meta.update(payload)
        candidate_id = str(
            merged_meta.get("candidate_id")
            or merged_meta.get("artifact_id")
            or merged_meta.get("model_id")
            or folder.name
        ).strip() or folder.name
        model_name = str(merged_meta.get("model_name") or merged_meta.get("model_family") or "").strip()
        preset_family = str(merged_meta.get("preset_family") or merged_meta.get("preset") or "").strip()
        side_policy = str(merged_meta.get("side_policy") or merged_meta.get("side") or "").strip()
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
        threshold_source = "default"
        threshold_value = _candidate_threshold(candidate_stub, fallback_threshold, use_candidate_thresholds=True)
        if any(key in merged_meta for key in ("threshold", "selected_threshold", "decision_threshold", "min_probability", "min_confidence")):
            threshold_source = "metadata"
        elif re.search(r"(?:^|_)t\d{2,3}(?:_|$)", candidate_id):
            threshold_source = "metadata"
        model_path = folder / "model.pkl"
        if not model_path.exists():
            bundles = sorted([p for p in folder.glob("*.pkl") if p.is_file()])
            model_path = bundles[0] if bundles else None
        feature_order_path = ""
        for name in ("feature_schema.json", "feature_order.json", "candidate_profile.json", "candidate_manifest.json", "paper_forward_manifest.json"):
            if (folder / name).exists():
                feature_order_path = str((folder / name).resolve())
                break
        rows.append(
            CandidateArtifact(
                candidate_id=candidate_id,
                artifact_dir=folder.resolve(),
                model_name=model_name or "unknown",
                preset_family=preset_family or "unknown",
                side_policy=side_policy or "BOTH",
                threshold_value=float(threshold_value),
                threshold_source=threshold_source,
                metadata=merged_meta,
                model_path=model_path.resolve() if model_path else None,
                feature_order_path=feature_order_path,
                filters=candidate_stub.get("filters") or {},
                discovered_feature_order=[],
            )
        )
    return rows


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


def _candidate_required_feature_union(
    candidate_artifacts: List[CandidateArtifact],
    fallback_threshold: float,
) -> set[str]:
    feature_union: set[str] = set()
    for candidate in candidate_artifacts:
        features = candidate.discovered_feature_order or _candidate_feature_order(candidate, fallback_threshold)
        candidate.discovered_feature_order = features
        feature_union.update(str(col) for col in features if str(col).strip())
    return feature_union


def _choose_best_dataset(candidate_artifacts: List[CandidateArtifact], fallback_threshold: float) -> Path:
    candidates = _dataset_candidates()
    if not candidates:
        raise FileNotFoundError("no candidate datasets found under data/processed")
    feature_union = _candidate_required_feature_union(candidate_artifacts, fallback_threshold)
    if not feature_union:
        return candidates[0]

    ranked: List[Tuple[int, int, float, float, Path]] = []
    for path in candidates:
        try:
            header = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        header_set = set(header)
        matched = len(feature_union & header_set)
        missing = len(feature_union - header_set)
        coverage = matched / max(len(feature_union), 1)
        ranked.append((matched, -missing, coverage, path.stat().st_mtime, path))
    if not ranked:
        return candidates[0]
    ranked.sort(reverse=True)
    return ranked[0][-1]


def _activity_label(trade_count: int) -> str:
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


def _base_audit_row(candidate: CandidateArtifact, *, reject_reason: str = "") -> CandidateAuditRow:
    return CandidateAuditRow(
        candidate_id=candidate.candidate_id,
        artifact_dir=str(candidate.artifact_dir),
        model_name=candidate.model_name,
        side_profile=candidate.side_policy or "BOTH",
        preset_family=candidate.preset_family or "unknown",
        threshold_used=float(candidate.threshold_value),
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
        profit_factor=0.0,
        win_rate=0.0,
        max_drawdown=0.0,
        final_equity=0.0,
        eligible_paper_forward=False,
        activity_label="LOW_ACTIVITY",
        reject_reason=reject_reason,
    )


def _extract_prediction_arrays(estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(X), dtype=float)
        if probs.ndim != 2 or probs.shape[0] != len(X):
            raise RuntimeError("PREDICT_PROBA_SHAPE_INVALID")
        if hasattr(estimator, "classes_"):
            classes = list(np.asarray(getattr(estimator, "classes_")).tolist())
            if 1 in classes:
                pos_idx = classes.index(1)
            else:
                pos_idx = min(1, probs.shape[1] - 1)
        else:
            pos_idx = min(1, probs.shape[1] - 1)
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


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def _select_outcome_columns(df: pd.DataFrame) -> Tuple[str, str]:
    net_candidates = ["net_forward_return", "expected_return_after_cost", "net_pnl", "realized_return"]
    gross_candidates = ["gross_forward_return", "gross_pnl", "raw_forward_return"]
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


def evaluate_candidate(
    candidate: CandidateArtifact,
    *,
    dataset: pd.DataFrame,
    gross_col: str,
    net_col: str,
    fallback_threshold: float,
    include_low_activity: bool,
) -> CandidateAuditRow:
    audit = _base_audit_row(candidate)
    if not candidate.model_path or not candidate.model_path.exists():
        audit.reject_reason = "ARTIFACT_LOAD_FAILED"
        audit.prediction_error = "model artifact missing"
        return audit
    if candidate.threshold_source == "default":
        audit.threshold_used = float(fallback_threshold)
        audit.threshold_source = "default"
    try:
        loaded = joblib.load(candidate.model_path)
    except Exception as exc:
        audit.reject_reason = "ARTIFACT_LOAD_FAILED"
        audit.prediction_error = f"{type(exc).__name__}: {exc}"
        return audit
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
        audit.reject_reason = "ARTIFACT_LOAD_FAILED"
        audit.prediction_error = f"{type(exc).__name__}: {exc}"
        return audit

    feature_cols = [str(col) for col in (normalized.feature_cols or []) if str(col).strip()]
    audit.required_feature_count = len(feature_cols)
    if not feature_cols:
        audit.reject_reason = "FEATURE_ORDER_MISSING"
        audit.prediction_error = "feature order missing"
        return audit
    if candidate.metadata:
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
            audit.reject_reason = "ARTIFACT_IDENTITY_MISMATCH"
            audit.prediction_error = ",".join(mismatches)
            return audit

    missing_features = [col for col in feature_cols if col not in dataset.columns]
    audit.missing_feature_count = len(missing_features)
    audit.missing_features_sample = missing_features[:10]
    if missing_features:
        audit.reject_reason = "FEATURES_MISSING"
        audit.prediction_error = f"missing feature columns: {missing_features[:10]}"
        return audit

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
        audit.reject_reason = "NO_TRADES"
        audit.prediction_error = "no rows after market/side prefilter"
        return audit

    X_df = subset[feature_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(X_df.to_numpy(dtype=float)).all(axis=1)
    subset = subset.loc[finite_mask].copy()
    if subset.empty:
        audit.reject_reason = "FEATURE_VECTOR_INVALID"
        audit.prediction_error = "no finite feature rows available"
        return audit
    X = subset[feature_cols].to_numpy(dtype=float)

    try:
        raw_pred, confidence, predict_method = _extract_prediction_arrays(normalized.estimator, X)
    except Exception as exc:
        audit.reject_reason = "PREDICTION_ERROR"
        audit.prediction_error = f"{type(exc).__name__}: {exc}"
        return audit

    audit.prediction_attempted = True
    if not np.isfinite(raw_pred).all() or not np.isfinite(confidence).all():
        audit.reject_reason = "NON_FINITE_PREDICTION"
        audit.prediction_error = "NaN/inf prediction values"
        return audit
    if np.allclose(raw_pred, 0.0) and np.allclose(confidence, 0.0):
        audit.reject_reason = "CONSTANT_ZERO_PREDICTION"
        audit.prediction_error = predict_method
        return audit

    audit.raw_prediction_min = float(np.min(raw_pred))
    audit.raw_prediction_max = float(np.max(raw_pred))
    audit.raw_prediction_mean = float(np.mean(raw_pred))
    audit.confidence_min = float(np.min(confidence))
    audit.confidence_max = float(np.max(confidence))
    audit.confidence_mean = float(np.mean(confidence))

    threshold = float(normalized.threshold if normalized.threshold is not None else fallback_threshold)
    audit.threshold_used = threshold
    selected = subset.loc[confidence >= threshold].copy()
    selected["_confidence"] = confidence[confidence >= threshold]
    if candidate.filters:
        selected = selected[selected.apply(lambda row: _row_passes_filters(row, candidate.filters or {}), axis=1)]
    if selected.empty:
        audit.reject_reason = "NO_TRADES"
        return audit

    gross = pd.to_numeric(selected[gross_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    net = pd.to_numeric(selected[net_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(gross).all() or not np.isfinite(net).all():
        audit.reject_reason = "NON_FINITE_PNL"
        audit.prediction_error = "non-finite gross/net outcome values"
        return audit
    costs = gross - net
    pf = _profit_factor(net)
    final_equity = float(np.cumsum(net)[-1]) if net.size else 0.0
    max_dd = _max_drawdown(net)
    trade_count = int(net.size)

    audit.trade_count = trade_count
    audit.gross_pnl = float(np.sum(gross))
    audit.total_cost = float(np.sum(costs))
    audit.net_pnl = float(np.sum(net))
    audit.profit_factor = float(pf)
    audit.win_rate = float(np.mean(net > 0)) if trade_count else 0.0
    audit.max_drawdown = max_dd
    audit.final_equity = final_equity
    audit.activity_label = _activity_label(trade_count)

    if trade_count == 0:
        audit.reject_reason = "NO_TRADES"
    elif audit.net_pnl <= 0.0:
        audit.reject_reason = "NOT_PROFITABLE_NET_PNL"
    elif audit.final_equity <= 0.0:
        audit.reject_reason = "EQUITY_NOT_POSITIVE"
    elif math.isfinite(audit.profit_factor) and audit.profit_factor <= 1.0:
        audit.reject_reason = "PF_NOT_ABOVE_1"
    elif not include_low_activity and audit.activity_label == "LOW_ACTIVITY":
        audit.reject_reason = "LOW_ACTIVITY_EXCLUDED"
    else:
        audit.reject_reason = ""
        audit.eligible_paper_forward = True
    return audit


def _candidate_to_config_entry(candidate: CandidateArtifact, audit: CandidateAuditRow) -> Dict[str, Any]:
    payload = dict(candidate.metadata)
    payload.update(
        {
            "candidate_id": candidate.candidate_id,
            "artifact_dir": str(candidate.artifact_dir),
            "artifact_path": str(candidate.artifact_dir),
            "model_path": str(candidate.model_path) if candidate.model_path else "",
            "model_name": candidate.model_name,
            "preset_family": candidate.preset_family,
            "side_policy": candidate.side_policy,
            "threshold": float(audit.threshold_used),
            "selected_threshold": float(audit.threshold_used),
            "threshold_source": audit.threshold_source,
            "feature_order_source": candidate.feature_order_path,
            "enabled": True,
            "paper_forward_only": True,
            "classification": "paper_forward_only",
            "eligible_paper_forward": True,
            "paper_forward_passed": False,
            "shadow_ready": False,
        }
    )
    return payload


def _write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def run_selection(
    *,
    artifacts_dir: Path,
    dataset_path: Optional[Path],
    output_config: Path,
    apply: bool,
    dry_run: bool,
    no_min_trades: bool,
    max_candidates: Optional[int],
    side: str,
    include_low_activity: bool,
    fallback_threshold: float = 0.35,
) -> Dict[str, Any]:
    del no_min_trades  # explicit no-op: no minimum trade-count gate is applied.

    discovered = _discover_candidate_artifacts(artifacts_dir, fallback_threshold=fallback_threshold)
    if side:
        discovered = [row for row in discovered if _side_matches_cli(row, side)]
    if max_candidates is not None and max_candidates >= 0:
        discovered = discovered[: max_candidates]

    dataset_path = dataset_path.resolve() if dataset_path is not None else _choose_best_dataset(discovered, fallback_threshold)
    dataset = _normalize_columns(pd.read_csv(dataset_path))
    required_feature_union = _candidate_required_feature_union(discovered, fallback_threshold)
    if required_feature_union:
        _materialize_offline_model_features(dataset, required_feature_union, log_fn=None)
    gross_col, net_col = _select_outcome_columns(dataset)

    audits: List[CandidateAuditRow] = []
    eligible_rows: List[Dict[str, Any]] = []
    broken_rejections: List[str] = []
    evaluated = 0
    for candidate in discovered:
        audit = evaluate_candidate(
            candidate,
            dataset=dataset,
            gross_col=gross_col,
            net_col=net_col,
            fallback_threshold=fallback_threshold,
            include_low_activity=include_low_activity,
        )
        audits.append(audit)
        if audit.prediction_attempted:
            evaluated += 1
        if audit.reject_reason in {
            "ARTIFACT_LOAD_FAILED",
            "FEATURES_MISSING",
            "FEATURE_ORDER_MISSING",
            "ARTIFACT_IDENTITY_MISMATCH",
            "CONSTANT_ZERO_PREDICTION",
            "FEATURE_VECTOR_INVALID",
            "PREDICTION_ERROR",
            "NON_FINITE_PREDICTION",
        }:
            broken_rejections.append(f"{audit.candidate_id}:{audit.reject_reason}")
        if audit.eligible_paper_forward:
            eligible_rows.append(_candidate_to_config_entry(candidate, audit))

    stamp = _now_stamp()
    audit_payload = [asdict(row) for row in audits]
    eligible_payload = [row for row in audit_payload if row["eligible_paper_forward"]]

    report_paths = {
        "eligible_csv": REPORTS_DIR / f"paper_forward_eligible_candidates_{stamp}.csv",
        "eligible_json": REPORTS_DIR / f"paper_forward_eligible_candidates_{stamp}.json",
        "audit_csv": REPORTS_DIR / f"paper_forward_all_candidate_audit_{stamp}.csv",
        "audit_json": REPORTS_DIR / f"paper_forward_all_candidate_audit_{stamp}.json",
    }
    _write_report(report_paths["eligible_csv"], eligible_payload)
    _write_report(report_paths["eligible_json"], eligible_payload)
    _write_report(report_paths["audit_csv"], audit_payload)
    _write_report(report_paths["audit_json"], audit_payload)

    config_written = False
    if apply and not dry_run:
        output_config.parent.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "mode": "paper_forward_multi",
            "paper_only": True,
            "real_trading_enabled": False,
            "broker_orders_enabled": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset_path),
            "source_reports": {k: str(v) for k, v in report_paths.items()},
            "candidates": eligible_rows,
        }
        output_config.write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
        config_written = True

    eligible_sorted = sorted(
        [row for row in audits if row.eligible_paper_forward],
        key=lambda row: (row.net_pnl, row.profit_factor if math.isfinite(row.profit_factor) else float("inf")),
        reverse=True,
    )
    return {
        "dataset_path": str(dataset_path),
        "gross_col": gross_col,
        "net_col": net_col,
        "candidate_folders_scanned": len(discovered),
        "valid_model_candidates_evaluated": evaluated,
        "eligible_profitable_candidates_selected": len(eligible_rows),
        "rejected_candidates_count": len(discovered) - len(eligible_rows),
        "top_10_eligible": [asdict(row) for row in eligible_sorted[:10]],
        "generated_reports": {k: str(v) for k, v in report_paths.items()},
        "eligible_config_path": str(output_config),
        "eligible_config_written": config_written,
        "broken_candidates": broken_rejections,
        "all_audits": audit_payload,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select profitable paper-forward candidates from all artifact folders.")
    parser.add_argument("--artifacts-dir", default=str(ROOT / "artifacts" / "candidates"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-config", default=str(DEFAULT_OUTPUT_CONFIG))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-min-trades", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--side", choices=["CE_ONLY", "PE_ONLY", "BOTH", "ALL"], default="ALL")
    parser.add_argument("--include-low-activity", default="true")
    parser.add_argument("--default-threshold", type=float, default=0.35)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    dataset_path = _find_dataset_path(args.dataset) if args.dataset else None
    include_low_activity = _truthy(args.include_low_activity, default=True)
    summary = run_selection(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        dataset_path=dataset_path,
        output_config=Path(args.output_config).resolve(),
        apply=bool(args.apply),
        dry_run=bool(args.dry_run),
        no_min_trades=bool(args.no_min_trades),
        max_candidates=args.max_candidates,
        side=args.side,
        include_low_activity=include_low_activity,
        fallback_threshold=float(args.default_threshold),
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
