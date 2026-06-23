"""Helpers for the Historical ML Backtest tab ensemble retrain/test workflow."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENSEMBLE_DYNAMIC_CONFIG_NAME = "ensemble_xgb_rf_dynamic_candidate.json"
ENSEMBLE_ARTIFACT_GLOB = "*xgb_rf_ensemble*profitable_trade_label*.pkl"


def _artifact_freshness_key(path: Path | None) -> tuple[str, float]:
    if path is None:
        return ("", -1.0)
    text = str(path)
    match = re.search(r"(?:research_retrain|ensemble_gui_retrain)_(\d{8}_\d{6})", text)
    stamp = match.group(1) if match else ""
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = -1.0
    return (stamp, mtime)


def load_ensemble_feature_names(artifact_dir: Path) -> list[str]:
    """Load feature names from the artifacts retrain scripts actually write."""
    if not artifact_dir.exists():
        return []

    candidate_paths: list[Path] = [
        artifact_dir / "feature_manifest.json",
        artifact_dir / "feature_list_used.json",
    ]
    candidate_paths.extend(
        sorted(artifact_dir.glob("*_feature_order.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    )
    candidate_paths.extend(
        sorted(artifact_dir.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    )

    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("features", "feature_order", "feature_names"):
            vals = payload.get(key)
            if isinstance(vals, list) and vals:
                return [str(x) for x in vals if str(x).strip()]
    return []


def find_latest_ensemble_artifact(output_root: Path) -> tuple[Path | None, Path | None]:
    """Return the newest xgb_rf_ensemble profitable_trade_label artifact under a retrain output root."""
    candidates = [output_root] if output_root.exists() and output_root.is_dir() else []
    if output_root.exists():
        candidates.extend(path for path in output_root.iterdir() if path.is_dir())
    candidates = sorted(
        {path.resolve() for path in candidates},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for folder in candidates:
        try:
            model_files = sorted(
                folder.glob(ENSEMBLE_ARTIFACT_GLOB),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            model_files = []
        if model_files:
            return model_files[0], folder
    return None, None


def _resolve_repo_path(repo_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_dataset_path(repo_root: Path, raw: str | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return str(path) if path.is_file() else ""


def _load_ensemble_metrics(artifact_dir: Path) -> dict[str, Any]:
    for path in sorted(artifact_dir.glob("*xgb_rf_ensemble*profitable_trade_label*_metrics.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def load_ensemble_selected_threshold(artifact_dir: Path) -> float | None:
    """Read only the validation-selected threshold without parsing huge metrics blobs."""
    for path in sorted(artifact_dir.glob("*xgb_rf_ensemble*profitable_trade_label*_metadata.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            for key in ("selected_threshold", "threshold"):
                val = payload.get(key)
                if val is not None:
                    return float(val)
        except Exception:
            continue

    for path in sorted(artifact_dir.glob("*xgb_rf_ensemble*profitable_trade_label*_metrics.json"), reverse=True):
        try:
            with path.open("r", encoding="utf-8") as fh:
                head = fh.read(16384)
            match = re.search(
                r'"selected_threshold_from_validation"\s*:\s*\{[^{}]*"threshold"\s*:\s*([0-9.]+)',
                head,
            )
            if match:
                return float(match.group(1))
        except Exception:
            continue
    return None


def _dataset_hint_from_artifact_dir(artifact_dir: Path) -> str:
    candidate_paths: list[Path] = [
        artifact_dir / "feature_manifest.json",
        artifact_dir / "feature_list_used.json",
    ]
    candidate_paths.extend(
        sorted(artifact_dir.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    )

    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            dataset = str((payload or {}).get("dataset_path") or "").strip()
            if dataset:
                return dataset
        except Exception:
            continue

    preprocessing_path = artifact_dir / "preprocessing_metadata.json"
    if preprocessing_path.is_file():
        try:
            payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
            dataset = str((payload or {}).get("dataset_path") or "").strip()
            if dataset:
                return dataset
            source_artifact_dir = str((payload or {}).get("source_artifact_dir") or "").strip()
            if source_artifact_dir:
                source_dir = _resolve_repo_path(artifact_dir.parents[2], source_artifact_dir)
                if source_dir and source_dir.is_dir():
                    dataset = _dataset_hint_from_artifact_dir(source_dir)
                    if dataset:
                        return dataset
        except Exception:
            pass
    return ""


def _dataset_hint_from_candidate(repo_root: Path, cand: dict[str, Any]) -> str:
    for raw_dir in (
        cand.get("artifact_dir"),
        cand.get("model_dir"),
    ):
        resolved_dir = _resolve_repo_path(repo_root, str(raw_dir or ""))
        if resolved_dir and resolved_dir.is_dir():
            dataset = _dataset_hint_from_artifact_dir(resolved_dir)
            if dataset:
                return resolve_dataset_path(repo_root, dataset)

    model_path = _resolve_repo_path(repo_root, str(cand.get("model_path") or cand.get("artifact_path") or ""))
    if model_path and model_path.is_file():
        dataset = _dataset_hint_from_artifact_dir(model_path.parent)
        if dataset:
            return resolve_dataset_path(repo_root, dataset)
    return ""


def _candidate_from_config(repo_root: Path, config_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    cands = payload.get("candidates") or []
    if not isinstance(cands, list):
        return None
    for cand in cands:
        if not isinstance(cand, dict) or not cand.get("enabled", True):
            continue
        if str(cand.get("model_name") or "").lower() == "xgb_rf_ensemble":
            return cand
    return cands[0] if cands and isinstance(cands[0], dict) else None


def _config_model_path_exists(repo_root: Path, config_path: Path) -> bool:
    cand = _candidate_from_config(repo_root, config_path)
    if not cand:
        return False
    model_path = _resolve_repo_path(repo_root, str(cand.get("model_path") or cand.get("artifact_path") or ""))
    return bool(model_path and model_path.is_file())


def discover_latest_ensemble_gui_retrain_root(repo_root: Path) -> Path | None:
    models_dir = repo_root / "models"
    if not models_dir.is_dir():
        return None
    roots = sorted(
        (p for p in models_dir.glob("ensemble_gui_retrain_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return roots[0] if roots else None


def discover_ensemble_test_target(
    repo_root: Path,
    *,
    last_config: str = "",
    last_artifact: str = "",
) -> dict[str, Any]:
    """Resolve the ensemble config/artifact and backtest defaults for the GUI Test button."""
    last_config_path = Path(last_config) if last_config else None
    if last_config_path and last_config_path.is_file():
        cand = _candidate_from_config(repo_root, last_config_path) or {}
        return {
            "config_path": str(last_config_path),
            "artifact_path": str(_resolve_repo_path(repo_root, str(cand.get("model_path") or "")) or ""),
            "candidate_id": str(cand.get("candidate_id") or ""),
            "dataset_hint": _dataset_hint_from_candidate(repo_root, cand),
            "selected_threshold": cand.get("selected_threshold"),
            "max_trades_per_day": cand.get("max_trades_per_day"),
            "source": "session_config",
        }

    last_artifact_path = Path(last_artifact) if last_artifact else None
    if last_artifact_path and last_artifact_path.is_file():
        threshold = load_ensemble_selected_threshold(last_artifact_path.parent)
        return {
            "config_path": "",
            "artifact_path": str(last_artifact_path),
            "candidate_id": "",
            "dataset_hint": resolve_dataset_path(
                repo_root,
                _dataset_hint_from_artifact_dir(last_artifact_path.parent),
            ),
            "selected_threshold": threshold,
            "max_trades_per_day": 1,
            "source": "session_artifact",
        }

    latest_root = discover_latest_ensemble_gui_retrain_root(repo_root)
    latest_gui_target: dict[str, Any] | None = None
    latest_gui_artifact_key = ("", -1.0)
    if latest_root is not None:
        artifact_path, artifact_dir = find_latest_ensemble_artifact(latest_root)
        if artifact_path and artifact_dir:
            latest_gui_artifact_key = _artifact_freshness_key(artifact_path)
            threshold = load_ensemble_selected_threshold(artifact_dir)
            latest_gui_target = {
                "config_path": "",
                "artifact_path": str(artifact_path),
                "candidate_id": "",
                "dataset_hint": resolve_dataset_path(
                    repo_root,
                    _dataset_hint_from_artifact_dir(artifact_dir),
                ),
                "selected_threshold": threshold,
                "max_trades_per_day": 1,
                "source": "latest_gui_retrain",
            }

    static_config = repo_root / "config" / ENSEMBLE_DYNAMIC_CONFIG_NAME
    if static_config.is_file() and _config_model_path_exists(repo_root, static_config):
        static_candidate = _candidate_from_config(repo_root, static_config) or {}
        static_artifact = _resolve_repo_path(
            repo_root,
            str(static_candidate.get("model_path") or static_candidate.get("artifact_path") or ""),
        )
        static_artifact_key = _artifact_freshness_key(static_artifact if static_artifact and static_artifact.is_file() else None)
        if latest_gui_target is not None and latest_gui_artifact_key > static_artifact_key:
            return latest_gui_target
        cand = _candidate_from_config(repo_root, static_config) or {}
        return {
            "config_path": str(static_config),
            "artifact_path": str(_resolve_repo_path(repo_root, str(cand.get("model_path") or "")) or ""),
            "candidate_id": str(cand.get("candidate_id") or ""),
            "dataset_hint": _dataset_hint_from_candidate(repo_root, cand),
            "selected_threshold": cand.get("selected_threshold"),
            "max_trades_per_day": cand.get("max_trades_per_day"),
            "source": "static_config",
        }

    if latest_gui_target is not None:
        return latest_gui_target

    return {
        "error": "Retrain the ensemble first or select a model artifact/config path.",
    }


def materialize_ensemble_dynamic_candidate(
    repo_root: Path,
    artifact_path: Path,
    artifact_dir: Path,
) -> tuple[Path | None, Path | None]:
    """Build the ensemble dynamic-preset wrapper and config JSON used by the GUI test flow."""
    try:
        features = load_ensemble_feature_names(artifact_dir)
        if not features:
            return None, None

        metrics_payload = _load_ensemble_metrics(artifact_dir)
        raw_threshold = (metrics_payload.get("selected_threshold_from_validation") or {}).get("threshold")
        wrapper_threshold = float(raw_threshold) if raw_threshold is not None else 0.45
        source_threshold = float(raw_threshold) if raw_threshold is not None else wrapper_threshold

        created_at = datetime.now(timezone.utc).isoformat()
        artifact_suffix = artifact_dir.name.replace("research_retrain_", "")
        candidate_id = f"xgb_rf_ensemble_high_confidence_low_frequency_hi_conf_low_freq_t45_{artifact_suffix}"
        wrapper_dir = repo_root / "models" / "candidates" / candidate_id
        wrapper_dir.mkdir(parents=True, exist_ok=True)

        model_rel = os.path.relpath(artifact_path, wrapper_dir).replace("/", "\\")
        metrics_path = next(iter(sorted(artifact_dir.glob("*xgb_rf_ensemble*profitable_trade_label*_metrics.json"), reverse=True)), None)
        metrics_rel = (
            str(metrics_path.relative_to(repo_root)).replace("/", "\\")
            if metrics_path and metrics_path.exists()
            else ""
        )
        feature_rel = ""
        for candidate in (
            artifact_dir / "feature_manifest.json",
            artifact_dir / "feature_list_used.json",
            next(iter(sorted(artifact_dir.glob("*_feature_order.json"), reverse=True)), None),
        ):
            if candidate and candidate.exists():
                feature_rel = str(candidate.relative_to(repo_root)).replace("/", "\\")
                break

        preset_t45 = {
            "name": "hi_conf_low_freq_t45",
            "preset_id": "hi_conf_low_freq_t45",
            "preset_family": "high_confidence_low_frequency",
            "option_side_policy": "BOTH",
            "threshold": wrapper_threshold,
            "min_confidence": wrapper_threshold,
            "entry_threshold": wrapper_threshold,
            "max_trades_per_day": 1,
            "top_n_confidence_per_day": 1,
            "spread_limit_pct": 0.08,
            "max_spread_pct": 0.08,
            "liquidity_min": 100000.0,
            "premium_band": "all",
            "dte_range": "all",
            "regime_filter": "all",
            "expiry_filter": "all",
            "avoid_first_n_minutes": 0,
            "avoid_last_n_minutes": 0,
            "entry_time_start": "09:30",
            "entry_time_end": "15:00",
        }
        preset_t50 = dict(preset_t45)
        preset_t50.update(
            {
                "name": "hi_conf_low_freq_t50",
                "preset_id": "hi_conf_low_freq_t50",
                "threshold": 0.50,
                "min_confidence": 0.50,
                "entry_threshold": 0.50,
            }
        )
        dynamic_presets = {"hi_conf_low_freq_t45": preset_t45, "hi_conf_low_freq_t50": preset_t50}
        wrapper_metrics = {
            "model_name": "xgb_rf_ensemble",
            "source_artifact_threshold": source_threshold,
            "wrapper_selected_threshold": wrapper_threshold,
            "validation_metrics": metrics_payload.get("validation_metrics") or {},
            "validation_trade_metrics": metrics_payload.get("validation_trade_metrics") or {},
            "test_metrics": metrics_payload.get("test_metrics") or {},
        }
        candidate_profile = {
            "candidate_id": candidate_id,
            "model_name": "xgb_rf_ensemble",
            "feature_set_name": "ensemble_retrain_feature_manifest",
            "target_name": "profitable_trade_label",
            "side_policy": "BOTH",
            "preset_family": "high_confidence_low_frequency",
            "dynamic_presets": dynamic_presets,
            "threshold_policy": {"entry_threshold": wrapper_threshold, "min_confidence": wrapper_threshold},
            "selection_policy": {"max_trades_per_day": 1},
            "risk_policy": {"max_spread_pct": 0.08, "liquidity_min": 100000.0},
            "cost_policy": {},
            "artifact_paths": {"model_path": model_rel},
            "validation_metrics": wrapper_metrics,
            "gate_results": {"paper_only": True, "real_trading_enabled": False, "research_only": True},
            "live_computable_features": features,
            "created_at": created_at,
            "shadow_mode_metadata": {},
        }
        manifest = {
            "candidate_id": candidate_id,
            "paper_only": True,
            "real_trading_enabled": False,
            "created_at": created_at,
            "status": "RESEARCH_ONLY_DYNAMIC_PRESET_WRAPPER",
            "model_name": "xgb_rf_ensemble",
            "target": "profitable_trade_label",
            "model_pkl": model_rel,
            "selected_threshold": wrapper_threshold,
            "feature_schema_path": "feature_schema.json",
            "preprocessing_path": "preprocessing_metadata.json",
            "filter_definition_path": "filter_definition.json",
            "gate_results_path": "gate_results.json",
            "required_live_fields": features,
            "filter_name": "dynamic_preset",
            "filter_rule": "high_confidence_low_frequency_t45",
            "source_report_paths": [metrics_rel] if metrics_rel else [],
        }
        feature_schema = {
            "feature_names": features,
            "features": features,
            "required_live_fields": features,
            "feature_count": len(features),
            "source": feature_rel,
        }
        preprocessing_metadata = {
            "source_artifact_dir": str(artifact_dir.relative_to(repo_root)).replace("/", "\\"),
            "feature_manifest_path": feature_rel,
            "dataset_path": _dataset_hint_from_artifact_dir(artifact_dir),
            "notes": "Ensemble dynamic preset wrapper generated from GUI retrain flow",
        }
        filter_definition = {
            "filter_name": "dynamic_preset",
            "filter_rule": "high_confidence_low_frequency_t45",
            "filter_type": "dynamic_preset",
            "required_fields": ["option_type", "volume", "oi", "bid_ask_spread_pct"],
            "filters": {"max_spread_pct": 0.08, "spread_limit_pct": 0.08, "liquidity_min": 100000.0},
        }
        gate_results = {
            "gates_passed": 3,
            "gates_total": 3,
            "gates": [
                {"name": "paper_only", "pass": True},
                {"name": "real_trading_disabled", "pass": True},
                {"name": "feature_manifest_present", "pass": True},
            ],
        }
        files_to_write = {
            "candidate_manifest.json": manifest,
            "candidate_profile.json": candidate_profile,
            "dynamic_preset.json": preset_t45,
            "dynamic_presets.json": dynamic_presets,
            "feature_schema.json": feature_schema,
            "preprocessing_metadata.json": preprocessing_metadata,
            "filter_definition.json": filter_definition,
            "gate_results.json": gate_results,
            "metrics.json": wrapper_metrics,
        }
        for name, payload in files_to_write.items():
            (wrapper_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

        config_path = repo_root / "config" / ENSEMBLE_DYNAMIC_CONFIG_NAME
        config_payload = {
            "mode": "paper_forward_multi",
            "live_orders_enabled": False,
            "broker_orders_enabled": False,
            "created_at": created_at,
            "source_report": "GUI ensemble retrain dynamic preset wrapper",
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "enabled": True,
                    "classification": "paper_forward_only",
                    "paper_forward_only": True,
                    "paper_only": True,
                    "real_trading_enabled": False,
                    "model_name": "xgb_rf_ensemble",
                    "preset_family": "high_confidence_low_frequency",
                    "side_policy": "BOTH",
                    "artifact_dir": str(wrapper_dir.relative_to(repo_root)).replace("/", "\\"),
                    "model_path": str(artifact_path.relative_to(repo_root)).replace("/", "\\"),
                    "feature_order_source": str((wrapper_dir / "feature_schema.json").relative_to(repo_root)).replace("/", "\\"),
                    "direct_model_artifact": True,
                    "selected_threshold": wrapper_threshold,
                    "max_trades_per_day": 1,
                    "filters": {"max_spread_pct": 0.08, "spread_limit_pct": 0.08, "liquidity_min": 100000.0},
                    "preset": preset_t45,
                }
            ],
        }
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        return wrapper_dir, config_path
    except Exception:
        return None, None
