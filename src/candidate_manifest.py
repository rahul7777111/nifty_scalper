#!/usr/bin/env python3
"""
candidate_manifest.py
=====================
Standard loader and validator for NiftyScalper candidate manifests.

Every persisted candidate lives under:
    models/candidates/<candidate_id>/
        candidate_manifest.json   (required)
        model.pkl                 (required)
        feature_schema.json       (required)
        preprocessing_metadata.json (required)
        filter_definition.json    (required)
        threshold_sweep.json      (optional but required for routing)
        gate_results.json         (required)
        metrics.json              (optional)

Safety constraints (fail-closed on violation):
  - paper_only must be True
  - real_trading_enabled must be False
  - model.pkl must exist and be readable as a pickle
  - feature_schema.json must exist and be valid JSON
  - filter_definition.json must exist and be valid JSON
  - required_live_fields must be non-empty

Real trading mode: candidate loading is always blocked regardless of manifest contents.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, failing with a clear error on decode failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to read JSON from {path}: {exc}") from exc


def _resolve_path(manifest_dir: Path, rel_path: str) -> Path:
    """Resolve a relative path from the manifest directory to an absolute path."""
    return (manifest_dir / rel_path).resolve()


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_candidate_manifest(
    manifest_path: Path | str,
) -> Dict[str, Any]:
    """Load and validate a candidate_manifest.json.

    Parameters
    ----------
    manifest_path : Path | str
        Path to the candidate_manifest.json file.

    Returns
    -------
    dict
        Validated manifest dict with resolved paths and defaults filled.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    ValueError
        If the manifest is missing required fields, or if safety constraints
        (paper_only, real_trading_enabled) are violated.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Candidate manifest not found: {manifest_path}\n"
            f"Provide a valid path to a candidate_manifest.json file."
        )

    # Load raw JSON
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Candidate manifest is not valid JSON: {manifest_path}\n"
            f"Error: {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Candidate manifest is unreadable: {manifest_path}\n"
            f"Error: {exc}"
        ) from exc

    # Resolve paths
    manifest_dir = manifest_path.parent.resolve()

    # Accept candidate_id OR model_id (both names used across artifacts)
    candidate_id = payload.get("candidate_id") or payload.get("model_id")
    if not candidate_id:
        raise ValueError(
            f"Candidate manifest is missing required field: 'candidate_id' (or 'model_id').\n"
            f"Manifest: {manifest_path}"
        )

    # --- Safety constraint: paper_only must be True ---
    paper_only = payload.get("paper_only")
    if paper_only is not True:
        raise ValueError(
            f"Candidate manifest safety constraint violated: 'paper_only' must be True.\n"
            f"Found: {paper_only!r}\n"
            f"Manifest: {manifest_path}\n"
            f"Candidate loading is blocked for real-trading mode."
        )

    # --- Safety constraint: real_trading_enabled must be False ---
    real_trading_enabled = payload.get("real_trading_enabled")
    if real_trading_enabled is not False:
        raise ValueError(
            f"Candidate manifest safety constraint violated: 'real_trading_enabled' must be False.\n"
            f"Found: {real_trading_enabled!r}\n"
            f"Manifest: {manifest_path}\n"
            f"Candidate loading is blocked for real-trading mode."
        )

    # --- Required fields ---
    required_fields = ["model_pkl", "selected_threshold"]
    missing = [
        f for f in required_fields
        if f not in payload
        or payload[f] is None
        or (isinstance(payload[f], str) and not payload[f].strip())
    ]
    if missing:
        raise ValueError(
            f"Candidate manifest is missing required field(s): {missing}.\n"
            f"Manifest: {manifest_path}\n"
            f"Required: {required_fields} + 'candidate_id'/'model_id'"
        )

    # --- Resolve model.pkl ---
    model_pkl_rel = payload["model_pkl"]
    model_pkl_path = _resolve_path(manifest_dir, model_pkl_rel)
    if not model_pkl_path.exists():
        raise FileNotFoundError(
            f"model.pkl does not exist: {model_pkl_path}\n"
            f"Resolved from 'model_pkl' field: {model_pkl_rel}\n"
            f"Manifest: {manifest_path}"
        )

    # --- Resolve feature_schema.json ---
    feature_schema_path = _resolve_path(
        manifest_dir,
        payload.get("feature_schema_path", "feature_schema.json")
    )
    if not feature_schema_path.exists():
        raise FileNotFoundError(
            f"feature_schema.json does not exist: {feature_schema_path}\n"
            f"Manifest: {manifest_path}"
        )

    # --- Resolve preprocessing_metadata.json ---
    preprocessing_path = _resolve_path(
        manifest_dir,
        payload.get("preprocessing_path", "preprocessing_metadata.json")
    )
    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"preprocessing_metadata.json does not exist: {preprocessing_path}\n"
            f"Manifest: {manifest_path}"
        )

    # --- Resolve filter_definition.json ---
    filter_def_path = _resolve_path(
        manifest_dir,
        payload.get("filter_definition_path", "filter_definition.json")
    )
    if not filter_def_path.exists():
        raise FileNotFoundError(
            f"filter_definition.json does not exist: {filter_def_path}\n"
            f"Manifest: {manifest_path}"
        )

    # --- Resolve gate_results.json ---
    gate_results_path = _resolve_path(
        manifest_dir,
        payload.get("gate_results_path", "gate_results.json")
    )
    if not gate_results_path.exists():
        raise FileNotFoundError(
            f"gate_results.json does not exist: {gate_results_path}\n"
            f"Manifest: {manifest_path}"
        )

    # --- Resolve metrics.json (optional but warn if missing) ---
    metrics_path = _resolve_path(manifest_dir, "metrics.json")
    has_metrics = metrics_path.exists()

    # --- Build validated manifest dict ---
    filter_def = _load_json(filter_def_path)
    gate_results = _load_json(gate_results_path)

    # Count gates passed
    # Handle multiple gate_results.json formats:
    # Format A: {"gate_status": {"G1": {"pass": true}, ...}}
    # Format B: {"gates_passed": 10, "gates_evaluated": 10}
    # Format C: {"gates": [{"name": ..., "pass": true}, ...]}
    gate_status = gate_results.get("gate_status", {})
    if isinstance(gate_status, dict) and gate_status:
        gates_passed = sum(1 for v in gate_status.values() if v.get("pass") is True)
        gates_total = len(gate_status)
    elif isinstance(gate_results.get("gates_passed"), int):
        gates_passed = gate_results.get("gates_passed", 0)
        # Infer total from gates_evaluated if gates_total missing
        gates_total = gate_results.get("gates_total") or gate_results.get("gates_evaluated") or gates_passed
    else:
        # Try "gates" list format
        gates_list = gate_results.get("gates", [])
        if isinstance(gates_list, list) and gates_list:
            gates_passed = sum(1 for g in gates_list if g.get("pass") is True)
            gates_total = len(gates_list)
        else:
            gates_passed = None
            gates_total = None

    return {
        # Identity
        "candidate_id": str(candidate_id),
        "model_id": str(candidate_id),
        "paper_only": True,
        "real_trading_enabled": False,
        "status": payload.get("status", "UNKNOWN"),
        "created_at": payload.get("created_at", payload.get("persistence_timestamp", "")),
        # Model
        "model_name": payload.get("model_name", payload.get("model", {}).get("model_name", "unknown")),
        "model_pkl": str(model_pkl_path),
        "model_pkl_relative": model_pkl_rel,
        "target": payload.get("target", payload.get("model", {}).get("target", "unknown")),
        # Feature schema
        "feature_schema": _load_json(feature_schema_path),
        "feature_schema_path": str(feature_schema_path),
        # Preprocessing
        "preprocessing_metadata": _load_json(preprocessing_path),
        "preprocessing_path": str(preprocessing_path),
        # Filter
        "filter_definition": filter_def,
        "filter_definition_path": str(filter_def_path),
        "filter_name": filter_def.get("filter_name", payload.get("filter_name", "unknown")),
        "filter_rule": filter_def.get("filter_rule", payload.get("filter_rule", "")),
        "filter_type": filter_def.get("filter_type", ""),
        "required_live_fields": payload.get(
            "required_live_fields",
            filter_def.get("required_fields", [])
        ),
        # Threshold
        "selected_threshold": float(payload["selected_threshold"]),
        # Gates
        "gate_results": gate_results,
        "gate_results_path": str(gate_results_path),
        "gates_passed": gates_passed,
        "gates_total": gates_total,
        "gate_status": gate_status,
        # Metrics (may be empty dict if missing)
        "metrics": _load_json(metrics_path) if has_metrics else {},
        "metrics_path": str(metrics_path) if has_metrics else None,
        # Source tracking
        "source_report_paths": payload.get("source_report_paths", []),
        "source_manifest": str(manifest_path),
        "source_manifest_dir": str(manifest_dir),
        "candidate_dir": str(manifest_dir),
    }


def validate_candidate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate a loaded manifest dict.

    Returns a list of error strings (empty if valid).
    Does NOT raise — accumulate all errors.
    """
    errors: List[str] = []
    cid = manifest.get("candidate_id", "UNKNOWN")

    if not manifest.get("paper_only") is True:
        errors.append(f"[{cid}] paper_only must be True")

    if not manifest.get("real_trading_enabled") is False:
        errors.append(f"[{cid}] real_trading_enabled must be False")

    if not manifest.get("model_pkl"):
        errors.append(f"[{cid}] model_pkl is missing")

    if not manifest.get("selected_threshold"):
        errors.append(f"[{cid}] selected_threshold is missing")

    if not isinstance(manifest.get("selected_threshold"), (int, float)):
        errors.append(f"[{cid}] selected_threshold must be numeric")

    if not manifest.get("filter_name"):
        errors.append(f"[{cid}] filter_name is missing")

    if not manifest.get("filter_rule"):
        errors.append(f"[{cid}] filter_rule is missing")

    # Check required_live_fields is non-empty list
    rlf = manifest.get("required_live_fields", [])
    if not isinstance(rlf, list) or len(rlf) == 0:
        errors.append(f"[{cid}] required_live_fields must be a non-empty list")

    # Check gate results exist and have some passing gates
    gates_passed = manifest.get("gates_passed")
    gates_total = manifest.get("gates_total")
    if gates_passed is None or gates_total is None:
        errors.append(f"[{cid}] gate results are incomplete (gates_passed={gates_passed}, gates_total={gates_total})")
    elif gates_passed == 0:
        errors.append(f"[{cid}] no gates passed (gates_passed=0)")

    return errors


def resolve_candidate_paths(manifest: Dict[str, Any]) -> Dict[str, Path]:
    """Return a dict of name -> absolute Path for all artifact files."""
    return {
        "model_pkl": Path(manifest["model_pkl"]),
        "feature_schema": Path(manifest["feature_schema_path"]),
        "preprocessing": Path(manifest["preprocessing_path"]),
        "filter_definition": Path(manifest["filter_definition_path"]),
        "gate_results": Path(manifest["gate_results_path"]),
        "metrics": Path(manifest["metrics_path"]) if manifest.get("metrics_path") else None,
    }


def load_candidate_model(manifest: Dict[str, Any]):
    """Unpickle and return the candidate model bundle.

    Returns
    -------
    object
        The unpickled model object (typically an sklearn estimator or similar).

    Raises
    ------
    FileNotFoundError
        If model.pkl does not exist.
    pickle.UnpicklingError
        If the file is not a valid pickle.
    """
    model_path = Path(manifest["model_pkl"])
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found: {model_path}")

    try:
        with model_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        raise pickle.UnpicklingError(
            f"Failed to unpickle model at {model_path}: {exc}"
        ) from exc


def load_candidate_feature_schema(manifest: Dict[str, Any]) -> List[str]:
    """Return the list of feature names from the candidate's feature_schema.json."""
    schema = manifest.get("feature_schema", {})
    if isinstance(schema, list):
        return schema
    if isinstance(schema, dict):
        return schema.get("features", [])
    return []


def load_candidate_filter_definition(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Return the filter definition dict."""
    return manifest.get("filter_definition", {})


# ---------------------------------------------------------------------------
# Bulk loader
# ---------------------------------------------------------------------------

def load_candidates_from_dir(
    candidates_dir: Path | str,
    require_all_artifacts: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Scan a directory for candidate subdirectories and load each manifest.

    Parameters
    ----------
    candidates_dir : Path | str
        Directory containing candidate subdirectories (each with a candidate_manifest.json).
    require_all_artifacts : bool
        If True, skip candidates missing any required artifact file.
        If False, load whatever is present and let validation catch issues.

    Returns
    -------
    (valid_candidates, load_errors)
        valid_candidates : list of loaded + validated manifest dicts
        load_errors      : list of {"path": str, "error": str} for failed loads
    """
    candidates_dir = Path(candidates_dir)
    if not candidates_dir.exists():
        return [], [{"path": str(candidates_dir), "error": "Directory does not exist"}]

    valid: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for subdir in sorted(candidates_dir.iterdir()):
        if not subdir.is_dir():
            continue
        manifest_path = subdir / "candidate_manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = load_candidate_manifest(manifest_path)
            validation_errors = validate_candidate_manifest(manifest)
            if validation_errors:
                for err in validation_errors:
                    errors.append({"path": str(manifest_path), "error": err})
                continue
            valid.append(manifest)
        except Exception as exc:
            errors.append({"path": str(manifest_path), "error": str(exc)})

    return valid, errors