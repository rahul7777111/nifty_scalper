#!/usr/bin/env python3
"""Strict candidate artifact identity resolver.

Single source of truth for historical backtest and paper-forward loaders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CANDIDATES_SUBDIR = "artifacts/candidates"
MODEL_FILENAME = "model.pkl"

KNOWN_MODEL_FAMILIES = (
    "calibrated_logistic_regression",
    "logistic_regression",
    "elasticnet",
    "xgboost",
    "random_forest",
    "extra_trees",
    "ensemble",
)

_IDENTITY_SIDECARS = (
    "candidate_manifest.json",
    "paper_forward_manifest.json",
    "candidate_profile.json",
    "metadata.json",
    "model_card.json",
    "manifest.json",
)


@dataclass
class CandidateArtifactResolution:
    candidate_id: str
    expected_artifact_path: str
    configured_artifact_path: Optional[str] = None
    selected_artifact_path: Optional[str] = None
    artifact_identity_status: str = "MISSING"  # EXACT_MATCH | MISMATCH | MISSING
    mismatch_fields: List[str] = field(default_factory=list)
    loaded: bool = False
    disable_reason: Optional[str] = None
    resolved_dir: Optional[str] = None
    model_file: Optional[str] = None
    metadata_file: Optional[str] = None
    strict: bool = True
    allow_fallback: bool = False

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Map to the legacy paper_forward_engine resolve payload shape."""
        load_status = "candidate_loaded_ok"
        status = "ok"
        if self.artifact_identity_status == "MISSING":
            load_status = "ARTIFACT_NOT_FOUND"
            status = "ARTIFACT_MISSING"
        elif self.artifact_identity_status == "MISMATCH":
            load_status = "ARTIFACT_IDENTITY_MISMATCH"
            status = "ARTIFACT_IDENTITY_MISMATCH"
        elif not self.loaded:
            load_status = "ARTIFACT_NOT_FOUND"
            status = "ARTIFACT_MISSING"

        return {
            "candidate_id": self.candidate_id,
            "expected_artifact_path": self.expected_artifact_path,
            "configured_artifact_path": self.configured_artifact_path,
            "selected_artifact_path": self.selected_artifact_path,
            "artifact_identity_status": self.artifact_identity_status,
            "mismatch_fields": list(self.mismatch_fields),
            "loaded": self.loaded,
            "disable_reason": self.disable_reason,
            "expected_dir": str(Path(self.expected_artifact_path).parent).replace("\\", "/"),
            "configured_artifact_dir": self.configured_artifact_path,
            "resolved_dir": self.resolved_dir,
            "model_file": self.model_file,
            "metadata_file": self.metadata_file,
            "status": status,
            "load_status": load_status,
            "strict": self.strict,
            "allow_fallback": self.allow_fallback,
            "exists": {"model_pkl": self.loaded},
            "files": {"model_pkl": self.model_file},
        }


def _repo_rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_side(value: Any) -> str:
    text = str(value or "").upper().replace(" ", "").replace("-", "_")
    if text in {"PE", "PEONLY"}:
        return "PE_ONLY"
    if text in {"CE", "CEONLY"}:
        return "CE_ONLY"
    if text == "BOTH":
        return "BOTH"
    return text


def _candidate_t_tag(value: str) -> str:
    match = re.search(r"(?:^|_)t(\d+)(?:_|$)", value or "")
    return match.group(1) if match else ""


def _infer_model_family(name: str) -> str:
    low = name.lower()
    for family in KNOWN_MODEL_FAMILIES:
        if family in low:
            return family
    return ""


def expected_artifact_paths(candidate_id: str, project_root: Path | str) -> Tuple[Path, Path]:
    root = Path(project_root).resolve()
    folder = root / CANDIDATES_SUBDIR / candidate_id
    model = folder / MODEL_FILENAME
    return folder, model


def _iter_configured_paths(candidate: Dict[str, Any]) -> Iterable[str]:
    keys = (
        "artifact_path",
        "model_path",
        "artifact_dir",
        "candidate_dir",
        "manifest_path",
    )
    for key in keys:
        value = candidate.get(key)
        if value:
            yield str(value)
    paths = candidate.get("artifact_paths")
    if isinstance(paths, dict):
        for key in keys + ("model", "bundle", "model_pkl"):
            value = paths.get(key)
            if value:
                yield str(value)


def _resolve_repo_path(value: str, project_root: Path) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _artifact_folder_for_path(path: Path, project_root: Path) -> Optional[Path]:
    if not path:
        return None
    p = path.resolve()
    candidates_root = (project_root / CANDIDATES_SUBDIR).resolve()
    if p.is_file():
        if p.name != MODEL_FILENAME:
            return None
        try:
            p.relative_to(candidates_root)
        except Exception:
            return None
        return p.parent
    if p.is_dir():
        try:
            p.relative_to(candidates_root)
        except Exception:
            return None
        return p
    return None


def _load_artifact_metadata(folder: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    search_dirs = [folder]
    nested = folder / folder.name
    if nested.is_dir():
        search_dirs.append(nested)
    for base in search_dirs:
        for fname in _IDENTITY_SIDECARS:
            fp = base / fname
            if not fp.exists():
                continue
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            for key in ("candidate_id", "artifact_id", "model_id", "id"):
                if payload.get(key) and not meta.get("candidate_id"):
                    meta["candidate_id"] = str(payload[key])
            for key in ("model_name", "model_family", "model"):
                if payload.get(key) and not meta.get("model_name"):
                    meta["model_name"] = str(payload[key])
            for key in ("preset_family", "preset", "selected_preset"):
                if payload.get(key) and not meta.get("preset_family"):
                    meta["preset_family"] = str(payload[key])
            for key in ("side_policy", "side", "filter_name"):
                if payload.get(key) and not meta.get("side_policy"):
                    meta["side_policy"] = str(payload[key])
    if not meta.get("model_name"):
        meta["model_name"] = _infer_model_family(folder.name)
    meta["folder_name"] = folder.name
    meta["t_tag"] = _candidate_t_tag(folder.name)
    return meta


def validate_artifact_identity(
    candidate: Dict[str, Any],
    artifact_folder: Path,
    *,
    allow_timestamp_mismatch: bool = False,
) -> Tuple[bool, List[str]]:
    """Return (ok, mismatch_fields)."""
    cid = str(candidate.get("candidate_id") or candidate.get("id") or "")
    mismatches: List[str] = []
    folder_name = artifact_folder.name

    if folder_name != cid:
        if not (allow_timestamp_mismatch and _normalize_cid_base(folder_name) == _normalize_cid_base(cid)):
            mismatches.append("candidate_id")

    meta = _load_artifact_metadata(artifact_folder)
    meta_id = str(meta.get("candidate_id") or "")
    if meta_id and meta_id != cid:
        if not (allow_timestamp_mismatch and _normalize_cid_base(meta_id) == _normalize_cid_base(cid)):
            if "candidate_id" not in mismatches:
                mismatches.append("candidate_id")

    cm = _norm(candidate.get("model_name")) or _infer_model_family(cid)
    am = _norm(meta.get("model_name")) or _infer_model_family(folder_name)
    if cm and am and cm != am:
        mismatches.append("model_family")

    cp = _norm(candidate.get("preset_family"))
    ap = _norm(meta.get("preset_family"))
    if cp and ap and cp != ap:
        mismatches.append("preset_family")
    elif cp and not ap and folder_name != cid:
        if cp not in _norm(folder_name):
            mismatches.append("preset_family")

    cs = _norm_side(candidate.get("side_policy") or candidate.get("side"))
    ms = _norm_side(meta.get("side_policy"))
    if cs and ms and cs != ms:
        mismatches.append("side_policy")
    elif cs and folder_name != cid:
        side_tokens = {
            "PE_ONLY": ("pe_only", "peonly"),
            "CE_ONLY": ("ce_only", "ceonly"),
            "BOTH": ("both",),
            "AUTO_DIRECTIONAL": ("directional", "auto_directional"),
            "VOLATILITY_BREAKOUT": ("volatility_breakout", "vol_breakout"),
            "COST_SURVIVOR": ("cost_survivor", "cost_surv_bal"),
        }
        tokens = side_tokens.get(cs, (cs.lower(),))
        if not any(tok in _norm(folder_name) for tok in tokens):
            mismatches.append("side_policy")

    ct = _candidate_t_tag(cid)
    at = _candidate_t_tag(folder_name) or str(meta.get("t_tag") or "")
    if ct and at and ct != at:
        if candidate.get("shared_artifact_ok") is not True:
            mismatches.append("threshold_suffix")

    return len(mismatches) == 0, mismatches


def _normalize_cid_base(cid: str) -> str:
    if not cid:
        return ""
    s = re.sub(r"_\d{6}$", "", cid)
    s = re.sub(r"(\d{8})_\d{0,6}$", r"\1", s)
    match = re.match(r"^(.+_t\d+_\d{8})", s)
    if match:
        return match.group(1)
    return re.sub(r"_\d{8}_\d{6}$", "", cid)


def _validate_configured_path(
    candidate: Dict[str, Any],
    configured_path: Path,
    project_root: Path,
    *,
    allow_timestamp_mismatch: bool,
) -> Tuple[Optional[Path], List[str]]:
    folder = _artifact_folder_for_path(configured_path, project_root)
    if folder is None:
        return None, ["configured_path"]
    ok, mismatches = validate_artifact_identity(
        candidate,
        folder,
        allow_timestamp_mismatch=allow_timestamp_mismatch,
    )
    if not ok:
        return folder, mismatches
    model_path = folder / MODEL_FILENAME
    if not model_path.exists():
        return folder, ["model_file"]
    return folder, []


def resolve_candidate_artifact(
    candidate: Dict[str, Any],
    project_root: Path | str,
    *,
    strict: bool = True,
    allow_fallback: bool = False,
) -> CandidateArtifactResolution:
    """Resolve exactly one artifact path for a candidate.

    In strict mode (default):
      - only artifacts/candidates/<candidate_id>/model.pkl is selected
      - configured paths are validated but never substituted with another candidate
      - missing exact artifact => MISSING / ARTIFACT_NOT_FOUND
      - identity mismatch => MISMATCH / ARTIFACT_IDENTITY_MISMATCH
    """
    root = Path(project_root).resolve()
    cid = str(candidate.get("candidate_id") or candidate.get("id") or "unknown")
    allow_timestamp_mismatch = bool(
        candidate.get("allow_timestamp_mismatch")
        or candidate.get("shared_artifact_ok")
    )

    expected_dir, expected_model = expected_artifact_paths(cid, root)
    expected_rel = _repo_rel(expected_model, root)

    configured_values = list(_iter_configured_paths(candidate))
    configured_rel = configured_values[0].replace("\\", "/") if configured_values else None

    resolution = CandidateArtifactResolution(
        candidate_id=cid,
        expected_artifact_path=expected_rel,
        configured_artifact_path=configured_rel,
        strict=bool(strict),
        allow_fallback=bool(allow_fallback),
    )

    configured_mismatches: List[str] = []
    configured_folder: Optional[Path] = None
    for raw in configured_values:
        path = _resolve_repo_path(raw, root)
        if path is None or not path.exists():
            continue
        folder, mismatches = _validate_configured_path(
            candidate,
            path,
            root,
            allow_timestamp_mismatch=allow_timestamp_mismatch,
        )
        if folder is not None and not mismatches:
            configured_folder = folder
            break
        if mismatches:
            configured_mismatches = mismatches
            configured_folder = folder

    selected_folder: Optional[Path] = None
    selected_model: Optional[Path] = None
    mismatch_fields: List[str] = []

    if strict and configured_mismatches:
        mismatch_fields = configured_mismatches
        resolution.artifact_identity_status = "MISMATCH"
        resolution.disable_reason = "ARTIFACT_IDENTITY_MISMATCH"
    elif expected_model.exists():
        ok, mismatches = validate_artifact_identity(
            candidate,
            expected_dir,
            allow_timestamp_mismatch=allow_timestamp_mismatch,
        )
        if ok:
            selected_folder = expected_dir
            selected_model = expected_model
            resolution.artifact_identity_status = "EXACT_MATCH"
        else:
            mismatch_fields = mismatches
            resolution.artifact_identity_status = "MISMATCH"
            resolution.disable_reason = "ARTIFACT_IDENTITY_MISMATCH"
    elif strict and not allow_fallback:
        if configured_folder is not None and configured_mismatches:
            mismatch_fields = configured_mismatches
            resolution.artifact_identity_status = "MISMATCH"
            resolution.disable_reason = "ARTIFACT_IDENTITY_MISMATCH"
        else:
            resolution.artifact_identity_status = "MISSING"
            resolution.disable_reason = "ARTIFACT_NOT_FOUND"
    elif allow_fallback and configured_folder is not None:
        model_path = configured_folder / MODEL_FILENAME
        if model_path.exists() and not configured_mismatches:
            selected_folder = configured_folder
            selected_model = model_path
            if configured_folder.name == cid:
                resolution.artifact_identity_status = "EXACT_MATCH"
            else:
                resolution.artifact_identity_status = "MISMATCH"
                resolution.disable_reason = "ARTIFACT_IDENTITY_MISMATCH"
                mismatch_fields = configured_mismatches or ["candidate_id"]
        elif configured_mismatches:
            mismatch_fields = configured_mismatches
            resolution.artifact_identity_status = "MISMATCH"
            resolution.disable_reason = "ARTIFACT_IDENTITY_MISMATCH"
        else:
            resolution.artifact_identity_status = "MISSING"
            resolution.disable_reason = "ARTIFACT_NOT_FOUND"
    else:
        if configured_mismatches:
            mismatch_fields = configured_mismatches
            resolution.artifact_identity_status = "MISMATCH"
            resolution.disable_reason = "ARTIFACT_IDENTITY_MISMATCH"
        else:
            resolution.artifact_identity_status = "MISSING"
            resolution.disable_reason = "ARTIFACT_NOT_FOUND"

    resolution.mismatch_fields = mismatch_fields
    if selected_model is not None and resolution.artifact_identity_status == "EXACT_MATCH":
        resolution.selected_artifact_path = _repo_rel(selected_model, root)
        resolution.resolved_dir = _repo_rel(selected_folder, root) if selected_folder else None
        resolution.model_file = str(selected_model)
        resolution.loaded = True
        for fname in _IDENTITY_SIDECARS:
            fp = (selected_folder or expected_dir) / fname
            if fp.exists():
                resolution.metadata_file = str(fp)
                break
    else:
        resolution.loaded = False
        resolution.selected_artifact_path = None
        resolution.resolved_dir = None
        resolution.model_file = None

    return resolution


def log_artifact_resolution(resolution: CandidateArtifactResolution | Dict[str, Any], *, prefix: str = "[ARTIFACT]") -> None:
    if isinstance(resolution, CandidateArtifactResolution):
        payload = resolution
    else:
        payload = CandidateArtifactResolution(
            candidate_id=str(resolution.get("candidate_id") or ""),
            expected_artifact_path=str(resolution.get("expected_artifact_path") or ""),
            configured_artifact_path=resolution.get("configured_artifact_path"),
            selected_artifact_path=resolution.get("selected_artifact_path"),
            artifact_identity_status=str(resolution.get("artifact_identity_status") or "MISSING"),
            mismatch_fields=list(resolution.get("mismatch_fields") or []),
            loaded=bool(resolution.get("loaded")),
            disable_reason=resolution.get("disable_reason"),
        )
    print(
        f"{prefix} candidate_id={payload.candidate_id} "
        f"expected_artifact_path={payload.expected_artifact_path} "
        f"configured_artifact_path={payload.configured_artifact_path or ''} "
        f"selected_artifact_path={payload.selected_artifact_path or ''} "
        f"artifact_identity_status={payload.artifact_identity_status} "
        f"mismatch_fields={payload.mismatch_fields} "
        f"loaded={str(payload.loaded).lower()}"
    )


def resolve_candidate_artifacts(
    candidate: Dict[str, Any],
    project_root: Path | str,
    *,
    strict: bool = True,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Backward-compatible wrapper used by paper_forward_engine."""
    resolution = resolve_candidate_artifact(
        candidate,
        project_root,
        strict=strict,
        allow_fallback=allow_fallback,
    )
    log_artifact_resolution(resolution)
    return resolution.to_legacy_dict()
