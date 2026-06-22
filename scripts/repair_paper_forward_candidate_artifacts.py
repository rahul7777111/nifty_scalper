#!/usr/bin/env python3
"""Strictly repair Paper Forward candidate artifact paths.

This script intentionally fails closed. It never maps by model family alone,
never picks a latest artifact, and never borrows across model/preset/threshold
identities unless compatibility is explicitly declared in artifact metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "paper_forward_candidates.json"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "candidates"
SIDECARS = (
    "manifest.json",
    "metadata.json",
    "model_card.json",
    "candidate_config.json",
    "training_metadata.json",
    "candidate_manifest.json",
    "paper_forward_manifest.json",
    "candidate_profile.json",
    "shadow_manifest.json",
    "feature_schema.json",
    "preprocessing_metadata.json",
)


@dataclass
class ArtifactInfo:
    folder: Path
    candidate_id: str = ""
    artifact_id: str = ""
    model_family: str = ""
    preset_family: str = ""
    side_policy: str = ""
    threshold: Any = None
    feature_order_len: int = 0
    feature_order_source: str = ""
    model_path: Path | None = None
    scaler_path: Path | None = None
    calibrator_path: Path | None = None
    compatible_candidate_ids: set[str] = field(default_factory=set)
    shared_artifact_ok: bool = False
    metadata_files: list[Path] = field(default_factory=list)


def repo_rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("/", os.sep)
    except Exception:
        return str(path)


def configure_repo_root(path: str | Path) -> None:
    global REPO_ROOT, CONFIG_PATH, ARTIFACT_ROOT
    REPO_ROOT = Path(path).resolve()
    CONFIG_PATH = REPO_ROOT / "config" / "paper_forward_candidates.json"
    ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "candidates"


def resolve_repo_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def stale_absolute_root(value: Any) -> bool:
    path = resolve_repo_path(value)
    if path is None or not Path(str(value or "")).is_absolute():
        return False
    try:
        path.resolve().relative_to(REPO_ROOT)
        return False
    except Exception:
        return True


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def extract_threshold(payload: dict[str, Any]) -> Any:
    for key in ("threshold", "selected_threshold", "entry_threshold", "effective_threshold"):
        if payload.get(key) not in (None, ""):
            return payload.get(key)
    policy = payload.get("threshold_policy")
    if isinstance(policy, dict):
        return first_nonempty(policy.get("entry_threshold"), policy.get("threshold")) or None
    return None


def threshold_key(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def extract_feature_len(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return 0
    for key in ("features", "feature_order", "live_computable_features", "model_features"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    schema = payload.get("feature_schema")
    if isinstance(schema, list):
        return len(schema)
    if isinstance(schema, dict):
        return extract_feature_len(schema)
    return 0


def infer_model_from_name(name: str) -> str:
    low = name.lower()
    if "calibrated_logistic_regression" in low:
        return "calibrated_logistic_regression"
    if "logistic_regression" in low:
        return "logistic_regression"
    if "xgboost" in low:
        return "xgboost"
    if "elasticnet" in low:
        return "elasticnet"
    return ""


def candidate_t_tag(value: str) -> str:
    match = re.search(r"(?:^|_)t(\d+)(?:_|$)", value or "")
    return match.group(1) if match else ""


def t_tag_threshold(value: str) -> str:
    tag = candidate_t_tag(value)
    if not tag:
        return ""
    try:
        return f"{float(tag) / 100.0:.4f}"
    except Exception:
        return f"t{tag}"


def safe_model_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        head = path.read_bytes()[:1]
    except Exception:
        return False
    return head not in (b"{", b"[") and path.stat().st_size > 512


def artifact_search_dirs(folder: Path) -> list[Path]:
    dirs = [folder]
    nested = folder / folder.name
    if nested.is_dir():
        dirs.append(nested)
    return dirs


def read_artifact(folder: Path) -> ArtifactInfo:
    info = ArtifactInfo(folder=folder)
    info.artifact_id = folder.name
    for base in artifact_search_dirs(folder):
        for sidecar in SIDECARS:
            path = base / sidecar
            if not path.exists():
                continue
            payload = load_json(path)
            if payload is None:
                continue
            info.metadata_files.append(path)
            if isinstance(payload, dict):
                info.candidate_id = first_nonempty(
                    info.candidate_id,
                    payload.get("candidate_id"),
                    payload.get("candidate"),
                    payload.get("id"),
                )
                info.artifact_id = first_nonempty(
                    info.artifact_id,
                    payload.get("artifact_id"),
                    payload.get("model_id"),
                    payload.get("id"),
                    folder.name,
                )
                info.model_family = first_nonempty(
                    info.model_family,
                    payload.get("model_family"),
                    payload.get("model_name"),
                    payload.get("model"),
                )
                info.preset_family = first_nonempty(
                    info.preset_family,
                    payload.get("preset_family"),
                    payload.get("preset"),
                    payload.get("selected_preset"),
                )
                info.side_policy = first_nonempty(
                    info.side_policy,
                    payload.get("side_policy"),
                    payload.get("side"),
                    payload.get("filter_name"),
                )
                if info.threshold in (None, ""):
                    info.threshold = extract_threshold(payload)
                length = extract_feature_len(payload)
                if length > info.feature_order_len:
                    info.feature_order_len = length
                    info.feature_order_source = repo_rel(path)
                for key in ("compatible_candidate_ids", "compatible_candidates", "aliases"):
                    values = payload.get(key)
                    if isinstance(values, list):
                        info.compatible_candidate_ids.update(str(v) for v in values if v)
                if payload.get("shared_artifact_ok") is True:
                    info.shared_artifact_ok = True
                for key in ("model_path", "model_pkl", "model_file"):
                    if payload.get(key) and info.model_path is None:
                        candidate = base / str(payload[key])
                        if safe_model_file(candidate):
                            info.model_path = candidate
                for key in ("scaler_path", "scaler_file"):
                    if payload.get(key) and info.scaler_path is None:
                        candidate = base / str(payload[key])
                        if candidate.exists():
                            info.scaler_path = candidate
                for key in ("calibrator_path", "calibrator_file"):
                    if payload.get(key) and info.calibrator_path is None:
                        candidate = base / str(payload[key])
                        if candidate.exists():
                            info.calibrator_path = candidate
    if not info.model_family:
        info.model_family = infer_model_from_name(folder.name)
    if not info.candidate_id:
        info.candidate_id = folder.name
    if info.model_path is None:
        for base in artifact_search_dirs(folder):
            for path in sorted(base.glob("*.pkl")):
                if "metric" in path.name.lower():
                    continue
                if safe_model_file(path):
                    info.model_path = path
                    break
            if info.model_path:
                break
    return info


def build_artifact_index() -> list[ArtifactInfo]:
    if not ARTIFACT_ROOT.exists():
        return []
    return [read_artifact(path) for path in sorted(ARTIFACT_ROOT.iterdir()) if path.is_dir()]


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def norm_side(value: Any) -> str:
    v = str(value or "").upper().replace(" ", "").replace("-", "_")
    if v == "PE":
        return "PE_ONLY"
    if v == "CE":
        return "CE_ONLY"
    if v == "BOTH":
        return "BOTH"
    return v


def threshold_matches(candidate: dict[str, Any], artifact: ArtifactInfo) -> tuple[bool, str]:
    ct = candidate_t_tag(str(candidate.get("candidate_id") or ""))
    at = candidate_t_tag(artifact.folder.name) or candidate_t_tag(artifact.candidate_id) or candidate_t_tag(artifact.artifact_id)
    if ct and at and ct != at:
        if candidate.get("shared_artifact_ok") is True or artifact.shared_artifact_ok:
            return True, "threshold tag differs but shared_artifact_ok=true"
        return False, f"t{ct} cannot use t{at}"
    return True, ""


def identity_matches(candidate: dict[str, Any], artifact: ArtifactInfo) -> tuple[bool, str]:
    reasons: list[str] = []
    cm = norm(candidate.get("model_name"))
    am = norm(artifact.model_family)
    if cm and am and cm != am:
        reasons.append(f"model {cm}!={am}")
    cp = norm(candidate.get("preset_family"))
    ap = norm(artifact.preset_family)
    if cp and ap and cp != ap:
        reasons.append(f"preset {cp}!={ap}")
    cs = norm_side(candidate.get("side_policy"))
    as_ = norm_side(artifact.side_policy)
    if cs and as_ and cs != as_:
        reasons.append(f"side {cs}!={as_}")
    ok_threshold, threshold_reason = threshold_matches(candidate, artifact)
    if not ok_threshold:
        reasons.append(threshold_reason)
    if artifact.model_path is None:
        reasons.append("model_path missing or unsafe")
    if artifact.feature_order_len <= 0:
        reasons.append("feature_order length 0")
    return not reasons, "; ".join(reasons)


def declared_compatible(candidate_id: str, artifact: ArtifactInfo) -> bool:
    return candidate_id in artifact.compatible_candidate_ids


def find_match(candidate: dict[str, Any], artifacts: list[ArtifactInfo]) -> tuple[ArtifactInfo | None, str, str]:
    cid = str(candidate.get("candidate_id") or "")
    artifact_id = str(candidate.get("artifact_id") or "")
    by_candidate = {a.candidate_id: a for a in artifacts if a.candidate_id}
    by_artifact = {a.artifact_id: a for a in artifacts if a.artifact_id}
    by_folder = {a.folder.name: a for a in artifacts}
    checks: list[tuple[str, ArtifactInfo | None, bool]] = [
        ("candidate_id", by_candidate.get(cid), True),
        ("artifact_id", by_artifact.get(cid) or (by_artifact.get(artifact_id) if artifact_id else None), True),
        ("folder_basename", by_folder.get(cid), True),
    ]
    for rule, artifact, require_identity_id in checks:
        if not artifact:
            continue
        if require_identity_id and rule != "artifact_id" and artifact.candidate_id != cid and artifact.artifact_id != cid and artifact.folder.name != cid:
            continue
        ok, reason = identity_matches(candidate, artifact)
        if ok:
            return artifact, "MATCHED_" + rule.upper(), reason or "exact identity"
        return None, "NOT_FOUND", f"{rule} rejected: {reason}"
    for artifact in artifacts:
        if declared_compatible(cid, artifact):
            ok, reason = identity_matches(candidate, artifact)
            if ok:
                return artifact, "MATCHED_DECLARED_COMPATIBILITY", reason or "declared compatibility"
            return None, "NOT_FOUND", f"declared compatibility rejected: {reason}"
    return None, "NOT_FOUND", "no exact candidate_id/artifact_id/folder/declared compatibility match"


def load_config() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(CONFIG_PATH)
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid config JSON: {CONFIG_PATH}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("paper_forward_candidates.json must contain a candidates list")
    return payload, candidates


def candidate_path_report(candidate: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = str(candidate.get("artifact_dir") or "")
    model_path = str(candidate.get("model_path") or "")
    artifact_path = resolve_repo_path(artifact_dir)
    model = resolve_repo_path(model_path)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "enabled": candidate.get("enabled"),
        "model": candidate.get("model_name"),
        "preset": candidate.get("preset_family"),
        "side": candidate.get("side_policy"),
        "threshold": candidate.get("threshold") or candidate.get("selected_threshold"),
        "artifact_id": candidate.get("artifact_id"),
        "artifact_dir": artifact_dir or None,
        "stale_artifact_root": stale_absolute_root(artifact_dir),
        "artifact_dir_exists": bool(artifact_path and artifact_path.exists()),
        "model_path": model_path or None,
        "model_path_exists": bool(model and model.exists()),
    }


def candidate_threshold(candidate: dict[str, Any], artifact: ArtifactInfo | None = None) -> str:
    explicit = candidate.get("threshold") or candidate.get("selected_threshold")
    if explicit not in (None, ""):
        return threshold_key(explicit)
    if artifact and artifact.threshold not in (None, ""):
        return threshold_key(artifact.threshold)
    return t_tag_threshold(str(candidate.get("candidate_id") or "")) or t_tag_threshold(str(candidate.get("artifact_id") or ""))


def logical_key(candidate: dict[str, Any], artifact: ArtifactInfo | None = None, *, include_threshold: bool = True) -> tuple[str, ...]:
    base = (
        norm(candidate.get("model_name") or (artifact.model_family if artifact else "")),
        norm(candidate.get("preset_family") or (artifact.preset_family if artifact else "")),
        norm_side(candidate.get("side_policy") or (artifact.side_policy if artifact else "")),
    )
    if include_threshold:
        return (*base, candidate_threshold(candidate, artifact))
    return base


def artifact_is_valid(artifact: ArtifactInfo | None) -> bool:
    return bool(artifact and artifact.model_path and artifact.feature_order_len > 0)


def print_candidate_inspection(candidates: list[dict[str, Any]]) -> None:
    print("CONFIG CANDIDATES")
    print("candidate_id | enabled | model | preset | side | threshold | artifact_id | artifact_dir | stale_artifact_root | artifact_dir_exists | model_path | model_path_exists")
    print("-" * 220)
    for candidate in candidates:
        row = candidate_path_report(candidate)
        print(" | ".join(str(row[k]) for k in row))


def print_artifact_index(artifacts: list[ArtifactInfo]) -> None:
    print("\nARTIFACT INDEX")
    print("folder | candidate_id | artifact_id | model | preset | side | threshold | feature_order_len | model_path | scaler_path | calibrator_path")
    print("-" * 220)
    for artifact in artifacts:
        print(
            " | ".join(
                str(x)
                for x in (
                    artifact.folder.name,
                    artifact.candidate_id,
                    artifact.artifact_id,
                    artifact.model_family,
                    artifact.preset_family,
                    artifact.side_policy,
                    artifact.threshold,
                    artifact.feature_order_len,
                    repo_rel(artifact.model_path) or None,
                    repo_rel(artifact.scaler_path) or None,
                    repo_rel(artifact.calibrator_path) or None,
                )
            )
        )


def apply_artifact(candidate: dict[str, Any], artifact: ArtifactInfo) -> None:
    candidate["candidate_id"] = artifact.candidate_id
    candidate["artifact_id"] = artifact.artifact_id or artifact.folder.name
    candidate["artifact_dir"] = repo_rel(artifact.folder)
    candidate["model_path"] = repo_rel(artifact.model_path)
    candidate["feature_order_source"] = artifact.feature_order_source
    candidate["artifact_identity_verified"] = True
    candidate["enabled"] = True
    candidate.pop("disabled_reason", None)


def disable_candidate(candidate: dict[str, Any], reason: str = "ARTIFACT_NOT_FOUND") -> None:
    candidate["enabled"] = False
    candidate["disabled_reason"] = reason
    candidate["artifact_identity_verified"] = False
    candidate.pop("model_path", None)
    candidate.pop("feature_order_source", None)
    candidate.pop("artifact_id", None)


def dedupe_candidates(candidates: list[dict[str, Any]], artifacts: list[ArtifactInfo]) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    artifact_by_dir = {repo_rel(a.folder): a for a in artifacts}
    artifact_by_dir.update({str(a.folder): a for a in artifacts})

    def row_artifact(candidate: dict[str, Any]) -> ArtifactInfo | None:
        ad = str(candidate.get("artifact_dir") or "")
        return artifact_by_dir.get(ad) or artifact_by_dir.get(ad.replace("/", os.sep)) or artifact_by_dir.get(ad.replace("\\", os.sep))

    removals: list[tuple[str, str, str]] = []
    keep_indexes: set[int] = set(range(len(candidates)))

    id_seen: dict[str, int] = {}
    for idx, candidate in enumerate(candidates):
        cid = str(candidate.get("candidate_id") or "")
        if not cid:
            continue
        if cid not in id_seen:
            id_seen[cid] = idx
            continue
        prev = id_seen[cid]
        prev_art = row_artifact(candidates[prev])
        cur_art = row_artifact(candidate)
        drop = idx
        if artifact_is_valid(cur_art) and not artifact_is_valid(prev_art):
            drop = prev
            id_seen[cid] = idx
        candidates[drop]["disabled_reason"] = "DUPLICATE_STALE_ARTIFACT"
        removals.append((cid, "DUPLICATE_STALE_ARTIFACT", "duplicate candidate_id"))
        keep_indexes.discard(drop)

    by_strategy: dict[tuple[str, ...], list[int]] = {}
    for idx, candidate in enumerate(candidates):
        if idx not in keep_indexes:
            continue
        artifact = row_artifact(candidate)
        key = logical_key(candidate, artifact, include_threshold=True)
        by_strategy.setdefault(key, []).append(idx)

    for key, indexes in by_strategy.items():
        if len(indexes) <= 1:
            continue
        best = sorted(
            indexes,
            key=lambda i: (
                artifact_is_valid(row_artifact(candidates[i])),
                bool(candidates[i].get("enabled")),
                -i,
            ),
            reverse=True,
        )[0]
        for idx in indexes:
            if idx == best:
                continue
            candidates[idx]["enabled"] = False
            candidates[idx]["disabled_reason"] = "DUPLICATE_STALE_ARTIFACT"
            removals.append((str(candidates[idx].get("candidate_id") or ""), "DUPLICATE_STALE_ARTIFACT", "duplicate logical strategy"))
            keep_indexes.discard(idx)

    by_family: dict[tuple[str, ...], list[int]] = {}
    for idx, candidate in enumerate(candidates):
        if idx not in keep_indexes or not candidate.get("enabled", True):
            continue
        artifact = row_artifact(candidate)
        key = logical_key(candidate, artifact, include_threshold=False)
        by_family.setdefault(key, []).append(idx)

    for key, indexes in by_family.items():
        if len(indexes) <= 1:
            continue
        if any(candidates[i].get("allow_multiple_threshold_variants") is True for i in indexes):
            continue
        def rank(i: int) -> tuple[int, int, int]:
            threshold = candidate_threshold(candidates[i], row_artifact(candidates[i]))
            is_t30 = 1 if threshold in ("0.3000", "t30") or "_t30_" in str(candidates[i].get("candidate_id") or "") else 0
            valid = 1 if artifact_is_valid(row_artifact(candidates[i])) else 0
            return (is_t30, valid, -i)
        best = sorted(indexes, key=rank, reverse=True)[0]
        for idx in indexes:
            if idx == best:
                continue
            candidates[idx]["enabled"] = False
            candidates[idx]["disabled_reason"] = "DUPLICATE_THRESHOLD_VARIANT"
            removals.append((str(candidates[idx].get("candidate_id") or ""), "DUPLICATE_THRESHOLD_VARIANT", "threshold variant not explicitly allowed"))
            keep_indexes.discard(idx)

    return [c for i, c in enumerate(candidates) if i in keep_indexes], removals


def repair_config(
    apply: bool,
    *,
    add_new_artifacts: bool = False,
    dedupe: bool = False,
    drop_invalid: bool = False,
) -> int:
    config, candidates = load_config()
    original_count = len(candidates)
    artifacts = build_artifact_index()
    stale_root_count = sum(1 for c in candidates if stale_absolute_root(c.get("artifact_dir")))
    print_candidate_inspection(candidates)
    print_artifact_index(artifacts)
    print(
        f"[ARTIFACT-ROOT] repo_root={REPO_ROOT} stale_root_detected={stale_root_count} "
        f"converted=0 found={len(artifacts)} missing=0"
    )

    rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    for candidate in candidates:
        old_id = str(candidate.get("candidate_id") or "")
        old_artifact = str(candidate.get("artifact_dir") or candidate.get("artifact_id") or "-")
        artifact, status, reason = find_match(candidate, artifacts)
        if artifact:
            apply_artifact(candidate, artifact)
            new_artifact = repo_rel(artifact.folder)
            feature_len = str(artifact.feature_order_len)
        else:
            disable_candidate(candidate)
            new_artifact = "-"
            feature_len = "0"
        rows.append((
            old_id,
            str(candidate.get("model_name") or ""),
            str(candidate.get("preset_family") or ""),
            old_artifact,
            new_artifact,
            feature_len,
            status,
            reason,
        ))

    matched_ok_count = sum(1 for row in rows if row[6].startswith("MATCHED"))
    path_converted_count = sum(1 for row in rows if row[6].startswith("MATCHED") and row[3] != row[4])
    already_correct_count = sum(1 for row in rows if row[6].startswith("MATCHED") and row[3] == row[4])
    repaired_count = path_converted_count

    if add_new_artifacts:
        seen_candidate_ids = {str(c.get("candidate_id") or "") for c in candidates}
        for artifact in artifacts:
            if artifact.candidate_id in seen_candidate_ids:
                continue
            if artifact.model_path is None or artifact.feature_order_len <= 0:
                continue
            new_candidate = {
                "candidate_id": artifact.candidate_id,
                "enabled": True,
                "classification": "paper_forward_only",
                "model_name": artifact.model_family,
                "preset_family": artifact.preset_family,
                "side_policy": artifact.side_policy,
                "paper_forward_only": True,
                "shadow_ready": False,
                "notes": "added by strict artifact repair from exact materialized artifact",
            }
            apply_artifact(new_candidate, artifact)
            candidates.append(new_candidate)
            seen_candidate_ids.add(artifact.candidate_id)
            rows.append((
                artifact.candidate_id,
                artifact.model_family,
                artifact.preset_family,
                "-",
                repo_rel(artifact.folder),
                str(artifact.feature_order_len),
                "ADDED_EXACT_ARTIFACT",
                "real materialized artifact not present in config",
            ))

    invalid_drop_rows: list[tuple[str, str]] = []
    if drop_invalid:
        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            invalid_artifact = (
                candidate.get("enabled") is False
                and str(candidate.get("disabled_reason") or "").upper() == "ARTIFACT_NOT_FOUND"
                and not candidate.get("artifact_identity_verified")
            )
            if invalid_artifact:
                invalid_drop_rows.append((
                    str(candidate.get("candidate_id") or ""),
                    "invalid artifact row removed from active paper-forward config",
                ))
                continue
            kept.append(candidate)
        candidates[:] = kept
        config["candidates"] = candidates

    dedupe_rows: list[tuple[str, str, str]] = []
    if dedupe:
        new_candidates, dedupe_rows = dedupe_candidates(candidates, artifacts)
        candidates[:] = new_candidates
        config["candidates"] = candidates

    print("\ncandidate_id | model | preset | old_artifact | new_artifact | feature_order_len | status | reason")
    print("-" * 240)
    for row in rows:
        print(" | ".join(row))
    if dedupe_rows:
        print("\nremoved_candidate_id | disabled_reason | reason")
        print("-" * 160)
        for row in dedupe_rows:
            print(" | ".join(row))
    if invalid_drop_rows:
        print("\ndropped_invalid_candidate_id | reason")
        print("-" * 160)
        for row in invalid_drop_rows:
            print(" | ".join(row))

    if apply:
        disabled_missing_count = sum(
            1
            for candidate in candidates
            if not candidate.get("enabled", True)
            and str(candidate.get("disabled_reason") or "").upper() == "ARTIFACT_NOT_FOUND"
        )
        active_enabled_count = sum(1 for candidate in candidates if candidate.get("enabled", True))
        missing_disabled_count = disabled_missing_count
        config["repair_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "pre_repair_candidate_count": original_count,
            "post_repair_candidate_count": len(candidates),
            "add_new_artifacts": bool(add_new_artifacts),
            "dedupe": bool(dedupe),
            "drop_invalid": bool(drop_invalid),
            "repaired_count": repaired_count,
            "matched_ok_count": matched_ok_count,
            "path_converted_count": path_converted_count,
            "missing_disabled_count": missing_disabled_count,
            "already_correct_count": already_correct_count,
            "disabled_missing_count": disabled_missing_count,
            "active_enabled_count": active_enabled_count,
            "removed_invalid_count": len(invalid_drop_rows),
            "removed_duplicate_count": len(dedupe_rows),
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = CONFIG_PATH.with_name(f"paper_forward_candidates.backup_{timestamp}.json")
        shutil.copy2(CONFIG_PATH, backup)
        CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"\nBackup written: {repo_rel(backup)}")
        print(f"Applied repaired config: {repo_rel(CONFIG_PATH)}")
    else:
        print("\nDRY-RUN: no config written")
    disabled_missing_count = sum(
        1
        for candidate in candidates
        if not candidate.get("enabled", True)
        and str(candidate.get("disabled_reason") or "").upper() == "ARTIFACT_NOT_FOUND"
    )
    active_enabled_count = sum(1 for candidate in candidates if candidate.get("enabled", True))
    missing_count = disabled_missing_count
    print(
        f"[ARTIFACT-ROOT] repo_root={REPO_ROOT} stale_root_detected={stale_root_count} "
        f"converted={path_converted_count} found={matched_ok_count} missing={missing_count}"
    )
    print(
        "\nREPAIR REPORT "
        f"repaired_count={repaired_count} "
        f"matched_ok_count={matched_ok_count} "
        f"path_converted_count={path_converted_count} "
        f"missing_disabled_count={missing_count} "
        f"already_correct_count={already_correct_count} "
        f"disabled_missing_count={disabled_missing_count} "
        f"duplicate_removed_count={len(dedupe_rows)} "
        f"active_enabled_count={active_enabled_count}"
    )
    if not add_new_artifacts and len(candidates) > original_count:
        print(f"ERROR: candidate count grew from {original_count} to {len(candidates)} without --add-new-artifacts")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    parser.add_argument("--add-new-artifacts", action="store_true", help="Allow adding exact materialized artifacts not present in config.")
    parser.add_argument("--no-add-new-artifacts", action="store_true", help="Preserve/trim candidate list; this is the default.")
    parser.add_argument("--dedupe", action="store_true", help="Remove stale duplicate rows and unapproved threshold variants.")
    parser.add_argument("--drop-invalid", action="store_true", help="Remove rows that still resolve to invalid/missing artifacts after repair.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Project root containing config/ and artifacts/.")
    args = parser.parse_args(argv)
    configure_repo_root(args.repo_root)
    return repair_config(
        apply=bool(args.apply),
        add_new_artifacts=bool(args.add_new_artifacts),
        dedupe=bool(args.dedupe),
        drop_invalid=bool(args.drop_invalid),
    )


if __name__ == "__main__":
    raise SystemExit(main())
