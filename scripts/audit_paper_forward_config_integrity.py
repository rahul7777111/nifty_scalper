#!/usr/bin/env python3
"""Audit Paper Forward candidate config invariants."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_paper_forward_candidate_artifacts import (  # noqa: E402
    build_artifact_index,
    candidate_threshold,
    load_config,
    logical_key,
    repo_rel,
)


def resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def artifact_for_candidate(candidate: dict[str, Any], artifact_by_dir: dict[Path, Any]) -> Any:
    path = resolve_path(candidate.get("artifact_dir"))
    if path is None:
        return None
    return artifact_by_dir.get(path.resolve())


def main() -> int:
    config, candidates = load_config()
    artifacts = build_artifact_index()
    artifact_by_dir = {artifact.folder.resolve(): artifact for artifact in artifacts}
    errors: list[str] = []

    meta = config.get("repair_metadata") if isinstance(config.get("repair_metadata"), dict) else {}
    if meta:
        pre = int(meta.get("pre_repair_candidate_count") or len(candidates))
        post = int(meta.get("post_repair_candidate_count") or len(candidates))
        add_new = bool(meta.get("add_new_artifacts"))
        dedupe = bool(meta.get("dedupe"))
        if post > pre and not add_new:
            errors.append(f"candidate count grew from {pre} to {post} without --add-new-artifacts")
        if post != len(candidates):
            errors.append(f"repair_metadata post_repair_candidate_count={post} but actual={len(candidates)}")
        if post < pre and not dedupe:
            errors.append(f"candidate count decreased from {pre} to {post} without --dedupe")

    seen_ids: dict[str, int] = {}
    active_by_logical: dict[tuple[str, ...], list[str]] = {}
    active_by_family: dict[tuple[str, ...], list[str]] = {}

    for idx, candidate in enumerate(candidates, 1):
        cid = str(candidate.get("candidate_id") or "")
        if not cid:
            errors.append(f"row {idx}: missing candidate_id")
            continue
        if cid in seen_ids:
            errors.append(f"duplicate candidate_id: {cid} rows={seen_ids[cid]},{idx}")
        seen_ids[cid] = idx

        if not candidate.get("enabled", True):
            continue
        disabled_reason = str(candidate.get("disabled_reason") or "")
        if disabled_reason == "ARTIFACT_NOT_FOUND":
            errors.append(f"{cid}: enabled candidate has ARTIFACT_NOT_FOUND")
        artifact_dir = resolve_path(candidate.get("artifact_dir"))
        if artifact_dir is None or not artifact_dir.exists():
            errors.append(f"{cid}: enabled candidate has missing artifact_dir={candidate.get('artifact_dir')}")
            continue
        model_path = resolve_path(candidate.get("model_path"))
        if model_path is None or not model_path.exists():
            errors.append(f"{cid}: enabled candidate has missing model_path={candidate.get('model_path')}")
        artifact = artifact_for_candidate(candidate, artifact_by_dir)
        feature_len = int(getattr(artifact, "feature_order_len", 0) or 0)
        if feature_len == 0:
            errors.append(f"{cid}: enabled candidate has feature_order_len == 0")

        lk = logical_key(candidate, artifact, include_threshold=True)
        active_by_logical.setdefault(lk, []).append(cid)
        family_key = logical_key(candidate, artifact, include_threshold=False)
        active_by_family.setdefault(family_key, []).append(cid)

    for key, ids in active_by_logical.items():
        if len(ids) > 1:
            errors.append(f"duplicate active logical strategy {key}: {ids}")

    for key, ids in active_by_family.items():
        if len(ids) > 1:
            allowed = [
                c for c in candidates
                if c.get("candidate_id") in ids and c.get("allow_multiple_threshold_variants") is True
            ]
            if not allowed:
                thresholds = {
                    cid: candidate_threshold(next(c for c in candidates if c.get("candidate_id") == cid), artifact_for_candidate(next(c for c in candidates if c.get("candidate_id") == cid), artifact_by_dir))
                    for cid in ids
                }
                errors.append(f"multiple active threshold variants for {key} without allow_multiple_threshold_variants=true: {thresholds}")

    if errors:
        print("PAPER FORWARD CONFIG INTEGRITY FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PAPER FORWARD CONFIG INTEGRITY PASSED candidates={len(candidates)}")
    for candidate in candidates:
        if candidate.get("enabled", True):
            print(
                "ACTIVE "
                f"candidate_id={candidate.get('candidate_id')} "
                f"artifact_id={candidate.get('artifact_id')} "
                f"artifact_dir={Path(str(candidate.get('artifact_dir'))).name if candidate.get('artifact_dir') else None} "
                f"model_path={repo_rel(resolve_path(candidate.get('model_path')))}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
