#!/usr/bin/env python3
"""Audit Paper Forward candidate artifact mappings."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_paper_forward_candidate_artifacts import (  # noqa: E402
    REPO_ROOT,
    build_artifact_index,
    candidate_t_tag,
    identity_matches,
    load_config,
    norm,
    repo_rel,
)


def resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    _config, candidates = load_config()
    artifacts = build_artifact_index()
    by_dir = {artifact.folder.resolve(): artifact for artifact in artifacts}
    errors: list[str] = []

    for candidate in candidates:
        cid = str(candidate.get("candidate_id") or "")
        if not candidate.get("enabled", True):
            continue
        artifact_dir = resolve_path(candidate.get("artifact_dir"))
        model_path = resolve_path(candidate.get("model_path"))
        if artifact_dir is None:
            errors.append(f"{cid}: enabled candidate has no artifact_dir")
            continue
        if not artifact_dir.exists():
            errors.append(f"{cid}: artifact_dir does not exist: {repo_rel(artifact_dir)}")
            continue
        artifact = by_dir.get(artifact_dir.resolve())
        if artifact is None:
            errors.append(f"{cid}: artifact_dir is not indexed: {repo_rel(artifact_dir)}")
            continue
        if model_path is None:
            errors.append(f"{cid}: enabled candidate has no model_path")
        elif not model_path.exists():
            errors.append(f"{cid}: model_path does not exist: {repo_rel(model_path)}")
        if artifact.feature_order_len <= 0:
            errors.append(f"{cid}: feature_order length is 0")
        identity_ok = (
            artifact.candidate_id == cid
            or artifact.artifact_id == cid
            or artifact.folder.name == cid
            or cid in artifact.compatible_candidate_ids
        )
        if not identity_ok:
            errors.append(
                f"{cid}: artifact identity does not match candidate "
                f"(artifact_candidate_id={artifact.candidate_id}, artifact_id={artifact.artifact_id}, folder={artifact.folder.name})"
            )
        ok, reason = identity_matches(candidate, artifact)
        if not ok:
            errors.append(f"{cid}: artifact metadata mismatch: {reason}")

        cm = norm(candidate.get("model_name"))
        am = norm(artifact.model_family)
        cp = norm(candidate.get("preset_family"))
        ap = norm(artifact.preset_family)
        if "xgboost" in cm and "xgboost" not in am:
            errors.append(f"{cid}: xgboost resolves to non-xgboost artifact ({artifact.model_family})")
        if "calibrated" in cm and "calibrated" not in am:
            errors.append(f"{cid}: calibrated resolves to non-calibrated artifact ({artifact.model_family})")
        if "balanced" in cp and "conservative" in ap:
            errors.append(f"{cid}: balanced resolves to conservative artifact ({artifact.preset_family})")
        ct = candidate_t_tag(cid)
        at = candidate_t_tag(artifact.folder.name) or candidate_t_tag(artifact.candidate_id) or candidate_t_tag(artifact.artifact_id)
        if ct and at and ct != at and candidate.get("shared_artifact_ok") is not True and artifact.shared_artifact_ok is not True:
            errors.append(f"{cid}: t{ct} resolves to t{at} without shared_artifact_ok=true")

    if errors:
        print("PAPER FORWARD ARTIFACT AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PAPER FORWARD ARTIFACT AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
