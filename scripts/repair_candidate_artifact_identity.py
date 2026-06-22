#!/usr/bin/env python3
"""Repair paper-forward candidate artifact paths to exact identity-safe mappings."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_artifact_resolver import expected_artifact_paths, resolve_candidate_artifact, validate_artifact_identity  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "paper_forward_candidates.json"


def repo_rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def load_config(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit(f"Invalid candidates list in {config_path}")
    return payload, candidates


def repair_candidate(candidate: dict[str, Any], project_root: Path) -> dict[str, Any]:
    cid = str(candidate.get("candidate_id") or "")
    _, expected_model = expected_artifact_paths(cid, project_root)
    expected_rel = repo_rel(expected_model, project_root)
    expected_dir_rel = repo_rel(expected_model.parent, project_root)

    resolution = resolve_candidate_artifact(candidate, project_root, strict=True, allow_fallback=False)
    changed_fields: list[str] = []

    for stale_key in ("artifact_path",):
        if stale_key in candidate:
            candidate.pop(stale_key)
            changed_fields.append(f"removed:{stale_key}")

    expected_identity_ok = False
    expected_mismatches: list[str] = []
    if expected_model.exists():
        expected_identity_ok, expected_mismatches = validate_artifact_identity(candidate, expected_model.parent)

    if expected_model.exists() and expected_identity_ok:
        if candidate.get("artifact_dir") != expected_dir_rel:
            candidate["artifact_dir"] = expected_dir_rel
            changed_fields.append("artifact_dir")
        if candidate.get("model_path") != expected_rel:
            candidate["model_path"] = expected_rel
            changed_fields.append("model_path")
        candidate["artifact_id"] = cid
        candidate["artifact_identity_verified"] = True
        candidate["enabled"] = True
        candidate.pop("disabled_reason", None)
        status = "REPAIRED_EXACT"
    else:
        candidate["enabled"] = False
        if expected_model.exists() and expected_mismatches:
            candidate["disabled_reason"] = "ARTIFACT_IDENTITY_MISMATCH"
        else:
            candidate["disabled_reason"] = resolution.disable_reason or "ARTIFACT_NOT_FOUND"
        candidate["artifact_identity_verified"] = False
        candidate.pop("model_path", None)
        candidate.pop("feature_order_source", None)
        if candidate.get("artifact_dir"):
            candidate.pop("artifact_dir", None)
            changed_fields.append("removed:artifact_dir")
        if candidate.get("artifact_id"):
            candidate.pop("artifact_id", None)
            changed_fields.append("removed:artifact_id")
        status = "DISABLED_MISSING" if not expected_model.exists() else "DISABLED_MISMATCH"

    return {
        "candidate_id": cid,
        "status": status,
        "expected_artifact_path": expected_rel,
        "selected_artifact_path": resolution.selected_artifact_path,
        "artifact_identity_status": resolution.artifact_identity_status,
        "changed_fields": changed_fields,
        "enabled": candidate.get("enabled"),
        "disabled_reason": candidate.get("disabled_reason"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair candidate artifact identity paths.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--apply", action="store_true", help="Write repaired config (creates backup first)")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    payload, candidates = load_config(config_path)
    rows = [repair_candidate(candidate, root) for candidate in candidates]

    print("candidate_id | status | enabled | expected | selected | changed_fields")
    print("-" * 180)
    for row in rows:
        print(
            f"{row['candidate_id']} | {row['status']} | {row['enabled']} | {row['expected_artifact_path']} | "
            f"{row['selected_artifact_path'] or '-'} | {row['changed_fields']}"
        )

    if args.apply:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = config_path.with_name(f"paper_forward_candidates.backup_{timestamp}.json")
        shutil.copy2(config_path, backup)
        payload["candidates"] = candidates
        payload["artifact_identity_repair"] = {
            "timestamp": datetime.now().isoformat(),
            "backup": repo_rel(backup, root),
            "repaired_rows": len(rows),
        }
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nBackup written: {backup}")
        print(f"Applied repaired config: {config_path}")
    else:
        print("\nDRY-RUN: no config written (pass --apply to persist)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
