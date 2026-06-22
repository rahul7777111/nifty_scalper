#!/usr/bin/env python3
"""Audit paper-forward candidate artifact identity against on-disk artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_artifact_resolver import resolve_candidate_artifact  # noqa: E402


def load_candidates(config_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit(f"Invalid candidates list in {config_path}")
    return candidates


def audit_candidates(
    candidates: list[dict[str, Any]],
    *,
    project_root: Path,
    strict: bool = True,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    exact_match_count = 0
    missing_artifact_count = 0
    identity_mismatch_count = 0
    disabled_count = 0
    failures: list[str] = []

    for candidate in candidates:
        cid = str(candidate.get("candidate_id") or "")
        enabled = bool(candidate.get("enabled", True))
        resolution = resolve_candidate_artifact(
            candidate,
            project_root,
            strict=strict,
            allow_fallback=allow_fallback,
        )
        row = {
            "candidate_id": cid,
            "enabled": enabled,
            "expected_artifact_path": resolution.expected_artifact_path,
            "configured_artifact_path": resolution.configured_artifact_path,
            "selected_artifact_path": resolution.selected_artifact_path,
            "artifact_identity_status": resolution.artifact_identity_status,
            "mismatch_fields": resolution.mismatch_fields,
            "loaded": resolution.loaded,
            "disable_reason": resolution.disable_reason,
        }
        reports.append(row)

        if resolution.artifact_identity_status == "EXACT_MATCH":
            exact_match_count += 1
        elif resolution.artifact_identity_status == "MISSING":
            missing_artifact_count += 1
        elif resolution.artifact_identity_status == "MISMATCH":
            identity_mismatch_count += 1

        if not enabled:
            disabled_count += 1
        elif not resolution.loaded or resolution.artifact_identity_status != "EXACT_MATCH":
            failures.append(
                f"{cid}: enabled candidate lacks exact identity-safe artifact "
                f"(status={resolution.artifact_identity_status}, reason={resolution.disable_reason})"
            )

    return {
        "total_candidates": len(candidates),
        "exact_match_count": exact_match_count,
        "missing_artifact_count": missing_artifact_count,
        "identity_mismatch_count": identity_mismatch_count,
        "disabled_count": disabled_count,
        "enabled_without_exact_match": len(failures),
        "per_candidate": reports,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit candidate artifact identity mappings.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "paper_forward_candidates.json"),
        help="Candidate config JSON path",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Project root")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--allow-artifact-fallback", action="store_true", default=False)
    parser.add_argument("--json-out", default=None, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    candidates = load_candidates(config_path)
    report = audit_candidates(
        candidates,
        project_root=root,
        strict=args.strict,
        allow_fallback=args.allow_artifact_fallback,
    )

    print("CANDIDATE ARTIFACT IDENTITY AUDIT")
    print(f"total_candidates={report['total_candidates']}")
    print(f"exact_match_count={report['exact_match_count']}")
    print(f"missing_artifact_count={report['missing_artifact_count']}")
    print(f"identity_mismatch_count={report['identity_mismatch_count']}")
    print(f"disabled_count={report['disabled_count']}")
    print(f"enabled_without_exact_match={report['enabled_without_exact_match']}")
    print("")
    print("candidate_id | enabled | status | loaded | expected | selected | mismatch_fields")
    print("-" * 180)
    for row in report["per_candidate"]:
        print(
            f"{row['candidate_id']} | {row['enabled']} | {row['artifact_identity_status']} | {row['loaded']} | "
            f"{row['expected_artifact_path']} | {row['selected_artifact_path'] or '-'} | {row['mismatch_fields']}"
        )

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON report: {out}")

    if report["failures"]:
        print("\nAUDIT FAILED")
        for failure in report["failures"]:
            print(f"- {failure}")
        return 1

    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())