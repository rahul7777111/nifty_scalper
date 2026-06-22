#!/usr/bin/env python3
"""Audit model artifacts for extractable feature metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from candidate_artifact_resolver import resolve_candidate_artifact  # noqa: E402
from scripts.backtest_ml_models_from_csv import (  # noqa: E402
    _load_pickle_or_joblib,
    extract_model_feature_order,
)


def load_candidates(config_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit(f"Invalid candidates list in {config_path}")
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit candidate model feature metadata.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "paper_forward_candidates.json"))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for candidate in load_candidates(config_path):
        cid = str(candidate.get("candidate_id") or "")
        enabled = bool(candidate.get("enabled", True))
        resolution = resolve_candidate_artifact(candidate, root, strict=True, allow_fallback=False)
        row: dict[str, Any] = {
            "candidate_id": cid,
            "enabled": enabled,
            "artifact_identity_status": resolution.artifact_identity_status,
            "artifact_path": resolution.selected_artifact_path,
            "feature_count": 0,
            "feature_source": "",
            "model_type": "",
            "first_20_features": [],
            "status": "ARTIFACT_NOT_READY",
        }
        if resolution.loaded and resolution.model_file:
            try:
                obj = _load_pickle_or_joblib(Path(resolution.model_file))
                info = extract_model_feature_order(obj, Path(resolution.model_file))
                features = list(info.get("feature_order") or [])
                row.update(
                    {
                        "feature_count": len(features),
                        "feature_source": info.get("feature_source"),
                        "model_type": info.get("model_type"),
                        "first_20_features": features[:20],
                        "status": "OK" if features else "MODEL_FEATURES_MISSING",
                    }
                )
            except Exception as exc:
                row["status"] = "MODEL_LOAD_FAILED"
                row["error"] = str(exc)
        if enabled and row["status"] != "OK":
            failures.append(f"{cid}: {row['status']}")
        rows.append(row)

    report = {
        "total_candidates": len(rows),
        "enabled_candidates": sum(1 for r in rows if r["enabled"]),
        "feature_metadata_ok_count": sum(1 for r in rows if r["status"] == "OK"),
        "missing_feature_metadata_count": sum(1 for r in rows if r["status"] == "MODEL_FEATURES_MISSING"),
        "failures": failures,
        "per_candidate": rows,
    }

    print("MODEL FEATURE METADATA AUDIT")
    for key in ("total_candidates", "enabled_candidates", "feature_metadata_ok_count", "missing_feature_metadata_count"):
        print(f"{key}={report[key]}")
    print("")
    print("candidate_id | enabled | status | feature_count | feature_source | artifact_path")
    print("-" * 180)
    for row in rows:
        print(
            f"{row['candidate_id']} | {row['enabled']} | {row['status']} | {row['feature_count']} | "
            f"{row.get('feature_source') or '-'} | {row.get('artifact_path') or '-'}"
        )

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON report: {out}")

    if failures:
        print("\nAUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
