#!/usr/bin/env python3
"""Repair/archive paper-forward candidates using strict lifecycle gates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidate_lifecycle import (  # noqa: E402
    build_lifecycle_context,
    evaluate_candidate_lifecycle,
    lifecycle_status_visible,
    write_lifecycle_report,
)

CONFIG_PATH = ROOT / "config" / "paper_forward_candidates.json"
ARCHIVE_PATH = ROOT / "config" / "archived_paper_forward_candidates.json"
BACKUP_DIR = ROOT / "config" / "backups"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"candidates": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"candidates": data}
    if isinstance(data, dict):
        return data
    return {"candidates": []}


def _rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("candidates", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _archive_record(row: Dict[str, Any], reason: str, stage: str) -> Dict[str, Any]:
    out = dict(row)
    out["archived_at"] = datetime.now().isoformat(timespec="seconds")
    out["archive_reason"] = reason
    out["lifecycle_stage"] = stage
    out["enabled"] = False
    return out


def repair_config(*, apply: bool) -> Dict[str, Any]:
    payload = _load_json(CONFIG_PATH)
    candidates = _rows(payload)
    context = build_lifecycle_context(candidates)
    kept: List[Dict[str, Any]] = []
    archived: List[Dict[str, Any]] = []
    repaired = 0
    errors = 0

    for row in candidates:
        try:
            status = evaluate_candidate_lifecycle(row, {}, {}, context)
        except Exception as exc:
            status = None
            errors += 1
            archived.append(_archive_record(row, f"EVALUATION_ERROR:{type(exc).__name__}:{exc}", "INVALID_CONFIG"))
            continue
        if lifecycle_status_visible(status, "Show Qualified Only"):
            kept.append(row)
            continue
        reason = status.block_reason or status.current_stage
        archived.append(_archive_record(row, reason, status.current_stage))

    existing_archive = _load_json(ARCHIVE_PATH)
    archived_all = _rows(existing_archive) + archived
    summary = {
        "kept": len(kept),
        "archived": len(archived),
        "disabled": sum(1 for row in archived if row.get("lifecycle_stage") == "DISABLED"),
        "repaired": repaired,
        "errors": errors,
    }

    statuses = [evaluate_candidate_lifecycle(row, {}, {}, context) for row in candidates]
    write_lifecycle_report(statuses, archived=archived_all)

    if apply:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"paper_forward_candidates_{ts}.json"
        shutil.copy2(CONFIG_PATH, backup)
        new_payload = dict(payload)
        new_payload["candidates"] = kept
        new_payload["lifecycle_repaired_at"] = datetime.now().isoformat(timespec="seconds")
        new_payload["lifecycle_repair_summary"] = summary
        CONFIG_PATH.write_text(json.dumps(new_payload, indent=2), encoding="utf-8")
        archive_payload = dict(existing_archive)
        archive_payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        archive_payload["candidates"] = archived_all
        ARCHIVE_PATH.write_text(json.dumps(archive_payload, indent=2), encoding="utf-8")
        summary["backup"] = str(backup)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    summary = repair_config(apply=bool(args.apply))
    print("kept archived disabled repaired errors")
    print(f"{summary['kept']} {summary['archived']} {summary['disabled']} {summary['repaired']} {summary['errors']}")
    if "backup" in summary:
        print(f"backup {summary['backup']}")
    print("reports reports/candidate_lifecycle/latest_candidate_lifecycle_report.json reports/candidate_lifecycle/latest_candidate_lifecycle_report.csv")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
