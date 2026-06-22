from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from audit_invalid_candles import ROUNDING_TOLERANCE_PCT, evaluate_row, parse_timestamp, safe_float


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
BACKUPS_DIR = REPO_ROOT / "backups"


def repair_row(row: Dict[str, Any]) -> tuple[Dict[str, Any] | None, str]:
    repaired = dict(row)
    open_px = safe_float(repaired.get("open"))
    high_px = safe_float(repaired.get("high"))
    low_px = safe_float(repaired.get("low"))
    close_px = safe_float(repaired.get("close"))
    volume_px = safe_float(repaired.get("volume"))

    if open_px is None or high_px is None or low_px is None or close_px is None:
        return None, "dropped_invalid_ohlc"

    if high_px < low_px:
        swapped_high = low_px
        swapped_low = high_px
        if swapped_low <= open_px <= swapped_high and swapped_low <= close_px <= swapped_high:
            repaired["high"] = swapped_high
            repaired["low"] = swapped_low
            if volume_px is None:
                repaired["volume"] = 0.0
            return repaired, "repaired_swap_high_low"
        return None, "dropped_invalid_ohlc"

    max_outside_pct = 0.0
    if open_px > high_px:
        max_outside_pct = max(max_outside_pct, ((open_px - high_px) / max(abs(open_px), 1e-9)) * 100.0)
        high_px = open_px
    if open_px < low_px:
        max_outside_pct = max(max_outside_pct, ((low_px - open_px) / max(abs(open_px), 1e-9)) * 100.0)
        low_px = open_px
    if close_px > high_px:
        max_outside_pct = max(max_outside_pct, ((close_px - high_px) / max(abs(close_px), 1e-9)) * 100.0)
        high_px = close_px
    if close_px < low_px:
        max_outside_pct = max(max_outside_pct, ((low_px - close_px) / max(abs(close_px), 1e-9)) * 100.0)
        low_px = close_px

    if max_outside_pct > 0.0:
        if max_outside_pct <= ROUNDING_TOLERANCE_PCT:
            repaired["high"] = high_px
            repaired["low"] = low_px
            if volume_px is None:
                repaired["volume"] = 0.0
            return repaired, "repaired_rounding_bound"
        return None, "dropped_invalid_ohlc"

    if volume_px is None:
        repaired["volume"] = 0.0
        return repaired, "volume_zero_allowed"

    return repaired, "kept"


def backup_dataset() -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_root = BACKUPS_DIR / f"mstock_candle_clean_{stamp}"
    backup_data = backup_root / "data"
    backup_data.mkdir(parents=True, exist_ok=True)
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        shutil.copy2(path, backup_data / path.name)
    return backup_root


def clean_dataset(*, create_backup: bool, mode: str) -> Dict[str, Any]:
    if mode != "safe":
        raise RuntimeError("Only --mode safe is supported.")

    backup_path = backup_dataset() if create_backup else None
    rows_scanned = 0
    rows_repaired = 0
    rows_dropped = 0
    kept_rows = 0
    action_counter: Counter[str] = Counter()
    per_file_actions: Dict[str, List[Dict[str, Any]]] = {}

    for path in sorted(DATA_DIR.glob("candles_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        cleaned_rows: List[Dict[str, Any]] = []
        file_actions: List[Dict[str, Any]] = []

        for row in payload.get("candles") or []:
            rows_scanned += 1
            issue = evaluate_row(path, row, seen)
            if issue is None:
                repaired_row, action = repair_row(row)
                if repaired_row is None:
                    rows_dropped += 1
                    action_counter["dropped_invalid_ohlc"] += 1
                else:
                    cleaned_rows.append(repaired_row)
                    kept_rows += 1
                    if action != "kept":
                        rows_repaired += 1
                        action_counter[action] += 1
                continue

            repaired_row, action = repair_row(row)
            file_actions.append(
                {
                    "timestamp": issue.timestamp,
                    "invalid_reason": issue.invalid_reason,
                    "action": action,
                    "source_file": path.name,
                }
            )
            if repaired_row is None:
                rows_dropped += 1
                action_counter["dropped_invalid_ohlc"] += 1
                continue
            cleaned_rows.append(repaired_row)
            kept_rows += 1
            if action != "kept":
                rows_repaired += 1
                action_counter[action] += 1

        payload["candles"] = cleaned_rows
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if file_actions:
            per_file_actions[path.name] = file_actions

    from audit_invalid_candles import audit_invalid_candles

    remaining_invalid = audit_invalid_candles()
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows_scanned": rows_scanned,
        "rows_repaired": rows_repaired,
        "rows_dropped": rows_dropped,
        "rows_kept": kept_rows,
        "remaining_invalid_rows": remaining_invalid["invalid_row_count"],
        "backup_path": str(backup_path) if backup_path else None,
        "output_dataset_path": str(DATA_DIR),
        "actions_summary": dict(action_counter),
        "file_actions": per_file_actions,
    }
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Candle Cleaning Report",
        "",
        f"- Rows scanned: `{report['rows_scanned']}`",
        f"- Rows repaired: `{report['rows_repaired']}`",
        f"- Rows dropped: `{report['rows_dropped']}`",
        f"- Remaining invalid rows: `{report['remaining_invalid_rows']}`",
        f"- Backup path: `{report['backup_path']}`",
        f"- Output dataset path: `{report['output_dataset_path']}`",
        "",
        "## Action Summary",
        "",
        "| Action | Count |",
        "|---|---:|",
    ]
    for action, count in sorted((report.get("actions_summary") or {}).items()):
        lines.append(f"| {action} | {count} |")
    lines.extend(["", "## File Actions", ""])
    for file_name, actions in sorted((report.get("file_actions") or {}).items()):
        lines.append(f"### {file_name}")
        lines.append("")
        lines.append("| Timestamp | Invalid Reason | Action |")
        lines.append("|---|---|---|")
        for action in actions:
            lines.append(f"| {action['timestamp']} | {action['invalid_reason']} | {action['action']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely repair or drop invalid m.Stock candles.")
    parser.add_argument("--backup", action="store_true", help="Create a backup of all candle files before cleaning.")
    parser.add_argument("--mode", default="safe", help="Cleaning mode. Only 'safe' is supported.")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = clean_dataset(create_backup=bool(args.backup), mode=str(args.mode or "safe"))
    json_path = REPORTS_DIR / "candle_cleaning_report.json"
    md_path = REPORTS_DIR / "candle_cleaning_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(json.dumps({k: report[k] for k in ('rows_scanned', 'rows_repaired', 'rows_dropped', 'remaining_invalid_rows')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
