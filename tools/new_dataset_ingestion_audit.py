from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

KNOWN_BASELINE = {
    "raw_candles": 21750,
    "resampled_candles": 2175,
    "real_trading_days": 58,
}


def parse_timestamp(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    base = text.split(".")[0]
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(base, fmt)
        except Exception:
            continue
    return None


def weekday_dates_between(start_date, end_date):
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def load_raw_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candles = payload.get("candles") or []
            if candles:
                rows.extend(candles)
                sources.append(str(path.relative_to(REPO_ROOT)))
        except Exception:
            continue
    return rows, sources


def build_report() -> Dict[str, Any]:
    raw_rows, source_files = load_raw_rows()
    timestamps: List[datetime] = []
    invalid_ohlc = 0
    parsed_rows: List[Dict[str, Any]] = []

    for row in raw_rows:
        ts = parse_timestamp(row.get("time"))
        if ts is None:
            invalid_ohlc += 1
            continue
        try:
            open_px = float(row.get("open", 0.0) or 0.0)
            high_px = float(row.get("high", 0.0) or 0.0)
            low_px = float(row.get("low", 0.0) or 0.0)
            close_px = float(row.get("close", 0.0) or 0.0)
            volume = float(row.get("volume", 0.0) or 0.0)
        except Exception:
            invalid_ohlc += 1
            continue
        if min(open_px, high_px, low_px, close_px) <= 0.0 or high_px < max(open_px, close_px) or low_px > min(open_px, close_px) or high_px < low_px:
            invalid_ohlc += 1
        parsed_rows.append(
            {
                "time": ts,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "volume": volume,
            }
        )
        timestamps.append(ts)

    parsed_rows.sort(key=lambda item: item["time"])
    counter = Counter(timestamps)
    duplicate_timestamps = int(sum(count - 1 for count in counter.values() if count > 1))
    earliest = parsed_rows[0]["time"] if parsed_rows else None
    latest = parsed_rows[-1]["time"] if parsed_rows else None

    candles_by_day: Dict[str, List[datetime]] = defaultdict(list)
    for row in parsed_rows:
        candles_by_day[row["time"].date().isoformat()].append(row["time"])

    observed_daily_counts = [len({ts for ts in day_rows}) for day_rows in candles_by_day.values()]
    expected_candles_per_day = int(sorted(observed_daily_counts)[len(observed_daily_counts) // 2]) if observed_daily_counts else 0
    active_trading_days = len(candles_by_day)
    latest_day = latest.date().isoformat() if latest else None
    latest_day_count = len({ts for ts in candles_by_day.get(latest_day, [])}) if latest_day else 0
    current_day_status = "unknown"
    if latest_day:
        current_day_status = "complete" if latest_day_count >= expected_candles_per_day else "partial"

    missing_trading_days: List[str] = []
    if earliest and latest:
        actual_dates = {datetime.fromisoformat(day).date() for day in candles_by_day}
        missing_trading_days = [
            day.isoformat()
            for day in weekday_dates_between(earliest.date(), latest.date())
            if day not in actual_dates
        ]

    raw_after = len(parsed_rows)
    resampled_after = len(parsed_rows[::10])
    new_raw_candles_added = raw_after - KNOWN_BASELINE["raw_candles"]
    new_days_added = active_trading_days - KNOWN_BASELINE["real_trading_days"]
    latest_is_newer_than_baseline = raw_after > KNOWN_BASELINE["raw_candles"] or active_trading_days > KNOWN_BASELINE["real_trading_days"]

    return {
        "active_dataset_glob": str((DATA_DIR / "candles_*.json").relative_to(REPO_ROOT)),
        "active_dataset_files": source_files,
        "raw_candle_count_before_update": KNOWN_BASELINE["raw_candles"],
        "raw_candle_count_after_update": raw_after,
        "resampled_candle_count_before_update": KNOWN_BASELINE["resampled_candles"],
        "resampled_candle_count_after_update": resampled_after,
        "earliest_timestamp": earliest.isoformat() if earliest else None,
        "latest_timestamp": latest.isoformat() if latest else None,
        "active_trading_days_before_update": KNOWN_BASELINE["real_trading_days"],
        "active_trading_days_after_update": active_trading_days,
        "missing_trading_days": missing_trading_days,
        "missing_trading_day_count": len(missing_trading_days),
        "duplicate_timestamps": duplicate_timestamps,
        "invalid_ohlc_rows": invalid_ohlc,
        "latest_day_candle_count": latest_day_count,
        "expected_candles_per_day": expected_candles_per_day,
        "current_day_status": current_day_status,
        "new_raw_candles_added": new_raw_candles_added,
        "new_active_trading_days_added": new_days_added,
        "latest_timestamp_newer_than_known_baseline": latest_is_newer_than_baseline,
        "has_new_real_data": new_raw_candles_added > 0 or new_days_added > 0,
        "decision": "proceed" if (new_raw_candles_added > 0 or new_days_added > 0) else "skip_retraining",
        "decision_reason": "New real candles detected." if (new_raw_candles_added > 0 or new_days_added > 0) else "No new real data found. Retraining skipped.",
    }


def render_markdown(report: Dict[str, Any]) -> str:
    missing_preview = ", ".join(report.get("missing_trading_days", [])[:20]) or "None"
    files_preview = "\n".join(f"- `{path}`" for path in report.get("active_dataset_files", [])) or "- None"
    return "\n".join(
        [
            "# New Dataset Ingestion Audit",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Reason: {report['decision_reason']}",
            f"- Active dataset glob: `{report['active_dataset_glob']}`",
            "",
            "## Active Files",
            "",
            files_preview,
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Raw candle count before update | {report['raw_candle_count_before_update']} |",
            f"| Raw candle count after update | {report['raw_candle_count_after_update']} |",
            f"| Resampled candle count before update | {report['resampled_candle_count_before_update']} |",
            f"| Resampled candle count after update | {report['resampled_candle_count_after_update']} |",
            f"| Earliest timestamp | {report['earliest_timestamp']} |",
            f"| Latest timestamp | {report['latest_timestamp']} |",
            f"| Active trading days before update | {report['active_trading_days_before_update']} |",
            f"| Active trading days after update | {report['active_trading_days_after_update']} |",
            f"| Missing trading day count | {report['missing_trading_day_count']} |",
            f"| Duplicate timestamps | {report['duplicate_timestamps']} |",
            f"| Invalid OHLC rows | {report['invalid_ohlc_rows']} |",
            f"| Latest day candle count | {report['latest_day_candle_count']} |",
            f"| Expected candles per day | {report['expected_candles_per_day']} |",
            f"| Current day status | {report['current_day_status']} |",
            f"| New raw candles added | {report['new_raw_candles_added']} |",
            f"| New active trading days added | {report['new_active_trading_days_added']} |",
            "",
            f"Missing trading days preview: {missing_preview}",
        ]
    ) + "\n"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = REPORTS_DIR / "new_dataset_ingestion_audit.json"
    md_path = REPORTS_DIR / "new_dataset_ingestion_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
