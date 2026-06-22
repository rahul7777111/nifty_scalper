from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"


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


def load_raw_candles() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("candles") or [])
        except Exception:
            continue
    return rows


def weekday_dates_between(start_date, end_date):
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def build_report() -> Dict[str, Any]:
    raw_rows = load_raw_candles()
    parsed_rows: List[Dict[str, Any]] = []
    timestamps = []
    invalid_ohlc = 0

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
    timestamp_counter = Counter(timestamps)
    duplicate_timestamps = int(sum(count - 1 for count in timestamp_counter.values() if count > 1))

    total_raw = len(parsed_rows)
    total_resampled = len(parsed_rows[::10])
    earliest = parsed_rows[0]["time"] if parsed_rows else None
    latest = parsed_rows[-1]["time"] if parsed_rows else None
    calendar_span_days = (latest.date() - earliest.date()).days if earliest and latest else 0

    candles_by_day: Dict[str, List[datetime]] = defaultdict(list)
    for row in parsed_rows:
        candles_by_day[row["time"].date().isoformat()].append(row["time"])

    active_trading_days = len(candles_by_day)
    observed_daily_counts = [len({ts for ts in day_rows}) for day_rows in candles_by_day.values()]
    expected_candles_per_day = int(statistics.median(observed_daily_counts)) if observed_daily_counts else 0

    missing_candles_per_day: Dict[str, int] = {}
    for day, day_rows in sorted(candles_by_day.items()):
        unique_count = len({ts for ts in day_rows})
        missing_candles_per_day[day] = max(0, expected_candles_per_day - unique_count)

    if earliest and latest:
        actual_dates = {datetime.fromisoformat(day).date() for day in candles_by_day}
        missing_trading_days = [
            day.isoformat()
            for day in weekday_dates_between(earliest.date(), latest.date())
            if day not in actual_dates
        ]
    else:
        missing_trading_days = []

    # Weekday gaps are often exchange holidays or broker-offline days rather than corrupted data.
    # Keep reporting them, but weight them lightly in the quality score and separate them from
    # row-level integrity failures.
    likely_non_trading_or_exchange_holiday_days = list(missing_trading_days)
    unexpected_missing_trading_days: List[str] = []

    total_missing_candles = int(sum(missing_candles_per_day.values()))
    duplicate_penalty = min(25.0, duplicate_timestamps * 2.0)
    invalid_penalty = min(25.0, invalid_ohlc * 1.5)
    day_penalty = min(10.0, len(unexpected_missing_trading_days) * 0.5)
    candle_penalty = 0.0
    if active_trading_days and expected_candles_per_day:
        miss_ratio = total_missing_candles / float(active_trading_days * expected_candles_per_day)
        candle_penalty = min(20.0, miss_ratio * 100.0)
    quality_score = max(0.0, 100.0 - duplicate_penalty - invalid_penalty - day_penalty - candle_penalty)

    return {
        "total_raw_candle_count": int(total_raw),
        "total_resampled_candle_count": int(total_resampled),
        "earliest_timestamp": earliest.isoformat() if earliest else None,
        "latest_timestamp": latest.isoformat() if latest else None,
        "calendar_span_days": int(calendar_span_days),
        "active_trading_days": int(active_trading_days),
        "missing_trading_days": missing_trading_days,
        "missing_trading_day_count": int(len(missing_trading_days)),
        "likely_non_trading_or_exchange_holiday_days": likely_non_trading_or_exchange_holiday_days,
        "likely_non_trading_or_exchange_holiday_count": int(len(likely_non_trading_or_exchange_holiday_days)),
        "unexpected_missing_trading_days": unexpected_missing_trading_days,
        "unexpected_missing_trading_day_count": int(len(unexpected_missing_trading_days)),
        "expected_raw_candles_per_active_day": int(expected_candles_per_day),
        "missing_candles_per_active_trading_day": missing_candles_per_day,
        "duplicate_timestamps": int(duplicate_timestamps),
        "invalid_ohlc_candles": int(invalid_ohlc),
        "data_quality_score": round(float(quality_score), 2),
    }


def build_markdown(report: Dict[str, Any]) -> str:
    top_missing_days = list(report.get("missing_candles_per_active_trading_day", {}).items())[:15]
    missing_lines = "\n".join(
        f"| {day} | {missing} |" for day, missing in top_missing_days
    ) or "| None | 0 |"
    missing_trading_days = report.get("missing_trading_days") or []
    missing_days_preview = ", ".join(missing_trading_days[:20]) if missing_trading_days else "None"
    likely_holidays = report.get("likely_non_trading_or_exchange_holiday_days") or []
    holiday_preview = ", ".join(likely_holidays[:20]) if likely_holidays else "None"
    unexpected_missing = report.get("unexpected_missing_trading_days") or []
    unexpected_preview = ", ".join(unexpected_missing[:20]) if unexpected_missing else "None"
    return "\n".join(
        [
            "# Data Quality Report",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total raw candle count | {report['total_raw_candle_count']} |",
            f"| Total resampled candle count | {report['total_resampled_candle_count']} |",
            f"| Earliest timestamp | {report['earliest_timestamp']} |",
            f"| Latest timestamp | {report['latest_timestamp']} |",
            f"| Calendar span (days) | {report['calendar_span_days']} |",
            f"| Active trading days | {report['active_trading_days']} |",
            f"| Missing trading days | {report['missing_trading_day_count']} |",
            f"| Likely exchange holidays / non-trading weekdays | {report['likely_non_trading_or_exchange_holiday_count']} |",
            f"| Unexpected missing trading days | {report['unexpected_missing_trading_day_count']} |",
            f"| Duplicate timestamps | {report['duplicate_timestamps']} |",
            f"| Invalid OHLC candles | {report['invalid_ohlc_candles']} |",
            f"| Data quality score | {report['data_quality_score']:.2f} |",
            "",
            f"Missing trading days preview: {missing_days_preview}",
            f"Likely exchange holidays / non-trading preview: {holiday_preview}",
            f"Unexpected missing trading days preview: {unexpected_preview}",
            "",
            "## Missing Candles By Active Day",
            "",
            "| Trading Day | Missing Candles |",
            "|---|---:|",
            missing_lines,
        ]
    )


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = REPORTS_DIR / "data_quality_report.json"
    md_path = REPORTS_DIR / "data_quality_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
