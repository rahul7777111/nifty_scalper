from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
ROUNDING_TOLERANCE_PCT = 0.01


@dataclass
class CandleIssue:
    timestamp: str
    trading_day: str
    source_file: str
    open: Any
    high: Any
    low: Any
    close: Any
    volume: Any
    invalid_reason: str
    suggested_action: str


def parse_timestamp(raw: Any) -> tuple[datetime | None, str | None]:
    text = str(raw or "").strip()
    if not text:
        return None, "missing timestamp"
    base = text.split(".")[0]
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(base, fmt), None
        except Exception:
            continue
    return None, "malformed timezone or timestamp"


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return None
        return out
    except Exception:
        return None


def evaluate_row(path: Path, row: Dict[str, Any], seen: set[str]) -> CandleIssue | None:
    ts, ts_error = parse_timestamp(row.get("time"))
    timestamp_text = str(row.get("time") or "")
    trading_day = ts.date().isoformat() if ts else ""
    if ts_error:
        return CandleIssue(
            timestamp=timestamp_text,
            trading_day=trading_day,
            source_file=path.name,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            invalid_reason=ts_error,
            suggested_action="dropped_invalid_ohlc",
        )

    if ts.time() < MARKET_OPEN or ts.time() > MARKET_CLOSE:
        return CandleIssue(
            timestamp=timestamp_text,
            trading_day=trading_day,
            source_file=path.name,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            invalid_reason="timestamp outside market hours",
            suggested_action="dropped_invalid_ohlc",
        )

    if timestamp_text in seen:
        return CandleIssue(
            timestamp=timestamp_text,
            trading_day=trading_day,
            source_file=path.name,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            invalid_reason="duplicate timestamp",
            suggested_action="dropped_invalid_ohlc",
        )
    seen.add(timestamp_text)

    open_px = safe_float(row.get("open"))
    high_px = safe_float(row.get("high"))
    low_px = safe_float(row.get("low"))
    close_px = safe_float(row.get("close"))

    missing_fields = [name for name, value in (("open", open_px), ("high", high_px), ("low", low_px), ("close", close_px)) if value is None]
    if missing_fields:
        reason = "missing or non-numeric " + ", ".join(missing_fields)
        return CandleIssue(
            timestamp=timestamp_text,
            trading_day=trading_day,
            source_file=path.name,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            invalid_reason=reason,
            suggested_action="dropped_invalid_ohlc",
        )

    reasons: List[str] = []
    if min(open_px, high_px, low_px, close_px) <= 0.0:
        reasons.append("zero or negative open/high/low/close")
    if high_px < low_px:
        reasons.append("high < low")
    if open_px > high_px:
        reasons.append("open > high")
    if open_px < low_px:
        reasons.append("open < low")
    if close_px > high_px:
        reasons.append("close > high")
    if close_px < low_px:
        reasons.append("close < low")
    if not reasons:
        return None

    suggested_action = "dropped_invalid_ohlc"
    if high_px < low_px:
        swapped_high = low_px
        swapped_low = high_px
        if swapped_low <= open_px <= swapped_high and swapped_low <= close_px <= swapped_high:
            suggested_action = "repaired_swap_high_low"
    elif ("open > high" in reasons or "open < low" in reasons or "close > high" in reasons or "close < low" in reasons):
        max_outside_pct = 0.0
        if open_px > high_px:
            max_outside_pct = max(max_outside_pct, ((open_px - high_px) / max(abs(open_px), 1e-9)) * 100.0)
        if open_px < low_px:
            max_outside_pct = max(max_outside_pct, ((low_px - open_px) / max(abs(open_px), 1e-9)) * 100.0)
        if close_px > high_px:
            max_outside_pct = max(max_outside_pct, ((close_px - high_px) / max(abs(close_px), 1e-9)) * 100.0)
        if close_px < low_px:
            max_outside_pct = max(max_outside_pct, ((low_px - close_px) / max(abs(close_px), 1e-9)) * 100.0)
        if max_outside_pct <= ROUNDING_TOLERANCE_PCT:
            suggested_action = "repaired_rounding_bound"

    return CandleIssue(
        timestamp=timestamp_text,
        trading_day=trading_day,
        source_file=path.name,
        open=row.get("open"),
        high=row.get("high"),
        low=row.get("low"),
        close=row.get("close"),
        volume=row.get("volume"),
        invalid_reason="; ".join(reasons),
        suggested_action=suggested_action,
    )


def audit_invalid_candles() -> Dict[str, Any]:
    issues: List[CandleIssue] = []
    scanned_rows = 0
    seen: set[str] = set()
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("candles") or []:
            scanned_rows += 1
            issue = evaluate_row(path, row, seen)
            if issue is not None:
                issues.append(issue)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows_scanned": scanned_rows,
        "invalid_row_count": len(issues),
        "issues": [issue.__dict__ for issue in issues],
    }
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Invalid Candle Audit",
        "",
        f"- Rows scanned: `{report['rows_scanned']}`",
        f"- Invalid rows: `{report['invalid_row_count']}`",
        "",
        "| Timestamp | Trading Day | Source File | Open | High | Low | Close | Volume | Invalid Reason | Suggested Action |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for issue in report["issues"]:
        lines.append(
            f"| {issue['timestamp']} | {issue['trading_day']} | {issue['source_file']} | "
            f"{issue['open']} | {issue['high']} | {issue['low']} | {issue['close']} | {issue['volume']} | "
            f"{issue['invalid_reason']} | {issue['suggested_action']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = audit_invalid_candles()
    json_path = REPORTS_DIR / "invalid_candle_audit.json"
    md_path = REPORTS_DIR / "invalid_candle_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(json.dumps({"rows_scanned": report["rows_scanned"], "invalid_row_count": report["invalid_row_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
