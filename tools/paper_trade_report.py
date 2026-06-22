from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DB_PATH = REPO_ROOT / "trades.db"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db import DatabaseManager


def parse_dt(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return None


def load_trade_rows() -> List[Dict[str, Any]]:
    DatabaseManager(str(DB_PATH))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT o.trade_id, o.prediction_id, o.entry_time, o.exit_time, o.strategy, o.regime, o.strategy_signal,
               o.entry_reason, o.exit_reason, o.model_version, o.label_policy_version, o.feature_set_version,
               o.probability, o.threshold, o.confidence, o.confidence_bucket, o.gross_pnl, o.estimated_costs,
               o.estimated_slippage, o.pnl, o.timestamp, p.direction, p.symbol, p.source
        FROM trade_outcomes o
        LEFT JOIN predictions p ON p.prediction_id = o.prediction_id OR p.trade_id = o.trade_id
        WHERE o.trade_id NOT LIKE 'NS_SHADOW_%'
        ORDER BY o.id ASC
        """
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    latest_by_trade: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest_by_trade[str(row["trade_id"])] = row
    return list(latest_by_trade.values())


def compute_drawdown_periods(pnls: List[float]) -> List[Dict[str, float]]:
    periods: List[Dict[str, float]] = []
    equity = 0.0
    peak = 0.0
    start_idx = None
    trough = 0.0
    for idx, pnl in enumerate(pnls):
        equity += pnl
        if equity >= peak:
            if start_idx is not None:
                periods.append(
                    {
                        "start_index": float(start_idx),
                        "end_index": float(idx),
                        "depth": float(trough),
                    }
                )
                start_idx = None
                trough = 0.0
            peak = equity
            continue
        dd = peak - equity
        if start_idx is None:
            start_idx = idx
            trough = dd
        else:
            trough = max(trough, dd)
    if start_idx is not None:
        periods.append(
            {
                "start_index": float(start_idx),
                "end_index": float(len(pnls) - 1),
                "depth": float(trough),
            }
        )
    return periods


def compute_report() -> Dict[str, Any]:
    rows = load_trade_rows()
    pnls = [float(row.get("pnl") or 0.0) for row in rows]
    wins = [pnl for pnl in pnls if pnl > 0.0]
    losses = [pnl for pnl in pnls if pnl < 0.0]
    total = len(rows)
    win_rate = (len(wins) / total) if total else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (99.0 if wins else 0.0)
    expectancy = (sum(pnls) / total) if total else 0.0

    daily_pnl: Dict[str, float] = defaultdict(float)
    durations: List[float] = []
    by_strategy: Dict[str, List[float]] = defaultdict(list)
    by_confidence_bucket: Dict[str, List[float]] = defaultdict(list)
    by_hour: Dict[str, List[float]] = defaultdict(list)
    by_weekday: Dict[str, List[float]] = defaultdict(list)
    by_signal_type: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        pnl = float(row.get("pnl") or 0.0)
        ts = parse_dt(row.get("timestamp"))
        if ts is not None:
            daily_pnl[ts.date().isoformat()] += pnl
            by_hour[f"{ts.hour:02d}:00"].append(pnl)
            by_weekday[ts.strftime("%A")].append(pnl)
        entry_dt = parse_dt(row.get("entry_time"))
        exit_dt = parse_dt(row.get("exit_time"))
        if entry_dt is not None and exit_dt is not None and exit_dt >= entry_dt:
            durations.append((exit_dt - entry_dt).total_seconds() / 60.0)
        strategy = str(row.get("strategy") or "unknown")
        by_strategy[strategy].append(pnl)
        confidence = float(row.get("confidence") or 0.0)
        lower = math.floor(confidence * 10.0) / 10.0
        bucket = f"{lower:.1f}-{min(1.0, lower + 0.1):.1f}"
        by_confidence_bucket[bucket].append(pnl)
        signal_type = str(row.get("direction") or row.get("strategy") or "unknown")
        by_signal_type[signal_type].append(pnl)

    daily_returns = list(daily_pnl.values())
    if len(daily_returns) > 1:
        mean_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((ret - mean_ret) ** 2 for ret in daily_returns) / (len(daily_returns) - 1)
        std_dev = math.sqrt(variance)
        sharpe = (mean_ret / std_dev) * math.sqrt(252.0) if std_dev > 0 else 0.0
    else:
        sharpe = 0.0

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    strategy_breakdown = {
        strategy: {
            "count": len(values),
            "expectancy": (sum(values) / len(values)) if values else 0.0,
            "profit_factor": (sum(v for v in values if v > 0.0) / abs(sum(v for v in values if v < 0.0)))
            if any(v < 0.0 for v in values)
            else (99.0 if any(v > 0.0 for v in values) else 0.0),
        }
        for strategy, values in sorted(by_strategy.items())
    }
    confidence_breakdown = {
        bucket: {
            "count": len(values),
            "expectancy": (sum(values) / len(values)) if values else 0.0,
        }
        for bucket, values in sorted(by_confidence_bucket.items())
    }
    pnl_by_hour = {bucket: {"count": len(values), "pnl": float(sum(values))} for bucket, values in sorted(by_hour.items())}
    pnl_by_weekday = {bucket: {"count": len(values), "pnl": float(sum(values))} for bucket, values in sorted(by_weekday.items())}
    signal_type_breakdown = {bucket: {"count": len(values), "pnl": float(sum(values))} for bucket, values in sorted(by_signal_type.items())}
    linkage_coverage = sum(1 for row in rows if str(row.get("prediction_id") or "").strip())

    return {
        "paper_trades": total,
        "linked_predictions": linkage_coverage,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "daily_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "average_trade_duration_minutes": (sum(durations) / len(durations)) if durations else None,
        "strategy_breakdown": strategy_breakdown,
        "model_confidence_breakdown": confidence_breakdown,
        "expectancy_by_confidence_bucket": confidence_breakdown,
        "pnl_by_hour": pnl_by_hour,
        "pnl_by_weekday": pnl_by_weekday,
        "signal_type_breakdown": signal_type_breakdown,
        "drawdown_periods": compute_drawdown_periods(pnls),
        "slippage_estimate": None,
        "linkage_preview": rows[:25],
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Paper Trade Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| paper_trades | {report['paper_trades']} |",
        f"| linked_predictions | {report['linked_predictions']} |",
        f"| win_rate | {report['win_rate']:.4f} |",
        f"| profit_factor | {report['profit_factor']:.4f} |",
        f"| expectancy | {report['expectancy']:.4f} |",
        f"| daily_sharpe | {report['daily_sharpe']:.4f} |",
        f"| max_drawdown | {report['max_drawdown']:.4f} |",
        f"| average_trade_duration_minutes | {report['average_trade_duration_minutes']} |",
        "",
        "## Strategy Breakdown",
        "",
        "| Strategy | Count | Expectancy | Profit Factor |",
        "|---|---:|---:|---:|",
    ]
    for strategy, values in report.get("strategy_breakdown", {}).items():
        lines.append(f"| {strategy} | {values['count']} | {values['expectancy']:.4f} | {values['profit_factor']:.4f} |")
    lines.extend(["", "## Expectancy By Confidence Bucket", "", "| Bucket | Count | Expectancy |", "|---|---:|---:|"])
    for bucket, values in report.get("expectancy_by_confidence_bucket", {}).items():
        lines.append(f"| {bucket} | {values['count']} | {values['expectancy']:.4f} |")
    lines.extend(["", "## PnL By Hour", "", "| Hour | Count | PnL |", "|---|---:|---:|"])
    for bucket, values in report.get("pnl_by_hour", {}).items():
        lines.append(f"| {bucket} | {values['count']} | {values['pnl']:.4f} |")
    lines.extend(["", "## PnL By Weekday", "", "| Weekday | Count | PnL |", "|---|---:|---:|"])
    for bucket, values in report.get("pnl_by_weekday", {}).items():
        lines.append(f"| {bucket} | {values['count']} | {values['pnl']:.4f} |")
    lines.extend(["", "## Signal Type Breakdown", "", "| Signal Type | Count | PnL |", "|---|---:|---:|"])
    for bucket, values in report.get("signal_type_breakdown", {}).items():
        lines.append(f"| {bucket} | {values['count']} | {values['pnl']:.4f} |")
    lines.extend(["", "## Drawdown Periods", ""])
    for period in report.get("drawdown_periods", []):
        lines.append(
            f"- Start idx `{int(period['start_index'])}`, end idx `{int(period['end_index'])}`, depth `{period['depth']:.4f}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = compute_report()
    json_path = REPORTS_DIR / "paper_trade_report.json"
    md_path = REPORTS_DIR / "paper_trade_report.md"
    linkage_json_path = REPORTS_DIR / "paper_trade_ml_linkage_report.json"
    linkage_md_path = REPORTS_DIR / "paper_trade_ml_linkage_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    linkage_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    linkage_md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(f"Saved {linkage_json_path}")
    print(f"Saved {linkage_md_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
