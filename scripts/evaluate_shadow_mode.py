#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ML shadow mode JSONL logs.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def evaluate_shadow_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    enter_rows = [row for row in rows if row.get("shadow_decision") == "WOULD_ENTER"]
    realized = [float(row.get("realized_return") or 0.0) for row in enter_rows if row.get("realized_return") is not None]
    gross_profit = sum(v for v in realized if v > 0)
    gross_loss = abs(sum(v for v in realized if v < 0))
    pf = (gross_profit / gross_loss) if gross_loss else (999.0 if gross_profit > 0 else 0.0)
    avg_return = float(np.mean(realized)) if realized else 0.0
    sharpe = float((np.mean(realized) / np.std(realized, ddof=1)) * np.sqrt(252.0)) if len(realized) > 1 and float(np.std(realized, ddof=1)) > 0 else 0.0
    equity = np.cumsum(realized) if realized else np.asarray([])
    peaks = np.maximum.accumulate(equity) if realized else np.asarray([])
    max_dd = float((peaks - equity).max()) if realized else 0.0
    by_option = defaultdict(int)
    by_time = defaultdict(int)
    for row in enter_rows:
        by_option[str(row.get("option_type") or "unknown")] += 1
        try:
            ts = datetime.fromisoformat(str(row.get("timestamp")).replace("Z", "+00:00"))
            bucket = "opening" if ts.hour < 10 or (ts.hour == 10 and ts.minute < 30) else "midday" if ts.hour < 14 else "closing"
            by_time[bucket] += 1
        except Exception:
            by_time["unknown"] += 1
    return {
        "number_of_predictions": len(rows),
        "number_of_would_enter_signals": len(enter_rows),
        "realized_outcome_count": len(realized),
        "win_rate": float(sum(1 for v in realized if v > 0) / len(realized)) if realized else 0.0,
        "average_return": avg_return,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "ce_pe_breakdown": dict(by_option),
        "time_of_day_breakdown": dict(by_time),
        "daily_pnl_stability": {
            "best_day_contribution_pct": 0.0,
            "active_days": 1 if realized else 0,
        },
    }


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_dir) / f"ml_shadow_predictions_{args.date}.jsonl"
    rows = _load_jsonl(log_path)
    payload = evaluate_shadow_rows(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"shadow_mode_evaluation_{stamp}.json"
    md_path = out_dir / f"shadow_mode_evaluation_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Shadow Mode Evaluation",
                "",
                f"- number_of_predictions: `{payload['number_of_predictions']}`",
                f"- number_of_would_enter_signals: `{payload['number_of_would_enter_signals']}`",
                f"- realized_outcome_count: `{payload['realized_outcome_count']}`",
                f"- win_rate: `{payload['win_rate']}`",
                f"- average_return: `{payload['average_return']}`",
                f"- profit_factor: `{payload['profit_factor']}`",
                f"- sharpe: `{payload['sharpe']}`",
                f"- max_drawdown: `{payload['max_drawdown']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
