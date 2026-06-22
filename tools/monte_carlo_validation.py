from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "trades.db"
REPORTS_DIR = REPO_ROOT / "reports"
STARTING_CAPITAL = 100000.0
SIMULATIONS = 1500
RUIN_THRESHOLD = 0.20


def load_trade_pnls() -> List[float]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pnl
        FROM trade_outcomes
        WHERE trade_id NOT LIKE 'NS_SHADOW_%'
        ORDER BY id ASC
        """
    )
    values = [float(row[0] or 0.0) for row in cur.fetchall()]
    conn.close()
    return values


def max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def recovery_time(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    trough_idx = 0
    peak_idx = 0
    worst_dd = 0.0
    for idx, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_idx = idx
        dd = peak - value
        if dd > worst_dd:
            worst_dd = dd
            trough_idx = idx
    if worst_dd <= 0.0:
        return 0.0
    target = max(equity_curve[: trough_idx + 1])
    for idx in range(trough_idx + 1, len(equity_curve)):
        if equity_curve[idx] >= target:
            return float(idx - trough_idx)
    return float(len(equity_curve) - trough_idx)


def cagr(final_equity: float, periods: int) -> float:
    if periods <= 0 or final_equity <= 0.0:
        return 0.0
    years = max(periods / 252.0, 1.0 / 252.0)
    return (final_equity / STARTING_CAPITAL) ** (1.0 / years) - 1.0


def simulate_equity_curves(pnls: List[float], simulations: int = SIMULATIONS) -> Dict[str, float]:
    if not pnls:
        return {
            "simulations": 0,
            "expected_cagr": 0.0,
            "average_max_drawdown": 0.0,
            "probability_of_ruin": 0.0,
            "average_recovery_time": 0.0,
        }
    rng = random.Random(42)
    max_dds = []
    ruined = 0
    recovery_times = []
    cagrs = []
    block = min(5, len(pnls))
    for _ in range(int(simulations)):
        sampled = []
        while len(sampled) < len(pnls):
            start = rng.randrange(0, len(pnls))
            sampled.extend(pnls[start : start + block] or [pnls[start]])
        sampled = sampled[: len(pnls)]
        equity = [STARTING_CAPITAL]
        for pnl in sampled:
            equity.append(equity[-1] + pnl)
        dd = max_drawdown(equity)
        max_dds.append(dd)
        if min(equity) <= STARTING_CAPITAL * (1.0 - RUIN_THRESHOLD):
            ruined += 1
        recovery_times.append(recovery_time(equity))
        cagrs.append(cagr(equity[-1], len(sampled)))
    return {
        "simulations": int(simulations),
        "expected_cagr": float(sum(cagrs) / len(cagrs)) if cagrs else 0.0,
        "average_max_drawdown": float(sum(max_dds) / len(max_dds)) if max_dds else 0.0,
        "probability_of_ruin": float(ruined / simulations) if simulations else 0.0,
        "average_recovery_time": float(sum(recovery_times) / len(recovery_times)) if recovery_times else 0.0,
        "max_drawdown_p95": float(sorted(max_dds)[int(0.95 * (len(max_dds) - 1))]) if max_dds else 0.0,
    }


def render_markdown(report: Dict[str, float]) -> str:
    return "\n".join(
        [
            "# Monte Carlo Validation Report",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| simulations | {report['simulations']} |",
            f"| expected_cagr | {report['expected_cagr']:.6f} |",
            f"| average_max_drawdown | {report['average_max_drawdown']:.6f} |",
            f"| max_drawdown_p95 | {report['max_drawdown_p95']:.6f} |",
            f"| probability_of_ruin | {report['probability_of_ruin']:.6f} |",
            f"| average_recovery_time | {report['average_recovery_time']:.6f} |",
            "",
            "Method: block-bootstrap on realized paper-trade PnL with 1,500 simulated equity curves.",
        ]
    ) + "\n"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pnls = load_trade_pnls()
    report = simulate_equity_curves(pnls)
    json_path = REPORTS_DIR / "monte_carlo_report.json"
    md_path = REPORTS_DIR / "monte_carlo_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
