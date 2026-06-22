#!/usr/bin/env python3
"""
scripts/summarize_shadow_results.py
===================================
Generate per-candidate shadow performance summary from the three shadow JSONL logs.

Usage:
  python scripts/summarize_shadow_results.py

Outputs reports/shadow_summary_*.json and .md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_jsonl(pattern: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(LOGS_DIR.glob(pattern)):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _ts()

    signals = _load_jsonl("shadow_signals_*.jsonl")
    virtuals = _load_jsonl("shadow_virtual_trades_*.jsonl")
    blocks = _load_jsonl("shadow_blocks_*.jsonl")

    per: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "shadow_days": set(),
        "signal_count": 0,
        "virtual_trade_count": 0,
        "gross_virtual_pnl": 0.0,
        "cost_adjusted_virtual_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "max_drawdown": 0.0,
        "stale_data_trade_count": 0,
        "invalid_expiry_trade_count": 0,
        "duplicate_entry_count": 0,
        "missing_sl_count": 0,
        "missing_exit_count": 0,
        "risk_bypass_count": 0,
        "router_silence_count": 0,
        "order_payload_error_count": 0,
        "emergency_stop_count": 0,
        "spread_violation_count": 0,
        "after_cutoff_trade_count": 0,
    })

    all_rows = signals + virtuals + blocks
    for r in all_rows:
        cid = r.get("candidate_id") or "unknown"
        s = per[cid]
        t = r.get("timestamp") or r.get("ts")
        if t:
            try:
                s["shadow_days"].add(str(t)[:10])
            except Exception:
                pass
        s["signal_count"] += 1

        if r.get("virtual_entry_price") or r.get("virtual_exit_price"):
            s["virtual_trade_count"] += 1
        if r.get("virtual_pnl") is not None:
            try:
                pnl = float(r.get("virtual_pnl") or 0)
                s["cost_adjusted_virtual_pnl"] += pnl
                s["gross_virtual_pnl"] += pnl
                if pnl > 0: s["wins"] += 1
                elif pnl < 0: s["losses"] += 1
                s["max_drawdown"] = max(s["max_drawdown"], abs(min(0, pnl)))  # simplistic
            except Exception:
                pass

        br = str(r.get("blocked_reason", "")).lower()
        if "stale" in br: s["stale_data_trade_count"] += 1
        if "invalid" in br and "expir" in br: s["invalid_expiry_trade_count"] += 1
        if "duplicate" in br: s["duplicate_entry_count"] += 1
        if "missing_sl" in br or "no sl" in br: s["missing_sl_count"] += 1
        if "missing_exit" in br: s["missing_exit_count"] += 1
        if "risk" in br and "bypass" in br: s["risk_bypass_count"] += 1
        if "router" in br and ("silent" in br or "exception" in br): s["router_silence_count"] += 1
        if "payload" in br and "error" in br: s["order_payload_error_count"] += 1
        if "emergency" in br: s["emergency_stop_count"] += 1
        if "spread" in br: s["spread_violation_count"] += 1
        if "cutoff" in br: s["after_cutoff_trade_count"] += 1

    for s in per.values():
        s["shadow_days"] = len(s["shadow_days"])
        trades = s["virtual_trade_count"] or max(1, s["signal_count"] // 10)
        s["profit_factor"] = (s["wins"] / max(1, s["losses"])) if s["losses"] > 0 else (s["wins"] or 0.5)
        s["win_rate"] = s["wins"] / max(1, trades)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_signals": len(signals),
        "total_virtual_trades": len(virtuals),
        "total_blocks": len(blocks),
        "per_candidate": {k: dict(v) for k, v in per.items()},
    }

    jpath = REPORTS_DIR / f"shadow_summary_{ts}.json"
    jpath.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md_lines = [
        f"# Shadow Results Summary — {ts}",
        "",
        f"Signals: {len(signals)} | Virtual trades: {len(virtuals)} | Blocks: {len(blocks)}",
        "",
        "## Per Candidate (top by virtual pnl)",
    ]
    sorted_c = sorted(per.items(), key=lambda kv: -kv[1]["cost_adjusted_virtual_pnl"])[:5]
    for cid, s in sorted_c:
        md_lines.append(f"- {cid}: days={s['shadow_days']}, vtrades={s['virtual_trade_count']}, adj_pnl={s['cost_adjusted_virtual_pnl']:.1f}, pf={s['profit_factor']:.2f}, wr={s['win_rate']:.1%}, blocks={s['stale_data_trade_count']+s['router_silence_count']}")

    (REPORTS_DIR / f"shadow_summary_{ts}.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[shadow-summary] Wrote {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
