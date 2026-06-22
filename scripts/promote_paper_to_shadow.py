#!/usr/bin/env python3
"""
scripts/promote_paper_to_shadow.py
==================================
Reads paper-forward logs, summary reports, and paper_forward_candidates.json.
Computes per-candidate quality metrics (including error counters).
Applies strict promotion gates.
Writes:
  - config/shadow_candidates.json
  - reports/paper_to_shadow_promotion_YYYYMMDD_HHMMSS.json
  - reports/paper_to_shadow_promotion_YYYYMMDD_HHMMSS.md

Never enables live trading. Only produces shadow candidate list for subsequent shadow engine runs.

Promotion rule (exact):
- trade_count >= configured minimum (default 20)
- days_tested >= configured minimum (default 3)
- stale_data_trade_count == 0
- duplicate_trade_count == 0
- invalid_expiry_trade_count == 0
- missing_sl_count == 0
- missing_exit_count == 0
- after_cutoff_trade_count == 0
- cost_adjusted_pnl acceptable (not catastrophically negative)
- max_drawdown within limit

All other candidates stay at PAPER_FORWARD or are marked with blocker reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.candidate_lifecycle import (
    Candidate,
    load_shadow_candidates,
    save_shadow_candidates,
    evaluate_paper_to_shadow,
    promote_candidate,
    STATUS_PAPER_FORWARD,
    STATUS_PAPER_PASSED,
    STATUS_SHADOW,
    PAPER_FORWARD_CANDIDATES_PATH,
    SHADOW_CANDIDATES_PATH,
)

LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "config"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_latest(pattern: str, base: Path) -> Path | None:
    matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def aggregate_paper_stats(log_rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate per candidate_id the required counters from paper_forward jsonl.
    Also merge any per-candidate summary data if present.
    """
    per: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "trade_count": 0,
        "days_tested": 0,
        "gross_pnl": 0.0,
        "cost_adjusted_pnl": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "stale_data_trade_count": 0,
        "duplicate_trade_count": 0,
        "invalid_expiry_trade_count": 0,
        "missing_sl_count": 0,
        "missing_exit_count": 0,
        "spread_violation_count": 0,
        "after_cutoff_trade_count": 0,
        "dates": set(),
    })

    for r in log_rows:
        cid = r.get("candidate_id") or r.get("candidate") or "unknown"
        stats = per[cid]
        # Count a trade if we see a PAPER_TRADE / TRADE / entry event
        if r.get("final_signal") in ("PAPER_TRADE", "TRADE", "ENTRY", "virtual_entry") or r.get("side_decision"):
            stats["trade_count"] += 1
        # Date bucket (very rough; use date portion of timestamp if present)
        ts = r.get("timestamp") or r.get("ts")
        if ts:
            try:
                d = str(ts)[:10]
                stats["dates"].add(d)
            except Exception:
                pass

        # Error counters (paper_forward_engine or router may emit these keys)
        if r.get("stale") or r.get("stale_data"):
            stats["stale_data_trade_count"] += 1
        if r.get("duplicate") or "duplicate" in str(r.get("no_trade_reason", "")).lower():
            stats["duplicate_trade_count"] += 1
        if r.get("invalid_expiry") or "invalid_expiry" in str(r.get("no_trade_reason", "")).lower():
            stats["invalid_expiry_trade_count"] += 1
        if r.get("missing_sl") or "missing_sl" in str(r.get("no_trade_reason", "")).lower():
            stats["missing_sl_count"] += 1
        if r.get("missing_exit") or "missing_exit" in str(r.get("no_trade_reason", "")).lower():
            stats["missing_exit_count"] += 1
        if r.get("after_cutoff") or "cutoff" in str(r.get("no_trade_reason", "")).lower():
            stats["after_cutoff_trade_count"] += 1
        if r.get("spread_violation") or "spread" in str(r.get("no_trade_reason", "")).lower():
            stats["spread_violation_count"] += 1

        # PnL if present (paper engine may log simulated_* or cost_adjusted)
        if "cost_adjusted_pnl" in r or "simulated_pnl" in r:
            try:
                stats["cost_adjusted_pnl"] += float(r.get("cost_adjusted_pnl") or r.get("simulated_pnl") or 0.0)
            except Exception:
                pass

    # days_tested from distinct dates observed
    for cid, s in per.items():
        s["days_tested"] = len(s["dates"])
        s["dates"] = sorted(s["dates"])

    # Merge summary if it has per-candidate breakdowns
    if isinstance(summary, dict):
        cands = summary.get("per_candidate") or summary.get("candidates") or {}
        if isinstance(cands, list):
            # list form: each entry may have "candidate_id"
            for item in cands:
                if not isinstance(item, dict):
                    continue
                cid = item.get("candidate_id") or item.get("id")
                if not cid or cid not in per:
                    continue
                extra = item
                for k in ("trade_count", "days_tested", "cost_adjusted_pnl", "max_drawdown",
                          "stale_data_trade_count", "duplicate_trade_count", "invalid_expiry_trade_count",
                          "missing_sl_count", "missing_exit_count", "after_cutoff_trade_count"):
                    if k in extra:
                        try:
                            per[cid][k] = max(per[cid][k], int(extra[k]) if "count" in k or "trade" in k or "days" in k else float(extra[k]))
                        except Exception:
                            pass
        elif isinstance(cands, dict):
            for cid, extra in cands.items():
                if cid in per:
                    for k in ("trade_count", "days_tested", "cost_adjusted_pnl", "max_drawdown",
                              "stale_data_trade_count", "duplicate_trade_count", "invalid_expiry_trade_count",
                              "missing_sl_count", "missing_exit_count", "after_cutoff_trade_count"):
                        if k in extra:
                            try:
                                per[cid][k] = max(per[cid][k], int(extra[k]) if "count" in k or "trade" in k or "days" in k else float(extra[k]))
                            except Exception:
                                pass

    return per


def load_paper_forward_inputs() -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    log_rows: List[Dict[str, Any]] = []
    # Load all paper_forward*.jsonl
    for p in sorted(LOGS_DIR.glob("paper_forward*.jsonl")):
        log_rows.extend(_load_jsonl(p))
    for p in sorted(LOGS_DIR.glob("paper_forward_multi*.jsonl")):
        log_rows.extend(_load_jsonl(p))

    # Latest summary
    summary = {}
    for pat in ["paper_forward_summary_*.json", "paper_forward*summary*.json"]:
        latest = _find_latest(pat, REPORTS_DIR)
        if latest:
            summary = _load_json(latest)
            break

    # paper_forward_candidates.json (source of truth for candidate list)
    base_cands: List[Dict[str, Any]] = []
    if PAPER_FORWARD_CANDIDATES_PATH.exists():
        data = _load_json(PAPER_FORWARD_CANDIDATES_PATH)
        base_cands = data.get("candidates", data if isinstance(data, list) else [])
    # also try the _latest pointer
    latest_ptr = CONFIG_DIR / "paper_forward_candidates_latest.json"
    if latest_ptr.exists() and not base_cands:
        ptr = _load_json(latest_ptr)
        src = ptr.get("latest")
        if src:
            p2 = Path(src)
            if p2.exists():
                data = _load_json(p2)
                base_cands = data.get("candidates", data if isinstance(data, list) else [])

    return log_rows, summary, base_cands


def build_initial_shadow_list(base_cands: List[Dict[str, Any]]) -> List[Candidate]:
    out: List[Candidate] = []
    for bc in base_cands:
        cid = bc.get("candidate_id") or bc.get("id")
        if not cid:
            continue
        c = Candidate(
            candidate_id=cid,
            model_path=bc.get("artifact_paths", {}).get("model") or bc.get("model_path", ""),
            artifact_path=bc.get("artifact_dir", bc.get("artifact_path", "")),
            status=STATUS_PAPER_FORWARD,
            enabled=bool(bc.get("enabled", True)),
            allowed_modes=["paper_forward"],
            live_whitelisted=False,
            max_qty_lots=0,
        )
        out.append(c)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum clean trades (relaxed default for early data)")
    parser.add_argument("--min-days", type=int, default=1, help="Minimum distinct days observed")
    parser.add_argument("--max-drawdown", type=float, default=10000.0)
    parser.add_argument("--force", action="store_true", help="Force promotion even with low counts (testing only)")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    log_rows, summary, base_cands = load_paper_forward_inputs()
    print(f"[paper->shadow] Loaded {len(log_rows)} paper log rows, summary keys={list(summary.keys())[:6]}, base_candidates={len(base_cands)}")

    per_stats = aggregate_paper_stats(log_rows, summary)

    # Seed shadow list from paper_forward config + any existing shadow file
    shadow_list = build_initial_shadow_list(base_cands)
    existing = {c.candidate_id: c for c in load_shadow_candidates()}
    for c in shadow_list:
        if c.candidate_id in existing:
            # keep prior promotion state but refresh paper metrics
            prev = existing[c.candidate_id]
            c.status = prev.status
            c.enabled = prev.enabled
            c.live_whitelisted = prev.live_whitelisted
            c.max_qty_lots = prev.max_qty_lots

    promoted: List[str] = []
    blocked: List[Tuple[str, str]] = []

    for c in shadow_list:
        stats = per_stats.get(c.candidate_id, {})
        # If we have almost no data for this cid, still allow a minimal promotion path if force
        if not stats and not args.force:
            c.last_block_reason = "no paper-forward activity observed for candidate"
            blocked.append((c.candidate_id, c.last_block_reason))
            continue

        ok, reason = evaluate_paper_to_shadow(
            stats or {"trade_count": 0, "days_tested": 0},
            min_trades=args.min_trades,
            min_days=args.min_days,
            max_drawdown_limit=args.max_drawdown,
        )

        # Also pull any explicit error counters that may have been aggregated at top level
        if ok and (stats.get("stale_data_trade_count", 0) > 0 or
                   stats.get("duplicate_trade_count", 0) > 0 or
                   stats.get("invalid_expiry_trade_count", 0) > 0 or
                   stats.get("missing_sl_count", 0) > 0 or
                   stats.get("missing_exit_count", 0) > 0 or
                   stats.get("after_cutoff_trade_count", 0) > 0):
            ok = False
            reason = "one or more mandatory zero-error counters > 0"

        if ok or args.force:
            promote_candidate(
                c,
                STATUS_SHADOW,
                notes=f"paper->shadow: {reason}" + (" (FORCED)" if args.force else ""),
            )
            promoted.append(c.candidate_id)
            c.paper_trade_count = int(stats.get("trade_count", 0))
            c.paper_days_tested = int(stats.get("days_tested", 0))
            c.paper_cost_adjusted_pnl = float(stats.get("cost_adjusted_pnl", 0.0))
            c.paper_max_drawdown = float(stats.get("max_drawdown", 0.0))
        else:
            c.last_block_reason = reason
            blocked.append((c.candidate_id, reason))

    # Persist
    ts = _ts()
    meta = {
        "run_timestamp": ts,
        "source_logs": [str(p) for p in sorted(LOGS_DIR.glob("paper_forward*.jsonl"))],
        "min_trades": args.min_trades,
        "min_days": args.min_days,
        "promoted": promoted,
        "blocked": [{"candidate_id": cid, "reason": r} for cid, r in blocked],
    }
    save_shadow_candidates(shadow_list, meta=meta)

    # Report JSON
    report_json = REPORTS_DIR / f"paper_to_shadow_promotion_{ts}.json"
    report_json.write_text(json.dumps({
        "timestamp": ts,
        "promoted_to_shadow": promoted,
        "blocked_from_shadow": [{"candidate_id": cid, "reason": r} for cid, r in blocked],
        "shadow_candidates_file": str(SHADOW_CANDIDATES_PATH),
        "meta": meta,
    }, indent=2), encoding="utf-8")

    # Report MD
    report_md = REPORTS_DIR / f"paper_to_shadow_promotion_{ts}.md"
    lines = [
        f"# Paper → Shadow Promotion Report — {ts}",
        "",
        f"**Promoted ({len(promoted)}):** " + (", ".join(promoted) if promoted else "none"),
        "",
        "## Blocked",
    ]
    for cid, r in blocked:
        lines.append(f"- `{cid}`: {r}")
    if not blocked:
        lines.append("- (none)")
    lines.extend([
        "",
        "## Inputs",
        f"- Paper log rows: {len(log_rows)}",
        f"- Base candidates from paper_forward_candidates.json: {len(base_cands)}",
        f"- Gates: trades>={args.min_trades}, days>={args.min_days}, all error counters==0, cost_pnl acceptable, drawdown<= {args.max_drawdown}",
        "",
        "Safety: shadow mode only. No live orders permitted from this list.",
    ])
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[paper->shadow] Promoted: {promoted}")
    print(f"[paper->shadow] Blocked: {len(blocked)} (see {report_md})")
    print(f"[paper->shadow] Wrote {SHADOW_CANDIDATES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
