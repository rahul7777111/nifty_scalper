#!/usr/bin/env python3
"""
scripts/promote_shadow_to_live_dryrun.py
========================================
Reads shadow_candidates.json + shadow_*.jsonl logs + any shadow_summary reports.
Applies strict gates.
Promotes AT MOST ONE candidate to LIVE_DRY_RUN (the best/safest).
Writes:
  - config/live_candidate_whitelist.json  (only the chosen one has live_whitelisted=true + status LIVE_DRY_RUN)
  - reports/shadow_to_live_dryrun_promotion_YYYYMMDD_HHMMSS.json
  - reports/shadow_to_live_dryrun_promotion_YYYYMMDD_HHMMSS.md

Default rule (per spec): "Only the single best and safest candidate should be enabled for live dry-run.
All other candidates remain shadow-only."
"""

from __future__ import annotations

import argparse
import json
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
    save_live_whitelist,
    evaluate_shadow_to_live_dryrun,
    select_best_for_live_dryrun,
    promote_candidate,
    STATUS_SHADOW,
    STATUS_SHADOW_PASSED,
    STATUS_LIVE_DRY_RUN,
    STATUS_LIVE_1_LOT,
    STATUS_LIVE_SCALED,
    LIVE_WHITELIST_PATH,
)

LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "config"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_jsonl_glob(pattern: str) -> List[Dict[str, Any]]:
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


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aggregate_shadow_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    per: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "shadow_days": 0,
        "shadow_trade_count": 0,
        "stale_data_trade_count": 0,
        "invalid_expiry_trade_count": 0,
        "duplicate_trade_count": 0,
        "missing_sl_count": 0,
        "missing_exit_count": 0,
        "risk_bypass_count": 0,
        "router_silence_count": 0,
        "order_payload_error_count": 0,
        "emergency_stop_events": 0,
        "cost_adjusted_pnl": 0.0,
        "max_drawdown": 0.0,
        "dates": set(),
    })
    for r in rows:
        cid = r.get("candidate_id") or "unknown"
        s = per[cid]
        ts = r.get("timestamp") or r.get("ts")
        if ts:
            try:
                s["dates"].add(str(ts)[:10])
            except Exception:
                pass
        # Count virtual trades
        if r.get("virtual_entry_price") or r.get("side_decision") or "ENTER" in str(r.get("router_reason", "")):
            s["shadow_trade_count"] += 1
        # Error signals from shadow logs
        br = str(r.get("blocked_reason", "")).lower()
        if "stale" in br:
            s["stale_data_trade_count"] += 1
        if "invalid" in br and "expir" in br:
            s["invalid_expiry_trade_count"] += 1
        if "duplicate" in br:
            s["duplicate_trade_count"] += 1
        if "missing_sl" in br or "no sl" in br:
            s["missing_sl_count"] += 1
        if "missing_exit" in br:
            s["missing_exit_count"] += 1
        if "risk" in br and "bypass" in br:
            s["risk_bypass_count"] += 1
        if "router" in br and "silent" in br:
            s["router_silence_count"] += 1
        if "payload" in br and "error" in br:
            s["order_payload_error_count"] += 1
        if "emergency" in br:
            s["emergency_stop_events"] += 1
        if r.get("virtual_pnl") is not None:
            try:
                s["cost_adjusted_pnl"] += float(r.get("virtual_pnl") or 0.0)
            except Exception:
                pass
        s["shadow_days"] = len(s["dates"])
    return per


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-shadow-days", type=int, default=1)
    parser.add_argument("--min-shadow-trades", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load current shadow candidates
    shadow_cands: List[Candidate] = load_shadow_candidates()
    if not shadow_cands:
        print("[shadow->dryrun] No shadow_candidates.json found or empty. Nothing to promote.")
        return 0

    # Load shadow logs
    shadow_rows = _load_jsonl_glob("shadow_*.jsonl")
    print(f"[shadow->dryrun] Loaded {len(shadow_rows)} shadow log rows for {len(shadow_cands)} candidates")

    per_stats = aggregate_shadow_stats(shadow_rows)

    promoted: List[str] = []
    blocked: List[Tuple[str, str]] = []

    # First pass: mark who is eligible on paper
    eligible: List[Candidate] = []
    for c in shadow_cands:
        if c.status not in (STATUS_SHADOW, STATUS_SHADOW_PASSED):
            c.last_block_reason = f"current status {c.status} not eligible for live dry-run promotion"
            blocked.append((c.candidate_id, c.last_block_reason))
            continue

        stats = per_stats.get(c.candidate_id, {})
        ok, reason = evaluate_shadow_to_live_dryrun(
            stats or {},
            min_shadow_days=args.min_shadow_days,
            min_shadow_trades=args.min_shadow_trades,
        )
        if ok or args.force:
            eligible.append(c)
            c.shadow_days = int(stats.get("shadow_days", 0))
            c.shadow_trade_count = int(stats.get("shadow_trade_count", 0))
            c.shadow_cost_adjusted_pnl = float(stats.get("cost_adjusted_pnl", 0.0))
        else:
            c.last_block_reason = reason
            blocked.append((c.candidate_id, reason))

    # Select at most one
    best = select_best_for_live_dryrun(eligible) if eligible else None
    if best or args.force:
        target = best or (eligible[0] if eligible else shadow_cands[0])
        promote_candidate(
            target,
            STATUS_LIVE_DRY_RUN,
            notes="shadow->live_dry_run: single best/safest candidate per policy" + (" (FORCED)" if args.force else ""),
            set_live_whitelist=True,
            max_qty=1,
            max_trades=2,
            max_loss=1500.0,
        )
        promoted.append(target.candidate_id)
        # All others that were eligible but not chosen stay in shadow
        for c in eligible:
            if c.candidate_id != target.candidate_id:
                c.last_block_reason = "not the single best/safest; kept in shadow-only"
                blocked.append((c.candidate_id, c.last_block_reason))
    else:
        print("[shadow->dryrun] No candidate met the strict shadow->dry-run gates.")

    # Write the live whitelist (only those with live_whitelisted or LIVE_DRY_RUN status)
    live_list = [c for c in shadow_cands if c.live_whitelisted or c.status in (STATUS_LIVE_DRY_RUN, STATUS_LIVE_1_LOT, STATUS_LIVE_SCALED)]
    # If nothing qualified, still write a (mostly empty) whitelist file for auditability
    ts = _ts()
    meta = {
        "run_timestamp": ts,
        "promoted_to_live_dry_run": promoted,
        "blocked_from_live_dry_run": [{"candidate_id": cid, "reason": r} for cid, r in blocked],
        "policy": "only the single best and safest candidate may receive LIVE_DRY_RUN + live_whitelisted=true",
    }
    save_live_whitelist(live_list, meta=meta)

    # Reports
    jpath = REPORTS_DIR / f"shadow_to_live_dryrun_promotion_{ts}.json"
    jpath.write_text(json.dumps({
        "timestamp": ts,
        "promoted_to_live_dry_run": promoted,
        "blocked_from_live_dry_run": [{"candidate_id": cid, "reason": r} for cid, r in blocked],
        "live_whitelist_file": str(LIVE_WHITELIST_PATH),
        "meta": meta,
    }, indent=2), encoding="utf-8")

    md = REPORTS_DIR / f"shadow_to_live_dryrun_promotion_{ts}.md"
    md_lines = [
        f"# Shadow → Live Dry-Run Promotion — {ts}",
        "",
        f"**Promoted to LIVE_DRY_RUN (at most 1):** {', '.join(promoted) if promoted else 'none'}",
        "",
        "## Blocked / Kept in Shadow",
    ]
    for cid, r in blocked:
        md_lines.append(f"- `{cid}`: {r}")
    if not blocked:
        md_lines.append("- (none)")
    md_lines.extend([
        "",
        f"**live_candidate_whitelist.json** written to: `{LIVE_WHITELIST_PATH}`",
        "",
        "Safety: LIVE_ORDER_DRY_RUN path only. Real orders still require ORDER_PLACEMENT_ENABLED=false (or the full gate matrix).",
    ])
    md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[shadow->dryrun] Promoted to dry-run: {promoted}")
    print(f"[shadow->dryrun] Wrote {LIVE_WHITELIST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
