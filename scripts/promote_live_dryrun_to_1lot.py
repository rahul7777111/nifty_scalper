#!/usr/bin/env python3
"""
scripts/promote_live_dryrun_to_1lot.py
======================================
Reads the live_candidate_whitelist.json (populated by shadow->dryrun promotion).
Examines live_dryrun_orders_*.jsonl for payload volume and error counts.
Promotes the (single) LIVE_DRY_RUN candidate to LIVE_1_LOT only if ALL of:
  - dryrun_days >= 2
  - dryrun_payload_count >= configured minimum (default 5)
  - zero payload errors
  - zero risk bypasses (inferred from absence of risk flags in dry-run records)
  - zero accidental live orders (none of the records have live_order_sent=true)
  - (broker reconciliation + emergency stop tested are assumed if the above pass; real ops sign-off still required)

Writes:
  - config/live_candidate_whitelist.json (updated status + live_whitelisted)
  - reports/live_dryrun_to_1lot_promotion_YYYYMMDD_HHMMSS.json
  - reports/live_dryrun_to_1lot_promotion_YYYYMMDD_HHMMSS.md

Per spec: do not promote more than one candidate to LIVE_1_LOT without explicit approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.candidate_lifecycle import (
    Candidate,
    load_live_whitelist,
    save_live_whitelist,
    promote_candidate,
    STATUS_LIVE_DRY_RUN,
    STATUS_LIVE_1_LOT,
    LIVE_WHITELIST_PATH,
)

LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "config"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_dryrun_logs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(LOGS_DIR.glob("live_dryrun_orders_*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def aggregate_dryrun(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    per: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "days": set(),
        "payload_count": 0,
        "payload_errors": 0,
        "live_order_sent_count": 0,
        "risk_bypass_flags": 0,
    })
    for r in rows:
        cid = r.get("candidate_id") or "unknown"
        s = per[cid]
        ts = r.get("timestamp") or r.get("recorded_at")
        if ts:
            try:
                s["days"].add(str(ts)[:10])
            except Exception:
                pass
        s["payload_count"] += 1
        if r.get("validation_error"):
            s["payload_errors"] += 1
        if r.get("live_order_sent"):
            s["live_order_sent_count"] += 1
        # risk bypass would appear in a real decision record; in pure dry-run we look for explicit flags
        if "risk" in str(r.get("validation_error", "")).lower() or r.get("risk_bypass"):
            s["risk_bypass_flags"] += 1
    for s in per.values():
        s["dryrun_days"] = len(s["days"])
    return per


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dryrun-days", type=int, default=2)
    parser.add_argument("--min-payloads", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Bypass some checks (ops sign-off still required)")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    whitelist = load_live_whitelist()
    if not whitelist:
        print("[dryrun->1lot] No live_candidate_whitelist.json. Run shadow->dryrun promotion first.")
        return 0

    dry_rows = _load_dryrun_logs()
    print(f"[dryrun->1lot] Loaded {len(dry_rows)} live_dryrun order records")

    per = aggregate_dryrun(dry_rows)

    promoted: List[str] = []
    blocked: List[Tuple[str, str]] = []

    # Only consider the current LIVE_DRY_RUN candidate(s) — policy is at most one
    candidates_in_dry = [c for c in whitelist if c.status == STATUS_LIVE_DRY_RUN or c.live_whitelisted]
    if len(candidates_in_dry) > 1 and not args.force:
        # Enforce single-candidate policy
        for c in candidates_in_dry[1:]:
            c.last_block_reason = "more than one candidate in LIVE_DRY_RUN state; policy forbids promoting >1 to 1-lot without explicit approval"
            blocked.append((c.candidate_id, c.last_block_reason))

    target = candidates_in_dry[0] if candidates_in_dry else None

    if not target:
        print("[dryrun->1lot] No candidate currently at LIVE_DRY_RUN.")
        return 0

    stats = per.get(target.candidate_id, {})
    days = int(stats.get("dryrun_days", 0))
    cnt = int(stats.get("payload_count", 0))
    errs = int(stats.get("payload_errors", 0))
    sent = int(stats.get("live_order_sent_count", 0))
    risk = int(stats.get("risk_bypass_flags", 0))

    reasons: List[str] = []
    if days < args.min_dryrun_days:
        reasons.append(f"dryrun_days {days} < {args.min_dryrun_days}")
    if cnt < args.min_payloads:
        reasons.append(f"dryrun_payload_count {cnt} < {args.min_payloads}")
    if errs > 0:
        reasons.append(f"payload_errors={errs} (must be 0)")
    if sent > 0:
        reasons.append(f"live_order_sent_count={sent} (must be 0; no accidental real orders)")
    if risk > 0:
        reasons.append(f"risk_bypass_flags={risk} (must be 0)")

    ok = len(reasons) == 0 or args.force
    reason_str = "; ".join(reasons) if reasons else "all dry-run safety checks passed"

    if ok:
        promote_candidate(
            target,
            STATUS_LIVE_1_LOT,
            notes=f"dryrun->1lot: {reason_str}" + (" (FORCED - manual ops sign-off required)" if args.force else ""),
            set_live_whitelist=True,
            max_qty=1,
            max_trades=2,
            max_loss=1000.0,
        )
        promoted.append(target.candidate_id)
    else:
        target.last_block_reason = reason_str
        blocked.append((target.candidate_id, reason_str))

    # Persist whitelist (with updated status)
    ts = _ts()
    meta = {
        "run_timestamp": ts,
        "promoted_to_live_1_lot": promoted,
        "blocked": [{"candidate_id": cid, "reason": r} for cid, r in blocked],
        "policy": "At most one candidate may be LIVE_1_LOT. Real orders still require the full 8-gate matrix + ORDER_PLACEMENT_ENABLED=true + LIVE_ORDER_DRY_RUN=false.",
    }
    save_live_whitelist(whitelist, meta=meta)

    # Reports
    j = REPORTS_DIR / f"live_dryrun_to_1lot_promotion_{ts}.json"
    j.write_text(json.dumps({
        "timestamp": ts,
        "promoted_to_live_1_lot": promoted,
        "blocked_from_1_lot": [{"candidate_id": cid, "reason": r} for cid, r in blocked],
        "live_whitelist_file": str(LIVE_WHITELIST_PATH),
        "dryrun_records_analyzed": len(dry_rows),
        "meta": meta,
    }, indent=2), encoding="utf-8")

    md = REPORTS_DIR / f"live_dryrun_to_1lot_promotion_{ts}.md"
    blocked_lines = [f"- `{cid}`: {r}" for cid, r in blocked] if blocked else ["- (none)"]
    md.write_text("\n".join([
        f"# Live Dry-Run → LIVE_1_LOT Promotion — {ts}",
        "",
        f"**Promoted:** {', '.join(promoted) if promoted else 'none'}",
        "",
        "## Blocked",
    ] + blocked_lines + [
        "",
        f"dry-run records examined: {len(dry_rows)}",
        "",
        "NEXT MANUAL STEPS (do NOT automate):",
        "1. Verify broker reconciliation functions on a paper or micro position.",
        "2. Test emergency stop end-to-end (SCALPER_KILL_SWITCH=1 must halt entries and flatten).",
        "3. Confirm ORDER_PLACEMENT_ENABLED remains false until the operator is physically present.",
        "4. Set LIVE_ORDER_DRY_RUN=false and ORDER_PLACEMENT_ENABLED=true only after the above.",
        "5. Start with 1 lot, LIMIT orders, tight time window, and a human ready on the kill switch.",
    ]), encoding="utf-8")

    print(f"[dryrun->1lot] Promoted: {promoted}")
    print(f"[dryrun->1lot] Wrote {LIVE_WHITELIST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
