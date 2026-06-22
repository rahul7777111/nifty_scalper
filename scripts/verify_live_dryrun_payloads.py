#!/usr/bin/env python3
"""
scripts/verify_live_dryrun_payloads.py
======================================
Audit all live_dryrun_orders_*.jsonl for correctness and safety invariants.

Never marks anything as "sent". Reports violations.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.candidate_lifecycle import load_live_whitelist

LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_all_dryrun() -> List[Dict[str, Any]]:
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


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _ts()

    rows = _load_all_dryrun()
    wl = {c.candidate_id: c for c in load_live_whitelist()}
    violations: List[Dict[str, Any]] = []
    ok_count = 0

    for r in rows:
        vios = []
        cid = r.get("candidate_id", "unknown")

        # Basic field presence and sanity
        sym = str(r.get("symbol", "")).strip()
        if not sym or ":" in sym and not sym.startswith(("NSE:", "NFO:", "BSE:")):
            vios.append("symbol format invalid or missing exchange prefix when needed")

        exch = str(r.get("exchange", "")).upper()
        if exch not in ("NSE", "NFO", "BSE", "NSEFO"):
            vios.append(f"exchange {exch} not in allowed set")

        exp = str(r.get("expiry", ""))
        if not exp or len(exp) < 8:
            vios.append("expiry missing or too short")

        ot = str(r.get("option_type", "")).upper()
        if ot not in ("CE", "PE"):
            vios.append(f"option_type {ot} invalid")

        qty = int(r.get("quantity", 0) or 0)
        if qty <= 0 or qty % 65 != 0:  # NIFTY lot multiple
            vios.append(f"quantity {qty} not multiple of lot size")

        otype = str(r.get("order_type", "")).upper()
        if otype != "LIMIT":
            vios.append(f"order_type must be LIMIT in dry-run/first-live (got {otype})")

        price = r.get("price")
        if price is not None and (float(price) <= 0 or float(price) > 10000):
            vios.append(f"price {price} looks insane")

        if r.get("live_order_sent"):
            vios.append("dry-run record incorrectly has live_order_sent=true")

        # Candidate check
        c = wl.get(cid)
        if not c or not c.live_whitelisted:
            vios.append("candidate not whitelisted in live_candidate_whitelist.json")

        if r.get("validation_error"):
            vios.append(f"validation_error present: {r['validation_error']}")

        if vios:
            violations.append({"record": r, "violations": vios})
        else:
            ok_count += 1

    verdict = "PASS" if not violations else "FAIL"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "total_records": len(rows),
        "clean": ok_count,
        "violations": violations,
        "whitelist_candidates": list(wl.keys()),
    }

    jpath = REPORTS_DIR / f"live_dryrun_payload_audit_{ts}.json"
    jpath.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md_lines = [
        f"# Live Dry-Run Payload Audit — {ts}",
        "",
        f"**VERDICT: {verdict}**",
        f"Total records: {len(rows)} | Clean: {ok_count} | Violations: {len(violations)}",
        "",
    ]
    if violations:
        md_lines.append("## Violations")
        for v in violations[:20]:  # cap output
            md_lines.append(f"- candidate={v['record'].get('candidate_id')} : {'; '.join(v['violations'])}")
    else:
        md_lines.append("All dry-run payloads passed structural, whitelist, and first-live rules.")

    md_lines.extend([
        "",
        "Rules enforced: symbol/exchange/expiry/option_type, qty lot multiple, LIMIT only, no live_order_sent=true, candidate whitelisted.",
    ])
    (REPORTS_DIR / f"live_dryrun_payload_audit_{ts}.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[verify-dryrun] {verdict} - {ok_count}/{len(rows)} clean")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
