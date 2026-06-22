#!/usr/bin/env python3
"""
scripts/run_paper_forward_multi_live.py

Multi-candidate paper-forward live observation runner.
All trades simulated. Hard safety: no real orders.

Usage examples:
  # Dry run with one fake snapshot (recommended first)
  python scripts/run_paper_forward_multi_live.py --candidate-file config/paper_forward_candidates.json --paper-only --no-real-orders --dry-run-one-snapshot --write-jsonl

  # Live mode (provide your own snapshot loop or integrate with broker feed)
  python scripts/run_paper_forward_multi_live.py --candidate-file config/paper_forward_candidates.json --paper-only --no-real-orders --write-jsonl --snapshot-interval-seconds 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from paper_forward_engine import PaperForwardEngine

def load_candidate_ids_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    cands = data.get("candidates", data.get("selected_candidates", []))
    ids = []
    for c in cands:
        if c.get("paper_forward_only") or c.get("classification") == "paper_forward_only" or c.get("recommended_mode") == "paper_forward_only":
            if c.get("enabled", True):
                ids.append(c["candidate_id"])
    return ids

def main():
    parser = argparse.ArgumentParser(description="Multi-candidate paper-forward live observation (SIM ONLY)")
    parser.add_argument("--candidate-file", type=str, default="config/paper_forward_candidates.json")
    parser.add_argument("--candidate-id", action="append", default=[], help="Repeatable. Limits to these paper_forward_only ids")
    parser.add_argument("--all-paper-forward", action="store_true", help="Use all paper_forward_only from candidate-file")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/candidates")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--paper-only", action="store_true", required=True, help="REQUIRED for safety")
    parser.add_argument("--no-real-orders", action="store_true", required=True, help="REQUIRED for safety")
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--dry-run-one-snapshot", action="store_true")
    parser.add_argument("--snapshot-interval-seconds", type=int, default=5)
    parser.add_argument("--max-runtime-minutes", type=int, default=0)
    args = parser.parse_args()

    if not args.paper_only:
        print("[FATAL] --paper-only is required")
        sys.exit(1)
    if not args.no_real_orders:
        print("[FATAL] --no-real-orders is required")
        sys.exit(1)

    # Hard env check
    if os.getenv("MSTOCK_ENABLE_LIVE_ORDERS", "false").lower() in ("1", "true", "yes"):
        print("[FATAL] MSTOCK_ENABLE_LIVE_ORDERS=true detected. Refusing.")
        sys.exit(1)

    # Load candidates
    ids: List[str] = []
    if args.candidate_id:
        ids = args.candidate_id
    else:
        ids = load_candidate_ids_from_file(args.candidate_file)
        if args.all_paper_forward:
            pass  # already filtered in loader

    if args.max_candidates > 0:
        ids = ids[:args.max_candidates]

    if not ids:
        print("[FATAL] No paper_forward_only candidates found/selected")
        sys.exit(1)

    print(f"[PAPER-FORWARD-MULTI] real_orders=false candidates={len(ids)} ids={ids[:3]}...")

    engine = PaperForwardEngine(
        candidate_ids=ids,
        candidate_file=args.candidate_file if not args.candidate_id else None,
        artifacts_dir=args.artifacts_dir,
        log_dir=args.log_dir,
        reports_dir=args.reports_dir,
        broker_safe_mode=True,
    )

    start = time.time()
    snap_count = 0

    if args.dry_run_one_snapshot:
        fake_snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 24050.0,
            "regime": "TRENDING",
            "market_regime": "TRENDING",
            "last": 24050.0,
            "close": 24050.0,
        }
        fake_chain = {"best_bid": 82.3, "best_ask": 82.7, "volume": 150000}
        decisions = engine.on_market_snapshot(fake_snap, fake_chain)
        for d in decisions:
            if args.write_jsonl:
                engine.write_jsonl(d)
        engine.write_summary()
        print(f"[DRY] Processed {len(decisions)} candidates. Summary written.")
        return

    # Simple live loop (user can replace with real feed)
    print("[INFO] Starting simple snapshot loop (replace with real broker feed for production paper obs)")
    try:
        while True:
            snap_count += 1
            snap = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": 24000.0 + (snap_count % 50) * 2.0,  # toy moving price
                "regime": "CHOPPY",
                "market_regime": "CHOPPY",
                "last": 24000.0 + (snap_count % 50) * 2.0,
            }
            chain = {"best_bid": 80.0 + (snap_count % 10) * 0.1, "best_ask": 80.5 + (snap_count % 10) * 0.1, "volume": 120000}
            decisions = engine.on_market_snapshot(snap, chain)
            for d in decisions:
                if args.write_jsonl:
                    engine.write_jsonl(d)
            if snap_count % 10 == 0:
                engine.write_summary()
            if args.max_runtime_minutes > 0 and (time.time() - start) > args.max_runtime_minutes * 60:
                break
            time.sleep(max(0.1, args.snapshot_interval_seconds))
    except KeyboardInterrupt:
        print("[INFO] Stopped by user")
    finally:
        engine.write_summary()
        print("[INFO] Final summary written. All simulated paper only.")

if __name__ == "__main__":
    main()
