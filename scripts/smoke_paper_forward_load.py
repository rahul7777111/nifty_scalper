#!/usr/bin/env python3
"""
scripts/smoke_paper_forward_load.py

PHASE 12 smoke: load candidates + presets + artifacts.
Prints table of load status.
Exits non-zero only if ZERO candidates can load (OK or fallback).
Safe: no broker, no orders.
"""
import json
import sys
from pathlib import Path

# src import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_forward_engine import (
    load_candidate_registry,
    load_dynamic_presets,
    resolve_candidate_artifacts,
    resolve_candidate_preset,
    LOAD_OK,
)

def main() -> int:
    print("[SMOKE-LOAD] Starting paper forward load smoke")
    reg = load_candidate_registry(ROOT)
    print(f"[SMOKE-LOAD] registry entries: {len(reg)}")

    preset_reg = load_dynamic_presets(ROOT)
    print(f"[SMOKE-LOAD] presets source: {preset_reg.get('source')} loaded={preset_reg.get('loaded')}")

    cand_file = ROOT / "config" / "paper_forward_candidates.json"
    cfg = json.loads(cand_file.read_text(encoding="utf-8"))
    cands = [c for c in cfg.get("candidates", []) if c.get("paper_forward_only") or c.get("classification") == "paper_forward_only"]

    ok = 0
    fallback = 0
    failed = 0
    print("\ncandidate_id | model | preset | artifact_status | preset_status | resolved")
    print("-" * 110)
    for c in cands:
        cid = c.get("candidate_id")
        art = resolve_candidate_artifacts(c, ROOT)
        pres = resolve_candidate_preset(c, preset_reg)
        astat = art.get("load_status") or art.get("status", "candidate_missing_artifact_paths")
        pstat = "OK" if pres.get("preset_loaded_ok") else ("FALLBACK" if pres.get("preset_fallback_generated") else "MISSING")
        if astat in (LOAD_OK, "candidate_loaded_ok"):
            ok += 1
        elif pres.get("preset_fallback_generated"):
            fallback += 1
        else:
            failed += 1
        print(f"{cid[:45]:<46} | {c.get('model_name','?'):<12} | {c.get('preset_family','?'):<18} | {astat:<28} | {pstat:<9} | {art.get('resolved_dir')}")
    print("-" * 110)
    print(f"SUMMARY: OK={ok} FALLBACK={fallback} FAILED={failed} total={len(cands)}")
    if ok + fallback == 0:
        print("[SMOKE-LOAD] FAIL: zero loadable candidates")
        return 2
    print("[SMOKE-LOAD] PASS (paper safe)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
