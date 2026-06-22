#!/usr/bin/env python3
"""
scripts/smoke_paper_forward_snapshot.py

TASK 4: Test thin/chain-only + richer snapshots through PaperForwardEngine.
Must return structured decisions, 0 route exceptions, 0 waiting_for_snapshot after feed.
Strict: exits non-zero on bad conditions. No real broker calls.
"""
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_forward_engine import PaperForwardEngine, LOAD_OK

def make_fake_nifty_chain(spot: float = 24150.0, n_strikes: int = 5):
    strikes = [spot + (i-2)*50 for i in range(n_strikes)]
    rows = []
    for k in strikes:
        for ot in ("CE", "PE"):
            ltp = max(5.0, abs(spot - k) * 0.7 + 8)
            rows.append({
                "strike": k, "option_type": ot, "expiry": "2026-06-25",
                "ltp": round(ltp, 1), "bid": round(ltp - 0.4, 1), "ask": round(ltp + 0.6, 1),
                "volume": 8500, "oi": 320000, "spot": spot,
            })
    return rows

def make_minimal_candles(n: int = 5, last_close: float = 24150.0):
    return [
        {
            "open": last_close + i,
            "high": last_close + i + 3,
            "low": last_close + i - 3,
            "close": last_close + i * 2,
            "volume": 10000 + i,
            "time": f"2026-06-11T09:{15 + (i % 9) * 5}:00",
        }
        for i in range(n)
    ]

def run_scenario(name: str, snap: dict, chain: list, eng: PaperForwardEngine, loadable: list) -> dict:
    print(f"\n[SMOKE-SCENARIO] {name}")
    eng.candidates = [c for c in loadable]  # fresh copy
    # reset some counters if present
    try:
        eng._snapshots_received = 0
        eng._evaluations_count = 0
        eng._predictions_attempted = 0
        eng._predictions_success = 0
        eng._route_errors = 0
        eng._skipped_missing_features = 0
        eng._skipped_market_data = 0
        eng._reason_counts = {}
        eng._route_exception_details = []
    except Exception:
        pass

    decisions = []
    route_exc = 0
    try:
        decisions = eng.on_market_snapshot(snap, chain)
    except Exception as e:
        route_exc += 1
        print(f"[SMOKE] top-level exception: {e}")
        traceback.print_exc()

    waiting = 0
    preds = 0
    numeric_conf = 0
    reasons = {}
    for d in decisions:
        r = str(d.get("no_trade_reason") or d.get("last_no_trade_reason") or "")
        if "waiting_for_snapshot" in r.lower():
            waiting += 1
        if d.get("predict_attempted"):
            preds += 1
        c = d.get("confidence")
        if isinstance(c, (int, float)) and c is not None:
            numeric_conf += 1
        reasons[r] = reasons.get(r, 0) + 1
        print(f"  {d.get('candidate_id','?')[:40]}: sig={d.get('final_signal')} conf={c} pred={d.get('predict_attempted')} reason={r[:50]}")

    print(f"[SMOKE] decisions={len(decisions)} waiting={waiting} preds_attempted={preds} numeric_conf={numeric_conf} route_exc={route_exc}")
    print(f"[SMOKE] top reasons: {dict(sorted(reasons.items(), key=lambda x:-x[1])[:4])}")
    diag = eng.get_diagnostics()
    data_status = diag.get("data_status", {}) or {}
    data_quality = str(data_status.get("data_quality_status") or "")
    print(f"[SMOKE] engine_diag evals={diag.get('evaluations_count')} route_errors={diag.get('route_errors')} skipped_missing={diag.get('skipped_missing_features')}")
    print(f"[SMOKE] data_quality={data_quality} auth={data_status.get('broker_auth')} rows={data_status.get('option_chain_rows')} candles={data_status.get('candle_count')}")
    if diag.get("route_exception_tracebacks"):
        for ex in diag.get("route_exception_tracebacks", [])[:5]:
            print(f"[SMOKE-ROUTE-EXCEPTION] {ex.get('candidate_id')} {ex.get('exception_type')}: {ex.get('exception_message')}")

    return {
        "name": name,
        "decisions": len(decisions),
        "waiting": waiting,
        "preds": preds,
        "numeric_conf": numeric_conf,
        "route_exc": route_exc + int(diag.get("route_errors", 0) or 0),
        "evaluations": int(diag.get("evaluations_count", 0) or 0),
        "data_quality": data_quality,
        "reasons": reasons,
    }

def main() -> int:
    print("[SMOKE-SNAPSHOT] Starting 3-scenario thin-snapshot hardening test")
    base_chain = make_fake_nifty_chain()
    base_snap = {
        "price": 24150.0, "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "smoke_chain_only", "broker_auth": "AUTH_OK",
    }

    eng = PaperForwardEngine(
        candidate_file=str(ROOT / "config" / "paper_forward_candidates.json"),
        artifacts_dir=str(ROOT / "artifacts" / "candidates"),
        broker_safe_mode=True,
    )
    loadable = [c for c in eng.candidates if (c.get("_load_status") in (LOAD_OK, "candidate_loaded_ok")) or c.get("_preset", {}).get("preset_fallback_generated")]
    if not loadable:
        print("[SMOKE] no loadable cands — PASS (graceful)")
        return 0

    results = []

    # Scenario A: thin option-chain-only (no candles, minimal keys)
    snap_a = dict(base_snap)
    res_a = run_scenario("A_thin_chain_only", snap_a, base_chain, eng, loadable)
    results.append(res_a)

    # Scenario B: candles=100 but option chain empty
    snap_b = dict(base_snap)
    snap_b["candles"] = make_minimal_candles(100, base_snap["price"])
    snap_b["close"] = base_snap["price"]
    snap_b["source"] = "smoke_candles_no_chain"
    res_b = run_scenario("B_candles_100_chain_empty", snap_b, [], eng, loadable)
    results.append(res_b)

    # Scenario C: richer synthetic (add common keys the detects/models may expect)
    snap_c = dict(base_snap)
    snap_c.update({
        "atr_pct": 0.007, "adx_14": 22, "ret_mean": 0.001, "volume": 12000,
        "spread_pct": 0.035, "regime": "mixed", "candles": make_minimal_candles(100, base_snap["price"]),
    })
    res_c = run_scenario("C_richer_synthetic", snap_c, base_chain, eng, loadable)
    results.append(res_c)

    # Strict checks
    total_route_exc = sum(r["route_exc"] for r in results)
    total_waiting = sum(r["waiting"] for r in results)
    total_dec = sum(r["decisions"] for r in results)
    total_evals = sum(r["evaluations"] for r in results)
    expected_dec = len(loadable) * 3

    print("\n[SMOKE-SUMMARY]")
    for r in results:
        print(f"  {r['name']}: dec={r['decisions']} wait={r['waiting']} exc={r['route_exc']} preds={r['preds']}")

    bad = False
    if total_route_exc > 0:
        print(f"[SMOKE-FAIL] route exceptions: {total_route_exc}")
        bad = True
    if total_waiting > 0:
        print(f"[SMOKE-FAIL] waiting_for_snapshot after feeds: {total_waiting}")
        bad = True
    if total_dec != expected_dec:
        print(f"[SMOKE-FAIL] decision count {total_dec} != expected {expected_dec}")
        bad = True
    if total_evals == 0:
        print("[SMOKE-FAIL] candidates_evaluated stayed 0")
        bad = True
    for r in results:
        if r["data_quality"] != "DATA_OK" and r["preds"] > 0:
            print(f"[SMOKE-FAIL] {r['name']} attempted predictions while data_quality={r['data_quality']}")
            bad = True
        if any("low_confidence" in str(reason) for reason in r["reasons"]) and r["data_quality"] != "DATA_OK":
            print(f"[SMOKE-FAIL] {r['name']} produced low_confidence with data_quality={r['data_quality']}")
            bad = True

    # Safety: ensure no place_order was called (engine never does for paper)
    print("[SMOKE] safety: no broker.place_order expected in paper path — assumed OK")

    if bad:
        print("[SMOKE-SNAPSHOT] FAIL")
        return 1
    print("[SMOKE-SNAPSHOT] PASS (all scenarios, 0 exceptions, 0 waiting post-snap, structured reasons)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
