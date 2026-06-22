#!/usr/bin/env python3
"""Offline verification for Paper Forward prediction pipeline.

Uses config/paper_forward_candidates.json and a synthetic live-like snapshot.
Asserts candidates load, artifacts resolve, features align, prediction runs when
DATA_OK, and UI mapping keys match prediction candidate_id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import (  # noqa: E402
    PaperForwardEngine,
    build_paper_forward_feature_frame,
    format_paper_forward_confidence,
)


def _synthetic_snapshot() -> dict:
    candles = [
        {"open": 23490 + i, "high": 23530 + i, "low": 23480 + i, "close": 23505 + i, "volume": 70000 + i}
        for i in range(100)
    ]
    chain = [
        {
            "strike": 23500 + i * 50,
            "option_type": "CE" if i % 2 == 0 else "PE",
            "ltp": 118.5,
            "bid": 116.0,
            "ask": 121.0,
            "iv": 0.165,
            "volume": 1000,
            "oi": 5000,
            "expiry": "2026-06-19",
        }
        for i in range(224)
    ]
    return {
        "spot": 23512.0,
        "price": 23512.0,
        "broker_auth": "AUTH_OK",
        "auth_status": "AUTH_OK",
        "data_quality_status": "READY_FOR_PREDICTION",
        "chain_source": "BLACK_SCHOLES_SYNTHETIC",
        "candle_count": len(candles),
        "candles": candles,
        "option_chain": chain,
    }


def main() -> int:
    cfg_path = REPO / "config" / "paper_forward_candidates.json"
    if not cfg_path.exists():
        print(f"FAIL missing config {cfg_path}")
        return 2

    snap = _synthetic_snapshot()
    eng = PaperForwardEngine(candidate_file=str(cfg_path))
    assert eng.candidates, "no paper_forward candidates loaded"

    enabled = [c for c in eng.candidates if c.get("enabled", True)]
    assert enabled, "no enabled candidates"

    failures: list[str] = []
    for cand in enabled[:6]:  # sample first 6 for speed
        cid = cand["candidate_id"]
        load_st = str(cand.get("_load_status") or "")
        fo_len = len(cand.get("_feature_order") or [])
        is_ml = str(cand.get("model_name") or "").lower() not in ("rule", "rules", "heuristic", "none", "")
        if is_ml and fo_len == 0 and load_st == "candidate_loaded_ok":
            failures.append(f"{cid}: empty feature_order for ML model")
        if "ARTIFACT" in load_st.upper() and cand.get("enabled", True):
            failures.append(f"{cid}: enabled but artifact status={load_st}")

        try:
            _, missing, dbg = build_paper_forward_feature_frame(cand, snap)
            if dbg.get("feature_build_error"):
                failures.append(f"{cid}: feature_build_error={dbg.get('feature_build_error')}")
        except Exception as exc:
            failures.append(f"{cid}: feature_build_exception={exc}")

    decs = eng.on_market_snapshot(snap, snap["option_chain"])
    assert len(decs) == len(eng.candidates), "decision count mismatch"

    table = {r["candidate_id"]: r for r in eng.get_status_table()}
    for dec in decs:
        cid = dec["candidate_id"]
        row = table.get(cid)
        if row is None:
            failures.append(f"{cid}: missing status table row")
            continue
        if row.get("candidate_id") != cid:
            failures.append(f"{cid}: UI row candidate_id mismatch {row.get('candidate_id')}")
        conf_disp = row.get("confidence_display") or format_paper_forward_confidence(
            row.get("confidence"),
            predict_attempted=bool(row.get("predict_attempted")),
            reason=str(row.get("raw_reason") or row.get("last_no_trade_reason") or ""),
            load_status=str(row.get("_load_status") or ""),
        )
        if conf_disp in ("N/A", "", None):
            failures.append(f"{cid}: confidence_display is N/A")
        if cand_ok := next((c for c in eng.candidates if c["candidate_id"] == cid), None):
            if (
                cand_ok.get("enabled", True)
                and cand_ok.get("_load_status") == "candidate_loaded_ok"
                and snap.get("data_quality_status") in ("DATA_OK", "READY_FOR_PREDICTION")
                and not dec.get("predict_attempted")
                and str(dec.get("no_trade_reason") or "").upper() not in (
                    "FEATURES_MISSING",
                    "WAITING_FOR_CANDLES",
                    "NOT_READY",
                )
            ):
                failures.append(f"{cid}: predict not attempted despite DATA_OK + valid artifact")

    dbg_path = eng.write_debug_export()
    print(f"debug_export={dbg_path}")

    if failures:
        print("VERIFY_FAIL")
        for f in failures:
            print(" -", f)
        return 1

    print(
        f"VERIFY_OK candidates={len(eng.candidates)} enabled={len(enabled)} "
        f"predictions_attempted={sum(1 for d in decs if d.get('predict_attempted'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())