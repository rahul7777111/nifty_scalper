#!/usr/bin/env python3
"""
scripts/diagnose_paper_forward_prediction.py

Diagnostic for paper-forward ML router prediction path.
- Loads config/paper_forward_candidates.json
- Builds live-like features via build_paper_forward_feature_frame
- Runs model prediction per candidate
- Prints PF-PREDICT-DIAG-style table and explains zero-confidence cases
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from src.candidate_router import route_candidate_decision, _predict_confidence_from_artifact, _load_active_candidate_profile
    from src.paper_forward_engine import build_paper_forward_feature_frame
except Exception:
    from candidate_router import route_candidate_decision, _predict_confidence_from_artifact, _load_active_candidate_profile  # type: ignore
    from paper_forward_engine import build_paper_forward_feature_frame  # type: ignore


def load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cands = data.get("candidates", data if isinstance(data, list) else [])
    return {"candidates": [c for c in cands if c.get("enabled", True)]}


def find_debug_snapshot() -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    logs = REPO_ROOT / "logs"
    if logs.exists():
        candidates.extend(sorted(logs.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(logs.glob("paper*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        candidates.extend(sorted(data_dir.glob("**/*snapshot*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5])
    for p in candidates[:8]:
        try:
            if p.suffix == ".jsonl":
                for line in reversed(p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and (obj.get("spot") or obj.get("price") or obj.get("option_chain") or obj.get("candles")):
                        return obj
            else:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    if obj.get("snapshots") and isinstance(obj["snapshots"], list) and obj["snapshots"]:
                        return obj["snapshots"][-1]
                    if obj.get("spot") or obj.get("price") or obj.get("option_chain"):
                        return obj
        except Exception:
            continue
    return None


def make_synthetic_snapshot(feature_order: List[str] | None = None) -> Dict[str, Any]:
    import random
    from datetime import datetime, timezone

    snap: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spot": 23500.0 + random.uniform(-50, 50),
        "price": 23500.0 + random.uniform(-50, 50),
        "ltp": 120.0 + random.uniform(-10, 10),
        "strike": 23500.0,
        "option_type": "PE",
        "bid": 118.0,
        "ask": 122.0,
        "iv": 0.17,
        "delta": -0.45,
        "gamma": 0.03,
        "theta": -12.0,
        "vega": 45.0,
        "oi": 120000,
        "volume": 45000,
        "dte": 0.8,
        "dte_days": 0.8,
        "moneyness": 0.01,
        "spread": 4.0,
        "spread_pct": 0.033,
        "open": 23480.0,
        "high": 23560.0,
        "low": 23450.0,
        "close": 23510.0,
        "candle_volume": 120000,
        "rsi_14": 48.0,
        "atr_14": 85.0,
        "ret_1": 0.0008,
        "ret_3": -0.0012,
        "broker_auth": "AUTH_OK",
        "auth_status": "AUTH_OK",
        "data_quality_status": "DATA_OK",
        "candles": [
            {"open": 23480, "high": 23520, "low": 23470, "close": 23495, "volume": 80000},
        ] * 25,
        "candle_count": 25,
        "candle_source": "synthetic",
    }
    if feature_order:
        for f in feature_order:
            if f not in snap:
                if any(x in f for x in ("ret", "pct", "ratio", "moneyness", "spread")):
                    snap[f] = random.uniform(-0.02, 0.02)
                elif any(x in f for x in ("oi", "volume", "count")):
                    snap[f] = random.uniform(1000, 200000)
                elif any(x in f for x in ("iv", "vol")):
                    snap[f] = random.uniform(0.12, 0.28)
                else:
                    snap[f] = random.uniform(0.0, 100.0)
    snap["option_chain"] = [{
        "strike": snap["strike"], "option_type": snap["option_type"], "ltp": snap["ltp"],
        "bid": snap["bid"], "ask": snap["ask"], "iv": snap["iv"], "delta": snap["delta"],
        "oi": snap["oi"], "volume": snap["volume"],
    }]
    return snap


def _build_route_snapshot(cand: Dict[str, Any], base_snap: Dict[str, Any], art_path: Path) -> tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    profile, _ = _load_active_candidate_profile(str(art_path), cand["candidate_id"])
    fo = (profile or {}).get("_feature_order") or cand.get("_feature_order") or []
    merge_cand = dict(cand)
    if profile:
        merge_cand.update({k: profile[k] for k in ("_feature_order", "artifact_paths", "side_policy") if k in profile})
    ff, missing, feature_debug = build_paper_forward_feature_frame(merge_cand, base_snap)
    route_snap = dict(base_snap)
    if ff is not None and not ff.empty:
        for fk, fv in ff.iloc[0].to_dict().items():
            if fv is None:
                continue
            if isinstance(fv, float) and math.isnan(fv):
                continue
            route_snap[fk] = fv
    route_snap["option_type"] = route_snap.get("option_type") or "PE"
    return route_snap, missing, feature_debug


def _why_zero(res: Dict[str, Any], dec: Dict[str, Any]) -> str:
    err = res.get("error") or ""
    if err:
        return str(err)
    if dec.get("no_trade_reason"):
        return str(dec.get("no_trade_reason"))
    conf = res.get("confidence")
    if conf is None:
        return "confidence_none"
    if isinstance(conf, (int, float)) and float(conf) <= 1e-12:
        return "model_output_near_zero"
    return "ok"


def run_diagnosis(config_path: Path, fail_on_zero: bool = True) -> int:
    cfg = load_config(config_path)
    cands = cfg["candidates"]
    print(f"[DIAG] loaded {len(cands)} enabled candidates from {config_path}")

    debug_snap = find_debug_snapshot()
    print(f"[DIAG] debug_snapshot_found={debug_snap is not None}")

    rows: List[Dict[str, Any]] = []
    all_zero = True

    for c in cands:
        cid = c["candidate_id"]
        ad = c.get("artifact_dir") or str(REPO_ROOT / "artifacts" / "candidates" / cid)
        adp = Path(ad)
        if not adp.is_absolute():
            adp = (REPO_ROOT / ad).resolve()

        fo: List[str] = []
        fsj = adp / "feature_schema.json"
        if fsj.exists():
            try:
                fsd = json.loads(fsj.read_text(encoding="utf-8"))
                if isinstance(fsd, list):
                    fo = [str(x) for x in fsd]
                elif isinstance(fsd, dict):
                    fo = [str(x) for x in (fsd.get("features") or fsd.get("live_computable_features") or [])]
            except Exception:
                pass
        if not fo:
            fo = c.get("_feature_order") or c.get("feature_order") or []

        base_snap = dict(debug_snap) if debug_snap else make_synthetic_snapshot(fo or None)
        route_snap, missing, feature_debug = _build_route_snapshot(c, base_snap, adp)

        required_n = len(fo) or int(c.get("required_features", 0) or 0)
        raw = None
        conf = None
        thr = float(c.get("threshold", 0.35) or 0.35)
        final_sig = "NO_TRADE"
        ntr = "not_run"
        model_cls = "unknown"
        pred_err = ""
        scaler_present = False
        scaler_applied = False

        try:
            dec = route_candidate_decision(
                market_snapshot=route_snap,
                option_chain_snapshot=route_snap.get("option_chain"),
                legacy_signal=None,
                mode="paper",
                active_candidate_id=cid,
                candidate_dir=str(adp),
                force_eval=True,
            )
            dec = dict(dec or {})
            conf = dec.get("confidence")
            thr = float(dec.get("threshold") or thr)
            final_sig = dec.get("final_signal", "NO_TRADE")
            ntr = dec.get("no_trade_reason", "")
            pdbg = (dec.get("debug") or {}).get("predict") or {}
            raw = pdbg.get("raw", conf)
            model_cls = pdbg.get("model_class", dec.get("model_name", "unknown"))
            scaler_present = bool(pdbg.get("scaler_present"))
            scaler_applied = bool(pdbg.get("scaler_applied"))
            pred_err = pdbg.get("error") or ""
            why = _why_zero(pdbg, dec)
            print(
                f"[PF-PREDICT-DIAG] candidate_id={cid} artifact_path={adp} model_type={model_cls} "
                f"required_features_count={required_n} feature_vector_count={pdbg.get('feature_count', required_n)} "
                f"missing_features={','.join((pdbg.get('missing_features') or missing or [])[:5]) or '-'} "
                f"nan_count={feature_debug.get('feature_nan_count', 0)} "
                f"zero_count={feature_debug.get('feature_zero_count', 0)} "
                f"scaler_present={scaler_present} model_present={bool((adp / 'model.pkl').exists())} "
                f"raw_output={raw} probability={pdbg.get('prob')} confidence={conf} threshold={thr} "
                f"decision={final_sig} why={why}"
            )
        except Exception as e:
            tb = traceback.format_exc()
            ntr = f"diag_exception:{e}"
            pred_err = str(e)
            why = "exception"
            print(f"[DIAG-EXC] {cid}: {e}\n{tb[:500]}")

        if raw is not None and abs(float(raw)) > 1e-12:
            all_zero = False
        elif conf is not None and abs(float(conf)) > 1e-12:
            all_zero = False
        elif ntr in ("FEATURE_VECTOR_INVALID", "ARTIFACT_IDENTITY_MISMATCH", "MODEL_OUTPUT_INVALID", "XGBOOST_NOT_INSTALLED"):
            all_zero = False

        rows.append({
            "candidate_id": cid,
            "artifact_dir": str(adp),
            "model_class": model_cls,
            "required_features_count": required_n,
            "missing_features_count": len(missing),
            "scaler_present/applied": f"{scaler_present}/{scaler_applied}",
            "raw_output": raw,
            "confidence": conf,
            "threshold": thr,
            "final_signal": final_sig,
            "no_trade_reason": (ntr or "")[:80],
            "predict_error": pred_err[:60] if pred_err else "",
        })

    headers = ["candidate_id", "model", "req_f", "miss_f", "scaler_p/a", "raw", "conf", "thr", "sig", "reason"]
    print("\n" + " | ".join(headers))
    print("-" * 120)
    for r in rows:
        print(" | ".join([
            str(r["candidate_id"])[:42],
            str(r["model_class"])[:16],
            str(r["required_features_count"]),
            str(r["missing_features_count"]),
            str(r["scaler_present/applied"]),
            str(r["raw_output"]),
            str(r["confidence"]),
            str(r["threshold"]),
            str(r["final_signal"]),
            str(r["no_trade_reason"])[:32],
        ]))
    print("-" * 120)

    try:
        from paper_forward_readiness import format_readiness_log, summarize_readiness
    except ImportError:
        from src.paper_forward_readiness import format_readiness_log, summarize_readiness  # type: ignore

    readiness = summarize_readiness(
        candidates_total=len(rows),
        results=[{"no_trade_reason": r["no_trade_reason"], "predict_attempted": True, "confidence": r["confidence"]} for r in rows],
        cost_qualities=[],
        ltp_sources=[],
        data_quality="DIAG",
    )
    print(format_readiness_log(readiness))

    blocked = [r["candidate_id"] for r in rows if r["no_trade_reason"] in (
        "FEATURE_VECTOR_INVALID", "ARTIFACT_IDENTITY_MISMATCH", "MODEL_OUTPUT_INVALID", "XGBOOST_NOT_INSTALLED",
    )]
    zero_cands = [r["candidate_id"] for r in rows if r["raw_output"] is not None and abs(float(r["raw_output"] or 0)) < 1e-12 and (r["confidence"] is None or abs(float(r["confidence"] or 0)) < 1e-12)]
    print(f"[DIAG] blocked_with_specific_reason={len(blocked)} zero_output={len(zero_cands)} total={len(rows)} all_zero={all_zero}")
    if blocked:
        print("  blocked:", blocked[:5], "..." if len(blocked) > 5 else "")

    if fail_on_zero and all_zero and rows:
        print("\n[FAIL] All candidates blocked or produced zero confidence. See PF-PREDICT-DIAG lines above.")
        return 2
    print("[DIAG] done.")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "paper_forward_candidates.json"))
    ap.add_argument("--no-fail-on-zero", action="store_true", help="Do not exit non-zero if all zero")
    args = ap.parse_args(argv or sys.argv[1:])
    return run_diagnosis(Path(args.config), fail_on_zero=not args.no_fail_on_zero)


if __name__ == "__main__":
    sys.exit(main())