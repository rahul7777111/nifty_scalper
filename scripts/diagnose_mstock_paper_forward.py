#!/usr/bin/env python3
"""
scripts/diagnose_mstock_paper_forward.py

Standalone diagnostic for m.Stock Paper Forward Monitor readiness.
Prints exactly the fields required by the repair brief.
Run from repo root after activating venv.
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Make src importable
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

def _env(k: str, default: str = "") -> str:
    return str(os.getenv(k, default) or "").strip()

def main() -> None:
    print("=== NiftyScalper m.Stock Paper Forward Diagnose ===")
    print(f"repo_root={REPO_ROOT}")
    broker = _env("SCALPER_BROKER", "mstock").lower()
    if broker in {"mstocks", "m.stock", "m_stock"}:
        broker = "mstock"
    print(f"selected_broker={broker}")

    mstock_auth = bool(_env("MSTOCK_ACCESS_TOKEN"))
    print(f"MSTOCK_AUTH_PRESENT={mstock_auth}")

    exch = _env("MSTOCK_OPTION_EXCHANGE_ID") or _env("MSTOCK_OPTION_EXCHANGE") or _env("MSTOCK_SCRIPMASTER_EXCH")
    exp = _env("MSTOCK_OPTION_EXPIRY") or _env("MSTOCK_TARGET_EXPIRY")
    tok = _env("MSTOCK_OPTION_TOKEN") or _env("MSTOCK_UNDERLYING_TOKEN") or _env("MSTOCK_NIFTY_INDEX_TOKEN")
    print(f"MSTOCK_OPTION_EXCHANGE_ID={exch or 'MISSING'}")
    print(f"MSTOCK_OPTION_EXPIRY={exp or 'MISSING'}")
    print(f"MSTOCK_OPTION_TOKEN (optional for multi-strike) present={bool(tok)}")
    print(f"MSTOCK_TARGET_EXPIRY(from GUI/config/env)={_env('MSTOCK_TARGET_EXPIRY') or 'MISSING'}")

    # Candle rows (best effort: look for common live cache or recent files)
    candle_count = 0
    candle_src = "none"
    try:
        # UI live cache not available; check data/ or logs for recent candle counts is out of scope; use 0 unless file hint
        # For real, the live UI would show 100; script reports what it can derive
        for cand_f in (REPO_ROOT / "data" / "live_candles.json", REPO_ROOT / "logs" / "last_candles.json"):
            if cand_f.exists():
                try:
                    data = json.loads(cand_f.read_text())
                    if isinstance(data, list):
                        candle_count = len(data)
                        candle_src = str(cand_f.name)
                        break
                except Exception:
                    pass
    except Exception:
        pass
    print(f"candle_rows={candle_count} source={candle_src}")

    # Option chain fetch using resolver + full client path (TASK 10)
    opt_raw = 0
    opt_norm = 0
    chain_status = "NOT_ATTEMPTED"
    last_err = ""
    fetch_attempted = False
    try:
        from mstock_client import resolve_mstock_option_exchange, MStockTypeBClient
        exr = resolve_mstock_option_exchange(underlying="NIFTY")
        print(f"MSTOCK_OPTION_EXCHANGE_ID={exr.get('exchange_id', 'NFO')}")
        fetch_attempted = True
        # Use real client get_option_chain (will use resolver + CSV first + deep logs)
        try:
            # light client
            client = MStockTypeBClient.__new__(MStockTypeBClient)
            # set minimal attrs the getter expects
            client.cfg = type("C", (), {"api_key": _env("MSTOCK_API_KEY")})()
            chain = client.get_option_chain("NIFTY") or []
            opt_raw = len(chain)
            opt_norm = opt_raw
            chain_status = "FETCHED" if opt_raw > 0 else "EMPTY_AFTER_FETCH"
        except Exception as e:
            last_err = str(e)[:300]
            # fallback to direct sm with resolver exch
            try:
                from scripmaster import ScripMaster
                smp = _env("MSTOCK_SCRIPMASTER_PATH")
                if not smp:
                    for nm in ("api-scrip-master.csv", "scripmaster.csv"):
                        p = REPO_ROOT / nm
                        if p.exists():
                            smp = str(p); break
                if smp and Path(smp).exists():
                    sm = ScripMaster(smp)
                    exid = exr.get("exchange_id", "NFO")
                    rws = sm.option_rows(symbol_root="NIFTY", exch=exid, min_expiry=datetime.now().date())
                    opt_norm = len(rws or [])
                    chain_status = "CSV_FALLBACK"
            except Exception as e2:
                last_err += f"; sm_fallback:{e2}"
    except Exception as e:
        last_err = str(e)
        chain_status = "CLIENT_ERR"

    print(f"option_chain_fetch_attempted={fetch_attempted}")
    print(f"option_chain_fetch_status={chain_status}")
    print(f"option_chain_raw_rows={opt_raw}")
    print(f"option_chain_normalized_rows={opt_norm}")
    if last_err:
        print(f"last_chain_error={last_err[:250]}")

    # Candidate config
    cand_path = REPO_ROOT / "config" / "paper_forward_candidates.json"
    total_c = 0
    enabled_c = 0
    artifact_ok = 0
    try:
        if cand_path.exists():
            payload = json.loads(cand_path.read_text(encoding="utf-8"))
            raw = payload.get("candidates", payload) if isinstance(payload, dict) else payload
            cands: List[Dict[str, Any]] = [c for c in (raw or []) if isinstance(c, dict)]
            total_c = len(cands)
            enabled_list = [c for c in cands if bool(c.get("enabled", False)) or bool(c.get("active", False))]
            enabled_c = len(enabled_list)
            for c in enabled_list:
                ad = str(c.get("artifact_dir") or "")
                # consider ok if dir exists or relative default
                if ad:
                    p = Path(ad)
                    if not p.is_absolute():
                        p = REPO_ROOT / ad
                    if p.exists():
                        artifact_ok += 1
                else:
                    # default location check
                    cid = c.get("candidate_id", "")
                    if cid and (REPO_ROOT / "artifacts" / "candidates" / str(cid)).exists():
                        artifact_ok += 1
    except Exception as e:
        print(f"CANDIDATE_LOAD_ERR={e}")
    print(f"candidate_config_total={total_c}")
    print(f"candidate_config_enabled={enabled_c}")
    print(f"artifact_ok_count={artifact_ok}")

    # missing after resolver
    mks = []
    if not exp: mks.append("MSTOCK_OPTION_EXPIRY")
    # exchange should be defaulted
    print(f"missing_config_keys={mks}")

    # Black-Scholes synthetic fallback probe
    bs_enabled = False
    bs_rows = 0
    bs_iv = ""
    final_chain_source = "broker"
    try:
        from synthetic_option_chain import is_synthetic_chain_enabled, generate_synthetic_option_chain, CHAIN_SOURCE
        bs_enabled = is_synthetic_chain_enabled(broker)
        print(f"bs_synthetic_enabled={bs_enabled}")
        if bs_enabled and exp and not opt_norm:
            spot_guess = 24500.0
            try:
                rows, meta = generate_synthetic_option_chain(spot=spot_guess, expiry=exp, candles=[])
                bs_rows = len(rows)
                bs_iv = meta.get("iv", "")
                final_chain_source = CHAIN_SOURCE
                print(f"bs_rows={bs_rows} bs_iv={bs_iv} bs_ce_rows={meta.get('ce_rows')} bs_pe_rows={meta.get('pe_rows')}")
                print(f"synthetic_chain_cache_written=false final_chain_source={final_chain_source}")
            except Exception as e:
                print(f"bs_synthetic_probe_err={e}")
    except Exception as e:
        print(f"bs_module_err={e}")

    # Final readiness (TASK 10)
    has_auth = mstock_auth
    has_expiry = bool(exp)
    has_chain = opt_norm > 0 or bs_rows > 0
    has_candles = candle_count > 0
    has_artifacts = artifact_ok > 0 or enabled_c == 0

    if not has_expiry:
        readiness = "BLOCKED_MSTOCK_EXPIRY_MISSING"
    elif not has_chain:
        readiness = "BLOCKED_OPTION_CHAIN_EMPTY"
    elif bs_rows > 0 and not has_candles:
        readiness = "BLOCKED_NO_CANDLES"
    elif bs_rows > 0 and has_candles:
        readiness = "READY_SYNTHETIC_CHAIN"
    elif enabled_c > 0 and artifact_ok == 0:
        readiness = "BLOCKED_NO_ENABLED_ARTIFACTS"
    elif has_auth and has_chain and (artifact_ok > 0 or enabled_c == 0):
        readiness = "READY_FOR_PAPER_FORWARD"
    else:
        readiness = "BLOCKED_UNKNOWN"

    print(f"real_chain_fetch_status={chain_status} real_chain_rows={opt_norm}")
    print(f"final_chain_source={final_chain_source}")
    print(f"final_readiness={readiness}")
    print("=== end diagnose ===")

if __name__ == "__main__":
    main()
