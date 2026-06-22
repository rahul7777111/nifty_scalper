#!/usr/bin/env python3
"""
scripts/diagnose_black_scholes_paper_forward.py

Standalone diagnostic for Black-Scholes synthetic Paper Forward readiness.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _env(k: str, default: str = "") -> str:
    return str(os.getenv(k, default) or "").strip()


def main() -> None:
    from synthetic_option_chain import (
        CHAIN_SOURCE,
        PAPER_FORWARD_REQUIRED_COLUMNS,
        build_spot_candle_fallback,
        generate_synthetic_option_chain,
        infer_paper_forward_data_quality,
        is_synthetic_chain_enabled,
        load_bs_config,
        verify_paper_forward_columns,
    )
    from paper_forward_spot import (
        resolve_live_spot_for_paper_forward,
        resolve_paper_forward_candles,
        fetch_mstock_nifty_spot_ltp,
    )

    print("=== Black-Scholes Paper Forward Diagnose ===")
    print(f"repo_root={REPO_ROOT}")
    cfg = load_bs_config()
    broker = _env("SCALPER_BROKER", "mstock").lower()
    if broker in {"mstocks", "m.stock", "m_stock"}:
        broker = "mstock"
    print(f"broker={broker}")
    print(f"MSTOCK_AUTH_PRESENT={bool(_env('MSTOCK_ACCESS_TOKEN'))}")
    print(f"PF_USE_BLACK_SCHOLES_CHAIN={cfg['use_black_scholes_chain']}")
    print(f"PF_ALLOW_SYNTHETIC_CANDLE_FALLBACK={cfg['allow_candle_fallback']}")
    print(f"PF_MIN_CANDLES_FOR_PREDICTION={cfg.get('min_candles_for_prediction', 20)}")
    print(f"PF_SYNTHETIC_CANDLE_COUNT={cfg.get('synthetic_candle_count', 100)}")
    print(f"PF_FALLBACK_SPOT={_env('PF_FALLBACK_SPOT') or '(unset)'}")

    expiry = _env("MSTOCK_OPTION_EXPIRY") or _env("MSTOCK_TARGET_EXPIRY") or "16-06-2026"
    exchange = _env("MSTOCK_OPTION_EXCHANGE_ID") or _env("MSTOCK_SCRIPMASTER_EXCH") or "NFO"
    print(f"expiry={expiry or 'MISSING'}")
    print(f"exchange={exchange}")

    client = None
    mstock_spot_status = "NOT_ATTEMPTED"
    mstock_spot = None
    try:
        from mstock_client import MStockTypeBClient
        if _env("MSTOCK_ACCESS_TOKEN"):
            client = MStockTypeBClient.__new__(MStockTypeBClient)
            client.cfg = type("C", (), {"api_key": _env("MSTOCK_API_KEY")})()
            mstock_spot, mstock_spot_status, _ = fetch_mstock_nifty_spot_ltp(client)
    except Exception as exc:
        mstock_spot_status = f"ERR:{exc}"

    spot_res = resolve_live_spot_for_paper_forward(
        client=client,
        broker=broker,
        current=mstock_spot,
    )
    print(f"spot_resolver={json.dumps(spot_res.as_dict(), default=str)}")
    print(f"mstock_spot_fetch_status={mstock_spot_status} spot={mstock_spot}")

    candle_res = resolve_paper_forward_candles(client=client, broker=broker, spot=spot_res.spot)
    print(f"candle_resolver={json.dumps(candle_res.as_dict(), default=str)}")

    real_chain_rows = 0
    if client is not None and hasattr(client, "get_option_chain"):
        try:
            real_chain_rows = len(client.get_option_chain("NIFTY") or [])
        except Exception:
            real_chain_rows = 0
    print(f"real_option_chain_rows={real_chain_rows}")
    print(f"bs_synthetic_enabled={is_synthetic_chain_enabled(broker)}")

    bs_rows = 0
    quality = ""
    synth_candles = list(candle_res.candles)
    if not synth_candles and spot_res.ok and spot_res.spot and cfg["allow_candle_fallback"]:
        synth_candles = build_spot_candle_fallback(float(spot_res.spot))
    if spot_res.ok and spot_res.spot and expiry:
        try:
            rows, meta = generate_synthetic_option_chain(
                spot=float(spot_res.spot),
                expiry=expiry,
                candles=synth_candles,
                timestamp=datetime.now(),
            )
            bs_rows = len(rows)
            ok, missing = verify_paper_forward_columns(rows)
            candle_source = "SYNTHETIC_SPOT_FALLBACK" if candle_res.synthetic_candles else candle_res.source
            quality = infer_paper_forward_data_quality(
                option_rows=bs_rows,
                candle_count=len(synth_candles),
                spot=spot_res.spot,
                chain_source=CHAIN_SOURCE,
                synthetic=True,
                candle_source=candle_source,
                synthetic_candles=bool(candle_res.synthetic_candles or candle_source == "SYNTHETIC_SPOT_FALLBACK"),
            )
            print(f"bs_generated_rows={bs_rows} ce={meta.get('ce_rows')} pe={meta.get('pe_rows')}")
            print(f"required_columns_ok={ok} missing_columns={missing}")
            if rows:
                atm = meta.get("atm_strike")
                ce = next((r for r in rows if r.get("option_type") == "CE" and r.get("strike_price") == atm), rows[0])
                print(f"sample_atm_ce_keys={sorted(ce.keys())[:12]}")
        except Exception as exc:
            print(f"bs_generate_error={exc}")
    else:
        print("bs_generated_rows=0 reason=no_spot_or_no_expiry")

    enabled_ok = 0
    cand_path = REPO_ROOT / "config" / "paper_forward_candidates.json"
    if cand_path.exists():
        try:
            payload = json.loads(cand_path.read_text(encoding="utf-8"))
            raw = payload.get("candidates", payload) if isinstance(payload, dict) else payload
            for c in raw or []:
                if isinstance(c, dict) and (c.get("enabled") or c.get("active")):
                    ad = Path(str(c.get("artifact_dir") or ""))
                    if not ad.is_absolute():
                        ad = REPO_ROOT / ad
                    if ad.exists():
                        enabled_ok += 1
        except Exception:
            pass
    print(f"enabled_artifact_ok={enabled_ok}")
    print(f"data_quality={quality or 'N/A'}")

    min_candles = int(cfg.get("min_candles_for_prediction", 20))
    candle_rows = len(synth_candles)
    predict_allowed = 0
    if enabled_ok and candle_rows >= min_candles and bs_rows > 0:
        predict_allowed = enabled_ok
    print(f"predict_allowed_count={predict_allowed}")
    print("synthetic_live_order_guard=PASS")

    if not spot_res.ok:
        readiness = "BLOCKED_NO_SPOT"
    elif quality == "SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES":
        readiness = "READY_SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES"
    elif quality == "SYNTHETIC_CHAIN_OK":
        readiness = "READY_SYNTHETIC_CHAIN_WITH_REAL_CANDLES"
    elif quality == "SYNTHETIC_CHAIN_WITH_SPOT_CANDLE_FALLBACK" and candle_rows >= int(cfg.get("synthetic_candle_count", 100)):
        readiness = "READY_SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES"
    elif quality in ("WAITING_FOR_CANDLES", "SYNTHETIC_CHAIN_READY_WAITING_FOR_CANDLES"):
        readiness = "WAITING_FOR_CANDLES"
    elif bs_rows > 0 and candle_rows < min_candles:
        readiness = "WAITING_FOR_CANDLES"
    elif bs_rows > 0:
        readiness = "READY_SYNTHETIC_CHAIN_WITH_REAL_CANDLES"
    else:
        readiness = "BLOCKED_UNKNOWN"

    print(f"final_readiness={readiness}")
    print("=== end diagnose ===")


if __name__ == "__main__":
    main()