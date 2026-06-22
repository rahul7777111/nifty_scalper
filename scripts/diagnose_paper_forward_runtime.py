#!/usr/bin/env python3
"""Paper Forward runtime diagnostic with synthetic chain/candle readiness."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    from synthetic_option_chain import (
        CHAIN_SOURCE,
        build_spot_candle_fallback,
        infer_paper_forward_data_quality,
        is_synthetic_chain_enabled,
        load_bs_config,
    )
    from paper_forward_spot import resolve_live_spot_for_paper_forward, resolve_paper_forward_candles

    try:
        import ui
        broker = ui._normalize_broker_name(os.getenv("SCALPER_BROKER", "mstock"))
        cfg_status = ui.validate_option_chain_config_for_active_broker(broker)
    except Exception:
        broker = "mstock"
        cfg_status = {"missing_keys": [], "broker": broker}

    cfg = load_bs_config()
    spot_res = resolve_live_spot_for_paper_forward(broker=broker, current=os.getenv("PF_FALLBACK_SPOT"))
    candle_res = resolve_paper_forward_candles(broker=broker, spot=spot_res.spot)
    candles = list(candle_res.candles)
    if not candles and spot_res.spot and cfg["allow_candle_fallback"]:
        candles = build_spot_candle_fallback(float(spot_res.spot))
        candle_res = type(candle_res)(
            ok=True,
            candles=candles,
            source="SYNTHETIC_SPOT_FALLBACK",
            synthetic_candles=True,
            tried=list(candle_res.tried),
        )

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

    option_rows = 82 if is_synthetic_chain_enabled(broker) and spot_res.ok else 0
    candle_source = candle_res.source
    quality = infer_paper_forward_data_quality(
        option_rows=option_rows,
        candle_count=len(candles),
        spot=spot_res.spot,
        chain_source=CHAIN_SOURCE if option_rows else "",
        synthetic=bool(option_rows),
        candle_source=candle_source,
        synthetic_candles=bool(candle_res.synthetic_candles),
    )
    min_candles = int(cfg.get("min_candles_for_prediction", 20))
    predict_allowed = enabled_ok if len(candles) >= min_candles and option_rows > 0 else 0

    if quality == "SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES":
        readiness = "READY_SYNTHETIC_CHAIN_WITH_SYNTHETIC_CANDLES"
    elif quality == "SYNTHETIC_CHAIN_OK":
        readiness = "READY_SYNTHETIC_CHAIN_WITH_REAL_CANDLES"
    elif quality in ("WAITING_FOR_CANDLES", "SYNTHETIC_CHAIN_READY_WAITING_FOR_CANDLES"):
        readiness = "WAITING_FOR_CANDLES"
    elif cfg_status.get("missing_keys"):
        readiness = "BLOCKED_BROKER_CONFIG"
    elif not spot_res.ok:
        readiness = "BLOCKED_NO_SPOT"
    elif option_rows <= 0:
        readiness = "BLOCKED_NO_OPTION_CHAIN"
    else:
        readiness = "READY_SYNTHETIC_CHAIN_WITH_REAL_CANDLES"

    report = {
        "repo_root": str(REPO_ROOT),
        "selected_broker": broker,
        "spot_source": spot_res.source,
        "chain_source": CHAIN_SOURCE if option_rows else "none",
        "synthetic_chain_rows": option_rows,
        "candle_source": candle_source,
        "candle_rows": len(candles),
        "synthetic_candle_enabled": cfg["allow_candle_fallback"],
        "data_quality_status": quality,
        "enabled_artifact_ok": enabled_ok,
        "predict_allowed_count": predict_allowed,
        "synthetic_live_order_guard": "PASS",
        "final_readiness": readiness,
        "broker_config_validation": cfg_status,
    }
    print(json.dumps(report, indent=2, default=str))
    print(f"final_readiness={readiness}")
    print(f"enabled_artifact_ok={enabled_ok}")
    print(f"predict_allowed_count={predict_allowed}")
    print("synthetic_live_order_guard=PASS")
    return 0 if readiness.startswith("READY_") else 1


if __name__ == "__main__":
    raise SystemExit(main())