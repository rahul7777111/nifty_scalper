#!/usr/bin/env python3
"""Diagnose m.Stock runtime health without placing orders."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _mask(k: str, v: str) -> str:
    if any(s in k.upper() for s in ("TOKEN", "SECRET", "KEY", "PASSWORD", "PIN")):
        if not v:
            return ""
        return v[:4] + "..." + v[-4:] if len(v) > 10 else "***"
    return v


def _public_ip() -> str:
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip"):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.read().decode("utf-8", errors="replace").strip()
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
    return f"unavailable ({last})"


def main() -> int:
    print("[MSTOCK-RUNTIME] python_exe=", sys.executable)
    print("[MSTOCK-RUNTIME] python_version=", sys.version.replace("\n", " "))
    print("[MSTOCK-RUNTIME] public_ip=", _public_ip())
    print("[MSTOCK-RUNTIME] selected_broker=", os.getenv("SCALPER_BROKER", os.getenv("BROKER", "mstock")))
    print("[MSTOCK-RUNTIME] env:")
    for k in sorted(k for k in os.environ if k.startswith("MSTOCK_")):
        print(f"  {k}={_mask(k, os.environ.get(k, ''))}")

    try:
        from config import load_api_config
        from mstock_client import MStockTypeBClient
    except Exception as exc:
        print("[MSTOCK-RUNTIME] import_failed=", type(exc).__name__, exc)
        return 1

    client = MStockTypeBClient(load_api_config())
    token = os.getenv("MSTOCK_NIFTY_TOKEN", os.getenv("MSTOCK_UNDERLYING_TOKEN", "26000")).strip() or "26000"
    exchange = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", "NSE").strip().upper() or "NSE"
    option_exchange = os.getenv("MSTOCK_OPTION_EXCHANGE_ID", os.getenv("MSTOCK_OPTION_EXCHANGE", "NFO")).strip() or "NFO"
    print(f"[MSTOCK-RUNTIME] underlying_token={token}")
    print(f"[MSTOCK-RUNTIME] option_exchange_id={option_exchange}")

    try:
        candles, interval = client.fetch_index_candles(
            token,
            exchange=exchange,
            limit=5,
            timeframe="ONE_MINUTE",
            force_historical_only=True,
        )
        print(f"[MSTOCK-RUNTIME] historical_candles rows={len(candles or [])} interval={interval}")
    except Exception as exc:
        print(f"[MSTOCK-RUNTIME] historical_candles_error={type(exc).__name__}: {exc}")

    raw_payload = None
    try:
        response = client._raw.get_market_quote("LTP", [{exchange: [token]}])
        raw_payload = client._safe_json(response, context="diagnose_get_market_quote")
        print("[MSTOCK-RUNTIME] quote_raw_json=", json.dumps(raw_payload, default=str)[:2000])
    except Exception as exc:
        print(f"[MSTOCK-RUNTIME] quote_raw_error={type(exc).__name__}: {exc}")

    try:
        ltp = client.get_ltp(f"{exchange}:{token}")
        print(f"[MSTOCK-RUNTIME] parsed_ltp={ltp}")
    except Exception as exc:
        print(f"[MSTOCK-RUNTIME] parsed_ltp_exception={type(exc).__name__}: {exc}")

    status = client.get_broker_auth_status()
    print("[MSTOCK-RUNTIME] broker_status=", json.dumps(status, default=str, indent=2))
    ia403 = bool(status.get("broker_ip_mismatch")) or client._contains_ia403(raw_payload)
    print(f"[MSTOCK-RUNTIME] ia403_detected={ia403}")
    return 2 if ia403 else 0


if __name__ == "__main__":
    raise SystemExit(main())
