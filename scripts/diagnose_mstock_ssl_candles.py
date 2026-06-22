#!/usr/bin/env python3
"""Diagnose m.Stock candle SSL/TLS setup using the same client path as the GUI."""
from __future__ import annotations

import os
import ssl
import sys
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _certifi_path() -> str:
    try:
        import certifi

        return str(certifi.where())
    except Exception as exc:
        return f"certifi unavailable: {type(exc).__name__}: {exc}"


def main() -> int:
    print("[MSTOCK-SSL-DIAG-SCRIPT] python_exe=", sys.executable)
    print("[MSTOCK-SSL-DIAG-SCRIPT] python_version=", sys.version.replace("\n", " "))
    print("[MSTOCK-SSL-DIAG-SCRIPT] certifi=", _certifi_path())
    print("[MSTOCK-SSL-DIAG-SCRIPT] MSTOCK_SSL_VERIFY=", os.getenv("MSTOCK_SSL_VERIFY", "true"))
    print("[MSTOCK-SSL-DIAG-SCRIPT] MSTOCK_CA_BUNDLE=", os.getenv("MSTOCK_CA_BUNDLE", ""))
    print("[MSTOCK-SSL-DIAG-SCRIPT] MSTOCK_ALLOW_INSECURE_SSL=", os.getenv("MSTOCK_ALLOW_INSECURE_SSL", "0"))

    ca_bundle = os.getenv("MSTOCK_CA_BUNDLE", "").strip() or _certifi_path()
    try:
        ctx = ssl.create_default_context(cafile=ca_bundle if Path(ca_bundle).exists() else None)
        print("[MSTOCK-SSL-DIAG-SCRIPT] ssl_context=OK verify_mode=", ctx.verify_mode)
    except Exception as exc:
        print("[MSTOCK-SSL-DIAG-SCRIPT] ssl_context=FAIL", type(exc).__name__, exc)
        return 1

    try:
        from config import load_api_config
        from mstock_client import MStockTypeBClient

        cfg = load_api_config()
        client = MStockTypeBClient(cfg)
        token = os.getenv("MSTOCK_NIFTY_TOKEN", "26000").strip() or "26000"
        exchange = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", "NSE").strip().upper() or "NSE"
        print(
            f"[MSTOCK-SSL-DIAG-SCRIPT] fetching broker=mstock token={token} "
            f"exchange={exchange} interval=ONE_MINUTE"
        )
        candles, used_tf = client.fetch_index_candles(
            token,
            exchange=exchange,
            limit=100,
            timeframe="ONE_MINUTE",
            force_historical_only=True,
        )
        rows = len(candles or [])
        print(f"[MSTOCK-SSL-DIAG-SCRIPT] result rows={rows} interval={used_tf}")
        if rows:
            first = candles[0]
            last = candles[-1]
            print(f"[MSTOCK-SSL-DIAG-SCRIPT] first={first}")
            print(f"[MSTOCK-SSL-DIAG-SCRIPT] last={last}")
            return 0
        print("[MSTOCK-SSL-DIAG-SCRIPT] no candle rows returned")
        return 2
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "CERTIFICATE_VERIFY_FAILED" in text or "SSL_CERTIFICATE_VERIFY_FAILED" in text:
            print("[MSTOCK-SSL-DIAG-SCRIPT] SSL_CERTIFICATE_VERIFY_FAILED")
        print("[MSTOCK-SSL-DIAG-SCRIPT] fetch=FAIL", text)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
