#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import datetime, time as dtime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

from config import load_api_config
from mstock_client import MStockTypeBClient
from ui import ScalperUI


def _market_hours_now() -> bool:
    now = datetime.now().time()
    return dtime(9, 15) <= now <= dtime(15, 30)


def main() -> int:
    token = (os.getenv("MSTOCK_ACCESS_TOKEN") or "").strip()
    print(f"[DEBUG-OC] token_present={bool(token)}")
    if not token:
        print("[DEBUG-OC] missing MSTOCK_ACCESS_TOKEN")
        return 2 if _market_hours_now() else 0

    client = MStockTypeBClient(load_api_config())
    try:
        setattr(client, "access_token", token)
        if hasattr(client, "_raw"):
            setattr(client._raw, "access_token", token)
            if hasattr(client._raw, "set_access_token"):
                client._raw.set_access_token(token)
    except Exception as exc:
        print(f"[DEBUG-OC] token_attach_warning={exc}")

    spot = None
    for sym in ("NIFTY", "NSE:NIFTY", os.getenv("MSTOCK_UNDERLYING_TOKEN", "")):
        if not sym:
            continue
        try:
            if hasattr(client, "get_ltp"):
                spot = client.get_ltp(sym)
                print(f"[DEBUG-OC] spot={spot} source=get_ltp:{sym}")
                break
        except Exception as exc:
            print(f"[DEBUG-OC] spot_fetch_error symbol={sym} error={exc}")

    raw = None
    try:
        underlying = os.getenv("MSTOCK_UNDERLYING", "NIFTY") or "NIFTY"
        print(f"[OC-FETCH] broker={type(client).__name__} expiry={os.getenv('MSTOCK_OPTION_EXPIRY') or 'nearest'} spot={spot}")
        raw = client.get_option_chain(underlying)
        raw_len = len(raw) if hasattr(raw, "__len__") else 0
        print(f"[OC-FETCH] success raw_rows={raw_len}")
    except Exception as exc:
        print(f"[OC-FETCH] error type={type(exc).__name__} message={exc}")
        raw = []

    helper = object.__new__(ScalperUI)
    rows = ScalperUI._normalize_option_chain_rows(helper, raw)
    print(f"[DEBUG-OC] normalized_rows={len(rows)}")
    for idx, row in enumerate(rows[:3], start=1):
        print(f"[DEBUG-OC] sample_{idx}_keys={sorted(list(row.keys()))}")
        print(f"[DEBUG-OC] sample_{idx}={row}")

    if not rows and _market_hours_now():
        print("[DEBUG-OC] rows=0 during market hours")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
