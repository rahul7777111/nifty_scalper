from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from config import load_api_config


def _row_token(row: Dict[str, Any]) -> str:
    for k in (
        "symboltoken",
        "symbolToken",
        "instrumentToken",
        "token",
        "Token",
        "symbol_token",
    ):
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s.isdigit():
            return s
    return ""


def _row_exchange(row: Dict[str, Any]) -> str:
    for k in ("exchange", "exch_seg", "exchangeSegment", "exchange_segment"):
        v = row.get(k)
        if v:
            return str(v).strip().upper()
    return ""


def _row_name(row: Dict[str, Any]) -> str:
    return str(
        row.get("tradingsymbol")
        or row.get("tradingSymbol")
        or row.get("symbol")
        or row.get("name")
        or row.get("tokenName")
        or ""
    ).strip()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search m.Stock instrument master and print matching tokens")
    parser.add_argument("query", nargs="?", default=os.getenv("MSTOCK_UNDERLYING", "NIFTY"), help="Search text")
    parser.add_argument("--exchange", default=os.getenv("MSTOCK_EXCHANGE", ""), help="Exchange filter (optional)")
    parser.add_argument(
        "--instrument-type",
        default="",
        help="Filter by instrument type substring (e.g. FUT, OPT). Matches instrumentType/instrumenttype/segment/series.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print")
    args = parser.parse_args(argv)

    try:
        from mstock_client import MStockTypeBClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: tradingapi_b.\n\n"
            "You're likely running with a different Python than this repo's venv.\n"
            "Run with the venv interpreter directly, e.g. on Windows PowerShell:\n"
            "  & .\\.venv\\Scripts\\python.exe -m src.find_token NIFTY\n"
        ) from exc

    api_cfg = load_api_config()
    client = MStockTypeBClient(api_cfg)

    # Requires MSTOCK_ACCESS_TOKEN already present in env.
    client.login(interactive=False)

    instruments = client._load_instruments()  # type: ignore[attr-defined]
    q = str(args.query or "").strip().upper()
    exch_filter = str(args.exchange or "").strip().upper()
    type_filter = str(args.instrument_type or "").strip().upper()

    matches: List[Dict[str, Any]] = []
    for row in instruments:
        if not isinstance(row, dict):
            continue
        ex = _row_exchange(row)
        if exch_filter and ex and ex != exch_filter:
            continue

        blob = " ".join(
            str(row.get(k) or "")
            for k in (
                "tradingsymbol",
                "tradingSymbol",
                "symbol",
                "name",
                "tokenName",
                "instrumenttype",
                "instrumentType",
                "series",
                "segment",
            )
        ).upper()
        if q and q not in blob:
            continue

        if type_filter:
            it = str(
                row.get("instrumenttype")
                or row.get("instrumentType")
                or row.get("segment")
                or row.get("series")
                or ""
            ).upper()
            if type_filter not in it and type_filter not in blob:
                continue

        tok = _row_token(row)
        if not tok:
            continue
        matches.append(row)

    print(f"Found {len(matches)} matches for {q!r}.")
    printed = 0
    for row in matches:
        if printed >= int(args.limit):
            break
        tok = _row_token(row)
        ex = _row_exchange(row)
        name = _row_name(row)
        it = str(row.get("instrumenttype") or row.get("instrumentType") or row.get("segment") or row.get("series") or "").strip()
        print(f"{ex}:{name} -> token={tok} type={it}")
        printed += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
