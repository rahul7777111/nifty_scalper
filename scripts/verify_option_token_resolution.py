#!/usr/bin/env python3
"""Verify ScripMaster resolves synthetic-style symbols to broker tokens."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripmaster import ScripMaster, parse_option_tradingsymbol


def main() -> int:
    errors: list[str] = []
    csv_path = REPO / "instrument (2).csv"
    if not csv_path.exists():
        for p in REPO.glob("instrument*.csv"):
            csv_path = p
            break
    if not csv_path.exists():
        print("verify_option_token_resolution SKIPPED (no instrument csv)")
        return 0

    sm = ScripMaster(str(csv_path))
    synthetic_sym = "NIFTY16JUN2623950CE"
    parsed = parse_option_tradingsymbol(synthetic_sym)
    if not parsed:
        errors.append("failed to parse synthetic symbol format")
    tok = sm.token_for_option_symbol(synthetic_sym, exch="NFO")
    if not tok:
        errors.append(f"no token for {synthetic_sym}")
    elif tok != "50613":
        errors.append(f"unexpected token for {synthetic_sym}: {tok}")

    row = sm.lookup_option_contract(
        underlying="NIFTY",
        expiry=date(2026, 6, 16),
        strike=23950,
        option_type="CE",
        exch="NFO",
    )
    if row is None:
        errors.append("structured lookup failed")
    elif str(row.tradingsymbol) != "NIFTY2661623950CE":
        errors.append(f"unexpected tradingsymbol: {row.tradingsymbol}")

    if errors:
        print("verify_option_token_resolution FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("verify_option_token_resolution PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())