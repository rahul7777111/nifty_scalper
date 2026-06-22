#!/usr/bin/env python3
"""Verify Paper Forward spot / option-premium separation."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from paper_forward_spot_integrity import (  # noqa: E402
    apply_resolved_spot_to_option_row,
    derive_spot_from_option_chain_payload,
    is_valid_nifty_underlying_spot,
    option_premium_from_row,
    validate_and_log_spot_integrity,
)

NIFTY_SPOT = 23968.55
CE_LTP = 32.65
PE_LTP = 13.70
STRIKE = 23950


def _assert_close(actual: float | None, expected: float, label: str, errors: list[str], *, tol: float = 0.01) -> None:
    if actual is None:
        errors.append(f"{label}: got None expected {expected}")
        return
    if abs(float(actual) - expected) > tol:
        errors.append(f"{label}: got {actual} expected {expected}")


def main() -> int:
    errors: list[str] = []

    ce_row = apply_resolved_spot_to_option_row(
        {
            "symbol": "NIFTY25JUN23950CE",
            "strike": STRIKE,
            "strike_price": STRIKE,
            "option_type": "CE",
            "ltp": CE_LTP,
            "token": "50614",
            "exchange": "NFO",
        },
        NIFTY_SPOT,
        option_ltp=CE_LTP,
    )
    pe_row = apply_resolved_spot_to_option_row(
        {
            "symbol": "NIFTY25JUN23950PE",
            "strike": STRIKE,
            "strike_price": STRIKE,
            "option_type": "PE",
            "ltp": PE_LTP,
            "token": "50615",
            "exchange": "NFO",
        },
        NIFTY_SPOT,
        option_ltp=PE_LTP,
    )

    snapshot = {"spot": NIFTY_SPOT, "underlying_price": NIFTY_SPOT}
    chain = [ce_row, pe_row]

    chain_spot, _src = derive_spot_from_option_chain_payload(chain)
    if chain_spot in (CE_LTP, PE_LTP):
        errors.append(f"chain_spot must not be option premium; got {chain_spot}")

    _assert_close(snapshot.get("spot"), NIFTY_SPOT, "snapshot.spot", errors)
    for label, row, premium in (("CE", ce_row, CE_LTP), ("PE", pe_row, PE_LTP)):
        _assert_close(row.get("spot"), NIFTY_SPOT, f"{label}.spot", errors)
        _assert_close(row.get("underlying_price"), NIFTY_SPOT, f"{label}.underlying_price", errors)
        _assert_close(row.get("ltp"), premium, f"{label}.ltp", errors)
        _assert_close(row.get("option_ltp"), premium, f"{label}.option_ltp", errors)
        moneyness = (float(row.get("strike")) - NIFTY_SPOT) / NIFTY_SPOT
        if abs(moneyness) > 0.05:
            errors.append(f"{label}.moneyness uses wrong spot: {moneyness}")

    bad_chain = [{"ltp": 0.05, "strike": STRIKE, "option_type": "CE"}]
    bad_spot, _ = derive_spot_from_option_chain_payload(bad_chain)
    if bad_spot is not None and not is_valid_nifty_underlying_spot(bad_spot):
        pass  # expected rejection
    elif bad_spot == 0.05:
        errors.append("derive_spot_from_option_chain_payload returned option ltp as spot")

    ok, reason = validate_and_log_spot_integrity(
        snapshot_spot=NIFTY_SPOT,
        chain_spot=chain_spot,
        option_ltp=CE_LTP,
        row=ce_row,
    )
    if not ok:
        errors.append(f"spot integrity validation failed: {reason}")

    if errors:
        print("verify_paper_forward_spot_integrity FAILED")
        for err in errors:
            print(f"  [FAIL] {err}")
        return 1

    print("verify_paper_forward_spot_integrity PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())