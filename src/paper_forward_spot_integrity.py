#!/usr/bin/env python3
"""
Paper Forward spot / option-price separation helpers.

Ensures NIFTY underlying spot is never confused with option premium in
option chain rows, snapshots, or derived chain_spot values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

NIFTY_SPOT_MIN = 10_000.0
NIFTY_SPOT_MAX = 50_000.0

# Keys that may carry underlying index spot (never option premium).
UNDERLYING_SPOT_KEYS: Tuple[str, ...] = (
    "underlyingValue",
    "underlying_value",
    "underlyingPrice",
    "underlying_price",
    "underlying_spot_price",
    "underlyingSpotPrice",
    "underlying_ltp",
    "underlyingLtp",
    "index_spot",
    "indexSpot",
    "index_ltp",
    "indexLtp",
    "spot_price",
    "spotPrice",
)

# Row-level spot keys — only accepted when value passes NIFTY range check.
ROW_SPOT_KEYS: Tuple[str, ...] = ("spot", "underlying", "price")

OPTION_PRICE_KEYS: Tuple[str, ...] = (
    "ltp",
    "LTP",
    "last_price",
    "last_price_value",
    "lastPrice",
    "last_traded_price",
    "option_ltp",
    "ctx_option_price",
    "mid",
    "theoretical_price",
    "traded_price",
    "lastTradedPrice",
)


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "n/a", "N/A", "nan"):
        return None
    try:
        v = float(value)
        if v == v and v > 0:
            return v
    except Exception:
        pass
    return None


def is_valid_nifty_underlying_spot(value: Any) -> bool:
    spot = _to_float(value)
    if spot is None:
        return False
    return NIFTY_SPOT_MIN <= spot <= NIFTY_SPOT_MAX


def option_premium_from_row(row: dict) -> Optional[float]:
    """Extract option premium from a chain row (not underlying spot)."""
    if not isinstance(row, dict):
        return None
    for key in OPTION_PRICE_KEYS:
        val = _to_float(row.get(key))
        if val is not None:
            return val
    try:
        bid = _to_float(row.get("bid") or row.get("bid_price") or row.get("best_bid") or row.get("best_bid_price"))
        ask = _to_float(row.get("ask") or row.get("ask_price") or row.get("best_ask") or row.get("best_ask_price"))
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        if bid is not None:
            return bid
        if ask is not None:
            return ask
    except Exception:
        pass
    return None


def _walk_dict_nodes(obj: object) -> List[dict]:
    nodes: List[dict] = []
    if isinstance(obj, dict):
        nodes.append(obj)
        for value in obj.values():
            nodes.extend(_walk_dict_nodes(value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            nodes.extend(_walk_dict_nodes(item))
    return nodes


def _spot_from_node(node: dict, keys: Sequence[str]) -> Tuple[Optional[float], str]:
    for key in keys:
        value = _to_float(node.get(key))
        if value is not None and is_valid_nifty_underlying_spot(value):
            return value, f"option_chain.{key}"
    return None, ""


def derive_spot_from_option_chain_payload(payload: object) -> Tuple[Optional[float], str]:
    """
    Derive NIFTY underlying spot from an option-chain payload.

    Never uses option premium fields (ltp, last_price, mid, etc.).
    Rejects values outside the NIFTY spot band.
    """
    if payload in (None, "", [], ()):
        return None, ""

    # Envelope / top-level dict first (broker often puts index spot here).
    if isinstance(payload, dict):
        spot, src = _spot_from_node(payload, UNDERLYING_SPOT_KEYS)
        if spot is not None:
            return spot, src
        for key in ROW_SPOT_KEYS:
            value = _to_float(payload.get(key))
            if value is not None and is_valid_nifty_underlying_spot(value):
                return value, f"option_chain.{key}"

    rows: List[dict] = []
    if isinstance(payload, (list, tuple)):
        rows = [r for r in payload if isinstance(r, dict)]
    elif isinstance(payload, dict):
        for subkey in ("option_chain", "rows", "data", "records", "result"):
            sub = payload.get(subkey)
            if isinstance(sub, (list, tuple)):
                rows = [r for r in sub if isinstance(r, dict)]
                if rows:
                    break

    # Aggregate from rows using only underlying-specific keys.
    for row in rows:
        spot, src = _spot_from_node(row, UNDERLYING_SPOT_KEYS)
        if spot is not None:
            return spot, src

    # Row spot/underlying/price only when in NIFTY band (rejects option premiums).
    for row in rows:
        for key in ROW_SPOT_KEYS:
            value = _to_float(row.get(key))
            if value is not None and is_valid_nifty_underlying_spot(value):
                return value, f"option_chain_row.{key}"

    # Last resort: nested nodes, still excluding option premium keys.
    for node in _walk_dict_nodes(payload):
        spot, src = _spot_from_node(node, UNDERLYING_SPOT_KEYS)
        if spot is not None:
            return spot, src

    return None, ""


def apply_resolved_spot_to_option_row(
    row: dict,
    resolved_spot: float,
    *,
    option_ltp: Optional[float] = None,
) -> dict:
    """Enforce separation of underlying spot vs option premium on one row."""
    out = dict(row or {})
    spot_f = float(resolved_spot)
    premium = option_ltp if option_ltp is not None else option_premium_from_row(out)

    out["underlying_price"] = spot_f
    out["ctx_spot"] = spot_f
    out["spot"] = spot_f
    out["underlying"] = spot_f
    out["underlying_ltp"] = spot_f

    if premium is not None and premium > 0:
        out["ltp"] = premium
        out["option_ltp"] = premium
        out["ctx_option_price"] = premium
        out.setdefault("last_price", premium)

    return out


def validate_and_log_spot_integrity(
    *,
    snapshot_spot: Any,
    chain_spot: Any = None,
    option_ltp: Any = None,
    row: Optional[dict] = None,
) -> Tuple[bool, str]:
    """
    Validate spot integrity and emit [PF-SPOT-INTEGRITY] log.

    Returns (ok, reason).
    """
    snap = _to_float(snapshot_spot)
    chain = _to_float(chain_spot) if chain_spot is not None else None
    opt = _to_float(option_ltp) if option_ltp is not None else None

    if row is not None and opt is None:
        opt = option_premium_from_row(row)
    if row is not None and chain is None:
        chain = _to_float(row.get("spot") or row.get("underlying_price"))

    reason = ""
    ok = True
    if snap is None or not is_valid_nifty_underlying_spot(snap):
        ok = False
        reason = "INVALID_UNDERLYING_SPOT"
    elif chain is not None and not is_valid_nifty_underlying_spot(chain):
        ok = False
        reason = "INVALID_CHAIN_SPOT"
    elif snap is not None and chain is not None and abs(snap - chain) > 500:
        ok = False
        reason = "CHAIN_SPOT_MISMATCH"
    elif opt is not None and snap is not None and abs(opt - snap) < 100:
        # Option premium should never be near index spot magnitude confusion.
        if not is_valid_nifty_underlying_spot(opt):
            ok = False
            reason = "OPTION_LTP_NEAR_SPOT_CONFUSION"

    status = "OK" if ok else "FAILED"
    print(
        f"[PF-SPOT-INTEGRITY] snapshot_spot={snap if snap is not None else 'n/a'} "
        f"chain_spot={chain if chain is not None else 'n/a'} "
        f"option_ltp={opt if opt is not None else 'n/a'} "
        f"status={status} reason={reason or 'none'}"
    )
    return ok, reason


def log_option_row_check(row: dict, *, resolved_spot: Optional[float] = None) -> None:
    """Emit a single [OC-ROW-CHECK] debug sample."""
    if not isinstance(row, dict):
        return
    sym = str(row.get("symbol") or row.get("trading_symbol") or row.get("tradingsymbol") or "")
    strike = row.get("strike") or row.get("strike_price") or ""
    opt_type = row.get("option_type") or row.get("type") or ""
    spot = resolved_spot if resolved_spot is not None else row.get("spot")
    opt_ltp = option_premium_from_row(row)
    up = row.get("underlying_price")
    print(
        f"[OC-ROW-CHECK] symbol={sym} strike={strike} type={opt_type} "
        f"spot={spot} option_ltp={opt_ltp} underlying_price={up}"
    )
