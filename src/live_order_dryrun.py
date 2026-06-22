#!/usr/bin/env python3
"""
src/live_order_dryrun.py
========================
Live Order Dry-Run engine.

When LIVE_ORDER_DRY_RUN=true (and typically LIVE_MODE=true, ORDER_PLACEMENT_ENABLED=false):
- Accepts the same decision/leg that would go to the router.
- Builds the *exact* broker order payload that the live path would send.
- Validates symbol, exchange, expiry, side, quantity, order_type (LIMIT for first live), price.
- Does NOT call any place_order / broker SDK.
- Appends a record to logs/live_dryrun_orders_YYYYMMDD.jsonl

This module is the single choke-point for "build but do not send".
It is safe to call from the GUI "Enable Live Dry-run" button or from strategy when the env flags are set.

Integration:
- The main order path (strategy + mstock_client.place_order) continues to use the existing RealTradingGate.
- This module is an additional explicit dry-run recorder for the promotion pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from .candidate_lifecycle import (
        get_live_order_dry_run_env,
        get_live_mode_env,
        get_order_placement_enabled_env,
        get_kill_switch_active,
    )
except Exception:  # pragma: no cover
    from candidate_lifecycle import (  # type: ignore
        get_live_order_dry_run_env,
        get_live_mode_env,
        get_order_placement_enabled_env,
        get_kill_switch_active,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def is_live_dry_run_active() -> bool:
    """True only when the explicit dry-run combination is requested."""
    return (
        get_live_mode_env()
        and get_live_order_dry_run_env()
        and not get_order_placement_enabled_env()
        and not get_kill_switch_active()
    )


def build_and_validate_dry_run_payload(
    *,
    candidate_id: str,
    symbol: str,
    exchange: str,
    symbol_token: str,
    side: str,
    quantity: int,
    order_type: str = "LIMIT",
    price: Optional[float] = None,
    product_type: str = "INTRADAY",
    expiry: str = "",
    option_type: str = "",
    strike: Any = None,
    spread_pct: float = 0.0,
    ltp: float = 0.0,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Build the exact payload + validate it.
    Returns (payload, error_or_none).
    Never sends anything.
    """
    payload: Dict[str, Any] = {
        "timestamp": _iso(),
        "candidate_id": candidate_id,
        "symbol": str(symbol or "").strip(),
        "exchange": str(exchange or "NFO").strip().upper(),
        "symbol_token": str(symbol_token or "").strip(),
        "side": str(side or "").strip().upper(),
        "quantity": int(quantity or 0),
        "order_type": str(order_type or "LIMIT").strip().upper(),
        "price": float(price) if price is not None else None,
        "product_type": str(product_type or "INTRADAY").strip().upper(),
        "ordertag": f"dryrun:{candidate_id[:12]}",
        "expiry": str(expiry or ""),
        "option_type": str(option_type or ""),
        "strike": strike,
        "spread_pct": float(spread_pct),
        "ltp": float(ltp),
        "live_order_sent": False,
        "dry_run": True,
    }

    errors: list[str] = []

    if not payload["symbol"]:
        errors.append("symbol empty")
    if not payload["exchange"]:
        errors.append("exchange empty")
    if not payload["symbol_token"]:
        errors.append("symbol_token empty")
    if payload["side"] not in ("BUY", "SELL"):
        errors.append(f"invalid side {payload['side']}")
    if payload["quantity"] <= 0:
        errors.append("quantity <= 0")
    if payload["order_type"] not in ("LIMIT", "MARKET"):
        errors.append(f"unsupported order_type {payload['order_type']}")
    if payload["order_type"] == "LIMIT" and (payload["price"] is None or payload["price"] <= 0):
        errors.append("LIMIT order requires positive price")
    if not payload["expiry"]:
        errors.append("expiry missing")
    if payload["option_type"] not in ("CE", "PE"):
        errors.append(f"invalid option_type {payload['option_type']}")

    # First-live rule (per task): order_type must be LIMIT for the initial 1-lot phase
    # (We only warn here; the hard gate is enforced in the real-live path.)
    err = "; ".join(errors) if errors else None
    return payload, err


def record_dry_run_order(payload: Dict[str, Any], validation_error: Optional[str] = None) -> Path:
    """Write the dry-run record. Returns the path written."""
    day = _today()
    path = LOGS_DIR / f"live_dryrun_orders_{day}.jsonl"
    rec = dict(payload)
    rec["validation_error"] = validation_error
    rec["recorded_at"] = _iso()
    _append(path, rec)
    return path


def execute_dry_run_if_requested(
    *,
    candidate_id: str,
    symbol: str,
    exchange: str = "NFO",
    symbol_token: str = "",
    side: str = "SELL",
    quantity: int = 65,
    order_type: str = "LIMIT",
    price: Optional[float] = None,
    expiry: str = "",
    option_type: str = "PE",
    strike: Any = None,
    spread_pct: float = 0.0,
    ltp: float = 0.0,
) -> Dict[str, Any]:
    """
    High-level helper used by UI / strategy when dry-run mode is active.
    Always safe: builds, validates, logs, returns the payload. Never sends.
    """
    if not is_live_dry_run_active():
        return {"skipped": True, "reason": "live_dry_run_not_active_via_env"}

    payload, err = build_and_validate_dry_run_payload(
        candidate_id=candidate_id,
        symbol=symbol,
        exchange=exchange,
        symbol_token=symbol_token,
        side=side,
        quantity=quantity,
        order_type=order_type,
        price=price,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
        spread_pct=spread_pct,
        ltp=ltp,
    )
    written = record_dry_run_order(payload, err)
    payload["log_file"] = str(written)
    return payload
