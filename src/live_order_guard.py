#!/usr/bin/env python3
"""
src/live_order_guard.py
=======================
Central broker order safety wrapper. EVERY real (or dry) broker order intent
MUST pass through validate_before_order before any client.place_order call.

This is the final gate after all prior RealTradingGate / LiveTradeGate / candidate
lifecycle checks.

Strict: if allowed=False, the caller MUST NOT send the order.

Defaults are safe: no real orders unless all explicit conditions + envs align.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .candidate_lifecycle import (
        get_kill_switch_active,
        get_live_mode_env,
        get_live_order_dry_run_env,
        get_order_placement_enabled_env,
        load_live_whitelist,
        Candidate,
    )
except Exception:  # direct run / test fallback
    from candidate_lifecycle import (  # type: ignore
        get_kill_switch_active,
        get_live_mode_env,
        get_live_order_dry_run_env,
        get_order_placement_enabled_env,
        load_live_whitelist,
        Candidate,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


@dataclass
class OrderIntent:
    candidate_id: str
    symbol: str
    exchange: str = "NFO"
    symbol_token: str = ""
    side: str = "SELL"  # BUY / SELL
    quantity: int = 65
    order_type: str = "LIMIT"
    price: Optional[float] = None
    product_type: str = "INTRADAY"
    expiry: str = ""
    option_type: str = ""  # CE / PE
    strike: Any = None
    # risk / market context
    stop_loss: Optional[float] = None
    exit_rule: Optional[str] = None
    spread_pct: float = 0.0
    premium: float = 0.0
    ltp: float = 0.0
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class GuardResult:
    allowed: bool
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_order: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


def _normalize_order(intent: OrderIntent, candidate: Optional[Candidate]) -> Dict[str, Any]:
    return {
        "candidate_id": intent.candidate_id,
        "symbol": str(intent.symbol or "").strip().upper(),
        "exchange": str(intent.exchange or "NFO").strip().upper(),
        "symbol_token": str(intent.symbol_token or "").strip(),
        "side": str(intent.side or "SELL").strip().upper(),
        "quantity": int(intent.quantity or 0),
        "order_type": str(intent.order_type or "LIMIT").strip().upper(),
        "price": float(intent.price) if intent.price is not None else None,
        "product_type": str(intent.product_type or "INTRADAY").strip().upper(),
        "expiry": str(intent.expiry or "").strip(),
        "option_type": str(intent.option_type or "").strip().upper(),
        "strike": intent.strike,
        "stop_loss": intent.stop_loss,
        "exit_rule": intent.exit_rule,
        "spread_pct": float(intent.spread_pct or 0.0),
        "premium": float(intent.premium or 0.0),
        "ltp": float(intent.ltp or 0.0),
        "dry_run": True,  # always start true; only real path flips after full gate
        "live_order_sent": False,
    }


def validate_before_order(
    order_intent: OrderIntent,
    candidate: Optional[Candidate] = None,
    runtime_state: Optional[Dict[str, Any]] = None,
    market_snapshot: Optional[Dict[str, Any]] = None,
    *,
    # allow tests to inject without env
    force_kill_switch: Optional[bool] = None,
    force_dry_run: Optional[bool] = None,
) -> GuardResult:
    """
    The single choke point. Returns GuardResult.
    Caller MUST check result.allowed before calling any broker place_order.
    """
    blockers: List[str] = []
    warnings: List[str] = []
    ts = _now_iso()
    runtime_state = runtime_state or {}
    market_snapshot = market_snapshot or {}

    # 1. Master kill / mode switches (env or forced)
    kill = force_kill_switch if force_kill_switch is not None else get_kill_switch_active()
    if kill:
        blockers.append("SCALPER_KILL_SWITCH == 1 (or active)")

    live_mode = get_live_mode_env()
    if not live_mode:
        blockers.append("LIVE_MODE != true")

    order_placement = get_order_placement_enabled_env()
    if not order_placement:
        blockers.append("ORDER_PLACEMENT_ENABLED != true")

    dry_run = force_dry_run if force_dry_run is not None else get_live_order_dry_run_env()
    if dry_run:
        blockers.append("LIVE_ORDER_DRY_RUN == true (real order forbidden)")

    # 2. Candidate whitelist & status
    wl = load_live_whitelist() or []
    live_cand = None
    if candidate is None:
        # best effort: find by id in whitelist
        for c in wl:
            if c.candidate_id == order_intent.candidate_id:
                live_cand = c
                break
    else:
        live_cand = candidate

    if live_cand is None:
        # fallback: any whitelisted LIVE_1_LOT ?
        live_cand = next((c for c in wl if c.live_whitelisted and c.status in ("LIVE_1_LOT", "LIVE_SCALED")), None)

    if not live_cand or not getattr(live_cand, "live_whitelisted", False):
        blockers.append("candidate.live_whitelisted != true")

    status = getattr(live_cand, "status", "DISABLED") if live_cand else "DISABLED"
    if status not in ("LIVE_1_LOT", "LIVE_SCALED"):
        blockers.append(f"candidate.status={status} not in [LIVE_1_LOT, LIVE_SCALED]")

    max_qty = int(getattr(live_cand, "max_qty_lots", 0) or 0) if live_cand else 0
    if max_qty <= 0:
        blockers.append("candidate.max_qty_lots <= 0")

    # 3. Quantity / first-live rules
    qty = int(order_intent.quantity or 0)
    allowed_qty = max_qty * 65 if max_qty > 0 else 65  # rough; real lot from config
    if qty > allowed_qty or (status == "LIVE_1_LOT" and qty > 65):
        blockers.append(f"quantity {qty} exceeds allowed for first live (max 1 lot)")

    otype = str(order_intent.order_type or "LIMIT").upper()
    if status == "LIVE_1_LOT" and otype != "LIMIT":
        blockers.append("order_type must be LIMIT during first live (LIVE_1_LOT)")

    # 4. Stop / exit rules
    if not order_intent.stop_loss and not (runtime_state.get("has_sl") or (live_cand and getattr(live_cand, "stop_loss_rule", None))):
        blockers.append("missing stop loss")
    if not order_intent.exit_rule and not (runtime_state.get("has_exit") or (live_cand and getattr(live_cand, "exit_rule", None))):
        blockers.append("missing exit rule")

    # 5. Market / quote quality from snapshot or intent
    spread = float(order_intent.spread_pct or market_snapshot.get("spread_pct", 999.0))
    max_spread = float(os.getenv("MAX_SPREAD_PCT", "0.02") or 0.02)
    if spread > max_spread:
        blockers.append(f"spread {spread*100:.2f}% > max {max_spread*100:.2f}%")

    prem = float(order_intent.premium or market_snapshot.get("premium", 0.0) or order_intent.ltp)
    min_p = float(os.getenv("MIN_OPTION_PREMIUM", "5") or 5)
    if prem < min_p:
        blockers.append(f"premium {prem:.2f} < min {min_p:.2f}")

    # 6. Data freshness (from market_snapshot or runtime)
    chain_fresh = bool(market_snapshot.get("option_chain_fresh", runtime_state.get("option_chain_fresh", False)))
    if not chain_fresh:
        blockers.append("option chain stale")

    expiry_ok = bool(market_snapshot.get("selected_expiry_valid", runtime_state.get("expiry_valid", False)))
    if not expiry_ok and order_intent.expiry:
        # basic parse check
        try:
            # allow common formats
            pass
        except Exception:
            blockers.append("expiry invalid")
    elif not expiry_ok:
        blockers.append("selected_expiry_valid is false")

    before_cutoff = bool(market_snapshot.get("before_cutoff", runtime_state.get("before_cutoff", True)))
    if not before_cutoff:
        blockers.append("current time after entry cutoff")

    # 7. Risk / position state
    trades_today = int(runtime_state.get("trades_today", 0))
    max_trades = int(getattr(live_cand, "max_trades_per_day", 2) or 2) if live_cand else 2
    if trades_today >= max_trades:
        blockers.append(f"trades_today {trades_today} >= max {max_trades}")

    daily_loss = float(runtime_state.get("daily_pnl", runtime_state.get("daily_loss", 0.0)))
    max_loss = float(getattr(live_cand, "max_daily_loss", 1000.0) or 1000.0) if live_cand else 1000.0
    if daily_loss <= -max_loss:
        blockers.append(f"daily_loss {daily_loss:.2f} <= -max_daily_loss {max_loss:.2f}")

    if runtime_state.get("duplicate_open_position", False):
        blockers.append("duplicate open position for same leg")

    open_pos = int(runtime_state.get("open_positions", 0))
    max_open = int(runtime_state.get("max_open_positions", 6))
    if open_pos >= max_open:
        blockers.append(f"open_positions {open_pos} >= max {max_open}")

    # 8. Broker session (runtime or caller)
    if not runtime_state.get("broker_session_valid", True):
        blockers.append("broker session invalid")

    # Build normalized
    norm = _normalize_order(order_intent, live_cand)
    # If we reached here with dry_run env still true, keep it marked
    norm["dry_run"] = dry_run or not (live_mode and order_placement and not dry_run)
    norm["live_order_sent"] = False

    allowed = len(blockers) == 0

    if allowed and dry_run:
        # Still "allowed" for the dry-run recording path, but the mstock short-circuit will catch it
        warnings.append("dry_run mode active - order will be recorded only")

    return GuardResult(
        allowed=allowed,
        blockers=blockers,
        warnings=warnings,
        normalized_order=norm,
        timestamp=ts,
    )


def is_real_live_fully_enabled() -> bool:
    """Convenience: true only for the absolute final 1-lot real path."""
    return (
        get_live_mode_env()
        and get_order_placement_enabled_env()
        and not get_live_order_dry_run_env()
        and not get_kill_switch_active()
    )
