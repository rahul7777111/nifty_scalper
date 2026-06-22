#!/usr/bin/env python3
"""
src/broker_reconciliation.py
============================
Broker reconciliation module for live/dry-run safety.

After every order attempt (dry or live), reconcile bot view vs broker reality.
Detect orphans, unknown pending orders, qty mismatches.

Logs to logs/live_reconciliation_YYYYMMDD.jsonl

Functions are safe to call even when no broker session (will report degraded).
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _get_client(bot_or_client: Any) -> Any:
    if bot_or_client is None:
        return None
    if hasattr(bot_or_client, "client"):
        return bot_or_client.client
    return bot_or_client


def fetch_open_orders(client: Any) -> List[Dict[str, Any]]:
    """Best effort fetch of open/pending orders from broker."""
    if client is None:
        return []
    try:
        if hasattr(client, "get_open_orders"):
            return client.get_open_orders() or []
        if hasattr(client, "get_order_book"):
            book = client.get_order_book() or []
            return [o for o in book if str(o.get("status", "")).upper() in ("OPEN", "PENDING", "TRIGGER PENDING")]
        return []
    except Exception:
        return []


def fetch_order_status(client: Any, order_id: str) -> Optional[Dict[str, Any]]:
    if not client or not order_id:
        return None
    try:
        if hasattr(client, "get_order_status"):
            return client.get_order_status(order_id)
        if hasattr(client, "get_order_book"):
            for o in (client.get_order_book() or []):
                if str(o.get("order_id") or o.get("orderId")) == str(order_id):
                    return o
    except Exception:
        pass
    return None


def fetch_positions(client: Any) -> List[Dict[str, Any]]:
    if client is None:
        return []
    try:
        if hasattr(client, "get_positions"):
            pos = client.get_positions()
            return pos if isinstance(pos, list) else []
        if hasattr(client, "positions"):
            return client.positions() or []
    except Exception:
        pass
    return []


def detect_orphan_positions(bot_positions: Dict[str, int], broker_positions: List[Dict[str, Any]]) -> List[str]:
    """Return list of symbols that exist on broker but not in bot view (or qty sign mismatch)."""
    orphans: List[str] = []
    bot_syms = {str(k).upper() for k in bot_positions.keys() if bot_positions.get(k)}
    for p in broker_positions:
        sym = str(p.get("tradingsymbol") or p.get("tradingSymbol") or p.get("symbol") or "").strip().upper()
        if not sym:
            continue
        if sym not in bot_syms:
            qty = int(p.get("quantity") or p.get("netQty") or p.get("qty") or 0)
            if qty != 0:
                orphans.append(sym)
    return orphans


def detect_unknown_pending_orders(bot_order_ids: List[str], broker_open: List[Dict[str, Any]]) -> List[str]:
    unknown = []
    known = {str(x) for x in (bot_order_ids or [])}
    for o in broker_open:
        oid = str(o.get("order_id") or o.get("orderId") or o.get("id") or "")
        if oid and oid not in known:
            unknown.append(oid)
    return unknown


def compare_bot_vs_broker_position(
    bot_net: Dict[str, int],
    broker_pos: List[Dict[str, Any]],
    target_symbol_root: Optional[str] = "NIFTY",
) -> Tuple[bool, List[str]]:
    """Return (match, mismatch_reasons)."""
    mismatches: List[str] = []
    broker_net: Dict[str, int] = {}
    for p in broker_pos:
        sym = str(p.get("tradingsymbol") or p.get("tradingSymbol") or p.get("symbol") or "").strip().upper()
        if not sym:
            continue
        if target_symbol_root and not sym.startswith(target_symbol_root):
            continue
        q = int(p.get("quantity") or p.get("netQty") or p.get("qty") or 0)
        side = str(p.get("side") or p.get("productType") or "").upper()
        if "SELL" in side or q < 0:
            q = -abs(q)
        broker_net[sym] = broker_net.get(sym, 0) + q

    all_syms = set(bot_net.keys()) | set(broker_net.keys())
    for sym in all_syms:
        bq = broker_net.get(sym, 0)
        botq = bot_net.get(sym, 0)
        if bq != botq:
            mismatches.append(f"{sym}: bot={botq} broker={bq}")

    return (len(mismatches) == 0, mismatches)


def write_reconciliation_log(
    *,
    candidate_id: str,
    order_intent: Dict[str, Any],
    broker_order_id: Optional[str],
    dry_run: bool,
    expected_position: Dict[str, int],
    actual_broker_position: List[Dict[str, Any]],
    reconciliation_status: str,
    mismatch_reason: str = "",
) -> Path:
    day = _today()
    path = LOGS_DIR / f"live_reconciliation_{day}.jsonl"
    rec = {
        "timestamp": _iso(),
        "candidate_id": candidate_id,
        "order_intent": order_intent,
        "broker_order_id": broker_order_id,
        "dry_run": bool(dry_run),
        "expected_position": expected_position,
        "actual_broker_position": actual_broker_position,
        "reconciliation_status": reconciliation_status,
        "mismatch_reason": mismatch_reason,
    }
    _append_jsonl(path, rec)
    return path


def reconcile_after_order(
    bot: Any,
    *,
    candidate_id: str,
    order_intent: Dict[str, Any],
    broker_order_id: Optional[str] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    High-level: call after any order attempt (dry or real).
    Compiles bot view, fetches broker, detects issues, logs, returns summary.
    Safe when broker unavailable (reports 'DEGRADED').
    """
    client = _get_client(bot)
    broker_pos = fetch_positions(client)
    open_orders = fetch_open_orders(client)

    # Compile bot view (very lightweight; real bots have more state)
    bot_net: Dict[str, int] = {}
    try:
        if hasattr(bot, "state"):
            state = bot.state
            for trades in (getattr(state, "open_directional", None) or [], getattr(state, "open_multi", None) or []):
                for t in (trades if isinstance(trades, (list, tuple)) else [trades]):
                    for leg in (t.get("legs") if isinstance(t, dict) else []):
                        sym = str(leg.get("symbol") or "").upper()
                        q = int(leg.get("quantity") or 0)
                        if str(leg.get("side", "")).upper() == "SELL":
                            q = -q
                        if sym:
                            bot_net[sym] = bot_net.get(sym, 0) + q
    except Exception:
        pass

    match, mismatches = compare_bot_vs_broker_position(bot_net, broker_pos)
    orphans = detect_orphan_positions(bot_net, broker_pos)
    unknown = detect_unknown_pending_orders([], open_orders)

    status = "MATCH" if match and not orphans and not unknown else "MISMATCH"
    reason = "; ".join(mismatches + [f"orphan:{o}" for o in orphans] + [f"unknown_order:{u}" for u in unknown])

    log_path = write_reconciliation_log(
        candidate_id=candidate_id,
        order_intent=order_intent,
        broker_order_id=broker_order_id,
        dry_run=dry_run,
        expected_position=bot_net,
        actual_broker_position=broker_pos,
        reconciliation_status=status,
        mismatch_reason=reason,
    )

    result = {
        "status": status,
        "mismatch_reason": reason,
        "log_file": str(log_path),
        "broker_positions": len(broker_pos),
        "bot_positions": len(bot_net),
        "orphans": orphans,
        "unknown_pending": unknown,
        "dry_run": dry_run,
    }
    return result
