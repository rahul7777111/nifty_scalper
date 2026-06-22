from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
RECON_AUDIT_PATH = REPORTS_DIR / "reconciliation_audit.json"


class OrderLifecycleState(str, Enum):
    PENDING_SUBMIT = "PENDING_SUBMIT"
    LIVE_ON_BOOK = "LIVE_ON_BOOK"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCEL_CONFIRMED = "CANCEL_CONFIRMED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FULLY_FILLED = "FULLY_FILLED"


@dataclass
class OrderStateMachine:
    symbol: str
    side: str
    quantity: int
    order_id: str = ""
    state: OrderLifecycleState = OrderLifecycleState.PENDING_SUBMIT
    submitted_ts: float = 0.0
    last_transition_ts: float = 0.0
    cancel_requested_ts: float = 0.0
    cancel_replace_latency_ms: Optional[float] = None
    unsolicited_fill_occurred: bool = False
    reconciliation_status_code: str = "NEW"
    filled_quantity: int = 0
    fill_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    _ALLOWED_TRANSITIONS = {
        OrderLifecycleState.PENDING_SUBMIT: {OrderLifecycleState.LIVE_ON_BOOK, OrderLifecycleState.FULLY_FILLED},
        OrderLifecycleState.LIVE_ON_BOOK: {OrderLifecycleState.PENDING_CANCEL, OrderLifecycleState.PARTIALLY_FILLED, OrderLifecycleState.FULLY_FILLED},
        OrderLifecycleState.PENDING_CANCEL: {OrderLifecycleState.CANCEL_CONFIRMED, OrderLifecycleState.FULLY_FILLED, OrderLifecycleState.PARTIALLY_FILLED},
        OrderLifecycleState.CANCEL_CONFIRMED: {OrderLifecycleState.PENDING_SUBMIT, OrderLifecycleState.LIVE_ON_BOOK, OrderLifecycleState.FULLY_FILLED},
        OrderLifecycleState.PARTIALLY_FILLED: {OrderLifecycleState.PENDING_CANCEL, OrderLifecycleState.FULLY_FILLED},
        OrderLifecycleState.FULLY_FILLED: set(),
    }

    def transition(self, new_state: OrderLifecycleState, *, status_code: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            allowed = self._ALLOWED_TRANSITIONS.get(self.state, set())
            if new_state not in allowed and new_state != self.state:
                raise RuntimeError(f"Illegal order transition: {self.state} -> {new_state}")
            self.state = new_state
            self.last_transition_ts = time.time()
            self.reconciliation_status_code = str(status_code or self.reconciliation_status_code)
            if metadata:
                self.metadata.update(dict(metadata))

    def mark_submitted(self, order_id: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.order_id = str(order_id or self.order_id)
        self.submitted_ts = time.time()
        self.transition(OrderLifecycleState.LIVE_ON_BOOK, status_code="LIVE", metadata=metadata)

    def request_cancel(self) -> None:
        self.cancel_requested_ts = time.time()
        self.transition(OrderLifecycleState.PENDING_CANCEL, status_code="CANCEL_REQUESTED")

    def confirm_cancel(self) -> None:
        self.cancel_replace_latency_ms = max(0.0, (time.time() - float(self.cancel_requested_ts or time.time())) * 1000.0)
        self.transition(
            OrderLifecycleState.CANCEL_CONFIRMED,
            status_code="CANCEL_CONFIRMED",
            metadata={"cancel_replace_latency_ms": self.cancel_replace_latency_ms},
        )

    def mark_fill(self, *, fill_qty: int, fill_price: Optional[float], unsolicited: bool = False) -> None:
        self.filled_quantity = max(self.filled_quantity, int(fill_qty or 0))
        self.fill_price = float(fill_price) if fill_price is not None else self.fill_price
        self.unsolicited_fill_occurred = bool(unsolicited)
        next_state = (
            OrderLifecycleState.FULLY_FILLED
            if self.filled_quantity >= int(self.quantity)
            else OrderLifecycleState.PARTIALLY_FILLED
        )
        status = "LATE_FILL_ABORT" if unsolicited and self.state == OrderLifecycleState.PENDING_CANCEL else "FILLED"
        self.transition(
            next_state,
            status_code=status,
            metadata={
                "unsolicited_fill_occurred": bool(unsolicited),
                "fill_price": self.fill_price,
                "filled_quantity": int(self.filled_quantity),
            },
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": int(self.quantity),
            "order_id": self.order_id,
            "state": self.state.value,
            "submitted_ts": float(self.submitted_ts or 0.0),
            "last_transition_ts": float(self.last_transition_ts or 0.0),
            "cancel_requested_ts": float(self.cancel_requested_ts or 0.0),
            "cancel_replace_latency_ms": self.cancel_replace_latency_ms,
            "unsolicited_fill_occurred": bool(self.unsolicited_fill_occurred),
            "reconciliation_status_code": str(self.reconciliation_status_code or ""),
            "filled_quantity": int(self.filled_quantity),
            "fill_price": self.fill_price,
            "metadata": dict(self.metadata or {}),
        }


class ReconciliationEngine:
    """Position reconciliation plus asynchronous order-state guards."""

    def __init__(self, check_interval_sec: float = 15.0):
        self.check_interval_sec = check_interval_sec
        self.last_check_ts = 0.0
        self.broker_cache = None
        self.mismatch_detected = False
        self.last_mismatch_reason = ""
        self.reconciliation_history: List[Dict[str, Any]] = []
        self._order_state_machines: Dict[str, OrderStateMachine] = {}
        self._halted_symbols: set[str] = set()
        self._lock = threading.RLock()

    def register_order(self, *, symbol: str, side: str, quantity: int, metadata: Optional[Dict[str, Any]] = None) -> OrderStateMachine:
        fsm = OrderStateMachine(symbol=str(symbol or ""), side=str(side or ""), quantity=int(quantity or 0), metadata=dict(metadata or {}))
        with self._lock:
            synthetic_id = f"pending::{uuid.uuid4().hex[:12]}"
            self._order_state_machines[synthetic_id] = fsm
            self._write_reconciliation_audit()
        return fsm

    def bind_order_id(self, fsm: OrderStateMachine, order_id: str) -> None:
        with self._lock:
            old_key = None
            for key, value in self._order_state_machines.items():
                if value is fsm:
                    old_key = key
                    break
            if old_key is not None:
                del self._order_state_machines[old_key]
            fsm.mark_submitted(str(order_id or ""))
            self._order_state_machines[str(order_id or old_key or uuid.uuid4().hex)] = fsm
            self._write_reconciliation_audit()

    def mark_cancel_requested(self, fsm: OrderStateMachine) -> None:
        with self._lock:
            fsm.request_cancel()
            self._write_reconciliation_audit()

    def resolve_cancel_result(
        self,
        *,
        fsm: OrderStateMachine,
        cancel_succeeded: bool,
        fill_qty: int = 0,
        fill_price: Optional[float] = None,
        unsolicited: bool = False,
    ) -> None:
        with self._lock:
            if cancel_succeeded:
                fsm.confirm_cancel()
                fsm.reconciliation_status_code = "CLEAN_REPLACE"
            else:
                fsm.mark_fill(fill_qty=fill_qty or int(fsm.quantity), fill_price=fill_price, unsolicited=unsolicited or True)
            self._write_reconciliation_audit()

    def is_symbol_halted(self, symbol: str) -> bool:
        with self._lock:
            return str(symbol or "").strip().upper() in self._halted_symbols

    def halt_symbol(self, symbol: str, reason: str) -> None:
        sym = str(symbol or "").strip().upper()
        with self._lock:
            if sym:
                self._halted_symbols.add(sym)
            self.reconciliation_history.append(
                {
                    "ts": time.time(),
                    "status": "HALT_SYMBOL",
                    "symbol": sym,
                    "reason": str(reason or ""),
                }
            )
            self._trim_history()
            self._write_reconciliation_audit()

    def handle_unsolicited_execution(self, strategy_bot, event: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(event.get("symbol") or "").strip().upper()
        order_id = str(event.get("order_id") or "").strip()
        fill_qty = int(event.get("filled_quantity") or event.get("quantity") or 0)
        fill_price = float(event.get("fill_price") or 0.0) if event.get("fill_price") is not None else None
        reason = "unsolicited_execution_report"
        self.halt_symbol(symbol, reason)
        strategy_bot._entries_paused = True

        fsm = None
        with self._lock:
            fsm = self._order_state_machines.get(order_id)
            if fsm is not None:
                fsm.mark_fill(fill_qty=fill_qty or int(fsm.quantity), fill_price=fill_price, unsolicited=True)
            self._write_reconciliation_audit()

        prediction_id = f"PRED_ORPHAN_{uuid.uuid4().hex[:8].upper()}"
        entry_price = float(fill_price or 0.0)
        meta = {
            "prediction_id": prediction_id,
            "reason": "unsolicited_execution",
            "reconciliation_status_code": "UNSOLICITED_FILL",
            "unsolicited_fill_occurred": True,
            "cancel_replace_latency_ms": None,
        }
        if entry_price > 0.0:
            try:
                meta["triple_barrier"] = strategy_bot._build_triple_barrier_metadata(
                    fill_price=entry_price,
                    prediction_id=prediction_id,
                    bar_timestamp=None,
                )
            except Exception:
                pass

        emergency_trade = {
            "trade_id": strategy_bot._new_trade_id("D"),
            "name": "orphan_execution_guard",
            "symbol": symbol,
            "token": event.get("token"),
            "exchange": event.get("exchange"),
            "side": str(event.get("side") or "BUY"),
            "quantity": int(fill_qty or 0),
            "entry_spot": entry_price,
            "entry_price": entry_price,
            "atr": 0.0,
            "legs": [
                {
                    "symbol": symbol,
                    "token": event.get("token"),
                    "exchange": event.get("exchange"),
                    "quantity": int(fill_qty or 0),
                    "side": str(event.get("side") or "BUY"),
                    "entry_price": entry_price,
                }
            ],
            "meta": meta,
            "opened_ts": time.time(),
            "entry_time": time.time(),
        }
        try:
            strategy_bot.state.open_directional.append(emergency_trade)
        except Exception:
            pass
        if getattr(strategy_bot, "notifier", None) is not None:
            try:
                strategy_bot.notifier.send_message(
                    f"<b>Unsolicited execution halted</b>\n<code>{symbol}</code> qty={fill_qty} order_id={order_id}"
                )
            except Exception:
                pass
        return emergency_trade

    def on_execution_event(self, strategy_bot, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        order_id = str(event.get("order_id") or "").strip()
        with self._lock:
            fsm = self._order_state_machines.get(order_id)
        if fsm is None:
            return self.handle_unsolicited_execution(strategy_bot, event)
        with self._lock:
            fsm.mark_fill(
                fill_qty=int(event.get("filled_quantity") or event.get("quantity") or 0),
                fill_price=float(event.get("fill_price")) if event.get("fill_price") is not None else None,
                unsolicited=False,
            )
            self._write_reconciliation_audit()
        return None

    def _trim_history(self) -> None:
        if len(self.reconciliation_history) > 100:
            self.reconciliation_history = self.reconciliation_history[-100:]

    def _write_reconciliation_audit(self) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.time(),
            "mismatch_detected": bool(self.mismatch_detected),
            "last_mismatch_reason": str(self.last_mismatch_reason or ""),
            "halted_symbols": sorted(self._halted_symbols),
            "order_state_machines": {
                key: fsm.snapshot()
                for key, fsm in self._order_state_machines.items()
            },
            "history": list(self.reconciliation_history[-50:]),
        }
        RECON_AUDIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def trigger_immediate_reconciliation(self, strategy_bot, reason: str = "FSM_TRANSITION") -> bool:
        cfg = getattr(strategy_bot, "cfg", None)
        is_live = bool(getattr(cfg, "enable_live_trading", False)) if cfg is not None else True
        if not is_live:
            return True
        logger.info("[RECONCILIATION] Immediate trigger fired: %s", reason)
        old_ts = self.last_check_ts
        self.last_check_ts = 0.0
        result = self.reconcile_positions(strategy_bot)
        if result:
            self.last_check_ts = time.time()
        else:
            self.last_check_ts = old_ts
        return result

    def reconcile_positions(self, strategy_bot) -> bool:
        cfg = getattr(strategy_bot, "cfg", None)
        is_live = bool(getattr(cfg, "enable_live_trading", False)) if cfg is not None else True
        if not is_live:
            if self.mismatch_detected:
                self.mismatch_detected = False
                self.last_mismatch_reason = ""
            return True
        now = time.time()
        if now - self.last_check_ts < self.check_interval_sec:
            return not self.mismatch_detected
        self.last_check_ts = now
        try:
            broker_pos = strategy_bot.client.get_positions()
            if broker_pos is None:
                logger.warning("[RECONCILIATION] Broker positions call returned None. Skipping this cycle.")
                return not self.mismatch_detected
            self.broker_cache = broker_pos
            bot_net_qty = self._compile_bot_positions(strategy_bot)
            broker_net_qty = self._compile_broker_positions(broker_pos)
            target_root = None
            if hasattr(strategy_bot, "cfg") and strategy_bot.cfg is not None:
                raw_sym = str(getattr(strategy_bot.cfg, "symbol", "") or "").strip().upper()
                if ":" in raw_sym:
                    raw_sym = raw_sym.split(":", 1)[1]
                for root in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"):
                    if raw_sym.startswith(root):
                        target_root = root
                        break
                if not target_root:
                    target_root = "".join(ch for ch in raw_sym if ch.isalpha())
            mismatches = self.detect_mismatches(bot_net_qty, broker_net_qty, target_symbol_root=target_root)
            if mismatches:
                reason = "; ".join(mismatches)
                self.emergency_freeze(strategy_bot, reason)
                return False
            if self.mismatch_detected:
                logger.info("[RECONCILIATION] Positions successfully re-reconciled. Clear status.")
                self.mismatch_detected = False
                self.last_mismatch_reason = ""
            self.reconciliation_history.append({"ts": time.time(), "status": "SUCCESS", "reason": "All positions match perfectly."})
            self._trim_history()
            self._write_reconciliation_audit()
            return True
        except Exception as exc:
            logger.error("[RECONCILIATION] Error during execution: %s", exc)
            return not self.mismatch_detected

    def _compile_bot_positions(self, bot) -> Dict[str, int]:
        net_qty: Dict[str, int] = {}

        def _add(symbol: str, qty: int, side: str):
            sym = str(symbol or "").strip().upper()
            if not sym:
                return
            q = int(qty)
            if str(side).upper() == "SELL":
                q = -q
            net_qty[sym] = net_qty.get(sym, 0) + q

        if hasattr(bot.state, "open_multi") and bot.state.open_multi:
            for trade in bot.state.open_multi:
                for leg in trade.get("legs") or []:
                    _add(leg.get("symbol"), leg.get("quantity"), leg.get("side"))
        if hasattr(bot.state, "open_directional") and bot.state.open_directional:
            for trade in bot.state.open_directional:
                for leg in trade.get("legs") or []:
                    _add(leg.get("symbol"), leg.get("quantity"), leg.get("side"))
        return {k: v for k, v in net_qty.items() if v != 0}

    def _compile_broker_positions(self, broker_pos: List[Dict[str, Any]]) -> Dict[str, int]:
        net_qty: Dict[str, int] = {}
        for pos in broker_pos:
            if not isinstance(pos, dict):
                continue
            symbol = pos.get("tradingsymbol") or pos.get("tradingSymbol") or pos.get("symbol")
            if not symbol:
                continue
            symbol = str(symbol).strip().upper()
            if ":" in symbol:
                symbol = symbol.split(":", 1)[1]
            try:
                qty = int(pos.get("quantity") or pos.get("netqty") or pos.get("netQty") or 0)
            except Exception:
                qty = 0
            if qty != 0:
                net_qty[symbol] = net_qty.get(symbol, 0) + qty
        return net_qty

    def detect_mismatches(self, bot_qty: Dict[str, int], broker_qty: Dict[str, int], target_symbol_root: Optional[str] = None) -> List[str]:
        mismatches = []
        all_symbols = set(bot_qty.keys()) | set(broker_qty.keys())
        for sym in all_symbols:
            if target_symbol_root:
                sym_clean = str(sym).strip().upper()
                if ":" in sym_clean:
                    sym_clean = sym_clean.split(":", 1)[1]
                if not sym_clean.startswith(target_symbol_root.strip().upper()):
                    continue
            b_qty = bot_qty.get(sym, 0)
            br_qty = broker_qty.get(sym, 0)
            if b_qty != br_qty:
                if br_qty == 0:
                    mismatches.append(f"Missing position: Bot expected {b_qty} for {sym}, Broker reports 0")
                elif b_qty == 0:
                    mismatches.append(f"Ghost position: Broker has {br_qty} for {sym}, Bot expected 0")
                else:
                    mismatches.append(f"Quantity Mismatch on {sym}: Bot has {b_qty}, Broker has {br_qty}")
        return mismatches

    def emergency_freeze(self, strategy_bot, reason: str) -> None:
        self.mismatch_detected = True
        self.last_mismatch_reason = reason
        cfg = getattr(strategy_bot, "cfg", None)
        is_live = bool(getattr(cfg, "enable_live_trading", False)) if cfg is not None else True
        if not is_live:
            logger.warning("[RECONCILIATION][PAPER] Simulation mismatch detected: %s", reason)
            return
        strategy_bot._entries_paused = True
        self.reconciliation_history.append({"ts": time.time(), "status": "FREEZE", "reason": reason})
        self._trim_history()
        self._write_reconciliation_audit()
        print("\n==================================================")
        print("!!! EMERGENCY RECONCILIATION FREEZE ACTIVE !!!")
        print(f"Reason: {reason}")
        print("==================================================\n")
        if getattr(strategy_bot, "notifier", None) is not None:
            text = (
                f"🚨 <b>EMERGENCY FREEZE ALERT</b> 🚨\n\n"
                f"Position reconciliation mismatch detected:\n"
                f"<code>{reason}</code>\n\n"
                f"<i>Bot trade entries have been frozen. Live monitoring required.</i>"
            )
            strategy_bot.notifier.send_message(text)
