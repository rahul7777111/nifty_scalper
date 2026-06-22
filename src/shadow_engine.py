#!/usr/bin/env python3
"""
src/shadow_engine.py
====================
Shadow mode execution engine.

- Consumes real option-chain snapshots and real candle snapshots when available.
- Re-uses the production candidate router (route_candidate_decision) for every decision.
- Produces virtual entries/exits only (no broker state, no paper positions).
- Builds the exact "would_order_payload" for audit (same shape as a real order would use).
- Logs three rotating daily JSONL files:
    logs/shadow_signals_YYYYMMDD.jsonl
    logs/shadow_virtual_trades_YYYYMMDD.jsonl
    logs/shadow_blocks_YYYYMMDD.jsonl
- Every record contains the fields mandated by the task:
    timestamp, candidate_id, option_type, strike, expiry, side, ltp, bid, ask, spread_pct,
    signal_score, router_reason, risk_allowed, blocked_reason, virtual_entry_price,
    virtual_exit_price, virtual_pnl, would_order_payload, live_order_sent=false

Safety contract:
- live_order_sent is ALWAYS false.
- place_order / any broker client is NEVER imported or called.
- Respects SCALPER_KILL_SWITCH.
- Only candidates present in config/shadow_candidates.json with status in {SHADOW, SHADOW_PASSED, LIVE_DRY_RUN...} and enabled=True participate.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Router (production path)
try:
    from .candidate_router import route_candidate_decision, get_last_router_decision
except Exception:  # pragma: no cover
    from candidate_router import route_candidate_decision, get_last_router_decision  # type: ignore

try:
    from .candidate_lifecycle import (
        load_shadow_candidates,
        Candidate,
        get_kill_switch_active,
        STATUS_SHADOW,
        STATUS_SHADOW_PASSED,
        STATUS_LIVE_DRY_RUN,
        STATUS_LIVE_1_LOT,
        STATUS_LIVE_SCALED,
    )
except Exception:  # pragma: no cover
    from candidate_lifecycle import (  # type: ignore
        load_shadow_candidates,
        Candidate,
        get_kill_switch_active,
        STATUS_SHADOW,
        STATUS_SHADOW_PASSED,
        STATUS_LIVE_DRY_RUN,
        STATUS_LIVE_1_LOT,
        STATUS_LIVE_SCALED,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
CONFIG_DIR = REPO_ROOT / "config"

SHADOW_ALLOWED_STATUSES = {STATUS_SHADOW, STATUS_SHADOW_PASSED, STATUS_LIVE_DRY_RUN, STATUS_LIVE_1_LOT, STATUS_LIVE_SCALED}


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class ShadowEngine:
    def __init__(
        self,
        candidate_file: Optional[str] = None,
        log_dir: str = "logs",
        artifacts_dir: str = "artifacts/candidates",
    ):
        self.log_dir = Path(log_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.candidates: List[Candidate] = []
        self._active: Dict[str, Dict[str, Any]] = {}  # per-cid virtual position state
        self._lock = threading.Lock()
        self._load_candidates(candidate_file)

        # daily log paths (rotated on date change)
        self._current_date = _today()
        self._sig_path = self.log_dir / f"shadow_signals_{self._current_date}.jsonl"
        self._trade_path = self.log_dir / f"shadow_virtual_trades_{self._current_date}.jsonl"
        self._block_path = self.log_dir / f"shadow_blocks_{self._current_date}.jsonl"

    def _load_candidates(self, candidate_file: Optional[str]) -> None:
        loaded: List[Candidate] = []
        # Prefer explicit shadow list
        if candidate_file:
            p = Path(candidate_file)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for item in data.get("candidates", []):
                    c = Candidate.from_dict(item) if hasattr(Candidate, "from_dict") else None
                    if c is None:
                        c = Candidate(
                            candidate_id=item.get("candidate_id", ""),
                            model_path=item.get("model_path", ""),
                            artifact_path=item.get("artifact_path", item.get("artifact_dir", "")),
                            status=item.get("status", STATUS_SHADOW),
                            enabled=bool(item.get("enabled", True)),
                        )
                    if c.status in SHADOW_ALLOWED_STATUSES and c.enabled:
                        loaded.append(c)
        # Fallback to the lifecycle loader
        if not loaded:
            for c in load_shadow_candidates():
                if c.status in SHADOW_ALLOWED_STATUSES and c.enabled:
                    loaded.append(c)

        self.candidates = loaded
        for c in self.candidates:
            self._active[c.candidate_id] = {
                "open": False,
                "side": None,
                "entry_price": None,
                "virtual_pnl": 0.0,
            }
        print(f"[SHADOW-ENGINE] Loaded {len(self.candidates)} shadow-eligible candidates (no broker orders will ever be sent)")

    def _rotate_logs_if_needed(self) -> None:
        d = _today()
        if d != self._current_date:
            self._current_date = d
            self._sig_path = self.log_dir / f"shadow_signals_{d}.jsonl"
            self._trade_path = self.log_dir / f"shadow_virtual_trades_{d}.jsonl"
            self._block_path = self.log_dir / f"shadow_blocks_{d}.jsonl"

    def _build_would_order_payload(
        self,
        cand: Candidate,
        decision: Dict[str, Any],
        snapshot: Dict[str, Any],
        option: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a dry-run broker order payload (never sent)."""
        # Use the same fields the live path would use.
        side = decision.get("side") or decision.get("side_decision") or "BUY"
        # Try to extract a concrete option from the router decision or the passed option snapshot
        sym = option.get("symbol") or option.get("tradingsymbol") or snapshot.get("symbol") or "NIFTY"
        token = option.get("token") or option.get("instrument_token") or snapshot.get("token")
        exch = option.get("exchange") or "NFO"
        qty = int(cand.max_qty_lots or 1) * 65  # conservative; real lot size resolved later
        ltp = float(option.get("ltp") or option.get("last_price") or 0.0)
        return {
            "symbol": sym,
            "exchange": exch,
            "symbol_token": str(token) if token else "",
            "side": str(side).upper(),
            "quantity": qty,
            "order_type": "LIMIT",  # first-live rule: LIMIT
            "price": round(ltp, 2) if ltp else None,
            "product_type": "INTRADAY",
            "ordertag": f"shadow:{cand.candidate_id[:16]}",
            "candidate_id": cand.candidate_id,
            "live_order_sent": False,
            "dry_run": True,
        }

    def on_market_snapshot(
        self,
        market_snapshot: Dict[str, Any],
        option_chain_snapshot: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main entry. Feed identical real snapshot to every shadow-eligible candidate.
        Returns list of decision records (one per candidate).
        """
        self._rotate_logs_if_needed()
        if get_kill_switch_active():
            # Log a global block
            rec = {
                "timestamp": _iso_now(),
                "event": "kill_switch_active",
                "live_order_sent": False,
            }
            _append_jsonl(self._block_path, rec)
            return []

        results: List[Dict[str, Any]] = []
        chain = option_chain_snapshot or []

        with self._lock:
            for cand in self.candidates:
                cid = cand.candidate_id
                state = self._active.get(cid, {"open": False})

                # Pick a representative option from the chain for this candidate (router will filter)
                # If chain empty, still let router decide (it may SKIP).
                rep_option: Dict[str, Any] = {}
                if chain:
                    # Prefer the first; router + filters will reject unsuitable strikes/expiries.
                    rep_option = dict(chain[0])

                try:
                    dec = route_candidate_decision(
                        market_snapshot=dict(market_snapshot),
                        option_chain_snapshot=chain if chain else None,
                        legacy_signal=None,
                        mode="shadow",
                        active_candidate_id=cid,
                    ) or {}
                except Exception as e:
                    dec = {"decision": "SKIP", "reason": f"router_exception:{e}", "no_order_sent": True}

                # Normalize common fields the router returns
                router_reason = dec.get("reason") or dec.get("router_reason") or dec.get("no_trade_reason") or ""
                signal_score = float(dec.get("probability") or dec.get("score") or dec.get("edge_score") or 0.0)
                final_sig = dec.get("decision") or dec.get("final_signal") or "SKIP"
                risk_allowed = bool(dec.get("risk_allowed", True)) and not dec.get("blocked", False)

                # Extract concrete leg info if router provided one
                leg = dec.get("selected_leg") or dec.get("option") or rep_option or {}
                option_type = leg.get("option_type") or leg.get("type") or "PE"
                strike = leg.get("strike") or leg.get("strike_price") or 0
                expiry = leg.get("expiry") or leg.get("expiry_date") or ""
                ltp = float(leg.get("ltp") or leg.get("last_price") or leg.get("price") or 0.0)
                bid = float(leg.get("bid") or leg.get("best_bid") or 0.0)
                ask = float(leg.get("ask") or leg.get("best_ask") or 0.0)
                spread_pct = float(leg.get("spread_pct") or (abs(ask - bid) / max(ltp, 1e-6) if ltp else 0.0))

                blocked_reason = ""
                if final_sig in ("SKIP", "NO_TRADE", "BLOCKED") or not risk_allowed:
                    blocked_reason = router_reason or dec.get("blocked_reason") or "router_skip"

                # Virtual execution (very simple one-position model per candidate for shadow)
                virtual_entry = state.get("entry_price")
                virtual_exit = None
                virtual_pnl = state.get("virtual_pnl", 0.0)
                side = dec.get("side") or dec.get("side_decision") or leg.get("side") or "SELL"

                if final_sig in ("ENTER", "PAPER_TRADE", "TRADE", "SHADOW_TRADE") and not state.get("open"):
                    # Virtual entry
                    state["open"] = True
                    state["side"] = side
                    state["entry_price"] = ltp or ask or bid
                    virtual_entry = state["entry_price"]
                    virtual_pnl = 0.0
                elif state.get("open") and final_sig in ("EXIT", "SQUARE_OFF", "TARGET", "STOP"):
                    # Virtual exit
                    exit_px = ltp or bid or ask
                    entry_px = state["entry_price"] or exit_px
                    # Crude side-aware PnL (1 lot notionally)
                    mult = 1 if state.get("side") in ("SELL", "SHORT") else -1
                    virtual_pnl = mult * (exit_px - entry_px)
                    state["virtual_pnl"] = virtual_pnl
                    virtual_exit = exit_px
                    state["open"] = False

                would_payload = self._build_would_order_payload(cand, dec, market_snapshot, leg or rep_option)

                rec = {
                    "timestamp": _iso_now(),
                    "candidate_id": cid,
                    "option_type": option_type,
                    "strike": strike,
                    "expiry": expiry,
                    "side": side,
                    "ltp": ltp,
                    "bid": bid,
                    "ask": ask,
                    "spread_pct": round(spread_pct, 6),
                    "signal_score": round(signal_score, 6),
                    "router_reason": router_reason,
                    "risk_allowed": risk_allowed,
                    "blocked_reason": blocked_reason,
                    "virtual_entry_price": virtual_entry,
                    "virtual_exit_price": virtual_exit,
                    "virtual_pnl": round(virtual_pnl, 2),
                    "would_order_payload": would_payload,
                    "live_order_sent": False,
                }

                # Always log signal
                _append_jsonl(self._sig_path, rec)

                if blocked_reason:
                    _append_jsonl(self._block_path, rec)
                else:
                    # Log a virtual trade event when we have entry or exit activity
                    if virtual_entry is not None or virtual_exit is not None:
                        _append_jsonl(self._trade_path, rec)

                results.append(rec)

        return results

    def get_status_summary(self) -> List[Dict[str, Any]]:
        out = []
        for c in self.candidates:
            st = self._active.get(c.candidate_id, {})
            out.append({
                "candidate_id": c.candidate_id,
                "status": c.status,
                "open": st.get("open", False),
                "virtual_pnl": st.get("virtual_pnl", 0.0),
            })
        return out


# Convenience for direct script runs / tests
if __name__ == "__main__":
    eng = ShadowEngine(candidate_file=str(CONFIG_DIR / "shadow_candidates.json"))
    print("ShadowEngine ready. Feed it snapshots via on_market_snapshot().")
    print("Loaded candidates:", [c.candidate_id for c in eng.candidates])
