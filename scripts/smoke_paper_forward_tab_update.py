#!/usr/bin/env python3
"""
Headless smoke for the Paper Forward tab update path.

This does not start Tk mainloop, does not call a broker, and does not place orders.
It verifies that decision attributes produced by the paper engine are merged into
the tab rows and rendered into the Treeview values.
"""
from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ui import ScalperUI  # noqa: E402


class FakeVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class FakeTree:
    def __init__(self) -> None:
        self.items: dict[str, tuple] = {}
        self.update_calls = 0

    def exists(self, iid: str) -> bool:
        return iid in self.items

    def item(self, iid: str, **kwargs):
        if "values" in kwargs:
            self.items[iid] = tuple(kwargs["values"])
        return {"values": self.items.get(iid)}

    def insert(self, _parent, _index, iid: str, values):
        self.items[iid] = tuple(values)
        return iid

    def get_children(self):
        return list(self.items)

    def delete(self, iid: str) -> None:
        self.items.pop(iid, None)

    def update_idletasks(self) -> None:
        self.update_calls += 1


class FakeEngine:
    def __init__(self) -> None:
        self.candidates = [
            {
                "candidate_id": "cand-1",
                "enabled": True,
                "model_name": "demo_model",
                "preset_family": "demo_preset",
                "side_policy": "BOTH",
                "final_signal": "NO_TRADE",
                "confidence": None,
                "position_status": "FLAT",
                "last_no_trade_reason": "candidate_loaded_ok_waiting_for_snapshot",
                "last_update": "",
            }
        ]
        self.rows = [dict(self.candidates[0])]

    def get_status_table(self):
        return [dict(row) for row in self.rows]

    def get_diagnostics(self):
        return {
            "snapshots_received": 3,
            "evaluations_count": 2,
            "route_errors": 0,
            "data_status": {
                "broker_name": "mstock",
                "broker_auth": "AUTH_OK",
                "candle_source": "smoke",
                "candle_count": 32,
                "spot": 23257.95,
                "option_chain_rows": 5310,
                "option_chain_status": "DATA_OK",
                "option_chain_source": "smoke_rows",
                "option_chain_age_sec": 1.2,
                "last_tick_ts": "2026-06-12T09:20:01+05:30",
                "data_quality_status": "DATA_OK",
            },
        }


def _new_headless_ui() -> ScalperUI:
    ui = object.__new__(ScalperUI)
    ui.pf_engine = FakeEngine()
    ui.pf_candidates = []
    ui.pf_tree = FakeTree()
    ui.pf_live_var = FakeVar()
    ui.pf_status_var = FakeVar()
    ui.pf_broker_status_var = FakeVar()
    ui._dash_spot_var = FakeVar()
    ui._dash_candles_var = FakeVar()
    ui._dash_last_tick_var = FakeVar()
    ui._option_chain_status_var = FakeVar()
    ui._ui_queue = queue.Queue()
    ui._closing = False
    ui._pf_latest_decisions_by_candidate = {}
    ui._pf_poll_cycle_id = 0
    ui._pf_auth_trace = lambda *args, **kwargs: None
    ui._safe_after_app = lambda *args, **kwargs: None
    ui.after = lambda _delay, callback=None, *args: callback(*args) if callback else None
    return ui


def _assert(name: str, condition: bool, detail: str) -> bool:
    if condition:
        print(f"[SMOKE-PF-TAB] OK {name}: {detail}")
        return True
    print(f"[SMOKE-PF-TAB] FAIL {name}: {detail}")
    return False


def main() -> int:
    print("[SMOKE-PF-TAB] Checking Paper Forward tab row update path")
    ui = _new_headless_ui()
    bad = False

    ui._pf_refresh_monitor_ui(source="initial")
    initial = ui.pf_tree.items.get("cand-1")
    bad |= not _assert("initial_row_inserted", initial is not None, str(initial))
    bad |= not _assert("initial_reason", initial[13] == "candidate_loaded_ok_waiting_for_snapshot", str(initial))

    decision = {
        "candidate_id": "cand-1",
        "final_signal": "NO_TRADE",
        "confidence": 0.0,
        "threshold": 0.35,
        "no_trade_reason": "low_confidence_0.0000_lt_0.3500",
        "timestamp": "2026-06-12T09:20:02+05:30",
        "predict_attempted": True,
        "unrealized_pnl": 12.5,
        "realized_pnl": -3.0,
        "total_trades": 4,
        "win_rate": 0.25,
        "position_status": "FLAT",
    }
    ui._pf_remember_decisions([decision])
    ui._pf_refresh_monitor_ui(source="decision_cache")

    updated = ui.pf_tree.items.get("cand-1")
    bad |= not _assert("confidence_rendered", updated[6] == "0.000", str(updated))
    bad |= not _assert("unrealized_rendered", updated[8] == "12.50", str(updated))
    bad |= not _assert("realized_rendered", updated[9] == "-3.00", str(updated))
    bad |= not _assert("trades_rendered", updated[10] == 4, str(updated))
    bad |= not _assert("win_rate_rendered", updated[11] == "25.00%", str(updated))
    bad |= not _assert("reason_rendered", updated[13] == "low_confidence_0.0000_lt_0.3500", str(updated))
    bad |= not _assert("timestamp_rendered", updated[14] == "2026-06-12T09:20:02+05:30", str(updated))
    bad |= not _assert("status_vars_updated", ui.pf_status_var.get() == "RUNNING_EVALUATING", ui.pf_status_var.get())
    bad |= not _assert("option_chain_var_updated", ui._option_chain_status_var.get() == "OK rows=5310", ui._option_chain_status_var.get())
    bad |= not _assert("live_var_counts", "Candidates: 1" in ui.pf_live_var.get() and "evals=2" in ui.pf_live_var.get(), ui.pf_live_var.get())
    bad |= not _assert("gui_update_metadata", ui._pf_last_gui_row_count == 1, str(getattr(ui, "_pf_last_gui_sample", {})))

    worker_decision = dict(decision)
    worker_decision["no_trade_reason"] = "spread_too_wide"
    worker_decision["timestamp"] = "2026-06-12T09:20:04+05:30"
    ui._pf_remember_decisions([worker_decision])
    t = threading.Thread(target=lambda: ui._pf_refresh_monitor_ui(source="worker_thread"))
    t.start()
    t.join(timeout=5)
    ui._drain_ui_queue()
    queued = ui.pf_tree.items.get("cand-1")
    bad |= not _assert("worker_queue_reason_rendered", queued[13] == "spread_too_wide", str(queued))
    bad |= not _assert("tree_updated_in_place", list(ui.pf_tree.items) == ["cand-1"] and ui.pf_tree.update_calls >= 3, str(ui.pf_tree.items))

    schedule_decision = dict(decision)
    schedule_decision["no_trade_reason"] = "liquidity_filter"
    schedule_decision["timestamp"] = "2026-06-12T09:20:06+05:30"
    ui._pf_remember_decisions([schedule_decision])
    ui._pf_schedule_update()
    scheduled = ui.pf_tree.items.get("cand-1")
    bad |= not _assert("schedule_loop_reason_rendered", scheduled[13] == "liquidity_filter", str(scheduled))

    if bad:
        print("[SMOKE-PF-TAB] FAIL")
        return 1
    print("[SMOKE-PF-TAB] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
