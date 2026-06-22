from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reconciliation import OrderLifecycleState, OrderStateMachine, ReconciliationEngine


def test_order_state_machine_clean_cancel_replace_transition() -> None:
    fsm = OrderStateMachine(symbol="NIFTYTEST", side="BUY", quantity=10)
    fsm.mark_submitted("ORD1")
    assert fsm.state == OrderLifecycleState.LIVE_ON_BOOK
    fsm.request_cancel()
    assert fsm.state == OrderLifecycleState.PENDING_CANCEL
    fsm.confirm_cancel()
    assert fsm.state == OrderLifecycleState.CANCEL_CONFIRMED
    assert fsm.reconciliation_status_code == "CANCEL_CONFIRMED"


def test_order_state_machine_late_fill_abort_transition() -> None:
    fsm = OrderStateMachine(symbol="NIFTYTEST", side="BUY", quantity=10)
    fsm.mark_submitted("ORD2")
    fsm.request_cancel()
    fsm.mark_fill(fill_qty=10, fill_price=100.25, unsolicited=True)
    assert fsm.state == OrderLifecycleState.FULLY_FILLED
    assert fsm.unsolicited_fill_occurred is True
    assert fsm.reconciliation_status_code == "LATE_FILL_ABORT"


def test_reconciliation_engine_registers_and_binds_order() -> None:
    engine = ReconciliationEngine()
    fsm = engine.register_order(symbol="NIFTYTEST", side="BUY", quantity=5)
    engine.bind_order_id(fsm, "ORD3")
    assert fsm.order_id == "ORD3"
    assert fsm.state == OrderLifecycleState.LIVE_ON_BOOK
