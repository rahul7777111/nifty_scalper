from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ui


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


def test_gui_trade_update_uses_queue_handoff():
    app = object.__new__(ui.ScalperUI)
    app._trade_q = _Queue()
    evt = object()
    ui.ScalperUI._queue_trade_event_from_worker(app, evt)
    assert app._trade_q.items == [evt]


def test_gui_tick_update_uses_safe_after():
    app = object.__new__(ui.ScalperUI)
    calls = []
    app._safe_after = lambda delay, fn: calls.append((delay, fn))
    fn = lambda: None
    ui.ScalperUI._schedule_tick_update_from_worker(app, fn)
    assert len(calls) == 1
    assert calls[0][0] == 0
    assert calls[0][1] is fn
