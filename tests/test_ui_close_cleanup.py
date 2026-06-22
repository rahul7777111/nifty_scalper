from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ui


class _Closer:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _DummyEvent:
    def __init__(self) -> None:
        self.calls = 0

    def set(self) -> None:
        self.calls += 1


def test_on_close_closes_db_dependencies_once():
    app = object.__new__(ui.ScalperUI)
    ui_db = _Closer()
    strat_db = _Closer()
    app._db_manager = ui_db
    app._scalper = type("ScalperStub", (), {"db_manager": strat_db})()
    app._bot_stop = _DummyEvent()
    app._ui_log_fp = None
    app._orig_stdout = sys.stdout
    app._orig_stderr = sys.stderr
    app.destroy = lambda: None

    ui.ScalperUI._on_close(app)

    assert app._bot_stop.calls == 1
    assert ui_db.closed == 1
    assert strat_db.closed == 1
