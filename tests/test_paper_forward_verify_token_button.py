import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_verify_token_button_method_exists_and_forces_recheck(monkeypatch):
    ui = object.__new__(ScalperUI)
    called = {"force": None}

    def fake_ensure(force=False):
        called["force"] = force
        return "AUTH_OK", "", 23172.5

    monkeypatch.setattr(ui, "_pf_ensure_auth_terminal", fake_ensure)
    monkeypatch.setattr(ui, "_pf_auth_status_fields", lambda: {
        "auth_validation_attempted": True,
        "auth_validation_in_progress": False,
        "auth_validation_endpoint": "mock",
        "auth_success_count": 1,
        "auth_failure_count": 0,
    })
    monkeypatch.setattr(ui, "_pf_apply_data_status_to_ui", lambda status: None)
    monkeypatch.setattr(ui, "_trigger_paper_forward_data_poll_once", lambda: None)
    monkeypatch.setattr(ui, "after", lambda _delay, fn=None: fn() if fn else None)
    monkeypatch.setattr("ui.messagebox.showinfo", lambda *args, **kwargs: None)

    # Run worker path synchronously by replacing Thread with a tiny stub.
    class T:
        def __init__(self, target=None, daemon=None):
            self.target = target
        def start(self):
            self.target()

    monkeypatch.setattr("ui.threading.Thread", T)
    ui._pf_verify_token_button()

    assert called["force"] is True
