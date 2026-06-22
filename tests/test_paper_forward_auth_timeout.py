import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_verifying_timeout_becomes_auth_failed(monkeypatch):
    monkeypatch.setenv("PAPER_FORWARD_AUTH_TIMEOUT_SEC", "10")
    ui = object.__new__(ScalperUI)
    ui._pf_auth_validation_in_progress = True
    ui._pf_auth_validation_attempted = False
    ui._pf_auth_validation_started_ts = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()

    ui._pf_enforce_auth_timeout()

    assert ui._pf_auth_validation_in_progress is False
    assert ui._pf_auth_validation_attempted is True
    assert ui._pf_last_auth_status == "AUTH_FAILED:auth_validation_timeout"
    assert ui._pf_last_auth_error == "auth_validation_timeout"
