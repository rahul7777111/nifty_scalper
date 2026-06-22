import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


class _Engine:
    def __init__(self):
        self.calls = 0
    def on_market_snapshot(self, snap, chain):
        self.calls += 1
    def get_status_table(self):
        return []


def test_terminal_auth_failure_snapshot_published_once():
    ui = object.__new__(ScalperUI)
    ui.pf_engine = _Engine()
    ui._pf_refresh_table = lambda rows: None

    snap = {
        "broker_auth": "SESSION_EXPIRED:IA401",
        "auth_error": "IA401",
        "auth_validation_attempted": True,
        "token_hash_prefix": "abc",
    }
    ui._publish_market_snapshot_to_paper_forward(dict(snap), [])
    ui._publish_market_snapshot_to_paper_forward(dict(snap), [])

    assert ui.pf_engine.calls == 1
