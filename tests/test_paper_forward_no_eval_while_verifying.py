import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


class Engine:
    def __init__(self):
        self.calls = 0

    def on_market_snapshot(self, snap, chain):
        self.calls += 1

    def get_status_table(self):
        return []


def test_pending_auth_snapshot_not_republished_repeatedly():
    ui = object.__new__(ScalperUI)
    ui.pf_engine = Engine()
    ui._pf_pending_auth_snapshot_published = False
    ui._pf_refresh_table = lambda rows: None

    snap = {
        "broker_auth": "TOKEN_PRESENT_UNCHECKED",
        "auth_validation_attempted": False,
        "spot": 23257.95,
    }
    ui._publish_market_snapshot_to_paper_forward(dict(snap), [])
    ui._publish_market_snapshot_to_paper_forward(dict(snap), [])

    assert ui.pf_engine.calls == 1
