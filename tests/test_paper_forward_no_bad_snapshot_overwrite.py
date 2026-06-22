import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI
from paper_forward_engine import PaperForwardRuntime

def test_no_overwrite_after_good():
    ui = object.__new__(ScalperUI)
    ui._pf_runtime = PaperForwardRuntime()
    ui.pf_engine = type('E',(), {
        'on_market_snapshot': lambda s,c: setattr(ui.pf_engine, 'call_count', getattr(ui.pf_engine,'call_count',0)+1 ),
        'get_status_table': lambda s: []
    })()
    ui._pf_refresh_table = lambda r: None
    ui._pf_set_table_auth_block_reason = lambda x: None
    ui._pf_auth_trace = lambda *a,**k: None
    ui._pf_runtime.auth.status = "AUTH_OK"
    ui._pf_runtime.last_good_snapshot_rows = 5310
    good_snap = {"broker_auth":"AUTH_OK", "spot":23172.5, "option_rows":5310, "candles":[1]*100}
    ui._publish_market_snapshot_to_paper_forward(good_snap, [{}]*5310, "poller")
    calls_before = getattr(ui.pf_engine, 'call_count', 0)
    bad = {"broker_auth":"TOKEN_NOT_VERIFIED", "spot":23172.5, "option_rows":0, "candles":[] , "source":"ui_or_engine"}
    ui._publish_market_snapshot_to_paper_forward(bad, [], "ui_or_engine")
    calls_after = getattr(ui.pf_engine, 'call_count', 0)
    assert calls_after == calls_before  # no additional call for bad
    assert ui._pf_runtime.last_good_snapshot_rows == 5310
