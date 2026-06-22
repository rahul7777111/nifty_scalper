import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI
from paper_forward_engine import PaperForwardRuntime

def test_live_chart_merges_not_replaces():
    ui = object.__new__(ScalperUI)
    ui._pf_runtime = PaperForwardRuntime()
    ui.pf_engine = type('E',(),{'on_market_snapshot':lambda s,c:None, 'get_status_table':lambda s:[]})()
    ui._pf_refresh_table = lambda r: None
    ui._pf_set_table_auth_block_reason = lambda x: None
    ui._pf_auth_trace = lambda *a,**k: None
    ui._pf_runtime.auth.status = "AUTH_OK"
    ui._pf_runtime.option_chain_rows = 5310
    ui._pf_runtime.last_good_snapshot_rows = 5310
    ui._pf_runtime.last_good_snapshot_candles = 100
    ui._pf_runtime.update_from_good_snapshot({"broker_auth":"AUTH_OK", "spot":23172.5, "option_rows":5310, "candles":[1]*100}, "poller")
    # simulate live chart candles only update via publish (arbitration should keep rows)
    lc = {"broker_auth": "AUTH_OK", "spot": 23172.5, "option_rows": 0, "candles": [2]*50, "candle_count":50, "source": "live_chart_candles_only_update"}
    ui._publish_market_snapshot_to_paper_forward(lc, [], "live_chart_candles_only_update")
    assert ui._pf_runtime.option_chain_rows == 5310  # preserved
    assert ui._pf_runtime.candle_count >= 50  # merged/updated candles
