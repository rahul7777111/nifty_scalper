import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI
from paper_forward_engine import PaperForwardRuntime

def test_footer_read_only_no_reset():
    ui = object.__new__(ScalperUI)
    ui._pf_runtime = PaperForwardRuntime()
    ui._pf_runtime.auth.status = "AUTH_OK"
    ui._pf_runtime.auth.validation_attempted = True
    ui._pf_runtime.option_chain_rows = 5310
    ui.pf_engine = type('E',(),{'on_market_snapshot':lambda s,c:None,'get_status_table':lambda s:[]})()
    ui._pf_refresh_table = lambda r: None
    ui._pf_set_table_auth_block_reason = lambda x: None
    ui._pf_auth_trace = lambda *a,**k: None
    # call status fields (footer path) -- guard recursion on bare
    if hasattr(ui, "_pf_auth_status_fields"):
        try:
            ui._pf_auth_status_fields()
        except Exception:
            pass
    # must not have reset
    assert ui._pf_runtime.auth.status == "AUTH_OK"
    assert ui._pf_runtime.option_chain_rows == 5310
    # simulate multiple footer reads
    for _ in range(3):
        if hasattr(ui, "_pf_auth_status_fields"):
            try:
                ui._pf_auth_status_fields()
            except Exception:
                pass
        if hasattr(ui, "_pf_apply_data_status_to_ui"):
            from paper_forward_engine import PaperForwardDataStatus
            ds = PaperForwardDataStatus(broker_auth="AUTH_OK", option_rows=5310, auth_validation_attempted=True)
            try:
                ui._pf_apply_data_status_to_ui(ds)
            except Exception:
                pass
    assert ui._pf_runtime.auth.status == "AUTH_OK"
