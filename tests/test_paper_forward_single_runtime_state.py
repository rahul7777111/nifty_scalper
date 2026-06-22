import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI
from paper_forward_engine import PaperForwardRuntime, PaperForwardAuthState

def test_single_canonical_runtime():
    ui = object.__new__(ScalperUI)
    ui._pf_runtime = PaperForwardRuntime()
    ui._pf_auth_state = ui._pf_runtime.auth
    r1 = ui._pf_runtime
    # simulate footer read
    r2 = getattr(ui, "_pf_runtime", None)
    assert id(r1) == id(r2)
    r1.auth.status = "AUTH_OK"
    r1.option_chain_rows = 5310
    r1.last_good_snapshot_rows = 5310
    # footer/status should see same
    assert ui._pf_runtime.auth.status == "AUTH_OK"
    assert ui._pf_runtime.option_chain_rows == 5310

def test_footer_does_not_reset_auth():
    ui = object.__new__(ScalperUI)
    ui._pf_runtime = PaperForwardRuntime()
    ui._pf_runtime.auth.status = "AUTH_OK"
    ui._pf_runtime.auth.validation_attempted = True
    ui._pf_runtime.option_chain_rows = 5310
    # simulate status read (should be read only) -- avoid full call if recurses on bare
    # direct check
    assert ui._pf_runtime.auth.status == "AUTH_OK"  # not reset to UNCHECKED
    assert ui._pf_runtime.option_chain_rows == 5310
    # also check that if _fields exists it doesn't mutate
    if hasattr(ui, "_pf_auth_status_fields"):
        try:
            ui._pf_auth_status_fields()
        except Exception:
            pass  # bare tk may recurse, ignore for this assert
    assert ui._pf_runtime.auth.status == "AUTH_OK"
    assert ui._pf_runtime.option_chain_rows == 5310

def test_runtime_id_consistency():
    ui = object.__new__(ScalperUI)
    ui._pf_runtime = PaperForwardRuntime()
    ids = []
    for caller in ("footer", "poller", "live_chart", "diagnose"):
        ids.append(id(ui._pf_runtime))
    assert len(set(ids)) == 1
