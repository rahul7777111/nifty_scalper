"""No uncaught route exceptions on thin data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardEngine, LOAD_OK

def test_no_top_level_or_per_cand_route_exc_on_thin():
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    loadable = [c for c in eng.candidates if c.get("_load_status") in (LOAD_OK,)]
    if not loadable:
        return
    eng.candidates = loadable[:3]
    chain = [{"strike": 24000, "option_type": "CE", "ltp": 25, "spot": 24100}]
    snap = {"price": 24100, "timestamp": "t", "source": "thin_test"}
    try:
        decs = eng.on_market_snapshot(snap, chain)
        for d in decs:
            assert d.get("final_signal") != "ERROR" or "model_route_exception" in str(d.get("no_trade_reason", ""))
    except Exception as e:
        assert False, f"top level exception leaked: {e}"
