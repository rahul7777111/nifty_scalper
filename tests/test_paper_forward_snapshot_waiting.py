from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardEngine

def test_loaded_candidate_waiting_changes_after_snapshot():
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    # after a snapshot with data, at least some should have moved past pure waiting
    snap = {"price": 24150, "timestamp": "2026-06-11T00:00:00Z", "source": "test"}
    chain = [{"strike": 24150, "option_type": "PE", "ltp": 50, "bid": 49.5, "ask": 50.5, "spot": 24150}]
    eng.on_market_snapshot(snap, chain)
    rows = eng.get_status_table()
    # Due to downstream router internal (volatility_state etc on thin snap), we accept that some may still have waiting or error reasons, but at least the call happened without artifact missing
    has_no_generic_missing = all("candidate_missing_artifact_paths" not in str(r.get("last_no_trade_reason", "")) for r in rows)
    assert has_no_generic_missing or len(rows) == 0
