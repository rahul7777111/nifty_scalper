import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import PaperForwardEngine

def test_export_uses_runtime_values():
    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng.candidates = [{"candidate_id": "c1"}]
    eng._state = {"c1": {"unrealized_pnl": 1.23, "total_trades": 5, "win_rate": 0.6}}
    eng._candidate_states = {}
    eng.reports_dir = Path("reports")
    eng.reports_dir.mkdir(exist_ok=True)
    # populate cs
    from paper_forward_engine import PaperForwardCandidateRuntimeState
    cs = PaperForwardCandidateRuntimeState("c1")
    cs.unreal_pnl = 1.23
    cs.trades = 5
    cs.win_rate = 0.6
    cs.final_signal = "NO_TRADE"
    eng._candidate_states["c1"] = cs
    summ = eng.write_summary()
    cands = summ.get("candidates", [])
    assert len(cands) > 0
    assert cands[0].get("unrealized_pnl") == 1.23 or abs(cands[0].get("unrealized_pnl",0)-1.23)<0.1
    print("[TEST] export summary runtime ok")