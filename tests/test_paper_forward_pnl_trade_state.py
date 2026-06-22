import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import PaperForwardEngine

def test_pnl_trade_update_on_sim_entry_exit():
    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng._state = {}
    eng._candidate_states = {}
    eng.candidates = [{"candidate_id": "c1", "side_policy": "CE"}]
    # minimal init
    cs = type('cs',(),{'pos':'FLAT','qty':0,'lot_size':65,'lots':1,'direction':'BUY','entry_price':0,'unreal_pnl':0,'realized_pnl':0,'trades':0,'wins':0,'losses':0,'win_rate':0,'max_drawdown':0,'last_action':'','side':'CE','candidate_id':'c1'})()
    eng._candidate_states['c1'] = cs
    eng._state['c1'] = {'open_position':False}
    # simulate enter via helper
    eng._pf_open_paper_position('c1', 'CE', 100.0, 't1')
    assert cs.pos == 'LONG_CE' or cs.pos == 'CE'
    assert cs.unreal_pnl == 0
    # update pnl
    eng._pf_update_paper_position_pnl('c1', 101.0)
    assert abs(cs.unreal_pnl - 65.0) < 0.01
    # close
    eng._pf_close_paper_position('c1', 101.5, 't2')
    assert cs.pos == 'FLAT'
    assert cs.trades == 1
    assert cs.realized_pnl > 0
    print("[TEST] pnl trade state update ok")
