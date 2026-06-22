import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import LOAD_OK, PaperForwardEngine


def _engine():
    eng = PaperForwardEngine(candidate_file=str(ROOT / "config" / "paper_forward_candidates.json"), artifacts_dir=str(ROOT), broker_safe_mode=True)
    eng.candidates = [{
        "candidate_id": "c1",
        "enabled": True,
        "_load_status": LOAD_OK,
        "artifact_dir": str(ROOT),
        "_feature_list": [],
    }]
    return eng


def test_token_not_verified_blocks_prediction():
    eng = _engine()
    decisions = eng.on_market_snapshot({"spot": 23172.5, "broker_auth": "TOKEN_SET_NOT_VERIFIED", "token_set": True}, [])

    assert decisions[0]["no_trade_reason"] == "token_not_verified"
    assert decisions[0]["predict_attempted"] is False
    assert decisions[0]["confidence"] is None
    assert eng.get_diagnostics()["predictions_attempted"] == 0


def test_rows_zero_blocks_with_option_chain_empty_when_auth_ok():
    eng = _engine()
    decisions = eng.on_market_snapshot({"spot": 23172.5, "broker_auth": "AUTH_OK"}, [])

    assert decisions[0]["no_trade_reason"] == "option_chain_empty"
    assert decisions[0]["predict_attempted"] is False


def test_rows_one_blocks_with_thin_option_chain():
    eng = _engine()
    chain = [{"strike": 23200, "option_type": "CE", "ltp": 100, "volume": 10, "oi": 20, "spot": 23172.5}]
    decisions = eng.on_market_snapshot({"spot": 23172.5, "broker_auth": "AUTH_OK", "option_chain": chain}, chain)

    assert decisions[0]["no_trade_reason"] == "thin_option_chain"
    assert decisions[0]["predict_attempted"] is False
