import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import LOAD_OK, PaperForwardEngine


def test_prediction_audit_marks_blocked_predictions(capsys):
    eng = PaperForwardEngine(candidate_file=str(ROOT / "config" / "paper_forward_candidates.json"), artifacts_dir=str(ROOT), broker_safe_mode=True)
    eng.candidates = [{
        "candidate_id": "c1",
        "enabled": True,
        "_load_status": LOAD_OK,
        "artifact_dir": str(ROOT),
        "_feature_list": [],
    }]

    eng.on_market_snapshot({"spot": 23172.5, "broker_auth": "TOKEN_SET_NOT_VERIFIED", "token_set": True}, [])
    out = capsys.readouterr().out

    assert "[PAPER-FWD-PREDICT]" in out
    assert "predict_allowed=false" in out
    assert "raw_output=None" in out
    assert "confidence=None" in out
