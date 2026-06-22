import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import LOAD_OK, PaperForwardEngine


def _chain(rows=20):
    return [{"strike": 24000 + i * 50, "option_type": "CE" if i % 2 == 0 else "PE", "ltp": 10, "spot": 24150} for i in range(rows)]


def test_candidates_evaluated_increments_when_prediction_skipped():
    art = Path("tests") / "_tmp_paper_forward_eval" / "c1"
    art.mkdir(parents=True, exist_ok=True)
    eng = PaperForwardEngine(broker_safe_mode=True)
    eng.candidates = [
        {
            "candidate_id": "c1",
            "enabled": True,
            "artifact_dir": str(art),
            "_load_status": LOAD_OK,
            "_feature_list": ["missing_a", "missing_b"],
            "model_name": "unit_model",
            "preset_family": "unit_preset",
        }
    ]
    eng._state = {"c1": {"last_no_trade_reason": "", "confidence": None, "predict_attempted": False}}

    decisions = eng.on_market_snapshot({"price": 24150, "timestamp": "t"}, _chain())
    diag = eng.get_diagnostics()

    assert len(decisions) == 1
    assert diag["snapshots_received"] == 1
    assert diag["evaluations_count"] == 1
    assert diag["predictions_attempted"] == 0
    assert diag["predictions_success"] == 0
    assert diag["skipped_missing_features"] == 1
    assert diag["route_errors"] == 0
