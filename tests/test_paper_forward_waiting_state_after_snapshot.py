import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import LOAD_OK, PaperForwardEngine


def _chain(rows=20):
    return [{"strike": 24000 + i * 50, "option_type": "CE" if i % 2 == 0 else "PE", "ltp": 10, "spot": 24150} for i in range(rows)]


def test_snapshot_replaces_waiting_for_snapshot_reason():
    art = Path("tests") / "_tmp_paper_forward_waiting" / "c1"
    art.mkdir(parents=True, exist_ok=True)
    eng = PaperForwardEngine(broker_safe_mode=True)
    eng.candidates = [
        {
            "candidate_id": "c1",
            "enabled": True,
            "artifact_dir": str(art),
            "_load_status": LOAD_OK,
            "_feature_list": ["missing_required_feature"],
            "model_name": "unit_model",
            "preset_family": "unit_preset",
        }
    ]
    eng._state = {"c1": {"last_no_trade_reason": "", "confidence": None, "predict_attempted": False}}

    before = eng.get_status_table()[0]["last_no_trade_reason"]
    decisions = eng.on_market_snapshot({"price": 24150, "timestamp": "t"}, _chain())
    after = eng.get_status_table()[0]["last_no_trade_reason"]

    assert before == "candidate_loaded_ok_waiting_for_snapshot"
    assert len(decisions) == 1
    assert "waiting_for_snapshot" not in after
    assert after == "feature_vector_missing_columns"
