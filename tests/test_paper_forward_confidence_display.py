"""Confidence display."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_conf_none_becomes_dash_in_ui_logic():
    conf = None
    predict = False
    conf_str = "-" if (conf is None or (conf == 0.0 and not predict)) else f"{float(conf):.3f}"
    assert conf_str == "-"


def test_status_table_preserves_tiny_nonzero_confidence():
    from paper_forward_engine import PaperForwardEngine

    eng = PaperForwardEngine.__new__(PaperForwardEngine)
    eng.candidates = [{"candidate_id": "c1", "enabled": True, "_load_status": "OK"}]
    eng._state = {
        "c1": {
            "confidence": 9.68e-8,
            "predict_attempted": True,
            "last_signal": "NO_TRADE",
            "last_no_trade_reason": "low_confidence_9.68e-08_lt_0.3500",
        }
    }
    eng._candidate_states = {}
    eng._last_data_status = None
    eng._latest_market_snapshot = {}
    eng._latest_option_chain_snapshot = []
    eng.artifacts_dir = "artifacts/candidates"

    row = eng.get_status_table()[0]

    assert row["confidence"] == 9.68e-8
    assert row["confidence_display"] == "9.68e-08"
