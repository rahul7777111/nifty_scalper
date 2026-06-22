import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import build_paper_forward_feature_frame


def test_feature_builder_derives_common_live_fields_and_missing_exactly():
    candidate = {"_feature_list": ["strike_price", "dte_days", "option_type_ce", "return_1", "missing_x"]}
    snap = {
        "price": 24150,
        "timestamp": "2026-06-11T09:20:00+05:30",
        "candles": [
            {"open": 100 + i, "high": 103 + i, "low": 99 + i, "close": 101 + i, "volume": 1000 + i}
            for i in range(20)
        ],
        "option_chain": [{"strike": 24150, "option_type": "CE", "ltp": 10, "bid": 9.5, "ask": 10.5, "expiry": "2026-06-25", "spot": 24150}],
    }

    frame, missing, debug = build_paper_forward_feature_frame(candidate, snap)

    assert "strike_price" in frame.columns
    assert "dte_days" in frame.columns
    assert "option_type_ce" in frame.columns
    assert "return_1" in frame.columns
    assert missing == ["missing_x"]
    assert debug["feature_order"] == candidate["_feature_list"]


def test_required_feature_order_matches_training_list():
    feature_list = ["b", "a", "c"]
    _, _, debug = build_paper_forward_feature_frame({"_feature_list": feature_list}, {"price": 1, "option_chain": []})
    assert debug["feature_order"] == feature_list
