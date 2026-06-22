"""Feature contract tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardEngine

def _chain(rows=20):
    return [{"strike": 24000 + i * 50, "option_type": "CE" if i % 2 == 0 else "PE", "ltp": 10, "spot": 24150} for i in range(rows)]

def test_feature_check_skips_model_on_missing():
    eng = PaperForwardEngine(candidate_file="config/paper_forward_candidates.json", broker_safe_mode=True)
    for c in eng.candidates:
        c["_feature_list"] = ["nonexistent_feature_xyz_123"]
    eng.candidates = eng.candidates[:2]
    decs = eng.on_market_snapshot({"price": 24150, "timestamp": "t"}, _chain())
    for d in decs:
        assert d.get("predict_attempted") is False
        assert "feature" in str(d.get("no_trade_reason", "")).lower() or d.get("final_signal") == "NO_TRADE"
