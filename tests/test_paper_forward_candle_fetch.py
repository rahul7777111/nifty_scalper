import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardDataStatus


def test_candle_count_positive_never_reports_source_none():
    status = PaperForwardDataStatus.from_snapshot(
        {"broker_auth": "AUTH_OK", "price": 24150, "source": "none", "candles": [{"close": 1}] * 100},
        [{"strike": 24000 + i * 50, "option_type": "CE", "ltp": 10, "spot": 24150} for i in range(24)],
    )
    d = status.as_dict()
    assert d["candle_count"] == 100
    assert d["candle_source"] != "none"
