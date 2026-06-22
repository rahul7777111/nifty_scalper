import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardDataStatus


def test_header_footer_fields_come_from_same_status_object():
    status = PaperForwardDataStatus.from_snapshot(
        {
            "broker_name": "mstock",
            "broker_auth": "AUTH_OK",
            "price": 24150.5,
            "timestamp": "2026-06-11T09:20:00+05:30",
            "candle_source": "mstock_historical",
            "candles": [{"open": 1, "high": 2, "low": 1, "close": 2, "volume": 10}] * 100,
        },
        [{"strike": 24000 + i * 50, "option_type": "CE", "ltp": 10, "spot": 24150.5} for i in range(24)],
    )
    d = status.as_dict()

    assert d["spot"] == 24150.5
    assert d["spot_status"] == "spot_ok"
    assert d["candle_count"] == 100
    assert d["option_chain_rows"] == 24
    assert d["data_quality_status"] == "DATA_OK"
    footer = status.footer_text()
    assert "AUTH_OK" in footer
    assert "candles=100" in footer
    assert "rows=24" in footer
