import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_forward_engine import PaperForwardDataStatus


def _snap():
    return {
        "broker_auth": "AUTH_OK",
        "price": 24150,
        "candle_source": "mstock_historical",
        "candles": [{"open": 1, "high": 2, "low": 1, "close": 2, "volume": 10}] * 100,
    }


def test_single_row_chain_is_thin_option_chain():
    status = PaperForwardDataStatus.from_snapshot(_snap(), [{"strike": 24150, "option_type": "CE", "ltp": 10, "spot": 24150}])
    d = status.as_dict()
    assert d["option_chain_rows"] == 1
    assert d["option_chain_status"] == "THIN_OPTION_CHAIN"
    assert d["data_quality_status"] == "THIN_OPTION_CHAIN"


def test_full_chain_rows_are_data_ok_when_spot_and_candles_present():
    chain = [{"strike": 23000 + i * 50, "option_type": "CE" if i % 2 == 0 else "PE", "ltp": 10, "spot": 24150} for i in range(24)]
    status = PaperForwardDataStatus.from_snapshot(_snap(), chain)
    d = status.as_dict()
    assert d["option_chain_rows"] == 24
    assert d["option_chain_status"] == "DATA_OK"
    assert d["data_quality_status"] == "DATA_OK"
