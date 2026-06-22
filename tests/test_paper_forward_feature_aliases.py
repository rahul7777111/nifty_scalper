import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import build_paper_forward_feature_frame


def test_full_chain_and_candle_aliases_map_required_fields():
    candidate = {
        "candidate_id": "alias",
        "_feature_list": ["strike_price", "ltp", "volume", "oi", "dte_days", "open", "high", "low", "close"],
    }
    snapshot = {
        "spot": 23172.5,
        "timestamp": "2026-06-11T09:30:00+00:00",
        "option_chain": [{
            "strikePrice": 23200,
            "right": "CALL",
            "lastPrice": 95.5,
            "traded_volume": 12345,
            "openInterest": 67890,
            "expiry": "2026-06-25",
        }],
        "candles": [
            {"open": 23100, "high": 23150, "low": 23090, "close": 23120, "volume": 1000},
            {"open": 23120, "high": 23190, "low": 23110, "close": 23172.5, "volume": 2000},
        ],
    }

    frame, missing, debug = build_paper_forward_feature_frame(candidate, snapshot)

    assert missing == []
    row = frame.iloc[0].to_dict()
    assert row["strike_price"] == 23200
    assert row["ltp"] == 95.5
    assert row["volume"] == 12345
    assert row["oi"] == 67890
    assert row["dte_days"] > 0
    assert row["open"] == 23120
    assert row["high"] == 23190
    assert row["low"] == 23110
    assert row["close"] == 23172.5
    assert debug["feature_order"] == candidate["_feature_list"]


def test_pe_only_candidate_uses_pe_option_row():
    candidate = {
        "candidate_id": "pe-only",
        "side_policy": "PE_ONLY",
        "_feature_list": ["option_type_pe", "ltp", "oi", "volume"],
    }
    snapshot = {
        "spot": 23172.5,
        "timestamp": "2026-06-11T09:30:00+00:00",
        "option_chain": [
            {"strikePrice": 23200, "right": "CALL", "lastPrice": 95.5, "traded_volume": 111, "openInterest": 222},
            {"strikePrice": 23200, "right": "PUT", "lastPrice": 88.5, "traded_volume": 333, "openInterest": 444},
        ],
        "candles": [
            {"open": 23100, "high": 23150, "low": 23090, "close": 23120, "volume": 1000},
            {"open": 23120, "high": 23190, "low": 23110, "close": 23172.5, "volume": 2000},
        ],
    }

    frame, missing, _debug = build_paper_forward_feature_frame(candidate, snapshot)

    row = frame.iloc[0].to_dict()
    assert missing == []
    assert row["option_type_pe"] == 1
    assert row["ltp"] == 88.5
    assert row["oi"] == 444
    assert row["volume"] == 333


def test_live_chain_without_iv_quotes_fills_required_paper_fallbacks():
    required = [
        "spot_vwap",
        "bs_delta",
        "bs_gamma",
        "bs_theta",
        "bs_vega",
        "bid_ask_spread",
        "mid_price",
        "ltp_vs_mid_diff",
        "ltp_vs_mid_diff_pct",
    ]
    candidate = {
        "candidate_id": "paper-fallbacks",
        "side_policy": "CE_ONLY",
        "_feature_list": required,
    }
    snapshot = {
        "spot": 23360.15,
        "timestamp": "2026-06-12T09:55:00+05:30",
        "option_chain": [{
            "strikePrice": 23400,
            "right": "CALL",
            "lastPrice": 82.25,
            "traded_volume": 0,
            "openInterest": 5310,
            "expiry": "2026-06-25",
        }],
        "candles": [
            {"open": 23320 + i, "high": 23345 + i, "low": 23305 + i, "close": 23330 + i, "volume": 0}
            for i in range(20)
        ],
    }

    frame, missing, _debug = build_paper_forward_feature_frame(candidate, snapshot)

    row = frame.iloc[0].to_dict()
    assert missing == []
    for field in required:
        assert row[field] is not None
