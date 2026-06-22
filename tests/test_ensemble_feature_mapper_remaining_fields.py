from __future__ import annotations

from src.ml.ensemble_auto_router import build_ensemble_feature_frame


REMAINING_FEATURES = [
    "ltp",
    "ctx_option_price",
    "extrinsic_value",
    "bid_ask_spread",
    "ltp_vs_mid_diff",
]


def _base(side: str = "CE", **overrides):
    row = {
        "option_type": side,
        "spot": 24500.0,
        "strike_price": 24500.0,
        "expiry_date": "2026-06-25",
        "timestamp": "2026-06-18T10:00:00",
        "last_price": 120.0,
        "bid": 119.0,
        "ask": 121.0,
    }
    row.update(overrides)
    return row


def _row(snapshot, features=None):
    frame, diag = build_ensemble_feature_frame(snapshot, features or REMAINING_FEATURES, return_diagnostics=True)
    assert diag["truly_missing_count"] == 0
    return frame.iloc[0]


def test_ltp_is_filled_from_last_price():
    row = _row(_base(last_price=123.45, ltp=None))

    assert row["ltp"] == 123.45


def test_ctx_option_price_equals_ltp():
    row = _row(_base(last_price=111.25))

    assert row["ctx_option_price"] == row["ltp"] == 111.25


def test_bid_ask_spread_equals_ask_minus_bid():
    row = _row(_base(last_price=120.0, bid=118.5, ask=121.0))

    assert row["bid_ask_spread"] == 2.5


def test_ltp_vs_mid_diff_equals_ltp_minus_mid():
    row = _row(_base(last_price=120.0, bid=118.0, ask=121.0))

    assert row["ltp_vs_mid_diff"] == 0.5


def test_ce_extrinsic_value_calculation():
    row = _row(_base("CE", spot=24550.0, strike_price=24500.0, last_price=120.0))

    assert row["extrinsic_value"] == 70.0


def test_pe_extrinsic_value_calculation():
    row = _row(_base("PE", spot=24450.0, strike_price=24500.0, last_price=110.0))

    assert row["extrinsic_value"] == 60.0


def test_missing_bid_ask_gives_zero_spread_and_diff():
    row = _row(_base(last_price=120.0, bid=None, ask=None))

    assert row["bid_ask_spread"] == 0.0
    assert row["ltp_vs_mid_diff"] == 0.0


def test_no_remaining_missing_fields_for_required_set():
    frame, diag = build_ensemble_feature_frame(_base(last_price=120.0), REMAINING_FEATURES, return_diagnostics=True)

    assert list(frame.columns) == REMAINING_FEATURES
    assert diag["truly_missing_count"] == 0
    assert diag["truly_missing_features"] == []


def test_ltp_can_fallback_to_mid_price():
    row = _row(_base(ltp=None, last_price=None, close=None, option_price=None, bid=100.0, ask=104.0))

    assert row["ltp"] == 102.0
    assert row["ctx_option_price"] == 102.0


def test_ltp_can_fallback_to_intrinsic_plus_extrinsic():
    row = _row(
        _base(
            "CE",
            spot=24550.0,
            strike_price=24500.0,
            ltp=None,
            last_price=None,
            close=None,
            option_price=None,
            bid=None,
            ask=None,
            extrinsic_value=25.0,
        )
    )

    assert row["ltp"] == 75.0
    assert row["ctx_option_price"] == 75.0
