from __future__ import annotations

from types import SimpleNamespace

from src.ui import ScalperUI


def test_compute_cached_trade_mtm_uses_cached_leg_ltps() -> None:
    dummy = SimpleNamespace()

    mtm = ScalperUI._compute_cached_trade_mtm(
        dummy,
        [
            {"side": "SELL", "quantity": 50, "entry_price": 100.0, "ltp": 90.0},
            {"side": "BUY", "quantity": 50, "entry_price": 20.0, "ltp": 24.0},
        ],
    )

    assert mtm == 700.0


def test_build_live_option_update_event_refreshes_option_leg_prices() -> None:
    client = SimpleNamespace(get_ltp=lambda key: 110.0 if str(key).endswith(":123") else 95.0)

    dummy = SimpleNamespace()
    dummy._is_closed_trade_state = lambda trade_id, state: False
    dummy._is_option_leg = lambda leg: str(leg.get("symbol") or "").endswith(("CE", "PE"))
    dummy._try_get_live_ltp_for_leg = lambda client, leg: ScalperUI._try_get_live_ltp_for_leg(dummy, client, leg)
    dummy._compute_cached_trade_mtm = lambda legs: ScalperUI._compute_cached_trade_mtm(dummy, legs)

    state = {
        "pos_type": "multi",
        "strategy": "iron_condor",
        "status": "OPEN",
        "legs": [
            {"symbol": "NIFTY2631723950CE", "token": "123", "exchange": "NFO", "side": "SELL", "quantity": 50, "entry_price": 100.0},
            {"symbol": "NIFTY2631723750PE", "token": "124", "exchange": "NFO", "side": "BUY", "quantity": 50, "entry_price": 90.0},
        ],
    }

    evt = ScalperUI._build_live_option_update_event(dummy, client, "T-1", state)

    assert evt is not None
    assert evt.event == "UPDATE"
    assert evt.trade_id == "T-1"
    assert evt.mtm == -250.0
    assert [float(leg.get("ltp")) for leg in evt.legs] == [110.0, 95.0]


def test_build_live_option_update_event_ignores_zero_qty_legs() -> None:
    client = SimpleNamespace(get_ltp=lambda key: 111.0)

    dummy = SimpleNamespace()
    dummy._is_closed_trade_state = lambda trade_id, state: False
    dummy._is_option_leg = lambda leg: str(leg.get("symbol") or "").endswith(("CE", "PE"))
    dummy._try_get_live_ltp_for_leg = lambda client, leg: ScalperUI._try_get_live_ltp_for_leg(dummy, client, leg)
    dummy._compute_cached_trade_mtm = lambda legs: ScalperUI._compute_cached_trade_mtm(dummy, legs)

    state = {
        "pos_type": "multi",
        "strategy": "long_straddle",
        "status": "PARTIAL",
        "legs": [
            {
                "symbol": "NIFTY2631723950CE",
                "token": "123",
                "exchange": "NFO",
                "side": "BUY",
                "quantity": 0,
                "entry_price": 100.0,
            }
        ],
    }

    evt = ScalperUI._build_live_option_update_event(dummy, client, "M-closedish", state)
    assert evt is None


def test_build_live_option_update_event_refreshes_hedge_rows_when_requested() -> None:
    client = SimpleNamespace(get_ltp=lambda key: 242.0 if str(key).endswith("NIFTYBEES") else 0.0)

    dummy = SimpleNamespace()
    dummy._is_closed_trade_state = lambda trade_id, state: False
    dummy._is_option_leg = lambda leg: str(leg.get("symbol") or "").endswith(("CE", "PE"))
    dummy._try_get_live_ltp_for_leg = lambda client, leg: ScalperUI._try_get_live_ltp_for_leg(dummy, client, leg)
    dummy._compute_cached_trade_mtm = lambda legs: ScalperUI._compute_cached_trade_mtm(dummy, legs)

    state = {
        "pos_type": "hedge",
        "strategy": "delta_hedge",
        "status": "OPEN",
        "legs": [
            {"symbol": "NSE:NIFTYBEES", "exchange": "NSE", "side": "BUY", "quantity": 50, "entry_price": 240.0},
            {"symbol": "NSE:NIFTYBEES", "exchange": "NSE", "side": "SELL", "quantity": 20, "entry_price": 241.0},
        ],
    }

    evt = ScalperUI._build_live_option_update_event(
        dummy,
        client,
        "T-hedge-H",
        state,
        refresh_all_legs=True,
    )

    assert evt is not None
    assert evt.trade_id == "T-hedge-H"
    assert [float(leg.get("ltp")) for leg in evt.legs] == [242.0, 242.0]
    assert evt.mtm == 80.0


def test_is_leg_stop_hit_for_display_buy_and_sell() -> None:
    dummy = SimpleNamespace()

    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="BUY", ltp=74.2, stop_price=75.59) is True
    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="BUY", ltp=76.0, stop_price=75.59) is False
    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="SELL", ltp=102.0, stop_price=101.0) is True
    assert ScalperUI._is_leg_stop_hit_for_display(dummy, side="SELL", ltp=99.0, stop_price=101.0) is False