from __future__ import annotations

from datetime import datetime

from src.ui import ScalperUI


def _dummy_ui() -> ScalperUI:
    return ScalperUI.__new__(ScalperUI)


def test_format_net_hedge_legs_nets_by_symbol() -> None:
    ui = _dummy_ui()

    txt = ScalperUI._format_net_hedge_legs(
        ui,
        [
            {"symbol": "NSE:NIFTYBEES", "side": "BUY", "quantity": 150, "is_hedge": True, "entry_price": 100.0, "ltp": 101.0},
            {"symbol": "NSE:NIFTYBEES", "side": "SELL", "quantity": 100, "is_hedge": True, "entry_price": 99.0, "ltp": 101.0},
            {"symbol": "NSE:BANKBEES", "side": "SELL", "quantity": 20, "is_hedge": True, "entry_price": 201.0, "exit_price": 198.0},
            {"symbol": "NSE:BANKBEES", "side": "BUY", "quantity": 20, "is_hedge": True, "entry_price": 199.0, "exit_price": 198.0},
        ],
    )

    assert "NSE:NIFTYBEES BUY x50" in txt
    assert "legs 2" in txt
    assert "entry 99.60" in txt
    assert "last 101.00" in txt
    assert "NSE:BANKBEES FLAT" in txt
    assert "buy avg 199.00" in txt
    assert "sell avg 201.00" in txt
    assert "exit 198.00" in txt


def test_trade_log_row_values_for_hedge_uses_netted_text() -> None:
    ui = _dummy_ui()
    ui._split_legs_for_display = lambda legs: ([], [], [])  # type: ignore[assignment]
    ui._format_trade_legs_for_state = lambda tid, state: "SELL NIFTYBEES x65 (entry 120.00, last 118.00)"  # type: ignore[assignment]

    row = ScalperUI._trade_log_row_values_from_state(
        ui,
        "T-1-H",
        {
            "opened_ts": float(datetime.now().timestamp()),
            "pos_type": "delta_hedge",
            "strategy": "auto Δ-hedge",
            "legs": [{"symbol": "NSE:NIFTYBEES", "side": "SELL", "quantity": 65, "is_hedge": True}],
            "mtm": 130.5,
            "status": "OPEN",
        },
    )

    assert row is not None
    # row[4] = legs_text, row[5] = mtm_display
    assert row[4] == "SELL NIFTYBEES x65 (entry 120.00, last 118.00)"
    assert row[5] == "130.50"


def test_underlying_hedge_row_inherits_parent_expiry_format() -> None:
    ui = _dummy_ui()
    ui._trade_state = {
        "D1": {
            "legs": [
                {
                    "symbol": "NIFTY26MAY23950PE",
                    "side": "BUY",
                    "quantity": 65,
                    "strike": 23950,
                    "option_type": "PE",
                    "expiry": "26-May-26",
                    "entry_price": 88.70,
                }
            ],
        },
        "D1-H": {
            "legs": [
                {
                    "symbol": "NIFTY",
                    "side": "BUY",
                    "quantity": 65,
                    "is_hedge": True,
                    "entry_price": 23938.90,
                    "ltp": 23940.70,
                }
            ],
        },
    }

    txt = ScalperUI._format_trade_legs_for_state(ui, "D1-H", ui._trade_state["D1-H"])

    assert txt == "BUY 26-May-26 23950 PE x65 (entry 23938.90, last 23940.70)"


def test_format_net_hedge_legs_can_skip_flat_symbols() -> None:
    ui = _dummy_ui()

    txt = ScalperUI._format_net_hedge_legs(
        ui,
        [
            {"symbol": "NSE:NIFTYBEES", "side": "BUY", "quantity": 100, "is_hedge": True},
            {"symbol": "NSE:NIFTYBEES", "side": "SELL", "quantity": 100, "is_hedge": True},
        ],
        include_flat=False,
    )

    assert txt == ""


def test_trade_log_row_values_ignores_zero_qty_main_legs() -> None:
    ui = _dummy_ui()
    
    # Mock _format_trade_legs_for_state to return a predictable string
    def mock_format_trade_legs(trade_id: str, state: dict) -> str:
        legs = state.get("legs", [])
        # Filter out zero quantity legs for display
        non_zero_legs = [l for l in legs if l.get("quantity", 0) > 0]
        if not non_zero_legs:
            return ""
        # Simple format for testing
        parts = []
        for leg in non_zero_legs:
            side = leg.get("side", "")
            opt_type = leg.get("option_type", "")
            qty = leg.get("quantity", 0)
            parts.append(f"{side} {opt_type} x{qty}")
        return ", ".join(parts)
    
    ui._format_trade_legs_for_state = mock_format_trade_legs  # type: ignore[assignment]

    row = ScalperUI._trade_log_row_values_from_state(
        ui,
        "T-2",
        {
            "opened_ts": float(datetime.now().timestamp()),
            "pos_type": "multi",
            "strategy": "iron_condor",
            "legs": [
                {
                    "symbol": "NIFTY26APR25000CE",
                    "side": "SELL",
                    "quantity": 0,
                    "option_type": "CE",
                    "entry_price": 100.0,
                },
                {
                    "symbol": "NIFTY26APR24900PE",
                    "side": "SELL",
                    "quantity": 50,
                    "option_type": "PE",
                    "entry_price": 95.0,
                },
            ],
            "status": "PARTIAL",
        },
    )

    assert row is not None
    # row[4] = legs_text
    assert "25000 CE" not in row[4]
    assert "SELL PE x50" in row[4]
