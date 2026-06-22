from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cost_model import CostModel, CostModelAssumptions, OptionChargesCalculator
from cost_model import evaluate_live_execution_friction


def test_cost_model_round_trip_cost_and_viability():
    model = CostModel(
        CostModelAssumptions(
            brokerage_per_side_pct=0.0001,
            exchange_charges_pct=0.0001,
            taxes_pct=0.0001,
            stt_ctt_pct=0.0001,
            gst_pct=0.0001,
            sebi_charges_pct=0.0001,
            stamp_duty_pct=0.0001,
            spread_proxy_pct=0.0002,
            slippage_proxy_pct=0.0002,
            minimum_net_edge_pct=0.0010,
        )
    )
    assert model.assumptions.round_trip_cost_pct > 0.0
    assert model.trade_is_viable(0.01) is True
    assert model.trade_is_viable(0.0005) is False


def test_option_charges_calculator_matches_expected_sell_leg_breakdown():
    calc = OptionChargesCalculator()
    summary = calc.calculate_charges(quantity=50, strike_price=22000, premium=200, side="SELL")

    assert summary["turnover"] == 10000.0
    assert summary["brokerage"] == 5.0
    assert summary["sebi_fee"] == 0.01
    assert summary["exchange_txn"] == 3.5
    assert summary["gst"] == 1.53
    assert summary["stt"] == 15.0
    assert summary["stamp_duty"] == 0.0
    assert summary["non_brokerage_charges"] == 20.05
    assert summary["total_charges"] == 25.05


def test_option_charges_calculator_excludes_ctt_for_buy_leg():
    calc = OptionChargesCalculator()
    summary = calc.calculate_charges(quantity=50, strike_price=22000, premium=200, side="BUY")

    assert summary["stt"] == 0.0
    assert summary["stamp_duty"] == 0.3
    assert summary["non_brokerage_charges"] == 5.35
    assert summary["total_charges"] == 10.35


def test_option_charges_calculator_matches_nifty_buy_example():
    calc = OptionChargesCalculator()
    summary = calc.calculate_charges(quantity=75, strike_price=22000, premium=100, side="BUY")

    assert summary["premium_turnover"] == 7500.0
    assert summary["stt"] == 0.0
    assert summary["stamp_duty"] == 0.23
    assert summary["exchange_charges"] == 2.63
    assert summary["sebi_fees"] == 0.0075
    assert summary["gst"] == 1.37
    assert summary["non_brokerage_charges"] == 4.23
    assert summary["total_charges"] == 9.23


def test_evaluate_live_execution_friction_rejects_wide_spread():
    class DummyClient:
        def get_bid_ask(self, symbol, exchange_hint=None):
            return 100.0, 101.0, 100.5

    result = evaluate_live_execution_friction(
        {
            "client": DummyClient(),
            "symbol": "NIFTYTEST",
            "exchange": "NFO",
            "cost_model": CostModel(),
        }
    )
    assert result["status"] == "REJECT_LIQUIDITY_CEILING"
    assert result["relative_spread_pct"] > result["allowed_cost_pct"]


def test_evaluate_live_execution_friction_accepts_tight_spread():
    class DummyClient:
        def get_bid_ask(self, symbol, exchange_hint=None):
            return 100.0, 100.1, 100.05

    result = evaluate_live_execution_friction(
        {
            "client": DummyClient(),
            "symbol": "NIFTYTEST",
            "exchange": "NFO",
            "cost_model": CostModel(),
        }
    )
    assert result["status"] in {"OK", "DEGRADED_QUOTE_FALLBACK"}
