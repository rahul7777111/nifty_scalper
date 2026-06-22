"""Tests for option chain summary panel — no broker needed."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import option_chain_to_summary, OptionChainSummary


class TestOptionChainNonePayload:
    def test_none_returns_all_defaults(self) -> None:
        oc = option_chain_to_summary(None, None)
        assert oc.atm_strike is None
        assert oc.ce_ltp is None
        assert oc.pe_ltp is None
        assert oc.pcr is None
        assert oc.liquidity_score == "N/A"


class TestOptionChainAtmStrike:
    def test_atm_rounded_to_50(self) -> None:
        oc = option_chain_to_summary({}, spot=24517.0)
        assert oc.atm_strike == 24500.0

    def test_atm_rounded_up_for_32(self) -> None:
        oc = option_chain_to_summary({}, spot=24532.0)
        assert oc.atm_strike == 24550.0

    def test_atm_exactly_on_50(self) -> None:
        oc = option_chain_to_summary({}, spot=24550.0)
        assert oc.atm_strike == 24550.0


class TestOptionChainLTP:
    def test_ce_ltp_from_chain(self) -> None:
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 250.0, "bid": 249.0, "ask": 251.0, "iv": 18.0, "oi": 50000},
            {"option_type": "PE", "strike": 24500.0, "ltp": 240.0, "bid": 239.0, "ask": 241.0, "iv": 17.5, "oi": 55000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.ce_ltp == 250.0
        assert oc.pe_ltp == 240.0

    def test_ltp_also_from_last_price(self) -> None:
        chain = [
            {"option_type": "CE", "strike": 24500.0, "last_price": 248.0, "bid": 247.0, "ask": 249.0, "iv": 18.0, "oi": 50000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.ce_ltp == 248.0


class TestOptionChainIV:
    def test_iv_extracted(self) -> None:
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 199.0, "ask": 201.0, "iv": 22.5, "oi": 50000},
            {"option_type": "PE", "strike": 24500.0, "ltp": 190.0, "bid": 189.0, "ask": 191.0, "iv": 21.0, "oi": 55000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.ce_iv == 22.5
        assert oc.pe_iv == 21.0


class TestOptionChainPCR:
    def test_pcr_computed(self) -> None:
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 199.0, "ask": 201.0, "iv": 18.0, "oi": 50000},
            {"option_type": "CE", "strike": 24550.0, "ltp": 150.0, "bid": 149.0, "ask": 151.0, "iv": 18.0, "oi": 30000},
            {"option_type": "PE", "strike": 24500.0, "ltp": 190.0, "bid": 189.0, "ask": 191.0, "iv": 17.5, "oi": 80000},
            {"option_type": "PE", "strike": 24450.0, "ltp": 180.0, "bid": 179.0, "ask": 181.0, "iv": 17.5, "oi": 20000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        # CE OI = 50000 + 30000 = 80000, PE OI = 80000 + 20000 = 100000
        assert oc.pcr == pytest.approx(1.25, rel=0.01)


class TestOptionChainBidAskSpread:
    def test_spread_computed(self) -> None:
        # CE: bid=198, ask=202 → spread = 4
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 198.0, "ask": 202.0, "iv": 18.0, "oi": 50000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.ce_bid_ask_spread == 4.0


class TestOptionChainLiquidityScore:
    def test_ok_when_all_normal(self) -> None:
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 199.5, "ask": 200.5, "iv": 18.0, "oi": 50000},
            {"option_type": "PE", "strike": 24500.0, "ltp": 190.0, "bid": 189.5, "ask": 190.5, "iv": 17.5, "oi": 55000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.liquidity_score == "OK"

    def test_warning_when_spread_wide(self) -> None:
        # spread = 20 → > 2.0 threshold → wide_spread_warning
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 190.0, "ask": 210.0, "iv": 18.0, "oi": 50000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.wide_spread_warning is True
        assert oc.liquidity_score in ("WARNING", "DANGER")

    def test_warning_when_iv_spike(self) -> None:
        # IV = 35 > 30 threshold → iv_spike_warning
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 199.5, "ask": 200.5, "iv": 35.0, "oi": 50000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.iv_spike_warning is True
        assert oc.liquidity_score in ("WARNING", "DANGER")

    def test_danger_when_both_wide_and_spike(self) -> None:
        # spread > 2 AND IV > 30 → DANGER
        chain = [
            {"option_type": "CE", "strike": 24500.0, "ltp": 200.0, "bid": 190.0, "ask": 210.0, "iv": 35.0, "oi": 50000},
        ]
        oc = option_chain_to_summary({"chain": chain}, spot=24500.0)
        assert oc.liquidity_score == "DANGER"


class TestOptionChainStaleness:
    def test_stale_if_ts_too_old(self) -> None:
        import time
        old_ts = time.time() - 120
        payload = {"chain": [], "timestamp": old_ts}
        oc = option_chain_to_summary(payload, spot=24500.0)
        assert oc.is_stale is True

    def test_not_stale_if_recent(self) -> None:
        import time
        recent_ts = time.time() - 10
        payload = {"chain": [], "timestamp": recent_ts}
        oc = option_chain_to_summary(payload, spot=24500.0)
        assert oc.is_stale is False