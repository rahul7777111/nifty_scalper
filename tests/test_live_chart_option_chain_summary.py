"""
Smoke tests for option_chain_to_summary function.

Verifies end-to-end parsing of option chain payloads into OptionChainSummary.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import build_live_chart_snapshot, option_chain_to_summary


class TestOptionChainToSummaryBasic:
    """Basic option chain parsing tests."""

    def test_option_chain_to_summary_basic(self) -> None:
        """Simple CE/PE dict with strike, ltp, iv, oi."""
        payload = {
            "chain": [
                {
                    "option_type": "CE",
                    "strike": 25000.0,
                    "ltp": 250.0,
                    "bid": 248.0,
                    "ask": 252.0,
                    "iv": 18.5,
                    "oi": 75000,
                },
                {
                    "option_type": "PE",
                    "strike": 25000.0,
                    "ltp": 240.0,
                    "bid": 238.0,
                    "ask": 242.0,
                    "iv": 17.8,
                    "oi": 80000,
                },
            ],
        }

        summary = option_chain_to_summary(payload, spot=25000.0)

        # ATM strike should be rounded to nearest 50
        assert summary.atm_strike == 25000.0
        # CE fields
        assert summary.ce_ltp == 250.0
        assert summary.ce_iv == 18.5
        assert summary.ce_oi == 75000
        assert summary.ce_bid_ask_spread == 4.0  # 252 - 248
        # PE fields
        assert summary.pe_ltp == 240.0
        assert summary.pe_iv == 17.8
        assert summary.pe_oi == 80000
        assert summary.pe_bid_ask_spread == 4.0  # 242 - 238
        # PCR should be computed
        assert summary.pcr is not None
        assert abs(summary.pcr - (80000 / 75000)) < 0.01

    def test_option_chain_to_summary_nested(self) -> None:
        """Nested call_options/put_options structure is now correctly supported."""
        # Fixed: nested call_options/put_options is parsed via the non-empty list branch
        payload = {
            "call_options": [
                {
                    "instrument": "CE",
                    "strike": 25000.0,
                    "last_price": 250.0,
                    "bid": 248.0,
                    "ask": 252.0,
                    "implied_volatility": 18.5,
                    "open_interest": 75000,
                },
            ],
            "put_options": [
                {
                    "instrument": "PE",
                    "strike": 25000.0,
                    "last_price": 240.0,
                    "bid": 238.0,
                    "ask": 242.0,
                    "implied_volatility": 17.8,
                    "open_interest": 80000,
                },
            ],
        }

        summary = option_chain_to_summary(payload, spot=25000.0)
        assert summary.atm_strike == 25000.0
        # Now correctly parsed via the nested call_options/put_options branch
        assert summary.ce_ltp == 250.0
        assert summary.ce_iv == 18.5
        assert summary.ce_oi == 75000
        assert summary.pe_ltp == 240.0
        assert summary.pe_iv == 17.8
        assert summary.pe_oi == 80000
        assert summary.pcr is not None

    def test_option_chain_to_summary_empty(self) -> None:
        """Empty payload → graceful None defaults."""
        # None payload
        summary_none = option_chain_to_summary(None)
        assert summary_none.atm_strike is None
        assert summary_none.ce_ltp is None
        assert summary_none.pe_ltp is None
        assert summary_none.ce_iv is None
        assert summary_none.pe_iv is None
        assert summary_none.ce_oi is None
        assert summary_none.pe_oi is None
        assert summary_none.pcr is None

        # Empty dict
        summary_empty_dict = option_chain_to_summary({})
        assert summary_empty_dict.atm_strike is None

        # Empty chain list
        summary_empty_chain = option_chain_to_summary({"chain": []}, spot=25000.0)
        assert summary_empty_chain.atm_strike == 25000.0  # ATM still from spot
        assert summary_empty_chain.ce_ltp is None
        assert summary_empty_chain.pe_ltp is None

    def test_option_chain_to_summary_full_snapshot(self) -> None:
        """Verify option chain parsing works within full snapshot build."""
        payload = {
            "chain": [
                {
                    "option_type": "CE",
                    "strike": 24950.0,
                    "ltp": 280.0,
                    "bid": 278.0,
                    "ask": 282.0,
                    "iv": 19.0,
                    "oi": 60000,
                },
                {
                    "option_type": "CE",
                    "strike": 25000.0,
                    "ltp": 250.0,
                    "bid": 248.0,
                    "ask": 252.0,
                    "iv": 18.5,
                    "oi": 75000,
                },
                {
                    "option_type": "PE",
                    "strike": 25000.0,
                    "ltp": 240.0,
                    "bid": 238.0,
                    "ask": 242.0,
                    "iv": 17.8,
                    "oi": 80000,
                },
                {
                    "option_type": "PE",
                    "strike": 25050.0,
                    "ltp": 260.0,
                    "bid": 258.0,
                    "ask": 262.0,
                    "iv": 18.2,
                    "oi": 65000,
                },
            ],
        }

        snapshot = build_live_chart_snapshot(
            spot_price=25000.0,
            option_chain_payload=payload,
        )

        assert snapshot.option_chain_summary.atm_strike == 25000.0
        # Should pick nearest ATM CE (25000 strike)
        assert snapshot.option_chain_summary.ce_ltp == 250.0
        # Should pick nearest ATM PE (25000 strike)
        assert snapshot.option_chain_summary.pe_ltp == 240.0
        # PCR should be computed
        assert snapshot.option_chain_summary.pcr is not None

    def test_option_chain_staleness_detection(self) -> None:
        """Verify stale data is detected when timestamp is old."""
        old_ts = (datetime.now() - timedelta(minutes=5)).timestamp()
        payload = {
            "chain": [
                {"option_type": "CE", "strike": 25000.0, "ltp": 250.0, "bid": 248.0, "ask": 252.0},
                {"option_type": "PE", "strike": 25000.0, "ltp": 240.0, "bid": 238.0, "ask": 242.0},
            ],
            "timestamp": old_ts,
        }

        summary = option_chain_to_summary(payload, spot=25000.0)
        # Age is 5 minutes = 300 seconds > 60 seconds threshold
        assert summary.is_stale is True
        assert summary.liquidity_score == "DANGER"

    def test_option_chain_wide_spread_warning(self) -> None:
        """Verify wide spread warning is set correctly."""
        payload = {
            "chain": [
                {"option_type": "CE", "strike": 25000.0, "ltp": 250.0, "bid": 100.0, "ask": 400.0},
                {"option_type": "PE", "strike": 25000.0, "ltp": 240.0, "bid": 238.0, "ask": 242.0},
            ],
        }

        summary = option_chain_to_summary(payload, spot=25000.0)
        # CE spread = 400 - 100 = 300 > 2.0 threshold
        assert summary.wide_spread_warning is True
        # PE spread = 242 - 238 = 4 > 2.0 threshold
        assert summary.pe_bid_ask_spread == 4.0

    def test_option_chain_iv_spike_warning(self) -> None:
        """Verify IV spike warning is set correctly."""
        payload = {
            "chain": [
                {"option_type": "CE", "strike": 25000.0, "ltp": 250.0, "bid": 248.0, "ask": 252.0, "iv": 35.0},
                {"option_type": "PE", "strike": 25000.0, "ltp": 240.0, "bid": 238.0, "ask": 242.0, "iv": 25.0},
            ],
        }

        summary = option_chain_to_summary(payload, spot=25000.0)
        # CE IV = 35 > 30 threshold
        assert summary.iv_spike_warning is True
        assert summary.ce_iv == 35.0

    def test_option_chain_liquidity_score_ok(self) -> None:
        """Verify liquidity score = OK when all metrics are fine."""
        now_ts = datetime.now().timestamp()
        # Use tight spreads (< 2.0) and low IV (< 30) to get OK score
        payload = {
            "chain": [
                {"option_type": "CE", "strike": 25000.0, "ltp": 250.0, "bid": 249.0, "ask": 251.0, "iv": 18.0, "oi": 75000},
                {"option_type": "PE", "strike": 25000.0, "ltp": 240.0, "bid": 239.5, "ask": 240.5, "iv": 17.0, "oi": 80000},
            ],
            "timestamp": now_ts,
        }

        summary = option_chain_to_summary(payload, spot=25000.0)
        # Spreads: CE = 2.0, PE = 1.0 (both <= 2.0, so not wide)
        # IVs: CE = 18, PE = 17 (both < 30, so no spike)
        # Not stale since timestamp is current
        assert summary.liquidity_score == "OK"

    def test_option_chain_pcr_calculation(self) -> None:
        """Verify Put/Call Ratio calculation."""
        payload = {
            "chain": [
                {"option_type": "CE", "strike": 25000.0, "ltp": 250.0, "oi": 50000},
                {"option_type": "PE", "strike": 25000.0, "ltp": 240.0, "oi": 75000},
            ],
        }

        summary = option_chain_to_summary(payload, spot=25000.0)
        # PCR = 75000 / 50000 = 1.5
        assert summary.pcr == 1.5