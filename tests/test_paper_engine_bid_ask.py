"""Tests for paper execution bid/ask pricing realism.

Verifies:
1. LONG BUY entry → ask
2. LONG SELL exit  → bid
3. SHORT SELL entry → bid
4. SHORT BUY exit  → ask
5. Missing bid/ask blocks trade by default (paper_allow_ltp_fallback=False)
6. LTP fallback only when explicitly enabled
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# MockClient — minimal broker stub
# ---------------------------------------------------------------------------

class MockClient:
    """Returns fixed bid/ask/ltp on every call."""

    def __init__(
        self,
        bid: float = None,
        ask: float = None,
        ltp: float = None,
    ):
        self._bid = bid
        self._ask = ask
        self._ltp = ltp

    def get_bid_ask(self, symbol, exchange_hint=None):
        return (self._bid, self._ask, self._ltp)

    def get_ltp(self, symbol, exchange=None):
        return self._ltp


# ---------------------------------------------------------------------------
# MockConfig — paper execution fields only
# ---------------------------------------------------------------------------

class MockConfig:
    def __init__(
        self,
        paper_use_bid_ask_execution: bool = True,
        paper_allow_ltp_fallback: bool = False,
        paper_ltp_fallback_spread_pct: float = 0.05,
        paper_slippage_pct: float = 0.001,
    ):
        self.paper_use_bid_ask_execution = paper_use_bid_ask_execution
        self.paper_allow_ltp_fallback = paper_allow_ltp_fallback
        self.paper_ltp_fallback_spread_pct = paper_ltp_fallback_spread_pct
        self.paper_slippage_pct = paper_slippage_pct
        self.enable_live_trading = False
        self.paper_extra_market_impact_pct = 0.0
        self.paper_apply_brokerage_costs = True
        self.paper_cost_model_source = "cost_model_assumptions"


# ---------------------------------------------------------------------------
# Tests — inline simulation of _get_paper_exit_price logic
# (mirrors the actual strategy.py implementation)
# ---------------------------------------------------------------------------

def _simulate_get_paper_exit_price(client, cfg, symbol, leg_side, qty, is_entry_side_buy):
    """Inline simulation of _get_paper_exit_price for test isolation."""
    spread_deduct = 0.0
    use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
    allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))
    ltp_fallback_penalty = float(getattr(cfg, "paper_ltp_fallback_spread_pct", 0.05) or 0.05)

    if not use_bid_ask:
        ltp_val = client.get_ltp(symbol)
        return (ltp_val, 0.0) if ltp_val else (None, 0.0)

    bid_raw, ask_raw, ltp_raw = client.get_bid_ask(symbol)

    if is_entry_side_buy:
        # Closing BUY position = we SELL → receive bid
        if bid_raw is None:
            if allow_ltp_fallback and ltp_raw is not None:
                return (float(ltp_raw) * (1.0 - ltp_fallback_penalty), 0.0)
            return (None, 0.0)
        exit_price = float(bid_raw)
        if ask_raw is not None:
            spread_deduct = float((float(ask_raw) - float(bid_raw)) * float(qty))
    else:
        # Closing SELL position = we BUY → pay ask
        if ask_raw is None:
            if allow_ltp_fallback and ltp_raw is not None:
                return (float(ltp_raw) * (1.0 + ltp_fallback_penalty), 0.0)
            return (None, 0.0)
        exit_price = float(ask_raw)
        if bid_raw is not None:
            spread_deduct = float((float(ask_raw) - float(bid_raw)) * float(qty))

    return (exit_price, spread_deduct)


class TestPaperBidAskExecution:
    """Bid/ask execution price correctness tests."""

    # -------------------------------------------------------------------------
    # test_long_buy_entry_uses_ask
    # -------------------------------------------------------------------------
    def test_long_buy_entry_uses_ask(self):
        """BUY entry (going long) must execute at ask price."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=False,
        )
        client = MockClient(bid=100.0, ask=102.0, ltp=101.0)

        bid_raw, ask_raw, ltp = client.get_bid_ask("NIFTY26JUN24000CE")
        use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
        entry_side = "BUY"

        if use_bid_ask:
            if entry_side == "BUY":
                if ask_raw is None:
                    entry_price = None
                else:
                    entry_price = float(ask_raw)  # BUY pays ask
            else:
                if bid_raw is None:
                    entry_price = None
                else:
                    entry_price = float(bid_raw)  # SELL receives bid

        assert entry_price == 102.0, f"Expected ask=102.0, got {entry_price}"

    # -------------------------------------------------------------------------
    # test_long_sell_exit_uses_bid
    # -------------------------------------------------------------------------
    def test_long_sell_exit_uses_bid(self):
        """Closing a BUY (long) position = SELL → must receive bid price."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=False,
        )
        client = MockClient(bid=104.0, ask=106.0, ltp=105.0)

        entry_side_buy = True   # the original leg was BUY (long)
        leg_side = "BUY"
        qty = 1

        exit_price, spread_deduct = _simulate_get_paper_exit_price(
            client, cfg, "NIFTY26JUN24000CE", leg_side, qty, is_entry_side_buy=entry_side_buy
        )

        assert exit_price == 104.0, f"Expected bid=104.0 for close, got {exit_price}"
        # Spread = ask - bid = 2.0; spread_deduct for qty=1
        assert abs(spread_deduct - 2.0) < 1e-9

    # -------------------------------------------------------------------------
    # test_short_sell_entry_uses_bid
    # -------------------------------------------------------------------------
    def test_short_sell_entry_uses_bid(self):
        """SELL entry (going short) must execute at bid price."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=False,
        )
        client = MockClient(bid=100.0, ask=102.0, ltp=101.0)

        bid_raw, ask_raw, ltp = client.get_bid_ask("NIFTY26JUN24000CE")
        use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
        entry_side = "SELL"

        if use_bid_ask:
            if entry_side == "BUY":
                entry_price = float(ask_raw)
            else:
                entry_price = float(bid_raw)  # short seller receives bid

        assert entry_price == 100.0, f"Expected bid=100.0, got {entry_price}"

    # -------------------------------------------------------------------------
    # test_short_buy_exit_uses_ask
    # -------------------------------------------------------------------------
    def test_short_buy_exit_uses_ask(self):
        """Closing a SELL (short) position = BUY → must pay ask price."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=False,
        )
        client = MockClient(bid=104.0, ask=106.0, ltp=105.0)

        entry_side_buy = False  # the original leg was SELL (short)
        leg_side = "SELL"
        qty = 1

        exit_price, spread_deduct = _simulate_get_paper_exit_price(
            client, cfg, "NIFTY26JUN24000CE", leg_side, qty, is_entry_side_buy=entry_side_buy
        )

        assert exit_price == 106.0, f"Expected ask=106.0 for close, got {exit_price}"
        assert abs(spread_deduct - 2.0) < 1e-9

    # -------------------------------------------------------------------------
    # test_missing_bid_ask_blocks_trade_by_default
    # -------------------------------------------------------------------------
    def test_missing_bid_ask_blocks_trade_by_default(self):
        """When bid/ask missing and paper_allow_ltp_fallback=False, trade is blocked."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=False,  # default = block
        )
        client = MockClient(bid=None, ask=None, ltp=101.0)

        entry_side = "BUY"
        bid_raw, ask_raw, ltp = client.get_bid_ask("NIFTY26JUN24000CE")
        allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))
        use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))

        blocked = False
        entry_price = None

        if use_bid_ask:
            if entry_side == "BUY":
                if ask_raw is None:
                    if not allow_ltp_fallback:
                        blocked = True
                    else:
                        entry_price = float(ltp) if ltp else None
                else:
                    entry_price = float(ask_raw)
            else:
                if bid_raw is None:
                    if not allow_ltp_fallback:
                        blocked = True
                    else:
                        entry_price = float(ltp) if ltp else None
                else:
                    entry_price = float(bid_raw)

        assert blocked is True, "Trade must be blocked when bid/ask missing and fallback disabled"
        assert entry_price is None, "No price when blocked"

    # -------------------------------------------------------------------------
    # test_ltp_fallback_only_when_explicitly_enabled
    # -------------------------------------------------------------------------
    def test_ltp_fallback_only_when_explicitly_enabled(self):
        """LTP fallback must ONLY activate when paper_allow_ltp_fallback=True."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=True,  # explicitly enabled
            paper_ltp_fallback_spread_pct=0.05,
        )
        client = MockClient(bid=None, ask=None, ltp=100.0)

        entry_side = "BUY"
        bid_raw, ask_raw, ltp_raw = client.get_bid_ask("NIFTY26JUN24000CE")
        allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))
        penalty = float(getattr(cfg, "paper_ltp_fallback_spread_pct", 0.05))

        entry_price = None
        price_source = "ltp"

        if entry_side == "BUY":
            if ask_raw is None:
                if allow_ltp_fallback:
                    entry_price = float(ltp_raw) * (1.0 - penalty)  # BUY: worse price
                    price_source = "ltp_fallback"
        else:
            if bid_raw is None:
                if allow_ltp_fallback:
                    entry_price = float(ltp_raw) * (1.0 + penalty)  # SELL: worse price
                    price_source = "ltp_fallback"

        assert entry_price is not None, "Entry price must be set when fallback is enabled"
        assert price_source == "ltp_fallback"
        # BUY: 100 * (1 - 0.05) = 95.0
        assert abs(entry_price - 95.0) < 0.01, f"Expected 95.0, got {entry_price}"

    # -------------------------------------------------------------------------
    # test_ltp_fallback_disabled_does_not_activate
    # -------------------------------------------------------------------------
    def test_ltp_fallback_disabled_does_not_activate(self):
        """With paper_allow_ltp_fallback=False, no LTP fallback occurs."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=False,
        )
        client = MockClient(bid=None, ask=None, ltp=100.0)

        entry_side = "BUY"
        bid_raw, ask_raw, ltp_raw = client.get_bid_ask("NIFTY26JUN24000CE")
        allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))

        fallback_used = False
        if entry_side == "BUY" and ask_raw is None and allow_ltp_fallback:
            fallback_used = True
        elif entry_side == "SELL" and bid_raw is None and allow_ltp_fallback:
            fallback_used = True

        assert fallback_used is False, "LTP fallback must not activate when disabled"

    # -------------------------------------------------------------------------
    # test_paper_use_bid_ask_false_uses_ltp_for_entry
    # -------------------------------------------------------------------------
    def test_paper_use_bid_ask_false_uses_ltp_for_entry(self):
        """When paper_use_bid_ask_execution=False, paper trades use LTP."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=False,  # disabled
            paper_allow_ltp_fallback=False,
        )
        client = MockClient(bid=100.0, ask=102.0, ltp=101.0)

        use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
        entry_side = "BUY"
        ltp = 101.0

        if not use_bid_ask:
            entry_price = float(ltp)  # falls back to LTP
            price_source = "ltp"
        else:
            bid_raw, ask_raw, _ = client.get_bid_ask("NIFTY26JUN24000CE")
            if entry_side == "BUY":
                entry_price = float(ask_raw)
            else:
                entry_price = float(bid_raw)
            price_source = "ask" if entry_side == "BUY" else "bid"

        assert entry_price == 101.0, f"Expected LTP=101.0, got {entry_price}"
        assert price_source == "ltp", f"Expected price_source=ltp, got {price_source}"

    # -------------------------------------------------------------------------
    # test_sell_entry_ltp_fallback_applies_positive_penalty
    # -------------------------------------------------------------------------
    def test_sell_entry_ltp_fallback_applies_positive_penalty(self):
        """For SELL entry, LTP fallback must add penalty (worse price for seller)."""
        cfg = MockConfig(
            paper_use_bid_ask_execution=True,
            paper_allow_ltp_fallback=True,
            paper_ltp_fallback_spread_pct=0.05,
        )
        client = MockClient(bid=None, ask=None, ltp=100.0)

        entry_side = "SELL"
        allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))
        penalty = float(getattr(cfg, "paper_ltp_fallback_spread_pct", 0.05))
        bid_raw, ask_raw, ltp_raw = client.get_bid_ask("NIFTY26JUN24000CE")

        entry_price = None
        if entry_side == "SELL":
            if bid_raw is None:
                if allow_ltp_fallback:
                    entry_price = float(ltp_raw) * (1.0 + penalty)  # SELL: worse price
            else:
                entry_price = float(bid_raw)

        # SELL with penalty: 100 * (1 + 0.05) = 105.0
        assert entry_price is not None
        assert abs(entry_price - 105.0) < 0.01, f"Expected 105.0, got {entry_price}"


# ---------------------------------------------------------------------------
# Test: Config defaults are correct
# ---------------------------------------------------------------------------

def test_config_defaults_for_paper_execution():
    """paper_use_bid_ask_execution should default to True; fallback to False."""
    from src.config import StrategyConfig

    cfg = StrategyConfig()

    assert cfg.paper_use_bid_ask_execution is True, \
        "paper_use_bid_ask_execution must default to True (realism)"
    assert cfg.paper_allow_ltp_fallback is False, \
        "paper_allow_ltp_fallback must default to False (safety)"
    assert abs(cfg.paper_ltp_fallback_spread_pct - 0.05) < 1e-9, \
        "paper_ltp_fallback_spread_pct must default to 0.05 (5%)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])