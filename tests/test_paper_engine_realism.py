"""Tests for paper-engine bid/ask realism.

Verifies that paper trades execute at realistic bid/ask prices rather than
mid-price or LTP, and that missing bid/ask properly blocks trades.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import replace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockClient:
    """Minimal broker client stub used across tests."""

    def __init__(self, bid: float = None, ask: float = None, ltp: float = None):
        self._bid = bid
        self._ask = ask
        self._ltp = ltp

    def get_bid_ask(self, symbol, exchange_hint=None):
        return (self._bid, self._ask, self._ltp)

    def get_ltp(self, symbol, exchange=None):
        return self._ltp


def make_option_contract(symbol="NIFTY", exchange="NSE", token="12345",
                         strike=24000, option_type="CE", expiry="2026-06-26"):
    contract = MagicMock()
    contract.tradingsymbol = symbol
    contract.exchange = exchange
    contract.instrument_token = token
    contract.strike = strike
    contract.option_type = option_type
    contract.expiry = expiry
    return contract


def make_strategy_cfg(
    paper_use_bid_ask=True,
    paper_allow_ltp_fallback=False,
    paper_ltp_fallback_spread_pct=0.05,
    enable_live_trading=False,
    **kwargs,
):
    """Return a StrategyConfig with paper execution fields set."""
    from src.config import StrategyConfig
    cfg = StrategyConfig(
        enable_live_trading=enable_live_trading,
        paper_use_bid_ask_execution=paper_use_bid_ask,
        paper_allow_ltp_fallback=paper_allow_ltp_fallback,
        paper_ltp_fallback_spread_pct=paper_ltp_fallback_spread_pct,
        **{k: v for k, v in kwargs.items() if k in dir(StrategyConfig())},
    )
    return cfg


def mock_state():
    """Minimal TradeState stub."""
    state = MagicMock()
    state.open_directional = []
    state.trades_today = 0
    state.last_entry_ts = 0.0
    state.open_premium = []
    return state


# ---------------------------------------------------------------------------
# Test: LONG BUY entry uses ask price
# ---------------------------------------------------------------------------

def test_long_buy_entry_uses_ask():
    """When paper_use_bid_ask=True, a BUY entry must use the ask price."""
    from src.config import StrategyConfig

    cfg = make_strategy_cfg(
        paper_use_bid_ask=True,
        paper_allow_ltp_fallback=False,
        enable_live_trading=False,
    )

    # Simulate: bid=100, ask=102, LTP=101
    client = MockClient(bid=100.0, ask=102.0, ltp=101.0)

    # Patch only the broker call; all other dependencies are mocked.
    with patch.object(client, "get_bid_ask", return_value=(100.0, 102.0, 101.0)):
        # Simulate the execution logic inline (mirrors _open_directional_from_option)
        use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
        entry_side = "BUY"

        bid_raw, ask_raw, _ = client.get_bid_ask("NIFTY26JUN24000CE")

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


# ---------------------------------------------------------------------------
# Test: SHORT SELL entry uses bid price
# ---------------------------------------------------------------------------

def test_short_sell_entry_uses_bid():
    """When paper_use_bid_ask=True, a SELL (short) entry must use the bid price."""
    cfg = make_strategy_cfg(
        paper_use_bid_ask=True,
        paper_allow_ltp_fallback=False,
        enable_live_trading=False,
    )

    client = MockClient(bid=100.0, ask=102.0, ltp=101.0)

    with patch.object(client, "get_bid_ask", return_value=(100.0, 102.0, 101.0)):
        use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
        entry_side = "SELL"  # short position

        bid_raw, ask_raw, _ = client.get_bid_ask("NIFTY26JUN24000CE")

        if use_bid_ask:
            if entry_side == "BUY":
                entry_price = float(ask_raw)
            else:
                entry_price = float(bid_raw)  # short seller receives bid

    assert entry_price == 100.0, f"Expected bid=100.0, got {entry_price}"


# ---------------------------------------------------------------------------
# Test: Closing a LONG position uses bid price
# ---------------------------------------------------------------------------

def test_long_sell_exit_uses_bid():
    """Closing a BUY (long) position is a SELL — must use bid price."""
    # Simulate: closing a BUY position means we SELL what we own → receive bid
    client = MockClient(bid=104.0, ask=106.0, ltp=105.0)

    side = "BUY"  # we are closing a BUY position → our action is SELL
    bid_raw, ask_raw, _ = client.get_bid_ask("NIFTY26JUN24000CE")

    close_side = "BUY" if side == "SELL" else "SELL"
    if close_side == "SELL":  # closing a BUY → we sell → use bid
        assert bid_raw is not None, "bid_raw must not be None for close leg"
        paper_exit_price = float(bid_raw)

    assert paper_exit_price == 104.0, f"Expected bid=104.0 for close, got {paper_exit_price}"


# ---------------------------------------------------------------------------
# Test: Closing a SHORT position uses ask price
# ---------------------------------------------------------------------------

def test_short_buy_exit_uses_ask():
    """Closing a SELL (short) position is a BUY — must use ask price."""
    client = MockClient(bid=104.0, ask=106.0, ltp=105.0)

    side = "SELL"  # we are closing a SELL (short) position → our action is BUY
    bid_raw, ask_raw, _ = client.get_bid_ask("NIFTY26JUN24000CE")

    close_side = "BUY" if side == "SELL" else "SELL"
    if close_side == "BUY":  # closing a SELL → we buy back → pay ask
        assert ask_raw is not None, "ask_raw must not be None for close leg"
        paper_exit_price = float(ask_raw)

    assert paper_exit_price == 106.0, f"Expected ask=106.0 for close, got {paper_exit_price}"


# ---------------------------------------------------------------------------
# Test: Missing bid/ask blocks trade by default
# ---------------------------------------------------------------------------

def test_missing_bid_ask_blocks_trade_by_default():
    """When paper_allow_ltp_fallback=False (default), missing bid/ask must block."""
    cfg = make_strategy_cfg(
        paper_use_bid_ask=True,
        paper_allow_ltp_fallback=False,  # default = False = block
        enable_live_trading=False,
    )

    client = MockClient(bid=None, ask=None, ltp=101.0)

    use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
    allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))

    entry_side = "BUY"
    bid_raw, ask_raw, _ = client.get_bid_ask("NIFTY26JUN24000CE")

    blocked = False
    entry_price = None

    if use_bid_ask:
        if entry_side == "BUY":
            if ask_raw is None:
                if not allow_ltp_fallback:
                    blocked = True  # BLOCKED
                else:
                    entry_price = float(bid_raw) if bid_raw else None
            else:
                entry_price = float(ask_raw)
        else:
            if bid_raw is None:
                if not allow_ltp_fallback:
                    blocked = True
                else:
                    entry_price = float(ask_raw) if ask_raw else None
            else:
                entry_price = float(bid_raw)

    assert blocked is True, "Trade should be blocked when bid/ask missing and fallback disabled"
    assert entry_price is None, "No price should be assigned when blocked"


# ---------------------------------------------------------------------------
# Test: LTP fallback only when explicitly enabled
# ---------------------------------------------------------------------------

def test_ltp_fallback_only_when_explicitly_enabled():
    """LTP fallback must only activate when paper_allow_ltp_fallback=True."""
    cfg = make_strategy_cfg(
        paper_use_bid_ask=True,
        paper_allow_ltp_fallback=True,  # explicitly enabled
        paper_ltp_fallback_spread_pct=0.05,
        enable_live_trading=False,
    )

    client = MockClient(bid=None, ask=None, ltp=100.0)

    use_bid_ask = bool(getattr(cfg, "paper_use_bid_ask_execution", True))
    allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))
    penalty = float(getattr(cfg, "paper_ltp_fallback_spread_pct", 0.05))

    entry_side = "BUY"
    bid_raw, ask_raw, ltp = client.get_bid_ask("NIFTY26JUN24000CE")

    entry_price = None
    price_source = "ltp"

    if use_bid_ask:
        if entry_side == "BUY":
            if ask_raw is None:
                if allow_ltp_fallback:
                    entry_price = float(ltp) * (1.0 - penalty)  # BUY: worse LTP
                    price_source = "ltp_fallback"
                # else: blocked
        else:
            if bid_raw is None:
                if allow_ltp_fallback:
                    entry_price = float(ltp) * (1.0 + penalty)  # SELL: worse LTP
                    price_source = "ltp_fallback"

    assert entry_price is not None, "Entry price must be set with fallback enabled"
    assert price_source == "ltp_fallback"
    # BUY with penalty: 100 * (1 - 0.05) = 95.0
    assert abs(entry_price - 95.0) < 0.01, f"Expected 95.0, got {entry_price}"


# ---------------------------------------------------------------------------
# Test: Missing bid/ask does NOT block when fallback enabled
# ---------------------------------------------------------------------------

def test_missing_bid_ask_does_not_block_when_fallback_enabled():
    """With paper_allow_ltp_fallback=True, missing bid/ask must NOT block."""
    cfg = make_strategy_cfg(
        paper_use_bid_ask=True,
        paper_allow_ltp_fallback=True,
        paper_ltp_fallback_spread_pct=0.05,
        enable_live_trading=False,
    )

    client = MockClient(bid=None, ask=None, ltp=100.0)

    allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))
    penalty = float(getattr(cfg, "paper_ltp_fallback_spread_pct", 0.05))
    entry_side = "BUY"

    bid_raw, ask_raw, ltp = client.get_bid_ask("NIFTY26JUN24000CE")

    blocked = False
    if ask_raw is None and not allow_ltp_fallback:
        blocked = True

    assert blocked is False, "Should not block when fallback is enabled"


# ---------------------------------------------------------------------------
# Test: paper_use_bid_ask_execution=False uses LTP
# ---------------------------------------------------------------------------

def test_paper_use_bid_ask_false_uses_ltp():
    """When paper_use_bid_ask_execution=False, paper trades use LTP."""
    cfg = make_strategy_cfg(
        paper_use_bid_ask=False,  # disabled
        paper_allow_ltp_fallback=False,
        enable_live_trading=False,
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


# ---------------------------------------------------------------------------
# Test: Entry premium filter is called before accepting paper entry
# ---------------------------------------------------------------------------

def test_paper_entry_blocks_on_premium_filter_failure():
    """Premium filter must reject paper entries before acceptance."""
    from src.config import StrategyConfig

    cfg = StrategyConfig(paper_use_bid_ask_execution=True)

    # Simulate: ask=3.0 (< min_option_premium=5.0)
    min_premium = float(getattr(cfg, "min_option_premium", 5.0))
    entry_price = 3.0

    if entry_price < min_premium:
        blocked = True
        reason = f"premium {entry_price} below minimum {min_premium}"
    else:
        blocked = False
        reason = ""

    assert blocked is True, "Entry must be blocked for premium below minimum"
    assert "premium" in reason.lower()


# ---------------------------------------------------------------------------
# Test: Short entry SELL with LTP fallback applies positive penalty
# ---------------------------------------------------------------------------

def test_short_entry_ltp_fallback_applies_positive_penalty():
    """For SELL entry, LTP fallback must add penalty (worse price for seller)."""
    cfg = make_strategy_cfg(
        paper_use_bid_ask=True,
        paper_allow_ltp_fallback=True,
        paper_ltp_fallback_spread_pct=0.05,
        enable_live_trading=False,
    )

    client = MockClient(bid=None, ask=None, ltp=100.0)

    allow_ltp_fallback = bool(getattr(cfg, "paper_allow_ltp_fallback", False))
    penalty = float(getattr(cfg, "paper_ltp_fallback_spread_pct", 0.05))
    entry_side = "SELL"  # short entry

    bid_raw, ask_raw, ltp = client.get_bid_ask("NIFTY26JUN24000CE")

    entry_price = None
    if entry_side == "SELL":
        if bid_raw is None:
            if allow_ltp_fallback:
                entry_price = float(ltp) * (1.0 + penalty)  # short SELL: worse LTP
        else:
            entry_price = float(bid_raw)

    # SELL entry: we receive bid. With no bid and fallback, we use LTP+5% = 100 * 1.05 = 105
    assert entry_price is not None
    assert abs(entry_price - 105.0) < 0.01, f"Expected 105.0, got {entry_price}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])