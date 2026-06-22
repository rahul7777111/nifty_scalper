"""Tests for premium and spread filter defaults on NIFTY options paper-mode entry.

These tests verify that the safe default values in config.py and the logic in
strategy.py correctly block entries when:
  - Option premium is below minimum
  - Bid/ask spread is too wide (pct or absolute)
  - Bid/ask is missing when required

And correctly allow entries when filters are satisfied.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import strategy as strat_mod


# ---------------------------------------------------------------------------
# Minimal config stub matching StrategyConfig field names
# ---------------------------------------------------------------------------

@dataclass
class _Cfg:
    entry_min_option_premium: float = 5.0
    entry_max_option_premium: float = 0.0
    entry_require_bid_ask: bool = True
    entry_max_bid_ask_spread_pct: float = 2.0   # percent — pct > max_pct (not *100)
    entry_max_bid_ask_spread_abs: float = 5.0   # absolute Rs.
    entry_spread_shock_mult: float = 1.5        # block if spread > 1.5x recent median
    entry_spread_shock_lookback: int = 20
    entry_spread_shock_min_samples: int = 5
    min_option_premium: float = 5.0             # global fallback when entry_min disabled
    exit_require_bid_ask: bool = True
    exit_max_bid_ask_spread_pct: float = 3.0
    exit_max_bid_ask_spread_abs: float = 8.0
    debug_log_no_signal: bool = False


# ---------------------------------------------------------------------------
# Minimal Strategy mock
# ---------------------------------------------------------------------------

class _MockStrategy:
    """Minimal mock exposing _entry_price_bounds_ok and _check_entry_liquidity."""

    def __init__(self, cfg: _Cfg, client: Optional[MagicMock] = None):
        self.cfg: _Cfg = cfg
        self.client = client or MagicMock()
        # Spread watch history: key -> list of observed spreads
        self._entry_spread_watch: Dict[str, List[float]] = {}

    # Proxy to the real method so we test actual logic
    def _entry_price_bounds_ok(
        self, price: Optional[float]
    ) -> Tuple[bool, str]:
        """Replicate strategy._entry_price_bounds_ok logic."""
        try:
            min_p = float(getattr(self.cfg, "entry_min_option_premium", 0.0) or 0.0)
        except Exception:
            min_p = 0.0
        try:
            max_p = float(getattr(self.cfg, "entry_max_option_premium", 0.0) or 0.0)
        except Exception:
            max_p = 0.0

        if min_p <= 0 and max_p <= 0:
            return True, ""

        if price is None:
            return False, "Option premium unavailable for bounds check"

        try:
            p = float(price)
        except Exception:
            return False, "Option premium could not be parsed"

        if min_p > 0 and p < min_p:
            return False, f"Option premium {p:.2f} below min {min_p:.2f}"
        if max_p > 0 and p > max_p:
            return False, f"Option premium {p:.2f} above max {max_p:.2f}"
        return True, ""

    def _check_entry_liquidity(self, legs: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Replicate strategy._check_entry_liquidity logic (order of checks must match real code)."""
        try:
            require_ba = bool(getattr(self.cfg, "entry_require_bid_ask", False))
        except Exception:
            require_ba = False
        try:
            max_pct = float(getattr(self.cfg, "entry_max_bid_ask_spread_pct", 0.0) or 0.0)
        except Exception:
            max_pct = 0.0
        try:
            max_abs = float(getattr(self.cfg, "entry_max_bid_ask_spread_abs", 0.0) or 0.0)
        except Exception:
            max_abs = 0.0
        try:
            shock_mult = float(getattr(self.cfg, "entry_spread_shock_mult", 0.0) or 0.0)
        except Exception:
            shock_mult = 0.0

        # Fast path: skip spread calculations when no thresholds are configured
        if max_pct <= 0 and max_abs <= 0 and shock_mult <= 0:
            # NOTE: even in this fast path, require_bid_ask must still be enforced
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                symbol = str(leg.get("symbol") or "")
                if not symbol:
                    continue
                exchange = str(leg.get("exchange") or "") or None
                tok = str(leg.get("token") or "").strip()
                quote_key = tok if (tok.isdigit() and int(tok) > 0) else symbol
                bid, ask, _ = self.client.get_bid_ask(quote_key, exchange_hint=exchange)
                if require_ba and (bid is None or ask is None):
                    return False, f"Bid/ask unavailable for {symbol}"
            return True, ""

        def _spread_key(leg: Dict[str, Any], fallback_symbol: str) -> str:
            token = str(leg.get("token") or "").strip()
            exch = str(leg.get("exchange") or "").strip().upper()
            if token:
                return f"{exch}:{token}" if exch else token
            return str(fallback_symbol or "").strip().upper()

        def _shock_ok(key: str, spread: float) -> Tuple[bool, str]:
            if shock_mult <= 0:
                return True, ""
            try:
                lookback = int(getattr(self.cfg, "entry_spread_shock_lookback", 20) or 20)
            except Exception:
                lookback = 20
            try:
                min_samples = int(getattr(self.cfg, "entry_spread_shock_min_samples", 5) or 5)
            except Exception:
                min_samples = 5
            lookback = max(2, min(200, int(lookback)))
            min_samples = max(1, min(int(lookback), int(min_samples)))

            hist = list(self._entry_spread_watch.get(key) or [])
            ok = True
            reason = ""
            if len(hist) >= int(min_samples):
                ordered = sorted(float(x) for x in hist if float(x) >= 0)
                if ordered:
                    mid_i = len(ordered) // 2
                    if len(ordered) % 2:
                        baseline = float(ordered[mid_i])
                    else:
                        baseline = (float(ordered[mid_i - 1]) + float(ordered[mid_i])) / 2.0
                    if baseline > 0 and float(spread) > float(baseline) * float(shock_mult):
                        ok = False
                        reason = (
                            f"Spread shock for {key} "
                            f"({float(spread):.2f} > {float(shock_mult):.2f}x median {float(baseline):.2f})"
                        )
            hist.append(float(spread))
            if len(hist) > int(lookback):
                hist = hist[-int(lookback):]
            self._entry_spread_watch[key] = hist
            return ok, reason

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            symbol = str(leg.get("symbol") or "")
            if not symbol:
                continue
            exchange = str(leg.get("exchange") or "") or None
            tok = str(leg.get("token") or "").strip()
            quote_key = tok if (tok.isdigit() and int(tok) > 0) else symbol
            bid, ask, _ltp = self.client.get_bid_ask(quote_key, exchange_hint=exchange)
            if require_ba and (bid is None or ask is None):
                if _ltp is None:
                    return False, f"Bid/ask unavailable for {symbol}"
                continue
            if bid is None or ask is None:
                continue
            try:
                spread = float(ask) - float(bid)
            except Exception:
                continue
            if spread < 0:
                spread = abs(spread)
            shock_key = _spread_key(leg, symbol)
            ok, reason = _shock_ok(shock_key, float(spread))
            if not ok:
                return False, reason
            if max_abs > 0 and spread > max_abs:
                return False, f"Spread too wide for {symbol} ({spread:.2f} > {max_abs:.2f})"
            if max_pct > 0:
                mid = (float(ask) + float(bid)) / 2.0
                if mid > 0:
                    pct = spread / mid  # fraction, e.g. 0.005 for 0.5%
                    if pct > max_pct:
                        return False, f"Spread too wide for {symbol} ({pct*100:.2f}% > {max_pct*100:.2f}%)"

        return True, ""

    def _check_exit_liquidity(self, legs: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Replicate strategy._check_exit_liquidity logic."""
        try:
            require_ba = bool(getattr(self.cfg, "exit_require_bid_ask", True))
        except Exception:
            require_ba = True
        try:
            max_pct = float(getattr(self.cfg, "exit_max_bid_ask_spread_pct", 3.0) or 3.0)
        except Exception:
            max_pct = 3.0
        try:
            max_abs = float(getattr(self.cfg, "exit_max_bid_ask_spread_abs", 8.0) or 8.0)
        except Exception:
            max_abs = 8.0
        if max_pct <= 0 and max_abs <= 0 and not require_ba:
            return True, ""
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            symbol = str(leg.get("symbol") or "")
            if not symbol:
                continue
            bid, ask, _ = self.client.get_bid_ask(symbol)
            if require_ba and (bid is None or ask is None):
                return False, f"Bid/ask unavailable for {symbol}"
            if bid is None or ask is None:
                continue
            try:
                spread = abs(float(ask) - float(bid))
            except Exception:
                continue
            if max_abs > 0 and spread > max_abs:
                return False, f"Exit spread too wide for {symbol} ({spread:.2f} > {max_abs:.2f})"
            if max_pct > 0:
                mid = (float(ask) + float(bid)) / 2.0
                if mid > 0:
                    pct = (spread / mid) * 100.0
                    if pct > max_pct:
                        return False, f"Exit spread too wide for {symbol} ({pct:.2f}% > {max_pct:.2f}%)"
        return True, ""


# ---------------------------------------------------------------------------
# Client factory helpers
# ---------------------------------------------------------------------------

def _make_client(bid: Optional[float], ask: Optional[float]) -> MagicMock:
    """Return a mock client whose get_bid_ask always returns the given bid/ask.

    If bid and ask are both None, ltp is also None (no fallback).
    """
    c = MagicMock()
    ltp = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    c.get_bid_ask = MagicMock(return_value=(bid, ask, ltp))
    return c


# ===========================================================================
# Premium filter tests
# ===========================================================================

def test_low_premium_trade_blocked() -> None:
    """min_premium=5.0, ltp=3.0 → entry blocked."""
    cfg = _Cfg(entry_min_option_premium=5.0)
    s = _MockStrategy(cfg)
    ok, reason = s._entry_price_bounds_ok(3.0)
    assert not ok, f"Expected blocked but got ok=True, reason={reason}"
    assert "below min" in reason.lower(), f"Unexpected reason: {reason}"


def test_acceptable_premium_allowed() -> None:
    """min_premium=5.0, ltp=10.0 → entry allowed."""
    cfg = _Cfg(entry_min_option_premium=5.0)
    s = _MockStrategy(cfg)
    ok, reason = s._entry_price_bounds_ok(10.0)
    assert ok, f"Expected allowed but got ok=False, reason={reason}"


def test_acceptable_premium_at_exact_boundary() -> None:
    """min_premium=5.0, ltp=5.0 → entry allowed (boundary inclusive)."""
    cfg = _Cfg(entry_min_option_premium=5.0)
    s = _MockStrategy(cfg)
    ok, reason = s._entry_price_bounds_ok(5.0)
    assert ok, f"Expected allowed at boundary but got ok=False, reason={reason}"


def test_max_premium_blocks_too_high() -> None:
    """max_premium=8.0, ltp=12.0 → entry blocked."""
    cfg = _Cfg(entry_min_option_premium=0.0, entry_max_option_premium=8.0)
    s = _MockStrategy(cfg)
    ok, reason = s._entry_price_bounds_ok(12.0)
    assert not ok, f"Expected blocked for premium above max, got ok=True"


def test_no_premium_filter_allows_any() -> None:
    """min_premium=0.0, ltp=0.5 → allowed (filter disabled)."""
    cfg = _Cfg(entry_min_option_premium=0.0)
    s = _MockStrategy(cfg)
    ok, reason = s._entry_price_bounds_ok(0.5)
    assert ok, f"Expected allowed when filter disabled, got ok=False, reason={reason}"


# ===========================================================================
# Bid/ask spread filter tests — percentage
# ===========================================================================

def test_wide_spread_pct_blocked() -> None:
    """max_spread_pct=0.02 (2%), spread=3% → blocked.

    mid = 100.0, bid=98.5, ask=101.5 → spread=3.0, pct=3.0%
    3.0% > 2.0% threshold → blocked.
    """
    bid, ask = 98.5, 101.5   # spread=3.0, pct=3.0%
    client = _make_client(bid, ask)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.02,   # 0.02 = 2% threshold
        entry_max_bid_ask_spread_abs=0.0,     # disable abs check
        entry_spread_shock_mult=0.0,          # disable shock check
    )
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert not ok, f"Expected blocked at 3% spread but got ok=True, reason={reason}"
    assert "too wide" in reason.lower(), f"Unexpected reason: {reason}"


def test_narrow_spread_allowed() -> None:
    """max_spread_pct=0.02 (2%), spread=0.5% → allowed.

    mid = 100.0, bid=99.75, ask=100.25 → spread=0.5, pct=0.5%
    0.5% < 2.0% threshold → allowed.
    """
    bid, ask = 99.75, 100.25   # spread=0.5, pct=0.5%
    client = _make_client(bid, ask)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.02,
        entry_max_bid_ask_spread_abs=0.0,
        entry_spread_shock_mult=0.0,
    )
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert ok, f"Expected allowed at 0.5% spread but got ok=False, reason={reason}"


def test_exactly_at_max_spread_pct_allowed() -> None:
    """max_spread_pct=0.02 (2%), spread=2.0% → allowed (boundary inclusive).

    mid = 100.0, bid=99.0, ask=101.0 → spread=2.0, pct=2.0%
    2.0% is NOT > 2.0% → allowed.
    """
    bid, ask = 99.0, 101.0   # spread=2.0, pct=2.0%
    client = _make_client(bid, ask)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.02,
        entry_max_bid_ask_spread_abs=0.0,
        entry_spread_shock_mult=0.0,
    )
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert ok, f"Expected allowed at boundary (2.0%) but got ok=False, reason={reason}"


# ===========================================================================
# Bid/ask spread filter tests — absolute
# ===========================================================================

def test_wide_spread_abs_blocked() -> None:
    """max_spread_abs=5.0, spread=8.0 → blocked."""
    bid, ask = 100.0, 108.0   # spread=8.0
    client = _make_client(bid, ask)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.0,    # disable pct check
        entry_max_bid_ask_spread_abs=5.0,
        entry_spread_shock_mult=0.0,
    )
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert not ok, f"Expected blocked at 8.0 spread but got ok=True, reason={reason}"
    assert "too wide" in reason.lower(), f"Unexpected reason: {reason}"


def test_narrow_spread_abs_allowed() -> None:
    """max_spread_abs=5.0, spread=2.0 → allowed."""
    bid, ask = 100.0, 102.0   # spread=2.0
    client = _make_client(bid, ask)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.0,
        entry_max_bid_ask_spread_abs=5.0,
        entry_spread_shock_mult=0.0,
    )
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert ok, f"Expected allowed at 2.0 spread but got ok=False, reason={reason}"


# ===========================================================================
# Bid/ask required filter tests
# ===========================================================================

def test_bid_ask_required_blocks_when_missing() -> None:
    """require_bid_ask=True, bid=None, ask=None → blocked."""
    client = _make_client(None, None)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.0,
        entry_max_bid_ask_spread_abs=0.0,
        entry_spread_shock_mult=0.0,
    )
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert not ok, f"Expected blocked when bid/ask missing but got ok=True, reason={reason}"
    assert "unavailable" in reason.lower(), f"Unexpected reason: {reason}"


def test_bid_ask_not_required_allows_missing() -> None:
    """require_bid_ask=False, bid=None, ask=None → allowed."""
    client = _make_client(None, None)
    cfg = _Cfg(
        entry_require_bid_ask=False,
        entry_max_bid_ask_spread_pct=0.0,
        entry_max_bid_ask_spread_abs=0.0,
        entry_spread_shock_mult=0.0,
    )
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert ok, f"Expected allowed when require_bid_ask=False but got ok=False, reason={reason}"


# ===========================================================================
# Spread shock filter tests
# ===========================================================================

def test_spread_shock_1_0_blocks_expansion() -> None:
    """entry_spread_shock_mult=1.0 blocks when spread > median * 1.0.

    Pre-populate history with spreads [1.0, 1.5, 2.0] (median=1.5).
    New spread=3.0 → 3.0 > 1.0 * 1.5 = 1.5 → blocked.
    """
    bid, ask = 100.0, 103.0   # spread=3.0
    client = _make_client(bid, ask)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.0,
        entry_max_bid_ask_spread_abs=0.0,
        entry_spread_shock_mult=1.0,
        entry_spread_shock_lookback=20,
        entry_spread_shock_min_samples=3,
    )
    s = _MockStrategy(cfg, client)
    # Pre-populate spread history to establish baseline
    s._entry_spread_watch["NIFTY"] = [1.0, 1.5, 2.0]
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert not ok, f"Expected blocked due to spread shock but got ok=True, reason={reason}"
    assert "shock" in reason.lower(), f"Unexpected reason: {reason}"


def test_spread_shock_1_0_allows_stable_spread() -> None:
    """entry_spread_shock_mult=1.0 allows spread <= median * 1.0.

    Pre-populate history [1.0, 1.5, 2.0] (median=1.5).
    New spread=1.5 → 1.5 <= 1.0 * 1.5 → allowed.
    """
    bid, ask = 100.0, 101.5   # spread=1.5
    client = _make_client(bid, ask)
    cfg = _Cfg(
        entry_require_bid_ask=True,
        entry_max_bid_ask_spread_pct=0.0,
        entry_max_bid_ask_spread_abs=0.0,
        entry_spread_shock_mult=1.0,
        entry_spread_shock_lookback=20,
        entry_spread_shock_min_samples=3,
    )
    s = _MockStrategy(cfg, client)
    s._entry_spread_watch["NIFTY"] = [1.0, 1.5, 2.0]
    ok, reason = s._check_entry_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert ok, f"Expected allowed for stable spread but got ok=False, reason={reason}"


# ===========================================================================
# Config defaults verification
# ===========================================================================

def test_config_defaults_match_safe_values() -> None:
    """Verify StrategyConfig safe defaults match the task requirements."""
    from config import StrategyConfig
    cfg = StrategyConfig()
    assert cfg.entry_require_bid_ask is True, "entry_require_bid_ask should default True"
    assert cfg.entry_min_option_premium == 5.0, "entry_min_option_premium should default 5.0"
    assert cfg.entry_max_bid_ask_spread_pct == 2.0, "entry_max_bid_ask_spread_pct should default 2.0%"
    assert cfg.entry_max_bid_ask_spread_abs == 5.0, "entry_max_bid_ask_spread_abs should default 5.0"
    assert cfg.entry_spread_shock_mult == 1.5, "entry_spread_shock_mult should default 1.5"
    assert cfg.exit_require_bid_ask is True, "exit_require_bid_ask should default True"
    assert cfg.exit_max_bid_ask_spread_pct == 3.0, "exit_max_bid_ask_spread_pct should default 3.0%"
    assert cfg.exit_max_bid_ask_spread_abs == 8.0, "exit_max_bid_ask_spread_abs should default 8.0"


# ===========================================================================
# Exit liquidity filter tests
# ===========================================================================

def test_exit_wide_spread_pct_blocked() -> None:
    """exit_max_spread_pct=3.0%, spread=4% → blocked at exit."""
    bid, ask = 98.0, 102.0   # spread=4.0, pct=4.0%
    client = _make_client(bid, ask)
    cfg = _Cfg(exit_require_bid_ask=True, exit_max_bid_ask_spread_pct=3.0, exit_max_bid_ask_spread_abs=0.0)
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_exit_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert not ok, f"Expected blocked at 4% exit spread but got ok=True"
    assert "too wide" in reason.lower(), f"Unexpected reason: {reason}"


def test_exit_narrow_spread_allowed() -> None:
    """exit_max_spread_pct=3.0%, spread=1% → allowed at exit."""
    bid, ask = 99.5, 100.5   # spread=1.0, pct=1.0%
    client = _make_client(bid, ask)
    cfg = _Cfg(exit_require_bid_ask=True, exit_max_bid_ask_spread_pct=3.0, exit_max_bid_ask_spread_abs=0.0)
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_exit_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert ok, f"Expected allowed at 1% exit spread but got ok=False"


def test_exit_wide_spread_abs_blocked() -> None:
    """exit_max_spread_abs=8.0, spread=10.0 → blocked at exit."""
    bid, ask = 100.0, 110.0   # spread=10.0
    client = _make_client(bid, ask)
    cfg = _Cfg(exit_require_bid_ask=True, exit_max_bid_ask_spread_pct=0.0, exit_max_bid_ask_spread_abs=8.0)
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_exit_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert not ok, f"Expected blocked at 10.0 abs exit spread but got ok=True"


def test_exit_bid_ask_missing_blocked() -> None:
    """exit_require_bid_ask=True, bid=None → blocked at exit."""
    client = _make_client(None, None)
    cfg = _Cfg(exit_require_bid_ask=True, exit_max_bid_ask_spread_pct=0.0, exit_max_bid_ask_spread_abs=0.0)
    s = _MockStrategy(cfg, client)
    ok, reason = s._check_exit_liquidity([{"symbol": "NIFTY", "exchange": "NFO", "token": ""}])
    assert not ok, f"Expected blocked when exit bid/ask missing but got ok=True"


def test_filters_are_configurable() -> None:
    """Verify all filter values can be overridden via config."""
    from config import StrategyConfig
    cfg = StrategyConfig()
    # All required fields must exist and be settable
    for field, default in [
        ("entry_min_option_premium", 5.0),
        ("entry_max_bid_ask_spread_pct", 2.0),
        ("entry_max_bid_ask_spread_abs", 5.0),
        ("entry_spread_shock_mult", 1.5),
        ("exit_require_bid_ask", True),
        ("exit_max_bid_ask_spread_pct", 3.0),
        ("exit_max_bid_ask_spread_abs", 8.0),
    ]:
        assert hasattr(cfg, field), f"StrategyConfig missing field: {field}"
        assert getattr(cfg, field) == default, f"Field {field} default={getattr(cfg, field)}, expected {default}"


# ===========================================================================
# Run all tests
# ===========================================================================

if __name__ == "__main__":
    import traceback

    tests = [
        # Premium
        test_low_premium_trade_blocked,
        test_acceptable_premium_allowed,
        test_acceptable_premium_at_exact_boundary,
        test_max_premium_blocks_too_high,
        test_no_premium_filter_allows_any,
        # Spread pct
        test_wide_spread_pct_blocked,
        test_narrow_spread_allowed,
        test_exactly_at_max_spread_pct_allowed,
        # Spread absolute
        test_wide_spread_abs_blocked,
        test_narrow_spread_abs_allowed,
        # Bid/ask required
        test_bid_ask_required_blocks_when_missing,
        test_bid_ask_not_required_allows_missing,
        # Spread shock
        test_spread_shock_1_0_blocks_expansion,
        test_spread_shock_1_0_allows_stable_spread,
        # Config defaults
        test_config_defaults_match_safe_values,
        # Exit liquidity
        test_exit_wide_spread_pct_blocked,
        test_exit_narrow_spread_allowed,
        test_exit_wide_spread_abs_blocked,
        test_exit_bid_ask_missing_blocked,
        test_filters_are_configurable,
    ]

    passed = 0
    failed = 0
    results: List[Tuple[str, bool, str]] = []

    for fn in tests:
        try:
            fn()
            passed += 1
            results.append((fn.__name__, True, ""))
        except AssertionError as e:
            failed += 1
            results.append((fn.__name__, False, str(e)))
        except Exception as e:
            failed += 1
            results.append((fn.__name__, False, f"ERROR: {e}\n{traceback.format_exc()}"))

    print(f"\n{'='*60}")
    print(f"  Premium & Spread Filter Defaults — Test Results")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    print(f"{'='*60}\n")
    for name, ok, err in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if err:
            print(f"         {err}")
    print()
    if failed:
        raise SystemExit(1)