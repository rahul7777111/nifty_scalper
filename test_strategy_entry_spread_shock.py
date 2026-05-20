from __future__ import annotations

from src.config import StrategyConfig
from src.strategy import NiftyScalper


class QuoteClient:
    def __init__(self, spreads: list[float]) -> None:
        self.spreads = list(spreads)
        self.calls = 0

    def get_bid_ask(self, *_args, **_kwargs):
        spread = self.spreads[min(self.calls, len(self.spreads) - 1)]
        self.calls += 1
        return 100.0, 100.0 + float(spread), 100.0


def _bot_for_liquidity(cfg: StrategyConfig, client: object) -> NiftyScalper:
    bot = NiftyScalper.__new__(NiftyScalper)
    bot.cfg = cfg
    bot.client = client
    bot._entry_spread_watch = {}
    return bot


def test_entry_spread_shock_blocks_wide_spread_after_history() -> None:
    cfg = StrategyConfig(
        entry_spread_shock_mult=2.0,
        entry_spread_shock_lookback=5,
        entry_spread_shock_min_samples=3,
    )
    client = QuoteClient([1.0, 1.1, 0.9, 3.0])
    bot = _bot_for_liquidity(cfg, client)
    leg = {"symbol": "NIFTYTESTCE", "exchange": "NFO", "token": "123"}

    assert bot._check_entry_liquidity([leg]) == (True, "")
    assert bot._check_entry_liquidity([leg]) == (True, "")
    assert bot._check_entry_liquidity([leg]) == (True, "")

    ok, reason = bot._check_entry_liquidity([leg])
    assert ok is False
    assert "Spread shock" in reason


def test_entry_spread_shock_disabled_keeps_existing_fast_path() -> None:
    class NoQuoteClient:
        def get_bid_ask(self, *_args, **_kwargs):  # pragma: no cover - should not be called
            raise AssertionError("quote should not be fetched")

    cfg = StrategyConfig(entry_spread_shock_mult=0.0)
    bot = _bot_for_liquidity(cfg, NoQuoteClient())

    assert bot._check_entry_liquidity([{"symbol": "NIFTYTESTCE"}]) == (True, "")
