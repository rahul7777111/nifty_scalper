"""Minimal backtest harness for strategy smoke-testing.

This is a deliberately small simulator to validate wiring of enhancements.
It is NOT a production-quality backtester.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple


def simulate_simple(price_series: Iterable[float], signals: Iterable[int], position_size: int = 1) -> Tuple[float, List[float]]:
    """Simulate simple long-only trades where `signals` is 1 for long, 0 for flat.

    Entry/exit occurs on signal transitions. Returns total PnL and list of trade PnLs.
    """
    prices = list(price_series)
    sigs = list(signals)
    if not prices or not sigs or len(prices) != len(sigs):
        return 0.0, []
    in_trade = False
    entry_price = 0.0
    trade_pnls: List[float] = []
    for p, s in zip(prices, sigs):
        if not in_trade and s:
            in_trade = True
            entry_price = float(p)
        elif in_trade and not s:
            pnl = (float(p) - entry_price) * position_size
            trade_pnls.append(pnl)
            in_trade = False
    # close at end
    if in_trade:
        pnl = (float(prices[-1]) - entry_price) * position_size
        trade_pnls.append(pnl)
    return float(sum(trade_pnls)), trade_pnls
