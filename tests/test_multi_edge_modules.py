from __future__ import annotations

import os
import sys

SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from capital_allocator import allocate_capital_across_sleeves, build_sleeve_scores
from mean_reversion import evaluate_mean_reversion
from stat_arb import evaluate_pair_spread
from strategy_allocator import allocate_sleeves_for_regime, select_strategy_for_regime


def test_mean_reversion_engine_emits_buy_call_for_oversold_series():
    closes = [100.0] * 18 + [97.0, 95.0]
    sig = evaluate_mean_reversion(closes, lookback=20, entry_zscore=1.0)
    assert sig.signal == "buy_call"
    assert sig.recommended_option == "CE"
    assert sig.zscore < 0


def test_stat_arb_module_detects_wide_positive_spread():
    x = [100 + i * 0.2 for i in range(40)]
    y = [100 + i * 0.2 + (0.5 if i < 39 else 5.0) for i in range(40)]
    sig = evaluate_pair_spread(x, y, lookback=30, entry_zscore=1.2)
    assert sig.signal == "short_spread"
    assert sig.zscore > 0


def test_capital_allocator_respects_caps_and_reserve():
    scores = build_sleeve_scores(
        regime="trending",
        trend_score=0.9,
        mean_reversion_score=0.4,
        stat_arb_score=0.3,
    )
    allocations = allocate_capital_across_sleeves(
        account_capital=100000.0,
        sleeve_scores=scores,
        max_single_sleeve_weight=0.50,
        reserve_cash_weight=0.10,
    )
    assert allocations["trend"].weight <= 0.50
    total_weight = sum(row.weight for row in allocations.values())
    assert total_weight <= 0.90 + 1e-9


def test_strategy_allocator_exposes_multi_edge_sleeve_selection():
    routed = allocate_sleeves_for_regime(
        "mean_reverting",
        account_capital=100000.0,
        trend_score=0.3,
        mean_reversion_score=0.9,
        stat_arb_score=0.4,
    )
    assert routed["selected_sleeve"] == "mean_reversion"
    assert routed["selected_strategy"] == "mean_reversion"
    assert select_strategy_for_regime("quiet") == "stat_arb"
