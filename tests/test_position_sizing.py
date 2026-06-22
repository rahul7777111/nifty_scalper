from __future__ import annotations

from position_sizing import calculate_position_size, calculate_probabilistic_bet_size


def test_calculate_probabilistic_bet_size_bounded_and_zero_below_threshold() -> None:
    assert calculate_probabilistic_bet_size(0.40, 0.50) == 0.0
    mid = calculate_probabilistic_bet_size(0.60, 0.50)
    high = calculate_probabilistic_bet_size(0.90, 0.50)
    assert 0.0 <= mid <= 1.0
    assert 0.0 <= high <= 1.0
    assert high >= mid


def test_calculate_position_size_applies_min_lot_and_drawdown_throttle() -> None:
    sized = calculate_position_size(
        baseline_lots=10,
        calibrated_prob=0.55,
        optimal_threshold=0.54,
        standard_error=0.10,
        current_drawdown_pct=0.06,
        max_allowable_drawdown=0.05,
        enforce_min_lot=True,
    )
    assert sized["baseline_vol_lots"] == 10
    assert 0.0 <= sized["ml_bet_multiplier"] <= 1.0
    assert 0.0 <= sized["drawdown_modifier"] <= 1.0
    assert sized["final_allocated_lots"] == 1
