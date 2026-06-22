from __future__ import annotations

from datetime import datetime, timedelta

from src.exit_optimizer import (
    TripleBarrierState,
    evaluate_triple_barrier_state,
    initialize_triple_barrier_state,
)


def test_initialize_triple_barrier_state_sets_absolute_levels() -> None:
    ts = datetime(2026, 6, 3, 9, 20, 0)
    state = initialize_triple_barrier_state(
        fill_price=100.0,
        prediction_id="PRED_TEST01",
        policy_name="trade_quality_binary",
        profit_target_pct=0.01,
        stop_loss_pct=0.005,
        max_duration_bars=18,
        bar_timestamp=ts,
    )
    assert state.prediction_id == "PRED_TEST01"
    assert state.upper_profit_barrier == 101.0
    assert state.lower_stop_barrier == 99.5
    assert state.elapsed_bars == 0
    assert state.last_bar_timestamp == ts.isoformat()


def test_evaluate_triple_barrier_state_triggers_price_and_horizon() -> None:
    ts = datetime(2026, 6, 3, 9, 20, 0)
    state = initialize_triple_barrier_state(
        fill_price=100.0,
        prediction_id="PRED_TEST02",
        policy_name="trade_quality_binary",
        profit_target_pct=0.01,
        stop_loss_pct=0.005,
        max_duration_bars=2,
        bar_timestamp=ts,
    )

    hold_eval = evaluate_triple_barrier_state(
        state,
        current_price=100.25,
        bar_timestamp=ts + timedelta(minutes=5),
        count_bar_close=True,
    )
    assert hold_eval.trigger_source is None
    assert state.elapsed_bars == 1

    horizon_eval = evaluate_triple_barrier_state(
        state,
        current_price=100.10,
        bar_timestamp=ts + timedelta(minutes=10),
        count_bar_close=True,
    )
    assert horizon_eval.trigger_source == "horizon_expiry"
    assert state.final_execution_duration_bars == 2

    profit_state = TripleBarrierState.from_dict(state.to_dict())
    assert profit_state is not None
    profit_state.elapsed_bars = 0
    profit_state.last_bar_timestamp = ts.isoformat()
    profit_eval = evaluate_triple_barrier_state(
        profit_state,
        current_price=101.05,
        bar_timestamp=ts,
        count_bar_close=False,
    )
    assert profit_eval.trigger_source == "profit_target"
