#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_retraining_gates.py
=====================================================
Test the 9-gate threshold enforcement.

Covers: C1 positive expectancy, C2 pf_base >= 1.15, C3 pf_1.25x >= 1.05,
C4 pf_1.5x >= 1.00 (MANDATORY break-even, NOT weakened), C5 sharpe >= 0.75,
C6 trades >= 500, C7 profitable_folds >= n-1, C8 worst_fold_pf >= 0.90,
C9 daily stability <= 0.40, gate integrity (not weakened), all 9 present,
cost_stress has all 4 multipliers, passing candidate must pass all 9.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ml_only_dynamic_preset_retrainer import (
    DynamicPreset,
    PF_AT_1_00X,
    PF_AT_1_25X,
    PF_AT_1_50X,
    PF_AT_2_00X,
    MIN_TRADES,
    MIN_SHARPE,
    MAX_TOP_DAY_CONCENTRATION,
    evaluate_candidate_preset,
    load_and_audit_dataset,
)

REAL_DATASET = REPO_ROOT / "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"

# Expected gate keys
EXPECTED_GATE_KEYS = {
    'C1_positive_expectancy', 'C2_pf_base', 'C3_pf_1.25x', 'C4_pf_1.5x',
    'C5_sharpe', 'C6_trades', 'C7_profitable_folds', 'C8_worst_fold',
    'C9_daily_stability',
}


@pytest.fixture(scope="module")
def real_df_and_audit():
    """Load the real dataset once for the module."""
    if not REAL_DATASET.exists():
        pytest.skip(f"Real dataset not found at {REAL_DATASET}")
    df, audit = load_and_audit_dataset(REAL_DATASET)
    return df, audit


@pytest.fixture(scope="module")
def real_features(real_df_and_audit):
    """Extract live features from real dataset."""
    from ml_only_dynamic_preset_retrainer import get_live_features
    df, _ = real_df_and_audit
    return get_live_features(df)


@pytest.fixture(scope="module")
def pe_subset(real_df_and_audit):
    """Return a PE-only subset for faster testing."""
    df, _ = real_df_and_audit
    pe_df = df[df.get('option_type', '').astype(str).str.upper() == 'PE'].copy()
    return pe_df.head(50000)


# ─── Gate C1: Positive expectancy ───────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C1_positive_expectancy_requires_net_pf_positive(pe_subset, real_features):
    """avg_net_pf must be > 0 for C1_positive_expectancy to pass."""
    # Use a very loose preset to maximize trades
    preset = DynamicPreset(
        preset_id='test_C1_positive',
        preset_family='B_balanced',
        threshold=0.20,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    net_pf = result.get('net_pf', 0)
    c1_passed = gates.get('C1_positive_expectancy', False)

    if c1_passed:
        assert net_pf > 0, f"C1 passed but net_pf={net_pf} is not > 0"
    else:
        assert net_pf <= 0, f"C1 failed but net_pf={net_pf} is > 0"


# ─── Gate C2: PF base >= 1.15 ────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C2_pf_base_requires_gross_pf_above_1_15(pe_subset, real_features):
    """gross_pf (wins/losses ratio) >= 1.15 for C2_pf_base."""
    preset = DynamicPreset(
        preset_id='test_C2_pf_base',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    gross_pf = result.get('gross_pf', 0)
    c2_passed = gates.get('C2_pf_base', False)

    if c2_passed:
        assert gross_pf >= PF_AT_1_00X, (
            f"C2 passed but gross_pf={gross_pf} < {PF_AT_1_00X}"
        )


# ─── Gate C3: PF 1.25x >= 1.05 ──────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C3_pf_1_25x_requires_pf_above_1_05(pe_subset, real_features):
    """cost_stress['pf_at_1.25x'] >= 1.05 for C3_pf_1.25x."""
    preset = DynamicPreset(
        preset_id='test_C3_1_25x',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    cs = result.get('cost_stress', {})
    pf_1_25x = cs.get('pf_at_1.25x', 0)
    c3_passed = gates.get('C3_pf_1.25x', False)

    if c3_passed:
        assert pf_1_25x >= PF_AT_1_25X, (
            f"C3 passed but pf_at_1.25x={pf_1_25x} < {PF_AT_1_25X}"
        )


# ─── Gate C4: PF 1.5x >= 1.00 (MANDATORY break-even) ───────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C4_pf_1_5x_requires_pf_above_1_00(pe_subset, real_features):
    """cost_stress['pf_at_1.5x'] >= 1.00 for C4_pf_1.5x (MANDATORY break-even)."""
    preset = DynamicPreset(
        preset_id='test_C4_1_5x',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    cs = result.get('cost_stress', {})
    pf_1_5x = cs.get('pf_at_1.5x', 0)
    c4_passed = gates.get('C4_pf_1.5x', False)

    if c4_passed:
        assert pf_1_5x >= PF_AT_1_50X, (
            f"C4 passed but pf_at_1.5x={pf_1_5x} < {PF_AT_1_50X}"
        )


# ─── Gate C5: Sharpe >= 0.75 ─────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C5_sharpe_requires_sharpe_above_0_75(pe_subset, real_features):
    """avg_sharpe >= 0.75 for C5_sharpe."""
    preset = DynamicPreset(
        preset_id='test_C5_sharpe',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    sharpe = result.get('sharpe', 0)
    c5_passed = gates.get('C5_sharpe', False)

    if c5_passed:
        assert sharpe >= MIN_SHARPE, f"C5 passed but sharpe={sharpe} < {MIN_SHARPE}"


# ─── Gate C6: Trades >= 500 ─────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C6_trades_requires_min_500_trades(pe_subset, real_features):
    """trade_count >= 500 for C6_trades."""
    preset = DynamicPreset(
        preset_id='test_C6_trades',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    trade_count = result.get('trade_count', 0)
    c6_passed = gates.get('C6_trades', False)

    if c6_passed:
        assert trade_count >= MIN_TRADES, (
            f"C6 passed but trade_count={trade_count} < {MIN_TRADES}"
        )


# ─── Gate C7: Profitable folds >= n-1 ───────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C7_profitable_folds_requires_n_minus_1(pe_subset, real_features):
    """profitable_folds >= max(3, n_folds-1) for C7_profitable_folds."""
    preset = DynamicPreset(
        preset_id='test_C7_folds',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    n_folds = result.get('n_folds', 0)
    profitable_folds = result.get('profitable_folds', 0)
    required_min = max(3, n_folds - 1)
    c7_passed = gates.get('C7_profitable_folds', False)

    if c7_passed:
        assert profitable_folds >= required_min, (
            f"C7 passed but profitable_folds={profitable_folds} < {required_min}"
        )


# ─── Gate C8: Worst fold PF >= 0.90 ─────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C8_worst_fold_pf_requires_above_0_90(pe_subset, real_features):
    """worst_fold_pf >= 0.90 for C8_worst_fold."""
    preset = DynamicPreset(
        preset_id='test_C8_worst',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    worst_fold_pf = result.get('worst_fold_pf', 0)
    c8_passed = gates.get('C8_worst_fold', False)

    if c8_passed:
        assert worst_fold_pf >= 0.90, (
            f"C8 passed but worst_fold_pf={worst_fold_pf} < 0.90"
        )


# ─── Gate C9: Daily stability <= 0.40 ───────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_gate_C9_daily_stability_requires_top_day_below_0_40(pe_subset, real_features):
    """top_day_concentration <= 0.40 for C9_daily_stability."""
    preset = DynamicPreset(
        preset_id='test_C9_daily',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    top_day_conc = result.get('top_day_concentration', 0)
    c9_passed = gates.get('C9_daily_stability', False)

    if c9_passed:
        assert top_day_conc <= MAX_TOP_DAY_CONCENTRATION, (
            f"C9 passed but top_day_concentration={top_day_conc} > {MAX_TOP_DAY_CONCENTRATION}"
        )


# ─── Gate integrity ──────────────────────────────────────────────────────────

def test_gates_are_not_weakened():
    """PF_AT_1_50X must be exactly 1.00 (not weakened to 0.99 or lower)."""
    assert PF_AT_1_50X == 1.00, (
        f"PF_AT_1_50X={PF_AT_1_50X} is NOT 1.00. Gates have been weakened!"
    )
    # Also check it's not accidentally lowered
    assert PF_AT_1_50X >= 1.00, f"PF_AT_1_50X={PF_AT_1_50X} < 1.00 (weakened!)"


def test_all_9_gates_present_in_result(pe_subset, real_features):
    """Every evaluation result must have all 9 gate keys in gates dict."""
    preset = DynamicPreset(
        preset_id='test_gates_present_t30',
        preset_family='B_balanced',
        threshold=0.30,
        max_trades_per_day=2,
        top_n_confidence_per_day=2,
        spread_limit_pct=0.10,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates = result.get('gates', {})
    gate_keys = set(gates.keys())

    # SUSPICIOUS_PF may be present too (net_pf > 10)
    # But all 9 economic gates must be present
    missing = EXPECTED_GATE_KEYS - gate_keys
    assert len(missing) == 0, f"Missing gate keys: {missing}"


def test_cost_stress_has_all_4_multipliers(pe_subset, real_features):
    """cost_stress must have pf_at_1.0x, pf_at_1.25x, pf_at_1.5x, pf_at_2.0x."""
    preset = DynamicPreset(
        preset_id='test_cs_mults_t30',
        preset_family='B_balanced',
        threshold=0.30,
        max_trades_per_day=2,
        top_n_confidence_per_day=2,
        spread_limit_pct=0.10,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    cs = result.get('cost_stress', {})
    for mult_name, mult_val in [('1.0x', 1.0), ('1.25x', 1.25), ('1.5x', 1.5), ('2.0x', 2.0)]:
        key = f'pf_at_{mult_name}'
        assert key in cs, f"cost_stress missing '{key}'"
        # PF values should be finite numbers
        import math
        assert math.isfinite(cs[key]), f"cost_stress['{key}']={cs[key]} is not finite"


@pytest.mark.slow
@pytest.mark.real_dataset
def test_passing_candidate_must_pass_all_9_gates(pe_subset, real_features):
    """If gates_passed == 9, then every gate in gates dict must be True."""
    preset = DynamicPreset(
        preset_id='test_all9_pass_t25',
        preset_family='B_balanced',
        threshold=0.25,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    gates_passed = result.get('gates_passed', 0)
    gates = result.get('gates', {})

    if gates_passed == 9:
        # All 9 economic gates must be True
        for gate_key in EXPECTED_GATE_KEYS:
            assert gates.get(gate_key, False) is True, (
                f"gates_passed=9 but gate '{gate_key}' is False"
            )