#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_retrainer.py
==============================================
Test the core retrain loop functionality of ml_only_dynamic_preset_retrainer.

Covers: load_and_audit_dataset, get_live_features, walk_forward_splits,
build_preset_grid, evaluate_candidate_preset, evaluate_router_candidate.
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
    MIN_TRADES,
    MIN_SHARPE,
    build_preset_grid,
    evaluate_candidate_preset,
    evaluate_router_candidate,
    get_live_features,
    load_and_audit_dataset,
    walk_forward_splits,
)

# Real dataset path
REAL_DATASET = REPO_ROOT / "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"


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
    df, _ = real_df_and_audit
    return get_live_features(df)


@pytest.fixture(scope="module")
def pe_subset(real_df_and_audit):
    """Return a PE-only subset of the real dataset for faster testing."""
    df, _ = real_df_and_audit
    pe_df = df[df.get('option_type', '').astype(str).str.upper() == 'PE'].copy()
    # Take first 50000 rows of PE for speed
    return pe_df.head(50000)


# ─── load_and_audit_dataset ──────────────────────────────────────────────────

@pytest.mark.real_dataset
def test_load_and_audit_dataset_returns_correct_structure(real_df_and_audit):
    """Audit dict must have expected keys after loading real dataset."""
    df, audit = real_df_and_audit

    # Structural keys
    assert 'rows' in audit, "audit must contain 'rows'"
    assert 'cols' in audit, "audit must contain 'cols'"
    assert 'label_cols' in audit, "audit must contain 'label_cols'"
    assert 'timestamp_range' in audit, "audit must contain 'timestamp_range'"
    assert 'ce_count' in audit, "audit must contain 'ce_count'"
    assert 'pe_count' in audit, "audit must contain 'pe_count'"

    # Values make sense
    assert audit['rows'] >= 400_000, f"Expected ~405k rows, got {audit['rows']}"
    assert audit['cols'] >= 100, f"Expected ~183 cols, got {audit['cols']}"
    assert isinstance(audit['label_cols'], list), "label_cols must be a list"
    assert audit['ce_count'] > 0, "ce_count must be positive"
    assert audit['pe_count'] > 0, "pe_count must be positive"
    assert 'start' in audit['timestamp_range'], "timestamp_range must have 'start'"
    assert 'end' in audit['timestamp_range'], "timestamp_range must have 'end'"


# ─── get_live_features ───────────────────────────────────────────────────────

@pytest.mark.real_dataset
def test_get_live_features_excludes_forbidden_columns(real_features):
    """Feature list must NOT contain any forbidden tokens."""
    forbidden_tokens = [
        'return', 'forward', 'future', 'pnl', 'profit', 'loss',
        'label', 'target', 'outcome', 'exit', 'mfe', 'mae',
        'realized', 'net_', 'gross_', 'expected_', 'horizon_',
        'cost_survivor', 'strong_profitable', 'high_conviction',
        'paper_candidate', 'avoid_trade', 'weak_trade', 'no_trade',
        'trade_outcome', 'realized_after', 'cost_adjusted',
    ]
    features = real_features

    for feat in features:
        feat_lower = feat.lower()
        for token in forbidden_tokens:
            assert token not in feat_lower, (
                f"Feature '{feat}' contains forbidden token '{token}'"
            )

    # Exact-match exclusions
    exact_forbidden = {
        'net_forward_return', 'gross_forward_return', 'profitable_trade_label',
        'cost_survivor_label', 'cost_survivor_label_v2', 'strong_profitable_trade_label',
        'high_conviction_trade_label', 'paper_candidate_label', 'avoid_trade_label',
        'weak_trade_label', 'no_trade_label', 'mfe', 'mae', 'realized_pnl',
        'expected_return_after_cost', 'return_to_cost_ratio',
    }
    for feat in features:
        assert feat not in exact_forbidden, f"Feature '{feat}' is in exact-forbidden set"


# ─── walk_forward_splits ─────────────────────────────────────────────────────

@pytest.mark.real_dataset
def test_walk_forward_splits_are_time_ordered(real_df_and_audit):
    """Each fold's test start must be >= previous fold's test start (chronological)."""
    df, _ = real_df_and_audit
    splits = walk_forward_splits(df, n_folds=3)

    assert len(splits) >= 1, "Should produce at least 1 split"
    prev_test_start = None
    for i, (train, val, test) in enumerate(splits):
        # Get the timestamp_dt min of each split
        test_min_ts = test['timestamp_dt'].min()
        if prev_test_start is not None:
            assert test_min_ts >= prev_test_start, (
                f"Fold {i} test start ({test_min_ts}) is before fold {i-1} test start ({prev_test_start})"
            )
        prev_test_start = test_min_ts


@pytest.mark.real_dataset
def test_walk_forward_splits_respecting_minimum_size(real_df_and_audit):
    """Each split must have train >= 100, val >= 20, test >= 20 rows."""
    df, _ = real_df_and_audit
    splits = walk_forward_splits(df, n_folds=3)

    for i, (train, val, test) in enumerate(splits):
        assert len(train) >= 100, f"Fold {i}: train has {len(train)} rows, need >= 100"
        assert len(val) >= 20, f"Fold {i}: val has {len(val)} rows, need >= 20"
        assert len(test) >= 20, f"Fold {i}: test has {len(test)} rows, need >= 20"


# ─── build_preset_grid ───────────────────────────────────────────────────────

def test_build_preset_grid_returns_dynamic_preset_list():
    """build_preset_grid must return a list of DynamicPreset objects."""
    presets = build_preset_grid(max_presets=100)  # use 100 to ensure all families appear
    assert isinstance(presets, list), "build_preset_grid must return a list"
    assert all(isinstance(p, DynamicPreset) for p in presets), (
        "All items must be DynamicPreset instances"
    )

    # Check expected families are present (A+B+C+D fit in 50 slots; need 100 for all 6)
    families = {p.preset_family for p in presets}
    expected_families = {
        'A_conservative', 'B_balanced', 'C_low_cost',
        'D_dte', 'E_volatility', 'F_router',
    }
    assert expected_families.issubset(families), (
        f"Missing families: {expected_families - families}"
    )


# ─── evaluate_candidate_preset ───────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_evaluate_candidate_preset_returns_complete_dict(pe_subset, real_features):
    """evaluate_candidate_preset must return a dict with all expected keys."""
    # Use a balanced preset (should produce some trades)
    preset = DynamicPreset(
        preset_id='test_balanced_t30_n2',
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

    # Check top-level keys
    assert 'status' in result, "result must have 'status'"
    assert 'gates_passed' in result, "result must have 'gates_passed'"
    assert 'gates_total' in result, "result must have 'gates_total'"
    assert 'trade_count' in result, "result must have 'trade_count'"
    assert 'net_pf' in result, "result must have 'net_pf'"
    assert 'gross_pf' in result, "result must have 'gross_pf'"
    assert 'cost_stress' in result, "result must have 'cost_stress'"
    assert 'gates' in result, "result must have 'gates'"
    assert 'sharpe' in result, "result must have 'sharpe'"
    assert 'max_drawdown' in result, "result must have 'max_drawdown'"
    assert 'top_day_concentration' in result, "result must have 'top_day_concentration'"
    assert 'fold_details' in result, "result must have 'fold_details'"
    assert 'preset' in result, "result must have 'preset'"
    assert 'n_folds' in result, "result must have 'n_folds'"

    # cost_stress must have 4 multipliers
    cs = result['cost_stress']
    assert 'pf_at_1.0x' in cs, "cost_stress must have pf_at_1.0x"
    assert 'pf_at_1.25x' in cs, "cost_stress must have pf_at_1.25x"
    assert 'pf_at_1.5x' in cs, "cost_stress must have pf_at_1.5x"
    assert 'pf_at_2.0x' in cs, "cost_stress must have pf_at_2.0x"


@pytest.mark.slow
@pytest.mark.real_dataset
def test_evaluate_candidate_preset_handles_no_trades_gracefully(pe_subset, real_features):
    """Preset so strict it produces 0 trades must return status='NO_TRADES'."""
    # Very restrictive preset that should produce zero trades
    preset = DynamicPreset(
        preset_id='test_strict_t99_n1',
        preset_family='A_conservative',
        threshold=0.99,  # Extremely high threshold
        max_trades_per_day=1,
        top_n_confidence_per_day=1,
        spread_limit_pct=0.01,  # Very tight spread
        liquidity_min=1_000_000_000.0,  # Unrealistic liquidity
        premium_band='high',
        dte_range='0-1',
    )

    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    assert result.get('status') == 'NO_TRADES', (
        f"Strict preset should return NO_TRADES, got: {result.get('status')}"
    )


@pytest.mark.slow
@pytest.mark.real_dataset
def test_candidate_results_sortable_by_1_5x_pf(pe_subset, real_features):
    """Evaluate 2 presets, sort by cost_stress['pf_at_1.5x'], verify descending order."""
    preset_loose = DynamicPreset(
        preset_id='test_loose_t20_n3',
        preset_family='B_balanced',
        threshold=0.20,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        spread_limit_pct=0.15,
        liquidity_min=50000.0,
    )
    preset_tight = DynamicPreset(
        preset_id='test_tight_t50_n1',
        preset_family='A_conservative',
        threshold=0.50,
        max_trades_per_day=1,
        top_n_confidence_per_day=1,
        spread_limit_pct=0.10,
        liquidity_min=50000.0,
    )

    result_loose = evaluate_candidate_preset(
        df=pe_subset, features=real_features,
        label='cost_survivor_label_v2', model_type='elasticnet',
        preset=preset_loose, n_folds=2,
    )
    result_tight = evaluate_candidate_preset(
        df=pe_subset, features=real_features,
        label='cost_survivor_label_v2', model_type='elasticnet',
        preset=preset_tight, n_folds=2,
    )

    pf_loose = result_loose.get('cost_stress', {}).get('pf_at_1.5x', 0)
    pf_tight = result_tight.get('cost_stress', {}).get('pf_at_1.5x', 0)

    # Both should be finite numbers
    import math
    assert math.isfinite(pf_loose), f"pf_loose={pf_loose} is not finite"
    assert math.isfinite(pf_tight), f"pf_tight={pf_tight} is not finite"

    # Sorting by descending 1.5x PF should work
    results = [result_loose, result_tight]
    sorted_results = sorted(
        results,
        key=lambda r: r.get('cost_stress', {}).get('pf_at_1.5x', 0),
        reverse=True,
    )
    pfs_sorted = [r.get('cost_stress', {}).get('pf_at_1.5x', 0) for r in sorted_results]
    assert pfs_sorted[0] >= pfs_sorted[1], (
        f"Descending sort failed: {pfs_sorted}"
    )


# ─── evaluate_router_candidate ───────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_evaluate_router_candidate_returns_complete_dict(pe_subset, real_features):
    """evaluate_router_candidate must return dict with all expected keys."""
    preset = DynamicPreset(
        preset_id='test_router_t30_n2',
        preset_family='F_router',
        threshold=0.30,
        max_trades_per_day=2,
        top_n_confidence_per_day=2,
        spread_limit_pct=0.10,
        liquidity_min=50000.0,
    )

    result = evaluate_router_candidate(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=preset,
        n_folds=2,
    )

    # Complete dict check — same keys as evaluate_candidate_preset
    expected_keys = {
        'status', 'gates_passed', 'gates_total', 'trade_count',
        'net_pf', 'gross_pf', 'cost_stress', 'gates', 'sharpe',
        'max_drawdown', 'top_day_concentration', 'fold_details',
        'preset', 'n_folds',
    }
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )

    # cost_stress must have 4 multipliers
    cs = result['cost_stress']
    for mult in ['1.0x', '1.25x', '1.5x', '2.0x']:
        assert f'pf_at_{mult}' in cs, f"cost_stress missing pf_at_{mult}"

    # gates must have 9 gate keys
    gates = result.get('gates', {})
    assert len(gates) >= 9, f"Expected >= 9 gates, got {len(gates)}: {gates}"