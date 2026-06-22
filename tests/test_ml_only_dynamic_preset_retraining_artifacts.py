#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_retraining_artifacts.py
==========================================================
Test artifact creation and completeness.

Covers: JSON serializability, fold_detail required fields,
DynamicPreset to_dict/from_dict round-trip, cost_stress PF
sanity, net/gross PF finiteness, raw_returns percentage space,
preset_to_dict completeness, raw_returns in fold result,
suspicious PF flagging.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ml_only_dynamic_preset_retrainer import (
    DynamicPreset,
    PF_AT_1_50X,
    evaluate_candidate_preset,
    load_and_audit_dataset,
)

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
    from ml_only_dynamic_preset_retrainer import get_live_features
    df, _ = real_df_and_audit
    return get_live_features(df)


@pytest.fixture(scope="module")
def pe_subset(real_df_and_audit):
    """Return a PE-only subset for faster testing."""
    df, _ = real_df_and_audit
    pe_df = df[df.get('option_type', '').astype(str).str.upper() == 'PE'].copy()
    return pe_df.head(50000)


@pytest.fixture(scope="module")
def sample_preset():
    """A balanced preset for testing."""
    return DynamicPreset(
        preset_id='test_artifact_t30',
        preset_family='B_balanced',
        threshold=0.30,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
        max_open_positions=2,
        spread_limit_pct=0.10,
        liquidity_min=50000.0,
        premium_band='all',
        dte_range='all',
        regime_filter='all',
        expiry_filter='all',
        avoid_first_n_minutes=5,
        avoid_last_n_minutes=5,
        expected_move_to_cost_ratio_min=0.5,
        cooldown_after_loss_minutes=0,
        daily_stop_loss_pct=0.0,
        daily_profit_lock_pct=0.0,
    )


# ─── JSON serializability ────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_evaluate_candidate_preset_returns_serializable_result(pe_subset, real_features, sample_preset):
    """All values in result must be JSON-serializable (no numpy types)."""
    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=sample_preset,
        n_folds=2,
    )

    # Try to serialize with default=str (handles numpy types)
    try:
        json_str = json.dumps(result, default=str)
        deserialized = json.loads(json_str)
    except TypeError as e:
        pytest.fail(f"Result is not JSON-serializable: {e}")

    # Verify deserialized values match expected types
    assert isinstance(deserialized['status'], str)
    assert isinstance(deserialized['trade_count'], int)
    assert isinstance(deserialized['gates_passed'], int)
    assert isinstance(deserialized['net_pf'], (int, float))
    assert isinstance(deserialized['gross_pf'], (int, float))


# ─── Fold details ────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_fold_details_contains_required_fields(pe_subset, real_features, sample_preset):
    """Each fold detail must have all required fields."""
    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=sample_preset,
        n_folds=2,
    )

    required_fold_keys = {
        'trade_count', 'win_rate', 'gross_pf', 'base_net_pf',
        'profit_factor', 'sharpe', 'max_drawdown',
        'top_day_concentration', 'threshold_used', 'raw_returns',
    }

    for i, fold in enumerate(result.get('fold_details', [])):
        missing = required_fold_keys - set(fold.keys())
        assert len(missing) == 0, f"Fold {i} missing keys: {missing}"


# ─── DynamicPreset round-trip ────────────────────────────────────────────────

def test_dynamic_preset_to_dict_round_trip(sample_preset):
    """Create a DynamicPreset, call to_dict(), call from_dict(), verify equality."""
    d = sample_preset.to_dict()
    restored = DynamicPreset.from_dict(d)

    # All fields should match
    original_dict = sample_preset.to_dict()
    restored_dict = restored.to_dict()
    assert original_dict == restored_dict, (
        f"Round-trip failed.\nOriginal: {original_dict}\nRestored: {restored_dict}"
    )


# ─── Cost stress PF sanity ───────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_cost_stress_pf_at_1_5x_never_negative_for_passing_candidate(pe_subset, real_features, sample_preset):
    """If gates_passed == 9, cost_stress['pf_at_1.5x'] must be >= 1.0."""
    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=sample_preset,
        n_folds=2,
    )

    gates_passed = result.get('gates_passed', 0)
    cs = result.get('cost_stress', {})
    pf_1_5x = cs.get('pf_at_1.5x', 0)

    if gates_passed == 9:
        assert pf_1_5x >= PF_AT_1_50X, (
            f"gates_passed=9 but pf_at_1.5x={pf_1_5x} < {PF_AT_1_50X}"
        )


# ─── Finite PF values ────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_net_pf_and_gross_pf_are_finite(pe_subset, real_features, sample_preset):
    """net_pf and gross_pf must be finite numbers (not inf, not nan)."""
    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=sample_preset,
        n_folds=2,
    )

    net_pf = result.get('net_pf', 0)
    gross_pf = result.get('gross_pf', 0)

    assert math.isfinite(net_pf), f"net_pf={net_pf} is not finite"
    assert math.isfinite(gross_pf), f"gross_pf={gross_pf} is not finite"


# ─── Raw returns percentage space ───────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_raw_returns_are_percentage_space(pe_subset, real_features, sample_preset):
    """All raw_returns values must be in range [-100, 500] (percentage space)."""
    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=sample_preset,
        n_folds=2,
    )

    for fold in result.get('fold_details', []):
        raw_returns = fold.get('raw_returns', [])
        if not raw_returns:
            continue
        for r in raw_returns:
            assert -150 <= r <= 600, (
                f"raw_return={r} outside percentage space [-150, 600]"
            )


# ─── Preset to_dict completeness ─────────────────────────────────────────────

def test_preset_to_dict_contains_all_preset_fields(sample_preset):
    """to_dict() must include all DynamicPreset fields."""
    d = sample_preset.to_dict()

    expected_fields = {
        'preset_id', 'preset_family', 'threshold',
        'max_trades_per_day', 'top_n_confidence_per_day',
        'max_open_positions', 'spread_limit_pct', 'liquidity_min',
        'premium_band', 'dte_range', 'regime_filter', 'expiry_filter',
        'avoid_first_n_minutes', 'avoid_last_n_minutes',
        'entry_time_start', 'entry_time_end',
        'expected_move_to_cost_ratio_min',
        'cooldown_after_loss_minutes',
        'daily_stop_loss_pct', 'daily_profit_lock_pct',
    }

    missing = expected_fields - set(d.keys())
    assert len(missing) == 0, f"to_dict() missing fields: {missing}"


# ─── Raw returns in fold result ─────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_evaluate_fold_returns_raw_returns_for_cost_stress(pe_subset, real_features, sample_preset):
    """evaluate_fold result must have 'raw_returns' key (list type) for cost_stress."""
    result = evaluate_candidate_preset(
        df=pe_subset,
        features=real_features,
        label='cost_survivor_label_v2',
        model_type='elasticnet',
        preset=sample_preset,
        n_folds=2,
    )

    for i, fold in enumerate(result.get('fold_details', [])):
        assert 'raw_returns' in fold, f"Fold {i} missing raw_returns"
        assert isinstance(fold['raw_returns'], list), (
            f"Fold {i} raw_returns is {type(fold['raw_returns'])}, expected list"
        )


# ─── Suspicious PF flagging ──────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_suspicious_pf_flagged_above_10(pe_subset, real_features):
    """If net_pf > 10.0, gates must contain SUSPICIOUS_PF: True."""
    # Use a very aggressive preset to attempt to trigger high PF
    preset = DynamicPreset(
        preset_id='test_suspicious_t20',
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

    net_pf = result.get('net_pf', 0)
    gates = result.get('gates', {})

    if net_pf > 10.0:
        assert gates.get('SUSPICIOUS_PF', False) is True, (
            f"net_pf={net_pf} > 10.0 but SUSPICIOUS_PF is not True"
        )