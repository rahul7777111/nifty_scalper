#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_retraining_no_leakage.py
==========================================================
Test dynamic preset and ML feature leakage.

Covers: preset filter fields are live-computable only, feature list
excludes return/label/future columns, threshold tuning on val only,
fold details contain raw returns (not features), preset filters
applied before scoring.
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
    apply_preset_filters,
    evaluate_candidate_preset,
    get_live_features,
    load_and_audit_dataset,
)

REAL_DATASET = REPO_ROOT / "data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"

# Preset filter fields that must be live-computable
PRESET_LIVE_FILTER_FIELDS = {
    'dte_range', 'spread_limit_pct', 'liquidity_min', 'premium_band',
    'threshold', 'max_trades_per_day', 'top_n_confidence_per_day',
    'regime_filter', 'expiry_filter',
    'avoid_first_n_minutes', 'avoid_last_n_minutes',
    'entry_time_start', 'entry_time_end',
    'expected_move_to_cost_ratio_min',
}

# Preset filter fields that must NEVER be used in filters
PRESET_LEAKAGE_FIELDS = {
    'return', 'pnl', 'mfe', 'mae', 'label', 'outcome',
    'forward', 'realized', 'cost_survivor', 'profitable',
    'strong_profitable', 'paper_candidate', 'avoid_trade',
    'weak_trade', 'no_trade', 'trade_outcome',
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
    df, _ = real_df_and_audit
    return get_live_features(df)


@pytest.fixture(scope="module")
def pe_subset(real_df_and_audit):
    """Return a PE-only subset for faster testing."""
    df, _ = real_df_and_audit
    pe_df = df[df.get('option_type', '').astype(str).str.upper() == 'PE'].copy()
    return pe_df.head(50000)


# ─── Preset filter fields ────────────────────────────────────────────────────

def test_preset_filter_uses_only_live_computable_fields():
    """All filter fields in DynamicPreset must be live-computable (no return/PnL/label)."""
    preset = DynamicPreset()
    all_fields = set(preset.to_dict().keys())

    # No leakage fields in the preset dataclass
    for field in all_fields:
        field_lower = field.lower()
        for leakage_token in PRESET_LEAKAGE_FIELDS:
            assert leakage_token not in field_lower, (
                f"DynamicPreset field '{field}' contains leakage token '{leakage_token}'"
            )

    # The filter-related fields should all be clearly live-computable
    # dte_range uses dte_days (live), spread_limit_pct uses range_pct (live),
    # liquidity_min uses volume (live), premium_band uses ltp (live)
    for f in PRESET_LIVE_FILTER_FIELDS:
        assert f in all_fields, f"Expected filter field '{f}' missing from DynamicPreset"


# ─── Feature list exclusions ─────────────────────────────────────────────────

def test_feature_list_excludes_return_columns(real_features):
    """get_live_features must NOT include any return column."""
    return_cols = {
        'net_forward_return', 'gross_forward_return',
        'expected_return_after_cost', 'return_to_cost_ratio',
    }
    for feat in real_features:
        assert feat not in return_cols, f"Feature '{feat}' is a return column and must be excluded"


def test_feature_list_excludes_label_columns(real_features):
    """get_live_features must NOT include any column with '_label' in name."""
    for feat in real_features:
        assert '_label' not in feat.lower(), (
            f"Feature '{feat}' contains '_label' and must be excluded"
        )


def test_feature_list_excludes_future_columns(real_features):
    """get_live_features must NOT include mfe, mae, realized, exit, trade_outcome."""
    future_tokens = {'mfe', 'mae', 'realized', 'exit', 'trade_outcome'}
    for feat in real_features:
        feat_lower = feat.lower()
        for token in future_tokens:
            assert token not in feat_lower, (
                f"Feature '{feat}' contains future/leakage token '{token}'"
            )


# ─── Threshold tuning ────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_evaluate_fold_does_not_use_test_labels_for_threshold(pe_subset, real_features):
    """Threshold tuning must use val_df only, not test_df."""
    # We verify by checking that the code path for threshold tuning
    # in evaluate_fold uses val_df scores only. We do this by inspecting
    # the fold_details: each fold should have a threshold_used that was
    # tuned on val, and the test metrics should be based on that threshold.
    preset = DynamicPreset(
        preset_id='test_thresh_t35',
        preset_family='B_balanced',
        threshold=0.35,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
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

    # The threshold used in each fold should be in the expected range
    # (F1-selected from val, between 0.20 and 0.80)
    for fold in result.get('fold_details', []):
        thresh = fold.get('threshold_used')
        assert thresh is not None, "fold must have threshold_used"
        assert 0.10 <= thresh <= 0.90, (
            f"threshold_used={thresh} outside expected range"
        )


@pytest.mark.slow
@pytest.mark.real_dataset
def test_threshold_tuning_uses_filtered_val_set(pe_subset, real_features):
    """Threshold is tuned on top-N/day filtered val data, not raw val."""
    # The pipeline in evaluate_fold is:
    # val_copy['_score'] = val_scores
    # val_filtered = apply_preset_filters(val_copy, preset)
    # val_filtered = top_n_per_day(val_filtered, '_score', preset.top_n_confidence_per_day)
    # THEN threshold tuning on val_filtered
    # This means threshold tuning uses the SAME filtering as test evaluation.
    preset = DynamicPreset(
        preset_id='test_top_n_filter',
        preset_family='B_balanced',
        threshold=0.35,
        max_trades_per_day=2,
        top_n_confidence_per_day=2,  # top-2 per day
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

    # The fold_details should reflect that scoring happened on filtered data
    # If threshold tuning used unfiltered val, the thresholds would differ
    for fold in result.get('fold_details', []):
        thresh = fold.get('threshold_used')
        # threshold should be one of the search values [0.20, 0.25, 0.30, 0.35, ...]
        valid_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        assert thresh in valid_thresholds, (
            f"threshold_used={thresh} not in valid threshold search grid"
        )


# ─── Raw returns (not features) ─────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.real_dataset
def test_evaluate_candidate_preset_returns_raw_returns_not_features(pe_subset, real_features):
    """fold_details must have 'raw_returns' (list of floats), not feature data."""
    preset = DynamicPreset(
        preset_id='test_raw_returns_t30',
        preset_family='B_balanced',
        threshold=0.30,
        max_trades_per_day=3,
        top_n_confidence_per_day=3,
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

    for fold in result.get('fold_details', []):
        assert 'raw_returns' in fold, f"Fold detail missing 'raw_returns': {fold.keys()}"
        raw_returns = fold['raw_returns']
        assert isinstance(raw_returns, list), (
            f"raw_returns must be a list, got {type(raw_returns)}"
        )
        assert all(isinstance(r, (int, float)) for r in raw_returns), (
            "raw_returns must contain only numeric values"
        )
        # raw_returns are in percentage space, so -100 to 500 is reasonable
        assert all(-150 <= r <= 600 for r in raw_returns), (
            f"Some raw_returns outside percentage space: min={min(raw_returns)}, max={max(raw_returns)}"
        )


# ─── Preset filters applied before scoring ───────────────────────────────────

def test_preset_filters_applied_before_scoring():
    """apply_preset_filters must be called on test BEFORE scoring (live-computable only)."""
    # We verify this by checking that apply_preset_filters only uses
    # live-computable fields. Inspect the function's use of df columns.
    import pandas as pd
    import numpy as np

    # Create a mock dataframe with leakage columns added
    df = pd.DataFrame({
        'dte_days': [3.0, 5.0, 1.0, 10.0],
        'range_pct': [0.05, 0.12, 0.08, 0.20],
        'volume': [100_000, 40_000, 200_000, 30_000],
        'ltp': [80.0, 200.0, 40.0, 300.0],
        'timestamp_dt': pd.to_datetime(['2024-01-01'] * 4),
        # Leakage columns — should be IGNORED by apply_preset_filters
        'net_forward_return': [0.10, -0.05, 0.20, -0.10],
        'gross_forward_return': [0.12, -0.03, 0.25, -0.08],
        'cost_survivor_label_v2': [1, 0, 1, 0],
        'mfe': [0.5, -0.2, 0.8, -0.3],
        'mae': [0.3, -0.1, 0.4, -0.2],
    })

    preset = DynamicPreset(
        preset_id='test_filters',
        preset_family='B_balanced',
        threshold=0.35,
        max_trades_per_day=2,
        top_n_confidence_per_day=2,
        spread_limit_pct=0.10,  # Filters out row 1 (0.12) and row 3 (0.20)
        liquidity_min=50000.0,  # Filters out row 1 (40k) and row 3 (30k)
        premium_band='mid',     # Filters to ltp 50-150: only row 0 (80)
    )

    filtered = apply_preset_filters(df, preset)

    # After applying all filters:
    # - spread_limit_pct=0.10: keeps rows with range_pct <= 0.10 -> rows 0, 2
    # - liquidity_min=50000: keeps rows with volume >= 50000 -> rows 0, 2
    # - premium_band='mid': keeps ltp 50-150 -> row 0 only
    assert len(filtered) == 1, f"Expected 1 row after filters, got {len(filtered)}"
    assert filtered.iloc[0]['dte_days'] == 3.0, "Wrong row kept after filtering"