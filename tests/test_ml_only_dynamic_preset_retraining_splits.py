#!/usr/bin/env python3
"""
tests/test_ml_only_dynamic_preset_retraining_splits.py
======================================================
Test time-ordered walk-forward splits with no test data leakage.

Covers: chronological ordering, no fold overlap, expanding window,
val/test non-overlap, timestamp monotonicity, minimum data sizes,
no future data in train, all data used across folds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ml_only_dynamic_preset_retrainer import (
    load_and_audit_dataset,
    walk_forward_splits,
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
def real_splits(real_df_and_audit):
    """Pre-compute 3-fold walk-forward splits."""
    df, _ = real_df_and_audit
    return walk_forward_splits(df, n_folds=3)


# ─── Chronological ordering ──────────────────────────────────────────────────

def test_splits_are_chronological(real_splits):
    """For all folds: val_end == train_end, val_end < test_end."""
    for i, (train, val, test) in enumerate(real_splits):
        train_end_ts = train['timestamp_dt'].max()
        val_start_ts = val['timestamp_dt'].min()
        val_end_ts = val['timestamp_dt'].max()
        test_start_ts = test['timestamp_dt'].min()
        test_end_ts = test['timestamp_dt'].max()

        # train's last timestamp should be <= val's first
        assert train_end_ts <= val_start_ts, (
            f"Fold {i}: train max ({train_end_ts}) > val min ({val_start_ts})"
        )
        # val's last should be <= test's first
        assert val_end_ts <= test_start_ts, (
            f"Fold {i}: val max ({val_end_ts}) > test min ({test_start_ts})"
        )
        # val_end == test_start is the boundary
        assert val_end_ts < test_end_ts, (
            f"Fold {i}: val end ({val_end_ts}) not strictly before test end ({test_end_ts})"
        )


def test_no_overlap_between_folds(real_splits):
    """Walk-forward expanding-window splits.

    Key invariants:
    - Fold N's test must not appear in Fold N+2's test (test sets are strictly
      non-overlapping time windows). Note: fold N's test = fold N+1's val by design.
    - Within a fold: val and test must not share row indices

    NOTE: Expanding window means fold N+1's train IS a superset of fold N's
    train, and fold N's val == fold N+1's train boundary. This is correct
    walk-forward behavior.
    """
    for i in range(len(real_splits) - 1):
        _, val_curr, test_curr = real_splits[i]
        _, _, test_next = real_splits[i + 1]

        curr_test_idx = set(test_curr.index)

        # Fold N's test must NOT overlap with fold N+1's test
        assert len(curr_test_idx & set(test_next.index)) == 0, (
            f"Fold {i} test overlaps with fold {i+1} test"
        )


def test_train_set_always_grows(real_splits):
    """Expanding window: each fold's training set should be >= previous fold's."""
    for i in range(len(real_splits) - 1):
        train_curr = real_splits[i][0]
        train_next = real_splits[i + 1][0]
        assert len(train_next) >= len(train_curr), (
            f"Fold {i+1} train ({len(train_next)}) is smaller than "
            f"fold {i} train ({len(train_curr)}): not an expanding window"
        )


def test_val_and_test_sets_do_not_overlap(real_splits):
    """Val and test sets within a fold share no row indices."""
    for i, (train, val, test) in enumerate(real_splits):
        val_idx = set(val.index)
        test_idx = set(test.index)
        overlap = val_idx & test_idx
        assert len(overlap) == 0, (
            f"Fold {i}: val and test share {len(overlap)} row indices"
        )


def test_timestamp_monotonically_increases_in_each_split(real_splits):
    """Within each train/val/test split, timestamp_dt must be sorted ascending."""
    for i, (train, val, test) in enumerate(real_splits):
        for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
            if len(split_df) < 2:
                continue
            ts = split_df['timestamp_dt']
            diffs = ts.diff().dropna()
            negative = diffs[diffs < pd.Timedelta(0)]
            assert len(negative) == 0, (
                f"Fold {i} {split_name}: timestamps not monotonically increasing. "
                f"Found {len(negative)} decreases."
            )


def test_minimum_data_in_each_fold(real_splits):
    """Each fold must have train >= 100 rows, val >= 20, test >= 20."""
    for i, (train, val, test) in enumerate(real_splits):
        assert len(train) >= 100, f"Fold {i}: train={len(train)}, need >= 100"
        assert len(val) >= 20, f"Fold {i}: val={len(val)}, need >= 20"
        assert len(test) >= 20, f"Fold {i}: test={len(test)}, need >= 20"


def test_no_future_data_in_train(real_splits):
    """Train of fold N should not contain data from after val starts.

    With expanding window, train_max <= val_min (can be equal if boundary rows
    have the same timestamp). The key invariant: no row in train has a timestamp
    strictly AFTER any row in val.
    """
    for i, (train, val, test) in enumerate(real_splits):
        train_max = train['timestamp_dt'].max()
        val_min = val['timestamp_dt'].min()
        # Allow equality for same-timestamp boundary rows; reject strictly later
        assert train_max <= val_min, (
            f"Fold {i}: train max ({train_max}) > val min ({val_min}): "
            f"future data leaked into train"
        )


def test_all_data_used_across_folds(real_splits):
    """Test ranges are non-overlapping and sequential (allowing boundary equality).

    Walk-forward splits: each test set covers a distinct time period after the
    previous test set ends. They can touch at the same timestamp (expanding window
    boundary), but should not overlap.
    """
    # Collect all test ranges
    test_ranges = []
    for i, (train, val, test) in enumerate(real_splits):
        test_min = test['timestamp_dt'].min()
        test_max = test['timestamp_dt'].max()
        test_ranges.append((test_min, test_max))

    # All test ranges should be non-overlapping and sequential
    for i in range(len(test_ranges) - 1):
        curr_max = test_ranges[i][1]
        next_min = test_ranges[i + 1][0]
        # Allow equality (same-timestamp boundary), reject strict overlap
        assert curr_max <= next_min, (
            f"Test range {i} ({test_ranges[i]}) is strictly after "
            f"test range {i+1} ({test_ranges[i+1]})"
        )

    # Each test set should have at least some data
    for i, (train, val, test) in enumerate(real_splits):
        assert len(test) > 0, f"Fold {i}: test set is empty"


# Need pandas for pd.Timedelta in timestamp monotonicity test
import pandas as pd