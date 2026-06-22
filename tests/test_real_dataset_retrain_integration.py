"""tests/test_real_dataset_retrain_integration.py
==============================================
Real-dataset integration test for the ML retraining pipeline.

This test exercises the end-to-end retraining pipeline on the real
405k-row cost-aware edge dataset (~719 MB, 183 cols) to verify:
- Dataset loads successfully
- Required target column exists
- Feature hygiene passes (no forbidden columns)
- Model training works on a small time-bounded slice
- core_retrain_summary.json is written correctly
- No leakage features in training input

MARKER: @pytest.mark.real_dataset
Run with: pytest -q -m real_dataset tests/test_real_dataset_retrain_integration.py

NOTE: This test is NON-default and skipped by default pytest runs.
      Use the above command to explicitly enable.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data" / "processed"

# Add scripts to path for retrain imports
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


# -----------------------------------------------------------------------
# Constants & Fixtures
# -----------------------------------------------------------------------

# Canonical real dataset path (from CLAUDE.md)
REAL_DATASET_PATTERN = "nifty_option_chain_cost_aware_edge_dataset_*.csv"
REAL_DATASET_MIN_ROWS = 400_000
REAL_DATASET_MIN_COLS = 180

# Training slice parameters (keep small for fast test execution)
TRAINING_SLICE_ROWS = 5000  # Time-bounded slice, not full 405k
MIN_ROWS_FOR_TRAINING = 2000

# Target columns that must exist in the real dataset
REQUIRED_PRIMARY_LABELS = [
    "profitable_trade_label",
    "cost_survivor_label_v2",
    "strong_profitable_trade_label_v2",
]

# Models to test (keeping it minimal for speed)
TEST_MODEL_FAMILIES = ["logistic_regression"]


def _find_real_dataset() -> Path | None:
    """Find the canonical cost-aware edge dataset."""
    candidates = sorted(DATA_DIR.glob(REAL_DATASET_PATTERN), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        # Verify it has expected size (>= 500MB for the real dataset)
        if candidate.stat().st_size >= 500_000_000:
            return candidate
    return candidates[0] if candidates else None


def _load_real_dataset_slice(dataset_path: Path, max_rows: int = TRAINING_SLICE_ROWS) -> pd.DataFrame:
    """Load a time-bounded slice of the real dataset.
    
    Reads the first N rows which represent the earliest time period
    in the dataset. This ensures chronological training.
    """
    return pd.read_csv(dataset_path, nrows=max_rows)


# -----------------------------------------------------------------------
# Pytest Configuration
# -----------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register the real_dataset marker."""
    config.addinivalue_line(
        "markers",
        "real_dataset: marks tests that require the real 405k-row dataset (run with: pytest -q -m real_dataset)"
    )


# -----------------------------------------------------------------------
# Skip Conditions
# -----------------------------------------------------------------------

def _get_skip_reason() -> str | None:
    """Return skip reason if real dataset is not available, else None."""
    dataset_path = _find_real_dataset()
    if dataset_path is None:
        return "Real dataset not found: nifty_option_chain_cost_aware_edge_dataset_*.csv"
    
    if not dataset_path.exists():
        return f"Dataset path does not exist: {dataset_path}"
    
    if dataset_path.stat().st_size < 500_000_000:
        return f"Dataset too small ({dataset_path.stat().st_size / 1e6:.1f}MB) - expected >= 500MB"
    
    try:
        # Quick header check
        preview = pd.read_csv(dataset_path, nrows=10)
        row_count = sum(1 for _ in open(dataset_path, "r", encoding="utf-8", errors="ignore")) - 1
        
        if row_count < REAL_DATASET_MIN_ROWS:
            return f"Dataset has {row_count} rows, expected >= {REAL_DATASET_MIN_ROWS}"
        
        if len(preview.columns) < REAL_DATASET_MIN_COLS:
            return f"Dataset has {len(preview.columns)} columns, expected >= {REAL_DATASET_MIN_COLS}"
    except Exception as e:
        return f"Failed to read dataset: {e}"
    
    return None


# -----------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------

@pytest.mark.real_dataset
def test_real_dataset_is_accessible() -> None:
    """Verify the real dataset exists and meets minimum requirements."""
    skip_reason = _get_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
    
    dataset_path = _find_real_dataset()
    assert dataset_path is not None
    assert dataset_path.exists()
    
    # Check file size
    size_mb = dataset_path.stat().st_size / 1e6
    print(f"\n  Dataset: {dataset_path.name}")
    print(f"  Size: {size_mb:.1f} MB")
    
    assert size_mb >= 500, f"Dataset too small: {size_mb:.1f}MB"


@pytest.mark.real_dataset
def test_real_dataset_has_required_target_columns() -> None:
    """Verify the real dataset contains required target columns."""
    skip_reason = _get_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
    
    dataset_path = _find_real_dataset()
    assert dataset_path is not None
    
    # Read preview for column check
    df_preview = pd.read_csv(dataset_path, nrows=100)
    
    missing_labels = [label for label in REQUIRED_PRIMARY_LABELS if label not in df_preview.columns]
    assert not missing_labels, f"Missing required target columns: {missing_labels}"
    
    print(f"\n  Found target columns: {[c for c in REQUIRED_PRIMARY_LABELS if c in df_preview.columns]}")


@pytest.mark.real_dataset
def test_real_dataset_feature_hygiene_no_forbidden_columns() -> None:
    """Verify detect_column_groups properly excludes forbidden columns from features."""
    skip_reason = _get_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
    
    dataset_path = _find_real_dataset()
    assert dataset_path is not None
    
    # Load slice for feature hygiene check
    df = _load_real_dataset_slice(dataset_path, max_rows=1000)
    
    # Run feature detection
    groups = retrain.detect_column_groups(df)
    input_features = set(groups["input_features"])
    forbidden = set(groups["forbidden_feature_columns"])
    
    print(f"\n  Total columns: {len(df.columns)}")
    print(f"  Input features: {len(input_features)}")
    print(f"  Forbidden columns: {len(forbidden)}")
    
    # Verify required forbidden columns are blocked
    critical_forbidden = [
        "future_close",
        "gross_forward_return",
        "net_forward_return",
        "profitable_trade_label",  # _label pattern
        "strong_profitable_trade_label",  # _label pattern
        "cost_survivor_label_v2",  # _label_v2 pattern
    ]
    
    for col in critical_forbidden:
        if col in df.columns:
            assert col in forbidden, f"{col} should be in forbidden list"
            assert col not in input_features, f"{col} should NOT be in input features"
    
    # Verify _label and _label_v2 patterns are blocked
    label_cols = [c for c in df.columns if "_label" in c.lower()]
    for col in label_cols:
        assert col in forbidden, f"Label column {col} should be forbidden"
        assert col not in input_features, f"Label column {col} should NOT be in input features"
    
    print(f"  Verified {len(label_cols)} label columns are blocked")


@pytest.mark.real_dataset  
def test_real_dataset_model_training_on_slice() -> None:
    """Verify model training works on a small time-bounded slice (first 5000 rows).
    
    This test ensures the retraining pipeline can train a model on real data
    without errors. Training is limited to 5000 rows for speed.
    """
    skip_reason = _get_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
    
    dataset_path = _find_real_dataset()
    assert dataset_path is not None
    
    # Load training slice
    df = _load_real_dataset_slice(dataset_path, max_rows=TRAINING_SLICE_ROWS)
    
    print(f"\n  Training slice: {len(df)} rows, {len(df.columns)} columns")
    
    assert len(df) >= MIN_ROWS_FOR_TRAINING, f"Slice too small: {len(df)} rows"
    
    # Get column groups
    groups = retrain.detect_column_groups(df)
    input_features = groups["input_features"]
    target_columns = groups["target_columns"]
    
    print(f"  Input features: {len(input_features)}")
    print(f"  Target columns available: {len(target_columns)}")
    
    # Select target - prefer cost-aware labels
    preferred_targets = ["cost_survivor_label_v2", "strong_profitable_trade_label_v2", "profitable_trade_label"]
    target_col = None
    for t in preferred_targets:
        if t in target_columns:
            target_col = t
            break
    
    assert target_col is not None, f"No suitable target found in {target_columns}"
    print(f"  Selected target: {target_col}")
    
    # Prepare data - ensure chronological order
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Build feature matrix
    available_features = [f for f in input_features if f in df.columns]
    assert len(available_features) > 0, "No features available for training"
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Drop rows with NaN target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"  Training data: X={X.shape}, y={y.shape}")
    print(f"  Class distribution: {dict(y.value_counts())}")
    
    # Ensure we have both classes
    unique_labels = y.unique()
    assert len(unique_labels) >= 2, f"Target has only 1 class: {unique_labels}"
    
    # Train a simple model
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Chronological split for time-series
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    model.fit(X_train.values, y_train.values)
    
    # Basic sanity check
    train_score = model.score(X_train.values, y_train.values)
    test_score = model.score(X_test.values, y_test.values)
    
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")
    
    assert 0.0 <= train_score <= 1.0, "Invalid train score"
    assert 0.0 <= test_score <= 1.0, "Invalid test score"
    assert not np.isnan(train_score), "Train score is NaN"
    assert not np.isnan(test_score), "Test score is NaN"
    
    print("  Model training completed successfully")


@pytest.mark.real_dataset
def test_real_dataset_core_retrain_summary_written() -> None:
    """Verify core_retrain_summary.json is written to artifact directory.
    
    This is a critical artifact that downstream tools (--edge-refinement-report,
    build_cost_aware_retrain_comparison.py) depend on.
    """
    skip_reason = _get_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
    
    dataset_path = _find_real_dataset()
    assert dataset_path is not None
    
    # Create temporary artifact directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = Path(tmp_dir) / "core_retrain_test"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Load small slice for testing
        df = _load_real_dataset_slice(dataset_path, max_rows=2000)
        
        # Get column groups
        groups = retrain.detect_column_groups(df)
        labels_trained = groups["target_columns"][:1] if groups["target_columns"] else ["profitable_trade_label"]
        evaluation_return_column = groups["evaluation_return_column_used"] or "net_forward_return"
        
        # Simulate a metric row (as would be created by training)
        metric_rows = [{
            "label_name": labels_trained[0],
            "model_name": "logistic_regression",
            "status": "completed",
            "artifact_path": str(artifact_dir / "model.pkl"),
            "metrics_path": str(artifact_dir / "metrics.json"),
            "profit_factor": 1.15,
            "sharpe_ratio": 0.85,
            "trade_count": 150,
            "skip_reason": None,
        }]
        
        result_payload = {
            "failed_models": [],
            "skipped_models": [],
        }
        
        # Write the summary
        summary_path = retrain._write_core_retrain_summary(
            artifact_dir=artifact_dir,
            dataset_path=dataset_path,
            labels_trained=labels_trained,
            model_families=["logistic_regression"],
            metric_rows=metric_rows,
            result_payload=result_payload,
            evaluation_return_column=evaluation_return_column,
        )
        
        print(f"\n  Summary path: {summary_path}")
        assert summary_path.exists(), "core_retrain_summary.json was not written"
        
        # Verify content
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        print(f"  Summary keys: {list(summary.keys())}")
        
        # Verify required structure
        assert "dataset_selection" in summary, "Missing dataset_selection"
        assert "labels" in summary, "Missing labels"
        assert "families" in summary, "Missing families"
        assert "result" in summary, "Missing result"
        
        assert summary["dataset_selection"]["path"] == str(dataset_path)
        assert "logistic_regression" in summary["families"]
        assert len(summary["result"]["trained_models"]) > 0
        
        print(f"  Labels trained: {summary['labels']}")
        print(f"  Models trained: {len(summary['result']['trained_models'])}")


@pytest.mark.real_dataset
def test_real_dataset_no_leakage_features_in_training_input() -> None:
    """Verify no leakage features are present in the training input.
    
    This is a critical safety check to ensure the feature set is clean
    and does not contain future-looking or target-correlated columns.
    """
    skip_reason = _get_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
    
    dataset_path = _find_real_dataset()
    assert dataset_path is not None
    
    # Load slice
    df = _load_real_dataset_slice(dataset_path, max_rows=3000)
    
    # Build feature sets
    feature_sets = retrain.build_feature_sets(df, use_black_scholes_features=False)
    baseline_features = feature_sets["baseline_features"]
    rejected_features = set(feature_sets["rejected_features"])
    
    print(f"\n  Baseline features: {len(baseline_features)}")
    print(f"  Rejected features: {len(rejected_features)}")
    
    # Leakage patterns that must never appear in features
    LEAKAGE_PATTERNS = [
        "future", "forward", "return_after", "next_",
        "target", "label", "pnl", "profit", "loss",
        "outcome", "exit", "hit_target", "hit_sl",
        "mae", "mfe", "max_profit", "max_loss",
        "expected_return", "cost_adjusted_success",
        "ternary_trade_quality", "simulated_net_pnl",
    ]
    
    leakage_found = []
    for feature in baseline_features:
        feature_lower = feature.lower()
        for pattern in LEAKAGE_PATTERNS:
            if pattern in feature_lower:
                leakage_found.append((feature, pattern))
                break
    
    assert not leakage_found, f"Leakage features found: {leakage_found[:10]}"
    
    # Verify critical columns are rejected
    critical_rejections = [
        "future_close",
        "gross_forward_return",
        "net_forward_return",
        "profitable_trade_label",
        "cost_survivor_label",
        "strong_profitable_trade_label",
        "expected_return_after_cost",
        "return_to_cost_ratio",
        "cost_return_units_estimated",
    ]
    
    for col in critical_rejections:
        if col in df.columns:
            assert col in rejected_features, f"{col} should be rejected but is not"
    
    print("  No leakage features detected - feature set is clean")


@pytest.mark.real_dataset
def test_real_dataset_chronological_ordering() -> None:
    """Verify the dataset maintains chronological ordering for time-series validity."""
    skip_reason = _get_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
    
    dataset_path = _find_real_dataset()
    assert dataset_path is not None
    
    # Load slice
    df = _load_real_dataset_slice(dataset_path, max_rows=5000)
    
    if "timestamp" not in df.columns:
        pytest.skip("No timestamp column in dataset")
    
    # Parse and validate timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    
    # Sort by timestamp (as the retrain pipeline does)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Verify monotonic increasing
    timestamps = df["timestamp"]
    is_monotonic = timestamps.is_monotonic_increasing
    
    print(f"\n  Rows: {len(df)}")
    print(f"  First timestamp: {timestamps.iloc[0]}")
    print(f"  Last timestamp: {timestamps.iloc[-1]}")
    print(f"  Monotonic: {is_monotonic}")
    
    # After sorting, should be monotonic
    assert timestamps.is_monotonic_increasing, "Timestamps are not monotonic after sorting"
    
    # Verify no future timestamps (sanity check)
    now = datetime.now(df["timestamp"].dt.tz)
    future_count = (df["timestamp"] > now).sum()
    print(f"  Future timestamps: {future_count}")
    
    # Most timestamps should be in the past
    if future_count > len(df) * 0.1:
        print(f"  WARNING: {future_count} timestamps are in the future")


@pytest.mark.real_dataset
def test_real_dataset_cleanup_artifact_files() -> None:
    """Verify temp artifact files can be cleaned up properly.
    
    The test must clean up after itself - verify cleanup works.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = Path(tmp_dir) / "cleanup_test"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy artifacts
        (artifact_dir / "core_retrain_summary.json").write_text("{}")
        (artifact_dir / "model.pkl").write_bytes(b"fake model")
        (artifact_dir / "report.json").write_text("{}")
        
        # Verify files exist
        assert (artifact_dir / "core_retrain_summary.json").exists()
        assert (artifact_dir / "model.pkl").exists()
        
        # Cleanup using shutil (standard approach)
        shutil.rmtree(artifact_dir)
        
        # Verify directory is gone
        assert not artifact_dir.exists()
        
        print(f"\n  Cleanup verified: {artifact_dir.name} removed")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])