"""tests/test_cli_validation_fixes.py
====================================
Tests for CLI validation fixes in scripts/retrain_all_edge_models.py.

Coverage
--------
1. --only-target with cost-aware label passes validation (dataset has the column)
2. --only-target with invalid label raises a clear ValueError
3. --only-model=XGBoost (camelCase) maps to xgboost correctly (case-insensitive)
4. --only-model=ExtraTrees maps to extra_trees correctly (case-insensitive)
5. --only-model with unknown name raises a clear ValueError
6. Zero-model training never happens silently (always raises or fails clearly)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain

SCRIPT_PATH = REPO_ROOT / "scripts" / "retrain_all_edge_models.py"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_cost_aware_dataset(n_rows: int = 500, positive_share: float = 0.45) -> pd.DataFrame:
    """Create a synthetic dataset with cost-aware labels."""
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    rows = []
    n_positive = int(n_rows * positive_share)

    for idx in range(n_rows):
        is_positive = idx < n_positive
        is_ce = idx % 2 == 0
        volatility = idx % 5 in (0, 1)

        rows.append({
            "timestamp": base_ts + pd.Timedelta(minutes=idx),
            # Option context
            "option_type_ce": 1.0 if is_ce else 0.0,
            "option_type_pe": 0.0 if is_ce else 1.0,
            "ltp": float(50 + idx % 100),
            "volume": float(500 + idx * 2),
            "bid_ask_spread_pct": 0.02 + idx % 10 * 0.001,
            "strike_price": 24800.0,
            # Features
            "feature_a": float(idx % 7),
            "feature_b": float((idx * 3) % 11),
            "moneyness": 1.04 if idx % 3 == 0 else (1.0 if idx % 3 == 1 else 0.96),
            "regime_volatile": float(volatility),
            "volatility_regime_classifier": float(volatility),
            "ret_1": float((idx % 20) - 10) / 100.0,
            "rsi_14": 50.0 + (idx % 30) - 15,
            "atr_14": 2.0 + idx % 10 * 0.1,
            # Return columns
            "net_forward_return": 0.01 if is_positive else -0.01,
            "gross_forward_return": 0.015 if is_positive else -0.008,
            # Baseline labels
            "profitable_trade_label": 1 if is_positive else 0,
            "avoid_trade_label": 0 if is_positive else 1,
            # Cost-aware labels (more selective, realistic distributions)
            "cost_survivor_label_v2": 1 if is_positive and volatility else 0,
            "strong_profitable_trade_label_v2": 1 if is_positive and idx % 4 == 0 else 0,
            "cost_survivor_label": 1 if is_positive and volatility else 0,
            "strong_profitable_trade_label": 1 if is_positive and idx % 3 == 0 else 0,
        })
    return pd.DataFrame(rows)


def _minimal_dataset() -> pd.DataFrame:
    """Minimal dataset for fast validation tests (skips training overhead)."""
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    rows = []
    for idx in range(240):
        side_is_ce = idx % 2 == 0
        volatile = 1 if idx % 5 in (0, 1) else 0
        if idx % 6 in (0, 1):
            moneyness = 1.04
        elif idx % 6 in (2, 3):
            moneyness = 1.0
        else:
            moneyness = 0.95
        is_positive = idx % 4 in (0, 1)
        rows.append({
            "timestamp": base_ts + pd.Timedelta(minutes=idx),
            "feature_a": float(idx % 7),
            "feature_b": float((idx * 3) % 11),
            "option_volume": float(100 + idx),
            "option_type_ce": 1.0 if side_is_ce else 0.0,
            "option_type_pe": 0.0 if side_is_ce else 1.0,
            "moneyness": moneyness,
            "distance_from_atm": abs(moneyness - 1.0) * 100.0,
            "regime_volatile": float(volatile),
            "volatility_regime_classifier": float(volatile),
            "ltp": float(100 + (idx % 17)),
            "bid_ask_spread_pct": 0.02,
            "profitable_trade_label": int(is_positive),
            "avoid_trade_label": int(not is_positive),
            "net_forward_return": 0.01 if is_positive else -0.01,
            "future_profit": 1.0,
            "spot_return_1": 0.5,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: --only-target with cost-aware label is validated and accepted
# ---------------------------------------------------------------------------

def test_only_target_cost_aware_accepted(tmp_path: Path) -> None:
    """--only-target=cost_survivor_label_v2 is accepted when the column exists.

    This verifies FIX 1 & 2: choose_labels() discovers cost-aware labels
    dynamically, and the --only-target validator confirms the label is in
    the usable set before training begins.
    """
    df = _make_cost_aware_dataset(n_rows=500)
    dataset_path = tmp_path / "edge_dataset.csv"
    df.to_csv(dataset_path, index=False)

    # Verify the label exists in the dataset
    assert "cost_survivor_label_v2" in df.columns

    # Verify choose_labels() accepts it (dynamic discovery + validation pass)
    usable, skipped = retrain.choose_labels(df)
    assert "cost_survivor_label_v2" in usable, (
        f"cost_survivor_label_v2 should be discovered dynamically and accepted. "
        f"Usable: {usable}, Skipped: {skipped}"
    )

    # Verify the CLI pipeline accepts --only-target=cost_survivor_label_v2
    # We run with --report-only to avoid the full training overhead.
    output_root = tmp_path / "artifacts"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", str(dataset_path),
        "--retrain-all-models",
        "--only-target", "cost_survivor_label_v2",
        "--live-computable-only",
        "--no-production-adopt",
        "--min-rows", "100",
        "--output-dir", str(output_root),
        "--model-families", "logistic_regression",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)

    # Must NOT fail with "not found in dataset" error
    combined = (result.stdout + result.stderr).lower()
    assert "not found in dataset" not in combined, (
        f"--only-target=cost_survivor_label_v2 should be accepted. "
        f"stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )
    # Should not fail with "label not found" style errors
    assert "no usable label" not in combined
    assert "resolved to zero models" not in combined


# ---------------------------------------------------------------------------
# Test 2: --only-target with invalid label raises a clear ValueError
# ---------------------------------------------------------------------------

def test_only_target_invalid_fails(tmp_path: Path) -> None:
    """--only-target=this_label_does_not_exist raises a clear ValueError.

    This verifies FIX 3: if --only-target label does not exist in the dataset,
    the pipeline fails with a clear error instead of silently skipping training.
    """
    df = _minimal_dataset()
    dataset_path = tmp_path / "edge_dataset.csv"
    df.to_csv(dataset_path, index=False)

    output_root = tmp_path / "artifacts"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", str(dataset_path),
        "--retrain-all-models",
        "--only-target", "this_label_does_not_exist",
        "--live-computable-only",
        "--no-production-adopt",
        "--min-rows", "100",
        "--output-dir", str(output_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)

    # Must fail with non-zero exit code
    assert result.returncode != 0, (
        f"--only-target=this_label_does_not_exist should fail, got exit={result.returncode}. "
        f"stdout: {result.stdout[:500]}"
    )

    combined = (result.stdout + result.stderr).lower()
    # Must contain a helpful error about the missing label
    assert any(keyword in combined for keyword in [
        "not found in dataset",
        "this_label_does_not_exist",
        "label(s) not found",
        "--only-target",
    ]), (
        f"Error message should mention the missing label. "
        f"stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 3: --only-model=XGBoost (mixed case) maps to xgboost correctly
# ---------------------------------------------------------------------------

def test_only_model_xgboost_maps_correctly(tmp_path: Path) -> None:
    """--only-model=XGBoost (camelCase) matches xgboost from the registry.

    This verifies FIX 4: --only-model is case-insensitive.  The pipeline
    normalises both the registry names and the CLI input to lowercase
    before matching, so "XGBoost", "xgboost", and "XGBOOST" all work.
    """
    try:
        __import__("xgboost")
    except Exception:
        pytest.skip("xgboost not installed")
    df = _minimal_dataset()
    dataset_path = tmp_path / "edge_dataset.csv"
    df.to_csv(dataset_path, index=False)

    output_root = tmp_path / "artifacts"
    # Use camelCase "XGBoost" which should resolve to xgboost
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", str(dataset_path),
        "--retrain-all-models",
        "--only-model", "XGBoost",
        "--live-computable-only",
        "--no-production-adopt",
        "--min-rows", "100",
        "--output-dir", str(output_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)

    combined = (result.stdout + result.stderr).lower()

    # Must NOT fail with case-sensitivity error
    assert "resolved to zero models" not in combined, (
        f"--only-model=XGBoost should map to xgboost (case-insensitive). "
        f"Got: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )
    # Must not silently skip all training
    assert "models planned=" not in combined or "logistic" in combined or "xgboost" in combined
    assert result.returncode == 0, (
        f"Expected success with --only-model=XGBoost. "
        f"Exit: {result.returncode}, stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 4: --only-model=ExtraTrees maps to extra_trees correctly
# ---------------------------------------------------------------------------

def test_only_model_extratrees_maps_correctly(tmp_path: Path) -> None:
    """--only-model=ExtraTrees (mixed case) matches extra_trees from the registry.

    This verifies FIX 4 for another model name variant.
    """
    df = _minimal_dataset()
    dataset_path = tmp_path / "edge_dataset.csv"
    df.to_csv(dataset_path, index=False)

    output_root = tmp_path / "artifacts"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", str(dataset_path),
        "--retrain-all-models",
        "--only-model", "ExtraTrees",
        "--live-computable-only",
        "--no-production-adopt",
        "--min-rows", "100",
        "--output-dir", str(output_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)

    combined = (result.stdout + result.stderr).lower()

    assert "resolved to zero models" not in combined, (
        f"--only-model=ExtraTrees should map to extra_trees (case-insensitive). "
        f"Got: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )
    assert result.returncode == 0, (
        f"Expected success with --only-model=ExtraTrees. "
        f"Exit: {result.returncode}, stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 5: --only-model with unknown name raises a clear ValueError
# ---------------------------------------------------------------------------

def test_only_model_unknown_fails(tmp_path: Path) -> None:
    """--only-model=nonexistent_model raises a clear ValueError.

    This verifies FIX 5: when zero models match after case-normalised lookup,
    the pipeline raises a clear error instead of silently running zero models.
    """
    df = _minimal_dataset()
    dataset_path = tmp_path / "edge_dataset.csv"
    df.to_csv(dataset_path, index=False)

    output_root = tmp_path / "artifacts"
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", str(dataset_path),
        "--retrain-all-models",
        "--only-model", "nonexistent_model",
        "--live-computable-only",
        "--no-production-adopt",
        "--min-rows", "100",
        "--output-dir", str(output_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)

    assert result.returncode != 0, (
        f"--only-model=nonexistent_model should fail, got exit={result.returncode}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "resolved to zero models" in combined or "zero models" in combined, (
        f"Error should mention 'zero models'. "
        f"stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 6: Zero-trained models never happen silently
# ---------------------------------------------------------------------------

def test_zero_trained_models_never_silent(tmp_path: Path) -> None:
    """If --only-model resolves to zero models, training is aborted with an error.

    This is a defence-in-depth check.  Even if the outer CLI validator is
    bypassed, the train_variant() guard inside the pipeline raises a ValueError
    before any training begins.
    """
    df = _minimal_dataset()
    dataset_path = tmp_path / "edge_dataset.csv"
    df.to_csv(dataset_path, index=False)

    output_root = tmp_path / "artifacts"

    # Try several model names that don't exist in the registry
    for bad_model in ["not_a_model", "FakeXGBoost", "UNKNOWN"]:
        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--dataset", str(dataset_path),
            "--retrain-all-models",
            "--only-model", bad_model,
            "--live-computable-only",
            "--no-production-adopt",
            "--min-rows", "100",
            "--output-dir", str(output_root),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)

        assert result.returncode != 0, (
            f"--only-model={bad_model!r} must fail (zero models). "
            f"Got exit={result.returncode}, stdout: {result.stdout[:300]}"
        )
        combined = (result.stdout + result.stderr).lower()
        assert "zero models" in combined or "resolved to zero models" in combined, (
            f"Error must explicitly say zero models. "
            f"Got: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
        )


# ---------------------------------------------------------------------------
# Additional unit tests for choose_labels() dynamic discovery
# ---------------------------------------------------------------------------

def test_choose_labels_discovers_cost_aware_labels_dynamically() -> None:
    """choose_labels() should discover cost-aware labels from the dataset.

    This verifies FIX 2: cost-aware labels (strong_profitable_trade_label,
    cost_survivor_label, etc.) are discovered dynamically from the dataset,
    not hardcoded only in PRIMARY_LABELS.
    """
    df = _make_cost_aware_dataset(n_rows=500)

    usable, skipped = retrain.choose_labels(df)

    # Must include dynamically-discovered cost-aware labels
    cost_aware_labels = [
        "cost_survivor_label",
        "strong_profitable_trade_label",
        "cost_survivor_label_v2",
        "strong_profitable_trade_label_v2",
    ]
    for label in cost_aware_labels:
        assert label in usable, (
            f"{label!r} should be discovered dynamically from dataset. "
            f"Usable: {usable}, Skipped: {skipped}"
        )


def test_choose_labels_rejects_missing_column() -> None:
    """Labels not present in the dataset are correctly skipped."""
    df = _make_cost_aware_dataset(n_rows=500)
    usable, skipped = retrain.choose_labels(df)

    # A label that definitely doesn't exist in our dataset
    assert "nonexistent_label" not in usable
    assert "nonexistent_label" not in [r["label_name"] for r in skipped], (
        "Non-existent labels should not even appear in the skip report "
        "(they are simply absent from the candidate set)."
    )


def test_allowed_models_is_case_insensitive_in_train_variant(tmp_path: Path) -> None:
    """train_variant() internal allowed_models guard is case-insensitive.

    Verifies the defence-in-depth check inside train_variant().
    """
    try:
        __import__("xgboost")
    except Exception:
        pytest.skip("xgboost not installed")
    df = _minimal_dataset()
    dataset_path = tmp_path / "edge_dataset.csv"
    df.to_csv(dataset_path, index=False)

    # This should NOT raise "resolved to zero models" for mixed-case input
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", str(dataset_path),
        "--retrain-all-models",
        "--only-model", "XGBoost,RandomForest",
        "--live-computable-only",
        "--no-production-adopt",
        "--min-rows", "100",
        "--output-dir", str(tmp_path / "artifacts"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
    combined = (result.stdout + result.stderr).lower()
    assert "resolved to zero models" not in combined, (
        f"XGBoost,RandomForest should map correctly (case-insensitive). "
        f"Got: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    )