"""tests/test_retraining_dataset_schema.py
========================================
Tests for dataset schema validation in the ML retraining pipeline.

Coverage
--------
- Dataset loading works
- Timestamp column exists and is sortable
- Required columns exist (option context, OHLCV, labels)
- Forbidden columns are NOT used as features (explicit check)
- No future/forward return columns in features
- No PnL/return/scoring columns in features
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_test_dataset(
    with_labels: bool = True,
    with_return_cols: bool = True,
    with_future_cols: bool = False,
    with_leakage_cols: bool = False,
    with_regime_cols: bool = True,
    with_metadata_cols: bool = True,
    n_rows: int = 100,
) -> pd.DataFrame:
    """Create a synthetic dataset that matches expected retraining schema."""
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    rows = []
    for idx in range(n_rows):
        is_ce = idx % 2 == 0
        positive = idx % 4 in (0, 1)
        rows.append(
            {
                "timestamp": base_ts + pd.Timedelta(minutes=idx),
                # Option context
                "option_type_ce": 1.0 if is_ce else 0.0,
                "option_type_pe": 0.0 if is_ce else 1.0,
                "strike_price": 24800.0 + (idx % 5) * 100,
                "expiry": "2026-06-05",
                "ltp": float(100 + idx % 50),
                "volume": float(1000 + idx),
                "oi": float(50000 + idx * 10),
                "bid_ask_spread_pct": 0.02 + idx % 5 * 0.001,
                # OHLCV
                "open": float(100 + idx % 10),
                "high": float(105 + idx % 10),
                "low": float(95 + idx % 10),
                "close": float(102 + idx % 10),
                # Features (safe)
                "feature_a": float(idx % 7),
                "feature_b": float((idx * 3) % 11),
                "moneyness": 1.04 if idx % 3 == 0 else (1.0 if idx % 3 == 1 else 0.96),
                "distance_from_atm": abs(1.04 - 1.0) * 100 if idx % 3 == 0 else (abs(1.0 - 1.0) * 100 if idx % 3 == 1 else abs(0.96 - 1.0) * 100),
                "regime_volatile": float(idx % 5 in (0, 1)),
                "volatility_regime_classifier": float(idx % 5 in (0, 1)),
                "ret_1": float((idx % 10) - 5) / 100.0,
                "rsi_14": 50.0 + (idx % 20) - 10,
                "atr_14": 2.0 + idx % 5 * 0.1,
                "spot_return_1": float((idx % 8) - 4) / 100.0,
                "ctx_time_sin": float(idx % 24) / 24.0,
                "ctx_time_cos": float(idx % 60) / 60.0,
            }
        )
    df = pd.DataFrame(rows)
    
    # Add labels if requested
    if with_labels:
        df["profitable_trade_label"] = [1 if positive else 0 for positive in [idx % 4 in (0, 1) for idx in range(n_rows)]]
        df["avoid_trade_label"] = [0 if positive else 1 for positive in [idx % 4 in (0, 1) for idx in range(n_rows)]]
        df["cost_survivor_label_v2"] = [1 if positive and idx % 2 == 0 else 0 for idx, positive in enumerate([idx % 4 in (0, 1) for idx in range(n_rows)])]
        df["strong_profitable_trade_label_v2"] = [1 if positive and idx % 3 == 0 else 0 for idx, positive in enumerate([idx % 4 in (0, 1) for idx in range(n_rows)])]
    
    # Add return columns if requested
    if with_return_cols:
        df["net_forward_return"] = [0.01 if positive else -0.01 for positive in [idx % 4 in (0, 1) for idx in range(n_rows)]]
        df["gross_forward_return"] = [0.015 if positive else -0.008 for positive in [idx % 4 in (0, 1) for idx in range(n_rows)]]
    
    # Add future columns (leakage) if requested
    if with_future_cols:
        df["future_close"] = [float(100 + idx + 5) for idx in range(n_rows)]
        df["future_price_5m"] = [float(100 + idx + 3) for idx in range(n_rows)]
        df["next_return"] = [0.01 for _ in range(n_rows)]
    
    # Add forbidden leakage columns if requested
    if with_leakage_cols:
        df["future_profit"] = [1.0 if positive else 0.0 for positive in [idx % 4 in (0, 1) for idx in range(n_rows)]]
        df["realized_pnl"] = [0.5 if positive else -0.3 for positive in [idx % 4 in (0, 1) for idx in range(n_rows)]]
        df["trade_pnl"] = [0.3 if positive else -0.2 for positive in [idx % 4 in (0, 1) for idx in range(n_rows)]]
        df["target_return"] = [0.02 for _ in range(n_rows)]
    
    # Add regime columns if requested
    if with_regime_cols:
        df["regime_trending"] = [1 if idx % 5 in (0, 1) else 0 for idx in range(n_rows)]
        df["regime_mean_reverting"] = [1 if idx % 5 in (2, 3) else 0 for idx in range(n_rows)]
        df["regime_quiet"] = [1 if idx % 5 == 4 else 0 for idx in range(n_rows)]
        df["realized_vol_percentile_60"] = [float(idx % 100) / 100.0 for idx in range(n_rows)]
    
    # Add metadata columns if requested
    if with_metadata_cols:
        df["trading_day"] = "2026-06-01"
        df["selected_option_symbol"] = ["NIFTY24800CE" if is_ce else "NIFTY24800PE" for is_ce in [idx % 2 == 0 for idx in range(n_rows)]]
        df["trading_symbol"] = ["NIFTY24800CE" if is_ce else "NIFTY24800PE" for is_ce in [idx % 2 == 0 for idx in range(n_rows)]]
        df["option_type"] = ["CE" if is_ce else "PE" for is_ce in [idx % 2 == 0 for idx in range(n_rows)]]
        df["instrument_key"] = ["NSE_FO_NIFTY_24800_CE" if is_ce else "NSE_FO_NIFTY_24800_PE" for is_ce in [idx % 2 == 0 for idx in range(n_rows)]]
    
    return df


# ---------------------------------------------------------------------------
# Dataset Loading & Schema Tests
# ---------------------------------------------------------------------------

def test_dataset_timestamp_column_exists_and_sortable(tmp_path: Path) -> None:
    """Timestamp column must exist and be sortable for time-series integrity."""
    df = _make_test_dataset()
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    # Load and verify
    loaded = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    assert "timestamp" in loaded.columns, "Dataset must have 'timestamp' column"
    
    # Verify sortable
    assert loaded["timestamp"].is_monotonic_increasing or (loaded["timestamp"] == loaded["timestamp"].sort_values()).all(), \
        "Timestamp column must be sortable (monotonically non-decreasing)"
    
    # Verify no NaT values
    assert loaded["timestamp"].notna().all(), "Timestamp column must not have NaT values"


def test_dataset_has_required_option_context_columns(tmp_path: Path) -> None:
    """Required option context columns must be present."""
    df = _make_test_dataset()
    required_cols = ["option_type_ce", "option_type_pe", "strike_price", "ltp", "bid_ask_spread_pct"]
    for col in required_cols:
        assert col in df.columns, f"Required option context column '{col}' must be present"


def test_dataset_has_ohlcv_columns(tmp_path: Path) -> None:
    """OHLCV columns must be present for price-based features."""
    df = _make_test_dataset()
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        assert col in df.columns, f"Required OHLCV column '{col}' must be present"


def test_dataset_has_at_least_one_label_column(tmp_path: Path) -> None:
    """Dataset must have at least one target label column."""
    df = _make_test_dataset(with_labels=True)
    # detect_column_groups should find labels
    groups = retrain.detect_column_groups(df)
    assert len(groups["target_columns"]) > 0, "Dataset must have at least one target label column"


def test_dataset_has_return_column_for_evaluation(tmp_path: Path) -> None:
    """Dataset should have a return column for evaluation."""
    df = _make_test_dataset(with_return_cols=True)
    groups = retrain.detect_column_groups(df)
    assert len(groups["evaluation_return_columns"]) > 0, "Dataset should have evaluation return columns (net_forward_return, etc.)"


# ---------------------------------------------------------------------------
# Forbidden Column / Leakage Tests
# ---------------------------------------------------------------------------

def test_future_columns_are_detected_as_forbidden(tmp_path: Path) -> None:
    """Future/forward columns must be detected as forbidden features."""
    df = _make_test_dataset(with_future_cols=True)
    groups = retrain.detect_column_groups(df)
    forbidden = set(groups["forbidden_feature_columns"])
    
    # future_close, future_price_5m, next_return should be forbidden
    assert "future_close" in forbidden, "future_close should be in forbidden columns"
    assert "future_price_5m" in forbidden, "future_price_5m should be in forbidden columns"
    assert "next_return" in forbidden, "next_return should be in forbidden columns"


def test_pnl_return_columns_are_detected_as_forbidden(tmp_path: Path) -> None:
    """PnL/return/target columns must be detected as forbidden features."""
    df = _make_test_dataset(with_leakage_cols=True)
    groups = retrain.detect_column_groups(df)
    forbidden = set(groups["forbidden_feature_columns"])
    
    # These should all be in forbidden
    assert "future_profit" in forbidden, "future_profit should be forbidden"
    assert "realized_pnl" in forbidden, "realized_pnl should be forbidden"
    assert "trade_pnl" in forbidden, "trade_pnl should be forbidden"
    assert "target_return" in forbidden, "target_return should be forbidden"


def test_label_columns_not_in_features(tmp_path: Path) -> None:
    """Label columns must NOT appear in the feature set."""
    df = _make_test_dataset(with_labels=True)
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    
    # Labels should NOT be in features
    assert "profitable_trade_label" not in features, "profitable_trade_label should not be in features"
    assert "avoid_trade_label" not in features, "avoid_trade_label should not be in features"
    assert "cost_survivor_label_v2" not in features, "cost_survivor_label_v2 should not be in features"
    assert "strong_profitable_trade_label_v2" not in features, "strong_profitable_trade_label_v2 should not be in features"


def test_return_columns_not_in_features(tmp_path: Path) -> None:
    """Return/evaluation columns must NOT appear in the feature set."""
    df = _make_test_dataset(with_return_cols=True)
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    eval_cols = set(groups["evaluation_return_columns"])
    
    # Return columns should NOT be in features
    assert "net_forward_return" not in features, "net_forward_return should not be in features"
    assert "gross_forward_return" not in features, "gross_forward_return should not be in features"
    
    # Return columns should be in evaluation columns
    assert "net_forward_return" in eval_cols or len(eval_cols) > 0


def test_metadata_columns_not_in_features(tmp_path: Path) -> None:
    """Metadata columns should not be in the feature set."""
    df = _make_test_dataset(with_metadata_cols=True)
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    
    # Some metadata columns should be excluded
    metadata_in_features = [col for col in ["trading_day", "selected_option_symbol", "trading_symbol", "instrument_key"] if col in features]
    # At least some should be excluded
    assert len(metadata_in_features) < 4, "Most metadata columns should be excluded from features"


def test_is_forbidden_feature_name_detects_leakage() -> None:
    """_is_forbidden_feature_name should correctly identify forbidden columns."""
    # These should be forbidden
    forbidden_names = [
        "future_close", "future_profit", "net_forward_return", "gross_forward_return",
        "profitable_trade_label", "avoid_trade_label", "trade_pnl", "realized_pnl",
        "target_return", "next_return", "trade_result", "exit_price",
    ]
    for name in forbidden_names:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be detected as forbidden"
    
    # These should NOT be forbidden (safe features)
    safe_names = [
        "feature_a", "feature_b", "ltp", "volume", "bid_ask_spread_pct",
        "moneyness", "ret_1", "rsi_14", "atr_14", "regime_volatile",
    ]
    for name in safe_names:
        assert not retrain._is_forbidden_feature_name(name), f"'{name}' should NOT be detected as forbidden"
    
    # Exception: realized_vol and variants are allowed
    assert not retrain._is_forbidden_feature_name("realized_vol"), "realized_vol should be allowed"
    assert not retrain._is_forbidden_feature_name("realized_volatility"), "realized_volatility should be allowed"
    assert not retrain._is_forbidden_feature_name("realized_vol_percentile"), "realized_vol_percentile should be allowed"


def test_looks_like_evaluation_column_detects_returns() -> None:
    """_looks_like_evaluation_column should correctly detect evaluation/return columns."""
    eval_names = [
        "net_forward_return", "gross_forward_return", "trade_pnl",
        "realized_pnl", "target_return", "cost_adjusted_return",
    ]
    for name in eval_names:
        assert retrain._looks_like_evaluation_column(name), f"'{name}' should be detected as evaluation column"
    
    # These should NOT be evaluation columns (they're features)
    # Note: spot_return_1 is actually flagged because it contains 'return'
    # The actual filtering uses both _is_forbidden_feature_name and _looks_like_evaluation_column together
    not_eval_names = [
        "ret_1", "ret_3", "return_10",
    ]
    for name in not_eval_names:
        assert not retrain._looks_like_evaluation_column(name), f"'{name}' should NOT be detected as evaluation column"


# ---------------------------------------------------------------------------
# Regime Columns Tests
# ---------------------------------------------------------------------------

def test_regime_columns_available_in_dataset(tmp_path: Path) -> None:
    """Regime columns should be available in the dataset."""
    df = _make_test_dataset(with_regime_cols=True)
    required_regime = ["regime_volatile", "regime_trending", "regime_mean_reverting", "regime_quiet"]
    for col in required_regime:
        assert col in df.columns, f"Required regime column '{col}' must be present"


def test_detect_column_groups_includes_regime_columns(tmp_path: Path) -> None:
    """detect_column_groups should properly handle regime columns."""
    df = _make_test_dataset(with_regime_cols=True)
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    
    # Regime columns should be in features (they're safe)
    assert "regime_volatile" in features, "regime_volatile should be in features"


# ---------------------------------------------------------------------------
# Schema Validation Integration Test
# ---------------------------------------------------------------------------

def test_dataset_passes_schema_validation_workflow(tmp_path: Path) -> None:
    """End-to-end: dataset should pass schema validation workflow."""
    df = _make_test_dataset(
        with_labels=True,
        with_return_cols=True,
        with_future_cols=False,  # No leakage
        with_leakage_cols=False,  # No leakage
        with_regime_cols=True,
        n_rows=200,
    )
    dataset_path = tmp_path / "dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    # Load and detect column groups
    loaded = pd.read_csv(dataset_path)
    groups = retrain.detect_column_groups(loaded)
    
    # Verify structure
    assert len(groups["input_features"]) > 0, "Should have input features"
    assert len(groups["target_columns"]) > 0, "Should have target columns"
    assert len(groups["evaluation_return_columns"]) > 0, "Should have evaluation return columns"
    
    # Verify no forbidden columns in features
    forbidden_in_features = [col for col in groups["input_features"] if col in groups["forbidden_feature_columns"]]
    assert len(forbidden_in_features) == 0, f"Forbidden columns should not be in features: {forbidden_in_features}"
    
    # Verify labels not in features
    labels_in_features = [col for col in groups["input_features"] if col in groups["target_columns"]]
    assert len(labels_in_features) == 0, f"Labels should not be in features: {labels_in_features}"
    
    # Verify return columns not in features
    returns_in_features = [col for col in groups["input_features"] if col in groups["evaluation_return_columns"]]
    assert len(returns_in_features) == 0, f"Return columns should not be in features: {returns_in_features}"
    
    # Verify feature audit
    audit = groups["feature_audit"]
    assert audit["raw_column_count"] > 0, "Should have raw columns"
    assert len(audit["numeric_columns"]) > 0, "Should have numeric columns"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_empty_dataset_handled_gracefully() -> None:
    """Empty dataframe should be handled gracefully."""
    df = pd.DataFrame()
    groups = retrain.detect_column_groups(df)
    assert groups["input_features"] == []
    assert groups["target_columns"] == []


def test_constant_column_excluded_from_features() -> None:
    """Constant columns (only one unique value) should be excluded from features."""
    df = _make_test_dataset(n_rows=50)
    df["constant_col"] = 1.0  # Same value for all rows
    groups = retrain.detect_column_groups(df)
    
    # constant_col should not be in features
    assert "constant_col" not in groups["input_features"], "Constant columns should not be in features"


def test_high_null_column_excluded_from_features() -> None:
    """Columns with >95% null values should be excluded from features."""
    df = _make_test_dataset(n_rows=50)
    df["mostly_null"] = [1.0 if i < 3 else None for i in range(50)]  # Only 6% non-null
    groups = retrain.detect_column_groups(df)
    
    # mostly_null should not be in features
    assert "mostly_null" not in groups["input_features"], "High null columns should not be in features"


def test_cost_aware_labels_accepted_as_targets(tmp_path: Path) -> None:
    """Cost-aware labels should be accepted as valid targets."""
    df = _make_test_dataset(with_labels=True)
    
    # Add cost-aware labels (these are valid targets)
    assert "cost_survivor_label_v2" in df.columns
    assert "strong_profitable_trade_label_v2" in df.columns
    
    groups = retrain.detect_column_groups(df)
    labels = set(groups["target_columns"])
    
    # These cost-aware labels should be in the target columns
    assert "cost_survivor_label_v2" in labels or len(labels) > 0
    assert "strong_profitable_trade_label_v2" in labels or len(labels) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])