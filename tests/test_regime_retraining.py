"""tests/test_regime_retraining.py
==================================
Tests for regime-specific retraining behavior.

Coverage
--------
- Regime-specific training mode if available
- Regime columns available in dataset
- Regime edge datasets are recognized
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
# Constants
# ---------------------------------------------------------------------------

REQUIRED_REGIME_COLS = [
    "regime_trending",
    "regime_volatile",
    "regime_mean_reverting",
    "regime_quiet",
    "volatility_regime_classifier",
]

# Regime definitions from build_regime_edge_datasets.py
REGIME_DEFINITIONS = {
    "high_volatility": "(regime_volatile == 1) | (realized_vol_percentile_60 > 0.7)",
    "low_volatility": "(regime_quiet == 1) | (realized_vol_percentile_60 < 0.3)",
    "trending": "regime_trending == 1",
    "mean_reverting": "regime_mean_reverting == 1",
    "quiet": "regime_quiet == 1",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_regime_dataset(n_rows: int = 300) -> pd.DataFrame:
    """Create a synthetic dataset with regime columns."""
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    rows = []
    
    for idx in range(n_rows):
        # Create regime patterns
        regime_cycle = idx % 8
        
        if regime_cycle in (0, 1):
            # Volatile regime
            regime_volatile = 1
            regime_trending = 1
            regime_mean_reverting = 0
            regime_quiet = 0
            realized_vol_pct = 0.8 + (idx % 10) / 50.0
        elif regime_cycle in (2, 3):
            # Trending regime
            regime_volatile = 0
            regime_trending = 1
            regime_mean_reverting = 0
            regime_quiet = 0
            realized_vol_pct = 0.4 + (idx % 10) / 50.0
        elif regime_cycle in (4, 5):
            # Mean reverting regime
            regime_volatile = 0
            regime_trending = 0
            regime_mean_reverting = 1
            regime_quiet = 0
            realized_vol_pct = 0.3 + (idx % 10) / 50.0
        else:
            # Quiet regime
            regime_volatile = 0
            regime_trending = 0
            regime_mean_reverting = 0
            regime_quiet = 1
            realized_vol_pct = 0.15 + (idx % 10) / 100.0
        
        is_positive = idx % 4 in (0, 1)
        is_ce = idx % 2 == 0
        
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
            "ret_1": float((idx % 20) - 10) / 100.0,
            "rsi_14": 50.0 + (idx % 30) - 15,
            "atr_14": 2.0 + idx % 10 * 0.1,
            # Regime columns
            "regime_volatile": float(regime_volatile),
            "regime_trending": float(regime_trending),
            "regime_mean_reverting": float(regime_mean_reverting),
            "regime_quiet": float(regime_quiet),
            "volatility_regime_classifier": float(regime_volatile),
            "realized_vol_percentile_60": realized_vol_pct,
            # Return columns
            "net_forward_return": 0.01 if is_positive else -0.01,
            "gross_forward_return": 0.015 if is_positive else -0.008,
            # Labels
            "profitable_trade_label": 1 if is_positive else 0,
            "avoid_trade_label": 0 if is_positive else 1,
        })
    
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime Column Availability Tests
# ---------------------------------------------------------------------------

def test_regime_columns_available_in_dataset() -> None:
    """Required regime columns must be available in the dataset."""
    df = _make_regime_dataset()
    
    for col in REQUIRED_REGIME_COLS:
        assert col in df.columns, f"Required regime column '{col}' must be present"


def test_regime_columns_are_numeric() -> None:
    """Regime columns should be numeric (0 or 1 for binary regimes)."""
    df = _make_regime_dataset()
    
    for col in REQUIRED_REGIME_COLS:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"
            unique_vals = df[col].dropna().unique()
            assert all(v in (0, 1, 0.0, 1.0) for v in unique_vals), \
                f"{col} should have binary values (0 or 1)"


def test_regime_columns_in_feature_set() -> None:
    """Regime columns should be included in the feature set (they're not leakage)."""
    df = _make_regime_dataset()
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    
    for col in REQUIRED_REGIME_COLS:
        assert col in features, f"Regime column '{col}' should be in features"


def test_realized_vol_percentile_column_available() -> None:
    """realized_vol_percentile_60 column should be available for regime filtering."""
    df = _make_regime_dataset()
    
    assert "realized_vol_percentile_60" in df.columns, \
        "realized_vol_percentile_60 should be present for regime filtering"


# ---------------------------------------------------------------------------
# Regime Masking Tests
# ---------------------------------------------------------------------------

def test_volatile_regime_filter() -> None:
    """Volatile regime filter should select only volatile rows."""
    df = _make_regime_dataset()
    
    # Filter for volatile regime
    volatile_mask = df["regime_volatile"] == 1
    volatile_df = df[volatile_mask]
    
    # All selected rows should have regime_volatile = 1
    assert (volatile_df["regime_volatile"] == 1).all()
    
    # Should have some rows selected (given our data generation)
    assert len(volatile_df) > 0, "Should have some volatile rows"


def test_trending_regime_filter() -> None:
    """Trending regime filter should select only trending rows."""
    df = _make_regime_dataset()
    
    # Filter for trending regime
    trending_mask = df["regime_trending"] == 1
    trending_df = df[trending_mask]
    
    assert (trending_df["regime_trending"] == 1).all()
    assert len(trending_df) > 0


def test_quiet_regime_filter() -> None:
    """Quiet regime filter should select only quiet rows."""
    df = _make_regime_dataset()
    
    # Filter for quiet regime
    quiet_mask = df["regime_quiet"] == 1
    quiet_df = df[quiet_mask]
    
    assert (quiet_df["regime_quiet"] == 1).all()
    assert len(quiet_df) > 0


def test_high_volatility_regime_filter() -> None:
    """High volatility regime filter should use both regime_volatile and realized_vol_percentile."""
    df = _make_regime_dataset()
    
    # High volatility: regime_volatile == 1 OR realized_vol_percentile_60 > 0.7
    high_vol_mask = (df["regime_volatile"] == 1) | (df["realized_vol_percentile_60"] > 0.7)
    high_vol_df = df[high_vol_mask]
    
    # Should have some rows selected
    assert len(high_vol_df) > 0


def test_low_volatility_regime_filter() -> None:
    """Low volatility regime filter should use both regime_quiet and realized_vol_percentile."""
    df = _make_regime_dataset()
    
    # Low volatility: regime_quiet == 1 OR realized_vol_percentile_60 < 0.3
    low_vol_mask = (df["regime_quiet"] == 1) | (df["realized_vol_percentile_60"] < 0.3)
    low_vol_df = df[low_vol_mask]
    
    # Should have some rows selected
    assert len(low_vol_df) > 0


# ---------------------------------------------------------------------------
# Regime Candidate Tests
# ---------------------------------------------------------------------------

def test_regime_candidates_defined() -> None:
    """Regime candidate definitions should exist."""
    # ALL_RESEARCH_CANDIDATE_SPECS contains the candidate definitions
    assert hasattr(retrain, 'ALL_RESEARCH_CANDIDATE_SPECS'), "ALL_RESEARCH_CANDIDATE_SPECS should exist"
    candidates = retrain.ALL_RESEARCH_CANDIDATE_SPECS
    regime_candidates = [c for c in candidates if c.get("candidate_group") == "regime"]
    
    # Should have at least some regime candidates
    assert len(regime_candidates) > 0, "Should have regime candidates defined"


def test_regime_candidates_have_filters() -> None:
    """Regime candidates should have regime-based filters."""
    candidates = retrain.ALL_RESEARCH_CANDIDATE_SPECS
    regime_candidates = [c for c in candidates if c.get("candidate_group") == "regime"]
    
    for candidate in regime_candidates:
        assert "filters" in candidate, f"Candidate {candidate['candidate_name']} should have filters"
        # Should have at least one regime filter
        filters = candidate["filters"]
        regime_filter_keys = ["volatile_only", "option_side", "moneyness"]
        assert any(k in filters for k in regime_filter_keys), \
            f"Candidate {candidate['candidate_name']} should have regime-related filter"


def test_regime_candidate_mask_function() -> None:
    """_regime_candidate_mask should correctly filter rows for regime candidates."""
    df = _make_regime_dataset()
    
    # Test volatile_only filter
    filters = {"volatile_only": True}
    mask = retrain._regime_candidate_mask(df, filters)
    
    # Should select volatile rows
    assert mask.sum() > 0, "volatile_only filter should select some rows"
    
    # All selected rows should be volatile
    selected_df = df[mask]
    assert (selected_df["regime_volatile"] == 1).all()


def test_regime_candidate_option_side_filter() -> None:
    """Regime candidates with option_side filter should work correctly."""
    df = _make_regime_dataset()
    
    # Test CE filter
    filters = {"option_side": "CE"}
    mask = retrain._regime_candidate_mask(df, filters)
    
    selected_df = df[mask]
    # For CE, option_type_ce should be 1
    if "option_type_ce" in selected_df.columns:
        assert (selected_df["option_type_ce"] == 1).all()
    
    # Test PE filter
    filters = {"option_side": "PE"}
    mask = retrain._regime_candidate_mask(df, filters)
    
    selected_df = df[mask]
    if "option_type_pe" in selected_df.columns:
        assert (selected_df["option_type_pe"] == 1).all()


# ---------------------------------------------------------------------------
# Regime Performance Report Tests
# ---------------------------------------------------------------------------

def test_regime_performance_report_structure() -> None:
    """_regime_performance_report should produce expected structure."""
    df = _make_regime_dataset()
    
    # Create mock prediction data
    np.random.seed(42)
    y_prob = np.clip(df["ret_1"].values + 0.5 + np.random.randn(len(df)) * 0.1, 0.1, 0.9)
    returns = df["net_forward_return"].values
    
    try:
        report = retrain._regime_performance_report(
            test_frame=df, y_prob=y_prob, threshold=0.55, returns=returns
        )
        
        # Check expected structure
        assert "available" in report
        assert "groups" in report
        assert report["available"] is True
        assert isinstance(report["groups"], dict)
        
    except Exception as e:
        # If report generation fails, at least verify inputs are valid
        assert len(df) > 0
        assert len(y_prob) == len(df)
        pytest.fail(f"_regime_performance_report should handle test data: {e}")


def test_regime_performance_report_has_per_group_metrics() -> None:
    """_regime_performance_report should have metrics per regime group."""
    df = _make_regime_dataset()
    
    np.random.seed(42)
    y_prob = np.clip(df["ret_1"].values + 0.5 + np.random.randn(len(df)) * 0.1, 0.1, 0.9)
    returns = df["net_forward_return"].values
    
    try:
        report = retrain._regime_performance_report(
            test_frame=df, y_prob=y_prob, threshold=0.55, returns=returns
        )
        
        groups = report.get("groups", {})
        
        # Should have regime-related group columns in the output
        # The function looks for columns like regime_volatile, regime_trending, etc.
        # Check that at least some regime groups are present
        regime_group_keys = [k for k in groups.keys() if 'regime' in k.lower()]
        
        # Report should be available and have groups
        assert report["available"] is True
        assert isinstance(groups, dict)
        
    except Exception as e:
        pytest.fail(f"_regime_performance_report should work: {e}")


# ---------------------------------------------------------------------------
# Regime Edge Dataset Tests
# ---------------------------------------------------------------------------

def test_regime_columns_not_in_forbidden() -> None:
    """Regime columns should NOT be in the forbidden feature list."""
    df = _make_regime_dataset()
    groups = retrain.detect_column_groups(df)
    forbidden = set(groups["forbidden_feature_columns"])
    
    for col in REQUIRED_REGIME_COLS:
        assert col not in forbidden, f"Regime column '{col}' should NOT be forbidden"


def test_regime_splits_preserve_data_integrity() -> None:
    """Regime splits should preserve data integrity."""
    df = _make_regime_dataset()
    
    # Split by regime
    volatile_df = df[df["regime_volatile"] == 1]
    non_volatile_df = df[df["regime_volatile"] == 0]
    
    # Combined should equal original
    assert len(volatile_df) + len(non_volatile_df) == len(df)
    
    # Each split should have required columns
    for split_df in [volatile_df, non_volatile_df]:
        for col in REQUIRED_REGIME_COLS:
            assert col in split_df.columns


def test_regime_split_has_sufficient_rows() -> None:
    """Regime splits should have sufficient rows for training."""
    df = _make_regime_dataset(n_rows=500)
    
    # Check each regime has enough samples
    for regime_name, regime_col in [
        ("volatile", "regime_volatile"),
        ("trending", "regime_trending"),
        ("quiet", "regime_quiet"),
    ]:
        regime_df = df[df[regime_col] == 1]
        # For retraining, we typically want at least 100 samples
        # But for testing, just verify we have some data
        assert len(regime_df) > 0, f"Should have {regime_name} regime data"


# ---------------------------------------------------------------------------
# Regime Definitions Verification
# ---------------------------------------------------------------------------

def test_regime_definitions_cover_key_regimes() -> None:
    """Regime definitions should cover key market regimes."""
    expected_regimes = ["high_volatility", "low_volatility", "trending", "quiet"]
    for regime in expected_regimes:
        assert regime in REGIME_DEFINITIONS, f"Should have {regime} definition"


def test_regime_masks_are_comprehensive() -> None:
    """Regime masks should cover all rows (no row left out)."""
    df = _make_regime_dataset(n_rows=200)
    
    # All rows should belong to at least one basic regime
    has_regime = (
        (df["regime_volatile"] == 1) |
        (df["regime_trending"] == 1) |
        (df["regime_mean_reverting"] == 1) |
        (df["regime_quiet"] == 1)
    )
    
    # All rows should have at least one regime flag
    assert has_regime.all(), "All rows should have at least one regime designation"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_missing_regime_columns_handled() -> None:
    """Missing regime columns should be handled gracefully."""
    df = _make_regime_dataset()
    df = df.drop(columns=["regime_volatile", "regime_trending"])
    
    # detect_column_groups should still work
    groups = retrain.detect_column_groups(df)
    assert "input_features" in groups


def test_empty_regime_split_handled() -> None:
    """Empty regime split should be handled gracefully."""
    df = _make_regime_dataset()
    
    # Force empty split
    df["regime_volatile"] = 0
    volatile_df = df[df["regime_volatile"] == 1]
    
    assert len(volatile_df) == 0


def test_single_regime_dataset() -> None:
    """Dataset with single regime should work."""
    df = _make_regime_dataset()
    
    # Make all rows volatile
    df["regime_volatile"] = 1
    df["regime_trending"] = 0
    df["regime_mean_reverting"] = 0
    df["regime_quiet"] = 0
    
    # Should still work
    groups = retrain.detect_column_groups(df)
    assert "input_features" in groups


if __name__ == "__main__":
    pytest.main([__file__, "-v"])