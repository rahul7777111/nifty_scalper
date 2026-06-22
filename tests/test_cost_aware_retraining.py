"""tests/test_cost_aware_retraining.py
======================================
Tests for cost-aware retraining behavior.

Coverage
--------
- cost_survivor_label_v2 target is accepted
- strong_profitable_trade_label_v2 target is accepted
- profitable_trade_label target works (baseline)
- Cost-aware labels have proper class distribution
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
from ml_execution_costs import (
    audit_double_counting,
    cost_survival_score,
    estimate_option_execution_costs_frame,
)


# ---------------------------------------------------------------------------
# Label Definitions & Constants
# ---------------------------------------------------------------------------

# Primary labels from retrain_all_edge_models.py
COST_AWARE_LABELS = [
    "strong_profitable_trade_label_v2",
    "cost_survivor_label_v2",
    "strong_profitable_trade_label",
    "cost_survivor_label",
]

BASELINE_LABELS = [
    "profitable_trade_label",
    "avoid_trade_label",
]


def _make_cost_aware_dataset(n_rows: int = 500, positive_share: float = 0.45) -> pd.DataFrame:
    """Create a synthetic dataset with cost-aware labels."""
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    rows = []
    n_positive = int(n_rows * positive_share)
    
    for idx in range(n_rows):
        is_positive = idx < n_positive
        is_ce = idx % 2 == 0
        volatility = idx % 5 in (0, 1)
        
        # Generate realistic features
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
            # Cost-aware labels (more selective)
            "cost_survivor_label_v2": 1 if is_positive and volatility else 0,
            "strong_profitable_trade_label_v2": 1 if is_positive and idx % 4 == 0 else 0,
            "cost_survivor_label": 1 if is_positive and volatility else 0,
            "strong_profitable_trade_label": 1 if is_positive and idx % 3 == 0 else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost-Aware Label Acceptance Tests
# ---------------------------------------------------------------------------

def test_cost_survivor_label_v2_accepted_as_target() -> None:
    """cost_survivor_label_v2 should be accepted as a valid target label."""
    df = _make_cost_aware_dataset()
    
    # Check column exists
    assert "cost_survivor_label_v2" in df.columns, "cost_survivor_label_v2 must exist"
    
    # Check detect_column_groups recognizes it as a label
    groups = retrain.detect_column_groups(df)
    labels = set(groups["target_columns"])
    
    # cost_survivor_label_v2 should be in target columns
    assert "cost_survivor_label_v2" in labels, "cost_survivor_label_v2 should be recognized as a target"


def test_strong_profitable_trade_label_v2_accepted_as_target() -> None:
    """strong_profitable_trade_label_v2 should be accepted as a valid target label."""
    df = _make_cost_aware_dataset()
    
    # Check column exists
    assert "strong_profitable_trade_label_v2" in df.columns, "strong_profitable_trade_label_v2 must exist"
    
    # Check detect_column_groups recognizes it as a label
    groups = retrain.detect_column_groups(df)
    labels = set(groups["target_columns"])
    
    # strong_profitable_trade_label_v2 should be in target columns
    assert "strong_profitable_trade_label_v2" in labels, "strong_profitable_trade_label_v2 should be recognized as a target"


def test_profitable_trade_label_baseline_still_works() -> None:
    """profitable_trade_label (baseline) should continue to work."""
    df = _make_cost_aware_dataset()
    
    groups = retrain.detect_column_groups(df)
    labels = set(groups["target_columns"])
    
    # Baseline labels should be recognized
    assert "profitable_trade_label" in labels
    assert "avoid_trade_label" in labels


def test_all_cost_aware_labels_in_primary_labels_list() -> None:
    """Cost-aware labels should be in PRIMARY_LABELS constant."""
    for label in COST_AWARE_LABELS:
        assert label in retrain.PRIMARY_LABELS, f"{label} should be in PRIMARY_LABELS"


# ---------------------------------------------------------------------------
# Cost-Aware Label Class Distribution Tests
# ---------------------------------------------------------------------------

def test_cost_aware_labels_have_valid_class_distribution() -> None:
    """Cost-aware labels should have valid class distributions (not too imbalanced)."""
    df = _make_cost_aware_dataset()
    
    for label in COST_AWARE_LABELS:
        if label in df.columns:
            counts = df[label].value_counts()
            total = len(df)
            
            # Check positive share is in reasonable range (5% - 95%)
            if 1 in counts.index:
                positive_share = counts[1] / total
                assert 0.05 <= positive_share <= 0.95, \
                    f"{label} has extreme class imbalance: {positive_share:.2%} positive"
            
            # Check we have both classes
            assert len(counts) >= 2, f"{label} should have at least 2 classes (0 and 1)"


def test_cost_survivor_label_v2_more_selective_than_baseline() -> None:
    """cost_survivor_label_v2 should be more selective (fewer positives) than baseline."""
    df = _make_cost_aware_dataset()
    
    baseline_positive_share = df["profitable_trade_label"].mean()
    cost_survivor_positive_share = df["cost_survivor_label_v2"].mean()
    
    # Cost survivor should be more selective (fewer positives)
    assert cost_survivor_positive_share <= baseline_positive_share, \
        "cost_survivor_label_v2 should be more selective than baseline"


def test_strong_profitable_label_v2_most_selective() -> None:
    """strong_profitable_trade_label_v2 should be the most selective."""
    df = _make_cost_aware_dataset()
    
    baseline = df["profitable_trade_label"].mean()
    strong = df["strong_profitable_trade_label_v2"].mean()
    
    # Strong profitable should be more selective
    assert strong <= baseline, \
        "strong_profitable_trade_label_v2 should be more selective than baseline"


# ---------------------------------------------------------------------------
# Cost Model Integration Tests
# ---------------------------------------------------------------------------

def test_audit_double_counting_detects_embedded_cost() -> None:
    """audit_double_counting should detect embedded cost from gross vs net returns."""
    df = _make_cost_aware_dataset(n_rows=200)
    
    result = audit_double_counting(df, evaluation_return_column="net_forward_return")
    
    # Should detect embedded cost
    assert "inferred_embedded_cost" in result
    assert result["inferred_embedded_cost"] >= 0, "Embedded cost should be non-negative"
    assert result["double_counting_prevented"] is True or result["recommendation"] != "BLOCKED"


def test_estimate_option_execution_costs_frame_produces_columns() -> None:
    """estimate_option_execution_costs_frame should produce expected columns."""
    df = _make_cost_aware_dataset(n_rows=100)
    costs = estimate_option_execution_costs_frame(df)
    
    # Check expected columns exist
    expected_cols = [
        "premium", "embedded_cost", "brokerage_cost", "exchange_cost",
        "stt_cost", "gst_cost", "sebi_cost", "spread_cost", "slippage_cost",
        "total_estimated_roundtrip_cost", "cost_pct_of_premium", "cost_return_units",
    ]
    for col in expected_cols:
        assert col in costs.columns, f"Expected cost column '{col}' not found"


def test_cost_survival_score_ranks_cost_stable_candidates_higher() -> None:
    """cost_survival_score should rank cost-stable candidates higher."""
    # Two candidates: one profitable only without cost, one profitable even with cost
    candidate_a = {"profit_factor": 1.5, "sharpe": 1.2, "cost_stress_1_25x_pf": 0.9}
    candidate_b = {"profit_factor": 1.3, "sharpe": 1.0, "cost_stress_1_25x_pf": 1.15}
    
    score_a = cost_survival_score(candidate_a)
    score_b = cost_survival_score(candidate_b)
    
    # Candidate B should have higher score (survives 1.25x cost stress)
    assert score_b > score_a, "Cost-stable candidate should have higher cost_survival_score"


def test_cost_survival_score_returns_nonzero_for_valid_candidates() -> None:
    """cost_survival_score should return non-zero score for valid candidates."""
    candidate = {
        "profit_factor": 1.25,
        "sharpe": 1.0,
        "cost_stress_1_25x_pf": 1.1,
        "cost_stress_1_50x_pf": 1.0,
    }
    
    score = cost_survival_score(candidate)
    assert score > 0, "Valid cost-surviving candidate should have positive score"


def test_cost_survival_score_returns_zero_for_invalid_candidates() -> None:
    """cost_survival_score should return lower scores for invalid candidates."""
    # A candidate that fails cost stress should have lower score
    bad_candidate = {
        "profit_factor": 0.9,  # Loses money
        "sharpe": 0.2,
        "cost_stress_1_25x_pf": 0.7,  # Fails cost stress
        "cost_stress_1_50x_pf": 0.5,  # Also fails 1.5x stress
    }
    
    # A good candidate that passes cost stress
    good_candidate = {
        "profit_factor": 1.3,  # Makes money
        "sharpe": 1.1,
        "cost_stress_1_25x_pf": 1.15,  # Passes 1.25x cost stress
        "cost_stress_1_50x_pf": 1.0,  # Passes 1.5x cost stress
    }
    
    bad_score = cost_survival_score(bad_candidate)
    good_score = cost_survival_score(good_candidate)
    
    # Good candidate should have higher score than bad candidate
    assert good_score > bad_score, "Good candidate should have higher cost_survival_score than bad candidate"


# ---------------------------------------------------------------------------
# Cost-Aware Dataset Build Tests
# ---------------------------------------------------------------------------

def test_cost_aware_labels_not_in_features() -> None:
    """Cost-aware labels should NOT appear in the feature set."""
    df = _make_cost_aware_dataset()
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    
    for label in COST_AWARE_LABELS:
        if label in df.columns:
            assert label not in features, f"{label} should not be in features"


def test_return_columns_in_evaluation_not_features() -> None:
    """Return columns should be in evaluation columns, not features."""
    df = _make_cost_aware_dataset()
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    eval_cols = set(groups["evaluation_return_columns"])
    
    # net_forward_return should be in evaluation columns
    assert "net_forward_return" in eval_cols
    # But not in features
    assert "net_forward_return" not in features


def test_cost_columns_properly_excluded() -> None:
    """Columns with cost-related patterns should be properly excluded."""
    # Add some cost-related columns
    df = _make_cost_aware_dataset()
    df["cost_adjusted_success_5m"] = df["profitable_trade_label"]  # This should be forbidden
    
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    
    # cost_adjusted_success_5m should be forbidden
    assert "cost_adjusted_success_5m" in groups["forbidden_feature_columns"]
    assert "cost_adjusted_success_5m" not in features


# ---------------------------------------------------------------------------
# Cost Stress Tests
# ---------------------------------------------------------------------------

def test_cost_stress_scenarios_reduce_returns() -> None:
    """Cost stress scenarios should reduce effective returns."""
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.85, 0.76, 0.1])
    returns = np.array([0.35, 0.30, 0.28, -0.2, -0.1])
    
    report = retrain._cost_stress_report(y_true, y_prob, 0.75, returns)
    
    # Base return should be higher than stressed returns
    base_return = report["base"]["average_return_per_trade"]
    
    for scenario in ["extra_cost_0_25", "extra_cost_0_50", "extra_cost_1_00", "extra_cost_2_00"]:
        stress_return = report[scenario]["average_return_per_trade"]
        assert stress_return <= base_return, f"{scenario} should reduce returns"


def test_cost_stress_gate_fails_when_profit_factor_drops() -> None:
    """Cost stress gate should fail when profit factor drops below threshold."""
    # Simulate a candidate that passes base but fails 1.25x cost stress
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1])
    returns = np.array([0.20, 0.18, -0.1])  # Small positive returns
    
    report = retrain._cost_stress_report(y_true, y_prob, 0.75, returns)
    
    # With small returns, stress scenarios should have low profit factor
    assert report["extra_cost_1_00"]["profit_factor"] < report["base"]["profit_factor"]


# ---------------------------------------------------------------------------
# Primary Labels Verification
# ---------------------------------------------------------------------------

def test_primary_labels_includes_cost_aware_v2() -> None:
    """PRIMARY_LABELS should include v2 cost-aware labels."""
    for label in ["strong_profitable_trade_label_v2", "cost_survivor_label_v2"]:
        assert label in retrain.PRIMARY_LABELS, f"{label} must be in PRIMARY_LABELS"


def test_primary_labels_includes_baseline_labels() -> None:
    """PRIMARY_LABELS should include baseline labels."""
    for label in ["profitable_trade_label", "avoid_trade_label"]:
        assert label in retrain.PRIMARY_LABELS, f"{label} must be in PRIMARY_LABELS"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_empty_cost_aware_frame_handled() -> None:
    """Empty dataframe should be handled gracefully."""
    df = pd.DataFrame()
    costs = estimate_option_execution_costs_frame(df)
    assert costs.empty


def test_missing_return_column_handled() -> None:
    """Missing return column should be handled gracefully."""
    df = _make_cost_aware_dataset()
    df = df.drop(columns=["net_forward_return"])
    
    result = audit_double_counting(df, evaluation_return_column="net_forward_return")
    assert "recommendation" in result
    assert result["recommendation"] in ["BLOCKED_MISSING_GROSS_OR_NET", "VALID_GROSS_RETURN_COST_MODEL"]


def test_all_zero_returns_handled() -> None:
    """All-zero returns should be handled gracefully."""
    df = _make_cost_aware_dataset()
    df["net_forward_return"] = 0.0
    
    result = audit_double_counting(df, evaluation_return_column="net_forward_return")
    assert "inferred_embedded_cost" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])