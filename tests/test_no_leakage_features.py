"""tests/test_no_leakage_features.py
===================================
Tests for ML feature leakage prevention.

Coverage
--------
- STRICT_FORBIDDEN_TOKENS columns cannot enter feature set
- FUTURE_LEAK_TOKENS columns cannot enter feature set
- FORBIDDEN_COLUMN_TOKENS columns cannot enter feature set
- No *_label, *_label_v2 columns in features
- No future_close, gross_forward_return, net_forward_return in features
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
# Token-based Forbidden Name Tests
# ---------------------------------------------------------------------------

def test_strict_forbidden_tokens_detected() -> None:
    """STRICT_FORBIDDEN_TOKENS should block columns with these substrings."""
    tokens = retrain.STRICT_FORBIDDEN_TOKENS
    
    # These substrings should all be forbidden
    test_cases = [
        "future_close", "realized_pnl", "next_return", "target_col",
        "label_col", "pnl_total", "profit_amount", "loss_col",
        "return_pct", "outcome_result", "exit_price", "entry_result",
        "trade_result", "hit_target", "hit_sl", "mae_col", "mfe_col",
        "forward_return", "future_price", "future_return", "post_trade_result",
    ]
    for name in test_cases:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden due to STRICT_FORBIDDEN_TOKENS"


def test_future_leak_tokens_detected() -> None:
    """FUTURE_LEAK_TOKENS should block forward-looking columns."""
    tokens = retrain.FUTURE_LEAK_TOKENS
    
    # These substrings should be forbidden (from FUTURE_LEAK_TOKENS)
    test_cases = [
        "return_after_5m", "gross_pnl_10m", "net_pnl_5m",
        "max_favorable_excursion", "max_adverse_excursion",
        "option_trade_success_5m", "cost_adjusted_success_10m",
        "ternary_trade_quality_5m", "simulated_net_pnl_5m",
    ]
    for name in test_cases:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden due to FUTURE_LEAK_TOKENS"


def test_forbidden_column_tokens_detected() -> None:
    """FORBIDDEN_COLUMN_TOKENS should block columns with these substrings."""
    # These substrings are specifically in FORBIDDEN_COLUMN_TOKENS
    test_cases = [
        "net_forward_return", "gross_forward_return",
        "next_return_col", "return_5m",
    ]
    for name in test_cases:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden due to FORBIDDEN_COLUMN_TOKENS"


# ---------------------------------------------------------------------------
# Label Column Leakage Tests
# ---------------------------------------------------------------------------

def test_label_columns_blocked_from_features() -> None:
    """Any column ending with _label should be blocked from features."""
    label_names = [
        "profitable_trade_label",
        "avoid_trade_label",
        "cost_survivor_label",
        "strong_profitable_trade_label",
        "paper_candidate_label",
        "high_conviction_trade_label",
        "profitable_trade_label_v2",
        "cost_survivor_label_v2",
        "strong_profitable_trade_label_v2",
        "paper_candidate_label_v2",
        "high_conviction_trade_label_v2",
        "weak_trade_label",
        "no_trade_label",
        "big_move_label",
        "edge_label",
        "custom_label",
    ]
    for name in label_names:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden"


def test_label_suffix_variations_blocked() -> None:
    """Various _label suffix variations should all be blocked."""
    variations = [
        "trade_label", "signal_label", "result_label",
        "prediction_label", "quality_label", "score_label",
    ]
    for name in variations:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden"


# ---------------------------------------------------------------------------
# Specific Forbidden Column Tests
# ---------------------------------------------------------------------------

def test_future_close_blocked() -> None:
    """future_close should be blocked from features."""
    assert retrain._is_forbidden_feature_name("future_close")
    assert retrain._is_forbidden_feature_name("future_close_5m")
    assert retrain._is_forbidden_feature_name("future_close_10m")


def test_forward_return_columns_blocked() -> None:
    """Forward return columns should be blocked from features."""
    forbidden_names = [
        "gross_forward_return",
        "net_forward_return",
        "forward_return",
        "future_return_5m",
        "return_forward",
    ]
    for name in forbidden_names:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden"


def test_pnl_columns_blocked() -> None:
    """PnL columns should be blocked from features."""
    forbidden_names = [
        "realized_pnl",
        "net_pnl",
        "gross_pnl",
        "trade_pnl",
        "simulated_pnl",
    ]
    for name in forbidden_names:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden"


def test_target_outcome_columns_blocked() -> None:
    """Target/outcome columns should be blocked from features."""
    forbidden_names = [
        "target_return",
        "target_profit",
        "outcome_result",
        "trade_result",
        "exit_result",
    ]
    for name in forbidden_names:
        assert retrain._is_forbidden_feature_name(name), f"'{name}' should be forbidden"


# ---------------------------------------------------------------------------
# Safe Feature Tests
# ---------------------------------------------------------------------------

def test_safe_feature_columns_not_blocked() -> None:
    """Known safe features should NOT be blocked."""
    safe_names = [
        # Price/volume features
        "ltp", "close", "open", "high", "low", "volume", "oi",
        "bid_ask_spread", "bid_ask_spread_pct", "mid_price",
        # Return features (ret_ prefix is safe, return_ prefix triggers blocking)
        "ret_1", "ret_3", "ret_5", "ret_10",
        # Indicator features
        "rsi_14", "atr_14", "atr_pct", "adx_14", "ema_fast", "ema_slow",
        "ema_diff_pct", "supertrend_dir", "choppiness_14",
        # Option context features
        "moneyness", "log_moneyness", "distance_from_atm", "atm_distance",
        "strike_distance_pct", "dte_days", "time_to_expiry_days",
        # Regime features (safe)
        "regime_volatile", "regime_trending", "regime_mean_reverting", "regime_quiet",
        "volatility_regime_classifier", "regime_features_json",
        # Realized volatility (allowed)
        "realized_vol", "realized_volatility", "realized_vol_percentile",
        "realized_vol_percentile_60", "realized_vol_z_score",
        # Session/time features
        "ctx_time_sin", "ctx_time_cos", "weekday", "month",
        "is_opening_session", "is_closing_session",
        # Candlestick features
        "body_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
        "close_location_pct", "bullish_engulfing", "bearish_engulfing",
        # Generic features
        "feature_a", "feature_b", "feature_c", "oc_change_pct",
        "hl_change_pct", "oi_change_pct", "volume_change_pct",
        "oi_z_5", "volume_z_5", "range_pct",
    ]
    for name in safe_names:
        assert not retrain._is_forbidden_feature_name(name), f"'{name}' should NOT be forbidden (it's a safe feature)"


def test_realized_vol_not_blocked() -> None:
    """Realized volatility features should NOT be blocked (they're historical)."""
    safe_names = [
        "realized_vol",
        "realized_volatility",
        "realized_vol_percentile",
        "realized_vol_percentile_60",
        "realized_vol_z_score",
        "realized_vol_20",
    ]
    for name in safe_names:
        assert not retrain._is_forbidden_feature_name(name), f"'{name}' should NOT be forbidden (realized vol is historical)"


def test_spot_return_features_not_blocked() -> None:
    """Spot return features should NOT be blocked (except those with 'return' in name)."""
    # Note: columns with 'return' in the name are blocked by STRICT_FORBIDDEN_TOKENS
    # So spot_return_* columns are blocked by design
    # But spot_* columns without 'return' should be allowed
    safe_names = [
        "ctx_spot", "spot_close", "spot_open", "spot_high", "spot_low",
        "spot_atr", "spot_rsi", "spot_vwap", "spot_range_pct",
    ]
    for name in safe_names:
        assert not retrain._is_forbidden_feature_name(name), f"'{name}' should NOT be forbidden"


# ---------------------------------------------------------------------------
# Integration Test with Dataset
# ---------------------------------------------------------------------------

def _make_leakage_test_dataset() -> pd.DataFrame:
    """Create a dataset with both safe and forbidden columns."""
    n_rows = 100
    base_ts = pd.Timestamp("2026-06-01 09:15:00+05:30")
    rows = []
    for idx in range(n_rows):
        rows.append({
            "timestamp": base_ts + pd.Timedelta(minutes=idx),
            # Safe features
            "ltp": float(100 + idx % 50),
            "volume": float(1000 + idx),
            "rsi_14": 50.0 + idx % 20 - 10,
            "ret_1": float((idx % 10) - 5) / 100.0,
            "moneyness": 1.04 if idx % 3 == 0 else 1.0,
            "regime_volatile": float(idx % 2),
            "realized_vol": float(idx % 30) / 100.0,
            # Forbidden: leakage columns
            "future_close": float(100 + idx + 5),
            "net_forward_return": float(0.01 if idx % 2 == 0 else -0.01),
            "gross_forward_return": float(0.015 if idx % 2 == 0 else -0.008),
            "profitable_trade_label": int(idx % 2),
            "future_profit": 1.0 if idx % 2 == 0 else 0.0,
            "realized_pnl": float(0.5 if idx % 2 == 0 else -0.3),
            # Mixed: some that look like returns but are past
            "return_5m_past": float((idx % 10) - 5) / 100.0,  # Past return (safe)
        })
    return pd.DataFrame(rows)


def test_leakage_columns_not_in_feature_set() -> None:
    """Forbidden/leakage columns must not appear in the feature set."""
    df = _make_leakage_test_dataset()
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    forbidden = set(groups["forbidden_feature_columns"])
    
    # Check specific leakage columns are blocked
    assert "future_close" not in features, "future_close should not be in features"
    assert "net_forward_return" not in features, "net_forward_return should not be in features"
    assert "gross_forward_return" not in features, "gross_forward_return should not be in features"
    assert "profitable_trade_label" not in features, "profitable_trade_label should not be in features"
    assert "future_profit" not in features, "future_profit should not be in features"
    assert "realized_pnl" not in features, "realized_pnl should not be in features"
    
    # Check they ARE in forbidden
    assert "future_close" in forbidden, "future_close should be in forbidden"
    assert "net_forward_return" in forbidden, "net_forward_return should be in forbidden"
    assert "profitable_trade_label" in forbidden, "profitable_trade_label should be in forbidden"


def test_safe_features_included() -> None:
    """Safe features should be included in the feature set."""
    df = _make_leakage_test_dataset()
    groups = retrain.detect_column_groups(df)
    features = set(groups["input_features"])
    
    # Check safe features are included
    assert "ltp" in features, "ltp should be in features"
    assert "volume" in features, "volume should be in features"
    assert "rsi_14" in features, "rsi_14 should be in features"
    assert "ret_1" in features, "ret_1 should be in features"
    assert "moneyness" in features, "moneyness should be in features"
    assert "regime_volatile" in features, "regime_volatile should be in features"
    assert "realized_vol" in features, "realized_vol should be in features"
    # Note: columns with 'return' in name (like return_5m_past) are blocked by design


def test_return_column_candidates_detected() -> None:
    """Return column candidates should be detected."""
    df = _make_leakage_test_dataset()
    groups = retrain.detect_column_groups(df)
    eval_cols = set(groups["evaluation_return_columns"])
    
    # Return columns should be detected as evaluation columns
    assert "net_forward_return" in eval_cols
    assert "gross_forward_return" in eval_cols


# ---------------------------------------------------------------------------
# Token Definition Verification
# ---------------------------------------------------------------------------

def test_strict_forbidden_tokens_not_empty() -> None:
    """STRICT_FORBIDDEN_TOKENS should not be empty."""
    assert len(retrain.STRICT_FORBIDDEN_TOKENS) > 0, "STRICT_FORBIDDEN_TOKENS should not be empty"


def test_future_leak_tokens_not_empty() -> None:
    """FUTURE_LEAK_TOKENS should not be empty."""
    assert len(retrain.FUTURE_LEAK_TOKENS) > 0, "FUTURE_LEAK_TOKENS should not be empty"


def test_forbidden_column_tokens_not_empty() -> None:
    """FORBIDDEN_COLUMN_TOKENS should not be empty."""
    assert len(retrain.FORBIDDEN_COLUMN_TOKENS) > 0, "FORBIDDEN_COLUMN_TOKENS should not be empty"


def test_tokens_cover_all_leakage_patterns() -> None:
    """Token definitions should cover common leakage patterns."""
    tokens = retrain.STRICT_FORBIDDEN_TOKENS + retrain.FUTURE_LEAK_TOKENS + retrain.FORBIDDEN_COLUMN_TOKENS
    
    # Key leakage patterns that should be covered
    required_patterns = ["future", "label", "pnl", "forward", "realized", "return", "target"]
    all_tokens_str = " ".join(tokens).lower()
    
    for pattern in required_patterns:
        assert any(pattern in token for token in tokens), f"'{pattern}' pattern should be covered by tokens"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_case_insensitive_blocking() -> None:
    """Forbidden token matching should be case-insensitive."""
    # Uppercase, lowercase, mixed case should all be blocked
    assert retrain._is_forbidden_feature_name("FUTURE_CLOSE")
    assert retrain._is_forbidden_feature_name("Future_Close")
    assert retrain._is_forbidden_feature_name("PNL_TOTAL")
    assert retrain._is_forbidden_feature_name("Realized_PnL")


def test_whitespace_handling() -> None:
    """Column names should be stripped before checking."""
    assert retrain._is_forbidden_feature_name("  future_close  ")
    assert retrain._is_forbidden_feature_name("net_forward_return  ")


def test_partial_match_blocking() -> None:
    """Forbidding should work on partial matches within column names."""
    # "label" in the middle of a column name should still be blocked
    assert retrain._is_forbidden_feature_name("my_custom_label_v1")
    assert retrain._is_forbidden_feature_name("trade_label_test")
    
    # "future" in the middle should be blocked
    assert retrain._is_forbidden_feature_name("price_after_future_moves")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])