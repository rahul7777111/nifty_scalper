"""
ml_feature_contract.py
=======================
Shared feature contract module for NiftyScalper ML pipeline.

This module unifies training and live inference by enforcing a strict feature
contract that prevents training/inference feature mismatch.

Key principles:
- Features are versioned so retrain artifacts and live inference agree on feature set.
- Live inference FAILS CLOSED if required features are missing or forbidden features present.
- Forbidden tokens (labels, returns, PnL, future data) are enforced at BOTH training AND inference.

Version history:
- 1.0.0: Initial version with base feature set and forbidden token enforcement
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set, Tuple

# Contract version - must match between training and inference
CONTRACT_VERSION = "1.2.0"
CONTRACT_EPOCH = datetime(2026, 6, 19, 0, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# FORBIDDEN FEATURE PATTERNS
# These patterns match features that MUST NOT be used as live inference inputs.
# They are the canonical source of truth shared by both training and inference.
# ============================================================================

# Regex patterns for forbidden feature names (case-insensitive matching)
FORBIDDEN_FEATURE_PATTERNS: List[Pattern[str]] = [
    # Label columns (target variables)
    re.compile(r"_label(_v\d+)?$", re.IGNORECASE),
    re.compile(r"_target$", re.IGNORECASE),
    re.compile(r"_outcome$", re.IGNORECASE),
    
    # Return/PnL columns (future-looking or realized)
    re.compile(r"^future_", re.IGNORECASE),
    re.compile(r"_forward_return", re.IGNORECASE),
    re.compile(r"_return$", re.IGNORECASE),
    re.compile(r"realized_pnl", re.IGNORECASE),
    re.compile(r"trade_pnl", re.IGNORECASE),
    re.compile(r"net_pnl", re.IGNORECASE),
    re.compile(r"gross_pnl", re.IGNORECASE),
    re.compile(r"^pnl", re.IGNORECASE),
    
    # Trade result columns
    re.compile(r"_result$", re.IGNORECASE),
    re.compile(r"_exit$", re.IGNORECASE),
    re.compile(r"hit_target", re.IGNORECASE),
    re.compile(r"hit_sl", re.IGNORECASE),
    re.compile(r"mae$", re.IGNORECASE),
    re.compile(r"mfe$", re.IGNORECASE),
    
    # Other future-leaking features
    re.compile(r"_after_", re.IGNORECASE),
    re.compile(r"next_", re.IGNORECASE),
    re.compile(r"post_trade", re.IGNORECASE),
    re.compile(r"max_profit", re.IGNORECASE),
    re.compile(r"max_loss", re.IGNORECASE),
    re.compile(r"max_favorable_excursion", re.IGNORECASE),
    re.compile(r"max_adverse_excursion", re.IGNORECASE),
    re.compile(r"option_trade_success", re.IGNORECASE),
    re.compile(r"cost_adjusted_success", re.IGNORECASE),
    re.compile(r"ternary_trade_quality", re.IGNORECASE),
    re.compile(r"simulated_net_pnl", re.IGNORECASE),
    
    # Cost/expected return columns
    re.compile(r"expected_return_after_cost", re.IGNORECASE),
    re.compile(r"return_to_cost_ratio", re.IGNORECASE),
    re.compile(r"cost_return_units_estimated", re.IGNORECASE),
]

# Exact forbidden tokens (matched as substrings)
FORBIDDEN_EXACT_TOKENS: Set[str] = {
    "future_close",
    "gross_forward_return",
    "net_forward_return",
    "expected_return_after_cost",
    "return_to_cost_ratio",
    "cost_return_units_estimated",
}

# Legacy token sets from retrain_all_edge_models.py (for compatibility checking)
# NOTE: "realized" and "return" were removed from this list (2026-06-09) because
# they incorrectly flagged legitimate live-computable features:
# - "realized" flagged realized_vol_30, realized_vol_percentile_60 (historical vol)
# - "return" flagged spot_return_1/3/5 (historical spot returns)
# Specific regex patterns (_forward_return, realized_pnl) and FORBIDDEN_EXACT_TOKENS
# correctly target only truly forbidden return/PnL columns without false positives.
LEGACY_STRICT_FORBIDDEN_TOKENS = (
    "future", "next", "target", "label", "pnl",
    "profit", "loss", "outcome", "exit", "entry_result",
    "trade_result", "hit_target", "hit_sl", "mae", "mfe",
    "forward", "future_price", "future_return", "post_trade",
)

LEGACY_FUTURE_LEAK_TOKENS = (
    "_after_", "gross_pnl_", "net_pnl_", "max_favorable_excursion",
    "max_adverse_excursion", "option_trade_success_", "cost_adjusted_success_",
    "ternary_trade_quality_", "simulated_net_pnl_",
)

LEGACY_FORBIDDEN_COLUMN_TOKENS = (
    "pnl", "profit", "target", "label", "future", "forward",
    "next", "exit", "loss", "outcome", "next_return",
    "net_forward_return", "gross_forward_return",
    "trade_result", "post_entry_return", "max_profit", "max_loss",
)


# ============================================================================
# REQUIRED FEATURE SCHEMA
# These features are required for live inference. Any missing required feature
# will cause the validation to fail (fail-closed).
# ============================================================================

# Base features required from candles (OHLCV + derived)
# NOTE: These must match what build_market_feature_vector() ACTUALLY produces.
# Features excluded by EXCLUDED_MODEL_FEATURES in ml_pipeline.py are NOT included here:
# - ret_1, range_pct, roc_14, supertrend_gap_pct are EXCLUDED by ml_pipeline.py
# - ctx_time_sin, ctx_time_cos, price_to_spot_pct are also EXCLUDED
REQUIRED_CANDLE_FEATURES: Set[str] = {
    # Price data (last_* prefixed variants produced by build_market_feature_vector)
    "last_open", "last_high", "last_low", "last_close", "last_volume",
    
    # Candlestick patterns
    "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
    
    # Returns (ret_1, ret_3, ret_5, ret_10 produced but ret_1 is EXCLUDED by pipeline)
    # ret_mean/std/min/max from rolling_stats
    "ret_3", "ret_5", "ret_10", "ret_mean", "ret_std", "ret_min", "ret_max",
    
    # Technical indicators (roc_14, supertrend_gap_pct are EXCLUDED by pipeline)
    "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct",
    "adx_14", "choppiness_14", "supertrend_dir",
    
    # Pivot levels
    "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
    
    # Price action (range_pct is EXCLUDED by pipeline)
    "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
    
    # Candlestick structure (range_pct is EXCLUDED by pipeline)
    "body_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
    "close_location_pct",
}

# Regime features (categorical, one-hot encoded)
REQUIRED_REGIME_FEATURES: Set[str] = {
    "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
}

# Volatility regime features
REQUIRED_VOLATILITY_FEATURES: Set[str] = {
    "vol_mean", "vol_std", "vol_min", "vol_max", "vol_of_vol_14",
    "volatility_regime_classifier",
    "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60",
    "atr_pct_regime_10", "realized_vol_30",
}

# Option chain features (may be None if not available)
OPTION_CHAIN_FEATURES: Set[str] = {
    # Price data
    "ltp", "oi", "oi_CE", "oi_PE", "volume_CE", "volume_PE",
    "oi_change_pct", "volume_change_pct", "ce_pe_oi_ratio", "ce_pe_volume_ratio",
    
    # IV/Greeks
    "final_iv", "atm_distance", "strike_distance_pct", "distance_from_spot",
    "option_ltp", "strike_price",
    
    # Bid/ask
    "bid_ask_spread_pct", "option_bid_ask_spread_pct",
    
    # Context greeks
    "ctx_iv", "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
    "ctx_option_price", "delta_abs", "greeks_imbalance",
}

# Black-Scholes features (optional but tracked)
BS_FEATURES: Set[str] = {
    "bs_iv", "bs_delta", "bs_gamma", "bs_theta", "bs_vega", "bs_rho",
    "moneyness", "log_moneyness", "distance_from_atm", "distance_from_atm_pct",
    "intrinsic_value", "extrinsic_value", "time_to_expiry_days", "time_to_expiry_years",
    "bid_ask_spread", "bid_ask_spread_pct", "mid_price", "ltp_vs_mid_diff_pct",
    "greeks_quality_score",
}

# BS flag features (quality gates)
BS_FLAG_FEATURES: Set[str] = {
    "bs_iv_solve_ok", "bad_iv_flag", "bad_greek_flag", "wide_spread_flag",
    "low_price_flag", "deep_itm_flag", "deep_otm_flag", "row_enrichment_ok",
}

# Time-of-day features
TIME_FEATURES: Set[str] = {
    "is_opening_session", "is_closing_session", "is_midday_lull",
}

# Live-computable context / option metadata features
LIVE_CONTEXT_FEATURES: Set[str] = {
    "weekday", "month", "weekly", "dte_days", "is_weekly",
    "option_type_ce", "option_type_pe",
    "ctx_dte_norm", "ctx_volume_sma",
    "is_expiry_day", "is_near_expiry",
    "hl_change_pct", "oc_change_pct",
    "ctx_adx", "ctx_choppiness", "ctx_iv_change_pct", "ctx_iv_percentile",
    "ctx_spot", "ctx_time_cos", "ctx_time_sin", "ctx_trend_strength",
    "option_to_spot_pct", "theta_to_vega_ratio", "gamma_to_theta_ratio",
}

# Candle-derived optional features that are valid for live inference
OPTIONAL_CANDLE_FEATURES: Set[str] = {
    "range_pct", "ret_1", "roc_14", "supertrend_gap_pct",
}

# Rolling-history features available when enough live history exists
ROLLING_HISTORY_OPTIONAL_FEATURES: Set[str] = {
    "oi_z_5", "volume_z_5",
}

SPOT_CONTEXT_FEATURES: Set[str] = {
    "open_spot", "high_spot", "low_spot", "close_spot",
    "spot_close", "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap",
    "volume_spot", "weekday_spot",
}

# Strategy features computable from current/live history only
DIRECT_STRATEGY_FEATURES: Set[str] = {
    "trend_following_strength", "trend_following_confidence",
    "trend_following_ema_gap_pct", "trend_following_momentum_pct",
    "trend_following_breakout_score", "trend_following_pullback_score",
    "trend_following_buy_call", "trend_following_buy_put",
    "mean_reversion_zscore", "mean_reversion_entry_score",
    "mean_reversion_expected_reversion_pct", "mean_reversion_half_life_bars",
    "mean_reversion_buy_call", "mean_reversion_buy_put",
    "stat_arb_zscore", "stat_arb_confidence",
    "stat_arb_spread_pct", "stat_arb_hedge_ratio",
    "stat_arb_long_spread", "stat_arb_short_spread",
}

# Opening range features
OPENING_RANGE_FEATURES: Set[str] = {
    "dist_from_opening_high_pct", "dist_from_opening_low_pct",
    "opening_range_width_pct", "opening_range_breakout_strength",
}

# Market structure features
MARKET_STRUCTURE_FEATURES: Set[str] = {
    "dist_to_rolling_high_20", "dist_to_rolling_low_20",
    "rolling_range_width_20", "rolling_range_position_20",
}

# Union of all required features for live inference
REQUIRED_FEATURE_SCHEMA: Dict[str, str] = {
    **{f: "float" for f in REQUIRED_CANDLE_FEATURES},
    **{f: "float" for f in REQUIRED_REGIME_FEATURES},
    **{f: "float" for f in REQUIRED_VOLATILITY_FEATURES},
    **{f: "float" for f in OPTION_CHAIN_FEATURES},
    **{f: "float" for f in BS_FEATURES},
    **{f: "float" for f in BS_FLAG_FEATURES},
    **{f: "float" for f in TIME_FEATURES},
    **{f: "float" for f in LIVE_CONTEXT_FEATURES},
    **{f: "float" for f in OPTIONAL_CANDLE_FEATURES},
    **{f: "float" for f in ROLLING_HISTORY_OPTIONAL_FEATURES},
    **{f: "float" for f in DIRECT_STRATEGY_FEATURES},
    **{f: "float" for f in SPOT_CONTEXT_FEATURES},
    **{f: "float" for f in OPENING_RANGE_FEATURES},
    **{f: "float" for f in MARKET_STRUCTURE_FEATURES},
}

# Core required features that must be present for any prediction
CORE_REQUIRED_FEATURES: Set[str] = REQUIRED_CANDLE_FEATURES | REQUIRED_REGIME_FEATURES


# ============================================================================
# ALLOWED LIVE FEATURES
# This is the canonical list of features that can be used at live inference time.
# It excludes all forbidden patterns and is versioned.
# ============================================================================

def _build_allowed_live_features() -> Set[str]:
    """Build the set of allowed live features from required schema."""
    allowed = set(REQUIRED_FEATURE_SCHEMA.keys())
    allowed.difference_update(BS_FEATURES - OPTION_CHAIN_FEATURES)
    allowed.difference_update(BS_FLAG_FEATURES)
    # Add common derived features that might be computed at inference
    allowed.update({
        "atr_5", "atr_10", "atr_20",
        "rsi_7", "rsi_21",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "volume_ratio", "volume_change",
        "iv_rank", "iv_percentile",
        "skew", "kurtosis",
        "option_price", "spot", "dte_norm",
    })
    # Add raw OHLCV features - these are available in live candle data
    # even though the training pipeline (ml_pipeline.py) uses last_* prefixed versions.
    # live_feature_builder.py produces both prefixed and unprefixed OHLCV.
    allowed.update({"open", "high", "low", "close", "volume"})
    return allowed

ALLOWED_LIVE_FEATURES: Set[str] = _build_allowed_live_features()


# ============================================================================
# FEATURE CONTRACT RESULT TYPES
# ============================================================================

@dataclass
class FeatureCoverageResult:
    """Result of feature coverage computation."""
    coverage_pct: float
    total_required: int
    available_count: int
    missing_features: List[str]
    forbidden_features: List[str]
    available_features: List[str]
    all_valid: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_pct": self.coverage_pct,
            "total_required": self.total_required,
            "available_count": self.available_count,
            "missing_features": self.missing_features,
            "forbidden_features": self.forbidden_features,
            "available_features": self.available_features,
            "all_valid": self.all_valid,
        }


@dataclass
class FeatureValidationResult:
    """Result of full feature validation (fail-closed)."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    coverage_result: Optional[FeatureCoverageResult] = None
    contract_version: str = CONTRACT_VERSION
    validated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "coverage_result": self.coverage_result.to_dict() if self.coverage_result else None,
            "contract_version": self.contract_version,
            "validated_at": self.validated_at,
        }


# ============================================================================
# CORE CONTRACT FUNCTIONS
# ============================================================================

def is_feature_forbidden(feature_name: str) -> bool:
    """
    Check if a feature name is forbidden.
    
    A feature is forbidden if it matches any of the forbidden patterns or
    is an exact match for a forbidden token.
    
    Args:
        feature_name: The feature name to check.
        
    Returns:
        True if the feature is forbidden, False otherwise.
    """
    if not feature_name:
        return False
    
    # Check exact forbidden tokens
    if feature_name in FORBIDDEN_EXACT_TOKENS:
        return True
    
    # Check against legacy forbidden tokens (substring match)
    name_lower = feature_name.lower()
    for token in LEGACY_STRICT_FORBIDDEN_TOKENS:
        if token in name_lower:
            return True
    
    for token in LEGACY_FUTURE_LEAK_TOKENS:
        if token in name_lower:
            return True
    
    for token in LEGACY_FORBIDDEN_COLUMN_TOKENS:
        if token in name_lower:
            return True
    
    # Check against regex patterns
    for pattern in FORBIDDEN_FEATURE_PATTERNS:
        if pattern.search(feature_name):
            return True
    
    return False


def compute_feature_coverage(
    features_dict: Dict[str, Any],
    required_features: Optional[Set[str]] = None,
    check_forbidden: bool = True,
) -> FeatureCoverageResult:
    """
    Compute feature coverage for live inference.
    
    Args:
        features_dict: Dict of feature name -> value from live snapshot.
        required_features: Set of required feature names (defaults to CORE_REQUIRED_FEATURES).
        check_forbidden: If True, also check for forbidden features.
        
    Returns:
        FeatureCoverageResult with coverage statistics.
    """
    if required_features is None:
        required_features = CORE_REQUIRED_FEATURES
    
    feature_names = set(features_dict.keys())
    
    # Check for forbidden features
    forbidden_features: List[str] = []
    if check_forbidden:
        for fname in feature_names:
            if is_feature_forbidden(fname):
                forbidden_features.append(fname)
    
    # Compute missing features
    missing_features = [f for f in required_features if f not in feature_names]
    
    # Compute available features
    available_features = [f for f in required_features if f in feature_names]
    
    # Compute coverage percentage
    total_required = len(required_features)
    available_count = len(available_features)
    coverage_pct = (available_count / total_required * 100) if total_required > 0 else 0.0
    
    all_valid = len(missing_features) == 0 and len(forbidden_features) == 0
    
    return FeatureCoverageResult(
        coverage_pct=round(coverage_pct, 2),
        total_required=total_required,
        available_count=available_count,
        missing_features=missing_features,
        forbidden_features=forbidden_features,
        available_features=available_features,
        all_valid=all_valid,
    )


def validate_live_features(
    features_dict: Dict[str, Any],
    required_features: Optional[Set[str]] = None,
    check_forbidden: bool = True,
    min_coverage_pct: float = 95.0,
) -> Tuple[bool, List[str]]:
    """
    Validate live features for inference (fail-closed).
    
    This function FAILS CLOSED - it returns (False, errors) if:
    1. Any required feature is missing
    2. Any forbidden feature is detected
    3. Coverage is below minimum threshold
    
    Args:
        features_dict: Dict of feature name -> value from live snapshot.
        required_features: Set of required feature names (defaults to CORE_REQUIRED_FEATURES).
        check_forbidden: If True, also check for forbidden features.
        min_coverage_pct: Minimum coverage required (default 95%).
        
    Returns:
        Tuple of (is_valid, list of error messages).
        If is_valid is False, errors contains the reasons for failure.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    if required_features is None:
        required_features = CORE_REQUIRED_FEATURES
    
    # Step 1: Check for forbidden features (always a hard failure)
    if check_forbidden:
        forbidden_found = []
        for fname in features_dict.keys():
            if is_feature_forbidden(fname):
                forbidden_found.append(fname)
        
        if forbidden_found:
            errors.append(
                f"FORBIDDEN_FEATURES_DETECTED: {forbidden_found}. "
                "Forbidden features (labels, returns, PnL, future data) "
                "must never be used at live inference."
            )
    
    # Step 2: Compute coverage
    coverage_result = compute_feature_coverage(
        features_dict, 
        required_features=required_features,
        check_forbidden=False,  # Already checked above
    )
    
    # Step 3: Check for missing required features (hard failure)
    if coverage_result.missing_features:
        errors.append(
            f"MISSING_REQUIRED_FEATURES: {coverage_result.missing_features}. "
            "All required features must be present for inference."
        )
    
    # Step 4: Check coverage threshold (hard failure)
    if coverage_result.coverage_pct < min_coverage_pct:
        errors.append(
            f"COVERAGE_BELOW_THRESHOLD: {coverage_result.coverage_pct:.1f}% < {min_coverage_pct}%. "
            f"Missing {len(coverage_result.missing_features)} required features."
        )
    
    # Step 5: Warn about optional features that are missing
    all_optional = set(REQUIRED_FEATURE_SCHEMA.keys()) - required_features
    missing_optional = [f for f in all_optional if f not in features_dict]
    if missing_optional:
        warnings.append(f"OPTIONAL_FEATURES_MISSING: {missing_optional[:10]}...")
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        return (False, errors)
    
    return (True, [])


def get_live_feature_names() -> List[str]:
    """
    Get the canonical list of allowed live feature names.
    
    Returns:
        List of feature names that are valid for live inference.
    """
    return sorted(list(ALLOWED_LIVE_FEATURES))


def get_feature_contract_info() -> Dict[str, Any]:
    """
    Get information about the current feature contract.
    
    Returns:
        Dict with contract version, feature counts, and forbidden token counts.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_epoch": CONTRACT_EPOCH.isoformat(),
        "total_allowed_features": len(ALLOWED_LIVE_FEATURES),
        "total_required_features": len(REQUIRED_FEATURE_SCHEMA),
        "core_required_features": len(CORE_REQUIRED_FEATURES),
        "forbidden_patterns_count": len(FORBIDDEN_FEATURE_PATTERNS),
        "forbidden_exact_tokens_count": len(FORBIDDEN_EXACT_TOKENS),
    }


def align_features_to_contract(
    features_dict: Dict[str, Any],
    model_features: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Align live features to the feature contract and model requirements.
    
    This function:
    1. Validates against the feature contract
    2. Filters to only allowed features
    3. Returns aligned feature vector for model prediction
    
    Args:
        features_dict: Raw feature dict from live snapshot.
        model_features: Optional list of model-required features.
                       If provided, only these features will be returned.
                       If None, returns all contract-compliant features.
    
    Returns:
        Tuple of (aligned_features, warnings).
        aligned_features: Dict of feature_name -> float value.
        warnings: List of warning messages.
    """
    warnings: List[str] = []
    aligned: Dict[str, float] = {}
    
    # Determine which features to include
    target_features = set(model_features) if model_features else ALLOWED_LIVE_FEATURES
    
    for fname, fvalue in features_dict.items():
        # Skip forbidden features
        if is_feature_forbidden(fname):
            warnings.append(f"SKIPPED_FORBIDDEN: {fname}")
            continue
        
        # Skip if not in target features
        if fname not in target_features:
            continue
        
        # Convert to float
        try:
            aligned[fname] = float(fvalue) if fvalue is not None else 0.0
        except (ValueError, TypeError):
            warnings.append(f"SKIPPED_INVALID_VALUE: {fname}={fvalue}")
            aligned[fname] = 0.0
    
    # Fill missing model features with 0.0 (if model_features specified)
    if model_features:
        for fname in model_features:
            if fname not in aligned:
                warnings.append(f"FILLED_MISSING: {fname}=0.0")
                aligned[fname] = 0.0
    
    return aligned, warnings


def verify_training_feature_compatibility(
    training_features: Sequence[str],
    contract_version: str = CONTRACT_VERSION,
) -> Tuple[bool, List[str]]:
    """
    Verify that training features are compatible with the live inference contract.
    
    Call this at the start of training to ensure the trained model can be used
    for live inference.
    
    Args:
        training_features: List of feature names used in training.
        contract_version: Expected contract version.
        
    Returns:
        Tuple of (is_compatible, list of issues).
    """
    issues: List[str] = []
    
    # Check version
    if contract_version != CONTRACT_VERSION:
        issues.append(
            f"CONTRACT_VERSION_MISMATCH: expected {contract_version}, "
            f"got {CONTRACT_VERSION}"
        )
    
    # Check for forbidden features in training set
    forbidden_in_training = [f for f in training_features if is_feature_forbidden(f)]
    if forbidden_in_training:
        issues.append(
            f"FORBIDDEN_FEATURES_IN_TRAINING: {forbidden_in_training}. "
            "Training features must not include labels, returns, PnL, or future data."
        )
    
    # Check for features not in the allowed set
    unknown_features = [
        f for f in training_features 
        if f not in ALLOWED_LIVE_FEATURES and not is_feature_forbidden(f)
    ]
    if unknown_features:
        issues.append(
            f"UNKNOWN_FEATURES_IN_TRAINING: {unknown_features}. "
            "These features are not in the live inference contract."
        )
    
    is_compatible = len(issues) == 0
    return is_compatible, issues


# ============================================================================
# FEATURE CONTRACT VALIDATION FOR SPECIFIC USE CASES
# ============================================================================

def validate_model_features_for_live_inference(
    model_feature_names: List[str],
) -> Tuple[bool, List[str]]:
    """
    Validate that a model's feature set is suitable for live inference.
    
    Args:
        model_feature_names: List of feature names from a trained model.
        
    Returns:
        Tuple of (is_valid, list of issues).
    """
    issues: List[str] = []
    
    # Check for forbidden features
    forbidden = [f for f in model_feature_names if is_feature_forbidden(f)]
    if forbidden:
        issues.append(f"MODEL_CONTAINS_FORBIDDEN_FEATURES: {forbidden}")
    
    # Check for required features
    missing_required = [f for f in CORE_REQUIRED_FEATURES if f not in model_feature_names]
    if missing_required:
        issues.append(f"MODEL_MISSING_REQUIRED_FEATURES: {missing_required}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def create_feature_contract_summary() -> Dict[str, Any]:
    """
    Create a comprehensive summary of the feature contract state.
    
    Returns:
        Dict with contract details, feature counts, and statistics.
    """
    info = get_feature_contract_info()
    
    # Analyze features by category
    category_counts = {
        "required_candle": len(REQUIRED_CANDLE_FEATURES),
        "required_regime": len(REQUIRED_REGIME_FEATURES),
        "required_volatility": len(REQUIRED_VOLATILITY_FEATURES),
        "option_chain": len(OPTION_CHAIN_FEATURES),
        "black_scholes": len(BS_FEATURES),
        "time_features": len(TIME_FEATURES),
        "opening_range": len(OPENING_RANGE_FEATURES),
        "market_structure": len(MARKET_STRUCTURE_FEATURES),
    }
    
    return {
        **info,
        "categories": category_counts,
        "forbidden_tokens_sample": list(FORBIDDEN_EXACT_TOKENS)[:10],
        "required_features_sample": sorted(list(CORE_REQUIRED_FEATURES))[:20],
    }
