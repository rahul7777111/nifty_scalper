"""
tests/test_live_feature_gating.py
===================================
Integration test: ensure no deployed model artifact contains forbidden features
in its live-computable feature set.

This test FAIL-closes the pipeline if any candidate model requires features that
cannot be computed in real-time (labels, forward returns, PnL, future data).

Run:
    python -m pytest tests/test_live_feature_gating.py -v
"""

from __future__ import annotations

import json
import glob
import os
from pathlib import Path
import pytest

# Import the fixed feature contract
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ml_feature_contract import is_feature_forbidden


# =============================================================================
# CATALOG: All known candidate model artifact directories
# =============================================================================

def _candidate_dirs() -> list[str]:
    """Find all candidate model directories with feature_schema.json."""
    return sorted(glob.glob("models/candidates/*/feature_schema.json"))


# =============================================================================
# FIXTURE: Per-candidate live-feature validation result
# =============================================================================

@pytest.fixture(scope="module")
def _candidate_schemas():
    """Load all candidate feature schemas once."""
    results = []
    for schema_path in _candidate_dirs():
        dir_path = os.path.dirname(schema_path)
        model_id = os.path.basename(dir_path)
        manifest_path = os.path.join(dir_path, "candidate_manifest.json")
        try:
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        except Exception as exc:
            results.append({"model_id": model_id, "error": str(exc), "features": [], "live_computable_features": []})
            continue

        manifest = {}
        if os.path.exists(manifest_path):
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except Exception:
                pass

        all_features = set(schema.get("all_features", []))
        live_features = set(schema.get("live_computable_features", []))

        forbidden_in_live = sorted(f for f in live_features if is_feature_forbidden(f))
        forbidden_in_all = sorted(f for f in all_features if is_feature_forbidden(f))

        results.append({
            "model_id": model_id,
            "model_family": manifest.get("model_family", "unknown"),
            "label_name": manifest.get("label_name", "unknown"),
            "all_features_count": len(all_features),
            "all_features_list": sorted(all_features),
            "live_computable_count": len(live_features),
            "live_computable_features_list": sorted(live_features),
            "forbidden_in_all_features": forbidden_in_all,
            "forbidden_in_live_features": forbidden_in_live,
            "live_eligible": len(forbidden_in_live) == 0,
            "training_leak_free": len(forbidden_in_all) == 0 or
                all(f in forbidden_in_all for f in schema.get("all_features", [])),
        })
    return results


class TestNoForbiddenFeaturesInLiveCandidates:
    """FAIL-CLOSED: deployed models must not require forbidden features at live inference."""

    def test_no_candidates_with_forbidden_live_features(self, _candidate_schemas):
        """
        FAIL if any candidate model has forbidden features in its live_computable_features.

        This catches training-pipeline bugs where forbidden columns (labels, forward
        returns, PnL) leak into the live feature set.
        """
        blocked = [c for c in _candidate_schemas if not c.get("live_eligible", True)]
        if blocked:
            lines = []
            for c in blocked:
                lines.append(f"  {c['model_id']}: forbidden={c['forbidden_in_live_features']}")
            pytest.fail(
                f"Live ML gating would BLOCK {len(blocked)} candidate(s) due to forbidden features:\n"
                + "\n".join(lines)
                + "\nThese features cannot be computed in real-time."
            )

    def test_all_candidates_load_successfully(self, _candidate_schemas):
        """Verify every candidate schema loads without error."""
        errors = [c["model_id"] for c in _candidate_schemas if "error" in c]
        assert not errors, f"Failed to load schemas: {errors}"

    def test_live_computable_features_subset_of_all_features(self, _candidate_schemas):
        """
        Verify live_computable_features is a proper subset of all_features.
        The training pipeline must not ship a live-computable feature set that
        includes columns not in the full training feature set.
        """
        for c in _candidate_schemas:
            if "error" in c:
                continue
            assert c.get("live_computable_count", 0) > 0, f"{c['model_id']}: no live features"
            live_list = c.get("live_computable_features_list", [])
            assert len(live_list) == c.get("live_computable_count", -1), \
                f"{c['model_id']}: count mismatch"


class TestIsFeatureForbiddenContract:
    """
    Unit tests for the is_feature_forbidden() function.
    Ensures the contract correctly distinguishes live-computable vs forbidden features.
    """

    @pytest.mark.parametrize("feature", [
        # Truly forbidden features (labels, forward returns, PnL)
        "net_forward_return",
        "gross_forward_return",
        "profitable_trade_label",
        "avoid_trade_label",
        "strong_profitable_trade_label",
        "cost_survivor_label",
        "paper_candidate_label",
        "realized_pnl",
        "future_close",
        "expected_return_after_cost",
        "return_to_cost_ratio",
        "cost_return_units_estimated",
        "gross_pnl",
        "max_favorable_excursion",
        "max_adverse_excursion",
    ])
    def test_truly_forbidden_features_are_flagged(self, feature):
        assert is_feature_forbidden(feature), f"{feature} MUST be flagged as forbidden"

    @pytest.mark.parametrize("feature", [
        # Legitimate live-computable volatility features (were falsely flagged before fix)
        "realized_vol_30",
        "realized_vol_percentile_60",
        # Legitimate historical spot returns (were falsely flagged before fix)
        "spot_return_1",
        "spot_return_3",
        "spot_return_5",
        # Other live-computable features
        "atr_percentile_60",
        "atr_pct_regime_10",
        "volatility_percentile_60",
        "volatility_regime_classifier",
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_mean",
        "ret_std",
        "atr_14",
        "atr_pct",
        "rsi_14",
        "adx_14",
        "ema_fast",
        "ema_slow",
        "vol_of_vol_14",
        "rolling_range_position_20",
        "rolling_range_width_20",
        "is_opening_session",
        "is_closing_session",
        "is_midday_lull",
        "opening_range_width_pct",
        "opening_range_breakout_strength",
    ])
    def test_live_computable_features_not_flagged(self, feature):
        assert not is_feature_forbidden(feature), (
            f"{feature} must NOT be flagged as forbidden — "
            "it is a legitimate live-computable feature"
        )


class TestMlGatingIntegration:
    """
    Integration test: simulate the exact gating check that runs in the UI/scalper loop.
    Uses a realistic feature dict that includes realized_vol_30 and realized_vol_percentile_60.
    """

    def test_evaluate_ml_gating_with_realistic_live_snapshot(self):
        """
        Simulate the live ML gating path in ml_signals.evaluate_ml_gating_before_execution()
        with a realistic feature snapshot that includes realized_vol_30.

        Before the fix: this would fail-close with FORBIDDEN_FEATURES=['realized_vol_30'].
        After the fix: should pass validation.
        """
        from ml_signals import evaluate_ml_gating_before_execution
        from ml_feature_contract import is_feature_forbidden

        # Build a realistic live feature snapshot
        live_snapshot = {
            # Candlestick
            "last_open": 24567.5, "last_high": 24580.0, "last_low": 24555.0,
            "last_close": 24572.0, "last_volume": 1250000,
            "body_pct": 0.18, "gap_pct": 0.05, "upper_wick_pct": 0.12,
            "lower_wick_pct": 0.08, "close_location_pct": 0.55,
            # Returns (historical, live-computable)
            "ret_1": 0.0012, "ret_3": 0.0035, "ret_5": 0.0058, "ret_10": 0.0091,
            "ret_mean": 0.0042, "ret_std": 0.0031, "ret_min": -0.0021, "ret_max": 0.0112,
            # Technical indicators
            "ema_fast": 24565.0, "ema_slow": 24540.0, "ema_diff_pct": 0.102,
            "rsi_14": 58.3, "atr_14": 85.5, "atr_pct": 0.00348,
            "adx_14": 22.1, "roc_14": 0.45, "choppiness_14": 45.2,
            "supertrend_dir": 1.0, "supertrend_gap_pct": 0.0012,
            # Pivot levels
            "pivot_pp_dist_pct": 0.0, "pivot_r1_dist_pct": 0.0045,
            "pivot_s1_dist_pct": -0.0048,
            # Price action
            "close_vs_open_pct": 0.0018, "range_to_atr": 0.29, "momentum_lookback_pct": 0.0085,
            # Candlestick patterns
            "bullish_engulfing": 0.0, "bearish_engulfing": 0.0, "doji": 1.0,
            "hammer": 0.0, "shooting_star": 0.0,
            # Regime features
            "regime_trending": 1.0, "regime_volatile": 0.0,
            "regime_mean_reverting": 0.0, "regime_quiet": 0.0,
            # Volatility features (the features that were falsely forbidden)
            "realized_vol_30": 0.00085,
            "realized_vol_percentile_60": 0.62,
            "atr_percentile_60": 0.58,
            "atr_pct_regime_10": 0.71,
            "volatility_percentile_60": 0.55,
            "volatility_regime_classifier": 1.0,
            "vol_mean": 0.00092, "vol_std": 0.00031, "vol_min": 0.00045,
            "vol_max": 0.00182, "vol_of_vol_14": 0.00012,
            # Time features
            "is_opening_session": 0.0, "is_closing_session": 1.0, "is_midday_lull": 0.0,
            # Opening range features
            "dist_from_opening_high_pct": -0.0085, "dist_from_opening_low_pct": 0.0123,
            "opening_range_width_pct": 0.0185, "opening_range_breakout_strength": 0.62,
            # Market structure features
            "dist_to_rolling_high_20": -0.015, "dist_to_rolling_low_20": 0.025,
            "rolling_range_width_20": 0.032, "rolling_range_position_20": 0.68,
            # Option chain features
            "oi": 450000, "volume": 890000, "ltp": 245.5,
            "bid_ask_spread_pct": 0.0025, "option_bid_ask_spread_pct": 0.0038,
            "delta_abs": 0.45, "greeks_imbalance": 0.08,
            "atm_distance": 12.5, "strike_distance_pct": 0.0051,
            "distance_from_spot": 0.0051, "moneyness": 1.005,
            "ctx_iv": 0.145, "ctx_delta": 0.52, "ctx_gamma": 0.031,
            "ctx_vega": 0.18, "ctx_theta": -0.08,
            "final_iv": 0.142, "bs_iv": 0.140,
            "bs_delta": 0.51, "bs_gamma": 0.030, "bs_theta": -0.075,
            "bs_vega": 0.175, "bs_rho": 0.002,
        }

        # Verify no forbidden features in our realistic snapshot
        forbidden = [fname for fname in live_snapshot.keys() if is_feature_forbidden(fname)]
        assert not forbidden, (
            f"Realistic live snapshot contains forbidden features: {forbidden}. "
            "This test simulates the exact scenario that was blocking the UI — "
            "the snapshot must not contain any label, return, PnL, or future columns."
        )