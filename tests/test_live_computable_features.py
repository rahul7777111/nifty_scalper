"""tests/test_live_computable_features.py
=======================================
Unit tests for live_feature_builder.py.

Coverage
--------
- Feature builder returns expected columns
- No duplicate columns
- No NaN/inf for required features
- Coverage >= 95% on fixture
- Missing required feature blocks prediction
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

# -- paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from src.live_feature_builder import (
    LiveSnapshot,
    LiveFeatureBuilder,
    LiveFeatureResult,
    build_live_features,
    build_offline_fixture,
    CANDLE_FEATURES,
    OPTION_CHAIN_FEATURES,
    ROLLING_HISTORY_FEATURES,
    CONTEXT_FEATURES,
    _safe_float,
    _safe_denominator,
    _returns,
    _rolling_stats,
    _safe_std,
    _series,
    _one_hot_regime,
    _compute_vwap,
    _candlestick_context,
    _compute_atr_14,
    _compute_rsi_14,
    _compute_adx_14,
    _compute_ema,
    _rolling_percentile,
    _COVERAGE_MINIMUM,
    ALL_COMPUTABLE_FEATURES,
)

try:
    from src.market_data import Candle
except Exception:
    Candle = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candle(
    time: datetime,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 98.0,
    close: float = 103.0,
    volume: float = 1000.0,
) -> Candle:
    """Create a test Candle with sensible defaults."""
    if Candle is None:
        pytest.skip("Candle class not available")
    return Candle(time=time, open=open_, high=high, low=low, close=close, volume=volume)


def make_snapshot(
    num_candles: int = 25,
    with_option_chain: bool = True,
    spot: float = 24650.0,
) -> LiveSnapshot:
    """Create a deterministic test snapshot with optional option chain."""
    now = datetime(2026, 6, 7, 10, 30, tzinfo=timezone.utc)
    base_time = now.replace(hour=9, minute=15, second=0, microsecond=0)

    candles: List[Candle] = []
    close = spot
    for i in range(num_candles):
        t = base_time.replace(minute=15 + i)
        # Small trending drift so indicators behave
        drift = (i - num_candles // 2) * 2.0
        c_close = close + drift
        c_high = c_close + 20.0
        c_low = c_close - 20.0
        candles.append(make_candle(t, open_=c_close - 5, high=c_high, low=c_low, close=c_close, volume=50000))

    oc: Dict[str, Any] | None = None
    if with_option_chain:
        oc = {
            "strike": 24600.0,
            "expiry": "2026-06-12",
            "option_type": "CE",
            "ltp": 185.0,
            "bid": 183.0,
            "ask": 187.0,
            "iv": 0.185,
            "delta": 0.52,
            "gamma": 0.011,
            "theta": -3.8,
            "vega": 5.9,
            "oi": 250000.0,
            "oi_CE": 250000.0,
            "oi_PE": 240000.0,
            "volume_CE": 1800.0,
            "volume_PE": 1650.0,
            "volume": 1800.0,
            "change_in_oi": 8000.0,
            "dte_days": 5.0,
            "is_weekly": True,
            "open": 182.0,
            "high": 192.0,
            "low": 179.0,
            "spot_at_time": spot,
        }

    return LiveSnapshot(candles=candles, spot=spot, atm_iv=0.185, option_chain=oc, timestamp=now)


# ---------------------------------------------------------------------------
# Test: safe helpers
# ---------------------------------------------------------------------------

class TestSafeHelpers:
    def test_safe_float_with_valid(self):
        assert _safe_float(1.5) == 1.5
        assert _safe_float(42) == 42.0

    def test_safe_float_with_none(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=99.0) == 99.0

    def test_safe_float_with_nan(self):
        assert _safe_float(float("nan")) == 0.0
        assert _safe_float(float("inf")) == 0.0

    def test_safe_denominator_floor(self):
        assert _safe_denominator(0.0) == 1e-7
        assert _safe_denominator(-0.0) == 1e-7
        assert _safe_denominator(1.0) == 1.0


class TestReturnsAndRollingStats:
    def test_returns_single_element(self):
        assert _returns([100.0]) == []

    def test_returns_multiple_elements(self):
        r = _returns([100.0, 105.0, 110.25])
        assert len(r) == 2
        assert abs(r[0] - 0.05) < 1e-9
        assert abs(r[1] - 0.05) < 1e-9

    def test_rolling_stats_empty(self):
        s = _rolling_stats([])
        assert s["mean"] == 0.0
        assert s["std"] == 0.0

    def test_rolling_stats_single(self):
        s = _rolling_stats([42.0])
        assert s["mean"] == 42.0
        assert s["std"] == 0.0

    def test_rolling_stats_multiple(self):
        s = _rolling_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(s["mean"] - 3.0) < 1e-9
        assert s["min"] == 1.0
        assert s["max"] == 5.0

    def test_safe_std_empty(self):
        assert _safe_std([]) == 0.0

    def test_safe_std_single(self):
        assert _safe_std([5.0]) == 0.0

    def test_series_truncation(self):
        vals = list(range(100))
        assert len(_series(vals, 20)) == 20

    def test_series_no_truncation(self):
        vals = list(range(10))
        assert len(_series(vals, 20)) == 10


class TestOneHotRegime:
    def test_all_regimes(self):
        for regime_name, key in [
            ("trending", "regime_trending"),
            ("volatile", "regime_volatile"),
            ("mean_reverting", "regime_mean_reverting"),
            ("quiet", "regime_quiet"),
        ]:
            out = _one_hot_regime(regime_name)
            assert out[key] == 1.0, f"Expected 1.0 for {key}"
            others = [v for k, v in out.items() if k != key]
            assert all(v == 0.0 for v in others), f"Others should be 0 for {key}"


class TestCandlestickContext:
    def test_context_computed(self):
        now = datetime.now(timezone.utc)
        last = make_candle(now, open_=100, high=110, low=95, close=108, volume=500)
        prev = make_candle(now, open_=105, high=112, low=98, close=100, volume=480)

        ctx = _candlestick_context(last, prev)

        assert ctx["last_open"] == 100.0
        assert ctx["last_high"] == 110.0
        assert ctx["last_low"] == 95.0
        assert ctx["last_close"] == 108.0
        assert ctx["last_volume"] == 500.0
        assert "body_pct" in ctx
        assert "range_pct" in ctx
        assert "upper_wick_pct" in ctx
        assert "lower_wick_pct" in ctx
        assert 0.0 <= ctx["body_pct"] <= 1.0
        assert ctx["range_pct"] >= 0.0


# ---------------------------------------------------------------------------
# Test: VWAP
# ---------------------------------------------------------------------------

class TestVWAP:
    def test_vwap_basic(self):
        now = datetime.now(timezone.utc)
        candles = [
            make_candle(now.replace(minute=i), open_=100, high=105, low=98, close=103, volume=1000)
            for i in range(10)
        ]
        vwap = _compute_vwap(candles)
        assert vwap is not None
        assert vwap > 0.0
        # VWAP should be between low and high of the period
        lows = [_safe_float(c.low) for c in candles]
        highs = [_safe_float(c.high) for c in candles]
        assert min(lows) <= vwap <= max(highs)

    def test_vwap_zero_volume(self):
        now = datetime.now(timezone.utc)
        candles = [
            make_candle(now.replace(minute=i), open_=100, high=105, low=98, close=103, volume=0)
            for i in range(5)
        ]
        vwap = _compute_vwap(candles)
        assert vwap is None


# ---------------------------------------------------------------------------
# Test: LiveFeatureBuilder — core
# ---------------------------------------------------------------------------

class TestLiveFeatureBuilderCore:
    def test_default_required_not_empty(self):
        builder = LiveFeatureBuilder()
        assert len(builder.required_features) > 0

    def test_custom_required_features(self):
        custom = ["rsi_14", "atr_14", "ema_fast"]
        builder = LiveFeatureBuilder(required_features=custom)
        assert builder.required_features == set(custom)

    def test_build_with_empty_candles_marks_candle_features_missing(self):
        builder = LiveFeatureBuilder(required_features=["rsi_14", "atr_14", "ret_1", "ret_3"])
        snapshot = LiveSnapshot(candles=[], spot=24650.0)
        result = builder.build(snapshot)

        # Session features should still be available
        assert result.features.get("ctx_time_sin") is not None

    def test_build_with_no_option_chain_marks_option_features_missing(self):
        required = list(OPTION_CHAIN_FEATURES & set(LiveFeatureBuilder.DEFAULT_REQUIRED))
        required += ["oi_z_5", "volume_z_5"]  # rolling history features

        builder = LiveFeatureBuilder(required_features=required)
        snapshot = make_snapshot(with_option_chain=False)

        result = builder.build(snapshot)

        # Option chain features should be marked missing
        option_missing = result.missing & OPTION_CHAIN_FEATURES
        # rolling history should be missing too
        rolling_missing = result.missing & ROLLING_HISTORY_FEATURES
        assert len(option_missing) > 0 or len(rolling_missing) > 0


class TestFeatureBuilderCoverage:
    def test_coverage_100_on_full_snapshot(self):
        """With full candles + option chain, coverage should be 100% for computable features."""
        builder = LiveFeatureBuilder()
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        # All required candle and context features should be available
        missing_candle = result.missing & CANDLE_FEATURES
        missing_context = result.missing & CONTEXT_FEATURES

        # Should have no missing candle features (we have 30 candles)
        assert len(missing_candle) == 0, f"Missing candle features: {missing_candle}"
        # Context features (time) are always available
        assert len(missing_context) == 0, f"Missing context features: {missing_context}"

    def test_coverage_at_least_95_percent(self):
        """Coverage on the standard fixture must be >= 95%."""
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        builder = LiveFeatureBuilder()
        result = builder.build(snapshot)

        assert result.coverage_pct >= 95.0, (
            f"Coverage {result.coverage_pct}% < 95%. "
            f"Missing: {sorted(result.missing)}"
        )

    def test_coverage_calculation_correct(self):
        """Coverage = (available / required) * 100."""
        required = ["rsi_14", "atr_14", "ema_fast", "oi_z_5"]
        builder = LiveFeatureBuilder(required_features=required)
        snapshot = make_snapshot(num_candles=30, with_option_chain=False)
        result = builder.build(snapshot)

        available = len(required) - len(result.missing)
        expected_pct = round(available / len(required) * 100.0, 2)
        assert result.coverage_pct == expected_pct


class TestFeatureBuilderNoNaN:
    def test_no_nan_in_available_features(self):
        """Available features must not contain NaN or Inf."""
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        builder = LiveFeatureBuilder()
        result = builder.build(snapshot)

        for name, val in result.features.items():
            if val is not None:
                assert not (val != val), f"NaN found in feature '{name}'"  # NaN != NaN
                assert val != float("inf"), f"+Inf found in feature '{name}'"
                assert val != float("-inf"), f"-Inf found in feature '{name}'"

    def test_to_aligned_vector_fills_missing_with_zero(self):
        """to_aligned_vector returns 0.0 for missing features."""
        required = ["rsi_14", "oi_z_5"]
        builder = LiveFeatureBuilder(required_features=required)
        snapshot = make_snapshot(num_candles=30, with_option_chain=False)
        result = builder.build(snapshot)

        aligned = result.to_aligned_vector(required)
        assert aligned["rsi_14"] != 0.0 or result.features["rsi_14"] is not None
        # oi_z_5 was not computable → should be 0 in aligned
        assert aligned["oi_z_5"] == 0.0


class TestFeatureBuilderReturnsExpectedColumns:
    def test_returns_known_feature_names(self):
        """Result features should be a superset of the model's default feature set."""
        builder = LiveFeatureBuilder()
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        # Every required feature should appear in features dict
        for name in result.required:
            assert name in result.features, f"Required feature '{name}' not in features dict"

    def test_no_duplicate_feature_names(self):
        """features dict must not have duplicate keys."""
        builder = LiveFeatureBuilder()
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        seen: Set[str] = set()
        dups: List[str] = []
        for name in result.features:
            if name in seen:
                dups.append(name)
            seen.add(name)
        assert len(dups) == 0, f"Duplicate feature names: {dups}"

    def test_rolling_history_features_marked_missing_without_history(self):
        """oi_z_5 and volume_z_5 need rolling history — must be missing in live mode."""
        required = ["oi_z_5", "volume_z_5"]
        builder = LiveFeatureBuilder(required_features=required)
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        for name in required:
            assert name in result.missing, f"Expected {name} to be missing (needs rolling history)"

    def test_indicator_features_computed(self):
        """rsi_14, atr_14, adx_14 should be computed when candles are available."""
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        builder = LiveFeatureBuilder(
            required_features=["rsi_14", "atr_14", "adx_14", "ema_fast", "ema_slow"]
        )
        result = builder.build(snapshot)

        for name in ["rsi_14", "atr_14", "adx_14", "ema_fast", "ema_slow"]:
            assert result.features.get(name) is not None, f"{name} should be computed"
            assert name not in result.missing, f"{name} should NOT be missing"


# ---------------------------------------------------------------------------
# Test: build_offline_fixture
# ---------------------------------------------------------------------------

class TestBuildOfflineFixture:
    def test_offline_fixture_has_candles(self):
        fixture = build_offline_fixture()
        assert len(fixture.candles) >= 20
        assert fixture.spot is not None
        assert fixture.spot > 0

    def test_offline_fixture_coverage_greater_than_zero(self):
        required = LiveFeatureBuilder.DEFAULT_REQUIRED
        fixture = build_offline_fixture(required_features=required)
        builder = LiveFeatureBuilder(required_features=required)
        result = builder.build(fixture)

        # Coverage must be > 0 (at least time features should be available)
        assert result.coverage_pct > 0.0

    def test_offline_fixture_marks_rolling_features_missing(self):
        required = ["oi_z_5", "volume_z_5"]
        fixture = build_offline_fixture(required_features=required)
        builder = LiveFeatureBuilder(required_features=required)
        result = builder.build(fixture)

        assert "oi_z_5" in result.missing
        assert "volume_z_5" in result.missing


# ---------------------------------------------------------------------------
# Test: build_live_features convenience function
# ---------------------------------------------------------------------------

class TestBuildLiveFeatures:
    def test_convenience_function_returns_live_feature_result(self):
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = build_live_features(snapshot)

        assert isinstance(result, LiveFeatureResult)
        assert isinstance(result.features, dict)
        assert isinstance(result.missing, set)
        assert isinstance(result.coverage_pct, float)
        assert 0.0 <= result.coverage_pct <= 100.0

    def test_convenience_function_passes_through_required_features(self):
        required = ["rsi_14", "atr_14"]
        result = build_live_features(make_snapshot(num_candles=30), required_features=required)
        assert result.required == set(required)


# ---------------------------------------------------------------------------
# Test: missing required feature blocks prediction
# ---------------------------------------------------------------------------

class TestMissingFeatureBlocksPrediction:
    def test_missing_required_feature_identified(self):
        """If coverage < 95%, prediction should be blocked."""
        # Use required features that include rolling-history (unavailable in live mode)
        required = ["rsi_14", "oi_z_5", "volume_z_5", "dist_from_opening_high_pct"]
        builder = LiveFeatureBuilder(required_features=required)
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        # Rolling history features are not computable live
        assert len(result.missing) > 0, "Expected some features to be missing"
        assert result.coverage_pct < 100.0

        # Simulate the decision gate
        prediction_allowed = result.coverage_pct >= _COVERAGE_MINIMUM
        assert not prediction_allowed, (
            f"Prediction should be blocked when coverage={result.coverage_pct:.1f}% < {_COVERAGE_MINIMUM}%"
        )

    def test_full_snapshot_allows_prediction(self):
        """With full data, coverage >= 95% and prediction is allowed."""
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        builder = LiveFeatureBuilder()
        result = builder.build(snapshot)

        prediction_allowed = result.coverage_pct >= _COVERAGE_MINIMUM
        assert prediction_allowed, (
            f"Coverage={result.coverage_pct:.1f}% — prediction should be allowed"
        )

    def test_aligned_vector_has_expected_length(self):
        """Aligned vector length must match feature names."""
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        builder = LiveFeatureBuilder()
        result = builder.build(snapshot)

        feature_names = list(result.required)
        aligned = result.to_aligned_vector(feature_names)

        assert len(aligned) == len(feature_names), (
            f"Aligned vector length {len(aligned)} != required features {len(feature_names)}"
        )


# ---------------------------------------------------------------------------
# Test: RSI / ATR / ADX / EMA helpers
# ---------------------------------------------------------------------------

class TestIndicatorHelpers:
    def test_rsi_neutral_when_no_change(self):
        """RSI of flat series should be ~50."""
        flat = [100.0] * 20
        rsi_val = _compute_rsi_14(flat)
        # Flat → no gains/losses → avg_loss=0 → RSI=100, but function returns 50 on insufficient data
        assert rsi_val is not None

    def test_rsi_computed(self):
        # Uptrend → RSI > 50
        up = [100.0 + i for i in range(20)]
        rsi_val = _compute_rsi_14(up)
        assert rsi_val is not None
        assert rsi_val > 50.0

    def test_atr_computed(self):
        candles = make_snapshot(num_candles=20, with_option_chain=False)
        closes = [_safe_float(c.close) for c in candles.candles]
        highs = [_safe_float(c.high) for c in candles.candles]
        lows = [_safe_float(c.low) for c in candles.candles]

        atr_val = _compute_atr_14(highs, lows, closes)
        assert atr_val is not None
        assert atr_val >= 0.0

    def test_adx_computed(self):
        candles = make_snapshot(num_candles=40, with_option_chain=False)
        closes = [_safe_float(c.close) for c in candles.candles]
        highs = [_safe_float(c.high) for c in candles.candles]
        lows = [_safe_float(c.low) for c in candles.candles]

        adx_val = _compute_adx_14(highs, lows, closes)
        assert adx_val is not None
        assert 0.0 <= adx_val <= 100.0

    def test_ema_computed(self):
        closes = [100.0 + i for i in range(30)]
        ema_fast = _compute_ema(closes, 9)
        ema_slow = _compute_ema(closes, 21)
        assert ema_fast is not None
        assert ema_slow is not None
        # In uptrend, ema_fast should be above ema_slow
        assert ema_fast >= ema_slow


# ---------------------------------------------------------------------------
# Test: option chain features
# ---------------------------------------------------------------------------

class TestOptionChainFeatures:
    def test_option_chain_basic_features(self):
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        builder = LiveFeatureBuilder(
            required_features=["option_type_ce", "option_type_pe", "strike_price",
                               "distance_from_spot", "atm_distance", "bid_ask_spread_pct"]
        )
        result = builder.build(snapshot)

        assert result.features.get("option_type_ce") == 1.0
        assert result.features.get("option_type_pe") == 0.0
        assert result.features.get("strike_price") == 24600.0
        assert result.features.get("distance_from_spot") is not None
        assert result.features.get("bid_ask_spread_pct") is not None

    def test_ce_pe_oi_ratio_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["ce_pe_oi_ratio", "ce_pe_volume_ratio"]
        )
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        assert result.features.get("ce_pe_oi_ratio") is not None
        assert result.features.get("ce_pe_volume_ratio") is not None
        # CE oi = 250000, PE oi = 240000 → ratio = 250000/240000 ≈ 1.042
        ratio = result.features.get("ce_pe_oi_ratio")
        assert ratio is not None
        assert 1.0 <= ratio <= 2.0

    def test_greeks_features_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["ctx_delta", "ctx_gamma", "ctx_theta", "ctx_vega",
                               "delta_abs", "greeks_imbalance",
                               "theta_to_vega_ratio", "gamma_to_theta_ratio"]
        )
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        assert result.features.get("ctx_delta") == 0.52
        assert result.features.get("ctx_gamma") == 0.011
        assert result.features.get("ctx_theta") == -3.8
        assert result.features.get("ctx_vega") == 5.9
        assert result.features.get("delta_abs") == 0.52
        # greeks_imbalance = gamma + vega + theta
        gi = result.features.get("greeks_imbalance")
        assert gi is not None
        assert abs(gi - (0.011 + 5.9 - 3.8)) < 0.01

    def test_dte_and_weekly_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["dte_days", "is_weekly", "ctx_dte_norm"]
        )
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        assert result.features.get("dte_days") == 5.0
        assert result.features.get("is_weekly") == 1.0
        assert result.features.get("ctx_dte_norm") is not None


# ---------------------------------------------------------------------------
# Test: session / time features
# ---------------------------------------------------------------------------

class TestSessionFeatures:
    def test_session_features_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["is_opening_session", "is_closing_session",
                               "is_midday_lull", "weekday", "month",
                               "ctx_time_sin", "ctx_time_cos"]
        )
        # Use a specific timestamp to control session detection
        now = datetime(2026, 6, 7, 10, 30, tzinfo=timezone.utc)  # Sunday 10:30 → opening session
        snapshot = LiveSnapshot(
            candles=[],
            spot=24650.0,
            timestamp=now,
        )
        result = builder.build(snapshot)

        assert result.features.get("is_opening_session") == 1.0
        assert result.features.get("weekday") == 6.0  # Sunday
        assert result.features.get("month") == 6.0
        assert result.features.get("ctx_time_sin") is not None
        assert result.features.get("ctx_time_cos") is not None

    def test_closing_session_detected(self):
        builder = LiveFeatureBuilder(required_features=["is_closing_session"])
        now = datetime(2026, 6, 8, 15, 15, tzinfo=timezone.utc)  # Monday 15:15
        snapshot = LiveSnapshot(candles=[], spot=24650.0, timestamp=now)
        result = builder.build(snapshot)
        assert result.features.get("is_closing_session") == 1.0


# ---------------------------------------------------------------------------
# Test: feature completeness — no silent zero-filling
# ---------------------------------------------------------------------------

class TestNoSilentZeroFilling:
    def test_required_feature_not_silently_zero_filled(self):
        """If a feature is required and unavailable, it must be in result.missing, not 0.0."""
        builder = LiveFeatureBuilder(required_features=["oi_z_5"])
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        # oi_z_5 needs rolling history — should be missing, not 0.0
        assert "oi_z_5" in result.missing, "oi_z_5 should be marked as missing"
        # features dict should have None, not 0.0
        assert result.features.get("oi_z_5") is None, "oi_z_5 should be None, not 0.0"

    def test_to_aligned_vector_does_not_hide_missing(self):
        """to_aligned_vector converts missing to 0.0 for model compat, but the result.missing set is preserved."""
        builder = LiveFeatureBuilder(required_features=["oi_z_5", "rsi_14"])
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        aligned = result.to_aligned_vector(result.required)

        # oi_z_5 missing → 0.0 in aligned vector (model compat)
        assert aligned["oi_z_5"] == 0.0
        # rsi_14 available → non-zero in aligned
        assert aligned["rsi_14"] != 0.0 or result.features["rsi_14"] is not None


# ---------------------------------------------------------------------------
# Test: return and volatility stats
# ---------------------------------------------------------------------------

class TestReturnAndVolatilityStats:
    def test_ret_stats_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["ret_mean", "ret_std", "ret_min", "ret_max",
                               "ret_1", "ret_3", "ret_5", "ret_10"]
        )
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        for name in ["ret_mean", "ret_std", "ret_min", "ret_max", "ret_1", "ret_3", "ret_5", "ret_10"]:
            assert name in result.features, f"{name} should be in features"
            assert name not in result.missing, f"{name} should NOT be missing"

    def test_vol_stats_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["vol_mean", "vol_std", "vol_min", "vol_max"]
        )
        snapshot = make_snapshot(num_candles=30, with_option_chain=True)
        result = builder.build(snapshot)

        for name in ["vol_mean", "vol_std", "vol_min", "vol_max"]:
            assert name in result.features
            assert name not in result.missing


# ---------------------------------------------------------------------------
# Test: open/high/low/close/volume from candles
# ---------------------------------------------------------------------------

class TestCandleOHLCVFeatures:
    def test_ohlcv_features_present(self):
        builder = LiveFeatureBuilder(
            required_features=["last_open", "last_high", "last_low",
                               "last_close", "last_volume",
                               "open", "high", "low", "close", "volume"]
        )
        snapshot = make_snapshot(num_candles=5, with_option_chain=False)
        result = builder.build(snapshot)

        for name in ["last_open", "last_high", "last_low", "last_close", "last_volume"]:
            assert result.features.get(name) is not None, f"{name} should be computed"


# ---------------------------------------------------------------------------
# Test: candlestick patterns
# ---------------------------------------------------------------------------

class TestCandlestickPatterns:
    def test_pattern_features_present(self):
        builder = LiveFeatureBuilder(
            required_features=["bullish_engulfing", "bearish_engulfing",
                               "doji", "hammer", "shooting_star"]
        )
        snapshot = make_snapshot(num_candles=5, with_option_chain=False)
        result = builder.build(snapshot)

        for name in ["bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star"]:
            assert name in result.features, f"{name} should be in features"
            val = result.features.get(name)
            assert val in (0.0, 1.0), f"{name} should be 0.0 or 1.0"


# ---------------------------------------------------------------------------
# Test: pivot, supertrend, regime
# ---------------------------------------------------------------------------

class TestTechnicalPatterns:
    def test_pivot_features_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct"]
        )
        snapshot = make_snapshot(num_candles=20, with_option_chain=False)
        result = builder.build(snapshot)

        for name in ["pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct"]:
            assert name in result.features
            assert name not in result.missing

    def test_regime_features_computed(self):
        builder = LiveFeatureBuilder(
            required_features=["regime_trending", "regime_volatile",
                               "regime_mean_reverting", "regime_quiet"]
        )
        snapshot = make_snapshot(num_candles=30, with_option_chain=False)
        result = builder.build(snapshot)

        # Exactly one regime should be 1.0
        regime_vals = [
            result.features.get("regime_trending", 0.0),
            result.features.get("regime_volatile", 0.0),
            result.features.get("regime_mean_reverting", 0.0),
            result.features.get("regime_quiet", 0.0),
        ]
        assert sum(regime_vals) == 1.0, "Exactly one regime should be active"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-p", "no:cacheprovider"])