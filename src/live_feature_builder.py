"""live_feature_builder.py
=========================
Computes all ML features from a live market snapshot in real time.

Unlike the batch training pipeline (ml_pipeline.py / build_market_feature_vector),
this module operates on a single live snapshot and explicitly tracks which features
are unavailable so the caller can block predictions when coverage < 95%.

Design principles
-----------------
- Every feature is either computed or explicitly marked unavailable (None).
- No silent zero-filling of required features.
- Coverage is returned explicitly so the decision pipeline can gate on it.
- All computations mirror build_market_feature_vector / build_supervised_dataset_v2.
- Option-chain features are only available when option data is present.

Usage
-----
    from live_feature_builder import LiveSnapshot, LiveFeatureBuilder, build_live_features

    snapshot = LiveSnapshot(
        candles=[...],          # list of Candle (OHLCV)
        spot=24650.0,           # current spot price
        atm_iv=0.18,            # ATM implied volatility
        option_chain={...},     # optional option chain dict
    )
    result = build_live_features(snapshot)
    # result.features  → dict[str, float]
    # result.missing   → set[str]
    # result.coverage  → float (0-100)

    if result.coverage < 95.0:
        logger.warning("Feature coverage %.1f%% < 95%% — blocking prediction", result.coverage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Internal imports — all guarded so the module can be imported in non-live contexts
try:
    from market_data import Candle
except Exception:  # pragma: no cover
    Candle = Any  # type: ignore

try:
    from indicators import adx as _ind_adx, atr as _ind_atr, choppiness_index as _ind_choppiness
    from indicators import ema as _ind_ema, pivot_points as _ind_pivot
    from indicators import rsi as _ind_rsi, roc as _ind_roc, supertrend as _ind_supertrend
except Exception:  # pragma: no cover
    _ind_adx = _ind_atr = _ind_choppiness = None
    _ind_ema = _ind_pivot = _ind_rsi = _ind_roc = _ind_supertrend = None

try:
    from candlestick_patterns import (
        is_bearish_engulfing, is_bullish_engulfing,
        is_doji, is_hammer, is_shooting_star,
    )
except Exception:  # pragma: no cover
    is_bearish_engulfing = is_bullish_engulfing = None
    is_doji = is_hammer = is_shooting_star = None

try:
    from ml_pipeline import MLFeatureContext
except Exception:  # pragma: no cover
    MLFeatureContext = Any  # type: ignore

try:
    from strategy_allocator import detect_regime
except Exception:  # pragma: no cover
    detect_regime = None

try:
    from mean_reversion import evaluate_mean_reversion
except Exception:  # pragma: no cover
    evaluate_mean_reversion = None

try:
    from stat_arb import evaluate_pair_spread
except Exception:  # pragma: no cover
    evaluate_pair_spread = None

try:
    from trend_following import evaluate_trend_following
except Exception:  # pragma: no cover
    evaluate_trend_following = None

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FeatureComputationError(RuntimeError):
    """Raised when a feature cannot be computed and should be marked unavailable."""
    pass


# ---------------------------------------------------------------------------
# Snapshot input type
# ---------------------------------------------------------------------------


@dataclass
class LiveSnapshot:
    """Flat snapshot of everything available at prediction time.

    Attributes
    ----------
    candles : list[Candle]
        Last N OHLCV candles (oldest → newest).  At least 2 needed for
        meaningful indicators; 20+ recommended for rolling stats.
    spot : float | None
        Current underlying spot price.
    atm_iv : float | None
        ATM implied volatility (optional).
    option_chain : dict | None
        Option-chain snapshot with the following keys:
        - strike (float): option strike price
        - expiry (str | datetime): expiry date
        - option_type (str): "CE" or "PE"
        - ltp (float): last traded price of the contract
        - bid (float): bid price
        - ask (float): ask price
        - iv (float): contract IV
        - delta (float): option delta
        - gamma (float): option gamma
        - theta (float): option theta
        - vega (float): option vega
        - oi (float): open interest
        - oi_CE (float): CE open interest (for ATM strike)
        - oi_PE (float): PE open interest (for ATM strike)
        - volume_CE (float): CE volume
        - volume_PE (float): PE volume
        - volume (float): this contract's volume
        - change_in_oi (float): change in OI for this contract
        - dte_days (float): days to expiry
        - is_weekly (bool): whether this is a weekly expiry
        - open (float): today's open price of contract
        - high (float): today's high
        - low (float): today's low
        - spot_at_time (float): spot at the time of the option snapshot
    timestamp : datetime | None
        Snapshot timestamp (defaults to now UTC).
    """

    candles: List[Candle] = field(default_factory=list)
    spot: Optional[float] = None
    atm_iv: Optional[float] = None
    option_chain: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class LiveFeatureResult:
    """Return type of build_live_features."""

    #: Feature name → computed value.  Available features are float; unavailable are None.
    features: Dict[str, Optional[float]]

    #: Names of features that could not be computed.
    missing: Set[str]

    #: (available / required) * 100, rounded to 2 decimal places.
    coverage_pct: float

    #: Required feature names (the union of model features + always-required features).
    required: Set[str]

    def to_aligned_vector(self, feature_names: Sequence[str]) -> Dict[str, float]:
        """Return a dict suitable for model.predict with None → 0.0 for missing."""
        out: Dict[str, float] = {}
        for name in feature_names:
            val = self.features.get(name)
            out[name] = float(val) if val is not None else 0.0
        return out


# ---------------------------------------------------------------------------
# Feature source taxonomy (mirrors live_decision_dry_run.py)
# ---------------------------------------------------------------------------

# Features that need only OHLCV candles (no option chain, no rolling history)
CANDLE_FEATURES: Set[str] = {
    "last_open", "last_high", "last_low", "last_close", "last_volume",
    "body_pct", "range_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
    "close_location_pct", "open", "high", "low", "close", "volume",
    "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_mean", "ret_std", "ret_min", "ret_max",
    "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct",
    "adx_14", "roc_14", "choppiness_14", "supertrend_dir", "supertrend_gap_pct",
    "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
    "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
    "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
    "vol_mean", "vol_std", "vol_min", "vol_max", "vol_of_vol_14",
    "volatility_regime_classifier",
    "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60",
    "atr_pct_regime_10", "realized_vol_30",
}

# Features that need live option chain data
OPTION_CHAIN_FEATURES: Set[str] = {
    "ltp", "oi", "oi_CE", "oi_PE", "volume_CE", "volume_PE",
    "oi_change_pct", "volume_change_pct", "ce_pe_oi_ratio", "ce_pe_volume_ratio",
    "final_iv", "atm_distance", "strike_distance_pct", "distance_from_spot",
    "option_ltp", "strike_price", "expiry", "option_type",
    "bid_ask_spread_pct", "option_bid_ask_spread_pct",
    "ctx_iv", "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
    "ctx_option_price", "delta_abs", "greeks_imbalance",
    "theta_to_vega_ratio", "gamma_to_theta_ratio",
}

# Features that need rolling intraday history (z-scores, opening range, rolling highs)
ROLLING_HISTORY_FEATURES: Set[str] = {
    "oi_z_5", "volume_z_5",
    "dist_from_opening_high_pct", "dist_from_opening_low_pct",
    "opening_range_width_pct", "opening_range_breakout_strength",
    "dist_to_rolling_high_20", "dist_to_rolling_low_20",
    "rolling_range_width_20", "rolling_range_position_20",
}

# Context features derivable from current time / spot without history
CONTEXT_FEATURES: Set[str] = {
    "ctx_spot", "ctx_time_sin", "ctx_time_cos",
    "ctx_iv", "ctx_iv_change_pct", "ctx_iv_percentile",
    "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
    "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
    "ctx_dte_norm",
    "weekday", "month", "weekly", "is_weekly",
    "is_expiry_day", "is_near_expiry",
    "is_opening_session", "is_closing_session", "is_midday_lull",
    "spot", "open_spot", "high_spot", "low_spot", "close_spot",
    "spot_close", "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap",
    "volume_spot", "weekday_spot", "option_to_spot_pct", "dte_days",
    "option_type_ce", "option_type_pe",
    "hl_change_pct", "oc_change_pct",
}

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

ALL_COMPUTABLE_FEATURES: Set[str] = (
    CANDLE_FEATURES | OPTION_CHAIN_FEATURES | ROLLING_HISTORY_FEATURES | CONTEXT_FEATURES | DIRECT_STRATEGY_FEATURES
)


# ---------------------------------------------------------------------------
# Safe helpers (mirror ml_pipeline.py internals)
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if out != out or out == float("inf") or out == float("-inf"):  # NaN or inf
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_denominator(value: float, floor: float = 1e-7) -> float:
    val = abs(_safe_float(value, floor))
    return val if val >= floor else floor


def _series(values: Sequence[float], lookback: int) -> List[float]:
    vals = [_safe_float(v) for v in values]
    if lookback <= 0:
        return vals
    if len(vals) <= lookback:
        return vals
    return vals[-lookback:]


def _returns(closes: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(closes)):
        prev = _safe_float(closes[i - 1])
        cur = _safe_float(closes[i])
        out.append((cur - prev) / prev if prev else 0.0)
    return out


def _safe_std(values: Sequence[float]) -> float:
    vals = [_safe_float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    try:
        return float(pstdev(vals))
    except Exception:
        return 0.0


def _rolling_stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [_safe_float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(vals) == 1:
        v = vals[0]
        return {"mean": v, "std": 0.0, "min": v, "max": v}
    return {"mean": mean(vals), "std": pstdev(vals), "min": min(vals), "max": max(vals)}


def _rolling_percentile(values: Sequence[float], current: float, window: int) -> float:
    import numpy as np
    hist = [_safe_float(v) for v in values[-max(window, 1):]]
    if len(hist) <= 1:
        return 0.5
    return float(np.mean(np.asarray(hist, dtype=float) <= _safe_float(current)))


def _one_hot_regime(regime: str) -> Dict[str, float]:
    r = str(regime or "").strip().lower()
    return {
        "regime_trending": 1.0 if r == "trending" else 0.0,
        "regime_volatile": 1.0 if r == "volatile" else 0.0,
        "regime_mean_reverting": 1.0 if r == "mean_reverting" else 0.0,
        "regime_quiet": 1.0 if r == "quiet" else 0.0,
    }


# ---------------------------------------------------------------------------
# Candle-only feature computation
# ---------------------------------------------------------------------------

def _candlestick_context(last: Candle, prev: Optional[Candle]) -> Dict[str, Optional[float]]:
    high = _safe_float(last.high)
    low = _safe_float(last.low)
    open_ = _safe_float(last.open)
    close = _safe_float(last.close)
    prev_close = _safe_float(prev.close if prev is not None else open_)
    rng = max(high - low, 1e-9)
    body = close - open_
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    gap = open_ - prev_close

    return {
        "last_open": open_,
        "last_high": high,
        "last_low": low,
        "last_close": close,
        "last_volume": _safe_float(last.volume),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": _safe_float(last.volume),
        "body_pct": body / close if close else 0.0,
        "range_pct": rng / close if close else 0.0,
        "gap_pct": gap / prev_close if prev_close else 0.0,
        "upper_wick_pct": upper_wick / rng if rng else 0.0,
        "lower_wick_pct": lower_wick / rng if rng else 0.0,
        "close_location_pct": (close - low) / rng if rng else 0.5,
        "bullish_engulfing": (
            1.0 if (prev is not None and is_bullish_engulfing is not None and is_bullish_engulfing([prev, last]))
            else 0.0
        ),
        "bearish_engulfing": (
            1.0 if (prev is not None and is_bearish_engulfing is not None and is_bearish_engulfing([prev, last]))
            else 0.0
        ),
        "doji": 1.0 if (is_doji is not None and is_doji([last])) else 0.0,
        "hammer": 1.0 if (is_hammer is not None and is_hammer([last])) else 0.0,
        "shooting_star": 1.0 if (is_shooting_star is not None and is_shooting_star([last])) else 0.0,
    }


def _time_of_day_features(dt: Any) -> Dict[str, Optional[float]]:
    try:
        minutes = int(dt.hour) * 60 + int(dt.minute)
        return {
            "is_opening_session": 1.0 if (int(dt.hour) == 9 and 15 <= int(dt.minute) < 45) else 0.0,
            "is_closing_session": 1.0 if (int(dt.hour) == 15 and 0 <= int(dt.minute) <= 30) else 0.0,
            "is_midday_lull": 1.0 if (11 * 60 + 30) <= minutes <= (13 * 60 + 30) else 0.0,
        }
    except Exception:
        return {
            "is_opening_session": 0.0,
            "is_closing_session": 0.0,
            "is_midday_lull": 0.0,
        }


def _compute_atr_14(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> Optional[float]:
    if _ind_atr is None:
        return None
    return _ind_atr(list(highs), list(lows), list(closes), period=14)


def _compute_rsi_14(closes: Sequence[float]) -> Optional[float]:
    if _ind_rsi is None:
        return None
    return _ind_rsi(list(closes), period=14)


def _compute_adx_14(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> Optional[float]:
    if _ind_adx is None:
        return None
    return _ind_adx(list(highs), list(lows), list(closes), period=14)


def _compute_ema(values: Sequence[float], period: int) -> Optional[float]:
    if _ind_ema is None:
        return None
    return _ind_ema(list(values), period=period)


def _compute_roc(values: Sequence[float], period: int) -> Optional[float]:
    if _ind_roc is None:
        return None
    return _ind_roc(list(values), period=period)


def _compute_choppiness(values_h, values_l, values_c, period: int = 14) -> Optional[float]:
    if _ind_choppiness is None:
        return None
    return _ind_choppiness(list(values_h), list(values_l), list(values_c), period=period)


def _compute_supertrend(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    period: int = 10, multiplier: float = 3.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Returns (supertrend_value, direction).  direction: 1=up, -1=down, 0=unknown."""
    if _ind_supertrend is None:
        return None, None
    st_val = _ind_supertrend(list(highs), list(lows), list(closes), period=period, multiplier=multiplier)
    if st_val is None:
        return None, None
    close = _safe_float(closes[-1])
    direction = 1.0 if close >= st_val else -1.0
    return st_val, direction


def _compute_vwap(
    candles: Sequence[Candle],
) -> Optional[float]:
    """Compute VWAP from candles.  Returns None if not computable."""
    typical_sum = 0.0
    vol_sum = 0.0
    for c in candles:
        typical = (_safe_float(c.high) + _safe_float(c.low) + _safe_float(c.close)) / 3.0
        vol = _safe_float(c.volume)
        typical_sum += typical * vol
        vol_sum += vol
    if vol_sum <= 0:
        return None
    return typical_sum / vol_sum


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

_COVERAGE_MINIMUM = 95.0


class LiveFeatureBuilder:
    """Computes all model features from a LiveSnapshot.

    Parameters
    ----------
    required_features : sequence[str] | None
        Explicit list of features the model requires.  If None, all known
        model features (TARGET_FEATURES from ml_pipeline + option chain features)
        are assumed required.
    lookback : int
        Number of candles to use for rolling statistics.  Default 20.
    """

    DEFAULT_REQUIRED: List[str] = [
        # --- Candle OHLCV ---
        "last_open", "last_high", "last_low", "last_close", "last_volume",
        "body_pct", "range_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct",
        "close_location_pct", "open", "high", "low", "close", "volume",
        # --- Candlestick patterns ---
        "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
        # --- Return stats ---
        "ret_mean", "ret_std", "ret_min", "ret_max",
        "ret_1", "ret_3", "ret_5", "ret_10",
        # --- Volume stats ---
        "vol_mean", "vol_std", "vol_min", "vol_max",
        # --- Technical indicators ---
        "ema_fast", "ema_slow", "ema_diff_pct",
        "rsi_14", "atr_14", "atr_pct",
        "adx_14", "roc_14", "choppiness_14",
        "supertrend_dir", "supertrend_gap_pct",
        "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct",
        "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
        # --- Regime ---
        "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
        # --- Greeks context ---
        "ctx_iv", "ctx_iv_change_pct", "ctx_iv_percentile",
        "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
        "ctx_spot", "ctx_option_price",
        "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
        "ctx_time_sin", "ctx_time_cos", "ctx_dte_norm",
        "price_to_spot_pct", "option_to_spot_pct", "delta_abs", "greeks_imbalance",
        "mean_reversion_zscore", "mean_reversion_entry_score",
        "mean_reversion_expected_reversion_pct", "mean_reversion_half_life_bars",
        "mean_reversion_buy_call", "mean_reversion_buy_put",
        "stat_arb_zscore", "stat_arb_confidence", "stat_arb_spread_pct",
        "stat_arb_hedge_ratio", "stat_arb_long_spread", "stat_arb_short_spread",
        # --- Volatility regime ---
        "vol_of_vol_14", "realized_vol_30", "atr_pct_regime_10",
        "atr_percentile_60", "realized_vol_percentile_60",
        "volatility_percentile_60", "volatility_regime_classifier",
        # --- Session / time ---
        "is_opening_session", "is_closing_session", "is_midday_lull",
        # --- Rolling structure ---
        "dist_from_opening_high_pct", "dist_from_opening_low_pct",
        "opening_range_width_pct", "opening_range_breakout_strength",
        "dist_to_rolling_high_20", "dist_to_rolling_low_20",
        "rolling_range_width_20", "rolling_range_position_20",
        # --- Option chain (spot) ---
        "spot", "spot_close", "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap",
        "weekday", "month", "dte_days",
        "option_type_ce", "option_type_pe", "weekly", "is_weekly",
        "is_expiry_day", "is_near_expiry",
        "hl_change_pct", "oc_change_pct",
        # --- Option chain (contract) ---
        "ltp", "oi", "oi_CE", "oi_PE",
        "volume_CE", "volume_PE",
        "oi_change_pct", "volume_change_pct",
        "ce_pe_oi_ratio", "ce_pe_volume_ratio",
        "final_iv", "atm_distance", "strike_distance_pct", "distance_from_spot",
        "bid_ask_spread_pct", "option_bid_ask_spread_pct",
        "theta_to_vega_ratio", "gamma_to_theta_ratio",
    ]

    def __init__(
        self,
        required_features: Optional[Sequence[str]] = None,
        lookback: int = 20,
    ) -> None:
        self.required_features = set(required_features) if required_features else set(self.DEFAULT_REQUIRED)
        self.lookback = lookback

    def build(self, snapshot: LiveSnapshot) -> LiveFeatureResult:
        """Compute all required features from the live snapshot.

        Returns
        -------
        LiveFeatureResult
            features  : dict[str, Optional[float]]
            missing   : set[str]  — features that could not be computed
            coverage_pct : float  — (available / required) * 100
            required  : set[str]  — the required feature set used
        """
        features: Dict[str, Optional[float]] = {}
        missing: Set[str] = set()

        # ---- Base timestamp and session features (always available) ----
        ts = snapshot.timestamp or datetime.now(timezone.utc)
        session_minutes = ts.hour * 60 + ts.minute
        features["ctx_time_sin"] = _safe_float(
            __import__("math").sin(2 * 3.141592653589793 * session_minutes / 1440.0)
        )
        features["ctx_time_cos"] = _safe_float(
            __import__("math").cos(2 * 3.141592653589793 * session_minutes / 1440.0)
        )
        features["weekday"] = float(ts.weekday())
        features["month"] = float(ts.month)
        features["is_opening_session"] = 1.0 if 9 * 60 + 15 <= session_minutes <= 10 * 60 + 30 else 0.0
        features["is_closing_session"] = 1.0 if session_minutes >= 15 * 60 else 0.0
        features["is_midday_lull"] = 1.0 if 11 * 60 + 30 <= session_minutes <= 13 * 60 + 30 else 0.0

        # ---- Candle features ----
        candle_features, candle_missing = self._build_candle_features(snapshot)
        features.update(candle_features)
        missing.update(candle_missing)

        # ---- Option chain features ----
        option_features, option_missing = self._build_option_chain_features(snapshot)
        features.update(option_features)
        missing.update(option_missing)

        # ---- Rolling history features ----
        rolling_features, rolling_missing = self._build_rolling_history_features(snapshot)
        features.update(rolling_features)
        missing.update(rolling_missing)

        # ---- Spot features ----
        spot_features, spot_missing = self._build_spot_features(snapshot)
        features.update(spot_features)
        missing.update(spot_missing)

        # Normalize the contract so every required field is present in the output.
        for name in self.required_features:
            if name not in features:
                features[name] = None
                missing.add(name)

        # ---- Determine coverage ----
        required = self.required_features
        # Treat session/time features as always available (they are from clock)
        always_available = {
            "ctx_time_sin", "ctx_time_cos", "weekday", "month",
            "is_opening_session", "is_closing_session", "is_midday_lull",
        }
        available_count = sum(
            1 for name in required
            if name in always_available
            or (name in features and features[name] is not None)
        )
        coverage_pct = round(
            available_count / max(len(required), 1) * 100.0, 2
        )

        return LiveFeatureResult(
            features=features,
            missing=missing,
            coverage_pct=coverage_pct,
            required=required,
        )

    # ------------------------------------------------------------------
    # Candle feature computation
    # ------------------------------------------------------------------

    def _build_candle_features(self, snapshot: LiveSnapshot) -> Tuple[Dict, Set]:
        """Compute all OHLCV-derived features."""
        features: Dict[str, Optional[float]] = {}
        missing: Set[str] = set()
        candles = list(snapshot.candles)

        if len(candles) < 2:
            for name in CANDLE_FEATURES:
                if name in self.required_features:
                    missing.add(name)
            return features, missing

        last = candles[-1]
        prev = candles[-2] if len(candles) > 1 else None

        closes = [_safe_float(c.close) for c in candles]
        highs = [_safe_float(c.high) for c in candles]
        lows = [_safe_float(c.low) for c in candles]
        volumes = [_safe_float(c.volume) for c in candles]

        close = closes[-1]
        recent = _series(closes, self.lookback)
        recent_vols = _series(volumes, self.lookback)
        recent_rets = _returns(recent)
        ret_stats = _rolling_stats(recent_rets)
        vol_stats = _rolling_stats(recent_vols)

        # Candlestick context
        ctx = _candlestick_context(last, prev)
        features.update({k: v for k, v in ctx.items() if k in self.required_features})

        # Return stats
        for k, v in ret_stats.items():
            name = f"ret_{k}"
            if name in self.required_features:
                features[name] = v

        # Volume stats
        for k, v in vol_stats.items():
            name = f"vol_{k}"
            if name in self.required_features:
                features[name] = v

        # ---- ret_1, ret_3, ret_5, ret_10 ----
        prev_close = _safe_float(prev.close if prev else close)
        if "ret_1" in self.required_features and len(closes) >= 2:
            features["ret_1"] = (closes[-1] - closes[-2]) / _safe_denominator(closes[-2])
        elif "ret_1" in self.required_features:
            missing.add("ret_1")

        if "ret_3" in self.required_features and len(closes) >= 4:
            features["ret_3"] = (closes[-1] - closes[-4]) / _safe_denominator(closes[-4])
        elif "ret_3" in self.required_features:
            missing.add("ret_3")

        if "ret_5" in self.required_features and len(closes) >= 6:
            features["ret_5"] = (closes[-1] - closes[-6]) / _safe_denominator(closes[-6])
        elif "ret_5" in self.required_features:
            missing.add("ret_5")

        if "ret_10" in self.required_features and len(closes) >= 11:
            features["ret_10"] = (closes[-1] - closes[-11]) / _safe_denominator(closes[-11])
        elif "ret_10" in self.required_features:
            missing.add("ret_10")

        # ---- EMA ----
        if "ema_fast" in self.required_features or "ema_diff_pct" in self.required_features:
            ema_fast = _compute_ema(closes, 9)
            features["ema_fast"] = ema_fast
            if ema_fast is None and "ema_fast" in self.required_features:
                missing.add("ema_fast")

        if "ema_slow" in self.required_features or "ema_diff_pct" in self.required_features:
            ema_slow = _compute_ema(closes, 21)
            features["ema_slow"] = ema_slow
            if ema_slow is None and "ema_slow" in self.required_features:
                missing.add("ema_slow")

        if "ema_diff_pct" in self.required_features:
            ef = _safe_float(features.get("ema_fast"))
            es = _safe_float(features.get("ema_slow"))
            if ef and es:
                features["ema_diff_pct"] = (ef - es) / _safe_denominator(close)
            else:
                missing.add("ema_diff_pct")

        # ---- RSI 14 ----
        if "rsi_14" in self.required_features:
            rsi_val = _compute_rsi_14(closes)
            features["rsi_14"] = rsi_val if rsi_val is not None else 50.0

        # ---- ATR 14 and atr_pct ----
        if "atr_14" in self.required_features or "atr_pct" in self.required_features:
            atr_val = _compute_atr_14(highs, lows, closes)
            features["atr_14"] = atr_val
            if atr_val is not None and close:
                features["atr_pct"] = atr_val / close
            if atr_val is None and "atr_14" in self.required_features:
                missing.add("atr_14")

        # ---- ADX 14 ----
        if "adx_14" in self.required_features or "ctx_adx" in self.required_features or "ctx_trend_strength" in self.required_features:
            adx_val = _compute_adx_14(highs, lows, closes)
            features["adx_14"] = adx_val if adx_val is not None else 25.0

        if "ctx_adx" in self.required_features:
            adx_val = features.get("adx_14")
            if adx_val is not None:
                features["ctx_adx"] = adx_val
            else:
                missing.add("ctx_adx")

        # ctx_trend_strength is ADX-based trend strength (alias of adx_14)
        if "ctx_trend_strength" in self.required_features:
            adx_val = features.get("adx_14")
            if adx_val is not None:
                features["ctx_trend_strength"] = adx_val
            elif "adx_14" in missing:
                missing.add("ctx_trend_strength")

        # ---- ROC 14 ----
        if "roc_14" in self.required_features:
            roc_val = _compute_roc(closes, 14)
            features["roc_14"] = roc_val if roc_val is not None else 0.0

        # ---- Choppiness Index ----
        if "choppiness_14" in self.required_features or "ctx_choppiness" in self.required_features:
            chop_val = _compute_choppiness(highs, lows, closes, 14)
            features["choppiness_14"] = chop_val if chop_val is not None else 50.0

        if "ctx_choppiness" in self.required_features:
            chop_val = features.get("choppiness_14")
            if chop_val is not None:
                features["ctx_choppiness"] = chop_val
            else:
                missing.add("ctx_choppiness")

        # ---- Supertrend ----
        if "supertrend_dir" in self.required_features or "supertrend_gap_pct" in self.required_features:
            st_val, st_dir = _compute_supertrend(highs, lows, closes)
            features["supertrend_dir"] = st_dir
            if st_val is not None and close:
                features["supertrend_gap_pct"] = abs(close - st_val) / close
            else:
                if "supertrend_gap_pct" in self.required_features:
                    missing.add("supertrend_gap_pct")
            if st_val is None and "supertrend_dir" in self.required_features:
                missing.add("supertrend_dir")

        # ---- Pivot points ----
        if any(k in self.required_features for k in (
            "pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct"
        )):
            if _ind_pivot is not None:
                try:
                    piv = _ind_pivot(highs[-1], lows[-1], closes[-1])
                    for name in ("pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct"):
                        if name in self.required_features:
                            raw = _safe_float(piv.get(name.replace("pivot_", "").replace("_dist_pct", "")))
                            features[name] = (close - raw) / _safe_denominator(close)
                except Exception:
                    for name in ("pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct"):
                        if name in self.required_features:
                            missing.add(name)
            else:
                for name in ("pivot_pp_dist_pct", "pivot_r1_dist_pct", "pivot_s1_dist_pct"):
                    if name in self.required_features:
                        missing.add(name)

        # ---- Candle ratios ----
        if "close_vs_open_pct" in self.required_features:
            features["close_vs_open_pct"] = (close - features.get("last_open", close)) / _safe_denominator(close)

        if "range_to_atr" in self.required_features:
            atr_pct = _safe_float(features.get("atr_pct"))
            range_pct = _safe_float(features.get("range_pct"))
            features["range_to_atr"] = range_pct / atr_pct if atr_pct else 0.0

        if "momentum_lookback_pct" in self.required_features and len(recent) > 1:
            features["momentum_lookback_pct"] = (close - recent[0]) / _safe_denominator(recent[0])

        if "volume_ratio" in self.required_features:
            vol_mean = _safe_float(features.get("vol_mean"))
            last_vol = _safe_float(features.get("last_volume", 0))
            features["volume_ratio"] = last_vol / vol_mean if vol_mean else 0.0

        if "ctx_volume_sma" in self.required_features:
            features["ctx_volume_sma"] = _safe_float(features.get("vol_mean"))

        # ---- Regime ----
        if any(k in self.required_features for k in (
            "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet"
        )):
            regime = ""
            if detect_regime is not None:
                try:
                    atr_vals = [_safe_float(_compute_atr_14(highs[:i+1], lows[:i+1], closes[:i+1])) or 0.0
                                for i in range(len(closes))]
                    adx_vals = [_safe_float(_compute_adx_14(highs[:i+1], lows[:i+1], closes[:i+1])) or 25.0
                                for i in range(len(closes))]
                    rsi_vals = [_safe_float(_compute_rsi_14(closes[:i+1])) or 50.0
                                for i in range(len(closes))]
                    regime = detect_regime(atr_vals, adx_vals, rsi_vals)
                except Exception:
                    regime = "quiet"
            features.update(_one_hot_regime(regime))

        if any(name in self.required_features for name in DIRECT_STRATEGY_FEATURES):
            self._build_direct_strategy_features(closes, features)

        # ---- VWAP ----
        if "spot_vwap" in self.required_features:
            vwap = _compute_vwap(candles)
            features["spot_vwap"] = vwap
            if vwap is None:
                missing.add("spot_vwap")

        # ---- Vol of vol ----
        if "vol_of_vol_14" in self.required_features:
            try:
                import numpy as np
                recent_atrs = [
                    _safe_float(_compute_atr_14(highs[:i+1], lows[:i+1], closes[:i+1])) / _safe_denominator(closes[i])
                    for i in range(max(0, len(closes) - 14), len(closes))
                ]
                features["vol_of_vol_14"] = float(np.std(recent_atrs)) if len(recent_atrs) > 1 else 0.0
            except Exception:
                features["vol_of_vol_14"] = 0.0

        # ---- Realized vol 30 ----
        if "realized_vol_30" in self.required_features:
            all_rets = _returns(closes)
            features["realized_vol_30"] = _safe_std(all_rets[max(0, len(all_rets) - 30):])

        # ---- ATR percentile / regime ----
        if "atr_pct_regime_10" in self.required_features:
            try:
                recent_atrs = [
                    _safe_float(_compute_atr_14(highs[:i+1], lows[:i+1], closes[:i+1])) / _safe_denominator(closes[i])
                    for i in range(max(0, len(closes) - 10), len(closes))
                ]
                atr_regime_mean = mean(recent_atrs) if recent_atrs else 0.0
                features["atr_pct_regime_10"] = _safe_float(features.get("atr_pct")) / atr_regime_mean if atr_regime_mean else 0.0
            except Exception:
                features["atr_pct_regime_10"] = 0.0

        # ---- Volatility regime classifier ----
        if any(k in self.required_features for k in (
            "atr_percentile_60", "realized_vol_percentile_60",
            "volatility_percentile_60", "volatility_regime_classifier"
        )):
            try:
                import numpy as np
                atr_pct_history = [
                    _safe_float(_compute_atr_14(highs[:i+1], lows[:i+1], closes[:i+1])) / _safe_denominator(closes[i])
                    for i in range(len(closes))
                ]
                realized_vol_history = [
                    _safe_std(_returns(closes[max(0, i - 30):i]))
                    for i in range(len(closes))
                ]
                cur_atr_pct = _safe_float(features.get("atr_pct"))
                cur_rv = _safe_float(features.get("realized_vol_30"))

                atr_pct_perc = _rolling_percentile(atr_pct_history, cur_atr_pct, 60)
                rv_perc = _rolling_percentile(realized_vol_history, cur_rv, 60)

                features["atr_percentile_60"] = atr_pct_perc
                features["realized_vol_percentile_60"] = rv_perc
                features["volatility_percentile_60"] = rv_perc
                features["volatility_regime_classifier"] = 1.0 if (atr_pct_perc >= 0.7 or rv_perc >= 0.7) else 0.0
            except Exception:
                for name in ("atr_percentile_60", "realized_vol_percentile_60",
                             "volatility_percentile_60", "volatility_regime_classifier"):
                    if name in self.required_features:
                        features[name] = 0.0

        # ---- Rolling structure features ----
        if any(k in self.required_features for k in (
            "dist_from_opening_high_pct", "dist_from_opening_low_pct",
            "opening_range_width_pct", "opening_range_breakout_strength",
        )):
            or_feats = self._opening_range_features(candles, len(candles) - 1, close)
            for k, v in or_feats.items():
                if k in self.required_features:
                    features[k] = v

        if any(k in self.required_features for k in (
            "dist_to_rolling_high_20", "dist_to_rolling_low_20",
            "rolling_range_width_20", "rolling_range_position_20",
        )):
            ms_feats = self._market_structure_features(candles, len(candles) - 1, close)
            for k, v in ms_feats.items():
                if k in self.required_features:
                    features[k] = v

        return features, missing

    def _build_direct_strategy_features(
        self,
        closes: Sequence[float],
        features: Dict[str, Optional[float]],
    ) -> None:
        if evaluate_trend_following is not None:
            try:
                trend_signal = evaluate_trend_following(
                    list(closes),
                    fast_period=8,
                    slow_period=21,
                    momentum_lookback=10,
                    breakout_lookback=20,
                    strength_threshold=0.003,
                )
                features["trend_following_strength"] = _safe_float(trend_signal.strength)
                features["trend_following_confidence"] = _safe_float(trend_signal.confidence)
                features["trend_following_ema_gap_pct"] = _safe_float(trend_signal.ema_gap_pct)
                features["trend_following_momentum_pct"] = _safe_float(trend_signal.momentum_pct)
                features["trend_following_breakout_score"] = _safe_float(trend_signal.breakout_score)
                features["trend_following_pullback_score"] = _safe_float(trend_signal.pullback_score)
                features["trend_following_buy_call"] = 1.0 if str(trend_signal.signal) == "buy_call" else 0.0
                features["trend_following_buy_put"] = 1.0 if str(trend_signal.signal) == "buy_put" else 0.0
            except Exception:
                pass
        for name in (
            "trend_following_strength", "trend_following_confidence",
            "trend_following_ema_gap_pct", "trend_following_momentum_pct",
            "trend_following_breakout_score", "trend_following_pullback_score",
            "trend_following_buy_call", "trend_following_buy_put",
        ):
            features.setdefault(name, 0.0)

        close = _safe_float(closes[-1]) if closes else 0.0
        if evaluate_mean_reversion is not None:
            try:
                mr_signal = evaluate_mean_reversion(
                    list(closes),
                    lookback=20,
                    entry_zscore=1.25,
                    exit_zscore=0.35,
                )
                features["mean_reversion_zscore"] = _safe_float(mr_signal.zscore)
                features["mean_reversion_entry_score"] = _safe_float(mr_signal.entry_score)
                features["mean_reversion_expected_reversion_pct"] = _safe_float(mr_signal.expected_reversion) / _safe_denominator(close)
                features["mean_reversion_half_life_bars"] = _safe_float(mr_signal.half_life_bars)
                features["mean_reversion_buy_call"] = 1.0 if str(mr_signal.signal) == "buy_call" else 0.0
                features["mean_reversion_buy_put"] = 1.0 if str(mr_signal.signal) == "buy_put" else 0.0
            except Exception:
                pass
        for name in (
            "mean_reversion_zscore", "mean_reversion_entry_score",
            "mean_reversion_expected_reversion_pct", "mean_reversion_half_life_bars",
            "mean_reversion_buy_call", "mean_reversion_buy_put",
        ):
            features.setdefault(name, 0.0)

        if evaluate_pair_spread is not None and len(closes) >= 5:
            try:
                fair_value_series = []
                ema_value = 0.0
                alpha = 2.0 / (21.0 + 1.0)
                for idx, raw_close in enumerate(closes):
                    price = _safe_float(raw_close)
                    ema_value = price if idx == 0 else ((alpha * price) + ((1.0 - alpha) * ema_value))
                    fair_value_series.append(ema_value)
                stat_signal = evaluate_pair_spread(
                    list(closes),
                    fair_value_series,
                    lookback=30,
                    entry_zscore=1.5,
                    exit_zscore=0.5,
                )
                features["stat_arb_zscore"] = _safe_float(stat_signal.zscore)
                features["stat_arb_confidence"] = _safe_float(stat_signal.confidence)
                features["stat_arb_spread_pct"] = _safe_float(stat_signal.spread) / _safe_denominator(close)
                features["stat_arb_hedge_ratio"] = _safe_float(stat_signal.hedge_ratio)
                features["stat_arb_long_spread"] = 1.0 if str(stat_signal.signal) == "long_spread" else 0.0
                features["stat_arb_short_spread"] = 1.0 if str(stat_signal.signal) == "short_spread" else 0.0
            except Exception:
                pass
        for name in (
            "stat_arb_zscore", "stat_arb_confidence",
            "stat_arb_spread_pct", "stat_arb_hedge_ratio",
            "stat_arb_long_spread", "stat_arb_short_spread",
        ):
            features.setdefault(name, 0.0)

    def _opening_range_features(
        self, candles: Sequence[Candle], idx: int, close: float
    ) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "dist_from_opening_high_pct": 0.0,
            "dist_from_opening_low_pct": 0.0,
            "opening_range_width_pct": 0.0,
            "opening_range_breakout_strength": 0.0,
        }
        try:
            if idx <= 0:
                return out
            curr_date = candles[idx].time.date()
            opening_candles: List[Candle] = []
            scan_idx = idx
            while scan_idx >= 0 and candles[scan_idx].time.date() == curr_date:
                dt_time = candles[scan_idx].time
                if dt_time.hour == 9 and 15 <= dt_time.minute <= 45:
                    opening_candles.append(candles[scan_idx])
                scan_idx -= 1
            if not opening_candles:
                return out
            op_high = max(_safe_float(c.high) for c in opening_candles)
            op_low = min(_safe_float(c.low) for c in opening_candles)
            width = max(op_high - op_low, 1e-9)
            out["dist_from_opening_high_pct"] = (close - op_high) / op_high if op_high else 0.0
            out["dist_from_opening_low_pct"] = (close - op_low) / op_low if op_low else 0.0
            out["opening_range_width_pct"] = width / close if close else 0.0
            if close > op_high:
                out["opening_range_breakout_strength"] = (close - op_high) / width
            elif close < op_low:
                out["opening_range_breakout_strength"] = (close - op_low) / width
            return out
        except Exception:
            return out

    def _market_structure_features(
        self, candles: Sequence[Candle], idx: int, close: float, window: int = 20
    ) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "dist_to_rolling_high_20": 0.0,
            "dist_to_rolling_low_20": 0.0,
            "rolling_range_width_20": 0.0,
            "rolling_range_position_20": 0.0,
        }
        try:
            if idx <= 0:
                return out
            recent = list(candles[max(0, idx - window): idx])
            if not recent:
                return out
            roll_high = max(_safe_float(c.high) for c in recent)
            roll_low = min(_safe_float(c.low) for c in recent)
            width = max(roll_high - roll_low, 1e-9)
            out["dist_to_rolling_high_20"] = (roll_high - close) / close if close else 0.0
            out["dist_to_rolling_low_20"] = (close - roll_low) / close if close else 0.0
            out["rolling_range_width_20"] = width / close if close else 0.0
            out["rolling_range_position_20"] = (close - roll_low) / width
            return out
        except Exception:
            return out

    # ------------------------------------------------------------------
    # Option chain feature computation
    # ------------------------------------------------------------------

    def _build_option_chain_features(self, snapshot: LiveSnapshot) -> Tuple[Dict, Set]:
        """Compute features derivable from option chain snapshot."""
        features: Dict[str, Optional[float]] = {}
        missing: Set[str] = set()
        oc = snapshot.option_chain
        spot = _safe_float(snapshot.spot)
        atm_iv = _safe_float(snapshot.atm_iv)

        # If no option chain data, mark all option features as missing
        if oc is None:
            for name in OPTION_CHAIN_FEATURES:
                if name in self.required_features:
                    missing.add(name)
            return features, missing

        ltp = _safe_float(oc.get("ltp"))
        bid = _safe_float(oc.get("bid"))
        ask = _safe_float(oc.get("ask"))
        strike = _safe_float(oc.get("strike"))
        iv = _safe_float(oc.get("iv", atm_iv))
        delta = _safe_float(oc.get("delta"))
        gamma = _safe_float(oc.get("gamma"))
        theta = _safe_float(oc.get("theta"))
        vega = _safe_float(oc.get("vega"))
        oi = _safe_float(oc.get("oi"))
        oi_CE = _safe_float(oc.get("oi_CE"))
        oi_PE = _safe_float(oc.get("oi_PE"))
        vol_CE = _safe_float(oc.get("volume_CE"))
        vol_PE = _safe_float(oc.get("volume_PE"))
        vol = _safe_float(oc.get("volume"))
        change_in_oi = _safe_float(oc.get("change_in_oi"))
        dte = _safe_float(oc.get("dte_days"))
        is_weekly = bool(oc.get("is_weekly", False))
        option_type = str(oc.get("option_type", "")).strip().upper()
        expiry_val = oc.get("expiry")
        open_oc = _safe_float(oc.get("open"))
        high_oc = _safe_float(oc.get("high"))
        low_oc = _safe_float(oc.get("low"))
        spot_at_time = _safe_float(oc.get("spot_at_time", spot))

        # ---- Basic option metadata ----
        features["option_type_ce"] = 1.0 if option_type == "CE" else 0.0
        features["option_type_pe"] = 1.0 if option_type == "PE" else 0.0
        features["weekly"] = 1.0 if is_weekly else 0.0
        features["is_weekly"] = 1.0 if is_weekly else 0.0
        features["dte_days"] = dte if dte >= 0 else 0.0
        features["is_expiry_day"] = 1.0 if dte <= 0.0 else 0.0
        features["is_near_expiry"] = 1.0 if dte <= 2.0 else 0.0
        features["strike_price"] = strike
        features["expiry"] = str(expiry_val) if expiry_val else None  # type: ignore

        # ---- Price-based features ----
        features["ltp"] = ltp
        features["option_ltp"] = ltp
        features["ctx_option_price"] = ltp

        if ltp > 0 and spot > 0:
            features["option_to_spot_pct"] = ltp / spot
            features["distance_from_spot"] = (strike - spot) / spot
            features["distance_from_spot"] = abs(features["distance_from_spot"])
        else:
            if "option_to_spot_pct" in self.required_features:
                missing.add("option_to_spot_pct")
            if "distance_from_spot" in self.required_features:
                missing.add("distance_from_spot")

        if spot > 0 and strike > 0:
            atm_distance = (strike - spot) / spot
            features["atm_distance"] = atm_distance
            features["strike_distance_pct"] = abs(atm_distance)
        else:
            if "atm_distance" in self.required_features:
                missing.add("atm_distance")
            if "strike_distance_pct" in self.required_features:
                missing.add("strike_distance_pct")

        # ---- Greeks context ----
        features["ctx_iv"] = iv
        features["ctx_delta"] = delta
        features["ctx_gamma"] = gamma
        features["ctx_vega"] = vega
        features["ctx_theta"] = theta
        features["final_iv"] = iv
        features["delta_abs"] = abs(delta) if delta else 0.0
        features["greeks_imbalance"] = gamma + vega + theta
        features["ctx_dte_norm"] = dte / 30.0 if dte >= 0 else 0.0
        features["ctx_iv_change_pct"] = 0.0
        features["ctx_iv_percentile"] = 0.0

        # Greeks ratios
        features["theta_to_vega_ratio"] = theta / _safe_denominator(vega)
        features["gamma_to_theta_ratio"] = gamma / _safe_denominator(theta)

        # ---- Bid/ask spread ----
        if bid > 0 and ask > 0 and ask >= bid:
            midpoint = (bid + ask) / 2.0
            features["bid_ask_spread_pct"] = (ask - bid) / midpoint if midpoint else 0.0
            features["option_bid_ask_spread_pct"] = features["bid_ask_spread_pct"]
        else:
            # Fallback to rough estimate
            features["bid_ask_spread_pct"] = 0.02 * (1.0 + iv) if ltp > 0 else 0.0
            features["option_bid_ask_spread_pct"] = features["bid_ask_spread_pct"]

        # ---- OI and volume features ----
        features["oi"] = oi
        features["oi_CE"] = oi_CE
        features["oi_PE"] = oi_PE
        features["volume_CE"] = vol_CE
        features["volume_PE"] = vol_PE
        features["volume"] = vol

        if oi > 0 and change_in_oi is not None:
            features["oi_change_pct"] = change_in_oi / oi if oi else 0.0
        else:
            features["oi_change_pct"] = 0.0

        if vol > 0:
            # volume_change_pct from option chain is harder to compute live
            # without the previous bar; use 0 as a placeholder
            features["volume_change_pct"] = 0.0
        else:
            features["volume_change_pct"] = 0.0

        # CE/PE ratios
        if oi_PE > 0:
            features["ce_pe_oi_ratio"] = oi_CE / oi_PE if oi_PE else 0.0
        else:
            features["ce_pe_oi_ratio"] = 0.0

        if vol_PE > 0:
            features["ce_pe_volume_ratio"] = vol_CE / vol_PE if vol_PE else 0.0
        else:
            features["ce_pe_volume_ratio"] = 0.0

        # ---- Option OHLCV features ----
        if high_oc > 0 and low_oc > 0:
            features["hl_change_pct"] = (high_oc - low_oc) / _safe_denominator(low_oc)
        else:
            features["hl_change_pct"] = 0.0

        if open_oc > 0 and ltp > 0:
            features["oc_change_pct"] = (ltp - open_oc) / _safe_denominator(open_oc)
        else:
            features["oc_change_pct"] = 0.0

        # For oi_z_5 and volume_z_5, we need rolling history which we don't have in live mode
        for name in ("oi_z_5", "volume_z_5"):
            if name in self.required_features:
                missing.add(name)

        return features, missing

    # ------------------------------------------------------------------
    # Rolling / intraday-history features
    # ------------------------------------------------------------------

    def _build_rolling_history_features(self, snapshot: LiveSnapshot) -> Tuple[Dict, Set]:
        """Compute rolling intraday-history features (z-scores, opening range, rolling highs)."""
        features: Dict[str, Optional[float]] = {}
        missing: Set[str] = set()
        candles = list(snapshot.candles)

        if len(candles) < 5:
            for name in ROLLING_HISTORY_FEATURES:
                if name in self.required_features:
                    missing.add(name)
            return features, missing

        closes = [_safe_float(c.close) for c in candles]
        highs = [_safe_float(c.high) for c in candles]
        lows = [_safe_float(c.low) for c in candles]
        ois = [_safe_float(getattr(c, "oi", None)) for c in candles]
        volumes = [_safe_float(c.volume) for c in candles]

        last_close = closes[-1]

        # ---- Z-score features (oi_z_5, volume_z_5) ----
        for z_name, series in (("oi_z_5", ois), ("volume_z_5", volumes)):
            if z_name not in self.required_features:
                features[z_name] = None
                continue
            window_vals = _series(series, 5)
            if len(window_vals) < 3:
                missing.add(z_name)
                features[z_name] = None
                continue
            try:
                mean = sum(window_vals) / len(window_vals)
                variance = sum((v - mean) ** 2 for v in window_vals) / len(window_vals)
                std = math.sqrt(variance) if variance > 0 else 0.0
                current = window_vals[-1]
                # If all values are identical (std=0) or current equals mean, z-score is 0
                # but we can't distinguish signal from no-data — treat as unavailable.
                if std == 0.0 or variance == 0.0:
                    missing.add(z_name)
                    features[z_name] = None
                else:
                    features[z_name] = (current - mean) / std
            except Exception:
                missing.add(z_name)
                features[z_name] = None

        # ---- Opening range features ----
        first_candle_high = highs[0]
        first_candle_low = lows[0]
        opening_range = first_candle_high - first_candle_low

        features["dist_from_opening_high_pct"] = (
            (first_candle_high - last_close) / _safe_denominator(first_candle_high) * 100.0
            if first_candle_high > 0 else 0.0
        )
        features["dist_from_opening_low_pct"] = (
            (last_close - first_candle_low) / _safe_denominator(first_candle_low) * 100.0
            if first_candle_low > 0 else 0.0
        )
        features["opening_range_width_pct"] = (
            opening_range / _safe_denominator(first_candle_low) * 100.0
            if opening_range > 0 else 0.0
        )

        # Opening range breakout strength: how far price has moved beyond opening range
        if opening_range > 0:
            if last_close > first_candle_high:
                breakout = (last_close - first_candle_high) / opening_range
            elif last_close < first_candle_low:
                breakout = (first_candle_low - last_close) / opening_range
            else:
                breakout = 0.0
            features["opening_range_breakout_strength"] = float(breakout)
        else:
            features["opening_range_breakout_strength"] = 0.0

        # ---- Rolling 20-bar features ----
        recent_closes = _series(closes, self.lookback)
        recent_highs = _series(highs, self.lookback)
        recent_lows = _series(lows, self.lookback)

        rolling_high = max(recent_highs) if recent_highs else 0.0
        rolling_low = min(recent_lows) if recent_lows else 0.0
        rolling_range = rolling_high - rolling_low

        features["dist_to_rolling_high_20"] = (
            (rolling_high - last_close) / _safe_denominator(rolling_high) * 100.0
            if rolling_high > 0 else 0.0
        )
        features["dist_to_rolling_low_20"] = (
            (last_close - rolling_low) / _safe_denominator(rolling_low) * 100.0
            if rolling_low > 0 else 0.0
        )
        features["rolling_range_width_20"] = (
            rolling_range / _safe_denominator(rolling_low) * 100.0
            if rolling_range > 0 else 0.0
        )
        # Where is current close within the rolling range?
        if rolling_range > 0:
            features["rolling_range_position_20"] = (
                (last_close - rolling_low) / rolling_range * 100.0
            )
        else:
            features["rolling_range_position_20"] = 50.0

        return features, missing

    # ------------------------------------------------------------------
    # Spot context features
    # ------------------------------------------------------------------

    def _build_spot_features(self, snapshot: LiveSnapshot) -> Tuple[Dict, Set]:
        """Compute spot-level features from candles and spot price."""
        features: Dict[str, Optional[float]] = {}
        missing: Set[str] = set()
        spot = _safe_float(snapshot.spot)
        candles = list(snapshot.candles)

        features["spot"] = spot
        features["ctx_spot"] = spot

        if not candles or len(candles) < 2:
            return features, missing

        closes = [_safe_float(c.close) for c in candles]
        highs = [_safe_float(c.high) for c in candles]
        lows = [_safe_float(c.low) for c in candles]
        volumes = [_safe_float(c.volume) for c in candles]
        close = closes[-1]

        features["spot_close"] = close

        if spot > 0:
            features["spot_range_pct"] = (max(highs) - min(lows)) / spot
            features["price_to_spot_pct"] = (close - spot) / spot
        else:
            if "spot_range_pct" in self.required_features:
                missing.add("spot_range_pct")
            if "price_to_spot_pct" in self.required_features:
                missing.add("price_to_spot_pct")

        # Spot ATR
        atr_val = _compute_atr_14(highs, lows, closes)
        features["spot_atr"] = atr_val

        # Spot RSI
        rsi_val = _compute_rsi_14(closes)
        features["spot_rsi"] = rsi_val

        # VWAP
        vwap = _compute_vwap(candles)
        features["spot_vwap"] = vwap

        return features, missing


def build_live_features(
    snapshot: LiveSnapshot,
    required_features: Optional[Sequence[str]] = None,
    lookback: int = 20,
) -> LiveFeatureResult:
    """Convenience wrapper around LiveFeatureBuilder.build.

    Parameters
    ----------
    snapshot : LiveSnapshot
        Live market snapshot.
    required_features : sequence[str] | None
        Feature names required by the model.  None = use default feature set.
    lookback : int
        Number of candles for rolling stats.  Default 20.

    Returns
    -------
    LiveFeatureResult
        Result.features  : dict[str, Optional[float]]
        Result.missing   : set[str]
        Result.coverage_pct : float
        Result.required  : set[str]
    """
    builder = LiveFeatureBuilder(required_features=required_features, lookback=lookback)
    return builder.build(snapshot)


def build_offline_fixture(
    required_features: Optional[Sequence[str]] = None,
) -> LiveSnapshot:
    """Build a realistic synthetic fixture for offline testing / dry-runs.

    Populates candle and option chain data with stable synthetic values.
    This is NOT zero-filling — option-chain and rolling-history features
    will be marked as unavailable in the result.

    Parameters
    ----------
    required_features : sequence[str] | None
        Required feature names.  Used to decide which features to populate.

    Returns
    -------
    LiveSnapshot
        A fixture that can be passed to LiveFeatureBuilder.build.
    """
    import math
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    # Synthetic flat candle (stable values for reproducible tests)
    SYNTHETIC_CLOSE = 24500.0
    SYNTHETIC_RANGE = SYNTHETIC_CLOSE * 0.002  # 0.2% range
    s_open = SYNTHETIC_CLOSE - SYNTHETIC_RANGE * 0.3
    s_close = SYNTHETIC_CLOSE
    s_high = SYNTHETIC_CLOSE + SYNTHETIC_RANGE * 0.5
    s_low = SYNTHETIC_CLOSE - SYNTHETIC_RANGE * 0.6
    s_vol = 50000.0

    # Generate 30 synthetic candles (stable for tests)
    candles: List[Candle] = []
    base_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    for i in range(30):
        t = base_time.replace(minute=15 + i)
        # Slight variations so indicators have something to work with
        offset = (i - 15) * 5.0  # small drift
        c_close = SYNTHETIC_CLOSE + offset
        c_high = c_close + abs(SYNTHETIC_RANGE) * 0.3
        c_low = c_close - abs(SYNTHETIC_RANGE) * 0.3
        c_open = c_close - offset * 0.1
        candles.append(Candle(
            time=t,
            open=c_open,
            high=c_high,
            low=c_low,
            close=c_close,
            volume=s_vol,
        ))

    oc_required = required_features is None or bool(
        OPTION_CHAIN_FEATURES & set(required_features)
    )

    snapshot = LiveSnapshot(
        candles=candles,
        spot=SYNTHETIC_CLOSE,
        atm_iv=0.18,
        option_chain={
            "strike": 24500.0,
            "expiry": "2026-06-12",
            "option_type": "CE",
            "ltp": 150.0,
            "bid": 148.0,
            "ask": 152.0,
            "iv": 0.18,
            "delta": 0.48,
            "gamma": 0.012,
            "theta": -4.2,
            "vega": 6.1,
            "oi": 220000.0,
            "oi_CE": 220000.0,
            "oi_PE": 210000.0,
            "volume_CE": 1500.0,
            "volume_PE": 1400.0,
            "volume": 1500.0,
            "change_in_oi": 5000.0,
            "dte_days": 5.0,
            "is_weekly": True,
            "open": 148.0,
            "high": 155.0,
            "low": 145.0,
            "spot_at_time": SYNTHETIC_CLOSE,
        } if oc_required else None,
        timestamp=now,
    )
    return snapshot
