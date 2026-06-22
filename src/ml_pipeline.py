"""Feature engineering and ML evaluation helpers for the signal model.

The helpers here are intentionally lightweight and dependency-tolerant so the
project can run even when some scientific packages are missing. The feature
set mixes candle context, indicators, regime, IV, and Greeks context.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from candlestick_patterns import is_bearish_engulfing, is_bullish_engulfing, is_doji, is_hammer, is_shooting_star
from indicators import adx, atr, choppiness_index, ema, pivot_points, roc, rsi, supertrend
from market_data import Candle
from mean_reversion import evaluate_mean_reversion
from stat_arb import evaluate_pair_spread
from trend_following import evaluate_trend_following
from strategy_allocator import detect_regime

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
except Exception:  # pragma: no cover - optional dependency
    accuracy_score = precision_score = recall_score = f1_score = roc_auc_score = None
    TimeSeriesSplit = None


class FallbackClassifier:
    def __init__(self, **kwargs):
        self.class_prob = 0.5
        self.coef = []

    def fit(self, X, y):
        n_samples = len(X)
        if n_samples == 0:
            return self
        n_features = len(X[0])
        self.class_prob = sum(y) / n_samples
        x0 = [0.0] * n_features
        x1 = [0.0] * n_features
        c0 = 0
        c1 = 0
        for xi, yi in zip(X, y):
            if yi == 1:
                c1 += 1
                for j in range(n_features):
                    x1[j] += xi[j]
            else:
                c0 += 1
                for j in range(n_features):
                    x0[j] += xi[j]
        self.coef = [0.0] * n_features
        for j in range(n_features):
            m1 = x1[j] / c1 if c1 > 0 else 0.0
            m0 = x0[j] / c0 if c0 > 0 else 0.0
            self.coef[j] = m1 - m0
        return self

    def predict_proba(self, X):
        import math
        out = []
        for xi in X:
            score = sum(val * coef for val, coef in zip(xi, self.coef))
            try:
                prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, score))))
            except Exception:
                prob = 0.5
            prob = max(0.01, min(0.99, prob))
            out.append([1.0 - prob, prob])
        return out

    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if p[1] >= 0.5 else 0 for p in probs]


@dataclass
class MLFeatureContext:
    regime: str = ""
    iv: Optional[float] = None
    iv_change_pct: Optional[float] = None
    iv_percentile: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    spot: Optional[float] = None
    option_price: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    adx: Optional[float] = None
    trend_strength: Optional[float] = None
    choppiness: Optional[float] = None
    volume_sma: Optional[float] = None
    time_sin: Optional[float] = None
    time_cos: Optional[float] = None
    dte_norm: Optional[float] = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_denominator(value: float, floor: float = 1e-7) -> float:
    val = abs(_safe_float(value, floor))
    return val if val >= float(floor) else float(floor)


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


def _one_hot_regime(regime: str) -> Dict[str, float]:
    r = str(regime or "").strip().lower()
    return {
        "regime_trending": 1.0 if r == "trending" else 0.0,
        "regime_volatile": 1.0 if r == "volatile" else 0.0,
        "regime_mean_reverting": 1.0 if r == "mean_reverting" else 0.0,
        "regime_quiet": 1.0 if r == "quiet" else 0.0,
    }


def _candlestick_context(last: Candle, prev: Optional[Candle]) -> Dict[str, float]:
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
        "body_pct": body / close if close else 0.0,
        "range_pct": rng / close if close else 0.0,
        "gap_pct": gap / prev_close if prev_close else 0.0,
        "upper_wick_pct": upper_wick / rng if rng else 0.0,
        "lower_wick_pct": lower_wick / rng if rng else 0.0,
        "close_location_pct": (close - low) / rng if rng else 0.5,
        "bullish_engulfing": 1.0 if (prev is not None and is_bullish_engulfing([prev, last])) else 0.0,
        "bearish_engulfing": 1.0 if (prev is not None and is_bearish_engulfing([prev, last])) else 0.0,
        "doji": 1.0 if is_doji([last]) else 0.0,
        "hammer": 1.0 if is_hammer([last]) else 0.0,
        "shooting_star": 1.0 if is_shooting_star([last]) else 0.0,
    }


def _rolling_stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [_safe_float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(vals) == 1:
        v = vals[0]
        return {"mean": v, "std": 0.0, "min": v, "max": v}
    return {"mean": mean(vals), "std": pstdev(vals), "min": min(vals), "max": max(vals)}


def _safe_std(values: Sequence[float]) -> float:
    vals = [_safe_float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    try:
        return float(np.std(vals))
    except Exception:
        return 0.0


def _rolling_percentile(values: Sequence[float], current: float, window: int) -> float:
    hist = [_safe_float(v) for v in values[-max(window, 1) :]]
    if len(hist) <= 1:
        return 0.5
    current_val = _safe_float(current)
    return float(np.mean(np.asarray(hist, dtype=float) <= current_val))


def _volatility_regime_features(
    atr_pct_history: Sequence[float],
    realized_vol_history: Sequence[float],
    *,
    atr_pct: float,
    realized_vol: float,
    window: int = 60,
    threshold: float = 0.7,
) -> Dict[str, float]:
    atr_percentile = _rolling_percentile(atr_pct_history[:-1], atr_pct, window)
    realized_vol_percentile = _rolling_percentile(realized_vol_history[:-1], realized_vol, window)
    volatility_percentile = realized_vol_percentile
    return {
        "atr_percentile_60": atr_percentile,
        "realized_vol_percentile_60": realized_vol_percentile,
        "volatility_percentile_60": volatility_percentile,
        "volatility_regime_classifier": 1.0 if (atr_percentile >= threshold or realized_vol_percentile >= threshold) else 0.0,
    }


def _time_of_day_features(dt: Any) -> Dict[str, float]:
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


def _opening_range_features(candles: Sequence[Candle], idx: int, close: float) -> Dict[str, float]:
    out = {
        "dist_from_opening_high_pct": 0.0,
        "dist_from_opening_low_pct": 0.0,
        "opening_range_width_pct": 0.0,
        "opening_range_breakout_strength": 0.0,
    }
    try:
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


def _market_structure_features(candles: Sequence[Candle], idx: int, close: float, window: int = 20) -> Dict[str, float]:
    out = {
        "dist_to_rolling_high_20": 0.0,
        "dist_to_rolling_low_20": 0.0,
        "rolling_range_width_20": 0.0,
        "rolling_range_position_20": 0.0,
    }
    try:
        if idx <= 0:
            return out
        recent_candles = list(candles[max(0, idx - window) : idx])
        if not recent_candles:
            return out
        roll_high = max(_safe_float(c.high) for c in recent_candles)
        roll_low = min(_safe_float(c.low) for c in recent_candles)
        width = max(roll_high - roll_low, 1e-9)
        out["dist_to_rolling_high_20"] = (roll_high - close) / close if close else 0.0
        out["dist_to_rolling_low_20"] = (close - roll_low) / close if close else 0.0
        out["rolling_range_width_20"] = width / close if close else 0.0
        out["rolling_range_position_20"] = (close - roll_low) / width
        return out
    except Exception:
        return out


def build_market_feature_vector(
    candles: Sequence[Candle],
    *,
    context: Optional[MLFeatureContext | Dict[str, Any]] = None,
    lookback: int = 20,
) -> Tuple[List[float], List[str]]:
    """Build a fixed-length feature vector from candles and optional context."""

    c = list(candles or [])
    if not c:
        names = [
            "last_open", "last_high", "last_low", "last_close", "last_volume", "body_pct", "range_pct",
            "gap_pct", "upper_wick_pct", "lower_wick_pct", "close_location_pct",
        ]
        return [0.0] * len(names), names

    last = c[-1]
    prev = c[-2] if len(c) > 1 else None
    closes = [_safe_float(x.close) for x in c]
    highs = [_safe_float(x.high) for x in c]
    lows = [_safe_float(x.low) for x in c]
    volumes = [_safe_float(x.volume) for x in c]

    recent = _series(closes, lookback)
    recent_vols = _series(volumes, lookback)
    recent_rets = _returns(recent)
    ret_stats = _rolling_stats(recent_rets)
    vol_stats = _rolling_stats(recent_vols)

    feat: Dict[str, float] = {}
    feat.update(_candlestick_context(last, prev))
    feat.update({f"ret_{k}": v for k, v in ret_stats.items()})
    feat.update({f"vol_{k}": v for k, v in vol_stats.items()})

    close = feat["last_close"]
    prev_close = _safe_float(prev.close if prev is not None else close)
    feat["ret_1"] = (close - prev_close) / prev_close if prev_close else 0.0
    feat["ret_3"] = (close - _safe_float(c[-4].close if len(c) > 3 else prev_close)) / _safe_float(c[-4].close if len(c) > 3 else prev_close) if len(c) > 3 else feat["ret_1"]
    feat["ret_5"] = (close - _safe_float(c[-6].close if len(c) > 5 else prev_close)) / _safe_float(c[-6].close if len(c) > 5 else prev_close) if len(c) > 5 else feat["ret_1"]
    feat["ret_10"] = (close - _safe_float(c[-11].close if len(c) > 10 else prev_close)) / _safe_float(c[-11].close if len(c) > 10 else prev_close) if len(c) > 10 else feat["ret_1"]

    try:
        ema_fast = ema(closes, 9)
        ema_slow = ema(closes, 21)
        feat["ema_fast"] = _safe_float(ema_fast)
        feat["ema_slow"] = _safe_float(ema_slow)
        feat["ema_diff_pct"] = (feat["ema_fast"] - feat["ema_slow"]) / close if close else 0.0
    except Exception:
        feat["ema_fast"] = feat["ema_slow"] = feat["ema_diff_pct"] = 0.0

    try:
        feat["rsi_14"] = _safe_float(rsi(closes, period=14))
    except Exception:
        feat["rsi_14"] = 50.0
    try:
        feat["atr_14"] = _safe_float(atr(highs, lows, closes, period=14))
        feat["atr_pct"] = feat["atr_14"] / close if close else 0.0
    except Exception:
        feat["atr_14"] = feat["atr_pct"] = 0.0
    try:
        feat["adx_14"] = _safe_float(adx(highs, lows, closes, period=14))
    except Exception:
        feat["adx_14"] = 0.0
    try:
        feat["roc_14"] = _safe_float(roc(closes, period=14))
    except Exception:
        feat["roc_14"] = 0.0
    try:
        feat["choppiness_14"] = _safe_float(choppiness_index(highs, lows, closes, period=14))
    except Exception:
        feat["choppiness_14"] = 50.0
    try:
        st = supertrend(highs, lows, closes, period=10, multiplier=3.0)
        feat["supertrend_dir"] = 1.0 if st is not None and close >= _safe_float(st) else (-1.0 if st is not None else 0.0)
        feat["supertrend_gap_pct"] = abs(close - _safe_float(st)) / close if st is not None and close else 0.0
    except Exception:
        feat["supertrend_dir"] = 0.0
        feat["supertrend_gap_pct"] = 0.0

    try:
        piv = pivot_points(highs[-1], lows[-1], closes[-1])
        feat["pivot_pp_dist_pct"] = (close - _safe_float(piv.get("pp"))) / close if close else 0.0
        feat["pivot_r1_dist_pct"] = (close - _safe_float(piv.get("r1"))) / close if close else 0.0
        feat["pivot_s1_dist_pct"] = (close - _safe_float(piv.get("s1"))) / close if close else 0.0
    except Exception:
        feat["pivot_pp_dist_pct"] = feat["pivot_r1_dist_pct"] = feat["pivot_s1_dist_pct"] = 0.0

    feat["close_vs_open_pct"] = (close - feat["last_open"]) / close if close else 0.0
    feat["range_to_atr"] = feat["range_pct"] / feat["atr_pct"] if feat["atr_pct"] else 0.0
    feat["momentum_lookback_pct"] = (close - recent[0]) / recent[0] if len(recent) > 1 and recent[0] else 0.0
    feat["volume_ratio"] = feat["last_volume"] / feat["vol_mean"] if feat["vol_mean"] else 0.0

    regime = ""
    if context is not None:
        if isinstance(context, dict):
            regime = str(context.get("regime") or "")
        else:
            regime = str(getattr(context, "regime", "") or "")
    if not regime:
        try:
            regime = detect_regime([feat["atr_14"]] * 30, [feat["adx_14"]] * 30, [feat["rsi_14"]] * 30)
        except Exception:
            regime = "quiet"
    feat.update(_one_hot_regime(regime))

    def get_ctx(name: str, default: float = 0.0) -> float:
        if context is None:
            return float(default)
        if isinstance(context, dict):
            return _safe_float(context.get(name), default)
        return _safe_float(getattr(context, name, default), default)

    feat["ctx_iv"] = get_ctx("iv", 0.0)
    feat["ctx_iv_change_pct"] = get_ctx("iv_change_pct", 0.0)
    feat["ctx_iv_percentile"] = get_ctx("iv_percentile", 0.0)
    feat["ctx_delta"] = get_ctx("delta", 0.0)
    feat["ctx_gamma"] = get_ctx("gamma", 0.0)
    feat["ctx_vega"] = get_ctx("vega", 0.0)
    feat["ctx_theta"] = get_ctx("theta", 0.0)
    feat["ctx_spot"] = get_ctx("spot", close)
    feat["ctx_option_price"] = get_ctx("option_price", 0.0)
    feat["ctx_adx"] = get_ctx("adx", feat["adx_14"])
    feat["ctx_trend_strength"] = get_ctx("trend_strength", feat["ema_diff_pct"])
    feat["ctx_choppiness"] = get_ctx("choppiness", feat["choppiness_14"])
    feat["ctx_volume_sma"] = get_ctx("volume_sma", feat["vol_mean"])
    feat["ctx_time_sin"] = get_ctx("time_sin", 0.0)
    feat["ctx_time_cos"] = get_ctx("time_cos", 0.0)
    feat["ctx_dte_norm"] = get_ctx("dte_norm", 0.0)
    feat["ctx_bid_price"] = get_ctx("bid_price", 0.0)
    feat["ctx_ask_price"] = get_ctx("ask_price", 0.0)

    spot_denom = _safe_denominator(feat["ctx_spot"])
    feat["price_to_spot_pct"] = (close - feat["ctx_spot"]) / spot_denom
    feat["option_to_spot_pct"] = feat["ctx_option_price"] / spot_denom
    feat["delta_abs"] = abs(feat["ctx_delta"])
    feat["greeks_imbalance"] = feat["ctx_gamma"] + feat["ctx_vega"] + feat["ctx_theta"]

    try:
        trend_signal = evaluate_trend_following(
            closes,
            fast_period=8,
            slow_period=21,
            momentum_lookback=10,
            breakout_lookback=20,
            strength_threshold=0.003,
        )
        feat["trend_following_strength"] = _safe_float(trend_signal.strength)
        feat["trend_following_confidence"] = _safe_float(trend_signal.confidence)
        feat["trend_following_ema_gap_pct"] = _safe_float(trend_signal.ema_gap_pct)
        feat["trend_following_momentum_pct"] = _safe_float(trend_signal.momentum_pct)
        feat["trend_following_breakout_score"] = _safe_float(trend_signal.breakout_score)
        feat["trend_following_pullback_score"] = _safe_float(trend_signal.pullback_score)
        feat["trend_following_buy_call"] = 1.0 if str(trend_signal.signal) == "buy_call" else 0.0
        feat["trend_following_buy_put"] = 1.0 if str(trend_signal.signal) == "buy_put" else 0.0
    except Exception:
        feat["trend_following_strength"] = 0.0
        feat["trend_following_confidence"] = 0.0
        feat["trend_following_ema_gap_pct"] = 0.0
        feat["trend_following_momentum_pct"] = 0.0
        feat["trend_following_breakout_score"] = 0.0
        feat["trend_following_pullback_score"] = 0.0
        feat["trend_following_buy_call"] = 0.0
        feat["trend_following_buy_put"] = 0.0

    try:
        mr_signal = evaluate_mean_reversion(
            closes,
            lookback=20,
            entry_zscore=1.25,
            exit_zscore=0.35,
        )
        feat["mean_reversion_zscore"] = _safe_float(mr_signal.zscore)
        feat["mean_reversion_entry_score"] = _safe_float(mr_signal.entry_score)
        feat["mean_reversion_expected_reversion_pct"] = _safe_float(mr_signal.expected_reversion) / _safe_denominator(close)
        feat["mean_reversion_half_life_bars"] = _safe_float(mr_signal.half_life_bars)
        feat["mean_reversion_buy_call"] = 1.0 if str(mr_signal.signal) == "buy_call" else 0.0
        feat["mean_reversion_buy_put"] = 1.0 if str(mr_signal.signal) == "buy_put" else 0.0
    except Exception:
        feat["mean_reversion_zscore"] = 0.0
        feat["mean_reversion_entry_score"] = 0.0
        feat["mean_reversion_expected_reversion_pct"] = 0.0
        feat["mean_reversion_half_life_bars"] = 0.0
        feat["mean_reversion_buy_call"] = 0.0
        feat["mean_reversion_buy_put"] = 0.0

    try:
        fair_value_series = [float(ema(closes[: idx + 1], period=21) or closes[idx]) for idx in range(len(closes))]
        stat_arb_signal = evaluate_pair_spread(
            closes,
            fair_value_series,
            lookback=30,
            entry_zscore=1.5,
            exit_zscore=0.5,
        )
        feat["stat_arb_zscore"] = _safe_float(stat_arb_signal.zscore)
        feat["stat_arb_confidence"] = _safe_float(stat_arb_signal.confidence)
        feat["stat_arb_spread_pct"] = _safe_float(stat_arb_signal.spread) / _safe_denominator(close)
        feat["stat_arb_hedge_ratio"] = _safe_float(stat_arb_signal.hedge_ratio)
        feat["stat_arb_long_spread"] = 1.0 if str(stat_arb_signal.signal) == "long_spread" else 0.0
        feat["stat_arb_short_spread"] = 1.0 if str(stat_arb_signal.signal) == "short_spread" else 0.0
    except Exception:
        feat["stat_arb_zscore"] = 0.0
        feat["stat_arb_confidence"] = 0.0
        feat["stat_arb_spread_pct"] = 0.0
        feat["stat_arb_hedge_ratio"] = 0.0
        feat["stat_arb_long_spread"] = 0.0
        feat["stat_arb_short_spread"] = 0.0

    try:
        recent_atr_pcts = []
        for offset in range(min(14, len(c))):
            sub_highs = highs[: len(c) - offset]
            sub_lows = lows[: len(c) - offset]
            sub_closes = closes[: len(c) - offset]
            atr_val = _safe_float(atr(sub_highs, sub_lows, sub_closes, period=14))
            close_val = _safe_float(sub_closes[-1]) if sub_closes else 0.0
            recent_atr_pcts.append(atr_val / close_val if close_val else 0.0)
        feat["vol_of_vol_14"] = _safe_std(recent_atr_pcts)
    except Exception:
        feat["vol_of_vol_14"] = 0.0

    feat.update(_time_of_day_features(last.time if last is not None else None))
    feat.update(_opening_range_features(c, len(c) - 1, close))
    feat.update(_market_structure_features(c, len(c) - 1, close))

    recent_rets_30 = _returns(_series(closes[:-1], 30))
    feat["realized_vol_30"] = _safe_std(recent_rets_30)
    feat["atr_pct_regime_10"] = 0.0
    try:
        atr_regime_window: List[float] = []
        for offset in range(min(10, len(c))):
            sub_highs = highs[: len(c) - offset]
            sub_lows = lows[: len(c) - offset]
            sub_closes = closes[: len(c) - offset]
            atr_val = _safe_float(atr(sub_highs, sub_lows, sub_closes, period=14))
            close_val = _safe_float(sub_closes[-1]) if sub_closes else 0.0
            atr_regime_window.append(atr_val / close_val if close_val else 0.0)
        atr_regime_mean = mean(atr_regime_window) if atr_regime_window else 0.0
        feat["atr_pct_regime_10"] = feat["atr_pct"] / atr_regime_mean if atr_regime_mean else 0.0
    except Exception:
        feat["atr_pct_regime_10"] = 0.0
    atr_pct_history = []
    realized_vol_history = []
    for hist_end in range(len(c)):
        hist_close = closes[hist_end]
        atr_pct_history.append(_safe_float(atr(highs[: hist_end + 1], lows[: hist_end + 1], closes[: hist_end + 1], period=14)) / _safe_denominator(hist_close))
        hist_rets = _returns(closes[max(0, hist_end - 30) : hist_end])
        realized_vol_history.append(_safe_std(hist_rets))
    feat.update(
        _volatility_regime_features(
            atr_pct_history,
            realized_vol_history,
            atr_pct=feat["atr_pct"],
            realized_vol=feat["realized_vol_30"],
        )
    )

    opt_price = feat["ctx_option_price"]
    iv_val = feat["ctx_iv"]
    bid_price = max(feat["ctx_bid_price"], 0.0)
    ask_price = max(feat["ctx_ask_price"], 0.0)
    midpoint = (bid_price + ask_price) / 2.0 if bid_price > 0.0 and ask_price > 0.0 else max(opt_price, 0.0)
    if midpoint > 0.0 and ask_price >= bid_price > 0.0:
        feat["bid_ask_spread_pct"] = (ask_price - bid_price) / midpoint
    else:
        feat["bid_ask_spread_pct"] = 0.02 * (1.0 + iv_val) if opt_price > 0 else 0.0
    feat["option_bid_ask_spread_pct"] = feat["bid_ask_spread_pct"]
    feat["theta_to_vega_ratio"] = feat["ctx_theta"] / _safe_denominator(feat["ctx_vega"])
    feat["gamma_to_theta_ratio"] = feat["ctx_gamma"] / _safe_denominator(feat["ctx_theta"])

    raw_feature_order = [
        # OHLCV with last_ prefix (primary)
        "last_open", "last_high", "last_low", "last_close", "last_volume",
        # OHLCV without prefix (ml_feature_contract compatibility)
        "open", "high", "low", "close", "volume",
        # Candlestick ratios
        "body_pct", "range_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct", "close_location_pct",
        "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
        # Return stats
        "ret_mean", "ret_std", "ret_min", "ret_max",
        # Volume stats
        "vol_mean", "vol_std", "vol_min", "vol_max",
        # Return horizons
        "ret_1", "ret_3", "ret_5", "ret_10",
        # Technical indicators
        "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct", "adx_14", "roc_14",
        "choppiness_14", "supertrend_dir", "supertrend_gap_pct", "pivot_pp_dist_pct", "pivot_r1_dist_pct",
        "pivot_s1_dist_pct", "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct", "volume_ratio",
        # Regime
        "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
        # Direct strategy features
        "trend_following_strength", "trend_following_confidence", "trend_following_ema_gap_pct",
        "trend_following_momentum_pct", "trend_following_breakout_score", "trend_following_pullback_score",
        "trend_following_buy_call", "trend_following_buy_put",
        "mean_reversion_zscore", "mean_reversion_entry_score", "mean_reversion_expected_reversion_pct",
        "mean_reversion_half_life_bars", "mean_reversion_buy_call", "mean_reversion_buy_put",
        "stat_arb_zscore", "stat_arb_confidence", "stat_arb_spread_pct", "stat_arb_hedge_ratio",
        "stat_arb_long_spread", "stat_arb_short_spread",
        # Greeks context
        "ctx_iv", "ctx_iv_change_pct", "ctx_iv_percentile", "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
        "ctx_spot", "ctx_option_price", "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
        "ctx_time_sin", "ctx_time_cos", "ctx_dte_norm",
        # Ratio features
        "price_to_spot_pct", "option_to_spot_pct", "delta_abs", "greeks_imbalance",
        # Volatility regime
        "vol_of_vol_14", "is_opening_session", "is_closing_session", "is_midday_lull",
        # Opening range
        "dist_from_opening_high_pct", "dist_from_opening_low_pct", "opening_range_width_pct",
        "opening_range_breakout_strength",
        # Market structure
        "dist_to_rolling_high_20", "dist_to_rolling_low_20", "rolling_range_width_20", "rolling_range_position_20",
        # Volatility metrics
        "realized_vol_30", "atr_pct_regime_10",
        "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60", "volatility_regime_classifier",
        # Spread features
        "bid_ask_spread_pct", "theta_to_vega_ratio", "gamma_to_theta_ratio",
    ]
    feature_order = _active_feature_order(raw_feature_order)
    
    price_absolute_keys = {
        "last_open", "last_high", "last_low", "last_close",
        "open", "high", "low", "close",
        "ema_fast", "ema_slow", "ctx_spot", "ctx_option_price", "atr_14"
    }

    actual_close = _safe_float(last.close) if last is not None else 1.0
    if actual_close <= 0.0:
        actual_close = 1.0
        
    values = []
    for name in feature_order:
        val = float(feat.get(name, 0.0) or 0.0)
        if name in price_absolute_keys:
            val = val / actual_close
        values.append(val)
        
    return values, feature_order


def build_supervised_dataset(
    candles: Sequence[Candle],
    *,
    labels: Optional[Sequence[int]] = None,
    contexts: Optional[Sequence[MLFeatureContext | Dict[str, Any]]] = None,
    lookback: int = 20,
    horizon: int = 1,
    use_triple_barrier: bool = False,
    tb_profit_target_pct: float = 0.01,
    tb_stop_loss_pct: float = 0.005,
) -> Tuple[List[List[float]], List[int], List[str]]:
    c = list(candles or [])
    if len(c) <= max(lookback, horizon):
        return [], [], []

    X: List[List[float]] = []
    y: List[int] = []
    feature_names: List[str] = []

    for idx in range(lookback, len(c) - horizon):
        ctx = contexts[idx] if contexts is not None and idx < len(contexts) else None
        feats, feature_names = build_market_feature_vector(c[: idx + 1], context=ctx, lookback=lookback)
        if labels is not None and idx < len(labels):
            label = int(labels[idx])
        else:
            if use_triple_barrier:
                now_close = _safe_float(c[idx].close)
                label = 0  # default stop loss or quiet
                for f_idx in range(idx + 1, idx + horizon + 1):
                    f_close = _safe_float(c[f_idx].close)
                    ret = (f_close - now_close) / now_close if now_close else 0.0
                    if ret >= tb_profit_target_pct:
                        label = 1
                        break
                    elif ret <= -tb_stop_loss_pct:
                        label = 0
                        break
                else:
                    # temporal barrier hit
                    fut_close = _safe_float(c[idx + horizon].close)
                    label = 1 if fut_close > now_close else 0
            else:
                fut_close = _safe_float(c[idx + horizon].close)
                now_close = _safe_float(c[idx].close)
                label = 1 if fut_close > now_close else 0
        X.append(feats)
        y.append(label)
    return X, y, feature_names


def _metrics_from_predictions(y_true: Sequence[int], y_prob: Sequence[float], *, threshold: float = 0.5) -> Dict[str, float]:
    y_true_list = [1 if int(v) else 0 for v in y_true]
    y_pred = [1 if float(p) >= float(threshold) else 0 for p in y_prob]
    out: Dict[str, float] = {"samples": float(len(y_true_list)), "threshold": float(threshold)}
    if not y_true_list:
        return out

    # Compute metrics in pure Python as a baseline/fallback
    tp = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / len(y_true_list)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    out["accuracy"] = float(accuracy)
    out["precision"] = float(precision)
    out["recall"] = float(recall)
    out["f1"] = float(f1)

    if accuracy_score is not None:
        try:
            out["accuracy"] = float(accuracy_score(y_true_list, y_pred))
            out["precision"] = float(precision_score(y_true_list, y_pred, zero_division=0))
            out["recall"] = float(recall_score(y_true_list, y_pred, zero_division=0))
            out["f1"] = float(f1_score(y_true_list, y_pred, zero_division=0))
        except Exception:
            pass
        try:
            if len(set(y_true_list)) > 1:
                out["roc_auc"] = float(roc_auc_score(y_true_list, [float(p) for p in y_prob]))
        except Exception:
            pass
    return out


def walk_forward_backtest(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    n_splits: int = 5,
    gap: int = 10,
    threshold: float = 0.5,
    model_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run walk-forward model evaluation and return aggregate metrics."""

    X_list = [list(map(float, row)) for row in X]
    y_list = [int(v) for v in y]
    if not X_list or not y_list or len(X_list) != len(y_list):
        return {"ok": False, "reason": "invalid_data", "folds": [], "aggregate": {}}

    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:  # pragma: no cover - optional dependency
        RandomForestClassifier = FallbackClassifier

    if model_factory is None:
        def model_factory() -> Any:
            return RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
            )

    folds: List[Dict[str, Any]] = []
    aggregate_true: List[int] = []
    aggregate_prob: List[float] = []

    try:
        splitter = (
            TimeSeriesSplit(
                n_splits=max(2, int(n_splits)),
                gap=max(0, int(gap)),
            )
            if TimeSeriesSplit is not None
            else None
        )
    except Exception:
        splitter = None

    if splitter is not None:
        split_iter = splitter.split(X_list)
    else:
        split_points = []
        n = len(X_list)
        fold_count = max(2, int(n_splits))
        chunk = max(1, n // (fold_count + 1))
        for i in range(fold_count):
            train_end = min(n - 1, chunk * (i + 1))
            test_start = min(n, train_end + max(0, int(gap)))
            test_end = min(n, test_start + chunk)
            if train_end < 1 or test_end <= test_start:
                continue
            split_points.append((list(range(0, train_end)), list(range(test_start, test_end))))
        split_iter = iter(split_points)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        if len(train_idx) < 10 or len(test_idx) < 1:
            continue
        X_train = [X_list[i] for i in train_idx]
        y_train = [y_list[i] for i in train_idx]
        X_test = [X_list[i] for i in test_idx]
        y_test = [y_list[i] for i in test_idx]
        model = model_factory()
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                probs = [float(p[1]) if len(p) > 1 else float(p[0]) for p in model.predict_proba(X_test)]
            else:
                probs = [float(v) for v in model.predict(X_test)]
        except Exception as exc:
            folds.append({"fold": fold_idx, "ok": False, "error": str(exc)})
            continue

        metrics = _metrics_from_predictions(y_test, probs, threshold=threshold)
        metrics.update({"fold": float(fold_idx), "ok": True})
        folds.append(metrics)
        aggregate_true.extend(y_test)
        aggregate_prob.extend(probs)

    aggregate = _metrics_from_predictions(aggregate_true, aggregate_prob, threshold=threshold) if aggregate_true else {}
    return {"ok": bool(folds), "folds": folds, "aggregate": aggregate}


def verify_no_lookahead_leakage(
    candles: Sequence[Any],
    dataset_builder: Optional[Any] = None,
    lookback: int = 20,
    horizon: int = 5,
) -> Dict[str, Any]:
    """Audit feature construction for lookahead leakage by perturbing only future candles."""

    import copy
    import inspect

    audit_result: Dict[str, Any] = {
        "passed": True,
        "changed_features": [],
        "changed_rows": 0,
        "reason": "passed",
    }

    c = list(candles or [])
    dataset_builder = dataset_builder or build_supervised_dataset_v2
    perturbation_bars = 10

    if len(c) <= lookback + horizon + perturbation_bars:
        audit_result["reason"] = "insufficient_rows_for_audit"
        return audit_result

    base_candles = [
        Candle(
            time=src_candle.time,
            open=src_candle.open,
            high=src_candle.high,
            low=src_candle.low,
            close=src_candle.close,
            volume=src_candle.volume,
        )
        for src_candle in c
    ]

    builder_kwargs: Dict[str, Any] = {"lookback": lookback, "horizon": horizon}
    try:
        builder_sig = inspect.signature(dataset_builder)
        if "use_triple_barrier" in builder_sig.parameters:
            builder_kwargs["use_triple_barrier"] = True
    except Exception:
        builder_sig = None

    try:
        X_base, _, feature_names = dataset_builder(base_candles, **builder_kwargs)
    except Exception as exc:
        audit_result.update(
            {
                "passed": False,
                "reason": f"dataset_builder_failed_base: {exc}",
            }
        )
        return audit_result

    perturbed_candles = copy.deepcopy(base_candles)
    perturb_start = len(perturbed_candles) - perturbation_bars
    for idx in range(perturb_start, len(perturbed_candles)):
        perturbed_candles[idx].open *= 1.5
        perturbed_candles[idx].high *= 2.0
        perturbed_candles[idx].low *= 0.5
        perturbed_candles[idx].close *= 1.2
        perturbed_candles[idx].volume = (_safe_float(perturbed_candles[idx].volume) or 0.0) * 3.0

    try:
        X_pert, _, _ = dataset_builder(perturbed_candles, **builder_kwargs)
    except Exception as exc:
        audit_result.update(
            {
                "passed": False,
                "reason": f"dataset_builder_failed_perturbed: {exc}",
            }
        )
        return audit_result

    unaffected_rows = max(0, perturb_start - lookback)
    if unaffected_rows <= 0:
        audit_result["reason"] = "insufficient_unaffected_rows"
        return audit_result

    base_arr = np.array(X_base[:unaffected_rows], dtype=float)
    pert_arr = np.array(X_pert[:unaffected_rows], dtype=float)
    if base_arr.shape != pert_arr.shape:
        audit_result.update(
            {
                "passed": False,
                "reason": f"shape_mismatch_base_vs_perturbed ({base_arr.shape} != {pert_arr.shape})",
            }
        )
        return audit_result

    changed_mask = ~np.isclose(base_arr, pert_arr, atol=1e-8, equal_nan=True)
    if np.any(changed_mask):
        changed_rows = np.where(changed_mask.any(axis=1))[0]
        changed_cols = np.where(changed_mask.any(axis=0))[0]
        audit_result["passed"] = False
        audit_result["changed_rows"] = int(len(changed_rows))
        audit_result["changed_features"] = [feature_names[col_idx] for col_idx in changed_cols[:25]]
        first_row = int(changed_rows[0])
        first_col = int(changed_cols[0])
        audit_result["reason"] = (
            "future_perturbation_changed_current_features "
            f"(row={first_row}, feature={feature_names[first_col]})"
        )

    return audit_result


def walk_forward_validate_production(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    n_splits: int = 5,
    gap: int = 0,
    min_train_samples: int = 100,
    min_test_samples: int = 20,
    gate_roc_auc: float = 0.5,
    gate_accuracy: float = 0.5,
    gate_f1: float = 0.5,
    model_factory: Optional[Any] = None,
    candles: Optional[Sequence[Any]] = None,
    lookback: int = 20,
) -> Dict[str, Any]:
    """Run walk-forward validation and verify if metrics pass specific threshold gates."""
    X_list = [list(map(float, row)) for row in X]
    y_list = [int(v) for v in y]
    
    if not X_list or not y_list or len(X_list) != len(y_list):
        return {"passed": False, "reason": "invalid_data", "metrics": {}}

    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        RandomForestClassifier = FallbackClassifier

    if model_factory is None:
        def model_factory() -> Any:
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    n_samples = len(X_list)
    fold_count = max(2, int(n_splits))
    
    # Generate splits manually supporting min_train_samples, min_test_samples, and gap
    splits = []
    test_size = max(min_test_samples, (n_samples - min_train_samples - gap) // fold_count)
    if test_size <= 0:
        return {"passed": False, "reason": "insufficient_samples_for_splits", "metrics": {}}
        
    for i in range(fold_count):
        test_end = n_samples - i * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        
        if train_end < min_train_samples or test_start < 0:
            continue
            
        splits.append((list(range(0, train_end)), list(range(test_start, test_end))))
    
    if not splits:
        # Fallback to standard chunk-based splitting
        chunk = max(1, n_samples // (fold_count + 1))
        for i in range(fold_count):
            train_end = min(n_samples - 1, chunk * (i + 1))
            test_end = min(n_samples, train_end + chunk)
            if train_end > 10 and test_end > train_end:
                splits.append((list(range(0, train_end)), list(range(train_end, test_end))))
                
    if not splits:
        return {"passed": False, "reason": "no_valid_splits_generated", "metrics": {}}
        
    # Walk forward from past to future
    splits.reverse()
    
    aggregate_true = []
    aggregate_prob = []
    
    # Out of sample trading metrics variables
    all_test_dates = []
    all_trade_pnls = []
    wins = 0
    losses = 0
    profit_target = 0.01
    stop_loss = 0.005
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = [X_list[idx] for idx in train_idx]
        y_train = [y_list[idx] for idx in train_idx]
        X_test = [X_list[idx] for idx in test_idx]
        y_test = [y_list[idx] for idx in test_idx]
        
        if len(set(y_train)) < 2:
            continue
            
        model = model_factory()
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                probs = [float(p[1]) if len(p) > 1 else float(p[0]) for p in model.predict_proba(X_test)]
            else:
                probs = [float(v) for v in model.predict(X_test)]
        except Exception:
            continue
            
        aggregate_true.extend(y_test)
        aggregate_prob.extend(probs)
        
        # Simulate trades out of sample
        for i, idx in enumerate(test_idx):
            yt = y_test[i]
            yp = probs[i]
            pred = 1 if yp >= 0.5 else 0
            if pred == 1:
                candle_idx = lookback + idx
                c_date = candles[candle_idx].time.date() if (candles and candle_idx < len(candles)) else None
                if yt == 1:
                    pnl = profit_target
                    wins += 1
                else:
                    pnl = -stop_loss
                    losses += 1
                all_test_dates.append(c_date)
                all_trade_pnls.append(pnl)

    if not aggregate_true:
        return {"passed": False, "reason": "evaluation_failed_no_predictions", "metrics": {}}

    metrics = _metrics_from_predictions(aggregate_true, aggregate_prob)
    
    # Calculate daily Sharpe and Sortino ratios
    daily_returns_dict = {}
    for d, pnl in zip(all_test_dates, all_trade_pnls):
        if d:
            daily_returns_dict[d] = daily_returns_dict.get(d, 0.0) + pnl
            
    # Fill in zero returns for validation trading days with no active trades
    if candles and splits:
        all_test_indices = []
        for _, t_idx in splits:
            all_test_indices.extend(t_idx)
        all_test_indices.sort()
        start_idx = lookback + all_test_indices[0]
        end_idx = lookback + all_test_indices[-1]
        unique_days = sorted(list(set(candles[i].time.date() for i in range(start_idx, end_idx + 1) if i < len(candles))))
        for d in unique_days:
            if d not in daily_returns_dict:
                daily_returns_dict[d] = 0.0
                
    daily_returns = list(daily_returns_dict.values())
    
    import numpy as np
    
    # Sharpe Ratio
    if len(daily_returns) > 1:
        avg_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0.0 else 0.0
    else:
        sharpe = 0.0
        
    # Sortino Ratio
    if len(daily_returns) > 1:
        avg_ret = np.mean(daily_returns)
        downside_returns = [r for r in daily_returns if r < 0.0]
        if downside_returns:
            downside_std = np.std(downside_returns, ddof=1)
            sortino = (avg_ret / downside_std) * np.sqrt(252) if downside_std > 0.0 else 0.0
        else:
            sortino = 99.0 if avg_ret > 0.0 else 0.0
    else:
        sortino = 0.0
        
    # Profit Factor
    wins_sum = sum(p for p in all_trade_pnls if p > 0.0)
    losses_sum = sum(abs(p) for p in all_trade_pnls if p < 0.0)
    profit_factor = wins_sum / losses_sum if losses_sum > 0.0 else (99.0 if wins_sum > 0.0 else 1.0)
    
    # Max Drawdown
    cum_pnl = np.cumsum(all_trade_pnls) if all_trade_pnls else [0.0]
    peak = 0.0
    max_dd = 0.0
    for val in cum_pnl:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
            
    metrics.update({
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(profit_factor),
        "drawdown": float(max_dd),
        "win_rate": float(wins / len(all_trade_pnls) if all_trade_pnls else 0.0),
        "expectancy": float(np.mean(all_trade_pnls) if all_trade_pnls else 0.0),
        "trades_count": int(len(all_trade_pnls))
    })
    
    roc_auc = metrics.get("roc_auc", 0.0)
    accuracy = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1", 0.0)
    
    passed = True
    reason = "passed"
    
    # Check gates
    if roc_auc < gate_roc_auc:
        passed = False
        reason = f"roc_auc_below_gate ({roc_auc:.4f} < {gate_roc_auc})"
    elif accuracy < gate_accuracy:
        passed = False
        reason = f"accuracy_below_gate ({accuracy:.4f} < {gate_accuracy})"
    elif f1 < gate_f1:
        passed = False
        reason = f"f1_below_gate ({f1:.4f} < {gate_f1})"
        
    return {
        "passed": passed,
        "reason": reason,
        "metrics": metrics
    }


def _compute_global_ema(closes: List[float], period: int) -> List[Optional[float]]:
    n = len(closes)
    out = [None] * n
    if n < period:
        return out
    k = 2.0 / (period + 1.0)
    curr = closes[0]
    for i in range(n):
        if i == 0:
            curr = closes[0]
        else:
            curr = closes[i] * k + curr * (1.0 - k)
        if i >= period - 1:
            out[i] = curr
    return out


def _compute_global_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
    n = len(closes)
    out = [None] * n
    if n < 2 * period:
        return out

    trs = []
    plus_dms = []
    minus_dms = []

    for i in range(1, n):
        h = highs[i]
        l = lows[i]
        prev_c = closes[i-1]
        prev_h = highs[i-1]
        prev_l = lows[i-1]

        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        up_move = h - prev_h
        down_move = prev_l - l

        if up_move > down_move and up_move > 0:
            plus_dm = up_move
        else:
            plus_dm = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm = down_move
        else:
            minus_dm = 0.0

        trs.append(tr)
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)

    def wilders_smooth(values: list[float], n: int) -> list[float]:
        if len(values) < n:
            return []
        smoothed = []
        curr_smooth = sum(values[:n]) / n
        smoothed.append(curr_smooth)
        for i in range(n, len(values)):
            curr_smooth = ((curr_smooth * (n - 1)) + values[i]) / n
            smoothed.append(curr_smooth)
        return smoothed

    tr_smooth = wilders_smooth(trs, period)
    plus_dm_smooth = wilders_smooth(plus_dms, period)
    minus_dm_smooth = wilders_smooth(minus_dms, period)

    if not tr_smooth:
        return out

    dx_values = []
    for i in range(len(tr_smooth)):
        tr_val = tr_smooth[i]
        if tr_val == 0:
            dx_values.append(0.0)
            continue
        plus_di = (plus_dm_smooth[i] / tr_val) * 100.0
        minus_di = (minus_dm_smooth[i] / tr_val) * 100.0
        sum_di = plus_di + minus_di
        if sum_di == 0:
            dx = 0.0
        else:
            dx = (abs(plus_di - minus_di) / sum_di) * 100.0
        dx_values.append(dx)

    adx_smooth = wilders_smooth(dx_values, period)
    if not adx_smooth:
        return out

    for idx in range(2 * period - 1, n):
        out[idx] = adx_smooth[idx - 2 * period + 1]
    return out


def _compute_global_supertrend(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> List[Optional[float]]:
    n = len(closes)
    out = [None] * n
    if n < period + 1:
        return out

    tr_vals = [0.0] * n
    for i in range(1, n):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr_vals[i] = max(h - l, abs(h - pc), abs(l - pc))
    
    atr_vals = [0.0] * n
    if n > period:
        atr_vals[period] = sum(tr_vals[1:period+1]) / period
        for i in range(period + 1, n):
            atr_vals[i] = (atr_vals[i-1] * (period - 1) + tr_vals[i]) / period

    if n <= period:
        return out

    upper_band = [0.0] * n
    lower_band = [0.0] * n
    supertrend_vals = [0.0] * n
    trend = [1] * n

    first_valid = period
    hl2 = (highs[first_valid] + lows[first_valid]) / 2
    curr_atr = atr_vals[first_valid]
    upper_band[first_valid] = hl2 + (multiplier * curr_atr)
    lower_band[first_valid] = hl2 - (multiplier * curr_atr)
    supertrend_vals[first_valid] = upper_band[first_valid]

    for i in range(first_valid + 1, n):
        hl2 = (highs[i] + lows[i]) / 2
        curr_atr = atr_vals[i]
        ub = hl2 + (multiplier * curr_atr)
        lb = hl2 - (multiplier * curr_atr)
        prev_ub = upper_band[i-1]
        prev_lb = lower_band[i-1]
        prev_close = closes[i-1]
        
        if ub < prev_ub or prev_close > prev_ub:
            upper_band[i] = ub
        else:
            upper_band[i] = prev_ub
            
        if lb > prev_lb or prev_close < prev_lb:
            lower_band[i] = lb
        else:
            lower_band[i] = prev_lb
            
        prev_trend = trend[i-1]
        curr_close = closes[i]
        
        if prev_trend == 1:
            if curr_close < lower_band[i]:
                trend[i] = -1
                supertrend_vals[i] = upper_band[i]
            else:
                trend[i] = 1
                supertrend_vals[i] = lower_band[i]
        else:
            if curr_close > upper_band[i]:
                trend[i] = 1
                supertrend_vals[i] = lower_band[i]
            else:
                trend[i] = -1
                supertrend_vals[i] = upper_band[i]

    for idx in range(period, n):
        out[idx] = supertrend_vals[idx]
    return out


def compute_triple_barrier_labels_vectorized(
    closes: List[float],
    lookback: int,
    horizon: int,
    use_triple_barrier: bool = False,
    tb_profit_target_pct: float = 0.01,
    tb_stop_loss_pct: float = 0.005,
) -> List[int]:
    n = len(closes)
    num_samples = n - horizon - lookback
    if num_samples <= 0:
        return []
    
    closes_arr = np.array(closes, dtype=float)
    idx_range = np.arange(lookback, n - horizon)
    
    if not use_triple_barrier:
        fut_close_temp = closes_arr[idx_range + horizon]
        now_close_temp = closes_arr[idx_range]
        labels = np.where(fut_close_temp > now_close_temp, 1, 0)
        return labels.tolist()
    
    has_hit = np.zeros(num_samples, dtype=bool)
    labels = np.zeros(num_samples, dtype=int)
    now_closes = closes_arr[idx_range]
    
    for h in range(1, horizon + 1):
        fut_closes = closes_arr[idx_range + h]
        with np.errstate(divide='ignore', invalid='ignore'):
            rets = np.where(now_closes != 0.0, (fut_closes - now_closes) / now_closes, 0.0)
        
        tp_hit = rets >= tb_profit_target_pct
        sl_hit = rets <= -tb_stop_loss_pct
        
        active = ~has_hit
        any_hit_now = (tp_hit | sl_hit) & active
        
        if np.any(any_hit_now):
            labels[any_hit_now] = np.where(tp_hit[any_hit_now], 1, 0)
            has_hit[any_hit_now] = True
            
        if np.all(has_hit):
            break
            
    no_hit = ~has_hit
    if np.any(no_hit):
        fut_closes_end = closes_arr[idx_range + horizon]
        labels[no_hit] = np.where(fut_closes_end[no_hit] > now_closes[no_hit], 1, 0)
        
    return labels.tolist()


def build_supervised_dataset_v2(
    candles: Sequence[Candle],
    *,
    labels: Optional[Sequence[int]] = None,
    contexts: Optional[Sequence[MLFeatureContext | Dict[str, Any]]] = None,
    lookback: int = 20,
    horizon: int = 1,
    use_triple_barrier: bool = False,
    tb_profit_target_pct: float = 0.01,
    tb_stop_loss_pct: float = 0.005,
    **kwargs,
) -> Tuple[List[List[float]], List[int], List[str]]:
    c = list(candles or [])
    n = len(c)
    if n <= max(lookback, horizon):
        return [], [], []

    closes = [_safe_float(x.close) for x in c]
    highs = [_safe_float(x.high) for x in c]
    lows = [_safe_float(x.low) for x in c]
    volumes = [_safe_float(x.volume) for x in c]

    ema_fast_arr = _compute_global_ema(closes, 9)
    ema_slow_arr = _compute_global_ema(closes, 21)
    adx_14_arr = _compute_global_adx(highs, lows, closes, 14)
    supertrend_vals_arr = _compute_global_supertrend(highs, lows, closes, 10, 3.0)

    # Compute ATR globally
    atr_14_arr = [0.0] * n
    for i in range(n):
        try:
            atr_14_arr[i] = _safe_float(atr(highs[:i+1], lows[:i+1], closes[:i+1], period=14))
        except Exception:
            atr_14_arr[i] = 0.0

    global_rets = [0.0] * n
    for i in range(1, n):
        prev = closes[i - 1]
        global_rets[i] = (closes[i] - prev) / prev if prev else 0.0

    candlestick_feats = []
    for i in range(n):
        prev_candle = c[i - 1] if i > 0 else None
        candlestick_feats.append(_candlestick_context(c[i], prev_candle))

    raw_feature_order = [
        # OHLCV with last_ prefix (primary)
        "last_open", "last_high", "last_low", "last_close", "last_volume",
        # OHLCV without prefix (ml_feature_contract compatibility)
        "open", "high", "low", "close", "volume",
        # Candlestick ratios
        "body_pct", "range_pct", "gap_pct", "upper_wick_pct", "lower_wick_pct", "close_location_pct",
        "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
        # Return stats
        "ret_mean", "ret_std", "ret_min", "ret_max",
        # Volume stats
        "vol_mean", "vol_std", "vol_min", "vol_max",
        # Return horizons
        "ret_1", "ret_3", "ret_5", "ret_10",
        # Technical indicators
        "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct", "adx_14", "roc_14",
        "choppiness_14", "supertrend_dir", "supertrend_gap_pct", "pivot_pp_dist_pct", "pivot_r1_dist_pct",
        "pivot_s1_dist_pct", "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct", "volume_ratio",
        # Regime
        "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
        # Direct strategy features
        "trend_following_strength", "trend_following_confidence", "trend_following_ema_gap_pct",
        "trend_following_momentum_pct", "trend_following_breakout_score", "trend_following_pullback_score",
        "trend_following_buy_call", "trend_following_buy_put",
        "mean_reversion_zscore", "mean_reversion_entry_score", "mean_reversion_expected_reversion_pct",
        "mean_reversion_half_life_bars", "mean_reversion_buy_call", "mean_reversion_buy_put",
        "stat_arb_zscore", "stat_arb_confidence", "stat_arb_spread_pct", "stat_arb_hedge_ratio",
        "stat_arb_long_spread", "stat_arb_short_spread",
        # Greeks context
        "ctx_iv", "ctx_iv_change_pct", "ctx_iv_percentile", "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
        "ctx_spot", "ctx_option_price", "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
        "ctx_time_sin", "ctx_time_cos", "ctx_dte_norm",
        # Ratio features
        "price_to_spot_pct", "option_to_spot_pct", "delta_abs", "greeks_imbalance",
        # Volatility regime
        "vol_of_vol_14", "is_opening_session", "is_closing_session", "is_midday_lull",
        # Opening range
        "dist_from_opening_high_pct", "dist_from_opening_low_pct", "opening_range_width_pct",
        "opening_range_breakout_strength",
        # Market structure
        "dist_to_rolling_high_20", "dist_to_rolling_low_20", "rolling_range_width_20", "rolling_range_position_20",
        # Volatility metrics
        "realized_vol_30", "atr_pct_regime_10",
        "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60", "volatility_regime_classifier",
        # Spread features
        "bid_ask_spread_pct", "theta_to_vega_ratio", "gamma_to_theta_ratio",
    ]
    feature_order = _active_feature_order(raw_feature_order)

    price_absolute_keys = {
        "last_open", "last_high", "last_low", "last_close",
        "open", "high", "low", "close",
        "ema_fast", "ema_slow", "ctx_spot", "ctx_option_price", "atr_14"
    }

    X: List[List[float]] = []

    for idx in range(lookback, n - horizon):
        f = dict(candlestick_feats[idx])

        recent = closes[idx - lookback + 1 : idx + 1]
        recent_vols = volumes[idx - lookback + 1 : idx + 1]
        recent_rets = global_rets[idx - lookback + 2 : idx + 1]
        
        ret_stats = _rolling_stats(recent_rets)
        vol_stats = _rolling_stats(recent_vols)
        f.update({f"ret_{k}": v for k, v in ret_stats.items()})
        f.update({f"vol_{k}": v for k, v in vol_stats.items()})

        f["ret_1"] = global_rets[idx]
        f["ret_3"] = (closes[idx] - closes[idx-3]) / closes[idx-3] if idx >= 3 else f["ret_1"]
        f["ret_5"] = (closes[idx] - closes[idx-5]) / closes[idx-5] if idx >= 5 else f["ret_1"]
        f["ret_10"] = (closes[idx] - closes[idx-10]) / closes[idx-10] if idx >= 10 else f["ret_1"]

        f["ema_fast"] = _safe_float(ema_fast_arr[idx])
        f["ema_slow"] = _safe_float(ema_slow_arr[idx])
        f["ema_diff_pct"] = (f["ema_fast"] - f["ema_slow"]) / closes[idx] if closes[idx] else 0.0

        try:
            f["rsi_14"] = _safe_float(rsi(closes[:idx+1], period=14))
        except Exception:
            f["rsi_14"] = 50.0
        try:
            f["atr_14"] = _safe_float(atr(highs[:idx+1], lows[:idx+1], closes[:idx+1], period=14))
            f["atr_pct"] = f["atr_14"] / closes[idx] if closes[idx] else 0.0
        except Exception:
            f["atr_14"] = f["atr_pct"] = 0.0
        try:
            f["adx_14"] = _safe_float(adx_14_arr[idx])
        except Exception:
            f["adx_14"] = 0.0
        try:
            f["roc_14"] = _safe_float(roc(closes[:idx+1], period=14))
        except Exception:
            f["roc_14"] = 0.0
        try:
            f["choppiness_14"] = _safe_float(choppiness_index(highs[:idx+1], lows[:idx+1], closes[:idx+1], period=14))
        except Exception:
            f["choppiness_14"] = 50.0
        try:
            st = supertrend_vals_arr[idx]
            f["supertrend_dir"] = 1.0 if st is not None and closes[idx] >= _safe_float(st) else (-1.0 if st is not None else 0.0)
            f["supertrend_gap_pct"] = abs(closes[idx] - _safe_float(st)) / closes[idx] if st is not None and closes[idx] else 0.0
        except Exception:
            f["supertrend_dir"] = 0.0
            f["supertrend_gap_pct"] = 0.0

        try:
            piv = pivot_points(highs[idx], lows[idx], closes[idx])
            f["pivot_pp_dist_pct"] = (closes[idx] - _safe_float(piv.get("pp"))) / closes[idx] if closes[idx] else 0.0
            f["pivot_r1_dist_pct"] = (closes[idx] - _safe_float(piv.get("r1"))) / closes[idx] if closes[idx] else 0.0
            f["pivot_s1_dist_pct"] = (closes[idx] - _safe_float(piv.get("s1"))) / closes[idx] if closes[idx] else 0.0
        except Exception:
            f["pivot_pp_dist_pct"] = f["pivot_r1_dist_pct"] = f["pivot_s1_dist_pct"] = 0.0

        f["close_vs_open_pct"] = (closes[idx] - f["last_open"]) / closes[idx] if closes[idx] else 0.0
        f["range_to_atr"] = f["range_pct"] / f["atr_pct"] if f["atr_pct"] else 0.0
        f["momentum_lookback_pct"] = (closes[idx] - recent[0]) / recent[0] if len(recent) > 1 and recent[0] else 0.0
        f["volume_ratio"] = f["last_volume"] / f["vol_mean"] if f["vol_mean"] else 0.0

        ctx = contexts[idx] if contexts is not None and idx < len(contexts) else None
        regime = ""
        if ctx is not None:
            if isinstance(ctx, dict):
                regime = str(ctx.get("regime") or "")
            else:
                regime = str(getattr(ctx, "regime", "") or "")
        if not regime:
            try:
                regime = detect_regime([f["atr_14"]] * 30, [f["adx_14"]] * 30, [f["rsi_14"]] * 30)
            except Exception:
                regime = "quiet"
        f.update(_one_hot_regime(regime))

        def get_ctx(name: str, default: float = 0.0) -> float:
            if ctx is None:
                return float(default)
            if isinstance(ctx, dict):
                return _safe_float(ctx.get(name), default)
            return _safe_float(getattr(ctx, name, default), default)

        f["ctx_iv"] = get_ctx("iv", 0.0)
        f["ctx_iv_change_pct"] = get_ctx("iv_change_pct", 0.0)
        f["ctx_iv_percentile"] = get_ctx("iv_percentile", 0.0)
        f["ctx_delta"] = get_ctx("delta", 0.0)
        f["ctx_gamma"] = get_ctx("gamma", 0.0)
        f["ctx_vega"] = get_ctx("vega", 0.0)
        f["ctx_theta"] = get_ctx("theta", 0.0)
        f["ctx_spot"] = get_ctx("spot", closes[idx])
        f["ctx_option_price"] = get_ctx("option_price", 0.0)
        f["ctx_adx"] = get_ctx("adx", f["adx_14"])
        f["ctx_trend_strength"] = get_ctx("trend_strength", f["ema_diff_pct"])
        f["ctx_choppiness"] = get_ctx("choppiness", f["choppiness_14"])
        f["ctx_volume_sma"] = get_ctx("volume_sma", f["vol_mean"])
        f["ctx_time_sin"] = get_ctx("time_sin", 0.0)
        f["ctx_time_cos"] = get_ctx("time_cos", 0.0)
        f["ctx_dte_norm"] = get_ctx("dte_norm", 0.0)
        f["ctx_bid_price"] = get_ctx("bid_price", 0.0)
        f["ctx_ask_price"] = get_ctx("ask_price", 0.0)

        spot_denom = _safe_denominator(f["ctx_spot"])
        f["price_to_spot_pct"] = (closes[idx] - f["ctx_spot"]) / spot_denom
        f["option_to_spot_pct"] = f["ctx_option_price"] / spot_denom
        f["delta_abs"] = abs(f["ctx_delta"])
        f["greeks_imbalance"] = f["ctx_gamma"] + f["ctx_vega"] + f["ctx_theta"]

        actual_close = closes[idx]
        if actual_close <= 0.0:
            actual_close = 1.0

        # Vol of Vol
        try:
            recent_atrs = [atr_14_arr[idx - k] / closes[idx - k] if closes[idx - k] else 0.0 for k in range(14) if idx - k >= 0]
            f["vol_of_vol_14"] = float(np.std(recent_atrs)) if len(recent_atrs) > 1 else 0.0
        except Exception:
            f["vol_of_vol_14"] = 0.0

        f.update(_time_of_day_features(c[idx].time))
        f.update(_opening_range_features(c, idx, closes[idx]))
        f.update(_market_structure_features(c, idx, closes[idx]))
        f["realized_vol_30"] = _safe_std(global_rets[max(1, idx - 29) : idx])
        recent_atr_window = [atr_14_arr[pos] / _safe_denominator(closes[pos]) for pos in range(max(0, idx - 10), idx)]
        atr_regime_mean = mean(recent_atr_window) if recent_atr_window else 0.0
        f["atr_pct_regime_10"] = f["atr_pct"] / atr_regime_mean if atr_regime_mean else 0.0
        atr_pct_history = [atr_14_arr[pos] / _safe_denominator(closes[pos]) for pos in range(idx + 1)]
        realized_vol_history = [_safe_std(global_rets[max(1, pos - 29) : pos]) for pos in range(idx + 1)]
        f.update(
            _volatility_regime_features(
                atr_pct_history,
                realized_vol_history,
                atr_pct=f["atr_pct"],
                realized_vol=f["realized_vol_30"],
            )
        )

        # Option liquidity feature
        opt_price = get_ctx("option_price", 0.0)
        iv_val = get_ctx("iv", 0.16)
        bid_price = max(f["ctx_bid_price"], 0.0)
        ask_price = max(f["ctx_ask_price"], 0.0)
        midpoint = (bid_price + ask_price) / 2.0 if bid_price > 0.0 and ask_price > 0.0 else max(opt_price, 0.0)
        if midpoint > 0.0 and ask_price >= bid_price > 0.0:
            f["bid_ask_spread_pct"] = (ask_price - bid_price) / midpoint
        else:
            f["bid_ask_spread_pct"] = 0.02 * (1.0 + iv_val) if opt_price > 0 else 0.0
        f["option_bid_ask_spread_pct"] = f["bid_ask_spread_pct"]

        # Greeks ratios
        theta_val = get_ctx("theta", 0.0)
        vega_val = get_ctx("vega", 0.0)
        gamma_val = get_ctx("gamma", 0.0)
        f["theta_to_vega_ratio"] = theta_val / _safe_denominator(vega_val)
        f["gamma_to_theta_ratio"] = gamma_val / _safe_denominator(theta_val)

        row_values = []
        for name in feature_order:
            val = float(f.get(name, 0.0) or 0.0)
            if name in price_absolute_keys:
                val = val / actual_close
            row_values.append(val)
        X.append(row_values)

    if labels is not None:
        y = [int(labels[idx]) for idx in range(lookback, n - horizon)]
    else:
        y = compute_triple_barrier_labels_vectorized(
            closes=closes,
            lookback=lookback,
            horizon=horizon,
            use_triple_barrier=use_triple_barrier,
            tb_profit_target_pct=tb_profit_target_pct,
            tb_stop_loss_pct=tb_stop_loss_pct,
        )

    return X, y, feature_order


def build_model_registry_metadata(training_samples: int, walk_forward: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Helper to structure metadata entry for model registry."""
    return {
        "dataset_size": training_samples,
        "features": kwargs.get("features", 0),
        "walk_forward": walk_forward,
        "metrics": {
            "roc_auc": walk_forward.get("roc_auc", 0.0),
            "accuracy": walk_forward.get("accuracy", 0.0),
            "f1": walk_forward.get("f1", 0.0),
            "sharpe": kwargs.get("sharpe", 0.0),
            "profit_factor": kwargs.get("profit_factor", 0.0),
            "drawdown": kwargs.get("drawdown", 0.0)
        }
    }


LABEL_POLICY = "triple_barrier"

EXCLUDED_MODEL_FEATURES = {
    # Removed range_pct, ret_1, roc_14, supertrend_gap_pct - these are required
    # by ml_feature_contract.py CORE_REQUIRED_FEATURES and must not be excluded.
    "ctx_time_sin",
    "ctx_time_cos",
    "price_to_spot_pct",
}


def _active_feature_order(feature_order: Sequence[str]) -> List[str]:
    return [str(name) for name in feature_order if str(name) not in EXCLUDED_MODEL_FEATURES]

TARGET_FEATURES = [
    "last_open", "last_high", "last_low", "body_pct", "range_pct",
    "gap_pct", "upper_wick_pct", "lower_wick_pct", "close_location_pct", "bullish_engulfing",
    "bearish_engulfing", "doji", "ret_mean", "ret_std", "ret_min", "ret_max",
    "ret_1", "ret_3", "ret_5", "ret_10",
    "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct", "adx_14", "roc_14",
    "choppiness_14", "supertrend_dir", "supertrend_gap_pct", "pivot_pp_dist_pct", "pivot_r1_dist_pct",
    "pivot_s1_dist_pct", "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
    "regime_mean_reverting", "regime_quiet", "is_closing_session",
]
TARGET_FEATURES = _active_feature_order(TARGET_FEATURES)

