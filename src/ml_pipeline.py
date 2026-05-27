"""Feature engineering and ML evaluation helpers for the signal model.

The helpers here are intentionally lightweight and dependency-tolerant so the
project can run even when some scientific packages are missing. The feature
set mixes candle context, indicators, regime, IV, and Greeks context.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from candlestick_patterns import is_bearish_engulfing, is_bullish_engulfing, is_doji, is_hammer, is_shooting_star
from indicators import adx, atr, choppiness_index, ema, pivot_points, roc, rsi, supertrend
from market_data import Candle
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
    adx: Optional[float] = None
    trend_strength: Optional[float] = None
    choppiness: Optional[float] = None
    volume_sma: Optional[float] = None


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

    feat["price_to_spot_pct"] = (close - feat["ctx_spot"]) / feat["ctx_spot"] if feat["ctx_spot"] else 0.0
    feat["option_to_spot_pct"] = feat["ctx_option_price"] / feat["ctx_spot"] if feat["ctx_spot"] else 0.0
    feat["delta_abs"] = abs(feat["ctx_delta"])
    feat["greeks_imbalance"] = feat["ctx_gamma"] + feat["ctx_vega"] + feat["ctx_theta"]

    feature_order = [
        "last_open", "last_high", "last_low", "last_close", "last_volume", "body_pct", "range_pct",
        "gap_pct", "upper_wick_pct", "lower_wick_pct", "close_location_pct", "bullish_engulfing",
        "bearish_engulfing", "doji", "hammer", "shooting_star", "ret_mean", "ret_std", "ret_min", "ret_max",
        "vol_mean", "vol_std", "vol_min", "vol_max", "ret_1", "ret_3", "ret_5", "ret_10",
        "ema_fast", "ema_slow", "ema_diff_pct", "rsi_14", "atr_14", "atr_pct", "adx_14", "roc_14",
        "choppiness_14", "supertrend_dir", "supertrend_gap_pct", "pivot_pp_dist_pct", "pivot_r1_dist_pct",
        "pivot_s1_dist_pct", "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct", "volume_ratio",
        "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet",
        "ctx_iv", "ctx_iv_change_pct", "ctx_iv_percentile", "ctx_delta", "ctx_gamma", "ctx_vega", "ctx_theta",
        "ctx_spot", "ctx_option_price", "ctx_adx", "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma",
        "price_to_spot_pct", "option_to_spot_pct", "delta_abs", "greeks_imbalance",
    ]
    
    price_absolute_keys = {
        "last_open", "last_high", "last_low", "last_close",
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
                    future_close = _safe_float(c[idx + horizon].close)
                    label = 1 if future_close > now_close else 0
            else:
                future_close = _safe_float(c[idx + horizon].close)
                now_close = _safe_float(c[idx].close)
                label = 1 if future_close > now_close else 0
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
            return RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=2)

    folds: List[Dict[str, Any]] = []
    aggregate_true: List[int] = []
    aggregate_prob: List[float] = []

    try:
        splitter = TimeSeriesSplit(n_splits=max(2, int(n_splits))) if TimeSeriesSplit is not None else None
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
            test_end = min(n, train_end + chunk)
            if train_end < 1 or test_end <= train_end:
                continue
            split_points.append((list(range(0, train_end)), list(range(train_end, test_end))))
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
    if aggregate:
        aggregate["fold_count"] = float(len([f for f in folds if f.get("ok")]))
        aggregate["positive_rate"] = float(sum(aggregate_true) / len(aggregate_true)) if aggregate_true else 0.0
    return {"ok": bool(folds), "folds": folds, "aggregate": aggregate}
