from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class FinalDayCompleteness:
    trading_day: str
    candle_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    expected_candles: int
    is_complete: bool
    is_provisional: bool
    reason: str


def expected_session_candle_count() -> int:
    return 375


def compute_final_day_completeness(candles: Sequence[Any]) -> FinalDayCompleteness:
    if not candles:
        return FinalDayCompleteness(
            trading_day="",
            candle_count=0,
            first_timestamp=None,
            last_timestamp=None,
            expected_candles=expected_session_candle_count(),
            is_complete=False,
            is_provisional=False,
            reason="no_candles",
        )

    last_day = candles[-1].time.date()
    day_rows = [c for c in candles if c.time.date() == last_day]
    expected = expected_session_candle_count()
    first_ts = day_rows[0].time.isoformat() if day_rows else None
    last_ts = day_rows[-1].time.isoformat() if day_rows else None
    is_complete = bool(
        day_rows
        and len(day_rows) >= expected
        and day_rows[0].time.hour == 9
        and day_rows[0].time.minute == 15
        and day_rows[-1].time.hour == 15
        and day_rows[-1].time.minute >= 29
    )
    return FinalDayCompleteness(
        trading_day=last_day.isoformat(),
        candle_count=len(day_rows),
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        expected_candles=expected,
        is_complete=is_complete,
        is_provisional=not is_complete,
        reason="complete_session" if is_complete else "partial_or_missing_session_bars",
    )


def generate_purged_splits(
    n_samples: int,
    *,
    n_splits: int,
    gap: int,
    min_train_samples: int,
    min_test_samples: int,
) -> List[Dict[str, int]]:
    fold_count = max(2, int(n_splits))
    test_size = max(int(min_test_samples), (n_samples - min_train_samples - gap) // fold_count)
    if test_size <= 0:
        return []

    splits: List[Dict[str, int]] = []
    for idx in range(fold_count):
        test_end = n_samples - idx * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        if train_end < min_train_samples or test_start < 0:
            continue
        splits.append(
            {
                "train_start": 0,
                "train_end": train_end,
                "validation_start": test_start,
                "validation_end": test_end,
            }
        )

    if not splits:
        chunk = max(1, n_samples // (fold_count + 1))
        for idx in range(fold_count):
            train_end = min(n_samples - 1, chunk * (idx + 1))
            test_end = min(n_samples, train_end + chunk)
            if train_end > min_train_samples and test_end - train_end >= min_test_samples:
                splits.append(
                    {
                        "train_start": 0,
                        "train_end": train_end,
                        "validation_start": train_end,
                        "validation_end": test_end,
                    }
                )
    return list(reversed(splits))


def generate_purged_embargoed_cv_splits(
    df: pd.DataFrame,
    label_horizon_bars: int,
    embargo_pct: float = 0.01,
    *,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged and embargoed walk-forward splits for time-series labels.

    Purging removes training samples whose label evaluation windows would overlap
    the start of the validation window. Embargo removes a short post-validation
    slice to avoid contamination from immediately-following observations.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by a pandas DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df index must be timezone-aware")
    if int(label_horizon_bars) <= 0:
        raise ValueError("label_horizon_bars must be positive")
    if len(df.index) < 3:
        return []

    ordered = df.sort_index(kind="stable")
    sample_count = len(ordered.index)
    fold_count = max(2, int(n_splits))
    validation_size = max(1, sample_count // (fold_count + 1))
    embargo_bars = max(1, int(math.ceil(float(label_horizon_bars) * (1.0 + float(embargo_pct)))))
    ordered_index = pd.to_datetime(ordered.index)
    if sample_count >= 2:
        deltas = ordered_index.to_series().diff().dropna()
        bar_delta = deltas.median() if not deltas.empty else pd.Timedelta(minutes=1)
    else:
        bar_delta = pd.Timedelta(minutes=1)
    label_horizon_delta = bar_delta * int(label_horizon_bars)
    embargo_delta = bar_delta * embargo_bars

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(fold_count):
        validation_end = sample_count - (fold_count - 1 - fold_idx) * validation_size
        validation_start = max(validation_size, validation_end - validation_size)
        if validation_start >= validation_end or validation_end > sample_count:
            continue

        validation_start_ts = pd.to_datetime(ordered_index[validation_start])
        validation_end_ts = pd.to_datetime(ordered_index[validation_end - 1])
        train_mask = np.ones(sample_count, dtype=bool)
        train_mask[validation_start:validation_end] = False

        label_end_times = ordered_index + label_horizon_delta
        purge_mask = np.asarray((ordered_index < validation_start_ts) & (label_end_times >= validation_start_ts), dtype=bool)
        embargo_end_ts = validation_end_ts + embargo_delta
        embargo_mask = np.asarray((ordered_index > validation_end_ts) & (ordered_index <= embargo_end_ts), dtype=bool)
        train_mask[purge_mask] = False
        train_mask[embargo_mask] = False

        train_idx = np.flatnonzero(train_mask)
        validation_idx = np.arange(validation_start, validation_end, dtype=int)
        if train_idx.size == 0 or validation_idx.size == 0:
            continue
        if train_idx[-1] >= validation_idx[0]:
            train_idx = train_idx[train_idx < validation_idx[0]]
        if train_idx.size == 0:
            continue
        splits.append((train_idx, validation_idx))

    LOGGER.info(
        "Generated %s purged/embargoed splits for %s samples with horizon=%s embargo_bars=%s",
        len(splits),
        sample_count,
        label_horizon_bars,
        embargo_bars,
    )
    return splits


def optimize_threshold_for_economic_edge(
    y_true: Sequence[int],
    y_prob_positive: Sequence[float],
    min_trades: int = 20,
) -> Dict[str, float]:
    """Sweep thresholds and return the most stable F1-optimal cutoff.

    Thresholds that generate fewer than ``min_trades`` predicted executions are
    automatically discarded to avoid trivial optima on sparse positives.
    """

    y_true_list = np.asarray([1 if int(v) else 0 for v in y_true], dtype=int)
    y_prob_list = np.asarray([float(v) for v in y_prob_positive], dtype=float)
    if y_true_list.size == 0 or y_true_list.size != y_prob_list.size:
        return {"threshold": 0.5, "f1": 0.0, "trades_count": 0.0}

    best = {"threshold": 0.5, "f1": -1.0, "trades_count": 0.0}
    for threshold in np.arange(0.01, 1.00, 0.01, dtype=float):
        predictions = y_prob_list >= threshold
        trades_count = int(np.sum(predictions))
        if trades_count < int(min_trades):
            continue

        tp = int(np.sum((y_true_list == 1) & predictions))
        fp = int(np.sum((y_true_list == 0) & predictions))
        fn = int(np.sum((y_true_list == 1) & (~predictions)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if f1 > best["f1"] or (
            math.isclose(f1, best["f1"], rel_tol=1e-12, abs_tol=1e-12)
            and trades_count > best["trades_count"]
        ):
            best = {"threshold": float(threshold), "f1": float(f1), "trades_count": float(trades_count)}

    if best["f1"] < 0.0:
        LOGGER.warning(
            "No threshold cleared min_trades=%s; defaulting to 0.50 with zero F1",
            min_trades,
        )
        return {"threshold": 0.5, "f1": 0.0, "trades_count": 0.0}
    return best


KNOWN_REDUNDANT_FEATURES = {
    "close_vs_open_pct",
    "ctx_trend_strength",
    "atr_pct",
    "ctx_adx",
    "ctx_choppiness",
    "regime_quiet",
    "volatility_percentile_60",
}


def prune_redundant_features(df: pd.DataFrame, threshold: float = 0.98) -> pd.DataFrame:
    """Drop known redundant and highly collinear feature columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        return df.copy()

    pruned = df.copy()
    known_drops = [column for column in pruned.columns if column in KNOWN_REDUNDANT_FEATURES]
    if known_drops:
        pruned = pruned.drop(columns=known_drops)

    numeric_columns = [column for column in pruned.columns if pd.api.types.is_numeric_dtype(pruned[column])]
    if len(numeric_columns) < 2:
        return pruned

    corr = pruned[numeric_columns].corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    dynamic_drops: List[str] = []
    for left in upper.columns:
        for right, value in upper[left].dropna().items():
            if float(value) < float(threshold):
                continue
            keep, drop = sorted((str(left), str(right)))
            if drop not in dynamic_drops:
                dynamic_drops.append(drop)
                LOGGER.info(
                    "Dropping highly correlated feature %s (paired with %s, corr=%.6f)",
                    drop,
                    keep,
                    float(value),
                )

    if dynamic_drops:
        pruned = pruned.drop(columns=[column for column in dynamic_drops if column in pruned.columns])
    return pruned


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float], *, threshold: float = 0.5) -> Dict[str, float]:
    y_true_list = [1 if int(v) else 0 for v in y_true]
    y_prob_list = [float(v) for v in y_prob]
    y_pred = [1 if prob >= float(threshold) else 0 for prob in y_prob_list]
    if not y_true_list:
        return {
            "roc_auc": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    tp = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / len(y_true_list)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    roc_auc = 0.0
    if len(set(y_true_list)) > 1:
        pairs = sorted(zip(y_prob_list, y_true_list), key=lambda item: item[0])
        pos = sum(y_true_list)
        neg = len(y_true_list) - pos
        if pos and neg:
            rank_sum = 0.0
            for rank, (_, label) in enumerate(pairs, start=1):
                if label == 1:
                    rank_sum += rank
            roc_auc = (rank_sum - (pos * (pos + 1)) / 2.0) / (pos * neg)

    return {
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def aggregate_trade_metrics(
    pnls: Sequence[float],
    trade_days: Iterable[date],
) -> Dict[str, float]:
    pnl_list = [float(v) for v in pnls]
    wins = [p for p in pnl_list if p > 0.0]
    losses = [abs(p) for p in pnl_list if p < 0.0]
    profit_factor = sum(wins) / sum(losses) if losses else (99.0 if wins else 0.0)
    expectancy = float(np.mean(pnl_list)) if pnl_list else 0.0
    win_rate = len(wins) / len(pnl_list) if pnl_list else 0.0

    by_day: Dict[date, float] = defaultdict(float)
    for day, pnl in zip(trade_days, pnl_list):
        by_day[day] += pnl
    daily_returns = list(by_day.values())

    sharpe = 0.0
    sortino = 0.0
    if len(daily_returns) > 1:
        avg_ret = float(np.mean(daily_returns))
        std_ret = float(np.std(daily_returns, ddof=1))
        if std_ret > 0.0:
            sharpe = (avg_ret / std_ret) * math.sqrt(252.0)
        downside = [ret for ret in daily_returns if ret < 0.0]
        if downside:
            downside_std = float(np.std(downside, ddof=1))
            if downside_std > 0.0:
                sortino = (avg_ret / downside_std) * math.sqrt(252.0)
        elif avg_ret > 0.0:
            sortino = 99.0

    peak = 0.0
    equity = 0.0
    max_drawdown = 0.0
    for pnl in pnl_list:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return {
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "win_rate": float(win_rate),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "drawdown": float(max_drawdown),
        "trades_count": float(len(pnl_list)),
    }
