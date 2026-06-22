"""Tests for Shadow Mode Analytics panel in UI.

Tests the data-logic portion — pure computation without UI dependencies.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Pure computation helpers (simulate what the UI panel computes from rows)
# ---------------------------------------------------------------------------


def _shadow_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = len(rows)
    long_signals = sum(1 for r in rows if int(r.get("predicted_class", -1)) == 1)
    short_signals = sum(1 for r in rows if int(r.get("predicted_class", -1)) == 0)
    return {"total": total, "long": long_signals, "short": short_signals}


def _shadow_win_rate(rows: list[dict[str, Any]]) -> float:
    wins = sum(1 for r in rows if float(r.get("outcome_pnl", 0.0)) > 0.0)
    total = len(rows)
    if total == 0:
        return 0.0
    return wins / total


def _shadow_precision_recall(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(
        1
        for r in rows
        if int(r.get("predicted_class", -1)) == 1 and float(r.get("outcome_pnl", 0.0)) > 0.0
    )
    fp = sum(
        1
        for r in rows
        if int(r.get("predicted_class", -1)) == 1 and float(r.get("outcome_pnl", 0.0)) <= 0.0
    )
    fn = sum(
        1
        for r in rows
        if int(r.get("predicted_class", -1)) == 0 and float(r.get("outcome_pnl", 0.0)) > 0.0
    )
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall}


def _shadow_profit_factor(rows: list[dict[str, Any]]) -> float:
    gross_profit = sum(
        max(0.0, float(r.get("outcome_pnl", 0.0))) for r in rows
    )
    gross_loss = abs(
        sum(min(0.0, float(r.get("outcome_pnl", 0.0))) for r in rows)
    )
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _shadow_sharpe_ratio(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    pnls = [float(r.get("outcome_pnl", 0.0)) for r in rows]
    mean_pnl = sum(pnls) / len(pnls)
    variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
    std_pnl = math.sqrt(variance) if variance > 0 else 0.0
    if std_pnl == 0.0:
        return 0.0
    return (mean_pnl / std_pnl) * math.sqrt(252)


def _shadow_max_drawdown(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rows:
        equity += float(r.get("outcome_pnl", 0.0))
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _shadow_daily_pnl(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate PnL by calendar date (YYYY-MM-DD)."""
    daily: dict[str, float] = {}
    for r in rows:
        ts_raw = r.get("timestamp")
        if ts_raw is None:
            continue
        try:
            if isinstance(ts_raw, str):
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            elif isinstance(ts_raw, (int, float)):
                dt = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            else:
                continue
            day_key = dt.strftime("%Y-%m-%d")
        except Exception:
            day_key = "unknown"
        daily[day_key] = daily.get(day_key, 0.0) + float(r.get("outcome_pnl", 0.0))
    return daily


def _shadow_weekly_pnl(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate PnL by ISO week (YYYY-Www)."""
    weekly: dict[str, float] = {}
    for r in rows:
        ts_raw = r.get("timestamp")
        if ts_raw is None:
            continue
        try:
            if isinstance(ts_raw, str):
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            elif isinstance(ts_raw, (int, float)):
                dt = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            else:
                continue
            week_key = dt.strftime("%Y-W%W")
        except Exception:
            week_key = "unknown"
        weekly[week_key] = weekly.get(week_key, 0.0) + float(r.get("outcome_pnl", 0.0))
    return weekly


def _anomaly_highlight(win_rate: float, sharpe: float) -> bool:
    """Flag anomaly when win_rate < 0.4 or sharpe < 0.5."""
    return win_rate < 0.4 or sharpe < 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prediction_rows() -> list[dict[str, Any]]:
    """50 synthetic shadow-mode prediction rows with outcomes."""
    import random

    random.seed(42)
    rows = []
    base_ts = datetime(2026, 6, 1, 9, 15, tzinfo=timezone.utc)
    for i in range(50):
        predicted_class = 1 if i % 2 == 0 else 0
        # ~60% winners, 40% losers
        outcome_pnl = round(random.uniform(-150.0, 200.0) if random.random() > 0.4 else random.uniform(10.0, 200.0), 2)
        ts = base_ts.timestamp() + i * 900  # 15-min bars
        rows.append(
            {
                "timestamp": ts,
                "predicted_class": predicted_class,
                "probability": round(0.5 + random.random() * 0.45, 3),
                "trade_taken": random.random() > 0.3,
                "outcome_pnl": outcome_pnl,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shadow_metrics_computed_from_prediction_rows(synthetic_prediction_rows) -> None:
    counts = _shadow_counts(synthetic_prediction_rows)
    assert counts["total"] == 50
    assert counts["long"] == 25  # even indices
    assert counts["short"] == 25  # odd indices


def test_shadow_win_rate_calculation(synthetic_prediction_rows) -> None:
    wr = _shadow_win_rate(synthetic_prediction_rows)
    assert 0.0 <= wr <= 1.0
    # With 50 random rows, win_rate should be somewhere in reasonable range
    # Re-compute manually to cross-check
    wins = sum(1 for r in synthetic_prediction_rows if r["outcome_pnl"] > 0)
    expected = wins / 50
    assert abs(wr - expected) < 1e-9


def test_shadow_precision_recall_calculation() -> None:
    # 5 predictions: 3 buy (1,2,3) and 2 sell (4,5)
    # 1=WIN, 2=LOSS, 3=WIN, 4=LOSS, 5=WIN
    rows = [
        {"predicted_class": 1, "outcome_pnl": 100.0},   # TP
        {"predicted_class": 1, "outcome_pnl": -50.0},  # FP
        {"predicted_class": 1, "outcome_pnl": 80.0},   # TP
        {"predicted_class": 0, "outcome_pnl": -30.0},  # FN (missed win on sell)
        {"predicted_class": 0, "outcome_pnl": 60.0},   # TN
    ]
    result = _shadow_precision_recall(rows)
    # TP=2, FP=1, FN=1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)


def test_shadow_profit_factor() -> None:
    rows = [
        {"outcome_pnl": 100.0},
        {"outcome_pnl": 200.0},
        {"outcome_pnl": -50.0},
        {"outcome_pnl": -30.0},
    ]
    pf = _shadow_profit_factor(rows)
    # gross_profit = 300, gross_loss = 80
    assert pf == pytest.approx(300.0 / 80.0)

    # All winners -> inf
    all_winners = [{"outcome_pnl": 10.0}, {"outcome_pnl": 20.0}]
    assert _shadow_profit_factor(all_winners) == float("inf")

    # All losers -> 0
    all_losers = [{"outcome_pnl": -10.0}, {"outcome_pnl": -20.0}]
    assert _shadow_profit_factor(all_losers) == 0.0


def test_shadow_sharpe_ratio() -> None:
    # Zero PnL rows
    zero_rows = [{"outcome_pnl": 0.0} for _ in range(10)]
    sharpe = _shadow_sharpe_ratio(zero_rows)
    assert sharpe == 0.0

    # Identical positive returns -> should be high
    constant_rows = [{"outcome_pnl": 10.0} for _ in range(20)]
    sharpe = _shadow_sharpe_ratio(constant_rows)
    # std=0 -> Sharpe = 0
    assert sharpe == 0.0

    # Mix of positive and negative
    mixed = [
        {"outcome_pnl": 100.0},
        {"outcome_pnl": -50.0},
        {"outcome_pnl": 80.0},
        {"outcome_pnl": -30.0},
    ]
    sharpe = _shadow_sharpe_ratio(mixed)
    # mean = 25, deviations: [75, -75, 55, -55], variance = (5625+5625+3025+3025)/4 = 4312.5
    # std ≈ 65.66, Sharpe = (25/65.66)*sqrt(252) ≈ 6.03
    assert 5.5 < sharpe < 6.5, f"Expected ~6.0, got {sharpe}"

    # Single row
    single = [{"outcome_pnl": 50.0}]
    assert _shadow_sharpe_ratio(single) == 0.0


def test_shadow_max_drawdown() -> None:
    # Equity curve: 0 -> 100 -> 50 -> 200 -> 120 -> 250
    rows = [
        {"outcome_pnl": 100.0},
        {"outcome_pnl": -50.0},
        {"outcome_pnl": 150.0},
        {"outcome_pnl": -80.0},
        {"outcome_pnl": 130.0},
    ]
    mdd = _shadow_max_drawdown(rows)
    # peak=100 at row 0, low=50 after row 1, dd=50
    # then peak=200 at row 2, low=120 after row 3, dd=80
    # then peak=250 after row 4
    assert mdd == 80.0


def test_shadow_daily_pnl_aggregation() -> None:
    base_ts = datetime(2026, 6, 1, 9, 15, tzinfo=timezone.utc).timestamp()
    rows = [
        # Day 1: two rows
        {"timestamp": base_ts, "outcome_pnl": 100.0},
        {"timestamp": base_ts + 900, "outcome_pnl": 50.0},
        # Day 2: one row
        {"timestamp": base_ts + 86400, "outcome_pnl": -30.0},
    ]
    daily = _shadow_daily_pnl(rows)
    assert "2026-06-01" in daily
    assert daily["2026-06-01"] == pytest.approx(150.0)
    assert "2026-06-02" in daily
    assert daily["2026-06-02"] == pytest.approx(-30.0)


def test_shadow_weekly_pnl_aggregation() -> None:
    # June 1 2026 is a Monday = week 23
    base_ts = datetime(2026, 6, 1, 9, 15, tzinfo=timezone.utc).timestamp()
    rows = [
        {"timestamp": base_ts, "outcome_pnl": 100.0},
        {"timestamp": base_ts + 900, "outcome_pnl": 50.0},
        {"timestamp": base_ts + 86400 * 3, "outcome_pnl": -30.0},  # Thursday
    ]
    weekly = _shadow_weekly_pnl(rows)
    # All same ISO week
    assert len(weekly) == 1
    week_key = list(weekly.keys())[0]
    assert week_key.startswith("2026-W")
    assert abs(weekly[week_key] - 120.0) < 1e-9


def test_anomaly_highlight_thresholds() -> None:
    # Normal: win_rate >= 0.4 and sharpe >= 0.5
    assert _anomaly_highlight(win_rate=0.5, sharpe=1.0) is False
    assert _anomaly_highlight(win_rate=0.4, sharpe=0.5) is False
    # Anomaly: win_rate < 0.4
    assert _anomaly_highlight(win_rate=0.39, sharpe=1.0) is True
    # Anomaly: sharpe < 0.5
    assert _anomaly_highlight(win_rate=0.6, sharpe=0.49) is True
    # Anomaly: both
    assert _anomaly_highlight(win_rate=0.3, sharpe=0.2) is True


def test_empty_rows_handled_gracefully() -> None:
    empty: list[dict[str, Any]] = []
    assert _shadow_counts(empty) == {"total": 0, "long": 0, "short": 0}
    assert _shadow_win_rate(empty) == 0.0
    assert _shadow_precision_recall(empty) == {"precision": 0.0, "recall": 0.0}
    assert _shadow_profit_factor(empty) == 0.0
    assert _shadow_sharpe_ratio(empty) == 0.0
    assert _shadow_max_drawdown(empty) == 0.0
    assert _shadow_daily_pnl(empty) == {}
    assert _shadow_weekly_pnl(empty) == {}


def test_shadow_refresh_idempotent() -> None:
    rows = [
        {"timestamp": 1200.0, "predicted_class": 1, "outcome_pnl": 100.0},
        {"timestamp": 1300.0, "predicted_class": 0, "outcome_pnl": -40.0},
        {"timestamp": 1400.0, "predicted_class": 1, "outcome_pnl": 80.0},
    ]
    # Run computation twice
    counts1 = _shadow_counts(rows)
    counts2 = _shadow_counts(rows)
    assert counts1 == counts2

    wr1 = _shadow_win_rate(rows)
    wr2 = _shadow_win_rate(rows)
    assert wr1 == wr2

    pf1 = _shadow_profit_factor(rows)
    pf2 = _shadow_profit_factor(rows)
    assert pf1 == pf2

    mdd1 = _shadow_max_drawdown(rows)
    mdd2 = _shadow_max_drawdown(rows)
    assert mdd1 == mdd2