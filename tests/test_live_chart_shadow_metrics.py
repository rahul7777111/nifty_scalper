"""Tests for shadow metrics computation — no broker needed."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_chart_snapshot import compute_shadow_metrics


class TestShadowMetricsFromEmptyRows:
    def test_empty_prediction_rows(self) -> None:
        m = compute_shadow_metrics([], [])
        assert m.predictions_today == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0

    def test_none_prediction_rows(self) -> None:
        m = compute_shadow_metrics(None, None)
        assert m.predictions_today == 0


class TestShadowMetricsTodaysCount:
    def test_only_todays_rows_counted(self) -> None:
        now = datetime.now()
        today_ts = now.timestamp()
        yesterday_ts = (now - timedelta(days=1)).timestamp()
        rows = [
            {"ts": today_ts, "probability": 0.6, "trade_taken": True},
            {"ts": yesterday_ts, "probability": 0.6, "trade_taken": True},
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.predictions_today == 1


class TestShadowMetricsAcceptedVsBlocked:
    def test_trade_taken_true_is_accepted(self) -> None:
        now_ts = datetime.now().timestamp()
        rows = [
            {"ts": now_ts, "probability": 0.6, "trade_taken": True},
            {"ts": now_ts, "probability": 0.6, "trade_taken": True},
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.accepted_signals == 2
        assert m.blocked_signals == 0

    def test_trade_taken_false_is_blocked(self) -> None:
        now_ts = datetime.now().timestamp()
        rows = [
            {"ts": now_ts, "probability": 0.6, "trade_taken": False},
            {"ts": now_ts, "probability": 0.6, "trade_taken": False},
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.accepted_signals == 0
        assert m.blocked_signals == 2


class TestShadowMetricsWinRate:
    def test_win_rate_from_accepted_rows(self) -> None:
        now_ts = datetime.now().timestamp()
        rows = [
            {"ts": now_ts, "probability": 0.72, "trade_taken": True},  # win
            {"ts": now_ts, "probability": 0.35, "trade_taken": True},  # loss
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.win_rate == 50.0  # 1 out of 2 accepted = 50%

    def test_zero_accepted_gives_zero_winrate(self) -> None:
        now_ts = datetime.now().timestamp()
        rows = [
            {"ts": now_ts, "probability": 0.72, "trade_taken": False},
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.win_rate == 0.0


class TestShadowMetricsProfitFactor:
    def test_profit_factor_from_realized_pnl(self) -> None:
        trade_events = [
            {"realized": 200.0, "ts": datetime.now().timestamp(), "entry_ts": 0, "exit_ts": 0},
            {"realized": -100.0, "ts": datetime.now().timestamp(), "entry_ts": 0, "exit_ts": 0},
        ]
        m = compute_shadow_metrics([], trade_events)
        # gross_profit = 200, gross_loss = 100 → pf = 2.0
        assert m.profit_factor == 2.0

    def test_zero_gross_loss_avoids_division_error(self) -> None:
        trade_events = [{"realized": 100.0, "ts": datetime.now().timestamp()}]
        m = compute_shadow_metrics([], trade_events)
        # Only profit, no loss → pf = gross_profit (avoids div-by-zero)
        assert m.profit_factor == 100.0


class TestShadowMetricsMaxDrawdown:
    def test_max_drawdown_computed(self) -> None:
        # With 4 trades: +100, -130, +50, -20
        # Running: 100, -30, 20, 0 → peak=100 → max_drawdown=-130
        trade_events = [
            {"realized": 100.0, "ts": datetime.now().timestamp()},
            {"realized": -130.0, "ts": datetime.now().timestamp()},
            {"realized": 50.0, "ts": datetime.now().timestamp()},
            {"realized": -20.0, "ts": datetime.now().timestamp()},
        ]
        m = compute_shadow_metrics([], trade_events)
        assert m.max_drawdown < 0
        # Running: 100, -30, 20, 0 | peak=100 | max_drawdown = -130 (from 100 to -30)
        assert m.max_drawdown == pytest.approx(-130.0, rel=0.01)


class TestShadowMetricsDrift:
    def test_no_drift_with_low_variance(self) -> None:
        now_ts = datetime.now().timestamp()
        rows = [
            {"ts": now_ts + i * 60, "probability": 0.52, "trade_taken": True}
            for i in range(10)
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.drift_warning is False

    def test_drift_with_high_variance(self) -> None:
        now_ts = datetime.now().timestamp()
        probs = [0.9, 0.1, 0.85, 0.15, 0.88, 0.12, 0.9, 0.1, 0.87, 0.13]
        rows = [
            {"ts": now_ts + i * 60, "probability": probs[i], "trade_taken": True}
            for i in range(10)
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.drift_warning is True


class TestShadowMetricsLastPredictionTime:
    def test_last_prediction_time_from_rows(self) -> None:
        now_ts = datetime.now().timestamp()
        rows = [
            {"ts": now_ts - 300, "probability": 0.6, "trade_taken": False},
            {"ts": now_ts - 60, "probability": 0.6, "trade_taken": False},
        ]
        m = compute_shadow_metrics(rows, [])
        assert m.last_prediction_time is not None
        assert m.last_prediction_time.timestamp() == pytest.approx(now_ts - 60, rel=1.0)