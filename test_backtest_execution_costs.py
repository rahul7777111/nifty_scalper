from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from tools.backtest_today import BacktestStats, _print_report, run_backtest
from src.market_data import Candle


def test_print_report_labels_net_pnl_without_costs(capsys) -> None:
    _print_report(
        "sample",
        BacktestStats(
            trades=2,
            wins=1,
            losses=1,
            total_pnl_pts=12.5,
            total_win_pts=20.0,
            total_loss_pts=7.5,
            largest_win_pts=20.0,
            largest_loss_pts=7.5,
            max_consecutive_wins=1,
            max_consecutive_losses=1,
        ),
    )

    out = capsys.readouterr().out
    assert "net_pnl (index pts): 12.50" in out
    assert "expectancy=6.25 pts/trade" in out
    assert "profit_factor=2.67" in out
    assert "max_losses=1" in out
    assert "largest_loss=7.50" in out
    assert "gross_pnl" not in out
    assert "costs" not in out


def test_run_backtest_charges_round_trip_costs(monkeypatch) -> None:
    import tools.backtest_today as bt

    base = datetime(2026, 5, 18, 9, 15)
    candles = [
        Candle(time=base + timedelta(minutes=i), open=100 + i, high=101 + i, low=99 + i, close=100 + i, volume=1000)
        for i in range(40)
    ]

    monkeypatch.setattr(bt, "_warmup_needed", lambda cfg: 5)
    monkeypatch.setattr(bt, "_direction_signal", lambda window, cfg: bt.DirectionSignal("bull", 5, 0))

    cfg = SimpleNamespace(max_trades_per_day=1, cooldown_sec=0.0, max_open_positions=1, atr_period=14)
    stats = run_backtest(candles, cfg, horizon=3, slippage_pts=1.0, fees_pts=0.5)

    assert stats.entries == 1
    assert stats.trades == 1
    assert stats.gross_pnl_pts == 3.0
    assert stats.total_cost_pts == 2.5
    assert stats.total_pnl_pts == 0.5
    assert stats.total_win_pts == 0.5
    assert stats.total_loss_pts == 0.0
    assert stats.max_consecutive_wins == 1
    assert stats.max_consecutive_losses == 0
    assert stats.largest_win_pts == 0.5
    assert stats.largest_loss_pts == 0.0
