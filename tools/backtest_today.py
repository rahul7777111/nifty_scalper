from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional, Tuple

# Allow running as `python tools/backtest_today.py` without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_strategy_config
from indicators import adx, atr, choppiness_index, ema, roc, rsi, supertrend
from market_data import Candle
from candlestick_patterns import (
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_hammer,
    is_shooting_star,
)


def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _filter_candles_for_date(candles: List[Candle], d: date) -> List[Candle]:
    out = [c for c in candles if isinstance(c.time, datetime) and c.time.date() == d]
    out.sort(key=lambda c: c.time)
    return out


def _resample_candles(candles: List[Candle], *, target_minutes: int) -> List[Candle]:
    """Resample smaller-minute candles into larger-minute candles.

    This mirrors the aggregation logic in `MStockTypeBClient._resample_candles`, but is
    implemented here so the backtest tool can work with Yahoo 1m data.
    """

    try:
        m = int(target_minutes)
    except Exception:
        return candles
    if m <= 1:
        return candles
    if not candles:
        return []

    src = sorted(candles, key=lambda c: c.time)
    out: List[Candle] = []

    bucket_start: Optional[datetime] = None
    o: Optional[float] = None
    h: Optional[float] = None
    l: Optional[float] = None
    c: Optional[float] = None
    vol_sum: Optional[float] = None

    for row in src:
        t = row.time
        t0 = t.replace(second=0, microsecond=0)
        delta_min = int(t0.minute % m)
        b = t0.replace(second=0, microsecond=0) - __import__("datetime").timedelta(minutes=delta_min)

        if bucket_start is None or b != bucket_start:
            if bucket_start is not None and o is not None and h is not None and l is not None and c is not None:
                out.append(Candle(time=bucket_start, open=float(o), high=float(h), low=float(l), close=float(c), volume=vol_sum))

            bucket_start = b
            o = float(row.open)
            h = float(row.high)
            l = float(row.low)
            c = float(row.close)
            vol_sum = None
            if row.volume is not None:
                try:
                    vol_sum = float(row.volume)
                except Exception:
                    vol_sum = None
        else:
            if h is None or float(row.high) > float(h):
                h = float(row.high)
            if l is None or float(row.low) < float(l):
                l = float(row.low)
            c = float(row.close)
            if row.volume is not None:
                try:
                    v = float(row.volume)
                except Exception:
                    v = None
                if v is not None:
                    vol_sum = v if vol_sum is None else (vol_sum + v)

    if bucket_start is not None and o is not None and h is not None and l is not None and c is not None:
        out.append(Candle(time=bucket_start, open=float(o), high=float(h), low=float(l), close=float(c), volume=vol_sum))

    return out


def _warmup_needed(cfg) -> int:
    try:
        ema_f = int(getattr(cfg, "ema_fast", 9) or 9)
    except Exception:
        ema_f = 9
    try:
        ema_s = int(getattr(cfg, "ema_slow", 21) or 21)
    except Exception:
        ema_s = 21
    try:
        atr_p = int(getattr(cfg, "atr_period", 14) or 14)
    except Exception:
        atr_p = 14
    try:
        adx_p = int(getattr(cfg, "adx_period", 14) or 14)
    except Exception:
        adx_p = 14

    rsi_p = 14
    # ADX needs ~2*period to stabilize (see indicators.py)
    n = max(ema_s, ema_f, atr_p + 1, rsi_p + 1, 2 * adx_p)
    # Add a small safety margin
    return int(n + 5)


@dataclass(frozen=True)
class DirectionSignal:
    desired_dir: Optional[str]  # "bull" | "bear" | None
    bull_score: int
    bear_score: int


def _direction_signal(candles: List[Candle], cfg) -> DirectionSignal:
    if len(candles) < 25:
        return DirectionSignal(None, 0, 0)

    candles = sorted(candles, key=lambda c: c.time)
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    try:
        ema_f_p = int(getattr(cfg, "ema_fast", 9) or 9)
    except Exception:
        ema_f_p = 9
    try:
        ema_s_p = int(getattr(cfg, "ema_slow", 21) or 21)
    except Exception:
        ema_s_p = 21
    try:
        atr_p = int(getattr(cfg, "atr_period", 14) or 14)
    except Exception:
        atr_p = 14

    fast_ema = ema(closes, period=ema_f_p)
    slow_ema = ema(closes, period=ema_s_p)
    rsi_val = rsi(closes, period=14)
    atr_val = atr(highs, lows, closes, period=atr_p)
    if fast_ema is None or slow_ema is None or rsi_val is None or atr_val is None:
        return DirectionSignal(None, 0, 0)

    # Optional pre-filters (match the live strategy's structure, but keep it minimal).
    if bool(getattr(cfg, "enable_roc_filter", False)):
        r_val = roc(closes, period=int(getattr(cfg, "roc_period", 14) or 14))
        try:
            roc_min = float(getattr(cfg, "roc_min_threshold", 0.05) or 0.05)
        except Exception:
            roc_min = 0.05
        if r_val is not None and abs(float(r_val)) < float(roc_min):
            return DirectionSignal(None, 0, 0)

    if bool(getattr(cfg, "enable_choppiness_filter", False)):
        c_val = choppiness_index(highs, lows, closes, period=int(getattr(cfg, "choppiness_period", 14) or 14))
        try:
            chop_max = float(getattr(cfg, "choppiness_threshold", 61.8) or 61.8)
        except Exception:
            chop_max = 61.8
        if c_val is not None and float(c_val) > float(chop_max):
            return DirectionSignal(None, 0, 0)

    bullish_candle = is_bullish_engulfing(candles) or is_hammer(candles)
    bearish_candle = is_bearish_engulfing(candles) or is_shooting_star(candles)
    _indecision = is_doji(candles)

    # Directional entry quality knobs (same defaults as strategy).
    try:
        min_score = int(getattr(cfg, "dir_min_confirmations", 4) or 4)
    except Exception:
        min_score = 4
    if min_score < 1:
        min_score = 1
    try:
        min_diff = int(getattr(cfg, "dir_min_score_diff", 1) or 1)
    except Exception:
        min_diff = 1
    if min_diff < 0:
        min_diff = 0
    allow_tie = bool(getattr(cfg, "dir_allow_tie_break_entries", False))

    # ADX Filter
    if bool(getattr(cfg, "enable_adx_filter", True)):
        adx_p = int(getattr(cfg, "adx_period", 14) or 14)
        adx_val = adx(highs, lows, closes, period=adx_p)
        try:
            min_adx = float(getattr(cfg, "adx_min_strength", 25.0) or 25.0)
        except Exception:
            min_adx = 25.0
        if adx_val is not None and float(adx_val) < float(min_adx):
            return DirectionSignal(None, 0, 0)

    # Supertrend (vote-based)
    st_val = None
    st_dir = None
    if bool(getattr(cfg, "enable_supertrend_filter", True)):
        st_period = int(getattr(cfg, "supertrend_period", 10) or 10)
        st_mult = float(getattr(cfg, "supertrend_multiplier", 3.0) or 3.0)
        st_val = supertrend(highs, lows, closes, period=st_period, multiplier=st_mult)
        if st_val is not None:
            try:
                curr_price = float(closes[-1])
                st_dir = "UP" if curr_price > float(st_val) else "DOWN"
            except Exception:
                st_dir = None

    # Momentum filter scaling
    try:
        mom_mult = float(getattr(cfg, "dir_momentum_atr_mult", 0.10) or 0.10)
    except Exception:
        mom_mult = 0.10
    if mom_mult < 0:
        mom_mult = 0.0

    # RSI scoring params
    try:
        rsi_buy = float(getattr(cfg, "auto_dir_rsi_buy", 52.0) or 52.0)
    except Exception:
        rsi_buy = 52.0
    try:
        rsi_sell = float(getattr(cfg, "auto_dir_rsi_sell", 48.0) or 48.0)
    except Exception:
        rsi_sell = 48.0
    try:
        rsi_slop = float(getattr(cfg, "auto_dir_rsi_slop", 1.0) or 1.0)
    except Exception:
        rsi_slop = 1.0

    bull_score = 0
    bear_score = 0

    ema_up = bool(fast_ema > slow_ema)
    ema_down = bool(fast_ema < slow_ema)

    # EMA slope vote
    try:
        slope_mult = float(getattr(cfg, "dir_ema_slope_atr_mult", 0.0) or 0.0)
    except Exception:
        slope_mult = 0.0
    ema_slope = None
    if slope_mult > 0 and len(closes) > ema_f_p:
        try:
            ema_prev = ema(closes[:-1], period=ema_f_p)
            if ema_prev is not None:
                ema_slope = float(fast_ema) - float(ema_prev)
        except Exception:
            ema_slope = None
    if ema_slope is not None and float(atr_val) > 0 and slope_mult > 0:
        slope_thr = float(atr_val) * float(slope_mult)
        if float(ema_slope) > float(slope_thr):
            bull_score += 1
        elif float(ema_slope) < -float(slope_thr):
            bear_score += 1

    # 1) Trend direction
    if ema_up:
        bull_score += 1
    elif ema_down:
        bear_score += 1

    # 2) Candle pattern confirmation
    if bullish_candle:
        bull_score += 1
    if bearish_candle:
        bear_score += 1

    # 3) Spot move over last few minutes (filter noise with ATR if available)
    delta_spot = None
    try:
        lookback = 6
        if len(closes) >= lookback:
            delta_spot = float(closes[-1]) - float(closes[-lookback])
            thr = float(mom_mult) * float(atr_val) if atr_val is not None else 0.0
            if float(delta_spot) >= float(thr):
                bull_score += 1
            elif float(delta_spot) <= -float(thr):
                bear_score += 1
    except Exception:
        delta_spot = None

    # 4) RSI confirmation
    r = float(rsi_val)
    rsi_buy_line = float(rsi_buy) - float(rsi_slop)
    rsi_sell_line = float(rsi_sell) + float(rsi_slop)
    if r >= float(rsi_buy):
        bull_score += 2
    elif r >= float(rsi_buy_line):
        bull_score += 1
    elif r <= float(rsi_sell):
        bear_score += 2
    elif r <= float(rsi_sell_line):
        bear_score += 1
    else:
        midline_threshold = 2.0
        if r > (50.0 + midline_threshold):
            bull_score += 1
        elif r < (50.0 - midline_threshold):
            bear_score += 1

    # Supertrend vote (only when scores are close)
    if st_val is not None and st_dir is not None:
        try:
            score_diff = abs(int(bull_score) - int(bear_score))
        except Exception:
            score_diff = 999
        if score_diff <= 1:
            st_mode = str(getattr(cfg, "supertrend_mode", "counter") or "counter").strip().lower()
            st_bull = False
            st_bear = False
            if st_dir == "UP":
                if st_mode == "trend":
                    st_bull = True
                else:
                    st_bear = True
            elif st_dir == "DOWN":
                if st_mode == "trend":
                    st_bear = True
                else:
                    st_bull = True
            try:
                st_w = int(getattr(cfg, "supertrend_vote_weight", 1) or 1)
            except Exception:
                st_w = 1
            if st_w < 1:
                st_w = 1
            if st_bull:
                bull_score += int(st_w)
            if st_bear:
                bear_score += int(st_w)

    # Signal decision (copied from strategy logic)
    bearish_signal = int(bear_score) >= int(min_score) and int(bear_score) > int(bull_score)
    bearish_ok = bool(ema_down) or bool(int(bear_score) >= (int(bull_score) + 2))

    bullish_signal = int(bull_score) >= int(min_score) and int(bull_score) > int(bear_score)
    bullish_ok = bool(ema_up) or bool(int(bull_score) >= (int(bear_score) + 2))

    if bull_score == bear_score and bull_score >= int(min_score):
        if not bool(allow_tie):
            return DirectionSignal(None, int(bull_score), int(bear_score))
        if delta_spot is not None and abs(float(delta_spot)) > 0.01:
            bullish_signal = float(delta_spot) > 0
            bearish_signal = float(delta_spot) < 0
            bullish_ok = bullish_signal
            bearish_ok = bearish_signal
        elif ema_up or ema_down:
            bullish_signal = bool(ema_up)
            bearish_signal = bool(ema_down)
            bullish_ok = bullish_signal
            bearish_ok = bearish_signal
        else:
            bullish_signal = float(r) >= 50.0
            bearish_signal = not bullish_signal
            bullish_ok = bullish_signal
            bearish_ok = bearish_signal

    desired_dir = None
    if bearish_signal and bearish_ok:
        desired_dir = "bear"
    elif bullish_signal and bullish_ok:
        desired_dir = "bull"

    if desired_dir is not None:
        try:
            if abs(int(bull_score) - int(bear_score)) < int(min_diff):
                desired_dir = None
        except Exception:
            pass

    return DirectionSignal(desired_dir, int(bull_score), int(bear_score))


@dataclass
class BacktestStats:
    entries: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl_pts: float = 0.0
    total_cost_pts: float = 0.0
    total_pnl_pts: float = 0.0
    total_win_pts: float = 0.0
    total_loss_pts: float = 0.0
    largest_win_pts: float = 0.0
    largest_loss_pts: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    max_drawdown_pts: float = 0.0


def _max_drawdown(equity_curve: Iterable[float]) -> float:
    peak = None
    max_dd = 0.0
    for x in equity_curve:
        v = float(x)
        if peak is None or v > peak:
            peak = v
        if peak is not None:
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


def run_backtest(
    candles: List[Candle],
    cfg,
    *,
    horizon: int,
    slippage_pts: float = 0.0,
    fees_pts: float = 0.0,
    debug: bool = False,
) -> BacktestStats:
    candles = sorted(candles, key=lambda c: c.time)
    if horizon < 1:
        horizon = 1

    try:
        round_trip_cost_pts = max(0.0, (2.0 * float(slippage_pts)) + float(fees_pts))
    except Exception:
        round_trip_cost_pts = 0.0

    warmup = _warmup_needed(cfg)
    if len(candles) <= warmup + horizon:
        return BacktestStats()

    equity = 0.0
    equity_curve: List[float] = [0.0]
    stats = BacktestStats()

    try:
        max_trades = int(getattr(cfg, "max_trades_per_day", 0) or 0)
    except Exception:
        max_trades = 0
    if max_trades <= 0:
        max_trades = 10**9

    try:
        cooldown_sec = float(getattr(cfg, "cooldown_sec", 0.0) or 0.0)
    except Exception:
        cooldown_sec = 0.0
    if cooldown_sec < 0:
        cooldown_sec = 0.0

    try:
        max_open = int(getattr(cfg, "max_open_positions", 1) or 1)
    except Exception:
        max_open = 1
    if max_open < 1:
        max_open = 1

    last_entry_ts = 0.0
    last_exit_ts = 0.0
    consec_wins = 0
    consec_losses = 0
    open_positions: List[dict] = []  # {"dir": "bull"|"bear", "entry": float, "exit_i": int}

    def _current_atr(i: int) -> Optional[float]:
        try:
            atr_p = int(getattr(cfg, "atr_period", 14) or 14)
        except Exception:
            atr_p = 14
        if atr_p < 1:
            atr_p = 14
        window = candles[: i + 1]
        if len(window) < (atr_p + 2):
            return None
        highs = [c.high for c in window]
        lows = [c.low for c in window]
        closes = [c.close for c in window]
        try:
            v = atr(highs, lows, closes, period=int(atr_p))
        except Exception:
            v = None
        return float(v) if v is not None else None

    def _ts(i: int) -> float:
        try:
            return float(candles[i].time.timestamp())
        except Exception:
            return float(i)

    dbg_printed = 0
    for i in range(warmup, len(candles)):
        now_ts = _ts(i)

        # Close due positions at this candle.
        if open_positions:
            remaining: List[dict] = []
            for pos in open_positions:
                try:
                    exit_i = int(pos.get("exit_i"))
                except Exception:
                    exit_i = i
                if exit_i > i:
                    remaining.append(pos)
                    continue

                d = str(pos.get("dir") or "")
                entry = float(pos.get("entry") or 0.0)
                exit_ = float(candles[i].close)
                gross_pnl = (exit_ - entry) if d == "bull" else (entry - exit_)
                pnl = float(gross_pnl) - float(round_trip_cost_pts)

                stats.trades += 1
                if pnl > 0:
                    stats.wins += 1
                    stats.total_win_pts += float(pnl)
                    if float(pnl) > float(stats.largest_win_pts):
                        stats.largest_win_pts = float(pnl)
                    consec_wins += 1
                    consec_losses = 0
                    if consec_wins > int(stats.max_consecutive_wins):
                        stats.max_consecutive_wins = int(consec_wins)
                elif pnl < 0:
                    stats.losses += 1
                    stats.total_loss_pts += abs(float(pnl))
                    if abs(float(pnl)) > float(stats.largest_loss_pts):
                        stats.largest_loss_pts = abs(float(pnl))
                    consec_losses += 1
                    consec_wins = 0
                    if consec_losses > int(stats.max_consecutive_losses):
                        stats.max_consecutive_losses = int(consec_losses)
                else:
                    consec_wins = 0
                    consec_losses = 0

                equity += float(pnl)
                stats.gross_pnl_pts += float(gross_pnl)
                stats.total_cost_pts += float(round_trip_cost_pts)
                equity_curve.append(equity)
                last_exit_ts = float(now_ts)

            open_positions = remaining

        # Don't open new trades if we can't safely exit within data.
        if i + horizon >= len(candles):
            if bool(debug) and dbg_printed < 5:
                dbg_printed += 1
                print(f"[DEBUG] i={i} skip=insufficient_future")
            continue

        if int(stats.entries) >= int(max_trades):
            if bool(debug) and dbg_printed < 5:
                dbg_printed += 1
                print(f"[DEBUG] i={i} skip=max_trades entries={int(stats.entries)} max={int(max_trades)}")
            continue

        if len(open_positions) >= int(max_open):
            if bool(debug) and dbg_printed < 5:
                dbg_printed += 1
                print(f"[DEBUG] i={i} skip=max_open open={len(open_positions)} max={int(max_open)}")
            continue

        last_action_ts = max(float(last_entry_ts), float(last_exit_ts))
        if last_action_ts > 0 and (float(now_ts) - float(last_action_ts)) < float(cooldown_sec):
            if bool(debug) and dbg_printed < 5:
                dbg_printed += 1
                dt = float(now_ts) - float(last_action_ts)
                print(f"[DEBUG] i={i} skip=cooldown dt={dt:.2f} cd={float(cooldown_sec):.2f}")
            continue

        # ATR-based entry guardrails (match key live strategy blocks).
        atr_val = _current_atr(i)
        if atr_val is not None:
            try:
                min_atr = float(getattr(cfg, "min_atr", 0.0) or 0.0)
            except Exception:
                min_atr = 0.0
            try:
                max_atr = float(getattr(cfg, "max_atr", 1e9) or 1e9)
            except Exception:
                max_atr = 1e9
            if float(atr_val) < float(min_atr) or float(atr_val) > float(max_atr):
                if bool(debug) and dbg_printed < 5:
                    dbg_printed += 1
                    print(
                        f"[DEBUG] i={i} skip=atr_gate atr={float(atr_val):.4f} min={float(min_atr):.4f} max={float(max_atr):.4f}"
                    )
                continue


        window = candles[: i + 1]
        sig = _direction_signal(window, cfg)
        if sig.desired_dir not in {"bull", "bear"}:
            if bool(debug) and dbg_printed < 5:
                dbg_printed += 1
                print(
                    f"[DEBUG] i={i} skip=no_signal bull={int(sig.bull_score)} bear={int(sig.bear_score)} desired={sig.desired_dir} atr={atr_val}"
                )
            continue

        if bool(debug) and dbg_printed < 5:
            dbg_printed += 1
            try:
                t = candles[i].time
            except Exception:
                t = None
            print(
                f"[DEBUG] i={i} t={t} dir={sig.desired_dir} bull={sig.bull_score} bear={sig.bear_score} atr={atr_val}"
            )

        open_positions.append(
            {
                "dir": sig.desired_dir,
                "entry": float(candles[i].close),
                "exit_i": int(i + horizon),
            }
        )
        last_entry_ts = float(now_ts)
        stats.entries += 1

    stats.total_pnl_pts = float(equity)
    stats.max_drawdown_pts = _max_drawdown(equity_curve)
    return stats


def _print_report(label: str, stats: BacktestStats) -> None:
    trades = int(stats.trades)
    win_rate = (float(stats.wins) / float(trades) * 100.0) if trades else 0.0
    avg_win = (float(stats.total_win_pts) / float(stats.wins)) if int(stats.wins) > 0 else 0.0
    avg_loss = (float(stats.total_loss_pts) / float(stats.losses)) if int(stats.losses) > 0 else 0.0
    expectancy = (float(stats.total_pnl_pts) / float(trades)) if trades else 0.0
    if float(stats.total_loss_pts) > 0:
        profit_factor = float(stats.total_win_pts) / float(stats.total_loss_pts)
    elif float(stats.total_win_pts) > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    payoff = (float(avg_win) / float(avg_loss)) if float(avg_loss) > 0 else (float("inf") if float(avg_win) > 0 else 0.0)

    def _fmt(v: float) -> str:
        try:
            if math.isinf(float(v)):
                return "inf"
        except Exception:
            pass
        return f"{float(v):.2f}"

    print(f"\n=== {label} ===")
    print(
        f"entries: {int(stats.entries)} | closed: {trades} | wins: {int(stats.wins)} | losses: {int(stats.losses)} | win_rate: {win_rate:.1f}%"
    )
    if float(stats.total_cost_pts) > 0:
        print(f"gross_pnl (index pts): {float(stats.gross_pnl_pts):.2f}")
        print(f"costs (index pts): {float(stats.total_cost_pts):.2f}")
    print(f"net_pnl (index pts): {float(stats.total_pnl_pts):.2f}")
    print(f"max_drawdown (pts): {float(stats.max_drawdown_pts):.2f}")
    print(
        "quality: "
        f"expectancy={float(expectancy):.2f} pts/trade | "
        f"profit_factor={_fmt(profit_factor)} | "
        f"avg_win={float(avg_win):.2f} | avg_loss={float(avg_loss):.2f} | payoff={_fmt(payoff)}"
    )
    print(
        "streaks: "
        f"max_wins={int(stats.max_consecutive_wins)} | "
        f"max_losses={int(stats.max_consecutive_losses)} | "
        f"largest_win={float(stats.largest_win_pts):.2f} | "
        f"largest_loss={float(stats.largest_loss_pts):.2f}"
    )


def _load_yahoo_candles(underlying: str, timeframe: str, *, limit: int) -> List[Candle]:
    from yahoo_data import fetch_yahoo_candles, yahoo_symbol_for_underlying

    sym = yahoo_symbol_for_underlying(underlying)
    tf = str(timeframe or "").strip().lower()
    # Yahoo does not reliably support some intraday intervals for NSE indices (notably 3m,
    # and sometimes 5m "today" filtering behaves oddly). Fetch 1m and resample locally.
    if tf in {"3m", "5m"}:
        target_minutes = 3 if tf == "3m" else 5
        src_limit = int(limit) * int(target_minutes)
        if src_limit < 0:
            src_limit = 0
        if src_limit > 5000:
            src_limit = 5000
        base = fetch_yahoo_candles(sym, interval="1m", limit=int(src_limit))
        resampled = _resample_candles(base, target_minutes=int(target_minutes))
        return resampled[-int(limit) :] if limit else resampled

    return fetch_yahoo_candles(sym, interval=str(timeframe), limit=int(limit))


def _apply_backtest_preset(cfg, preset: str) -> None:
    p = str(preset or "").strip().lower()
    if not p:
        return
    # In the live bot, presets intentionally do not override explicit env vars.
    # For backtests, we *do* want the preset knobs to apply deterministically.
    if p in {"cons", "conservative", "safe"}:
        cfg.preset = "conservative"
        cfg.timeframe = "5m"
        cfg.max_trades_per_day = 6
        cfg.max_open_positions = 6
        cfg.cooldown_sec = 60.0
        cfg.cooldown_after_stopout_sec = 300.0
        cfg.max_pyramid_levels = 0
        cfg.entry_require_bid_ask = True
        cfg.entry_max_bid_ask_spread_pct = min(cfg.entry_max_bid_ask_spread_pct or 0.6, 0.4)
        cfg.min_atr = max(cfg.min_atr, 4.0)
        cfg.max_atr = min(cfg.max_atr, 55.0)

        # ADX filter
        cfg.enable_adx_filter = False
        cfg.adx_min_strength = 10.0

        # Directional scoring (tuned for today)
        cfg.dir_min_confirmations = 5
        cfg.dir_min_score_diff = 2
        cfg.auto_dir_rsi_buy = 52.0
        cfg.auto_dir_rsi_sell = 44.0
        cfg.auto_dir_rsi_slop = 1.5
        cfg.dir_momentum_atr_mult = 0.12
        cfg.dir_ema_slope_atr_mult = 0.03
    elif p in {"agg", "aggressive", "fast"}:
        cfg.preset = "aggressive"
        cfg.timeframe = "1m"
        cfg.max_trades_per_day = 10
        cfg.max_open_positions = 6
        cfg.cooldown_sec = 45.0
        cfg.cooldown_after_stopout_sec = 120.0
        cfg.max_pyramid_levels = max(cfg.max_pyramid_levels, 1)
        cfg.min_atr = max(cfg.min_atr, 0.5)
        cfg.max_atr = min(cfg.max_atr, 130.0)

        # ADX filter
        cfg.enable_adx_filter = False
        cfg.adx_min_strength = 25.0

        # Directional scoring (baseline aggressive)
        cfg.dir_min_confirmations = 5
        cfg.dir_min_score_diff = 0
        cfg.auto_dir_rsi_buy = 51.0
        cfg.auto_dir_rsi_sell = 49.0
        cfg.auto_dir_rsi_slop = 2.0
        cfg.dir_momentum_atr_mult = 0.05
        cfg.dir_ema_slope_atr_mult = 0.10


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay/backtest today's chart using directional signal scoring.")
    ap.add_argument("--source", choices=["yahoo"], default="yahoo", help="Candle source (default: yahoo).")
    ap.add_argument("--underlying", default=os.getenv("MSTOCK_UNDERLYING", "NIFTY"), help="Underlying (e.g. NIFTY).")
    ap.add_argument("--timeframe", default=os.getenv("MSTOCK_TIMEFRAME", "1m"), help="Timeframe (e.g. 1m, 5m).")
    ap.add_argument("--date", default=None, help="Date to backtest (YYYY-MM-DD). Default: today.")
    ap.add_argument("--limit", type=int, default=1200, help="Max candles to fetch.")
    ap.add_argument("--horizon", type=int, default=3, help="Exit horizon in candles (default: 3).")
    ap.add_argument(
        "--slippage-pts",
        type=float,
        default=0.0,
        help="Assumed one-way slippage in index points; charged on entry and exit.",
    )
    ap.add_argument(
        "--fees-pts",
        type=float,
        default=0.0,
        help="Assumed round-trip fees/taxes/brokerage converted to index points.",
    )
    ap.add_argument(
        "--preset",
        default=None,
        choices=["conservative", "aggressive"],
        help="Apply MSTOCK_PRESET for this run.",
    )
    ap.add_argument(
        "--compare-presets",
        action="store_true",
        help="Run both conservative and aggressive presets and print both reports.",
    )
    args = ap.parse_args()

    d = _parse_date(args.date) if args.date else datetime.now().date()

    if args.source == "yahoo":
        candles = _load_yahoo_candles(str(args.underlying), str(args.timeframe), limit=int(args.limit))
    else:
        candles = []

    candles = _filter_candles_for_date(candles, d)
    if not candles:
        print(f"No candles available for {args.underlying} on {d.isoformat()} using source={args.source}.")
        return

    def _run_for_preset(preset: Optional[str]) -> Tuple[str, BacktestStats]:
        # For preset comparisons, we want the preset bundle to take effect even when the
        # user's shell has explicit overrides set. We do this by temporarily clearing only
        # the knobs that the presets are intended to control.
        preset_keys = [
            "MSTOCK_TIMEFRAME",
            "MSTOCK_PRESET",
            "MSTOCK_MAX_TRADES_PER_DAY",
            "MSTOCK_MAX_OPEN_POSITIONS",
            "MSTOCK_COOLDOWN_SEC",
            "MSTOCK_COOLDOWN_AFTER_STOPOUT_SEC",
            "MSTOCK_MAX_PYRAMID_LEVELS",
            "MSTOCK_MIN_ATR",
            "MSTOCK_MAX_ATR",
            "MSTOCK_DIR_MIN_CONFIRMATIONS",
            "MSTOCK_DIR_MIN_SCORE_DIFF",
            "MSTOCK_DIR_MOMENTUM_ATR_MULT",
            "MSTOCK_DIR_EMA_SLOPE_ATR_MULT",
            "MSTOCK_AUTO_DIR_RSI_BUY",
            "MSTOCK_AUTO_DIR_RSI_SELL",
            "MSTOCK_AUTO_DIR_RSI_SLOP",
            "MSTOCK_ENABLE_ADX_FILTER",
            "MSTOCK_ADX_MIN_STRENGTH",
            "MSTOCK_ADX_PERIOD",
        ]

        saved: dict[str, Optional[str]] = {}
        if preset is not None:
            for k in preset_keys:
                saved[k] = os.environ.get(k)
                os.environ.pop(k, None)

        try:
            if preset:
                os.environ["MSTOCK_PRESET"] = str(preset)
            else:
                os.environ.pop("MSTOCK_PRESET", None)

            cfg = load_strategy_config()
            if preset:
                _apply_backtest_preset(cfg, preset)
            label = (
                f"preset={preset or '(none)'} timeframe={args.timeframe} horizon={int(args.horizon)} "
                f"slippage={float(args.slippage_pts):.2f} fees={float(args.fees_pts):.2f}"
            )
            return label, run_backtest(
                candles,
                cfg,
                horizon=int(args.horizon),
                slippage_pts=float(args.slippage_pts),
                fees_pts=float(args.fees_pts),
            )
        finally:
            # Restore cleared variables.
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    if bool(args.compare_presets):
        for preset in ["conservative", "aggressive"]:
            label, stats = _run_for_preset(preset)
            _print_report(label, stats)
        return

    label, stats = _run_for_preset(str(args.preset) if args.preset else None)
    _print_report(label, stats)


if __name__ == "__main__":
    main()
