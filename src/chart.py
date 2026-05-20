from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import os

import tkinter as tk

import matplotlib

# Ensure Tk-compatible backend before pyplot import.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


@dataclass
class _TradeMark:
    ts: float
    trade_id: str
    event: str  # OPEN | CLOSE


class LiveChartPlugin:
    """Embeds a live Matplotlib candlestick chart into a Tkinter frame.

    - Renders candles + overlays (Supertrend, EMA fast/slow)
    - Renders RSI panel
    - Can mark trade OPEN/CLOSE events as vertical lines
    """

    def __init__(
        self,
        parent_frame: tk.Frame,
        *,
        symbol: str = "NIFTY",
        timeframe: str = "1m",
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        supertrend_period: int = 10,
        supertrend_mult: float = 3.0,
    ) -> None:
        # Match the UI theme when possible.
        try:
            theme = (os.getenv("MSTOCK_UI_THEME", "dark") or "dark").strip().lower()
            if theme == "dark":
                plt.style.use("dark_background")
        except Exception:
            pass

        self.parent = parent_frame
        self.symbol = symbol
        self.timeframe = timeframe

        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)
        self.rsi_period = int(rsi_period)
        self.supertrend_period = int(supertrend_period)
        self.supertrend_mult = float(supertrend_mult)

        self._candles: list[Any] = []
        self._marks: list[_TradeMark] = []
        self._focus_trade_id: Optional[str] = None

        self.fig = plt.Figure(figsize=(7.8, 4.8), dpi=100)
        self.ax_price = self.fig.add_subplot(2, 1, 1)
        self.ax_rsi = self.fig.add_subplot(2, 1, 2, sharex=self.ax_price)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._style_axes()

    def _style_axes(self) -> None:
        # Keep it simple; rely on default theme. (We avoid heavy custom theming.)
        for ax in (self.ax_price, self.ax_rsi):
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        self.ax_price.set_title(f"{self.symbol} ({self.timeframe})")
        self.ax_rsi.set_ylabel("RSI")
        self.ax_price.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.axhline(70, color="gray", linewidth=0.8, alpha=0.6)
        self.ax_rsi.axhline(30, color="gray", linewidth=0.8, alpha=0.6)

    def set_symbol(self, symbol: str, timeframe: str) -> None:
        self.symbol = str(symbol or "").strip() or self.symbol
        self.timeframe = str(timeframe or "").strip() or self.timeframe
        try:
            self.ax_price.set_title(f"{self.symbol} ({self.timeframe})")
        except Exception:
            pass
        self._render()

    def focus_trade(self, trade_id: Optional[str]) -> None:
        tid = (str(trade_id).strip() if trade_id is not None else "")
        self._focus_trade_id = tid or None
        self._render()

    def push_trade_event(self, evt: Any) -> None:
        """Accepts a TradeLogEvent-like object and stores OPEN/CLOSE markers."""
        try:
            ev = str(getattr(evt, "event", "") or "").upper().strip()
            if ev not in {"OPEN", "CLOSE"}:
                return
            ts = float(getattr(evt, "ts", 0.0) or 0.0)
            tid = str(getattr(evt, "trade_id", "") or "").strip()
            if not ts or not tid:
                return
            self._marks.append(_TradeMark(ts=ts, trade_id=tid, event=ev))
        except Exception:
            return

        # Bound memory.
        if len(self._marks) > 2000:
            self._marks = self._marks[-1500:]

        self._render()

    def push_candles(self, candles_list: list[Any]) -> None:
        """Called periodically by the Strategy to feed history into the chart."""
        if not candles_list:
            return
        self._candles = list(candles_list)
        self._render()

    def _candle_dt(self, c: Any) -> Optional[datetime]:
        t = getattr(c, "time", None)
        if t is None:
            return None
        if isinstance(t, datetime):
            return t
        # Strategy Candle typically uses datetime; fall back to epoch seconds.
        try:
            return datetime.fromtimestamp(float(t))
        except Exception:
            return None

    def _ema_series(self, values: list[float], period: int) -> list[Optional[float]]:
        if period <= 0 or len(values) < period:
            return [None] * len(values)
        out: list[Optional[float]] = [None] * len(values)
        k = 2.0 / (period + 1.0)
        sma = sum(values[:period]) / float(period)
        out[period - 1] = float(sma)
        prev = float(sma)
        for i in range(period, len(values)):
            prev = (values[i] * k) + (prev * (1.0 - k))
            out[i] = float(prev)
        return out

    def _rsi_series(self, values: list[float], period: int) -> list[Optional[float]]:
        if period <= 0 or len(values) < period + 1:
            return [None] * len(values)
        out: list[Optional[float]] = [None] * len(values)
        gains: list[float] = []
        losses: list[float] = []
        for i in range(1, period + 1):
            ch = values[i] - values[i - 1]
            gains.append(max(0.0, ch))
            losses.append(max(0.0, -ch))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        out[period] = 100.0 - (100.0 / (1.0 + rs))
        for i in range(period + 1, len(values)):
            ch = values[i] - values[i - 1]
            gain = max(0.0, ch)
            loss = max(0.0, -ch)
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
            out[i] = 100.0 - (100.0 / (1.0 + rs))
        return out

    def _supertrend_series(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        *,
        period: int,
        multiplier: float,
    ) -> list[Optional[float]]:
        n = len(closes)
        out: list[Optional[float]] = [None] * n
        if period <= 0 or multiplier <= 0:
            return out
        if not (len(highs) == len(lows) == len(closes)):
            return out
        if n < period + 1:
            return out

        tr_vals = [0.0] * n
        for i in range(1, n):
            h = highs[i]
            l = lows[i]
            pc = closes[i - 1]
            tr_vals[i] = max(h - l, abs(h - pc), abs(l - pc))

        atr_vals = [0.0] * n
        atr_vals[period] = sum(tr_vals[1 : period + 1]) / period
        for i in range(period + 1, n):
            atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr_vals[i]) / period

        upper_band = [0.0] * n
        lower_band = [0.0] * n
        trend = [1] * n

        first_valid = period
        hl2 = (highs[first_valid] + lows[first_valid]) / 2.0
        curr_atr = atr_vals[first_valid]
        upper_band[first_valid] = hl2 + (multiplier * curr_atr)
        lower_band[first_valid] = hl2 - (multiplier * curr_atr)
        out[first_valid] = upper_band[first_valid]

        for i in range(first_valid + 1, n):
            hl2 = (highs[i] + lows[i]) / 2.0
            curr_atr = atr_vals[i]
            ub = hl2 + (multiplier * curr_atr)
            lb = hl2 - (multiplier * curr_atr)

            prev_ub = upper_band[i - 1]
            prev_lb = lower_band[i - 1]
            prev_close = closes[i - 1]

            upper_band[i] = ub if (ub < prev_ub or prev_close > prev_ub) else prev_ub
            lower_band[i] = lb if (lb > prev_lb or prev_close < prev_lb) else prev_lb

            prev_trend = trend[i - 1]
            curr_close = closes[i]
            if prev_trend == 1:
                if curr_close < lower_band[i]:
                    trend[i] = -1
                    out[i] = upper_band[i]
                else:
                    trend[i] = 1
                    out[i] = lower_band[i]
            else:
                if curr_close > upper_band[i]:
                    trend[i] = 1
                    out[i] = lower_band[i]
                else:
                    trend[i] = -1
                    out[i] = upper_band[i]
        return out

    def _render(self) -> None:
        self.ax_price.clear()
        self.ax_rsi.clear()
        self._style_axes()

        if not self._candles:
            self.canvas.draw_idle()
            return

        dts: list[datetime] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []

        for c in self._candles:
            dt = self._candle_dt(c)
            if dt is None:
                continue
            try:
                o = float(getattr(c, "open"))
                h = float(getattr(c, "high"))
                l = float(getattr(c, "low"))
                cl = float(getattr(c, "close"))
            except Exception:
                continue
            dts.append(dt)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(cl)

        if not closes:
            self.canvas.draw_idle()
            return

        xs = mdates.date2num(dts)

        # Candlesticks (simple rectangles + wicks)
        # Width in "days"; tuned for 1m-15m plots.
        width = max(1.0 / (24 * 60) * 0.7, (xs[-1] - xs[0]) / max(len(xs), 1) * 0.7)
        for x, o, h, l, c in zip(xs, opens, highs, lows, closes):
            up = c >= o
            color = "green" if up else "red"
            self.ax_price.vlines(x, l, h, color=color, linewidth=0.8, alpha=0.8)
            bottom = min(o, c)
            height = abs(c - o)
            if height < 1e-9:
                # Doji
                self.ax_price.hlines(o, x - width / 2, x + width / 2, color=color, linewidth=1.0)
            else:
                rect = matplotlib.patches.Rectangle(
                    (x - width / 2, bottom),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.4,
                    linewidth=0.8,
                )
                self.ax_price.add_patch(rect)

        # Overlays
        ema_f = self._ema_series(closes, self.ema_fast)
        ema_s = self._ema_series(closes, self.ema_slow)
        st = self._supertrend_series(
            highs,
            lows,
            closes,
            period=self.supertrend_period,
            multiplier=self.supertrend_mult,
        )

        def _plot_opt(ax, ys: list[Optional[float]], label: str, color: str) -> None:
            xs2: list[float] = []
            ys2: list[float] = []
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                xs2.append(x)
                ys2.append(float(y))
            if xs2:
                ax.plot(xs2, ys2, label=label, linewidth=1.0, color=color)

        _plot_opt(self.ax_price, ema_f, f"EMA({self.ema_fast})", "blue")
        _plot_opt(self.ax_price, ema_s, f"EMA({self.ema_slow})", "purple")
        _plot_opt(self.ax_price, st, f"Supertrend({self.supertrend_period},{self.supertrend_mult:g})", "orange")
        self.ax_price.legend(loc="upper left", fontsize=8)

        # RSI panel
        rsi_s = self._rsi_series(closes, self.rsi_period)
        _plot_opt(self.ax_rsi, rsi_s, f"RSI({self.rsi_period})", "black")

        # Trade markers
        focus = self._focus_trade_id
        for m in self._marks:
            if focus is not None and m.trade_id != focus:
                continue
            try:
                x = mdates.date2num(datetime.fromtimestamp(float(m.ts)))
            except Exception:
                continue
            col = "green" if m.event == "OPEN" else "red"
            self.ax_price.axvline(x, color=col, linewidth=1.0, alpha=0.5)

        self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.fig.tight_layout()
        self.canvas.draw_idle()


class TimeSeriesMultiLinePlugin:
    """Embeds a small live time-series chart into a Tkinter frame.

    Designed for dashboard-style plots (LTP/MTM) updated by the UI.
    """

    def __init__(
        self,
        parent_frame: tk.Frame,
        *,
        title: str,
        y_label: str = "",
        max_lines_in_legend: int = 8,
    ) -> None:
        # Match the UI theme when possible.
        try:
            theme = (os.getenv("MSTOCK_UI_THEME", "dark") or "dark").strip().lower()
            if theme == "dark":
                plt.style.use("dark_background")
        except Exception:
            pass

        self.parent = parent_frame
        self.title = str(title or "").strip() or "Live"
        self.y_label = str(y_label or "").strip()
        self.max_lines_in_legend = int(max_lines_in_legend)

        self.fig = plt.Figure(figsize=(7.8, 2.8), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._style_axes()

    def _style_axes(self) -> None:
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        self.ax.set_title(self.title)
        if self.y_label:
            self.ax.set_ylabel(self.y_label)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        try:
            self.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        except Exception:
            pass

    def update_series(self, series: dict[str, list[tuple[datetime, float]]]) -> None:
        """Render multiple named series.

        `series` maps label -> [(datetime, value), ...]
        """

        self.ax.clear()
        self._style_axes()

        if not series:
            self.canvas.draw_idle()
            return

        plotted = 0
        for label, pts in list(series.items()):
            if not pts:
                continue
            xs = []
            ys = []
            for dt, v in pts:
                if dt is None:
                    continue
                try:
                    xs.append(mdates.date2num(dt))
                    ys.append(float(v))
                except Exception:
                    continue
            if xs and ys:
                self.ax.plot(xs, ys, linewidth=1.0, label=str(label))
                plotted += 1

        if plotted:
            # Keep legends compact (dashboard can have multiple legs).
            try:
                handles, labels = self.ax.get_legend_handles_labels()
                if handles and labels:
                    if len(handles) > self.max_lines_in_legend:
                        handles = handles[: self.max_lines_in_legend]
                        labels = labels[: self.max_lines_in_legend]
                    self.ax.legend(handles, labels, loc="upper left", fontsize=8)
            except Exception:
                pass

        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()


class OptionChainIVSmilePlugin:
    """Simple option-chain snapshot: strike vs IV (CE/PE)."""

    def __init__(
        self,
        parent_frame: tk.Frame,
        *,
        title: str = "Option Chain (IV Smile)",
    ) -> None:
        # Match the UI theme when possible.
        try:
            theme = (os.getenv("MSTOCK_UI_THEME", "dark") or "dark").strip().lower()
            if theme == "dark":
                plt.style.use("dark_background")
        except Exception:
            pass

        self.parent = parent_frame
        self.title = str(title or "").strip() or "Option Chain"

        self.fig = plt.Figure(figsize=(7.8, 2.8), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._style_axes()

    def _style_axes(self) -> None:
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        self.ax.set_title(self.title)
        self.ax.set_xlabel("Strike")
        self.ax.set_ylabel("IV")
        try:
            self.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        except Exception:
            pass

    def update_chain(self, chain: list[dict[str, Any]], *, spot: Optional[float] = None) -> None:
        """Render IV smile using rows with strike + option_type + iv."""

        self.ax.clear()
        self._style_axes()

        if not chain:
            self.canvas.draw_idle()
            return

        ce_pts: list[tuple[float, float]] = []
        pe_pts: list[tuple[float, float]] = []
        for row in chain:
            if not isinstance(row, dict):
                continue
            ot = str(row.get("option_type") or "").upper().strip()
            try:
                strike = float(row.get("strike") or 0.0)
            except Exception:
                continue
            iv_raw = row.get("iv")
            if iv_raw is None:
                # CSV chain does not include IV; skip.
                continue
            try:
                iv = float(iv_raw)
            except Exception:
                continue
            if iv <= 0:
                continue
            if ot == "CE":
                ce_pts.append((strike, iv))
            elif ot == "PE":
                pe_pts.append((strike, iv))

        ce_pts.sort(key=lambda x: x[0])
        pe_pts.sort(key=lambda x: x[0])

        if ce_pts:
            self.ax.plot([p[0] for p in ce_pts], [p[1] for p in ce_pts], label="CE", linewidth=1.0, color="cyan")
        if pe_pts:
            self.ax.plot([p[0] for p in pe_pts], [p[1] for p in pe_pts], label="PE", linewidth=1.0, color="magenta")

        if spot is not None:
            try:
                self.ax.axvline(float(spot), color="gray", linewidth=1.0, alpha=0.6)
            except Exception:
                pass

        try:
            if ce_pts or pe_pts:
                self.ax.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()
