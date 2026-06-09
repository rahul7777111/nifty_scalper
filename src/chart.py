from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import os

import tkinter as tk
from matplotlib.collections import LineCollection, PolyCollection, PathCollection
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import numpy as np

import matplotlib
# Ensure Tk-compatible backend before pyplot import.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Patch PolyCollection to ensure backward-compatibility of get_verts for Matplotlib
if not hasattr(PolyCollection, "get_verts"):
    def _get_verts_compat(self):
        return [path.vertices for path in self.get_paths()]
    PolyCollection.get_verts = _get_verts_compat


@dataclass
class _TradeMark:
    ts: float
    trade_id: str
    event: str  # OPEN | CLOSE
    option_type: str = ""
    side: str = ""
    label: str = ""


class LiveChartPlugin:
    """Embeds a live Matplotlib candlestick chart into a Tkinter frame.

    - Renders candles + overlays (Supertrend, EMA fast/slow, EMA ultra/super, Bollinger Bands, VWAP, Opening Range)
    - Renders RSI panel
    - Can mark trade OPEN/CLOSE events as vertical lines
    - Supports ML signal visualization (BUY/SELL/EXIT markers)
    - Supports session/PREV day/ATM overlays
    """

    def __init__(
        self,
        parent_frame: tk.Frame,
        *,
        symbol: str = "NIFTY",
        timeframe: str = "1m",
        ema_fast: int = 9,
        ema_slow: int = 21,
        ema_ultra: int = 50,
        ema_super: int = 200,
        rsi_period: int = 14,
        supertrend_period: int = 10,
        supertrend_mult: float = 3.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        opening_range_period: int = 15,
        vwap_enabled: bool = True,
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
        self.ema_ultra = int(ema_ultra)
        self.ema_super = int(ema_super)
        self.rsi_period = int(rsi_period)
        self.supertrend_period = int(supertrend_period)
        self.supertrend_mult = float(supertrend_mult)
        self.bb_period = int(bb_period)
        self.bb_std = float(bb_std)
        self.opening_range_period = int(opening_range_period)
        self.vwap_enabled = bool(vwap_enabled)

        self._candles: list[Any] = []
        self._raw_candles: list[Any] = []
        self._marks: list[_TradeMark] = []
        self._focus_trade_id: Optional[str] = None
        self._active_trade: Optional[dict[str, Any]] = None
        self._indicator_cache: dict[str, dict[str, Any]] = {}
        self._suspend_view_tracking = False
        self._has_user_view = False
        self._stored_xlim: Optional[tuple[float, float]] = None
        self._stored_price_ylim: Optional[tuple[float, float]] = None
        self._stored_rsi_ylim: Optional[tuple[float, float]] = None

        # --- New state for enhanced features ---
        self._lock_to_live = False
        self._dirty = True
        self._crosshair_enabled = True  # Disables the built-in Matplotlib crosshair when a Tkinter overlay takes over
        self._max_candles: Optional[int] = None  # None = unlimited; set via set_max_candles()

        # Overlay visibility (name -> bool)
        self._overlays: dict[str, bool] = {
            "session_hl": True,
            "prevday_hl": True,
            "vwap": True,
            "opening_range": True,
            "atm": True,
            "ema_fast": True,
            "ema_slow": True,
            "ema_ultra": True,
            "ema_super": True,
            "bollinger": True,
            "volume": True,
        }

        # Market info (set from external)
        self._market_info: dict[str, Any] = {
            "spot": None,
            "futures": None,
            "atm_strike": None,
            "iv": None,
            "pcr": None,
        }

        # Session / Prev day / ATM levels
        self._session_high: Optional[float] = None
        self._session_low: Optional[float] = None
        self._session_open: Optional[float] = None
        self._prevday_high: Optional[float] = None
        self._prevday_low: Optional[float] = None
        self._prevday_open: Optional[float] = None
        self._atm_strike: Optional[float] = None
        self._opening_range_high: Optional[float] = None
        self._opening_range_low: Optional[float] = None

        # VWAP cache
        self._vwap_cache: list[Optional[float]] = []

        # Bollinger cache
        self._bb_upper_cache: list[Optional[float]] = []
        self._bb_lower_cache: list[Optional[float]] = []

        # ML signals
        self._prediction_rows: list[dict[str, Any]] = []
        self._signal_filter: str = "all"  # "all"|"executed"|"blocked"|"shadow"|"buy_only"|"sell_only"

        self.fig = plt.Figure(figsize=(7.8, 4.8), dpi=100)
        self.ax_price = self.fig.add_subplot(2, 1, 1)
        self.ax_rsi = self.fig.add_subplot(2, 1, 2, sharex=self.ax_price)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._style_axes()
        self._init_artists()
        self.ax_price.callbacks.connect("xlim_changed", self._on_limits_changed)
        self.ax_price.callbacks.connect("ylim_changed", self._on_limits_changed)
        self.ax_rsi.callbacks.connect("ylim_changed", self._on_limits_changed)

        # Connect crosshair cursor (suppressed when Tkinter overlay crosshair is active)
        self._connect_crosshair()

        # Apply institutional dark style
        self._apply_institutional_style()

    @property
    def _overlay_visible(self) -> dict[str, bool]:
        return self._overlays

    @property
    def _prevday_levels(self) -> dict[str, Any]:
        return {"high": self._prevday_high, "low": self._prevday_low}

    @property
    def _session_levels(self) -> dict[str, Any]:
        return {"high": self._session_high, "low": self._session_low}

    def _connect_crosshair(self) -> None:
        """Set up vertical/horizontal crosshairs and floating OHLC HUD."""
        # Crosshair lines
        self._cross_hair_vert = self.ax_price.axvline(0, color="#64748b", linestyle=":", linewidth=0.8, visible=False, zorder=10)
        self._cross_hair_horiz = self.ax_price.axhline(0, color="#64748b", linestyle=":", linewidth=0.8, visible=False, zorder=10)
        
        # HUD text box near top-left
        self._hud_text = self.ax_price.text(
            0.01, 0.96, "", transform=self.ax_price.transAxes,
            fontsize=8, color="#ffffff", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#020617", ec="#334155", alpha=0.85),
            zorder=11, visible=False
        )

        def _on_mouse_move(event):
            # Skip entirely when Tkinter crosshair overlay is active
            if not self._crosshair_enabled:
                return
            if event.inaxes != self.ax_price:
                self._cross_hair_vert.set_visible(False)
                self._cross_hair_horiz.set_visible(False)
                self._hud_text.set_visible(False)
                self.canvas.draw_idle()
                return

            x, y = event.xdata, event.ydata
            if x is None or y is None or not self._candles:
                return

            # Update crosshair positions
            self._cross_hair_vert.set_xdata([x, x])
            self._cross_hair_horiz.set_ydata([y, y])
            self._cross_hair_vert.set_visible(True)
            self._cross_hair_horiz.set_visible(True)

            # Find nearest candle
            try:
                xs = mdates.date2num([self._candle_dt(c) for c in self._candles if self._candle_dt(c) is not None])
                if len(xs) > 0:
                    idx = np.argmin(np.abs(xs - x))
                    c = self._candles[idx]
                    o = float(getattr(c, "open", 0))
                    h = float(getattr(c, "high", 0))
                    l = float(getattr(c, "low", 0))
                    cl = float(getattr(c, "close", 0))
                    vol = float(getattr(c, "volume", 0))
                    dt = self._candle_dt(c)
                    dt_str = dt.strftime("%Y-%m-%d %H:%M") if dt else ""

                    # Check if VWAP is available
                    vwap_val = None
                    if hasattr(self, "_vwap_cache") and len(self._vwap_cache) > idx:
                        vwap_val = self._vwap_cache[idx]

                    hud_content = f"Time: {dt_str} | O: {o:.2f} | H: {h:.2f} | L: {l:.2f} | C: {cl:.2f} | Vol: {vol:.0f}"
                    if vwap_val is not None:
                        hud_content += f" | VWAP: {vwap_val:.2f}"

                    self._hud_text.set_text(hud_content)
                    self._hud_text.set_visible(True)
            except Exception:
                pass

            self.canvas.draw_idle()

        def _on_mouse_leave(event):
            # Skip when Tkinter overlay crosshair is active
            if not self._crosshair_enabled:
                return
            self._cross_hair_vert.set_visible(False)
            self._cross_hair_horiz.set_visible(False)
            self._hud_text.set_visible(False)
            self.canvas.draw_idle()

        # Connect events
        self.canvas.mpl_connect("motion_notify_event", _on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", _on_mouse_leave)

    def _style_axes(self) -> None:
        # Move y-axis ticks/labels to the right
        self.ax_price.yaxis.tick_right()
        self.ax_price.yaxis.set_label_position("right")
        self.ax_rsi.yaxis.tick_right()
        self.ax_rsi.yaxis.set_label_position("right")

        for ax in (self.ax_price, self.ax_rsi):
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25, color="#334155")
        
        self.ax_price.set_title(f"{self.symbol} ({self.timeframe})")
        self.ax_rsi.set_ylabel("RSI")
        self.ax_price.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.axhline(70, color="#334155", linewidth=0.8, alpha=0.6)
        self.ax_rsi.axhline(30, color="#334155", linewidth=0.8, alpha=0.6)

    def _init_artists(self) -> None:
        self._wick_collection = LineCollection([], linewidths=0.8, alpha=0.85, zorder=2)
        self._body_collection = PolyCollection([], closed=True, linewidths=0.8, alpha=0.75, zorder=3)
        self.ax_price.add_collection(self._wick_collection)
        self.ax_price.add_collection(self._body_collection)

        # Consistent institutional color scheme
        self._ema_fast_line, = self.ax_price.plot([], [], linewidth=1.0, color="#facc15", label=f"EMA({self.ema_fast})")  # EMA9
        self._ema_slow_line, = self.ax_price.plot([], [], linewidth=1.0, color="#a78bfa", label=f"EMA({self.ema_slow})")  # EMA21
        self._ema_ultra_line, = self.ax_price.plot([], [], linewidth=1.0, color="#fb923c", label=f"EMA({self.ema_ultra})")  # EMA50
        self._ema_super_line, = self.ax_price.plot([], [], linewidth=1.0, color="#e879f9", label=f"EMA({self.ema_super})")  # EMA200
        self._supertrend_line, = self.ax_price.plot([], [], linewidth=1.0, color="#f43f5e", label=f"Supertrend({self.supertrend_period},{self.supertrend_mult:g})")
        self._rsi_line, = self.ax_rsi.plot([], [], linewidth=1.0, color="#38bdf8", label=f"RSI({self.rsi_period})")

        # VWAP line
        self._vwap_line, = self.ax_price.plot([], [], linewidth=1.0, color="#38bdf8", label="VWAP", zorder=4)

        # Bollinger Bands
        self._bb_upper_line, = self.ax_price.plot([], [], linewidth=0.8, color="#94a3b8", linestyle="--", label=f"BB({self.bb_period},{self.bb_std})", zorder=3)
        self._bb_lower_line, = self.ax_price.plot([], [], linewidth=0.8, color="#94a3b8", linestyle="--", zorder=3)
        self._bb_fill_collection = PolyCollection([], closed=True, alpha=0.04, zorder=1)
        self.ax_price.add_collection(self._bb_fill_collection)

        # Session high/low
        self._session_high_line, = self.ax_price.plot([], [], linewidth=1.2, color="#22c55e", label="Session H", zorder=4)
        self._session_low_line, = self.ax_price.plot([], [], linewidth=1.2, color="#ef4444", label="Session L", zorder=4)

        # Prev day high/low
        self._prevday_high_line, = self.ax_price.plot([], [], linewidth=1.0, color="#fbbf24", linestyle="--", label="PDH", zorder=4)
        self._prevday_low_line, = self.ax_price.plot([], [], linewidth=1.0, color="#f59e0b", linestyle="--", label="PDL", zorder=4)

        # Opening range (first N min)
        self._opening_range_high_line, = self.ax_price.plot([], [], linewidth=0.9, color="#ffffff", linestyle=":", label="ORH", zorder=4)
        self._opening_range_low_line, = self.ax_price.plot([], [], linewidth=0.9, color="#ffffff", linestyle=":", label="ORL", zorder=4)

        # ATM strike — horizontal price-level line
        self._atm_hline = self.ax_price.axhline(0, color="#fbbf24", linewidth=1.0, linestyle="--", alpha=0.6, zorder=5, visible=False)
        self._atm_label = self.ax_price.text(0, 0, "", fontsize=7, color="#fbbf24", ha="left", va="bottom", zorder=6)

        # Current price horizontal line
        self._current_price_line = self.ax_price.axhline(0, color="#ffffff", linewidth=1.0, linestyle="-", alpha=0.6, zorder=4, visible=False)
        self._current_price_label = self.ax_price.text(0, 0, "", fontsize=7, color="#ffffff", ha="right", va="top", zorder=6)

        # Entry / Stop / Target
        self._entry_line = self.ax_price.axhline(0.0, color="#ffffff", linewidth=0.9, linestyle="--", alpha=0.75, visible=False)
        self._stop_line = self.ax_price.axhline(0.0, color="#ef4444", linewidth=0.9, linestyle="--", alpha=0.75, visible=False)
        self._target_line = self.ax_price.axhline(0.0, color="#22c55e", linewidth=0.9, linestyle="--", alpha=0.75, visible=False)

        # Signal scatter collection (empty initialized list for separate marker shapes)
        self._signal_scatters: list[PathCollection] = []
        
        # Keep _signal_scatter around for backward compatibility
        self._signal_scatter = self.ax_price.scatter([], [], marker="^", s=60, c="lime", edgecolors="black", linewidths=0.4, zorder=7, alpha=0.9)

        # Volume bars
        self._volume_bar_collection = PolyCollection([], closed=True, alpha=0.15, zorder=1)
        self.ax_price.add_collection(self._volume_bar_collection)

        self._marker_lines: list[Any] = []
        self._marker_texts: list[Any] = []
        try:
            self.ax_price.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

    def _apply_institutional_style(self) -> None:
        """Apply the institutional dark theme to the Matplotlib figure and axes."""
        bg_color = "#020617"   # App background (near-black)
        grid_color = "#334155" # Chart grid
        text_color = "#94a3b8" # Muted text

        # Watermark already placed by _init_artists; refresh its text if symbol/timeframe changed
        if hasattr(self, "_watermark_text"):
            self._watermark_text.set_text(f"{self.symbol} · {self.timeframe.upper()} · SHADOW MODE")

        self.fig.patch.set_facecolor(bg_color)
        
        # Watermark
        self._watermark_text = self.fig.text(
            0.5, 0.55, f"{self.symbol} · {self.timeframe.upper()} · SHADOW MODE",
            fontsize=22, color=grid_color, alpha=0.08,
            ha="center", va="center", zorder=0, weight="bold"
        )

        for ax in (self.ax_price, self.ax_rsi):
            ax.set_facecolor(bg_color)
            for spine in ax.spines.values():
                spine.set_color(grid_color)
                spine.set_linewidth(0.8)
            ax.tick_params(colors=text_color, labelsize=8)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.grid(True, which="both", color=grid_color, linestyle="-", linewidth=0.5, alpha=0.2)
            
        self.ax_price.title.set_color("#ffffff")
        self.ax_price.title.set_fontsize(10)
        self.ax_price.title.set_weight("bold")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  Backward-compat properties for existing tests                       #
    # ------------------------------------------------------------------ #

    @property
    def _prevday_levels(self) -> dict[str, Optional[float]]:
        return {
            "high": self._prevday_high,
            "low": self._prevday_low,
            "open": getattr(self, "_prevday_open", None),
        }

    @property
    def _session_levels(self) -> dict[str, Optional[float]]:
        return {
            "high": self._session_high,
            "low": self._session_low,
            "open": getattr(self, "_session_open", None),
        }

    #  Public API (preserving backward compatibility)                    #
    # ------------------------------------------------------------------ #

    def set_symbol(self, symbol: str, timeframe: str) -> None:
        self.symbol = str(symbol or "").strip() or self.symbol
        self.timeframe = str(timeframe or "").strip() or self.timeframe
        try:
            self.ax_price.set_title(f"{self.symbol} ({self.timeframe})")
        except Exception:
            pass
        self._has_user_view = False
        self._dirty = True
        self._render()

    def focus_trade(self, trade_id: Optional[str]) -> None:
        tid = (str(trade_id).strip() if trade_id is not None else "")
        self._focus_trade_id = tid or None
        self._dirty = True
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
            option_type, side, label = self._extract_trade_marker_fields(
                getattr(evt, "legs", None),
                fallback_name=str(getattr(evt, "name", "") or ""),
                fallback_event=ev,
            )
            self._marks.append(
                _TradeMark(
                    ts=ts,
                    trade_id=tid,
                    event=ev,
                    option_type=option_type,
                    side=side,
                    label=label,
                )
            )
        except Exception:
            return

        # Bound memory.
        if len(self._marks) > 2000:
            self._marks = self._marks[-1500:]

        self._dirty = True
        self._render()

    def set_active_trade(self, trade: Optional[Any]) -> None:
        if isinstance(trade, dict):
            self._active_trade = dict(trade)
        else:
            self._active_trade = None
        self._dirty = True
        self._render()

    def load_trade_event_rows(self, rows: list[dict[str, Any]]) -> None:
        marks: list[_TradeMark] = []
        if rows is None:
            row_items: list[Any] = []
        elif hasattr(rows, "to_dict"):
            try:
                row_items = list(rows.to_dict("records"))
            except Exception:
                row_items = []
        else:
            try:
                row_items = list(rows)
            except Exception:
                row_items = []
        for row in row_items:
            try:
                ev = str(row.get("event") or "").upper().strip()
                if ev not in {"OPEN", "CLOSE"}:
                    continue
                ts = float(row.get("ts") or 0.0)
                tid = str(row.get("trade_id") or "").strip()
                if ts > 0.0 and tid:
                    option_type, side, label = self._extract_trade_marker_fields(
                        row.get("legs"),
                        fallback_name=str(row.get("name") or ""),
                        fallback_event=ev,
                    )
                    marks.append(
                        _TradeMark(
                            ts=ts,
                            trade_id=tid,
                            event=ev,
                            option_type=option_type,
                            side=side,
                            label=label,
                        )
                    )
            except Exception:
                continue
        if marks:
            self._marks = marks[-2000:]
            self._dirty = True
            self._render()

    def push_candles(self, candles_list: list[Any]) -> None:
        """Called periodically by the Strategy to feed history into the chart."""
        if candles_list is None:
            return
        try:
            candles = list(candles_list)
        except Exception:
            return
        self._raw_candles = candles
        # Apply max candle limit if set
        if self._max_candles is not None and len(self._raw_candles) > self._max_candles:
            self._raw_candles = self._raw_candles[-self._max_candles:]
        self._dirty = True
        self._render()

    # ------------------------------------------------------------------ #
    #  New public API for enhanced features                              #
    # ------------------------------------------------------------------ #

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        """Toggle visibility of a named overlay."""
        self._overlays[str(name)] = bool(visible)
        self._dirty = True
        self._render()

    def set_market_info(
        self,
        spot: Optional[float] = None,
        futures: Optional[float] = None,
        atm_strike: Optional[float] = None,
        iv: Optional[float] = None,
        pcr: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Store market metadata for overlay rendering."""
        self._market_info["spot"] = spot
        self._market_info["futures"] = futures
        self._market_info["atm_strike"] = atm_strike
        self._market_info["iv"] = iv
        self._market_info["pcr"] = pcr
        for k, v in kwargs.items():
            self._market_info[k] = v
        # Track regime and market status for dynamic title
        self._market_info["regime_label"] = kwargs.get("regime_label", self._market_info.get("regime_label", ""))
        self._market_info["market_status"] = kwargs.get("market_status", self._market_info.get("market_status", ""))
        if atm_strike is not None:
            self._atm_strike = float(atm_strike)
        self._dirty = True
        self._render()
        self._render()

    def set_session_levels(self, high: Optional[float], low: Optional[float], open_price: Optional[float] = None) -> None:
        """Set session high/low horizontal lines."""
        self._session_high = float(high) if high is not None else None
        self._session_low = float(low) if low is not None else None
        if open_price is not None:
            self._session_open = float(open_price)
        self._dirty = True
        self._render()

    def set_prevday_levels(self, high: Optional[float], low: Optional[float], open_price: Optional[float] = None) -> None:
        """Set previous day high/low horizontal lines."""
        self._prevday_high = float(high) if high is not None else None
        self._prevday_low = float(low) if low is not None else None
        if open_price is not None:
            self._prevday_open = float(open_price)
        self._dirty = True
        self._render()

    def set_atm_strike(self, strike: Optional[float]) -> None:
        """Set ATM strike for vertical marker."""
        self._atm_strike = float(strike) if strike is not None else None
        self._dirty = True
        self._render()

    def set_lock_to_live(self, enabled: bool) -> None:
        """Enable/disable auto-scroll to latest candle."""
        self._lock_to_live = bool(enabled)

    def disable_crosshair(self) -> None:
        """Disable the built-in Matplotlib crosshair HUD. Call this when a Tkinter overlay crosshair is active."""
        self._crosshair_enabled = False
        # Hide existing crosshair elements immediately
        if hasattr(self, "_cross_hair_vert"):
            self._cross_hair_vert.set_visible(False)
        if hasattr(self, "_cross_hair_horiz"):
            self._cross_hair_horiz.set_visible(False)
        if hasattr(self, "_hud_text"):
            self._hud_text.set_visible(False)
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def enable_crosshair(self) -> None:
        """Re-enable the built-in Matplotlib crosshair HUD."""
        self._crosshair_enabled = True

    def reset_view(self) -> None:
        """Reset chart zoom to show all candles and auto-scale axes."""
        self._suspend_view_tracking = False
        self._has_user_view = False
        self._stored_xlim = None
        self._stored_price_ylim = None
        self._stored_rsi_ylim = None
        self._dirty = True
        if hasattr(self, "ax_price"):
            self.ax_price.autoscale_view()
            self.ax_rsi.autoscale_view()
        self._render()

    def set_max_candles(self, count: int) -> None:
        """Limit rendered candles to the most recent `count`."""
        self._max_candles = max(1, int(count))
        self._dirty = True
        self._render()

    def set_timeframe(self, tf: str) -> None:
        """Change chart timeframe label without re-fetching data."""
        self.timeframe = str(tf)
        self._dirty = True
        self._render()

    def focus_latest_signal(self) -> None:
        """Focus the chart on the most recent prediction marker."""
        rows = getattr(self, "_prediction_rows", []) or []
        if not rows:
            return
        latest = rows[-1]
        tid = str(latest.get("prediction_id", "") or latest.get("trade_id", "") or "")
        if tid:
            self.focus_trade(tid)

    def set_signal_filter(self, mode: str) -> None:
        """Set signal filter mode: 'all' | 'executed' | 'blocked' | 'shadow' | 'buy_only' | 'sell_only'."""
        valid = {"all", "executed", "blocked", "shadow", "buy_only", "sell_only"}
        mode_str = str(mode or "all").lower().strip()
        if mode_str not in valid:
            mode_str = "all"
        self._signal_filter = mode_str
        self._dirty = True
        self._render()

    def load_prediction_rows(self, rows: list[dict[str, Any]]) -> None:
        """Load ML prediction rows for signal visualization.

        Each row dict should contain:
          ts, probability, predicted_class, confidence_bucket,
          regime, trade_candidate, trade_taken
        """
        if rows is None:
            self._prediction_rows = []
            return
        try:
            self._prediction_rows = list(rows)
        except Exception:
            self._prediction_rows = []
        self._dirty = True
        self._render()

    def export_png(self, filename: str) -> None:
        """Export the current figure to a PNG file."""
        try:
            self.fig.savefig(str(filename), dpi=150, bbox_inches="tight")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Indicator computation methods                                     #
    # ------------------------------------------------------------------ #

    def _bollinger_series(
        self,
        closes: list[float],
        period: int,
        std: float,
    ) -> tuple[list[Optional[float]], list[Optional[float]]]:
        """Compute Bollinger upper and lower bands."""
        cache_key = f"bb_{period}_{std}"
        cached = self._indicator_cache.get(cache_key)
        closes_list = self._normalize_sequence(closes)
        n = len(closes_list)
        if period <= 0 or std <= 0 or n < period:
            return [None] * n, [None] * n
        if cached and self._same_sequence(cached.get("source"), closes_list):
            return list(cached.get("upper") or []), list(cached.get("lower") or [])
        upper: list[Optional[float]] = [None] * n
        lower: list[Optional[float]] = [None] * n
        for i in range(period - 1, n):
            slice_vals = closes_list[i - period + 1:i + 1]
            mean = sum(slice_vals) / period
            variance = sum((v - mean) ** 2 for v in slice_vals) / period
            std_val = variance ** 0.5
            upper[i] = mean + std * std_val
            lower[i] = mean - std * std_val
        self._indicator_cache[cache_key] = {
            "source": list(closes_list),
            "upper": list(upper),
            "lower": list(lower),
        }
        return upper, lower

    def _vwap_series(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
    ) -> list[Optional[float]]:
        """Compute VWAP series from OHLCV data."""
        n = min(len(opens), len(highs), len(lows), len(closes), len(volumes))
        if n == 0:
            return [None] * len(closes)
        vwap: list[Optional[float]] = [None] * n
        cum_pv = 0.0
        cum_vol = 0.0
        for i in range(n):
            typical = (highs[i] + lows[i] + closes[i]) / 3.0
            vol = max(0.0, volumes[i])
            cum_pv += typical * vol
            cum_vol += vol
            if cum_vol > 0:
                vwap[i] = cum_pv / cum_vol
            else:
                vwap[i] = typical
        return vwap

    def _opening_range_values(
        self,
        highs: list[float],
        lows: list[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Return high/low of first N candles (opening range)."""
        period = self.opening_range_period
        if period <= 0 or len(highs) < 1 or len(lows) < 1:
            return None, None
        n = min(period, len(highs), len(lows))
        if n == 0:
            return None, None
        or_high = max(highs[:n])
        or_low = min(lows[:n])
        return or_high, or_low

    # ------------------------------------------------------------------ #
    #  Signal rendering                                                  #
    # ------------------------------------------------------------------ #

    def _render_signals(self, xs: list[float], closes: list[float]) -> None:
        """Render ML signal markers on price chart."""
        # Clear existing signal scatters
        if not hasattr(self, "_signal_scatters"):
            self._signal_scatters = []
        for scat in self._signal_scatters:
            try:
                scat.remove()
            except Exception:
                pass
        self._signal_scatters = []

        # Clear backward-compatible single scatter
        try:
            self._signal_scatter.set_offsets(np.empty((0, 2)))
        except Exception:
            pass

        if xs is None or len(xs) == 0 or not closes or not self._prediction_rows:
            return

        # Map timestamps to x positions
        x_map: dict[float, float] = {}
        for i, x in enumerate(xs):
            c_ts = self._candle_dt(self._candles[i]) if i < len(self._candles) else None
            if c_ts:
                try:
                    x_map[mdates.date2num(c_ts)] = x
                except Exception:
                    pass

        # We'll group the points to draw them in batches
        groups = {
            "exec_buy": {"x": [], "y": [], "sizes": [], "color": "#10b981", "marker": "^"},
            "exec_sell": {"x": [], "y": [], "sizes": [], "color": "#f43f5e", "marker": "v"},
            "blocked": {"x": [], "y": [], "sizes": [], "color": "#f59e0b", "marker": "o"},
            "shadow": {"x": [], "y": [], "sizes": [], "color": "#38bdf8", "marker": "d"}
        }

        for row in self._prediction_rows:
            try:
                ts_val = row.get("ts")
                if ts_val is None:
                    continue
                ts_num: float
                if isinstance(ts_val, (int, float)):
                    ts_num = float(ts_val)
                elif isinstance(ts_val, datetime):
                    ts_num = mdates.date2num(ts_val)
                else:
                    try:
                        ts_num = float(ts_val)
                    except Exception:
                        continue

                x = x_map.get(ts_num)
                if x is None:
                    try:
                        x = mdates.date2num(datetime.fromtimestamp(ts_num))
                    except Exception:
                        continue

                # Find nearest y
                idx = np.argmin(np.abs(np.array(xs) - x)) if len(xs) > 0 else 0
                y = float(closes[min(idx, len(closes) - 1)])

                pred_class = int(row.get("predicted_class", -1))
                trade_candidate = str(row.get("trade_candidate", "")).upper().strip()
                trade_taken = row.get("trade_taken") in (True, 1, "1", "true", "True")
                no_trade_reason = str(row.get("no_trade_reason") or "").strip()

                # Apply filter
                filter_mode = self._signal_filter
                if filter_mode == "executed" and not trade_taken:
                    continue
                elif filter_mode == "blocked" and trade_taken:
                    continue
                elif filter_mode == "shadow":
                    if trade_taken:
                        continue
                elif filter_mode == "buy_only" and not is_buy:
                    continue
                elif filter_mode == "sell_only" and not is_sell:
                    continue

                prob = float(row.get("probability", 0.5))
                thresh = float(row.get("threshold", 0.5))
                confidence = float(row.get("confidence") or abs(prob - 0.5) * 2)

                is_buy = (pred_class == 1) or ("LONG" in trade_candidate) or ("BUY" in trade_candidate)
                is_sell = (pred_class == 0) or ("SHORT" in trade_candidate) or ("SELL" in trade_candidate)

                size = int(40 + 120 * confidence)

                if trade_taken:
                    if is_buy:
                        groups["exec_buy"]["x"].append(x)
                        groups["exec_buy"]["y"].append(y)
                        groups["exec_buy"]["sizes"].append(size)
                    elif is_sell:
                        groups["exec_sell"]["x"].append(x)
                        groups["exec_sell"]["y"].append(y)
                        groups["exec_sell"]["sizes"].append(size)
                else:
                    is_blocked = (prob < thresh) or bool(no_trade_reason)
                    if is_blocked:
                        groups["blocked"]["x"].append(x)
                        groups["blocked"]["y"].append(y)
                        groups["blocked"]["sizes"].append(size)
                    else:
                        groups["shadow"]["x"].append(x)
                        groups["shadow"]["y"].append(y)
                        groups["shadow"]["sizes"].append(int(30 + 60 * confidence))

            except Exception:
                continue

        # Draw grouped scatters
        for key, g in groups.items():
            if not g["x"]:
                continue
            if key == "blocked":
                # Hollow marker for blocked
                scat = self.ax_price.scatter(
                    g["x"], g["y"], s=g["sizes"],
                    marker=g["marker"], facecolors="none",
                    edgecolors=g["color"], linewidths=1.5,
                    zorder=7, alpha=0.9
                )
            else:
                scat = self.ax_price.scatter(
                    g["x"], g["y"], s=g["sizes"],
                    marker=g["marker"], c=g["color"],
                    edgecolors="black", linewidths=0.5,
                    zorder=7, alpha=0.9
                )
            self._signal_scatters.append(scat)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    def _on_limits_changed(self, _ax: Any) -> None:
        if self._suspend_view_tracking:
            return
        try:
            self._stored_xlim = tuple(self.ax_price.get_xlim())
            self._stored_price_ylim = tuple(self.ax_price.get_ylim())
            self._stored_rsi_ylim = tuple(self.ax_rsi.get_ylim())
            self._has_user_view = True
        except Exception:
            pass

    def _extract_trade_marker_fields(
        self,
        legs: Any,
        *,
        fallback_name: str = "",
        fallback_event: str = "",
    ) -> tuple[str, str, str]:
        option_type = ""
        side = ""
        leg_list = list(legs or []) if isinstance(legs, list) else []
        for leg in leg_list:
            if not isinstance(leg, dict):
                continue
            if bool(leg.get("is_hedge")):
                continue
            side = str(leg.get("side") or "").upper().strip()
            option_type = str(leg.get("option_type") or "").upper().strip()
            symbol = str(leg.get("symbol") or "").upper().strip()
            if option_type not in {"CE", "PE"}:
                if symbol.endswith("CE"):
                    option_type = "CE"
                elif symbol.endswith("PE"):
                    option_type = "PE"
            if side in {"BUY", "SELL"} or option_type in {"CE", "PE"}:
                break

        if option_type not in {"CE", "PE"}:
            name = str(fallback_name or "").lower()
            if "call" in name or name.endswith("ce"):
                option_type = "CE"
            elif "put" in name or name.endswith("pe"):
                option_type = "PE"

        if side not in {"BUY", "SELL"}:
            side = "SELL" if str(fallback_event).upper() == "CLOSE" else "BUY"

        label_parts = [part for part in (option_type, side.title()) if part]
        label = " ".join(label_parts) if label_parts else str(fallback_event or "TRADE").title()
        return option_type, side, label

    def _nearest_price_at_timestamp(self, timestamp: float, xs: list[float], closes: list[float]) -> Optional[float]:
        try:
            xs_len = len(xs)
            closes_len = len(closes)
        except Exception:
            return None
        if xs_len <= 0 or closes_len <= 0:
            return None
        try:
            target_x = mdates.date2num(datetime.fromtimestamp(float(timestamp)))
        except Exception:
            return None
        best_idx = min(range(min(xs_len, closes_len)), key=lambda idx: abs(xs[idx] - target_x))
        try:
            return float(closes[best_idx])
        except Exception:
            return None

    def _set_optional_line(self, line: Any, level: Optional[float]) -> None:
        if level is None:
            line.set_visible(False)
            return
        try:
            y = float(level)
        except Exception:
            line.set_visible(False)
            return
        line.set_ydata([y, y])
        line.set_visible(True)

    def _extract_trade_barriers(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        trade = self._active_trade if isinstance(self._active_trade, dict) else {}
        if not trade:
            return None, None, None
        meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
        tb = meta.get("triple_barrier") if isinstance(meta.get("triple_barrier"), dict) else {}
        entry = trade.get("entry_price", tb.get("entry_price"))
        stop = trade.get("stop_loss", tb.get("lower_stop_barrier"))
        target = trade.get("profit_target", tb.get("upper_profit_barrier"))
        return (
            float(entry) if entry not in (None, "") else None,
            float(stop) if stop not in (None, "") else None,
            float(target) if target not in (None, "") else None,
        )

    def _candle_dt(self, c: Any) -> Optional[datetime]:
        # Support both market_data.Candle (time=) and live_chart_snapshot.Candle (timestamp=).
        t = getattr(c, "time", None)
        if t is None:
            t = getattr(c, "timestamp", None)
        if t is None:
            return None
        if isinstance(t, datetime):
            return t
        # Strategy Candle typically uses datetime; fall back to epoch seconds.
        try:
            return datetime.fromtimestamp(float(t))
        except Exception:
            return None

    def _normalize_sequence(self, values: Any) -> list[Any]:
        try:
            return list(values) if values is not None else []
        except Exception:
            return []

    def _same_sequence(self, left: Any, right: Any) -> bool:
        left_list = self._normalize_sequence(left)
        right_list = self._normalize_sequence(right)
        if len(left_list) != len(right_list):
            return False
        for lval, rval in zip(left_list, right_list):
            if lval is None or rval is None:
                if lval is not rval:
                    return False
                continue
            try:
                if float(lval) != float(rval):
                    return False
            except Exception:
                if lval != rval:
                    return False
        return True

    def _ema_series(self, values: list[float], period: int) -> list[Optional[float]]:
        cache_key = f"ema_{int(period)}"
        cached = self._indicator_cache.get(cache_key)
        values_list = self._normalize_sequence(values)
        if period <= 0 or len(values) < period:
            return [None] * len(values)
        if cached and self._same_sequence(cached.get("source"), values_list):
            return list(cached.get("series") or [])
        out: list[Optional[float]] = [None] * len(values)
        start_idx = period - 1
        cached_source = self._normalize_sequence(cached.get("source") if cached else None)
        if cached and len(cached_source) < len(values_list) and self._same_sequence(values_list[: len(cached_source)], cached_source):
            prev_series = list(cached.get("series") or [])
            prev_len = len(prev_series)
            if prev_len >= period:
                out[:prev_len] = prev_series[:prev_len]
                start_idx = max(period - 1, prev_len - 1)
        k = 2.0 / (period + 1.0)
        if out[period - 1] is None:
            sma = sum(values[:period]) / float(period)
            out[period - 1] = float(sma)
            prev = float(sma)
            start = period
        else:
            prev = float(out[start_idx] or 0.0)
            start = start_idx + 1
        for i in range(start, len(values)):
            prev = (values[i] * k) + (prev * (1.0 - k))
            out[i] = float(prev)
        self._indicator_cache[cache_key] = {"source": list(values_list), "series": list(out)}
        return out

    def _rsi_series(self, values: list[float], period: int) -> list[Optional[float]]:
        if period <= 0 or len(values) < period + 1:
            return [None] * len(values)
        cache_key = f"rsi_{int(period)}"
        cached = self._indicator_cache.get(cache_key)
        values_list = self._normalize_sequence(values)
        if cached and self._same_sequence(cached.get("source"), values_list):
            return list(cached.get("series") or [])
        out: list[Optional[float]] = [None] * len(values)
        recompute_from = 1
        cached_source = self._normalize_sequence(cached.get("source") if cached else None)
        if cached and len(cached_source) < len(values_list) and self._same_sequence(values_list[: len(cached_source)], cached_source):
            prev_series = list(cached.get("series") or [])
            prev_len = len(prev_series)
            out[: max(0, prev_len - (period + 2))] = prev_series[: max(0, prev_len - (period + 2))]
            recompute_from = max(1, prev_len - (period + 2))
        gains: list[float] = []
        losses: list[float] = []
        seed_end = max(period + 1, recompute_from + period)
        seed_end = min(seed_end, len(values) - 1)
        seed_start = max(1, seed_end - period)
        for i in range(seed_start, seed_end + 1):
            ch = values[i] - values[i - 1]
            gains.append(max(0.0, ch))
            losses.append(max(0.0, -ch))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        out[seed_end] = 100.0 - (100.0 / (1.0 + rs))
        for i in range(seed_end + 1, len(values)):
            ch = values[i] - values[i - 1]
            gain = max(0.0, ch)
            loss = max(0.0, -ch)
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
            out[i] = 100.0 - (100.0 / (1.0 + rs))
        self._indicator_cache[cache_key] = {"source": list(values_list), "series": list(out)}
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
        cache_key = f"supertrend_{int(period)}_{float(multiplier)}"
        cached = self._indicator_cache.get(cache_key)
        highs_list = self._normalize_sequence(highs)
        lows_list = self._normalize_sequence(lows)
        closes_list = self._normalize_sequence(closes)
        if cached and self._same_sequence(cached.get("highs"), highs_list) and self._same_sequence(cached.get("lows"), lows_list) and self._same_sequence(cached.get("closes"), closes_list):
            return list(cached.get("series") or [])
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
        self._indicator_cache[cache_key] = {
            "highs": list(highs_list),
            "lows": list(lows_list),
            "closes": list(closes_list),
            "series": list(out),
        }
        return out

    def _resample_candles_to_timeframe(self, candles: list[Any], timeframe_str: str) -> list[Any]:
        if not candles:
            return []
        if timeframe_str == "1m":
            return candles
        try:
            minutes = int(timeframe_str.replace("m", ""))
        except ValueError:
            return candles
        if minutes <= 1:
            return candles

        from market_data import Candle

        resampled: list[Any] = []
        current_group_dt = None
        group_open = None
        group_high = -float('inf')
        group_low = float('inf')
        group_close = None
        group_volume = 0.0

        for c in candles:
            dt = self._candle_dt(c)
            if dt is None:
                continue
            try:
                o = float(getattr(c, "open"))
                h = float(getattr(c, "high"))
                l = float(getattr(c, "low"))
                cl = float(getattr(c, "close"))
                v = float(getattr(c, "volume", 0.0) or 0.0)
            except Exception:
                continue

            delta_minutes = dt.minute % minutes
            group_dt = dt - timedelta(minutes=delta_minutes, seconds=dt.second, microseconds=dt.microsecond)

            if current_group_dt is None or group_dt != current_group_dt:
                if current_group_dt is not None:
                    resampled.append(Candle(
                        time=current_group_dt,
                        open=group_open,
                        high=group_high,
                        low=group_low,
                        close=group_close,
                        volume=group_volume
                    ))
                current_group_dt = group_dt
                group_open = o
                group_high = h
                group_low = l
                group_close = cl
                group_volume = v
            else:
                group_high = max(group_high, h)
                group_low = min(group_low, l)
                group_close = cl
                group_volume += v

        if current_group_dt is not None:
            resampled.append(Candle(
                time=current_group_dt,
                open=group_open,
                high=group_high,
                low=group_low,
                close=group_close,
                volume=group_volume
            ))

        return resampled

    # ------------------------------------------------------------------ #
    #  Main render                                                       #
    # ------------------------------------------------------------------ #

    def _render(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        if not hasattr(self, "_raw_candles") or not self._raw_candles:
            self._raw_candles = self._candles

        if not self._raw_candles:
            self._clear_artists()
            self.canvas.draw_idle()
            return

        self._candles = self._resample_candles_to_timeframe(self._raw_candles, self.timeframe)

        if not self._candles:
            self._clear_artists()
            self.canvas.draw_idle()
            return

        dts: list[datetime] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []

        for c in self._candles:
            dt = self._candle_dt(c)
            if dt is None:
                continue
            try:
                o = float(getattr(c, "open"))
                h = float(getattr(c, "high"))
                l = float(getattr(c, "low"))
                cl = float(getattr(c, "close"))
                v = float(getattr(c, "volume", 0.0))
            except Exception:
                continue
            dts.append(dt)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(cl)
            volumes.append(v)

        if not closes:
            self._clear_artists()
            self.canvas.draw_idle()
            return

        prev_xlim = self._stored_xlim if self._has_user_view else None
        prev_price_ylim = self._stored_price_ylim if self._has_user_view else None
        prev_rsi_ylim = self._stored_rsi_ylim if self._has_user_view else None

        # Update dynamic title showing LTP and price change/percentage
        try:
            title_parts = [self.symbol, self.timeframe.upper()]
            regime = str(self._market_info.get("regime_label") or "")
            if regime and regime not in ("N/A", "", "unknown"):
                title_parts.append(regime)
            mkt_status = str(self._market_info.get("market_status") or "").upper()
            if mkt_status and mkt_status not in ("CLOSED", "UNKNOWN", ""):
                title_parts.append(mkt_status)
            title_parts.append(f"Updated {datetime.now().strftime('%H:%M:%S')}")
            if closes:
                last_close = float(closes[-1])
                title_parts.append(f"LTP {last_close:.2f}")
                if len(closes) > 1:
                    prev_close = float(closes[-2])
                    diff = last_close - prev_close
                    pct = (diff / prev_close) * 100 if prev_close != 0 else 0.0
                    sign = "+" if diff >= 0 else ""
                    title_parts.append(f"{sign}{diff:.2f} ({sign}{pct:.2f}%)")
            self.ax_price.set_title(" · ".join(title_parts))
        except Exception:
            pass

        xs = mdates.date2num(dts)

        width = max(1.0 / (24 * 60) * 0.7, (xs[-1] - xs[0]) / max(len(xs), 1) * 0.7)

        # ---- Candle bodies and wicks ----
        wick_segments = []
        wick_colors = []
        body_polys = []
        body_colors = []
        # Update dynamic watermark text
        if hasattr(self, "_watermark_text"):
            self._watermark_text.set_text(f"{self.symbol} · {self.timeframe.upper()} · SHADOW MODE")

        for x, o, h, l, c in zip(xs, opens, highs, lows, closes):
            up = c >= o
            color = "#22c55e" if up else "#ef4444"
            wick_segments.append([(x, l), (x, h)])
            wick_colors.append(color)
            bottom = min(o, c)
            height = abs(c - o)
            top = bottom + (height if height >= 1e-9 else 1e-9)
            body_polys.append(
                [
                    (x - width / 2, bottom),
                    (x - width / 2, top),
                    (x + width / 2, top),
                    (x + width / 2, bottom),
                ]
            )
            body_colors.append(color)
        self._wick_collection.set_segments(wick_segments)
        self._wick_collection.set_color(wick_colors)
        self._body_collection.set_verts(body_polys)
        self._body_collection.set_facecolor(body_colors)
        self._body_collection.set_edgecolor(body_colors)

        # ---- EMA overlays ----
        ema_f = self._ema_series(closes, self.ema_fast)
        ema_s = self._ema_series(closes, self.ema_slow)
        ema_u = self._ema_series(closes, self.ema_ultra)
        ema_sp = self._ema_series(closes, self.ema_super)

        def _line_data(ys: list[Optional[float]]) -> tuple[list[float], list[float]]:
            xs2: list[float] = []
            ys2: list[float] = []
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                xs2.append(x)
                ys2.append(float(y))
            return xs2, ys2

        self._ema_fast_line.set_data(*_line_data(ema_f))
        self._ema_slow_line.set_data(*_line_data(ema_s))
        self._ema_ultra_line.set_data(*_line_data(ema_u))
        self._ema_super_line.set_data(*_line_data(ema_sp))

        # Show/hide EMAs based on overlay settings
        self._ema_fast_line.set_visible(self._overlays.get("ema_fast", True))
        self._ema_slow_line.set_visible(self._overlays.get("ema_slow", True))
        self._ema_ultra_line.set_visible(self._overlays.get("ema_ultra", True))
        self._ema_super_line.set_visible(self._overlays.get("ema_super", True))

        # ---- Supertrend ----
        st = self._supertrend_series(
            highs,
            lows,
            closes,
            period=self.supertrend_period,
            multiplier=self.supertrend_mult,
        )
        self._supertrend_line.set_data(*_line_data(st))

        # ---- RSI ----
        rsi_s = self._rsi_series(closes, self.rsi_period)
        self._rsi_line.set_data(*_line_data(rsi_s))

        # ---- VWAP ----
        if self.vwap_enabled and self._overlays.get("vwap", True):
            self._vwap_cache = self._vwap_series(opens, highs, lows, closes, volumes)
            vwap_xs, vwap_ys = _line_data(self._vwap_cache)
            self._vwap_line.set_data(vwap_xs, vwap_ys)
            self._vwap_line.set_visible(True)
        else:
            self._vwap_line.set_visible(False)

        # ---- Bollinger Bands ----
        if self._overlays.get("bollinger", True):
            bb_upper, bb_lower = self._bollinger_series(closes, self.bb_period, self.bb_std)
            bb_xs_u, bb_ys_u = _line_data(bb_upper)
            bb_xs_l, bb_ys_l = _line_data(bb_lower)
            self._bb_upper_line.set_data(bb_xs_u, bb_ys_u)
            self._bb_lower_line.set_data(bb_xs_l, bb_ys_l)
            self._bb_upper_line.set_visible(bool(bb_xs_u))
            self._bb_lower_line.set_visible(bool(bb_xs_l))

            # Fill between
            if bb_xs_u and bb_ys_u and bb_xs_l and bb_ys_l:
                n_u = len(bb_xs_u)
                n_l = len(bb_xs_l)
                fill_verts = []
                for i in range(min(n_u, n_l)):
                    fill_verts.append([(bb_xs_u[i], bb_ys_u[i]), (bb_xs_u[i], bb_ys_l[i])])
                self._bb_fill_collection.set_verts(fill_verts)
                self._bb_fill_collection.set_visible(True)
            else:
                self._bb_fill_collection.set_visible(False)
        else:
            self._bb_upper_line.set_visible(False)
            self._bb_lower_line.set_visible(False)
            self._bb_fill_collection.set_visible(False)

        # ---- Opening Range ----
        if self._overlays.get("opening_range", True):
            or_h, or_l = self._opening_range_values(highs, lows)
            self._opening_range_high = or_h
            self._opening_range_low = or_l
            if or_h is not None:
                self._opening_range_high_line.set_ydata([or_h, or_h])
                self._opening_range_high_line.set_xdata([xs[0], xs[-1]])
                self._opening_range_high_line.set_visible(True)
            else:
                self._opening_range_high_line.set_visible(False)
            if or_l is not None:
                self._opening_range_low_line.set_ydata([or_l, or_l])
                self._opening_range_low_line.set_xdata([xs[0], xs[-1]])
                self._opening_range_low_line.set_visible(True)
            else:
                self._opening_range_low_line.set_visible(False)
        else:
            self._opening_range_high_line.set_visible(False)
            self._opening_range_low_line.set_visible(False)

        # ---- Session High/Low ----
        if self._overlays.get("session_hl", True):
            self._set_optional_line(self._session_high_line, self._session_high)
            self._set_optional_line(self._session_low_line, self._session_low)
        else:
            self._session_high_line.set_visible(False)
            self._session_low_line.set_visible(False)

        # ---- Prev Day High/Low ----
        if self._overlays.get("prevday_hl", True):
            self._set_optional_line(self._prevday_high_line, self._prevday_high)
            self._set_optional_line(self._prevday_low_line, self._prevday_low)
        else:
            self._prevday_high_line.set_visible(False)
            self._prevday_low_line.set_visible(False)

        # ---- Volume Bars (lower 15% of price pane) ----
        if self._overlays.get("volume", True) and len(xs) == len(volumes):
            price_min = min(lows)
            price_max = max(highs)
            price_span = max(price_max - price_min, 1e-6)
            vol_min = min(volumes) if volumes else 0.0
            vol_max = max(volumes) if volumes else 1.0
            vol_span = max(vol_max - vol_min, 1.0)
            vol_bottom = price_min - price_span * 0.15  # bottom of volume zone
            vol_height_px = price_span * 0.13           # height reserved for volume bars
            vol_bar_polys = []
            vol_bar_colors = []
            bar_w = max(1.0 / (24 * 60) * 0.5, (xs[-1] - xs[0]) / max(len(xs), 1) * 0.5)
            for i, (x, v) in enumerate(zip(xs, volumes)):
                if v is None or v <= 0:
                    continue
                # Normalize volume to bar height
                norm_v = vol_bottom + ((v - vol_min) / vol_span) * vol_height_px
                is_bull = closes[i] >= opens[i]
                color = "#22c55e" if is_bull else "#ef4444"
                vol_bar_polys.append([
                    (x - bar_w / 2, vol_bottom),
                    (x - bar_w / 2, norm_v),
                    (x + bar_w / 2, norm_v),
                    (x + bar_w / 2, vol_bottom),
                ])
                vol_bar_colors.append(color)
            self._volume_bar_collection.set_verts(vol_bar_polys)
            self._volume_bar_collection.set_facecolor(vol_bar_colors)
            self._volume_bar_collection.set_edgecolor(vol_bar_colors)
            self._volume_bar_collection.set_visible(bool(vol_bar_polys))
        else:
            self._volume_bar_collection.set_visible(False)

        # ---- ATM Strike — horizontal price-level line ----
        if self._overlays.get("atm", True) and self._atm_strike is not None and len(xs) > 0:
            atm_price = float(self._atm_strike)
            self._atm_hline.set_ydata([atm_price, atm_price])
            self._atm_hline.set_xdata([xs[0], xs[-1]])
            self._atm_hline.set_visible(True)
            # Label at right edge of chart
            self._atm_label.set_position((xs[-1], atm_price))
            self._atm_label.set_text(f"  ATM {self._atm_strike:.0f}")
            self._atm_label.set_visible(True)
        else:
            self._atm_hline.set_visible(False)
            self._atm_label.set_visible(False)

        # ---- Current Price Line ----
        if closes:
            last_close = float(closes[-1])
            self._current_price_line.set_ydata([last_close, last_close])
            self._current_price_line.set_xdata([xs[0], xs[-1]])
            self._current_price_line.set_visible(True)
            self._current_price_label.set_position((xs[-1], last_close))
            self._current_price_label.set_text(f"  LTP {last_close:.2f}")
            self._current_price_label.set_visible(True)
        else:
            self._current_price_line.set_visible(False)
            self._current_price_label.set_visible(False)

        # ---- Trade markers ----
        for line in self._marker_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._marker_lines = []
        for text in self._marker_texts:
            try:
                text.remove()
            except Exception:
                pass
        self._marker_texts = []
        focus = self._focus_trade_id
        price_min = min(lows)
        price_max = max(highs)
        price_span = max(price_max - price_min, 1e-6)
        for m in self._marks:
            if focus is not None and m.trade_id != focus:
                continue
            try:
                x = mdates.date2num(datetime.fromtimestamp(float(m.ts)))
            except Exception:
                continue
            price_y = self._nearest_price_at_timestamp(m.ts, xs, closes)
            if price_y is None:
                continue
            is_buy = str(m.side).upper() == "BUY"
            option_type = str(m.option_type).upper()
            color = "deepskyblue" if option_type == "CE" else ("gold" if option_type == "PE" else ("lime" if is_buy else "red"))
            marker = "^" if is_buy else "v"
            y_offset = price_span * (0.018 if is_buy else -0.018)
            text_offset = price_span * (0.05 if is_buy else -0.05)
            scatter = self.ax_price.scatter(
                [x],
                [price_y + y_offset],
                marker=marker,
                s=70,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=6,
                alpha=0.95,
            )
            self._marker_lines.append(scatter)
            self._marker_texts.append(
                self.ax_price.annotate(
                    m.label or m.event.title(),
                    xy=(x, price_y + y_offset),
                    xytext=(x, price_y + text_offset),
                    textcoords="data",
                    fontsize=7,
                    color=color,
                    ha="center",
                    va="bottom" if is_buy else "top",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "black",
                        "ec": color,
                        "alpha": 0.35,
                    },
                    zorder=7,
                )
            )

        # ---- Entry / Stop / Target lines ----
        entry_level, stop_level, target_level = self._extract_trade_barriers()
        self._set_optional_line(self._entry_line, entry_level)
        self._set_optional_line(self._stop_line, stop_level)
        self._set_optional_line(self._target_line, target_level)

        # ---- ML Signals ----
        self._render_signals(xs, closes)

        # ---- Axis formatting ----
        self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self._suspend_view_tracking = True
        try:
            if prev_xlim and prev_price_ylim and prev_rsi_ylim:
                self.ax_price.set_xlim(prev_xlim)
                self.ax_price.set_ylim(prev_price_ylim)
                self.ax_rsi.set_ylim(prev_rsi_ylim)
            else:
                pad = max((price_max - price_min) * 0.05, 1e-6)
                self.ax_price.set_xlim(xs[0] - width, xs[-1] + width)
                self.ax_price.set_ylim(price_min - pad, price_max + pad)
                self.ax_rsi.set_xlim(xs[0] - width, xs[-1] + width)
                self.ax_rsi.set_ylim(0, 100)

            # ---- Lock-to-live: auto-scroll ----
            if self._lock_to_live and not self._has_user_view and len(xs) > 0:
                self.ax_price.set_xlim(xs[0] - width, xs[-1] + width)
                self.ax_rsi.set_xlim(xs[0] - width, xs[-1] + width)
        finally:
            self._suspend_view_tracking = False

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _clear_artists(self) -> None:
        """Clear all artist data when no candles available."""
        self._wick_collection.set_segments([])
        self._body_collection.set_verts([])
        self._ema_fast_line.set_data([], [])
        self._ema_slow_line.set_data([], [])
        self._ema_ultra_line.set_data([], [])
        self._ema_super_line.set_data([], [])
        self._supertrend_line.set_data([], [])
        self._vwap_line.set_data([], [])
        self._bb_upper_line.set_data([], [])
        self._bb_lower_line.set_data([], [])
        self._bb_fill_collection.set_verts([])
        self._session_high_line.set_data([], [])
        self._session_low_line.set_data([], [])
        self._prevday_high_line.set_data([], [])
        self._prevday_low_line.set_data([], [])
        self._opening_range_high_line.set_data([], [])
        self._opening_range_low_line.set_data([], [])
        self._rsi_line.set_data([], [])
        
        # Clear custom signal scatters
        if hasattr(self, "_signal_scatters"):
            for scat in self._signal_scatters:
                try:
                    scat.remove()
                except Exception:
                    pass
            self._signal_scatters = []
        self._bb_fill_collection.set_verts([])
        self._session_high_line.set_data([], [])
        self._session_low_line.set_data([], [])
        self._prevday_high_line.set_data([], [])
        self._prevday_low_line.set_data([], [])
        self._opening_range_high_line.set_data([], [])
        self._opening_range_low_line.set_data([], [])
        self._rsi_line.set_data([], [])
        self._signal_scatter.set_offsets(np.empty((0, 2)))
        self._volume_bar_collection.set_verts([])
        self._atm_hline.set_visible(False)
        self._atm_label.set_visible(False)
        self._current_price_line.set_visible(False)
        self._current_price_label.set_visible(False)


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