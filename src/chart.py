from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import os
import time

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


_EPOCH_MIN = 946684800.0   # 2000-01-01 UTC
_EPOCH_MAX = 4102444800.0  # 2100-01-01 UTC
_MPL_DATE_MIN = float(mdates.date2num(datetime(2000, 1, 1)))
_MPL_DATE_MAX = float(mdates.date2num(datetime(2100, 1, 1)))
_NIFTY_Y_MIN = 0.01
_NIFTY_Y_MAX = 100_000.0
_CHART_SKIP_LOG_TS: dict[str, float] = {}
_CHART_SKIP_LOG_INTERVAL_SEC = 10.0


def _is_finite_number(x: Any) -> bool:
    try:
        f = float(x)
        return bool(np.isfinite(f))
    except Exception:
        return False


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        f = float(x)
        if not np.isfinite(f):
            return default
        return f
    except Exception:
        return default


def _safe_mpl_date(x: Any, default: float | None = None) -> float | None:
    """Return a matplotlib date number or None when invalid."""
    f = _safe_float(x)
    if f is None:
        return default
    # Unix seconds/ms/ns accidentally passed as x.
    if abs(f) >= 1e8:
        epoch = _normalize_epoch_seconds(f)
        if epoch is None:
            _log_chart_data_skip("safe_mpl_date", "x", x, None, "unix_epoch_not_mpl_date")
            return default
        try:
            f = float(mdates.date2num(datetime.fromtimestamp(epoch)))
        except Exception:
            _log_chart_data_skip("safe_mpl_date", "x", x, None, "epoch_to_mpl_failed")
            return default
    if not (_MPL_DATE_MIN <= f <= _MPL_DATE_MAX):
        _log_chart_data_skip("safe_mpl_date", "x", x, None, "out_of_mpl_date_range")
        return default
    return f


def _valid_chart_y(
    y: Any,
    *,
    near: float | None = None,
    rsi: bool = False,
    allow_negative: bool = False,
) -> bool:
    f = _safe_float(y)
    if f is None or not np.isfinite(f) or abs(f) > 1e9:
        return False
    if rsi:
        return 0.0 <= f <= 100.0
    if allow_negative:
        if abs(f) > _NIFTY_Y_MAX:
            return False
    elif f <= 0 or f < _NIFTY_Y_MIN or f > _NIFTY_Y_MAX:
        return False
    if near is not None:
        ref = _safe_float(near)
        if ref is not None and ((allow_negative and abs(ref) > 0) or (not allow_negative and ref > 0)):
            lhs = abs(f) if allow_negative else f
            rhs = abs(ref) if allow_negative else ref
            ratio = max(lhs, rhs) / max(min(lhs, rhs), 1e-9)
            if ratio > 50.0:
                return False
    return True


def _valid_chart_xy(x: Any, y: Any, *, near_y: float | None = None, allow_negative: bool = False) -> bool:
    xf = _safe_mpl_date(x)
    if xf is None:
        return False
    return _valid_chart_y(y, near=near_y, allow_negative=allow_negative)


def _log_chart_data_skip(source: str, field: str, x: Any, y: Any, reason: str) -> None:
    key = f"{source}:{field}:{reason}"
    now = time.time()
    last = float(_CHART_SKIP_LOG_TS.get(key, 0.0) or 0.0)
    if (now - last) < _CHART_SKIP_LOG_INTERVAL_SEC:
        return
    _CHART_SKIP_LOG_TS[key] = now
    print(
        f"[CHART-DATA-SKIP] source={source} field={field} x={x} y={y} reason={reason}",
        flush=True,
    )


def _finite_plot_coord(value: Any) -> bool:
    f = _safe_float(value)
    return f is not None and abs(f) < 1e12


def _normalize_epoch_seconds(value: Any) -> float | None:
    """Normalize unix seconds/ms/ns to seconds; reject absurd epochs."""
    try:
        f = float(value)
        if not np.isfinite(f):
            return None
        if f > 1e18:
            f = f / 1e9
        elif f > 1e14:
            f = f / 1e6
        elif f > 1e11:
            f = f / 1000.0
        if f < _EPOCH_MIN or f > _EPOCH_MAX:
            return None
        return f
    except Exception:
        return None


def _valid_price(value: Any) -> float | None:
    f = _safe_float(value)
    if f is None or f <= 0 or f > 1e9:
        return None
    return f


def _valid_level_price(value: Any, *, near: float | None = None) -> float | None:
    f = _valid_price(value)
    if f is None:
        return None
    if not _valid_chart_y(f, near=near):
        _log_chart_data_skip("level_price", "y", None, value, "out_of_sane_range")
        return None
    return f


def _valid_mpl_date_x(value: Any) -> bool:
    return _safe_mpl_date(value) is not None


def _filter_xy_pairs(
    xs: list[Any],
    ys: list[Any],
    *,
    source: str,
    near_y: float | None = None,
    allow_negative: bool = False,
) -> tuple[list[float], list[float]]:
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        xf = _safe_mpl_date(x)
        yf = _safe_float(y)
        if xf is None:
            _log_chart_data_skip(source, "x", x, y, "invalid_x")
            continue
        if yf is None or not _valid_chart_y(yf, near=near_y, allow_negative=allow_negative):
            _log_chart_data_skip(source, "y", x, y, "invalid_y")
            continue
        out_x.append(xf)
        out_y.append(yf)
    return out_x, out_y


def _safe_plot(
    ax: Any,
    x: Any,
    y: Any,
    *args: Any,
    source: str = "plot",
    allow_negative: bool = False,
    **kwargs: Any,
) -> Any:
    xs_in = list(x) if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)) else [x]
    ys_in = list(y) if hasattr(y, "__iter__") and not isinstance(y, (str, bytes)) else [y]
    near = _safe_float(ys_in[-1]) if ys_in else None
    xs, ys = _filter_xy_pairs(xs_in, ys_in, source=source, near_y=near, allow_negative=allow_negative)
    if not xs or not ys:
        return None
    try:
        return ax.plot(xs, ys, *args, **kwargs)
    except Exception as exc:
        _log_chart_data_skip(source, "plot", xs[:1], ys[:1], f"plot_failed:{type(exc).__name__}")
        return None


def _safe_scatter(
    ax: Any,
    x: Any,
    y: Any,
    *args: Any,
    source: str = "scatter",
    allow_negative: bool = False,
    **kwargs: Any,
) -> Any:
    xs_in = list(x) if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)) else [x]
    ys_in = list(y) if hasattr(y, "__iter__") and not isinstance(y, (str, bytes)) else [y]
    near = _safe_float(ys_in[-1]) if ys_in else None
    xs, ys = _filter_xy_pairs(xs_in, ys_in, source=source, near_y=near, allow_negative=allow_negative)
    if not xs or not ys:
        return None
    try:
        return ax.scatter(xs, ys, *args, **kwargs)
    except Exception as exc:
        _log_chart_data_skip(source, "scatter", xs[:1], ys[:1], f"scatter_failed:{type(exc).__name__}")
        return None


def _safe_axvline(ax: Any, x: Any, *args: Any, source: str = "axvline", **kwargs: Any) -> Any:
    xf = _safe_mpl_date(x)
    if xf is None:
        _log_chart_data_skip(source, "x", x, None, "invalid_x")
        return None
    try:
        return ax.axvline(xf, *args, **kwargs)
    except Exception as exc:
        _log_chart_data_skip(source, "x", x, None, f"axvline_failed:{type(exc).__name__}")
        return None


def _safe_axhline(ax: Any, y: Any, *args: Any, source: str = "axhline", **kwargs: Any) -> Any:
    yf = _safe_float(y)
    allow_negative = bool(kwargs.pop("_allow_negative", False))
    if yf is None or not _valid_chart_y(yf, allow_negative=allow_negative):
        _log_chart_data_skip(source, "y", None, y, "invalid_y")
        return None
    try:
        return ax.axhline(yf, *args, **kwargs)
    except Exception as exc:
        _log_chart_data_skip(source, "y", None, y, f"axhline_failed:{type(exc).__name__}")
        return None


def _safe_annotate(ax: Any, text: str, xy: Any, *args: Any, source: str = "annotate", **kwargs: Any) -> Any:
    try:
        x_raw, y_raw = xy[0], xy[1]
    except Exception:
        _log_chart_data_skip(source, "xy", xy, None, "bad_xy_tuple")
        return None
    near = _safe_float(y_raw)
    allow_negative = bool(kwargs.pop("_allow_negative", False))
    if not _valid_chart_xy(x_raw, y_raw, near_y=near, allow_negative=allow_negative):
        _log_chart_data_skip(source, "xy", x_raw, y_raw, "invalid_xy")
        return None
    xytext = kwargs.get("xytext")
    if xytext is not None:
        try:
            xt_raw, yt_raw = xytext[0], xytext[1]
            if str(kwargs.get("textcoords", "data")).lower() in {"data", "offset points", "offset pixels"}:
                if not _valid_chart_xy(xt_raw, yt_raw, near_y=near, allow_negative=allow_negative):
                    _log_chart_data_skip(source, "xytext", xt_raw, yt_raw, "invalid_xytext")
                    return None
        except Exception:
            _log_chart_data_skip(source, "xytext", xytext, None, "bad_xytext_tuple")
            return None
    try:
        xf = _safe_mpl_date(x_raw)
        yf = _safe_float(y_raw)
        return ax.annotate(text, (xf, yf), *args, **kwargs)
    except Exception as exc:
        _log_chart_data_skip(source, "annotate", x_raw, y_raw, f"annotate_failed:{type(exc).__name__}")
        return None


_tight_layout_done: dict[int, bool] = {}


def _axis_tick_labels(ax: Any) -> list[Any]:
    labels: list[Any] = []
    for getter_name in ("get_xticklabels", "get_yticklabels"):
        getter = getattr(ax, getter_name, None)
        if callable(getter):
            try:
                labels.extend(list(getter() or []))
            except Exception:
                continue
    return labels


def _safe_tight_layout(fig, *, force: bool = False) -> None:
    """Apply tight_layout at most once per figure unless forced."""
    fid = id(fig)
    if not force and _tight_layout_done.get(fid):
        return
    try:
        fig.tight_layout()
        _tight_layout_done[fid] = True
    except Exception:
        try:
            fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
            _tight_layout_done[fid] = True
        except Exception:
            pass


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
        self._last_candle_batch_key: tuple[int, str] | None = None
        self._last_candle_ts_key: str | None = None
        self._overlay_last_render_ts: float = 0.0
        self._overlay_refresh_sec: float = 3.0
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
        # [CHART] Default to 500 candles to bound memory. Users can override via set_max_candles(None) for unlimited.
        self._max_candles: Optional[int] = 500
        # [CHART] Batch-update flag — suppresses intermediate _render() calls
        # so callers can make multiple state changes with a single final render.
        self._batch_updates: bool = False
        # [STOP-BOT] Paused flag to stop renders while preserving visible candles
        self._paused: bool = False
        # [CHART] Throttle + UI-thread render scheduling
        self._render_pending: bool = False
        self._render_after_id = None
        self._last_render_ts: float = 0.0
        self._render_throttle_sec: float = 0.25
        self._last_render_xs: list[float] = []
        self._last_render_closes: list[float] = []
        self._last_render_opens: list[float] = []
        self._last_render_highs: list[float] = []
        self._last_render_lows: list[float] = []
        self._advanced_overlays_disabled: bool = False

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
        self._cross_hair_vert = _safe_axvline(
            self.ax_price, _MPL_DATE_MIN, color="#64748b", linestyle=":", linewidth=0.8, visible=False, zorder=10,
            source="crosshair_vert",
        ) or self.ax_price.axvline(_MPL_DATE_MIN, color="#64748b", linestyle=":", linewidth=0.8, visible=False, zorder=10)
        self._cross_hair_horiz = _safe_axhline(
            self.ax_price, 24000.0, color="#64748b", linestyle=":", linewidth=0.8, visible=False, zorder=10,
            source="crosshair_horiz",
        ) or self.ax_price.axhline(24000.0, color="#64748b", linestyle=":", linewidth=0.8, visible=False, zorder=10)
        
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
            if not _valid_mpl_date_x(x) or not _valid_chart_y(y, near=_safe_float(self._candles[-1].close if hasattr(self._candles[-1], "close") else None)):
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
            ha="center", va="center", zorder=0, weight="bold",
            transform=self.fig.transFigure,
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

    def _normalize_incoming_candles(self, candles_list: Any) -> list[Any]:
        """Normalize DataFrame, dict rows, or Candle objects into a sorted list."""
        if candles_list is None:
            return []
        rows: list[Any]
        try:
            if hasattr(candles_list, "empty") and hasattr(candles_list, "iterrows"):
                if bool(getattr(candles_list, "empty", True)):
                    return []
                rows = []
                cols = {str(c).lower(): c for c in list(getattr(candles_list, "columns", []))}
                time_col = next((cols[k] for k in ("time", "timestamp", "datetime", "date") if k in cols), None)
                for _, row in candles_list.iterrows():
                    try:
                        t = row[time_col] if time_col is not None else None
                        o = row.get(cols.get("open", "open")) if hasattr(row, "get") else row[cols.get("open", "open")]
                        h = row.get(cols.get("high", "high")) if hasattr(row, "get") else row[cols.get("high", "high")]
                        l = row.get(cols.get("low", "low")) if hasattr(row, "get") else row[cols.get("low", "low")]
                        c = row.get(cols.get("close", "close")) if hasattr(row, "get") else row[cols.get("close", "close")]
                        v = row.get(cols.get("volume", "volume")) if hasattr(row, "get") else row.get(cols.get("volume", "volume"), 0.0)
                        rows.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
                    except Exception:
                        continue
            else:
                rows = list(candles_list)
        except Exception as exc:
            print(f"[CANDLES-NORMALIZE][ERROR] {type(exc).__name__}: {exc}", flush=True)
            return []

        def _get_time(c: Any):
            if hasattr(c, "time"):
                return getattr(c, "time")
            if hasattr(c, "timestamp"):
                return getattr(c, "timestamp")
            if isinstance(c, dict):
                return c.get("time") or c.get("timestamp") or c.get("datetime") or c.get("date")
            return None

        valid: list[Any] = []
        for c in rows:
            try:
                dt = self._candle_dt(c) if hasattr(self, "_candle_dt") else None
                if dt is None:
                    # Allow pre-init normalization during construction.
                    t = _get_time(c)
                    if t is None:
                        continue
                    epoch = _normalize_epoch_seconds(t)
                    if epoch is None and not isinstance(t, datetime):
                        _log_chart_data_skip("normalize_incoming", "time", t, None, "invalid_timestamp")
                        continue
                if isinstance(c, dict):
                    o, h, l, cl = c.get("open"), c.get("high"), c.get("low"), c.get("close")
                    if o is None or h is None or l is None or cl is None:
                        continue
                    if _valid_price(cl) is None:
                        _log_chart_data_skip("normalize_incoming", "close", None, cl, "invalid_price")
                        continue
                    valid.append(c)
                else:
                    if _valid_price(getattr(c, "close", None)) is None:
                        _log_chart_data_skip("normalize_incoming", "close", None, getattr(c, "close", None), "invalid_price")
                        continue
                    valid.append(c)
            except Exception:
                continue

        try:
            valid = sorted(valid, key=_get_time)
        except Exception as sort_err:
            print(f"[CHART-DATA] Warning: Could not sort candles: {sort_err}", flush=True)

        print(
            f"[CANDLES-NORMALIZE] rows={len(valid)} last={_get_time(valid[-1]) if valid else 'N/A'}",
            flush=True,
        )
        return valid

    def _request_render(self) -> None:
        """Schedule a render on the Tk main thread (throttled)."""
        if getattr(self, "_paused", False):
            return
        try:
            if self.parent is None or not self.parent.winfo_exists():
                return
        except Exception:
            return
        now = time.time()
        if (now - float(getattr(self, "_last_render_ts", 0.0) or 0.0)) < float(
            getattr(self, "_render_throttle_sec", 0.25) or 0.25
        ):
            if not getattr(self, "_render_pending", False):
                self._render_pending = True
                try:
                    delay_ms = int(float(getattr(self, "_render_throttle_sec", 0.25) or 0.25) * 1000)
                    if self._render_after_id is not None:
                        try:
                            self.parent.after_cancel(self._render_after_id)
                        except Exception:
                            pass
                    self._render_after_id = self.parent.after(max(delay_ms, 50), self._render_on_ui_thread)
                except Exception:
                    self._render_pending = False
                    self._render()
            return
        if getattr(self, "_render_pending", False):
            return
        self._render_pending = True
        try:
            if self._render_after_id is not None:
                try:
                    self.parent.after_cancel(self._render_after_id)
                except Exception:
                    pass
            self._render_after_id = self.parent.after(0, self._render_on_ui_thread)
        except Exception:
            self._render_pending = False
            self._render()

    def _render_on_ui_thread(self) -> None:
        self._render_after_id = None
        self._render_pending = False
        try:
            self._render()
        except Exception as exc:
            import traceback
            print(f"[CHART-RENDER][ERROR] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    def push_candles(self, candles_list: list[Any]) -> None:
        """Called periodically by the Strategy to feed history into the chart."""
        # [STOP-BOT] Skip updates when paused (preserves visible candles)
        if getattr(self, "_paused", False):
            return

        print(f"[CHART-DATA] push_candles called rows={len(candles_list) if candles_list is not None else 0}", flush=True)
        candles = self._normalize_incoming_candles(candles_list)
        if not candles:
            print("[CHART-DATA][EMPTY] no candle rows received", flush=True)
            self._raw_candles = []
            self._candles = []
            self._dirty = True
            self._request_render()
            # Unit tests / headless callers may not pump the Tk loop.
            if self._dirty:
                self._render_on_ui_thread()
            return
        try:
            last = candles[-1]
            last_time = str(getattr(last, "time", getattr(last, "timestamp", "")) or "")
            batch_key = (len(candles), last_time)
            has_rendered_candles = bool(getattr(self, "_last_render_xs", None)) and bool(
                getattr(self, "_last_render_closes", None)
            )
            if has_rendered_candles and batch_key == getattr(self, "_last_candle_batch_key", None):
                return
            if has_rendered_candles and last_time and last_time == getattr(self, "_last_candle_ts_key", None):
                return
            self._last_candle_batch_key = batch_key
            self._last_candle_ts_key = last_time
        except Exception:
            pass

        try:
            first = candles[0]
            last = candles[-1]
            first_time = getattr(first, "time", getattr(first, "timestamp", "N/A"))
            last_time = getattr(last, "time", getattr(last, "timestamp", "N/A"))
            if not hasattr(self, "_chart_data_log_ts") or (time.time() - float(getattr(self, "_chart_data_log_ts", 0.0) or 0.0)) > 10.0:
                print(
                    f"[CHART-DATA] first={first_time} last={last_time} count={len(candles)}",
                    flush=True,
                )
                self._chart_data_log_ts = time.time()
        except Exception as e:
            print(f"[CHART-DATA] Error logging candle data: {e}", flush=True)

        self._raw_candles = candles
        limit = self._max_candles if self._max_candles is not None else 500
        if len(self._raw_candles) > limit:
            self._raw_candles = self._raw_candles[-limit:]
        self._dirty = True
        self._request_render()
        # Unit tests / headless callers may not pump the Tk loop.
        if self._dirty:
            self._render_on_ui_thread()

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
            self._atm_strike = _valid_level_price(atm_strike, near=_safe_float(spot))
        self._dirty = True
        # [CHART] Removed duplicate _render() call that wasted CPU cycles.
        self._render()

    def set_session_levels(self, high: Optional[float], low: Optional[float], open_price: Optional[float] = None) -> None:
        """Set session high/low horizontal lines."""
        ref = _safe_float(high) or _safe_float(low)
        self._session_high = _valid_level_price(high, near=ref) if high is not None else None
        self._session_low = _valid_level_price(low, near=ref) if low is not None else None
        if open_price is not None:
            self._session_open = float(open_price)
        self._dirty = True
        self._render()

    def set_prevday_levels(self, high: Optional[float], low: Optional[float], open_price: Optional[float] = None) -> None:
        """Set previous day high/low horizontal lines."""
        ref = _safe_float(high) or _safe_float(low)
        self._prevday_high = _valid_level_price(high, near=ref) if high is not None else None
        self._prevday_low = _valid_level_price(low, near=ref) if low is not None else None
        if open_price is not None:
            self._prevday_open = float(open_price)
        self._dirty = True
        self._render()

    def set_atm_strike(self, strike: Optional[float]) -> None:
        """Set ATM strike for vertical marker."""
        self._atm_strike = _valid_level_price(strike, near=_safe_float(self._market_info.get("spot"))) if strike is not None else None
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

    def set_max_candles(self, count: Optional[int]) -> None:
        """Limit rendered candles to the most recent `count`. Pass None for unlimited."""
        if count is None:
            self._max_candles = None
        else:
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

    def update_from_snapshot(
        self,
        candles: Optional[list[Any]] = None,
        spot: Optional[float] = None,
        futures: Optional[float] = None,
        atm_strike: Optional[float] = None,
        iv: Optional[float] = None,
        pcr: Optional[float] = None,
        session_high: Optional[float] = None,
        session_low: Optional[float] = None,
        prevday_high: Optional[float] = None,
        prevday_low: Optional[float] = None,
        prediction_rows: Optional[list[dict[str, Any]]] = None,
        trade_event_rows: Optional[list[dict[str, Any]]] = None,
        regime_label: Optional[str] = None,
        market_status: Optional[str] = None,
    ) -> None:
        """Atomically apply all snapshot fields and render once.

        [CHART] Batches all state mutations so the expensive _render()
        (matplotlib canvas draw) is called exactly once instead of up to
        7 times per refresh cycle. This dramatically reduces CPU/GPU
        overhead during long-hour UI operation.
        """
        self._batch_updates = True
        try:
            if candles is not None:
                self.push_candles(candles)
            self._market_info["spot"] = spot
            self._market_info["futures"] = futures
            self._market_info["iv"] = iv
            self._market_info["pcr"] = pcr
            self._market_info["regime_label"] = regime_label or self._market_info.get("regime_label", "")
            self._market_info["market_status"] = market_status or self._market_info.get("market_status", "")
            ref_spot = _safe_float(spot)
            if atm_strike is not None:
                self._atm_strike = _valid_level_price(atm_strike, near=ref_spot)
            if session_high is not None:
                self._session_high = _valid_level_price(session_high, near=ref_spot)
            if session_low is not None:
                self._session_low = _valid_level_price(session_low, near=ref_spot)
            if prevday_high is not None:
                self._prevday_high = _valid_level_price(prevday_high, near=ref_spot)
            if prevday_low is not None:
                self._prevday_low = _valid_level_price(prevday_low, near=ref_spot)
            if prediction_rows is not None:
                self.load_prediction_rows(prediction_rows)
            if trade_event_rows is not None:
                self.load_trade_event_rows(trade_event_rows)
            # Mark dirty and render ONCE after all updates
            self._dirty = True
            self._request_render()
        finally:
            self._batch_updates = False

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
                ts_val = row.get("ts") or row.get("timestamp")
                if ts_val is None:
                    continue
                x: float | None = None
                if isinstance(ts_val, datetime):
                    if ts_val.year < 2000 or ts_val.year > 2100:
                        print(f"[CHART-OVERLAY-SKIP] reason=invalid_datetime x={ts_val} y=-", flush=True)
                        continue
                    ts_num = mdates.date2num(ts_val)
                    x = x_map.get(ts_num)
                    if x is None:
                        x = ts_num
                else:
                    epoch = _normalize_epoch_seconds(ts_val)
                    if epoch is None:
                        print(f"[CHART-OVERLAY-SKIP] reason=invalid_timestamp x={ts_val} y=-", flush=True)
                        continue
                    dt = datetime.fromtimestamp(epoch)
                    ts_num = mdates.date2num(dt)
                    x = x_map.get(ts_num)
                    if x is None:
                        x = ts_num

                x = _safe_mpl_date(x)
                if x is None or not _valid_mpl_date_x(x):
                    _log_chart_data_skip("render_signals", "x", ts_val, None, "invalid_mpl_x")
                    continue

                idx = int(np.argmin(np.abs(np.array(xs) - x))) if len(xs) > 0 else 0
                y = _safe_float(closes[min(idx, len(closes) - 1)])
                if y is None or not _valid_chart_y(y, near=y):
                    _log_chart_data_skip("render_signals", "y", x, y, "invalid_y")
                    continue

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
            near = _safe_float(g["y"][-1]) if g["y"] else None
            sx, sy = _filter_xy_pairs(g["x"], g["y"], source=f"render_signals:{key}", near_y=near)
            if not sx:
                continue
            sizes = g["sizes"][-len(sx):] if len(g["sizes"]) >= len(sx) else g["sizes"]
            if key == "blocked":
                scat = _safe_scatter(
                    self.ax_price, sx, sy, s=sizes,
                    marker=g["marker"], facecolors="none",
                    edgecolors=g["color"], linewidths=1.5,
                    zorder=7, alpha=0.9, source=f"render_signals:{key}",
                )
            else:
                scat = _safe_scatter(
                    self.ax_price, sx, sy, s=sizes,
                    marker=g["marker"], c=g["color"],
                    edgecolors="black", linewidths=0.5,
                    zorder=7, alpha=0.9, source=f"render_signals:{key}",
                )
            if scat is not None:
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
        epoch = _normalize_epoch_seconds(timestamp)
        if epoch is None:
            return None
        target_x = mdates.date2num(datetime.fromtimestamp(epoch))
        if not _valid_mpl_date_x(float(target_x)):
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
        ref = _safe_float(self._last_render_closes[-1]) if getattr(self, "_last_render_closes", None) else None
        y = _valid_level_price(level, near=ref)
        if y is None:
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
        if t is None and isinstance(c, dict):
            t = c.get("time") or c.get("timestamp") or c.get("ts")
        if t is None:
            return None
        if isinstance(t, datetime):
            if t.year < 2000 or t.year > 2100:
                return None
            return t
        try:
            if hasattr(t, "to_pydatetime"):
                dt = t.to_pydatetime()
                if dt.year < 2000 or dt.year > 2100:
                    return None
                return dt
        except Exception:
            pass
        epoch = _normalize_epoch_seconds(t)
        if epoch is None:
            return None
        try:
            return datetime.fromtimestamp(epoch)
        except Exception:
            return None

    def _sanitize_candles_for_render(self, candles: list[Any]) -> list[Any]:
        raw_n = len(candles or [])
        valid: list[Any] = []
        dropped_bad_ts = dropped_bad_price = 0
        seen_ts: set[float] = set()
        min_ts = max_ts = None
        for c in candles or []:
            dt = self._candle_dt(c)
            if dt is None:
                dropped_bad_ts += 1
                continue
            ts_key = float(dt.timestamp())
            if ts_key in seen_ts:
                dropped_bad_ts += 1
                continue
            o = _valid_price(getattr(c, "open", None) if not isinstance(c, dict) else c.get("open"))
            h = _valid_price(getattr(c, "high", None) if not isinstance(c, dict) else c.get("high"))
            l = _valid_price(getattr(c, "low", None) if not isinstance(c, dict) else c.get("low"))
            cl = _valid_price(getattr(c, "close", None) if not isinstance(c, dict) else c.get("close"))
            if cl is None:
                dropped_bad_price += 1
                continue
            if o is None:
                o = cl
            if h is None:
                h = max(o, cl)
            if l is None:
                l = min(o, cl)
            seen_ts.add(ts_key)
            min_ts = dt if min_ts is None or dt < min_ts else min_ts
            max_ts = dt if max_ts is None or dt > max_ts else max_ts
            valid.append(c)
        valid.sort(key=lambda row: self._candle_dt(row).timestamp() if self._candle_dt(row) else 0.0)
        print(
            f"[CHART-SANITIZE] raw={raw_n} valid={len(valid)} dropped_bad_ts={dropped_bad_ts} "
            f"dropped_bad_price={dropped_bad_price} min_ts={min_ts} max_ts={max_ts}",
            flush=True,
        )
        return valid

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

    def safe_ax_text(self, ax: Any, x: Any, y: Any, text: str, *, transform: Any = None, **kwargs: Any) -> Any:
        if transform is not None:
            try:
                xf = float(x)
                yf = float(y)
                if not (np.isfinite(xf) and np.isfinite(yf) and -5.0 <= xf <= 5.0 and -5.0 <= yf <= 5.0):
                    _log_chart_data_skip("safe_ax_text", "axes_xy", x, y, "invalid_axes_coordinate")
                    return None
            except Exception:
                _log_chart_data_skip("safe_ax_text", "axes_xy", x, y, "invalid_axes_coordinate")
                return None
            try:
                return ax.text(x, y, text, transform=transform, **kwargs)
            except Exception as exc:
                _log_chart_data_skip("safe_ax_text", "axes_xy", x, y, f"text_failed:{type(exc).__name__}")
                return None
        near = _safe_float(y)
        if not _valid_chart_xy(x, y, near_y=near):
            _log_chart_data_skip("safe_ax_text", "data_xy", x, y, "invalid_data_coordinate")
            return None
        try:
            xf = _safe_mpl_date(x)
            yf = _safe_float(y)
            return ax.text(xf, yf, text, **kwargs)
        except Exception as exc:
            _log_chart_data_skip("safe_ax_text", "data_xy", x, y, f"text_failed:{type(exc).__name__}")
            return None

    def _sanitize_draw_artists(self) -> None:
        for ax in (self.ax_price, self.ax_rsi):
            for txt in list(getattr(ax, "texts", []) or []):
                try:
                    x, y = txt.get_position()
                    transform = txt.get_transform()
                    if transform is ax.transAxes:
                        if not _is_finite_number(x) or not _is_finite_number(y) or not (-5.0 <= float(x) <= 5.0 and -5.0 <= float(y) <= 5.0):
                            _log_chart_data_skip("sanitize_draw", "axes_text", x, y, "invalid_axes_artist")
                            txt.set_visible(False)
                    else:
                        label = str(txt.get_text() or "").strip()
                        if not label:
                            continue
                        near = _safe_float(self._last_render_closes[-1]) if getattr(self, "_last_render_closes", None) else None
                        if not _valid_chart_xy(x, y, near_y=near):
                            _log_chart_data_skip("sanitize_draw", "data_text", x, y, "invalid_data_artist")
                            txt.set_visible(False)
                except Exception:
                    try:
                        txt.set_visible(False)
                    except Exception:
                        pass
            for ann in list(getattr(ax, "annotations", []) or []):
                try:
                    x, y = ann.xy
                    near = _safe_float(y)
                    if not _valid_chart_xy(x, y, near_y=near):
                        _log_chart_data_skip("sanitize_draw", "annotation", x, y, "invalid_annotation")
                        ann.set_visible(False)
                except Exception:
                    try:
                        ann.set_visible(False)
                    except Exception:
                        pass
            for label in _axis_tick_labels(ax):
                try:
                    pos = label.get_position()
                    if not (_is_finite_number(pos[0]) and _is_finite_number(pos[1]) and abs(float(pos[0])) < 1e6 and abs(float(pos[1])) < 1e6):
                        label.set_visible(False)
                except Exception:
                    try:
                        label.set_visible(False)
                    except Exception:
                        pass
        fig = getattr(self, "fig", None)
        if fig is not None:
            for text in list(getattr(fig, "texts", []) or []):
                try:
                    x, y = text.get_position()
                    if not (_is_finite_number(x) and _is_finite_number(y)):
                        text.set_visible(False)
                except Exception:
                    try:
                        text.set_visible(False)
                    except Exception:
                        pass

    def _sanitize_text_artists(self) -> None:
        self._sanitize_draw_artists()

    def _clamp_axis_limits(self) -> None:
        xs_cache = getattr(self, "_last_render_xs", None) or []
        ys_cache = getattr(self, "_last_render_closes", None) or []
        if not xs_cache or not ys_cache:
            return
        try:
            xs = [float(x) for x in xs_cache if _valid_mpl_date_x(x)]
            ys = [float(y) for y in ys_cache if _valid_chart_y(y, near=_safe_float(y))]
            if not xs or not ys:
                return
            width = max((xs[-1] - xs[0]) / max(len(xs), 1) * 0.7, 1.0 / (24 * 60) * 0.7)
            pad = max((max(ys) - min(ys)) * 0.08, 1e-6)
            self.ax_price.set_xlim(xs[0] - width, xs[-1] + width)
            self.ax_price.set_ylim(min(ys) - pad, max(ys) + pad)
            self.ax_rsi.set_xlim(xs[0] - width, xs[-1] + width)
            self.ax_rsi.set_ylim(0, 100)
        except Exception as exc:
            _log_chart_data_skip("clamp_axis_limits", "limits", None, None, f"failed:{type(exc).__name__}")

    def _hide_unsafe_artists_for_draw(self) -> None:
        for ax in (self.ax_price, self.ax_rsi):
            for text in list(getattr(ax, "texts", []) or []):
                try:
                    text.set_visible(False)
                except Exception:
                    pass
            for ann in list(getattr(ax, "annotations", []) or []):
                try:
                    ann.set_visible(False)
                except Exception:
                    pass
            legend_getter = getattr(ax, "get_legend", None)
            if callable(legend_getter):
                try:
                    legend = legend_getter()
                    if legend is not None:
                        legend.set_visible(False)
                except Exception:
                    pass
            for label in _axis_tick_labels(ax):
                try:
                    label.set_visible(False)
                except Exception:
                    pass
        fig = getattr(self, "fig", None)
        if fig is not None:
            for text in list(getattr(fig, "texts", []) or []):
                try:
                    text.set_visible(False)
                except Exception:
                    pass

    def _render_emergency_basic_candles(self) -> None:
        """Last-resort render: sanitized close prices only."""
        xs = list(getattr(self, "_last_render_xs", None) or [])
        closes = list(getattr(self, "_last_render_closes", None) or [])
        paired = [(x, c) for x, c in zip(xs, closes) if _valid_mpl_date_x(x) and _valid_chart_y(c, near=c)]
        if not paired:
            self.ax_price.clear()
            self.ax_rsi.clear()
            self._style_axes()
            self.safe_ax_text(
                self.ax_price, 0.5, 0.5, "No valid candle data",
                transform=self.ax_price.transAxes,
                ha="center", va="center", fontsize=12, color="#94a3b8",
            )
            return
        px, py = zip(*paired)
        self.ax_price.clear()
        self.ax_rsi.clear()
        self._style_axes()
        _safe_plot(self.ax_price, px, py, linewidth=1.2, color="#38bdf8", source="emergency_basic")
        self.safe_ax_text(
            self.ax_price,
            0.5,
            0.92,
            "Advanced overlays disabled due invalid chart coordinate",
            transform=self.ax_price.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#f59e0b",
        )
        self._advanced_overlays_disabled = True
        self._clamp_axis_limits()

    def _safe_draw_idle(self, *, fallback_basic: bool = False) -> None:
        try:
            self._clamp_axis_limits()
            self._sanitize_draw_artists()
            self.canvas.draw()
            self._advanced_overlays_disabled = False
        except (TypeError, ValueError, OverflowError) as exc:
            print(f"[CHART-RENDER-ERROR] type={type(exc).__name__} message={exc}", flush=True)
            if fallback_basic:
                try:
                    self._hide_unsafe_artists_for_draw()
                    self.canvas.draw()
                    print("[CHART-RENDER] fallback=basic_candles_only", flush=True)
                    return
                except Exception:
                    pass
                try:
                    self._render_emergency_basic_candles()
                    self.canvas.draw()
                    print("[CHART-RENDER] fallback=emergency_basic_candles", flush=True)
                    return
                except Exception as fallback_exc:
                    print(f"[CHART-RENDER-ERROR] fallback_failed={type(fallback_exc).__name__}:{fallback_exc}", flush=True)
            raise

    def _render(self) -> None:
        # [STOP-BOT] Skip render when paused (preserves visible candles)
        if getattr(self, "_paused", False):
            return
        # [CHART] Skip render during batch mode — caller is batching multiple
        # state changes and will trigger a single render at the end.
        if self._batch_updates:
            return
        if not self._dirty:
            return
        self._dirty = False

        # [CHART-RENDER] Log render start
        raw_count = len(self._raw_candles) if hasattr(self, '_raw_candles') and self._raw_candles else 0
        print(f"[CHART-RENDER] starting render raw_candles={raw_count}")

        if not hasattr(self, "_raw_candles") or not self._raw_candles:
            self._raw_candles = self._candles

        if not self._raw_candles:
            self._clear_artists()
            self.ax_price.clear()
            self.ax_rsi.clear()
            self._style_axes()
            self._init_artists()
            # [CHART-FALLBACK] Show message when no candle data
            print("[CHART-RENDER][EMPTY] showing fallback message", flush=True)
            self.safe_ax_text(self.ax_price, 0.5, 0.5, "No candle data available",
                              transform=self.ax_price.transAxes,
                              ha='center', va='center', fontsize=12, color='#94a3b8')
            self._safe_draw_idle()
            self._last_render_ts = time.time()
            return

        self._raw_candles = self._sanitize_candles_for_render(self._raw_candles)
        self._candles = self._resample_candles_to_timeframe(self._raw_candles, self.timeframe)

        if not self._candles:
            self._clear_artists()
            # [CHART-FALLBACK] Show message when resampling returns empty
            self.safe_ax_text(self.ax_price, 0.5, 0.5, "No candle data available",
                              transform=self.ax_price.transAxes,
                              ha='center', va='center', fontsize=12, color='#94a3b8')
            self._safe_draw_idle()
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
            o = _valid_price(getattr(c, "open", None) if not isinstance(c, dict) else c.get("open"))
            h = _valid_price(getattr(c, "high", None) if not isinstance(c, dict) else c.get("high"))
            l = _valid_price(getattr(c, "low", None) if not isinstance(c, dict) else c.get("low"))
            cl = _valid_price(getattr(c, "close", None) if not isinstance(c, dict) else c.get("close"))
            if cl is None:
                continue
            if o is None:
                o = cl
            if h is None:
                h = max(o, cl)
            if l is None:
                l = min(o, cl)
            try:
                v = float(getattr(c, "volume", 0.0) if not isinstance(c, dict) else c.get("volume", 0.0))
            except Exception:
                v = 0.0
            dts.append(dt)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(cl)
            volumes.append(v)

        if not closes:
            self._clear_artists()
            self._safe_draw_idle()
            return

        # [CHART-RENDER] Use ax.clear() pattern: clear axes, re-init artists, redraw
        self.ax_price.clear()
        self.ax_rsi.clear()
        self._style_axes()
        self._init_artists()

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

        xs_raw = mdates.date2num(dts)
        paired = [
            (float(x), o, h, l, c)
            for x, o, h, l, c in zip(xs_raw, opens, highs, lows, closes)
            if _valid_mpl_date_x(x)
        ]
        if not paired:
            self._clear_artists()
            self.safe_ax_text(
                self.ax_price, 0.5, 0.5, "Invalid candle timestamps",
                transform=self.ax_price.transAxes,
                ha="center", va="center", fontsize=12, color="#94a3b8",
            )
            self._safe_draw_idle(fallback_basic=True)
            return
        xs, opens, highs, lows, closes = map(list, zip(*paired))
        self._last_render_xs = list(xs)
        self._last_render_opens = list(opens)
        self._last_render_highs = list(highs)
        self._last_render_lows = list(lows)
        self._last_render_closes = list(closes)

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

        def _line_data(ys: list[Optional[float]], *, rsi: bool = False) -> tuple[list[float], list[float]]:
            near = _safe_float(closes[-1]) if closes else None
            xs2: list[float] = []
            ys2: list[float] = []
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                yf = _safe_float(y)
                if yf is None or not _valid_chart_y(yf, near=near, rsi=rsi):
                    _log_chart_data_skip("line_data", "y", x, y, "invalid_indicator_y")
                    continue
                if not _valid_mpl_date_x(x):
                    _log_chart_data_skip("line_data", "x", x, y, "invalid_indicator_x")
                    continue
                xs2.append(float(x))
                ys2.append(yf)
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
        self._rsi_line.set_data(*_line_data(rsi_s, rsi=True))

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
            if _valid_mpl_date_x(xs[-1]) and _finite_plot_coord(atm_price):
                self._atm_label.set_position((xs[-1], atm_price))
                self._atm_label.set_text(f"  ATM {self._atm_strike:.0f}")
                self._atm_label.set_visible(True)
            else:
                print(f"[CHART-OVERLAY-SKIP] reason=invalid_coordinate x={xs[-1]} y={atm_price}", flush=True)
                self._atm_label.set_visible(False)
        else:
            self._atm_hline.set_visible(False)
            self._atm_label.set_visible(False)

        # ---- Current Price Line ----
        if closes:
            last_close = float(closes[-1])
            self._current_price_line.set_ydata([last_close, last_close])
            self._current_price_line.set_xdata([xs[0], xs[-1]])
            self._current_price_line.set_visible(True)
            if _valid_mpl_date_x(xs[-1]) and _finite_plot_coord(last_close):
                self._current_price_label.set_position((xs[-1], last_close))
                self._current_price_label.set_text(f"  LTP {last_close:.2f}")
                self._current_price_label.set_visible(True)
            else:
                print(f"[CHART-OVERLAY-SKIP] reason=invalid_coordinate x={xs[-1]} y={last_close}", flush=True)
                self._current_price_label.set_visible(False)
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
            epoch = _normalize_epoch_seconds(m.ts)
            if epoch is None:
                print(f"[CHART-OVERLAY-SKIP] reason=invalid_xy trade_id={m.trade_id} x={m.ts} y=n/a", flush=True)
                continue
            x = mdates.date2num(datetime.fromtimestamp(epoch))
            if not _valid_mpl_date_x(float(x)):
                print(f"[CHART-OVERLAY-SKIP] reason=invalid_xy trade_id={m.trade_id} x={x} y=n/a", flush=True)
                continue
            price_y = self._nearest_price_at_timestamp(m.ts, xs, closes)
            if price_y is None or not _finite_plot_coord(price_y):
                print(f"[CHART-OVERLAY-SKIP] reason=invalid_xy trade_id={m.trade_id} x={x} y={price_y}", flush=True)
                continue
            is_buy = str(m.side).upper() == "BUY"
            option_type = str(m.option_type).upper()
            color = "deepskyblue" if option_type == "CE" else ("gold" if option_type == "PE" else ("lime" if is_buy else "red"))
            marker = "^" if is_buy else "v"
            y_offset = price_span * (0.018 if is_buy else -0.018)
            text_offset = price_span * (0.05 if is_buy else -0.05)
            marker_y = price_y + y_offset
            text_y = price_y + text_offset
            if not (_finite_plot_coord(x) and _finite_plot_coord(marker_y) and _finite_plot_coord(text_y)):
                print(f"[CHART-OVERLAY-SKIP] reason=invalid_xy trade_id={m.trade_id} x={x} y={marker_y}", flush=True)
                continue
            scatter = _safe_scatter(
                self.ax_price,
                [x],
                [marker_y],
                marker=marker,
                s=70,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=6,
                alpha=0.95,
                source="trade_marker",
            )
            if scatter is not None:
                self._marker_lines.append(scatter)
            ann = _safe_annotate(
                self.ax_price,
                m.label or m.event.title(),
                (x, marker_y),
                xytext=(x, text_y),
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
                source="trade_marker",
            )
            if ann is not None:
                self._marker_texts.append(ann)

        # ---- Entry / Stop / Target lines ----
        entry_level, stop_level, target_level = self._extract_trade_barriers()
        self._set_optional_line(self._entry_line, entry_level)
        self._set_optional_line(self._stop_line, stop_level)
        self._set_optional_line(self._target_line, target_level)

        # ---- ML Signals ----
        overlays_ok = True
        try:
            self._render_signals(xs, closes)
        except Exception as overlay_exc:
            overlays_ok = False
            print(f"[CHART-OVERLAY-SKIP] reason=render_exception error={overlay_exc}", flush=True)

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

        _safe_tight_layout(self.fig)
        candle_count = len(self._candles) if hasattr(self, '_candles') and self._candles else 0
        now = time.time()
        if (now - float(getattr(self, "_last_render_log_ts", 0.0) or 0.0)) >= 5.0:
            print(f"[CHART-RENDER] overlays_ok={overlays_ok} candles={candle_count}")
            self._last_render_log_ts = now
        self._safe_draw_idle(fallback_basic=True)
        self._last_render_ts = now

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
            near = _safe_float(ys[-1]) if ys else None
            sx, sy = _filter_xy_pairs(
                xs,
                ys,
                source="timeseries_plugin",
                near_y=near,
                allow_negative=True,
            )
            if sx and sy:
                _safe_plot(
                    self.ax,
                    sx,
                    sy,
                    linewidth=1.0,
                    label=str(label),
                    source="timeseries_plugin",
                    allow_negative=True,
                )
                plotted += 1

        if plotted:
            try:
                _safe_axhline(
                    self.ax,
                    0.0,
                    color="#888888",
                    linewidth=0.8,
                    alpha=0.6,
                    linestyle=":",
                    source="timeseries_plugin_zero",
                    _allow_negative=True,
                )
            except Exception:
                pass
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

        _safe_tight_layout(self.fig)
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
            _safe_plot(
                self.ax, [p[0] for p in ce_pts], [p[1] for p in ce_pts],
                label="CE", linewidth=1.0, color="cyan", source="iv_smile_ce",
            )
        if pe_pts:
            _safe_plot(
                self.ax, [p[0] for p in pe_pts], [p[1] for p in pe_pts],
                label="PE", linewidth=1.0, color="magenta", source="iv_smile_pe",
            )

        if spot is not None:
            _safe_axvline(self.ax, spot, color="gray", linewidth=1.0, alpha=0.6, source="iv_smile_spot")

        try:
            if ce_pts or pe_pts:
                self.ax.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

        _safe_tight_layout(self.fig)
        self.canvas.draw_idle()
