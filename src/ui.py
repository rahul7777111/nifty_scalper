from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("ui.py script started...")

import os
import queue
import json
import math
import sys
import threading
import time
import tkinter as tk
import tkinter.ttk as ttk
from datetime import datetime
from dataclasses import dataclass
from datetime import timedelta
from tkinter import filedialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

from pathlib import Path

from auth import request_sms_otp, verify_sms_otp
from config import load_api_config, load_persisted_env, load_strategy_config, persist_settings_env
from mstock_client import MStockTypeBClient
from strategy import NiftyScalper, TradeLogEvent

# Optional analytics + GPT integrations (kept best-effort so UI stays usable
# even if the user hasn't configured GPT keys yet).
from market_data import Candle
from indicators import adx, atr, ema, rsi, sma, supertrend, roc, choppiness_index, pivot_points
from candlestick_patterns import (
    is_dark_cloud_cover,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_evening_star,
    is_falling_three_methods,
    is_hammer,
    is_hanging_man,
    is_inverted_hammer,
    is_morning_star,
    is_piercing_line,
    is_shooting_star,
    is_three_black_crows,
    is_three_white_soldiers,
    is_rising_three_methods,
)
from greeks import (
    delta as bs_delta,
    gamma as bs_gamma,
    implied_volatility,
    theta as bs_theta,
    vega as bs_vega,
)
from gpt_advisor import analyze_market, health_check


_GPT_STARTUP_ENV_KEYS: tuple[str, ...] = (
    "MSTOCK_GPT_API_KEY",
    "OPENAI_API_KEY",
    "MSTOCK_GPT_API_BASE_URL",
    "MSTOCK_GPT_MODEL",
    "MSTOCK_GPT_ENABLE",
    "MSTOCK_GPT_MODE",
    "MSTOCK_GPT_TIMEOUT_SEC",
)


def _read_windows_registry_env(name: str, hive_name: str) -> str:
    if os.name != "nt":
        return ""

    try:
        import winreg
    except Exception:
        return ""

    hive = getattr(winreg, hive_name, None)
    if hive is None:
        return ""

    try:
        with winreg.OpenKey(hive, r"Environment") as key:
            value, _ = winreg.QueryValueEx(key, str(name))
    except Exception:
        return ""

    try:
        return str(value or "").strip()
    except Exception:
        return ""


def _hydrate_startup_gpt_env() -> None:
    """Load GPT config into process env before the UI builds its fields.

    This keeps the GPT tab populated even when the parent process (for example,
    VS Code) was started before the user updated Windows user environment
    variables via setx.
    """

    for key in _GPT_STARTUP_ENV_KEYS:
        if str(os.getenv(key, "") or "").strip():
            continue

        value = _read_windows_registry_env(key, "HKEY_CURRENT_USER")
        if not value:
            value = _read_windows_registry_env(key, "HKEY_LOCAL_MACHINE")
        if value:
            os.environ[key] = value

    try:
        loaded = load_persisted_env(override_existing=False)
    except Exception:
        loaded = {}

    # Prefer repo-persisted GPT settings over inherited shell values.
    # This avoids stale process env keys causing intermittent 401 errors.
    for key in _GPT_STARTUP_ENV_KEYS:
        val = str(loaded.get(key, "") or "").strip()
        if val:
            os.environ[key] = val


_hydrate_startup_gpt_env()


def _apply_sv_ttk_theme(root: tk.Tk) -> None:
    """Best-effort theme setup with a safe default on Windows.

    `sv_ttk` can render as a black/blank window on some Windows setups.
    To keep the UI always usable, we default to native ttk themes on Windows
    unless the user explicitly opts in.
    """

    # Always try to keep a non-broken built-in ttk theme first.
    try:
        style = ttk.Style(root)
        names = {str(n).lower() for n in style.theme_names()}
        if "vista" in names:
            style.theme_use("vista")
        elif "clam" in names:
            style.theme_use("clam")
    except Exception:
        pass

    raw_theme = str(os.getenv("MSTOCK_UI_THEME", "") or "").strip().lower()
    explicit_theme = raw_theme in {"dark", "light"}
    use_sv_ttk_raw = str(os.getenv("MSTOCK_UI_USE_SV_TTK", "auto") or "auto").strip().lower()
    allow_sv_ttk = use_sv_ttk_raw in {"1", "true", "yes", "on"}

    # Auto policy: stick to native ttk on Windows unless explicitly opted-in.
    if not allow_sv_ttk:
        return

    try:
        import sv_ttk  # type: ignore

        theme = raw_theme if explicit_theme else ("light" if os.name == "nt" else "dark")
        sv_ttk.set_theme(theme)
    except Exception:
        return


def _apply_base_ttk_style(root: tk.Tk) -> None:
    """Apply a clean, consistent ttk style (best-effort).

    Tkinter styling varies across Windows themes; this keeps spacing/fonts
    consistent without depending on a custom theme.
    """

    try:
        default_font = ("Segoe UI", 10)
        root.option_add("*Font", default_font)
    except Exception:
        pass

    try:
        style = ttk.Style(root)
        try:
            style.configure("TLabel", padding=(0, 0, 0, 0))
            style.configure("TButton", padding=(10, 6))
            style.configure("TCheckbutton", padding=(4, 2))
            style.configure("TEntry", padding=(4, 2))
            style.configure("TLabelframe", padding=(10, 8))
            style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
            style.configure("TNotebook.Tab", padding=(12, 8))
        except Exception:
            pass
    except Exception:
        pass


"""UI constants."""

# Strategy keys shown in the UI.
# IMPORTANT: keep this list aligned with the strategies implemented in `strategy.py`.
# Listing unsupported strategies causes the bot to never take trades with no obvious error.
_STRATEGY_CHOICES: tuple[str, ...] = (
    "auto",
    "directional",
    "short_straddle",
    "short_strangle",
    "bull_call_spread",
    "bull_put_spread",
    "call_ratio_backspread",
    "put_ratio_backspread",
    "iron_condor",
    "iron_fly",
    "iron_butterfly",
    "long_straddle",
    "long_strangle",
    "delta_hedged_short_straddle",
    "delta_hedged_long_straddle",

    # Directional explicit variants
    "long_call",
    "long_put",
    "short_call",
    "short_put",

    # Common alias names (strategy engine normalizes these)
    "short_volatility_strategy",
    "long_volatility_strategy",
    "short_volatility",
    "long_volatility",
    "condor_call",
    "condor_put",
    "butterfly_spread_call",
    "butterfly_spread_put",
    "calendar_spread_call",
    "calendar_spread_put",
    "double_calendar_spread",
    "diagonal_spread",
    "covered_call",
    "protective_put",
)


class _QueueWriter:
    def __init__(self, q: queue.Queue[str]) -> None:
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:  # pragma: no cover
        return


@dataclass
class _LoginState:
    refresh_token: str | None = None


class ScalperUI(tk.Tk):
    def __init__(self) -> None:
        print("Initializing ScalperUI...")
        super().__init__()

        # Window baseline sizing (keeps layout usable on laptops).
        try:
            self.title("Scalper Bot")
            self.minsize(980, 720)
            self.geometry(os.getenv("MSTOCK_UI_GEOMETRY", "1150x820"))
        except Exception:
            pass

        # Theme + base styles (best-effort; UI must still run if missing).
        _apply_sv_ttk_theme(self)
        _apply_base_ttk_style(self)

        # Shared UI state (used across header/dashboard/signals).
        self.status_var = tk.StringVar(value="Idle")
        self._dash_last_tick_var = tk.StringVar(value="n/a")
        self._dash_candles_var = tk.StringVar(value="0")
        self._dash_spot_var = tk.StringVar(value="n/a")
        self.access_token_var = tk.StringVar(value=os.getenv("MSTOCK_ACCESS_TOKEN", ""))
        self._dash_token_var = tk.StringVar(value="(set)" if self.access_token_var.get().strip() else "(not set)")
        self._app_status_var = tk.StringVar(value="Ready.")

        try:
            def _sync_dash_token(*_a: object) -> None:
                self._dash_token_var.set("(set)" if self.access_token_var.get().strip() else "(not set)")
            self.access_token_var.trace_add("write", _sync_dash_token)
        except Exception:
            pass

        self._ui_log_fp = None

        self._log_q: queue.Queue[str] = queue.Queue()
        self._trade_q: queue.Queue[TradeLogEvent] = queue.Queue()
        self._trade_rows: dict[str, str] = {}
        self._trade_state: dict[str, dict[str, object]] = {}
        self._client: MStockTypeBClient | None = None
        self._scalper: NiftyScalper | None = None

        # Live dashboard portfolio snapshot.
        self._dash_portfolio_snapshot: dict[str, object] = {}
        self._dash_portfolio_ts: float = 0.0
        self._dash_portfolio_inflight = False
        try:
            self._dash_portfolio_refresh_sec = float(
                (os.getenv("MSTOCK_DASH_PORTFOLIO_REFRESH_SEC", "2") or "2").strip() or 2.0
            )
        except Exception:
            self._dash_portfolio_refresh_sec = 2.0
        if self._dash_portfolio_refresh_sec < 0.5:
            self._dash_portfolio_refresh_sec = 0.5

        # Live spot (LTP) refresh for header/dashboard.
        self._spot_ltp_live: float | None = None
        self._spot_ltp_live_ts: float = 0.0
        self._spot_refresh_inflight = False
        try:
            self._spot_refresh_interval_sec = float((os.getenv("MSTOCK_SPOT_REFRESH_SEC", "1") or "1").strip() or 1.0)
        except Exception:
            self._spot_refresh_interval_sec = 1.0
        if self._spot_refresh_interval_sec < 0.5:
            self._spot_refresh_interval_sec = 0.5
        self._spot_live_stale_sec = max(3.0, float(self._spot_refresh_interval_sec) * 3.0)
        self._option_ltp_refresh_inflight = False
        self._option_ltp_live_ts: float = 0.0
        try:
            self._option_ltp_refresh_interval_sec = float(
                (os.getenv("MSTOCK_OPTION_MTM_REFRESH_SEC", "0.75") or "0.75").strip() or 0.75
            )
        except Exception:
            self._option_ltp_refresh_interval_sec = 0.75
        if self._option_ltp_refresh_interval_sec < 0.25:
            self._option_ltp_refresh_interval_sec = 0.25
        self._margin_required_live_total: float | None = None
        self._margin_required_live_ts: float = 0.0
        self._margin_refresh_inflight = False
        try:
            self._margin_refresh_interval_sec = float(os.getenv("MSTOCK_MARGIN_REFRESH_SEC", "5").strip() or 5.0)
        except Exception:
            self._margin_refresh_interval_sec = 5.0
        if self._margin_refresh_interval_sec < 1.0:
            self._margin_refresh_interval_sec = 1.0
        self._margin_live_stale_sec = max(10.0, self._margin_refresh_interval_sec * 3.0)

        # Per-day closed option leg stats (used for today-only averages).
        # We accumulate on PARTIAL_CLOSE/CLOSE events because the table state only
        # keeps the latest legs snapshot and would otherwise drop earlier partials.
        self._closed_option_legs_day = datetime.now().date()
        self._closed_option_legs: list[dict[str, object]] = []

        # Live per-day traded quantity (option legs only, excludes hedges).
        self._qty_traded_day = datetime.now().date()
        self._qty_traded_total: int = 0
        # Track current open quantities per (trade_id, symbol) so we can count
        # incremental adds on UPDATE events (pyramiding) as "traded".
        self._open_qty_by_trade_symbol: dict[tuple[str, str], int] = {}
        # Remember recent UPDATE-side reductions so PARTIAL_CLOSE can avoid
        # double-counting when both events represent the same exit.
        self._recent_qty_reduction_by_track_key: dict[tuple[str, str], tuple[int, float]] = {}

        # Live per-day traded quantity for underlying hedges only.
        # This is tracked separately because hedge legs are in underlying units
        # (not option contracts) and are often rebalanced (net position changes).
        self._hedge_qty_traded_day = datetime.now().date()
        self._hedge_qty_traded_total: int = 0
        # Track signed net hedge qty per (trade_id, symbol). BUY is positive, SELL is negative.
        self._open_hedge_qty_by_trade_symbol: dict[tuple[str, str], int] = {}
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        # sys.stdout = _QueueWriter(self._log_q)  # type: ignore[assignment]
        # sys.stderr = _QueueWriter(self._log_q)  # type: ignore[assignment]

        # Persist logs to a file as a fallback when users run the UI without a console.
        try:
            repo_root = Path(__file__).resolve().parent.parent
            log_path = repo_root / ".scalper.ui.log"
            self._ui_log_fp = log_path.open("a", encoding="utf-8")
            self._ui_log_fp.write(f"\n--- UI session started {datetime.now().isoformat(timespec='seconds')} ---\n")
            self._ui_log_fp.flush()
        except Exception:
            self._ui_log_fp = None

        self._login_state = _LoginState()
        self._bot_thread: threading.Thread | None = None
        self._bot_stop = threading.Event()

        # Live analytics snapshots (fed via Strategy on_tick callback).
        self._latest_candles: list[Candle] = []
        self._latest_candles_ts: float = 0.0
        self._signals_last_render_ts: float = 0.0
        self._signals_render_throttle_sec: float = 1.0
        self._greeks_inflight = False
        self._greeks_pending = False
        self._greeks_pending_spot: float | None = None
        self._greeks_last_render_ts: float = 0.0
        self._greeks_render_throttle_sec: float = 2.0
        self._gpt_inflight = False

        self._cred_path = self._default_credential_path()

        self._build_widgets()
        self._load_prefilled_credentials()
        self._sync_credential_editability()
        self._sync_trade_log_visibility()
        self.after(100, self._pump_logs)
        self.after(150, self._pump_trades)
        self.after(1000, self._pump_margin_required)
        self.after(750, self._pump_dashboard_portfolio)
        self.after(800, self._pump_spot_ltp)
        self.after(650, self._pump_option_ltp)
        self.after(900, self._pump_engine_diagnostics)

        # Optional: bring window to front on startup (useful if launched from CLI).
        try:
            bring_front = (os.getenv("MSTOCK_UI_BRING_TO_FRONT", "true") or "").strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            bring_front = True
        if bring_front:
            try:
                self.lift()
                self.attributes("-topmost", True)
                self.after(800, lambda: self.attributes("-topmost", False))
                self.deiconify()
                self.focus_force()
            except Exception:
                pass

        # Optional: a startup popup (disabled by default; can be noisy).
        try:
            show_popup = (os.getenv("MSTOCK_UI_STARTUP_POPUP", "false") or "").strip().lower() in {"1", "true", "yes", "y"}
        except Exception:
            show_popup = False
        if show_popup:
            try:
                messagebox.showinfo("Scalper Bot", "UI has started successfully!")
            except Exception:
                pass

        # Keep the trade log visibility in sync with the checkbox.
        self.live_var.trace_add("write", lambda *_: self._sync_trade_log_visibility())

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _refresh_pnl_totals(self) -> None:
        profit = 0.0
        loss = 0.0
        for st in self._trade_state.values():
            try:
                r = st.get("realized")
                if r is None:
                    continue
                r_f = float(r)
            except Exception:
                continue
            if r_f >= 0:
                profit += r_f
            else:
                loss += -r_f

        try:
            self._pnl_profit_var.set(f"₹{profit:.2f}")
            self._pnl_loss_var.set(f"₹{loss:.2f}")
        except Exception:
            return

        try:
            avg, n = self._avg_entry_minus_exit_today()
            if avg is None or n <= 0:
                self._avg_entry_minus_exit_var.set("n/a")
            else:
                self._avg_entry_minus_exit_var.set(f"{avg:.3f}")
        except Exception:
            # Never let stats rendering break the UI.
            try:
                self._avg_entry_minus_exit_var.set("n/a")
            except Exception:
                pass

        # Qty traded is maintained live from events (counts entries too).
        try:
            self._qty_traded_today_var.set(str(int(self._qty_traded_total)))
        except Exception:
            pass

        try:
            self._hedge_qty_traded_today_var.set(str(int(self._hedge_qty_traded_total)))
        except Exception:
            pass

        # Prefer live margin refresh if it was updated recently.
        try:
            live_total = None
            try:
                if self._margin_required_live_total is not None:
                    age = time.time() - float(self._margin_required_live_ts or 0.0)
                    if age <= float(self._margin_live_stale_sec):
                        live_total = float(self._margin_required_live_total)
            except Exception:
                live_total = None

            if live_total is not None:
                self._margin_required_var.set(f"₹{live_total:.2f}")
            else:
                # Best-effort: show total required margin from trade snapshots.
                total_m = 0.0
                any_m = False
                for tid, st in self._trade_state.items():
                    try:
                        if str(tid).endswith("-H"):
                            continue
                    except Exception:
                        pass

                    try:
                        status = str(st.get("status") or "")
                    except Exception:
                        status = ""
                    if status.strip().upper().startswith("CLOSED"):
                        continue

                    mr = st.get("margin_required")
                    if mr is None:
                        continue
                    try:
                        total_m += float(mr)
                        any_m = True
                    except Exception:
                        continue

                if any_m:
                    self._margin_required_var.set(f"₹{total_m:.2f}")
                else:
                    self._margin_required_var.set("n/a")
        except Exception:
            try:
                self._margin_required_var.set("n/a")
            except Exception:
                pass

    def _spot_quote_key(self) -> str:
        """Return a symbol key suitable for client.get_ltp() for the underlying spot."""

        # Prefer explicit token/exchange from Settings (used for candles too).
        token = ""
        exch = ""
        try:
            if hasattr(self, "underlying_token_var"):
                token = str(self.underlying_token_var.get() or "").strip()
        except Exception:
            token = ""
        try:
            if hasattr(self, "underlying_exchange_var"):
                exch = str(self.underlying_exchange_var.get() or "").strip().upper()
        except Exception:
            exch = ""

        if not exch:
            exch = (os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE")) or "NSE").strip().upper() or "NSE"

        # Underlying symbol (spot index/equity).
        underlying = (os.getenv("MSTOCK_UNDERLYING") or os.getenv("MSTOCK_SYMBOL") or "NIFTY").strip()
        if ":" in underlying:
            # Caller already supplied an explicit exchange.
            return underlying

        if token and token.isdigit():
            return f"{exch}:{token}"
        return f"{exch}:{underlying}"

    def _pump_spot_ltp(self) -> None:
        """Refresh live spot LTP (best-effort, never blocks UI)."""

        try:
            if bool(getattr(self, "_spot_refresh_inflight", False)):
                return
            client = getattr(self, "_client", None)
            if client is None:
                return

            now = float(time.time())
            last_ts = float(getattr(self, "_spot_ltp_live_ts", 0.0) or 0.0)
            interval = float(getattr(self, "_spot_refresh_interval_sec", 1.0) or 1.0)
            if (now - last_ts) < interval:
                return

            key = ""
            try:
                key = self._spot_quote_key()
            except Exception:
                key = ""
            if not key:
                return

            self._spot_refresh_inflight = True

            def _worker() -> None:
                ltp = None
                try:
                    ltp = float(client.get_ltp(key))
                except Exception:
                    ltp = None

                def _apply() -> None:
                    try:
                        self._spot_ltp_live = ltp
                        self._spot_ltp_live_ts = float(time.time())
                        if ltp is not None and float(ltp) > 0:
                            try:
                                self._dash_spot_var.set(f"{float(ltp):.2f}")
                            except Exception:
                                pass
                    finally:
                        self._spot_refresh_inflight = False

                try:
                    self.after(0, _apply)
                except Exception:
                    # If the UI is shutting down, just drop the update.
                    self._spot_refresh_inflight = False

            threading.Thread(target=_worker, daemon=True).start()
        finally:
            # Keep the spot fairly fresh even when candles are slow.
            self.after(500, self._pump_spot_ltp)

    def _collect_open_legs_for_margin(self) -> list[dict]:
        legs_out: list[dict] = []
        for tid, st in self._trade_state.items():
            try:
                if str(tid).endswith("-H"):
                    continue
            except Exception:
                pass

            try:
                status = str(st.get("status") or "")
            except Exception:
                status = ""
            if status.strip().upper().startswith("CLOSED"):
                continue

            legs = st.get("legs")
            if not isinstance(legs, list):
                continue
            for leg in legs:
                if isinstance(leg, dict):
                    legs_out.append(dict(leg))
        return legs_out

    def _is_closed_trade_state(self, trade_id: object, state: object) -> bool:
        try:
            if str(trade_id).endswith("-H"):
                return True
        except Exception:
            pass
        if not isinstance(state, dict):
            return True
        try:
            status = str(state.get("status") or "").strip().upper()
        except Exception:
            status = ""
        return status.startswith("CLOSED")

    def _try_get_live_ltp_for_leg(self, client: MStockTypeBClient, leg: dict) -> float | None:
        token = str(leg.get("token") or "").strip()
        exchange_raw = str(leg.get("exchange") or "").strip().upper()
        symbol = str(leg.get("symbol") or "").strip()

        # Infer exchange for option legs when missing.
        if not exchange_raw:
            try:
                if self._is_option_leg(leg):
                    exchange_raw = "NFO"
            except Exception:
                pass
        if exchange_raw in {"NSEFO", "NFO"}:
            exchange_raw = "NFO"
        exchange = exchange_raw.strip()

        if token.isdigit():
            try:
                key = f"{exchange}:{token}" if exchange else token
                return float(client.get_ltp(key))
            except Exception:
                pass

        if symbol:
            try:
                if exchange and ":" not in symbol:
                    return float(client.get_ltp(f"{exchange}:{symbol}"))
                return float(client.get_ltp(symbol))
            except Exception:
                pass
        return None

    def _compute_cached_trade_mtm(self, legs: list[dict]) -> float | None:
        mtm = 0.0
        any_price = False
        for leg in legs or []:
            if not isinstance(leg, dict):
                continue
            try:
                side = str(leg.get("side") or "").strip().upper()
                qty = int(leg.get("quantity") or 0)
                entry = float(leg.get("entry_price")) if leg.get("entry_price") is not None else None
                ltp = float(leg.get("ltp")) if leg.get("ltp") is not None else None
            except Exception:
                continue
            if entry is None or ltp is None or qty <= 0 or side not in {"BUY", "SELL"}:
                continue
            any_price = True
            sign = 1.0 if side == "BUY" else -1.0
            mtm += (float(ltp) - float(entry)) * sign * float(abs(qty))
        return float(mtm) if any_price else None

    def _compute_parent_display_pnl_breakdown(self, trade_id: str, state: dict[str, object]) -> dict[str, float | None]:
        base_mtm: float | None
        base_realized: float | None
        hedge_mtm: float | None = None
        hedge_realized: float | None = None
        try:
            trade_state = object.__getattribute__(self, "__dict__").get("_trade_state", {})
        except Exception:
            trade_state = {}
        if not isinstance(trade_state, dict):
            trade_state = {}

        try:
            base_mtm = float(state.get("mtm")) if state.get("mtm") is not None else None
        except Exception:
            base_mtm = None
        try:
            base_realized = float(state.get("realized")) if state.get("realized") is not None else None
        except Exception:
            base_realized = None

        if not str(trade_id).endswith("-H") and isinstance(state, dict):
            hedge_trade_id = f"{trade_id}-H"
            hedge_state = trade_state.get(hedge_trade_id)
            if isinstance(hedge_state, dict):
                try:
                    hedge_mtm = float(hedge_state.get("mtm")) if hedge_state.get("mtm") is not None else None
                except Exception:
                    hedge_mtm = None
                try:
                    hedge_realized = float(hedge_state.get("realized")) if hedge_state.get("realized") is not None else None
                except Exception:
                    hedge_realized = None

            if hedge_mtm is None:
                legs_all = state.get("legs") if isinstance(state.get("legs"), list) else []
                _main_legs, _hedge_opt_legs, hedge_under_legs = self._split_legs_for_display(legs_all)
                hedge_mtm = self._compute_cached_trade_mtm(hedge_under_legs)

        parent_mtm: float | None
        if base_mtm is None and hedge_mtm is None:
            parent_mtm = None
        else:
            parent_mtm = float(base_mtm or 0.0) + float(hedge_mtm or 0.0)

        parent_realized: float | None
        if base_realized is None and hedge_realized is None:
            parent_realized = None
        else:
            parent_realized = float(base_realized or 0.0) + float(hedge_realized or 0.0)

        return {
            "base_mtm": base_mtm,
            "hedge_mtm": hedge_mtm,
            "parent_mtm": parent_mtm,
            "base_realized": base_realized,
            "hedge_realized": hedge_realized,
            "parent_realized": parent_realized,
        }

    def _compute_parent_display_pnl(
        self,
        trade_id: str,
        state: dict[str, object],
    ) -> tuple[float | None, float | None]:
        """Return parent-display (mtm, realized), including hedge-underlying contribution."""
        breakdown = self._compute_parent_display_pnl_breakdown(trade_id, state)
        return breakdown.get("parent_mtm"), breakdown.get("parent_realized")

    def _is_leg_stop_hit_for_display(self, *, side: str, ltp: float | None, stop_price: float | None) -> bool:
        try:
            side_u = str(side or "").strip().upper()
        except Exception:
            side_u = ""
        if side_u not in {"BUY", "SELL"}:
            return False

        try:
            ltp_f = float(ltp) if ltp is not None else None
        except Exception:
            ltp_f = None
        try:
            stop_f = float(stop_price) if stop_price is not None else None
        except Exception:
            stop_f = None
        if ltp_f is None or stop_f is None:
            return False

        if side_u == "BUY":
            return float(ltp_f) <= float(stop_f)
        return float(ltp_f) >= float(stop_f)

    def _build_live_option_update_event(
        self,
        client: MStockTypeBClient,
        trade_id: str,
        state: dict[str, object],
        *,
        refresh_all_legs: bool = False,
    ) -> TradeLogEvent | None:
        if not trade_id or self._is_closed_trade_state(trade_id, state):
            return None

        legs_src = state.get("legs")
        if not isinstance(legs_src, list) or not legs_src:
            return None

        refreshed_any = False
        cloned_legs: list[dict] = []
        for leg in legs_src:
            if not isinstance(leg, dict):
                continue
            leg_copy = dict(leg)
            try:
                qty_live = int(leg_copy.get("quantity") or 0)
            except Exception:
                qty_live = 0
            if qty_live <= 0:
                continue
            if refresh_all_legs or self._is_option_leg(leg_copy):
                try:
                    live_ltp = self._try_get_live_ltp_for_leg(client, leg_copy)
                except Exception:
                    live_ltp = None
                if live_ltp is not None and float(live_ltp) > 0:
                    leg_copy["ltp"] = float(live_ltp)
                    refreshed_any = True
            cloned_legs.append(leg_copy)

        if not cloned_legs:
            return None

        if not refreshed_any:
            return None

        mtm = self._compute_cached_trade_mtm(cloned_legs)
        return TradeLogEvent(
            ts=time.time(),
            event="UPDATE",
            trade_id=str(trade_id),
            position_type=str(state.get("pos_type") or ""),
            name=str(state.get("strategy") or ""),
            legs=cloned_legs,
            mtm=mtm,
        )

    def _pump_option_ltp(self) -> None:
        try:
            if bool(getattr(self, "_option_ltp_refresh_inflight", False)):
                return
            client = getattr(self, "_client", None)
            if client is None:
                return

            now = float(time.time())
            last_ts = float(getattr(self, "_option_ltp_live_ts", 0.0) or 0.0)
            interval = float(getattr(self, "_option_ltp_refresh_interval_sec", 0.75) or 0.75)
            if (now - last_ts) < interval:
                return

            trade_items: list[tuple[str, dict[str, object]]] = []
            for trade_id, state in list((self._trade_state or {}).items()):
                if not isinstance(state, dict):
                    continue
                if self._is_closed_trade_state(trade_id, state):
                    continue
                legs = state.get("legs")
                if not isinstance(legs, list) or not legs:
                    continue
                if str(trade_id).endswith("-H"):
                    trade_items.append((str(trade_id), state))
                    continue
                if not any(
                    isinstance(leg, dict)
                    and self._is_option_leg(leg)
                    and int(leg.get("quantity") or 0) > 0
                    for leg in legs
                ):
                    continue
                trade_items.append((str(trade_id), state))

            if not trade_items:
                self._option_ltp_live_ts = now
                return

            self._option_ltp_refresh_inflight = True

            def _worker() -> None:
                events: list[TradeLogEvent] = []
                try:
                    for trade_id, state in trade_items:
                        evt = self._build_live_option_update_event(
                            client,
                            trade_id,
                            state,
                            refresh_all_legs=str(trade_id).endswith("-H"),
                        )
                        if evt is not None:
                            events.append(evt)
                except Exception:
                    events = []

                def _apply() -> None:
                    try:
                        for evt in events:
                            self._trade_q.put(evt)
                        self._option_ltp_live_ts = float(time.time())
                    finally:
                        self._option_ltp_refresh_inflight = False

                try:
                    self.after(0, _apply)
                except Exception:
                    self._option_ltp_refresh_inflight = False

            threading.Thread(target=_worker, daemon=True).start()
        finally:
            self.after(250, self._pump_option_ltp)

    def _calc_live_margin_required(self, client: MStockTypeBClient, legs: list[dict]) -> float | None:
        total = 0.0
        any_ok = False
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            try:
                token = str(leg.get("token") or "").strip()
                exchange_raw = str(leg.get("exchange") or "").strip().upper()
                symbol_raw = str(leg.get("symbol") or "").strip()
                side = str(leg.get("side") or "").strip().upper() or None
                quantity = int(leg.get("quantity") or 0)
            except Exception:
                continue

            # Normalize "EXCH:TRADINGSYMBOL" into raw trading symbol.
            symbol = symbol_raw or None
            if symbol and ":" in symbol:
                try:
                    maybe_exch, maybe_sym = symbol.split(":", 1)
                    if maybe_exch.strip().upper() in {"NSE", "BSE", "NFO", "NSEFO", "NSECM", "BSECM"} and maybe_sym.strip():
                        symbol = maybe_sym.strip()
                        # Prefer explicit prefix exchange when exchange field is empty.
                        if not exchange_raw:
                            exchange_raw = maybe_exch.strip().upper()
                except Exception:
                    # Keep original symbol.
                    symbol = symbol_raw or None

            # Infer exchange for option legs when missing.
            if not exchange_raw:
                try:
                    if self._is_option_leg(leg):
                        exchange_raw = "NFO"
                except Exception:
                    pass
            if exchange_raw in {"NSEFO", "NFO"}:
                exchange_raw = "NFO"
            exchange = exchange_raw.strip() or None

            # Some strategy flows (notably some long-premium entries) may not
            # attach token/exchange to the leg snapshot. Best-effort resolve it.
            if (not token or not token.isdigit()) and symbol:
                try:
                    resolved_exch, resolved_tok = client.resolve_exchange_token(symbol, exchange_hint=exchange)
                    if resolved_tok:
                        token = str(resolved_tok).strip()
                    if resolved_exch and not exchange:
                        exchange = str(resolved_exch).strip().upper() or exchange
                    # Normalize derivatives exchange.
                    if exchange in {"NSEFO", "NFO"}:
                        exchange = "NFO"
                except Exception:
                    pass

            if not token or not exchange or not symbol or not side or quantity <= 0:
                continue

            price = None
            for key in ("entry_price", "ltp", "price", "exit_price"):
                if leg.get(key) is None:
                    continue
                try:
                    price = float(leg.get(key))  # type: ignore[arg-type]
                    break
                except Exception:
                    price = None

            if price is None:
                try:
                    price = self._try_get_live_ltp_for_leg(client, leg)
                except Exception:
                    price = None

            try:
                mr = client.calculate_order_margin_required(
                    exchange=exchange,
                    symbol=symbol,
                    token=token,
                    side=side,
                    quantity=quantity,
                    price=price,
                )
            except Exception:
                mr = None

            # Broker APIs sometimes report 0 required margin for BUY option legs.
            # In practice, long premium still requires cash outlay ~= premium * qty.
            # Treat that as required margin so UI doesn't misleadingly show ₹0.00.
            try:
                mr_f = float(mr) if mr is not None else None
            except Exception:
                mr_f = None
            if (mr_f is None or mr_f <= 0.0) and side == "BUY" and price is not None:
                try:
                    premium_required = float(abs(float(price)) * float(abs(int(quantity))))
                except Exception:
                    premium_required = None
                if premium_required is not None and premium_required > 0.0:
                    mr = float(premium_required)

            if mr is None:
                continue
            try:
                total += float(mr)
                any_ok = True
            except Exception:
                continue

        if not any_ok:
            return None
        return float(total)

    def _pump_margin_required(self) -> None:
        try:
            if self._margin_refresh_inflight:
                return
            client = self._client
            if client is None:
                return
            now = time.time()
            if (now - float(self._margin_required_live_ts or 0.0)) < float(self._margin_refresh_interval_sec):
                return

            legs_snapshot = self._collect_open_legs_for_margin()
            if not legs_snapshot:
                self._margin_required_live_total = None
                self._margin_required_live_ts = now
                try:
                    self._refresh_pnl_totals()
                except Exception:
                    pass
                return

            self._margin_refresh_inflight = True

            def _worker() -> None:
                total = None
                try:
                    total = self._calc_live_margin_required(client, legs_snapshot)
                except Exception:
                    total = None

                def _apply() -> None:
                    try:
                        self._margin_required_live_total = total
                        self._margin_required_live_ts = time.time()
                        try:
                            self._refresh_pnl_totals()
                        except Exception:
                            pass
                    finally:
                        self._margin_refresh_inflight = False

                self.after(0, _apply)

            threading.Thread(target=_worker, daemon=True).start()
        finally:
            self.after(1000, self._pump_margin_required)

    def _pump_dashboard_portfolio(self) -> None:
        try:
            if self._dash_portfolio_inflight:
                return

            now = time.time()
            if (now - float(self._dash_portfolio_ts or 0.0)) < float(self._dash_portfolio_refresh_sec):
                return

            # Build a stable snapshot to avoid holding references across threads.
            try:
                trade_state_items = list((self._trade_state or {}).items())
            except Exception:
                trade_state_items = []

            scalper = getattr(self, "_scalper", None)
            try:
                if scalper is not None:
                    scalper.enable_gpt_leg_ui_visibility_gate(True)
            except Exception:
                pass
            try:
                equity_positions = dict(getattr(scalper, "_equity_positions", {}) or {}) if scalper is not None else {}
            except Exception:
                equity_positions = {}

            client = getattr(self, "_client", None)

            self._dash_portfolio_inflight = True

            def _worker() -> None:
                def _to_float(x: object) -> float | None:
                    try:
                        if x is None:
                            return None
                        return float(x)
                    except Exception:
                        return None

                def _to_int(x: object) -> int:
                    try:
                        return int(x)
                    except Exception:
                        return 0

                eq_rows: list[dict[str, object]] = []
                eq_exposure = 0.0
                for sym, pos in (equity_positions or {}).items():
                    if not isinstance(pos, dict):
                        continue
                    symbol = str(sym or "").strip() or str(pos.get("symbol") or "").strip()
                    if not symbol:
                        continue
                    side = str(pos.get("side") or "").strip().upper()
                    qty = _to_int(pos.get("qty") or 0)
                    entry = _to_float(pos.get("entry"))
                    stop = _to_float(pos.get("stop"))
                    target = _to_float(pos.get("target"))

                    ltp = None
                    if client is not None:
                        keys = [symbol]
                        if ":" not in symbol:
                            keys.append(f"NSE:{symbol}")
                            if symbol.endswith("-EQ"):
                                keys.append(f"NSE:{symbol[:-3]}")
                        for key in keys:
                            try:
                                ltp = float(client.get_ltp(key))
                                if ltp > 0:
                                    break
                            except Exception:
                                ltp = None
                    pnl = None
                    if ltp is not None and entry is not None and qty:
                        try:
                            sign = 1.0 if side == "BUY" else -1.0
                            pnl = (float(ltp) - float(entry)) * sign * float(abs(qty))
                        except Exception:
                            pnl = None
                    if ltp is not None and qty:
                        try:
                            eq_exposure += abs(float(ltp) * float(qty))
                        except Exception:
                            pass

                    eq_rows.append(
                        {
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "entry": entry,
                            "ltp": ltp,
                            "pnl": pnl,
                            "stop": stop,
                            "target": target,
                        }
                    )

                opt_rows: list[dict[str, object]] = []
                opt_exposure = 0.0
                opt_leg_count = 0
                opt_trade_count = 0

                seen_trade_ids: set[str] = set()
                for tid, st in trade_state_items:
                    trade_id = str(tid or "").strip()
                    if not trade_id:
                        continue
                    # Skip synthetic hedge-only row.
                    if trade_id.endswith("-H"):
                        continue
                    if not isinstance(st, dict):
                        continue

                    status = str(st.get("status") or "").strip()
                    if status.upper().startswith("CLOSED"):
                        continue

                    legs = st.get("legs")
                    if not isinstance(legs, list):
                        legs = []

                    any_open_option_leg = False
                    for lg in legs:
                        if not isinstance(lg, dict):
                            continue
                        # Keep option hedges visible in option dashboards.
                        if bool(lg.get("is_hedge")) and not self._is_option_leg(lg):
                            continue
                        if self._get_leg_qty(lg) > 0:
                            any_open_option_leg = True
                            break
                    if not any_open_option_leg:
                        continue

                    seen_trade_ids.add(trade_id)
                    pnl_breakdown = self._compute_parent_display_pnl_breakdown(trade_id, st)
                    trade_mtm_display = pnl_breakdown.get("parent_mtm")
                    base_mtm_display = pnl_breakdown.get("base_mtm")
                    hedge_mtm_display = pnl_breakdown.get("hedge_mtm")

                    # For multi-leg option trades, the strategy attaches MTM stop/target
                    # thresholds (trade-level rupees). The dashboard table is leg-level
                    # (Entry/LTP are per-contract premium), so display stop/target as
                    # per-leg premium price levels derived from those thresholds.
                    entry_premium_abs: float | None = None
                    try:
                        entry_premium_signed = 0.0
                        any_entry = False
                        for lg in legs:
                            if not isinstance(lg, dict):
                                continue
                            if bool(lg.get("is_hedge")) and not self._is_option_leg(lg):
                                continue
                            e = _to_float(lg.get("entry_price"))
                            q = self._get_leg_qty(lg)
                            s = str(lg.get("side") or "").strip().upper()
                            if e is None or q <= 0 or s not in {"BUY", "SELL"}:
                                continue
                            any_entry = True
                            if s == "SELL":
                                entry_premium_signed += float(e) * float(q)
                            else:
                                entry_premium_signed -= float(e) * float(q)
                        if any_entry:
                            entry_premium_abs = float(abs(entry_premium_signed))
                    except Exception:
                        entry_premium_abs = None

                    for leg in legs:
                        if not isinstance(leg, dict):
                            continue
                        is_hedge_leg = bool(leg.get("is_hedge"))
                        if is_hedge_leg and not self._is_option_leg(leg):
                            continue

                        sym = self._format_leg_symbol_ui(leg, include_hedge_tag=is_hedge_leg)
                        side = str(leg.get("side") or "").strip().upper()
                        qty = self._get_leg_qty(leg)
                        if qty <= 0:
                            continue
                        entry = _to_float(leg.get("entry_price"))
                        ltp = _to_float(leg.get("ltp"))

                        leg_mtm = None
                        if ltp is not None and entry is not None and qty:
                            try:
                                sign = 1.0 if side == "BUY" else -1.0
                                leg_mtm = (float(ltp) - float(entry)) * sign * float(abs(qty))
                            except Exception:
                                leg_mtm = None

                        # Stop/target: strategy may attach either MTM thresholds (multi)
                        # or spot thresholds (directional). Display whichever exists.
                        prem_stop = _to_float(leg.get("prem_stop"))
                        prem_target = _to_float(leg.get("prem_target"))
                        mtm_stop = _to_float(leg.get("mtm_stop"))
                        mtm_target = _to_float(leg.get("mtm_target"))
                        spot_stop = _to_float(leg.get("spot_stop"))
                        spot_target = _to_float(leg.get("spot_target"))

                        stop_s = ""
                        tgt_s = ""
                        stop_px: float | None = None
                        if prem_stop is not None or prem_target is not None:
                            if prem_stop is not None:
                                stop_s = f"{float(prem_stop):.2f}"
                                stop_px = float(prem_stop)
                            if prem_target is not None:
                                tgt_s = f"{float(prem_target):.2f}"
                        elif mtm_stop is not None or mtm_target is not None:
                            # Convert trade-level MTM thresholds into per-leg premium levels
                            # using the entry premium % implied by (mtm_* / entry_premium_abs).
                            try:
                                e = entry
                                if e is not None and entry_premium_abs is not None and entry_premium_abs > 0:
                                    stop_pct_eff = abs(float(mtm_stop)) / float(entry_premium_abs) if mtm_stop is not None else None
                                    tgt_pct_eff = abs(float(mtm_target)) / float(entry_premium_abs) if mtm_target is not None else None
                                    if stop_pct_eff is not None and stop_pct_eff >= 0:
                                        if side == "SELL":
                                            sl_price = float(e) * (1.0 + float(stop_pct_eff))
                                        else:
                                            sl_price = float(e) * (1.0 - float(stop_pct_eff))
                                        if sl_price < 0:
                                            sl_price = 0.0
                                        stop_s = f"{sl_price:.2f}"
                                        stop_px = float(sl_price)
                                    if tgt_pct_eff is not None and tgt_pct_eff >= 0:
                                        if side == "SELL":
                                            tp_price = float(e) * (1.0 - float(tgt_pct_eff))
                                        else:
                                            tp_price = float(e) * (1.0 + float(tgt_pct_eff))
                                        if tp_price < 0:
                                            tp_price = 0.0
                                        tgt_s = f"{tp_price:.2f}"
                            except Exception:
                                stop_s = ""
                                tgt_s = ""
                        elif spot_stop is not None or spot_target is not None:
                            if spot_stop is not None:
                                stop_s = f"{spot_stop:.1f}"
                                stop_px = float(spot_stop)
                            if spot_target is not None:
                                tgt_s = f"{spot_target:.1f}"

                        row_status = status
                        try:
                            status_u = str(status or "").strip().upper()
                        except Exception:
                            status_u = ""
                        if status_u.startswith("OPEN") or status_u.startswith("PARTIAL"):
                            sl_hit_pending = False
                            # MTM stop is a trade-level condition. Do not infer SL hit
                            # from per-leg converted stop prices, which can be misleading.
                            if mtm_stop is not None:
                                trade_mtm_now = _to_float(st.get("mtm"))
                                if trade_mtm_now is None:
                                    trade_mtm_now = self._compute_cached_trade_mtm(legs)
                                if trade_mtm_now is not None:
                                    try:
                                        sl_hit_pending = float(trade_mtm_now) <= float(mtm_stop)
                                    except Exception:
                                        sl_hit_pending = False
                            else:
                                sl_hit_pending = self._is_leg_stop_hit_for_display(
                                    side=side,
                                    ltp=ltp,
                                    stop_price=stop_px,
                                )
                            if sl_hit_pending:
                                row_status = "SL HIT (pending close)"

                        if ltp is not None and qty:
                            try:
                                opt_exposure += abs(float(ltp) * float(qty))
                            except Exception:
                                pass

                        opt_leg_count += 1
                        opt_rows.append(
                            {
                                "trade_id": trade_id,
                                "strategy": str(st.get("strategy") or ""),
                                "status": row_status,
                                "symbol": sym,
                                "side": side,
                                "qty": qty,
                                "entry": entry,
                                "ltp": ltp,
                                "mtm": trade_mtm_display,
                                "base_mtm": base_mtm_display,
                                "hedge_mtm": hedge_mtm_display,
                                "leg_mtm": leg_mtm,
                                "stop": stop_s,
                                "target": tgt_s,
                            }
                        )

                opt_trade_count = len(seen_trade_ids)

                total_exposure = float(eq_exposure + opt_exposure)
                eq_ratio = (float(eq_exposure) / total_exposure) if total_exposure > 0 else 0.0
                opt_ratio = (float(opt_exposure) / total_exposure) if total_exposure > 0 else 0.0

                snapshot = {
                    "ts": time.time(),
                    "equity_rows": eq_rows,
                    "option_rows": opt_rows,
                    "equity_exposure": float(eq_exposure),
                    "option_exposure": float(opt_exposure),
                    "equity_ratio": float(eq_ratio),
                    "option_ratio": float(opt_ratio),
                    "equity_count": int(len(eq_rows)),
                    "option_leg_count": int(opt_leg_count),
                    "option_trade_count": int(opt_trade_count),
                }

                def _apply() -> None:
                    try:
                        self._dash_portfolio_snapshot = snapshot
                        self._dash_portfolio_ts = time.time()
                        self._render_dashboard_portfolio(snapshot)
                    finally:
                        self._dash_portfolio_inflight = False

                self.after(0, _apply)

            threading.Thread(target=_worker, daemon=True).start()
        finally:
            self.after(500, self._pump_dashboard_portfolio)

    def _pump_engine_diagnostics(self) -> None:
        try:
            scalper = getattr(self, "_scalper", None)
            if scalper is None:
                return
            getter = getattr(scalper, "get_runtime_diagnostics", None)
            if not callable(getter):
                return
            snap = getter()
            if not isinstance(snap, dict):
                return

            router = snap.get("router") if isinstance(snap.get("router"), dict) else {}
            sel = str(router.get("selected") or "n/a")
            src = str(router.get("source") or "n/a")
            cands = router.get("candidates")
            cand_txt = ", ".join(str(x) for x in list(cands or [])[:4]) if isinstance(cands, list) else "n/a"
            self._diag_router_var.set(f"Router: {sel} ({src}) | Candidates: {cand_txt}")

            lb = snap.get("last_block") if isinstance(snap.get("last_block"), dict) else {}
            lb_code = str(lb.get("code") or "n/a")
            lb_reason = str(lb.get("reason") or "n/a")
            alt_trade = str(lb.get("suggested_trade") or "").strip()
            alt_source = str(lb.get("suggestion_source") or "").strip()
            alt_suffix = f" | alt: {alt_trade} ({alt_source or 'policy'})" if alt_trade else ""
            self._diag_last_block_var.set(f"Last block: {lb_code} | {lb_reason[:120]}{alt_suffix}")

            top_blocks = snap.get("top_block_codes")
            if isinstance(top_blocks, list) and top_blocks:
                parts = []
                for row in top_blocks[:3]:
                    try:
                        code, cnt = row
                        parts.append(f"{code}:{int(cnt)}")
                    except Exception:
                        continue
                self._diag_top_block_var.set("Top block: " + (", ".join(parts) if parts else "n/a"))
            else:
                self._diag_top_block_var.set("Top block: n/a")

            top_dec = snap.get("top_decisions")
            if isinstance(top_dec, list) and top_dec:
                parts = []
                for row in top_dec[:4]:
                    try:
                        nm, cnt = row
                        parts.append(f"{nm}:{int(cnt)}")
                    except Exception:
                        continue
                self._diag_decisions_var.set("Decisions: " + (", ".join(parts) if parts else "n/a"))
            else:
                self._diag_decisions_var.set("Decisions: n/a")

            top_exec = snap.get("top_selected")
            if isinstance(top_exec, list) and top_exec:
                parts = []
                for row in top_exec[:4]:
                    try:
                        nm, cnt = row
                        parts.append(f"{nm}:{int(cnt)}")
                    except Exception:
                        continue
                self._diag_exec_var.set("Executed: " + (", ".join(parts) if parts else "n/a"))
            else:
                self._diag_exec_var.set("Executed: n/a")

            pr = snap.get("portfolio_risk") if isinstance(snap.get("portfolio_risk"), dict) else {}
            try:
                delta_abs = float(pr.get("delta_abs") or 0.0)
            except Exception:
                delta_abs = 0.0
            try:
                notional = float(pr.get("option_notional_abs") or 0.0)
            except Exception:
                notional = 0.0
            try:
                legs_n = int(pr.get("legs_count") or 0)
            except Exception:
                legs_n = 0
            self._diag_risk_var.set(f"Risk: |Δ|={delta_abs:.1f} | Notional={notional:.0f} | Legs={legs_n}")

            p_req = snap.get("gpt_preset_request") if isinstance(snap.get("gpt_preset_request"), dict) else {}
            req = str(p_req.get("preset_request") or "").strip().lower()
            if req in {"aggressive", "conservative"}:
                try:
                    conf = float(p_req.get("confidence") or 0.0)
                except Exception:
                    conf = 0.0
                cur = str(p_req.get("current_preset") or "(none)").strip() or "(none)"
                reason = str(p_req.get("preset_reason") or "").strip()
                self._diag_preset_req_var.set(
                    f"GPT preset request: {req} (current={cur}, conf={conf:.2f}) | {reason[:110]}"
                )
            else:
                self._diag_preset_req_var.set("GPT preset request: n/a")
        except Exception:
            pass
        finally:
            self.after(1000, self._pump_engine_diagnostics)

    def _render_dashboard_portfolio(self, snapshot: dict[str, object]) -> None:
        visible_trade_ids: set[str] = set()

        # Summary line
        try:
            eq_exp = float(snapshot.get("equity_exposure") or 0.0)
        except Exception:
            eq_exp = 0.0
        try:
            opt_exp = float(snapshot.get("option_exposure") or 0.0)
        except Exception:
            opt_exp = 0.0
        try:
            eq_ratio = float(snapshot.get("equity_ratio") or 0.0)
        except Exception:
            eq_ratio = 0.0
        try:
            opt_ratio = float(snapshot.get("option_ratio") or 0.0)
        except Exception:
            opt_ratio = 0.0

        try:
            eq_count = int(snapshot.get("equity_count") or 0)
        except Exception:
            eq_count = 0
        try:
            opt_trades = int(snapshot.get("option_trade_count") or 0)
        except Exception:
            opt_trades = 0
        try:
            opt_legs = int(snapshot.get("option_leg_count") or 0)
        except Exception:
            opt_legs = 0

        try:
            ts = float(snapshot.get("ts") or 0.0)
            ts_s = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 0 else "n/a"
        except Exception:
            ts_s = "n/a"

        try:
            if hasattr(self, "_dash_portfolio_summary_var"):
                self._dash_portfolio_summary_var.set(
                    f"Updated {ts_s} | Exposure: Equity ₹{eq_exp:.0f} ({eq_ratio*100:.0f}%) | "
                    f"Options ₹{opt_exp:.0f} ({opt_ratio*100:.0f}%) | "
                    f"Counts: Eq {eq_count} | Opt {opt_trades} trades / {opt_legs} legs"
                )
        except Exception:
            pass

        # Render options table
        try:
            tree = getattr(self, "dash_opt_tree", None)
            if tree is not None:
                for item in tree.get_children(""):
                    tree.delete(item)
                rows = snapshot.get("option_rows")
                if isinstance(rows, list):
                    for r in rows[:200]:
                        if not isinstance(r, dict):
                            continue
                        try:
                            trade_id = str(r.get("trade_id") or "").strip()
                        except Exception:
                            trade_id = ""
                        entry = r.get("entry")
                        ltp = r.get("ltp")
                        leg_mtm = r.get("leg_mtm")
                        mtm = r.get("mtm")
                        base_mtm = r.get("base_mtm")
                        hedge_mtm = r.get("hedge_mtm")
                        if trade_id and ltp is not None:
                            visible_trade_ids.add(trade_id)
                        entry_s = "" if entry is None else f"{float(entry):.2f}"
                        ltp_s = "" if ltp is None else f"{float(ltp):.2f}"
                        leg_mtm_s = "" if leg_mtm is None else f"{float(leg_mtm):.2f}"
                        mtm_s = "" if mtm is None else f"{float(mtm):.2f}"
                        base_mtm_s = "" if base_mtm is None else f"{float(base_mtm):.2f}"
                        hedge_mtm_s = "" if hedge_mtm is None else f"{float(hedge_mtm):.2f}"
                        tree.insert(
                            "",
                            tk.END,
                            values=(
                                r.get("trade_id"),
                                r.get("strategy"),
                                r.get("symbol"),
                                r.get("side"),
                                r.get("qty"),
                                entry_s,
                                ltp_s,
                                leg_mtm_s,
                                mtm_s,
                                base_mtm_s,
                                hedge_mtm_s,
                                r.get("stop"),
                                r.get("target"),
                                r.get("status"),
                            ),
                        )
        except Exception:
            pass

        try:
            scalper = getattr(self, "_scalper", None)
            if scalper is not None and visible_trade_ids:
                marker = getattr(scalper, "mark_trade_ui_visible", None)
                if callable(marker):
                    now_ts = time.time()
                    for trade_id in visible_trade_ids:
                        marker(trade_id, ts=now_ts)
        except Exception:
            pass

        # Render equities table
        try:
            tree = getattr(self, "dash_eq_tree", None)
            if tree is not None:
                for item in tree.get_children(""):
                    tree.delete(item)
                rows = snapshot.get("equity_rows")
                if isinstance(rows, list):
                    for r in rows[:200]:
                        if not isinstance(r, dict):
                            continue
                        entry = r.get("entry")
                        ltp = r.get("ltp")
                        pnl = r.get("pnl")
                        entry_s = "" if entry is None else f"{float(entry):.2f}"
                        ltp_s = "" if ltp is None else f"{float(ltp):.2f}"
                        pnl_s = "" if pnl is None else f"{float(pnl):.2f}"
                        stop = r.get("stop")
                        tgt = r.get("target")
                        stop_s = "" if stop is None else f"{float(stop):.2f}"
                        tgt_s = "" if tgt is None else f"{float(tgt):.2f}"
                        tree.insert(
                            "",
                            tk.END,
                            values=(
                                r.get("symbol"),
                                r.get("side"),
                                r.get("qty"),
                                entry_s,
                                ltp_s,
                                pnl_s,
                                stop_s,
                                tgt_s,
                            ),
                        )
        except Exception:
            pass

    def _is_option_leg(self, leg: dict[str, object]) -> bool:
        try:
            opt_type = str(leg.get("option_type") or "").upper().strip()
            if opt_type in {"CE", "PE"}:
                return True
        except Exception:
            pass
        try:
            sym = str(leg.get("symbol") or "").upper().strip()
        except Exception:
            sym = ""
        return bool(sym.endswith("CE") or sym.endswith("PE"))

    def _get_leg_qty(self, leg: dict[str, object]) -> int:
        """Return a best-effort absolute quantity for a leg."""
        try:
            raw = leg.get("quantity")
            if raw is None:
                raw = leg.get("qty")
            if raw is None:
                raw = leg.get("netQty")
            if raw is None:
                raw = leg.get("netqty")
            qty = int(raw or 0)
        except Exception:
            qty = 0
        if qty < 0:
            qty = abs(qty)
        return qty

    def _format_leg_symbol_ui(self, leg: dict[str, object], *, include_hedge_tag: bool = False) -> str:
        """Return a compact user-friendly leg symbol for UI tables."""
        try:
            sym = str(leg.get("symbol") or "").strip()
        except Exception:
            sym = ""

        label = sym
        try:
            strike = leg.get("strike")
            opt_type = str(leg.get("option_type") or "").upper().strip()
            expiry = leg.get("expiry")
        except Exception:
            strike, opt_type, expiry = None, "", None

        if strike is not None or opt_type:
            try:
                if strike is not None:
                    s_val = float(strike)
                    s_str = f"{int(s_val)}" if s_val.is_integer() else f"{s_val:g}"
                else:
                    s_str = ""
            except Exception:
                s_str = str(strike or "")

            exp_str = ""
            if expiry is not None:
                try:
                    if hasattr(expiry, "strftime"):
                        exp_str = expiry.strftime("%d-%b-%y")
                    else:
                        exp_str = str(expiry).split()[0]
                except Exception:
                    exp_str = str(expiry)

            pieces: list[str] = []
            if exp_str:
                pieces.append(exp_str)
            if s_str:
                pieces.append(f"{s_str}")
            if opt_type:
                pieces.append(opt_type)
            if pieces:
                label = " ".join(pieces)
        else:
            up = sym.upper()
            if up.endswith("CE") or up.endswith("PE"):
                opt = up[-2:]
                body = up[:-2]
                j = len(body) - 1
                while j >= 0 and body[j].isdigit():
                    j -= 1
                digits = body[j + 1 :]
                if digits:
                    label = f"{digits} {opt}"

        if include_hedge_tag:
            return f"HEDGE OPT: {label}"
        return label

    def _qty_track_key(self, trade_id: str, leg: dict[str, object]) -> tuple[str, str] | None:
        """Return a stable key for qty tracking across OPEN/UPDATE/PARTIAL_CLOSE/CLOSE.

        Some event snapshots may differ in symbol formatting (exchange prefixes, spaces,
        etc.). Prefer `token` when available; fall back to a normalized symbol.
        """

        tid = str(trade_id or "").strip()
        if not tid:
            return None

        try:
            tok = str(leg.get("token") or "").strip()
        except Exception:
            tok = ""
        if tok:
            return (tid, f"tok:{tok}")

        try:
            sym = str(leg.get("symbol") or "").strip().upper()
        except Exception:
            sym = ""
        if not sym:
            return None

        # Drop optional exchange prefix and whitespace to reduce accidental mismatches.
        if ":" in sym:
            sym = sym.split(":", 1)[-1]
        sym = sym.replace(" ", "")
        return (tid, f"sym:{sym}")

    def _qty_track_keys(self, trade_id: str, leg: dict[str, object]) -> tuple[tuple[str, str] | None, list[tuple[str, str]]]:
        """Return (primary_key, all_keys) for qty tracking.

        Some event payloads may omit `token` on PARTIAL_CLOSE (or vice-versa).
        To keep accounting consistent, we track both:
        - primary: prefer token when available, else normalized symbol
        - aliases: write the same qty to both token+symbol keys when possible
        """

        primary = self._qty_track_key(trade_id, leg)
        if primary is None:
            return None, []

        keys: list[tuple[str, str]] = [primary]

        tid = str(trade_id or "").strip()
        if not tid:
            return primary, keys

        # If primary is token-based, also add symbol alias when possible.
        # If primary is symbol-based, also add token alias when possible.
        try:
            tok = str(leg.get("token") or "").strip()
        except Exception:
            tok = ""
        try:
            sym = str(leg.get("symbol") or "").strip().upper()
        except Exception:
            sym = ""

        if tok:
            k_tok = (tid, f"tok:{tok}")
            if k_tok not in keys:
                keys.append(k_tok)

        if sym:
            if ":" in sym:
                sym = sym.split(":", 1)[-1]
            sym = sym.replace(" ", "")
            k_sym = (tid, f"sym:{sym}")
            if k_sym not in keys:
                keys.append(k_sym)

        # Prefer token as primary when available.
        if tok:
            primary = (tid, f"tok:{tok}")
        else:
            primary = keys[0]
        return primary, keys

    def _note_closed_option_legs(self, evt: TradeLogEvent) -> None:
        # Reset automatically on day change (based on event timestamp).
        try:
            evt_day = datetime.fromtimestamp(float(evt.ts)).date()
        except Exception:
            evt_day = datetime.now().date()

        if evt_day != self._closed_option_legs_day:
            self._closed_option_legs_day = evt_day
            self._closed_option_legs = []

        for leg in evt.legs or []:
            if not isinstance(leg, dict):
                continue
            if not self._is_option_leg(leg):
                continue

            exit_p = leg.get("exit_price")
            if exit_p is None:
                # CLOSE snapshots may only carry LTP, especially when the leg is already zeroed.
                exit_p = leg.get("ltp")
            if exit_p is None:
                # Ignore OPEN/UPDATE snapshots that do not yet have an exit value.
                continue

            entry = leg.get("entry_price")
            try:
                entry_f = float(entry) if entry is not None else None
                exit_f = float(exit_p) if exit_p is not None else None
            except Exception:
                entry_f, exit_f = None, None
            if entry_f is None or exit_f is None:
                continue

            try:
                qty = int(leg.get("quantity") or 0)
            except Exception:
                qty = 0
            if qty <= 0:
                continue

            try:
                side = str(leg.get("side") or "").strip().upper()
            except Exception:
                        qty = self._get_leg_qty(leg)

            self._closed_option_legs.append(
                {
                    "ts": float(evt.ts),
                    "symbol": str(leg.get("symbol") or ""),
                    "side": side,
                    "quantity": qty,
                    "entry_price": float(entry_f),
                    "exit_price": float(exit_f),
                }
            )

    def _split_legs_for_display(self, legs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """Split legs into (main, hedge_options, hedge_underlyings).

        - Main legs: is_hedge is false
        - Hedge option legs: is_hedge true AND option-like symbol
        - Hedge underlying legs: is_hedge true AND not option-like

        Hedge underlyings are shown as a separate row in the Paper Trade Log
        (new position type) to avoid confusing the trade as having multiple
        independent legs.
        """

        main: list[dict] = []
        hedge_opts: list[dict] = []
        hedge_under: list[dict] = []
        for leg in legs or []:
            if not isinstance(leg, dict):
                continue
            try:
                is_hedge = bool(leg.get("is_hedge"))
            except Exception:
                is_hedge = False
            if not is_hedge:
                main.append(leg)
            else:
                if self._is_option_leg(leg):
                    hedge_opts.append(leg)
                else:
                    hedge_under.append(leg)
        return main, hedge_opts, hedge_under

    def _format_net_hedge_legs(self, hedge_legs: list[dict], *, include_flat: bool = True) -> str:
        """Return a compact net view for underlying hedge legs.

        Example:
        "NSE:NIFTYBEES BUY x50 (B150/S100, entry 241.25, last 242.10)"
        """

        by_symbol: dict[str, dict[str, float]] = {}
        order: list[str] = []

        for leg in hedge_legs or []:
            if not isinstance(leg, dict):
                continue
            sym = str(leg.get("symbol") or "").strip()
            if not sym:
                continue
            try:
                qty = int(leg.get("quantity") or 0)
            except Exception:
                qty = 0
            if qty < 0:
                qty = abs(qty)
            if qty <= 0:
                continue

            side = str(leg.get("side") or "").upper().strip()
            if side.startswith("B"):
                sign = 1
            elif side.startswith("S"):
                sign = -1
            else:
                continue

            if sym not in by_symbol:
                by_symbol[sym] = {
                    "buy": 0.0,
                    "sell": 0.0,
                    "net": 0.0,
                    "legs_count": 0.0,
                    "buy_entry_notional": 0.0,
                    "buy_entry_qty": 0.0,
                    "sell_entry_notional": 0.0,
                    "sell_entry_qty": 0.0,
                    "entry_notional": 0.0,
                    "entry_qty": 0.0,
                    "last_notional": 0.0,
                    "last_qty": 0.0,
                    "exit_notional": 0.0,
                    "exit_qty": 0.0,
                }
                order.append(sym)

            rec = by_symbol[sym]
            if sign > 0:
                rec["buy"] += float(qty)
            else:
                rec["sell"] += float(qty)
            rec["net"] += float(sign * qty)
            rec["legs_count"] += 1.0

            try:
                entry_px = float(leg.get("entry_price")) if leg.get("entry_price") is not None else None
            except Exception:
                entry_px = None
            if entry_px is not None:
                rec["entry_notional"] += float(entry_px) * float(qty)
                rec["entry_qty"] += float(qty)
                if sign > 0:
                    rec["buy_entry_notional"] += float(entry_px) * float(qty)
                    rec["buy_entry_qty"] += float(qty)
                else:
                    rec["sell_entry_notional"] += float(entry_px) * float(qty)
                    rec["sell_entry_qty"] += float(qty)

            try:
                last_px = float(leg.get("ltp")) if leg.get("ltp") is not None else None
            except Exception:
                last_px = None
            if last_px is not None:
                rec["last_notional"] += float(last_px) * float(qty)
                rec["last_qty"] += float(qty)
            else:
                try:
                    exit_px = float(leg.get("exit_price")) if leg.get("exit_price") is not None else None
                except Exception:
                    exit_px = None
                if exit_px is not None:
                    rec["exit_notional"] += float(exit_px) * float(qty)
                    rec["exit_qty"] += float(qty)

        parts: list[str] = []
        for sym in order:
            rec = by_symbol.get(sym) or {}
            net = int(rec.get("net", 0.0) or 0.0)
            buy_q = int(rec.get("buy", 0.0) or 0.0)
            sell_q = int(rec.get("sell", 0.0) or 0.0)
            legs_count = int(rec.get("legs_count", 0.0) or 0.0)

            if net > 0:
                net_txt = f"BUY x{net}"
            elif net < 0:
                net_txt = f"SELL x{abs(net)}"
            else:
                if not include_flat:
                    continue
                net_txt = "FLAT"

            extra: list[str] = [f"B{buy_q}/S{sell_q}"]
            if legs_count > 0:
                extra.append(f"legs {legs_count}")

            buy_entry_qty = float(rec.get("buy_entry_qty", 0.0) or 0.0)
            if buy_entry_qty > 0:
                buy_avg = float(rec.get("buy_entry_notional", 0.0) or 0.0) / buy_entry_qty
                extra.append(f"buy avg {buy_avg:.2f}")

            sell_entry_qty = float(rec.get("sell_entry_qty", 0.0) or 0.0)
            if sell_entry_qty > 0:
                sell_avg = float(rec.get("sell_entry_notional", 0.0) or 0.0) / sell_entry_qty
                extra.append(f"sell avg {sell_avg:.2f}")

            e_qty = float(rec.get("entry_qty", 0.0) or 0.0)
            if e_qty > 0:
                e_avg = float(rec.get("entry_notional", 0.0) or 0.0) / e_qty
                extra.append(f"entry {e_avg:.2f}")

            l_qty = float(rec.get("last_qty", 0.0) or 0.0)
            x_qty = float(rec.get("exit_qty", 0.0) or 0.0)
            if l_qty > 0:
                l_avg = float(rec.get("last_notional", 0.0) or 0.0) / l_qty
                extra.append(f"last {l_avg:.2f}")
            elif x_qty > 0:
                x_avg = float(rec.get("exit_notional", 0.0) or 0.0) / x_qty
                extra.append(f"exit {x_avg:.2f}")

            parts.append(f"{sym} {net_txt} ({', '.join(extra)})")

        return " | ".join(parts)[:500]

    def _avg_entry_minus_exit_today(self) -> tuple[float | None, int]:
        try:
            today = datetime.now().date()
        except Exception:
            today = self._closed_option_legs_day
        if today != self._closed_option_legs_day:
            self._closed_option_legs_day = today
            self._closed_option_legs = []

        # Quantity-weighted average P/L per unit (points) for closed option legs today.
        # For BUY legs: P/L per unit = (exit - entry)
        # For SELL legs: P/L per unit = (entry - exit)
        total_qty = 0
        total_diff_qty = 0.0
        for r in self._closed_option_legs:
            try:
                q = int(r.get("quantity") or 0)
                e = float(r.get("entry_price"))
                x = float(r.get("exit_price"))
                side = str(r.get("side") or "").strip().upper()
            except Exception:
                continue
            if q <= 0:
                continue
            total_qty += q
            if side == "BUY":
                diff = (x - e)
            elif side == "SELL":
                diff = (e - x)
            else:
                # Without side we can't determine profit direction reliably.
                # Skip to avoid flipping the sign.
                total_qty -= q
                continue
            total_diff_qty += float(diff) * float(q)
        if total_qty <= 0:
            return None, 0
        avg = float(total_diff_qty) / float(total_qty)
        return float(avg), int(total_qty)

    # _toggle_pnl_totals removed: totals bar is always visible.

    def _build_header(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, expand=False, padx=10, pady=(10, 0))

        left = ttk.Frame(bar)
        left.pack(side=tk.LEFT, anchor="w")
        ttk.Label(left, text="Scalper Bot", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)

        mid = ttk.Frame(bar)
        mid.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(18, 18))
        ttk.Label(mid, text="Status:").pack(side=tk.LEFT)
        ttk.Label(mid, textvariable=self.status_var).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(mid, text="Spot:").pack(side=tk.LEFT)
        ttk.Label(mid, textvariable=self._dash_spot_var).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(mid, text="Last tick:").pack(side=tk.LEFT)
        ttk.Label(mid, textvariable=self._dash_last_tick_var).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(mid, text="Candles:").pack(side=tk.LEFT)
        ttk.Label(mid, textvariable=self._dash_candles_var).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(mid, text="Token:").pack(side=tk.LEFT)
        ttk.Label(mid, textvariable=self._dash_token_var).pack(side=tk.LEFT, padx=(6, 0))

        right = ttk.Frame(bar)
        right.pack(side=tk.RIGHT, anchor="e")

        # Use these as the canonical control buttons (methods assume these names).
        self.btn_request_otp = ttk.Button(right, text="Request OTP", command=self._on_request_otp)
        self.btn_verify_otp = ttk.Button(right, text="Verify OTP", command=self._on_verify_otp)
        ttk.Separator(right, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.btn_start = ttk.Button(right, text="Start Bot", command=self._on_start)
        self.btn_stop = ttk.Button(right, text="Stop Bot", command=self._on_stop, state=tk.DISABLED)

        self.btn_request_otp.pack(side=tk.LEFT)
        self.btn_verify_otp.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_start.pack(side=tk.LEFT, padx=(12, 0))
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=(10, 0))

    def _build_widgets(self) -> None:
        # --- Global header (primary actions + live status) ---
        self._build_header()

        # --- Tabbed Main Area ---
        # We keep the app in a single window, but split it into tabs so the UI
        # doesn't become a long, crowded form as features grow.
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.dashboard_frame = ttk.Frame(self.notebook)
        self.trade_frame = ttk.Frame(self.notebook)
        self.signals_frame = ttk.Frame(self.notebook)
        self.gpt_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.dashboard_frame, text="Live Dashboard")
        self.notebook.add(self.trade_frame, text="Trade Logs")
        self.notebook.add(self.signals_frame, text="Signals/Greeks")
        self.notebook.add(self.gpt_frame, text="GPT Advisor")
        self.notebook.add(self.settings_frame, text="Settings")

        # PnL/today stats (used in both the global totals bar and Live Dashboard).
        self._pnl_profit_var = tk.StringVar(value="₹0.00")
        self._pnl_loss_var = tk.StringVar(value="₹0.00")
        self._avg_entry_minus_exit_var = tk.StringVar(value="n/a")
        self._qty_traded_today_var = tk.StringVar(value="0")
        self._hedge_qty_traded_today_var = tk.StringVar(value="0")
        self._margin_required_var = tk.StringVar(value="n/a")
        self._diag_router_var = tk.StringVar(value="Router: n/a")
        self._diag_last_block_var = tk.StringVar(value="Last block: n/a")
        self._diag_top_block_var = tk.StringVar(value="Top block: n/a")
        self._diag_decisions_var = tk.StringVar(value="Decisions: n/a")
        self._diag_exec_var = tk.StringVar(value="Executed: n/a")
        self._diag_risk_var = tk.StringVar(value="Risk: n/a")
        self._diag_preset_req_var = tk.StringVar(value="GPT preset request: n/a")

        # --- Live Dashboard ---
        outer = ttk.Frame(self.dashboard_frame)
        outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(2, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="we")
        ttk.Label(header, text="Live Dashboard", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        cards = ttk.Frame(outer)
        cards.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        cards.grid_columnconfigure(0, weight=1)
        cards.grid_columnconfigure(1, weight=1)
        cards.grid_columnconfigure(2, weight=1)
        cards.grid_rowconfigure(0, weight=1)

        # Bot status card
        bot_card = ttk.LabelFrame(cards, text="Bot")
        bot_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        bot_card.grid_columnconfigure(1, weight=1)

        ttk.Label(bot_card, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(bot_card, textvariable=self.status_var).grid(row=0, column=1, sticky="w")

        ttk.Label(bot_card, text="Last tick:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(bot_card, textvariable=self._dash_last_tick_var).grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(bot_card, text="Candles:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(bot_card, textvariable=self._dash_candles_var).grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(bot_card, text="Spot:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Label(bot_card, textvariable=self._dash_spot_var).grid(row=3, column=1, sticky="w", pady=(6, 0))

        # Session card
        ses_card = ttk.LabelFrame(cards, text="Session")
        ses_card.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        ses_card.grid_columnconfigure(1, weight=1)

        ttk.Label(ses_card, text="Access token:").grid(row=0, column=0, sticky="w")
        ttk.Label(ses_card, textvariable=self._dash_token_var).grid(row=0, column=1, sticky="w")

        ttk.Label(ses_card, text="Use top bar for OTP / Start / Stop.").grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Engine diagnostics
        eng_card = ttk.LabelFrame(cards, text="Engine Diagnostics")
        eng_card.grid(row=0, column=2, sticky="nsew")
        eng_card.grid_columnconfigure(0, weight=1)
        ttk.Label(eng_card, textvariable=self._diag_router_var).grid(row=0, column=0, sticky="w")
        ttk.Label(eng_card, textvariable=self._diag_last_block_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(eng_card, textvariable=self._diag_top_block_var).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(eng_card, textvariable=self._diag_decisions_var).grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Label(eng_card, textvariable=self._diag_exec_var).grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Label(eng_card, textvariable=self._diag_risk_var).grid(row=5, column=0, sticky="w", pady=(4, 0))
        ttk.Label(eng_card, textvariable=self._diag_preset_req_var).grid(row=6, column=0, sticky="w", pady=(4, 0))

        # --- Portfolio (live) ---
        portfolio = ttk.Frame(outer)
        portfolio.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        portfolio.grid_columnconfigure(0, weight=1)
        # Give both tables space; summary stays compact.
        portfolio.grid_rowconfigure(1, weight=3)
        portfolio.grid_rowconfigure(2, weight=2)

        self._dash_portfolio_summary_var = tk.StringVar(value="Updated n/a | Exposure: Equity ₹0 (0%) | Options ₹0 (0%) | Counts: Eq 0 | Opt 0 trades / 0 legs")
        summary = ttk.LabelFrame(portfolio, text="Portfolio Summary")
        summary.grid(row=0, column=0, sticky="we")
        ttk.Label(summary, textvariable=self._dash_portfolio_summary_var).pack(anchor="w", padx=10, pady=6)

        # Also show today's realized stats + turnover + margin inside Live Dashboard.
        dash_stats = ttk.Frame(summary)
        dash_stats.pack(anchor="w", padx=10, pady=(0, 6), fill=tk.X)
        dash_stats_row1 = ttk.Frame(dash_stats)
        dash_stats_row2 = ttk.Frame(dash_stats)
        dash_stats_row1.pack(fill=tk.X, expand=False)
        dash_stats_row2.pack(fill=tk.X, expand=False, pady=(2, 0))

        ttk.Label(dash_stats_row1, text="Total Profit:").pack(side=tk.LEFT)
        ttk.Label(dash_stats_row1, textvariable=self._pnl_profit_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(dash_stats_row1, text="Total Loss:").pack(side=tk.LEFT)
        ttk.Label(dash_stats_row1, textvariable=self._pnl_loss_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(dash_stats_row1, text="Avg (E-X) Today:").pack(side=tk.LEFT)
        ttk.Label(dash_stats_row1, textvariable=self._avg_entry_minus_exit_var).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(dash_stats_row2, text="Turnover Qty:").pack(side=tk.LEFT)
        ttk.Label(dash_stats_row2, textvariable=self._qty_traded_today_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(dash_stats_row2, text="Hedge Turnover Qty:").pack(side=tk.LEFT)
        ttk.Label(dash_stats_row2, textvariable=self._hedge_qty_traded_today_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(dash_stats_row2, text="Margin Req:").pack(side=tk.LEFT)
        ttk.Label(dash_stats_row2, textvariable=self._margin_required_var).pack(side=tk.LEFT, padx=(6, 0))

        opt_card = ttk.LabelFrame(portfolio, text="Options (Open Legs)")
        opt_card.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        opt_card.grid_rowconfigure(0, weight=1)
        opt_card.grid_columnconfigure(0, weight=1)

        opt_cols = (
            "trade_id",
            "strategy",
            "symbol",
            "side",
            "qty",
            "entry",
            "ltp",
            "leg_mtm",
            "trade_mtm",
            "base_mtm",
            "hedge_mtm",
            "sl",
            "tgt",
            "status",
        )
        opt_area = ttk.Frame(opt_card)
        opt_area.grid(row=0, column=0, sticky="nsew")
        opt_area.grid_rowconfigure(0, weight=1)
        opt_area.grid_columnconfigure(0, weight=1)
        self.dash_opt_tree = ttk.Treeview(opt_area, columns=opt_cols, show="headings", height=8)
        for c, title in (
            ("trade_id", "ID"),
            ("strategy", "Strategy"),
            ("symbol", "Symbol"),
            ("side", "Side"),
            ("qty", "Qty"),
            ("entry", "Entry"),
            ("ltp", "LTP"),
            ("leg_mtm", "Leg MTM"),
            ("trade_mtm", "Trade MTM"),
            ("base_mtm", "Base MTM"),
            ("hedge_mtm", "Hedge MTM"),
            ("sl", "Stop"),
            ("tgt", "Target"),
            ("status", "Status"),
        ):
            self.dash_opt_tree.heading(c, text=title)
        self.dash_opt_tree.column("trade_id", width=60, stretch=False, anchor="w")
        self.dash_opt_tree.column("strategy", width=120, stretch=True, anchor="w")
        self.dash_opt_tree.column("symbol", width=210, stretch=True, anchor="w")
        self.dash_opt_tree.column("side", width=55, stretch=False, anchor="w")
        self.dash_opt_tree.column("qty", width=55, stretch=False, anchor="e")
        self.dash_opt_tree.column("entry", width=70, stretch=False, anchor="e")
        self.dash_opt_tree.column("ltp", width=70, stretch=False, anchor="e")
        self.dash_opt_tree.column("leg_mtm", width=80, stretch=False, anchor="e")
        self.dash_opt_tree.column("trade_mtm", width=80, stretch=False, anchor="e")
        self.dash_opt_tree.column("base_mtm", width=80, stretch=False, anchor="e")
        self.dash_opt_tree.column("hedge_mtm", width=80, stretch=False, anchor="e")
        self.dash_opt_tree.column("sl", width=90, stretch=False, anchor="e")
        self.dash_opt_tree.column("tgt", width=90, stretch=False, anchor="e")
        self.dash_opt_tree.column("status", width=140, stretch=True, anchor="w")

        opt_vsb = ttk.Scrollbar(opt_area, orient="vertical", command=self.dash_opt_tree.yview)
        opt_hsb = ttk.Scrollbar(opt_area, orient="horizontal", command=self.dash_opt_tree.xview)
        self.dash_opt_tree.configure(yscrollcommand=opt_vsb.set, xscrollcommand=opt_hsb.set)
        self.dash_opt_tree.grid(row=0, column=0, sticky="nsew")
        opt_vsb.grid(row=0, column=1, sticky="ns")
        opt_hsb.grid(row=1, column=0, sticky="ew")

        eq_card = ttk.LabelFrame(portfolio, text="Equities (Bot-managed Positions)")
        eq_card.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        eq_card.grid_rowconfigure(0, weight=1)
        eq_card.grid_columnconfigure(0, weight=1)

        eq_cols = ("symbol", "side", "qty", "entry", "ltp", "pnl", "sl", "tgt")
        eq_area = ttk.Frame(eq_card)
        eq_area.grid(row=0, column=0, sticky="nsew")
        eq_area.grid_rowconfigure(0, weight=1)
        eq_area.grid_columnconfigure(0, weight=1)
        self.dash_eq_tree = ttk.Treeview(eq_area, columns=eq_cols, show="headings", height=6)
        for c, title in (
            ("symbol", "Symbol"),
            ("side", "Side"),
            ("qty", "Qty"),
            ("entry", "Entry"),
            ("ltp", "LTP"),
            ("pnl", "PnL"),
            ("sl", "Stop"),
            ("tgt", "Target"),
        ):
            self.dash_eq_tree.heading(c, text=title)
        self.dash_eq_tree.column("symbol", width=200, stretch=True, anchor="w")
        self.dash_eq_tree.column("side", width=60, stretch=False, anchor="w")
        self.dash_eq_tree.column("qty", width=70, stretch=False, anchor="e")
        self.dash_eq_tree.column("entry", width=80, stretch=False, anchor="e")
        self.dash_eq_tree.column("ltp", width=80, stretch=False, anchor="e")
        self.dash_eq_tree.column("pnl", width=90, stretch=False, anchor="e")
        self.dash_eq_tree.column("sl", width=90, stretch=False, anchor="e")
        self.dash_eq_tree.column("tgt", width=90, stretch=False, anchor="e")

        eq_vsb = ttk.Scrollbar(eq_area, orient="vertical", command=self.dash_eq_tree.yview)
        eq_hsb = ttk.Scrollbar(eq_area, orient="horizontal", command=self.dash_eq_tree.xview)
        self.dash_eq_tree.configure(yscrollcommand=eq_vsb.set, xscrollcommand=eq_hsb.set)
        self.dash_eq_tree.grid(row=0, column=0, sticky="nsew")
        eq_vsb.grid(row=0, column=1, sticky="ns")
        eq_hsb.grid(row=1, column=0, sticky="ew")

        # Analytics/Charts removed from UI.

        # Settings tab: scrollable form (prevents cramped/overflow on smaller screens).
        settings_outer = ttk.Frame(self.settings_frame)
        settings_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_canvas = tk.Canvas(settings_outer, highlightthickness=0)
        try:
            style = ttk.Style(self)
            bg = style.lookup("TFrame", "background")
            if bg:
                settings_canvas.configure(background=bg)
        except Exception:
            pass
        settings_vsb = ttk.Scrollbar(settings_outer, orient="vertical", command=settings_canvas.yview)
        settings_canvas.configure(yscrollcommand=settings_vsb.set)
        settings_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frm = ttk.Frame(settings_canvas)
        frm_id = settings_canvas.create_window((0, 0), window=frm, anchor="nw")

        def _on_settings_configure(_evt: object = None) -> None:
            try:
                settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
            except Exception:
                return

        def _on_settings_canvas_configure(evt: object) -> None:
            try:
                width = int(getattr(evt, "width"))
            except Exception:
                return
            try:
                settings_canvas.itemconfigure(frm_id, width=width)
            except Exception:
                return

        frm.bind("<Configure>", lambda e: _on_settings_configure(e))
        settings_canvas.bind("<Configure>", _on_settings_canvas_configure)

        # Mouse wheel scrolling (Windows/Linux/macOS best-effort).
        def _settings_mousewheel(evt: object) -> None:
            try:
                delta = int(getattr(evt, "delta"))
            except Exception:
                delta = 0
            if delta:
                settings_canvas.yview_scroll(int(-delta / 120), "units")

        def _settings_linux_scroll(evt: object) -> None:
            num = getattr(evt, "num", None)
            if num == 4:
                settings_canvas.yview_scroll(-1, "units")
            elif num == 5:
                settings_canvas.yview_scroll(1, "units")

        # On Windows, <MouseWheel> is delivered to the focused widget.
        settings_canvas.bind("<Enter>", lambda _e: settings_canvas.focus_set())
        settings_canvas.bind("<MouseWheel>", _settings_mousewheel)
        settings_canvas.bind("<Button-4>", _settings_linux_scroll)
        settings_canvas.bind("<Button-5>", _settings_linux_scroll)

        # --- Credentials ---
        row = 0

        def _section(title: str) -> None:
            nonlocal row
            ttk.Label(frm, text=title, font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(14 if row else 0, 6))
            row += 1
            ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="we", pady=(0, 10))
            row += 1

        _section("Credentials")
        ttk.Label(frm, text="API Key").grid(row=row, column=0, sticky="w")
        self.api_key_var = tk.StringVar(value="")
        self.api_key_entry = ttk.Entry(frm, textvariable=self.api_key_var, width=60)
        self.api_key_entry.grid(row=row, column=1, sticky="we", padx=5)

        row += 1
        ttk.Label(frm, text="Username (Client code)").grid(row=row, column=0, sticky="w")
        self.username_var = tk.StringVar(value="")
        self.username_entry = ttk.Entry(frm, textvariable=self.username_var, width=60)
        self.username_entry.grid(row=row, column=1, sticky="we", padx=5)

        row += 1
        ttk.Label(frm, text="Password").grid(row=row, column=0, sticky="w")
        self.password_var = tk.StringVar(value="")
        self.password_entry = ttk.Entry(frm, textvariable=self.password_var, show="*", width=60)
        self.password_entry.grid(row=row, column=1, sticky="we", padx=5)

        row += 1
        ttk.Label(frm, text="OTP").grid(row=row, column=0, sticky="w")
        self.otp_var = tk.StringVar(value="")
        self.otp_entry = ttk.Entry(frm, textvariable=self.otp_var, width=20)
        self.otp_entry.grid(row=row, column=1, sticky="w", padx=5)

        # Strategy selection
        row += 1
        ttk.Label(frm, text="Strategy").grid(row=row, column=0, sticky="w")
        self.strategy_var = tk.StringVar(value=os.getenv("MSTOCK_STRATEGY", "directional"))
        if str(self.strategy_var.get() or "").strip() not in set(_STRATEGY_CHOICES):
            self.strategy_var.set(_STRATEGY_CHOICES[0])
        ttk.OptionMenu(frm, self.strategy_var, self.strategy_var.get(), *_STRATEGY_CHOICES).grid(
            row=row, column=1, sticky="w", padx=5
        )

        row += 1
        ttk.Label(frm, text="Delta hedge scope").grid(row=row, column=0, sticky="w")
        self.delta_hedge_scope_var = tk.StringVar(value=os.getenv("MSTOCK_DELTA_HEDGE_SCOPE", "strategy_only"))
        if str(self.delta_hedge_scope_var.get() or "").strip() not in {"strategy_only", "multi_only", "all_options"}:
            self.delta_hedge_scope_var.set("strategy_only")
        ttk.OptionMenu(
            frm,
            self.delta_hedge_scope_var,
            self.delta_hedge_scope_var.get(),
            "strategy_only",
            "multi_only",
            "all_options",
        ).grid(row=row, column=1, sticky="w", padx=5)

        row += 1
        ttk.Label(frm, text="OTM distance (pts) (for strangle)").grid(row=row, column=0, sticky="w")
        self.distance_var = tk.StringVar(value=os.getenv("MSTOCK_SHORT_STRIKE_DISTANCE", "50"))
        ttk.Entry(frm, textvariable=self.distance_var, width=12).grid(row=row, column=1, sticky="w", padx=5)



        row += 1
        ttk.Label(frm, text="Timeframe").grid(row=row, column=0, sticky="w")
        def _normalize_ui_timeframe(value: object, default: str = "1m") -> str:
            raw = str(value or "").strip().lower()
            aliases = {
                "1m": "1m",
                "m1": "1m",
                "one_minute": "1m",
                "3m": "3m",
                "m3": "3m",
                "three_minute": "3m",
                "5m": "5m",
                "m5": "5m",
                "five_minute": "5m",
                "10m": "10m",
                "m10": "10m",
                "ten_minute": "10m",
                "15m": "15m",
                "m15": "15m",
                "fifteen_minute": "15m",
                "30m": "30m",
                "m30": "30m",
                "thirty_minute": "30m",
            }
            return aliases.get(raw, default)

        self.timeframe_var = tk.StringVar(value=_normalize_ui_timeframe(os.getenv("MSTOCK_TIMEFRAME", "1m"), "1m"))
        ttk.OptionMenu(
            frm,
            self.timeframe_var,
            self.timeframe_var.get(),
            "1m",
            "3m",
            "5m",
            "10m",
            "15m",
            "30m",
        ).grid(row=row, column=1, sticky="w", padx=5)

        row += 1
        ttk.Label(frm, text="Preset").grid(row=row, column=0, sticky="w")
        preset_frame = ttk.Frame(frm)
        preset_frame.grid(row=row, column=1, sticky="w", padx=5)
        _preset_env = (os.getenv("MSTOCK_PRESET", "") or "").strip().lower()
        _preset_ui = "(none)"
        if _preset_env in {"aggressive", "aggresive"}:
            _preset_ui = "Aggressive"
        elif _preset_env == "conservative":
            _preset_ui = "Conservative"
        self.preset_var = tk.StringVar(value=_preset_ui)
        ttk.OptionMenu(
            preset_frame,
            self.preset_var,
            self.preset_var.get(),
            "(none)",
            "Aggressive",
            "Conservative",
        ).pack(side=tk.LEFT)

        def _apply_quick_preset() -> None:
            name = str(self.preset_var.get() or "").strip().lower()
            current_strategy = str(self.strategy_var.get() or "").strip().lower()
            if name == "aggressive":
                if current_strategy != "auto":
                    self.strategy_var.set("auto")
                self.timeframe_var.set("1m")
                if hasattr(self, "delta_hedge_scope_var"):
                    self.delta_hedge_scope_var.set("all_options")
            elif name == "conservative":
                if current_strategy != "auto":
                    self.strategy_var.set("auto")
                self.timeframe_var.set("3m")
                if hasattr(self, "delta_hedge_scope_var"):
                    self.delta_hedge_scope_var.set("all_options")

        ttk.Button(preset_frame, text="Apply", command=_apply_quick_preset).pack(side=tk.LEFT, padx=(6, 0))

        # Beginner defaults: keep UI minimal.
        row += 1
        self.live_var = tk.BooleanVar(
            value=os.getenv("MSTOCK_ENABLE_LIVE_TRADING", "false").lower() in {"1", "true", "yes", "y"}
        )
        ttk.Checkbutton(frm, text="Enable live trading", variable=self.live_var).grid(row=row, column=1, sticky="w", padx=5)

        row += 1
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(btns, text="Session actions are available in the top bar.").pack(side=tk.LEFT, padx=(0, 12))

        self.remember_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btns, text="Remember API key/user/password", variable=self.remember_var).pack(
            side=tk.LEFT, padx=(18, 6)
        )

        self.edit_creds_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            btns,
            text="Edit credentials",
            variable=self.edit_creds_var,
            command=self._sync_credential_editability,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.btn_save_creds = ttk.Button(btns, text="Save", command=self._on_save_credentials)
        self.btn_save_creds.pack(side=tk.LEFT, padx=(0, 6))

        # Quick access to view/tweak current strategy settings.
        self.btn_show_settings = ttk.Button(btns, text="Settings...", command=self._on_show_settings)
        self.btn_show_settings.pack(side=tk.LEFT, padx=(12, 0))

        # --- Token ---
        _section("Session")
        ttk.Label(frm, text="Access Token").grid(row=row, column=0, sticky="w", pady=(10, 0))
        self.access_token_entry = ttk.Entry(frm, textvariable=self.access_token_var, width=60)
        self.access_token_entry.grid(row=row, column=1, sticky="we", padx=5, pady=(10, 0))

        row += 1
        token_actions = ttk.Frame(frm)
        token_actions.grid(row=row, column=1, sticky="w", padx=5, pady=(6, 0))
        ttk.Button(token_actions, text="Copy token", command=self._copy_token).pack(side=tk.LEFT)
        ttk.Button(token_actions, text="Clear token", command=self._clear_token).pack(side=tk.LEFT, padx=(6, 0))

        # --- Underlying (for candles) ---
        _section("Market Data")
        ttk.Label(frm, text="Underlying Token (for candles)").grid(row=row, column=0, sticky="w", pady=(10, 0))
        self.underlying_token_var = tk.StringVar(value=os.getenv("MSTOCK_UNDERLYING_TOKEN", ""))
        ttk.Entry(frm, textvariable=self.underlying_token_var, width=24).grid(
            row=row, column=1, sticky="w", padx=5, pady=(10, 0)
        )

        row += 1
        ttk.Label(
            frm,
            text="Tip: If unsure, leave this blank and let the bot auto-resolve the m.Stock token.",
        ).grid(row=row, column=1, sticky="w", padx=5)

        row += 1
        ttk.Label(frm, text="Underlying Exchange").grid(row=row, column=0, sticky="w")
        self.underlying_exchange_var = tk.StringVar(
            value=os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE"))
        )
        ttk.Entry(frm, textvariable=self.underlying_exchange_var, width=10).grid(row=row, column=1, sticky="w", padx=5)

        row += 1
        self.use_intraday_chart_var = tk.BooleanVar(
            value=os.getenv("MSTOCK_USE_INTRADAY_CHART", "false").lower() in {"1", "true", "yes", "y"}
        )
        ttk.Checkbutton(
            frm,
            text="Use m.Stock intraday chart API for candles (current day)",
            variable=self.use_intraday_chart_var,
        ).grid(row=row, column=1, sticky="w", padx=5)

        row += 1
        self.startup_use_historical_candles_var = tk.BooleanVar(
            value=os.getenv("MSTOCK_STARTUP_USE_HISTORICAL_CANDLES", "true").lower() in {"1", "true", "yes", "y"}
        )
        ttk.Checkbutton(
            frm,
            text="At startup, warm up spot candles using m.Stock historical endpoint",
            variable=self.startup_use_historical_candles_var,
        ).grid(row=row, column=1, sticky="w", padx=5)

        # Target options expiry for contracts (e.g. 06-01-2026).
        row += 1
        ttk.Label(frm, text="Options Expiry (DD-MM-YYYY)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.target_expiry_var = tk.StringVar(value=os.getenv("MSTOCK_TARGET_EXPIRY", ""))
        self.target_expiry_combo = ttk.Combobox(frm, textvariable=self.target_expiry_var, width=14)
        self.target_expiry_combo.grid(
            row=row,
            column=1,
            sticky="w",
            padx=5,
            pady=(6, 0),
        )

        # --- Options token mapping (ScripMaster CSV) ---

        row += 1
        ttk.Label(frm, text="ScripMaster CSV (options)").grid(row=row, column=0, sticky="w", pady=(10, 0))
        self.scripmaster_path_var = tk.StringVar(value=os.getenv("MSTOCK_SCRIPMASTER_PATH", ""))
        sm_row = ttk.Frame(frm)
        sm_row.grid(row=row, column=1, sticky="we", padx=5, pady=(10, 0))
        ttk.Entry(sm_row, textvariable=self.scripmaster_path_var, width=52).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _refresh_main_expiries(*args) -> None:
            # Prefer the UI field so browsing a new CSV updates immediately
            # (even if an older value is still present in the process env).
            sm_path = self.scripmaster_path_var.get().strip() or os.getenv("MSTOCK_SCRIPMASTER_PATH", "")
            if sm_path and Path(sm_path).exists():
                try:
                    from scripmaster import ScripMaster
                    sm = ScripMaster(sm_path)
                    u = os.getenv("MSTOCK_UNDERLYING", os.getenv("MSTOCK_SYMBOL", "NIFTY"))
                    exps = sm.get_available_expiries(u)
                    if exps:
                        self.target_expiry_combo["values"] = [d.strftime("%d-%m-%Y") for d in exps]
                except Exception:
                    pass

        _refresh_main_expiries()
        self.scripmaster_path_var.trace_add("write", _refresh_main_expiries)


        def browse_sm() -> None:
            p = filedialog.askopenfilename(
                title="Select ScripMaster CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*")],
            )
            if p:
                self.scripmaster_path_var.set(p)

        ttk.Button(sm_row, text="Browse...", command=browse_sm).pack(side=tk.LEFT, padx=(6, 0))
        row += 1
        self.auto_fetch_scripmaster_var = tk.BooleanVar(
            value=os.getenv("MSTOCK_AUTO_FETCH_SCRIPMASTER", "true").lower() in {"1", "true", "yes", "y"}
        )
        ttk.Checkbutton(
            frm,
            text="Auto-fetch ScripMaster daily if no CSV configured",
            variable=self.auto_fetch_scripmaster_var,
        ).grid(row=row, column=1, sticky="w", padx=5)

        # Simple status label near the controls so it's obvious when the
        # bot is running, without needing to parse the full log.
        row += 1
        ttk.Label(frm, text="Status").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Label(frm, textvariable=self.status_var).grid(row=row, column=1, sticky="w", padx=5, pady=(10, 0))

        frm.grid_columnconfigure(1, weight=1)

        # --- Trade Logs tab ---
        # Configure treeview style for a modern look
        style = ttk.Style()
        try:
            style.configure("Treeview", rowheight=28)
            style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        except Exception:
            pass

        cols = ("time", "trade_id", "pos_type", "strategy", "legs", "mtm", "realized", "status")
        trade_tools = ttk.Frame(self.trade_frame)
        trade_tools.pack(fill=tk.X, expand=False, pady=(0, 8))
        ttk.Button(trade_tools, text="Export Today's PDF", command=self._export_today_trade_log_pdf).pack(side=tk.LEFT)

        # Make the trade log tall enough to show many positions without scrolling.
        tree_area = ttk.Frame(self.trade_frame)
        tree_area.pack(fill=tk.BOTH, expand=True)

        self.trade_tree = ttk.Treeview(tree_area, columns=cols, show="headings", height=16)
        # Enable sortable headings.
        self._trade_sort_desc: dict[str, bool] = {}
        self.trade_tree.heading("time", text="Time", command=lambda: self._sort_trade_log("time"))
        self.trade_tree.heading("trade_id", text="ID", command=lambda: self._sort_trade_log("trade_id"))
        self.trade_tree.heading("pos_type", text="Type", command=lambda: self._sort_trade_log("pos_type"))
        self.trade_tree.heading("strategy", text="Position", command=lambda: self._sort_trade_log("strategy"))
        self.trade_tree.heading("legs", text="Legs", command=lambda: self._sort_trade_log("legs"))
        self.trade_tree.heading("mtm", text="Live MTM", command=lambda: self._sort_trade_log("mtm"))
        self.trade_tree.heading("realized", text="P&L", command=lambda: self._sort_trade_log("realized"))
        self.trade_tree.heading("status", text="Status", command=lambda: self._sort_trade_log("status"))

        self.trade_tree.column("time", width=70, stretch=False, anchor="w")
        self.trade_tree.column("trade_id", width=60, stretch=False, anchor="w")
        self.trade_tree.column("pos_type", width=85, stretch=False, anchor="w")
        self.trade_tree.column("strategy", width=110, stretch=False, anchor="w")
        # Keep Legs wide and non-stretch so horizontal scrolling can reveal long text.
        self.trade_tree.column("legs", width=900, stretch=False, anchor="w")
        self.trade_tree.column("mtm", width=90, stretch=False, anchor="e")
        self.trade_tree.column("realized", width=80, stretch=False, anchor="e")
        # Status may also contain hedge summaries; keep it wide enough.
        self.trade_tree.column("status", width=260, stretch=False, anchor="w")

        vsb = ttk.Scrollbar(tree_area, orient="vertical", command=self.trade_tree.yview)
        hsb = ttk.Scrollbar(tree_area, orient="horizontal", command=self.trade_tree.xview)
        self.trade_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.trade_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree_area.grid_rowconfigure(0, weight=1)
        tree_area.grid_columnconfigure(0, weight=1)

        # Row color-coding (profit/loss).
        try:
            self.trade_tree.tag_configure("profit", foreground="green")
            self.trade_tree.tag_configure("loss", foreground="red")
            self.trade_tree.tag_configure("hedge", foreground="gray")
        except Exception:
            pass

        # Right-click context menu.
        self._trade_ctx_trade_id: str | None = None
        self._trade_ctx_menu = tk.Menu(self, tearoff=0)
        self._trade_ctx_menu.add_command(label="Manually Exit Leg", command=self._trade_ctx_manual_exit)
        self.trade_tree.bind("<Button-3>", self._trade_tree_right_click)

        # --- Signals/Greeks tab ---
        self._build_signals_tab()

        # --- GPT Advisor tab ---
        self._build_gpt_tab()

        self._pnl_totals_frame = ttk.Frame(self)
        # Use two rows so stats don't get clipped on smaller window widths.
        _pnl_row1 = ttk.Frame(self._pnl_totals_frame)
        _pnl_row2 = ttk.Frame(self._pnl_totals_frame)
        _pnl_row1.pack(fill=tk.X, expand=False)
        _pnl_row2.pack(fill=tk.X, expand=False, pady=(2, 0))

        ttk.Label(_pnl_row1, text="Total Profit:").pack(side=tk.LEFT)
        ttk.Label(_pnl_row1, textvariable=self._pnl_profit_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(_pnl_row1, text="Total Loss:").pack(side=tk.LEFT)
        ttk.Label(_pnl_row1, textvariable=self._pnl_loss_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(_pnl_row1, text="Avg (E-X) Today:").pack(side=tk.LEFT)
        ttk.Label(_pnl_row1, textvariable=self._avg_entry_minus_exit_var).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(_pnl_row2, text="Turnover Qty:").pack(side=tk.LEFT)
        ttk.Label(_pnl_row2, textvariable=self._qty_traded_today_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(_pnl_row2, text="Hedge Turnover Qty:").pack(side=tk.LEFT)
        ttk.Label(_pnl_row2, textvariable=self._hedge_qty_traded_today_var).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(_pnl_row2, text="Margin Req:").pack(side=tk.LEFT)
        ttk.Label(_pnl_row2, textvariable=self._margin_required_var).pack(side=tk.LEFT, padx=(6, 0))

        # --- Logs ---
        # Slightly shorter log area so the extra controls and trade log fit comfortably.
        self.log = ScrolledText(self, height=14)
        self.log.insert(tk.END, "Ready.\n")
        self.log.configure(state=tk.DISABLED)

        # PnL totals always above the log area.
        self._pnl_totals_frame.pack(fill=tk.X, expand=False, padx=10, pady=(0, 8))

        # Log toolbar (small QoL actions).
        log_tools = ttk.Frame(self)
        log_tools.pack(fill=tk.X, expand=False, padx=10, pady=(0, 6))

        def _copy_log_to_clipboard() -> None:
            try:
                txt = self.log.get("1.0", tk.END)
            except Exception:
                return
            try:
                self.clipboard_clear()
                self.clipboard_append(txt)
                self._app_status_var.set("Log copied to clipboard.")
            except Exception:
                pass

        def _clear_log() -> None:
            try:
                self.log.configure(state=tk.NORMAL)
                self.log.delete("1.0", tk.END)
                self.log.insert(tk.END, "Cleared.\n")
                self.log.configure(state=tk.DISABLED)
                self._app_status_var.set("Log cleared.")
            except Exception:
                pass

        ttk.Button(log_tools, text="Copy Log", command=_copy_log_to_clipboard).pack(side=tk.LEFT)
        ttk.Button(log_tools, text="Clear Log", command=_clear_log).pack(side=tk.LEFT, padx=(8, 0))

        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # Bottom status bar
        status_bar = ttk.Frame(self)
        status_bar.pack(fill=tk.X, expand=False, padx=10, pady=(0, 10))
        ttk.Separator(status_bar, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(status_bar, textvariable=self._app_status_var, anchor="w").pack(fill=tk.X, pady=(6, 0))

    def _build_signals_tab(self) -> None:
        root = ttk.Frame(self.signals_frame)
        root.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # Header / status
        self._sig_updated_var = tk.StringVar(value="Updated: n/a")
        self._sig_health_var = tk.StringVar(value="Waiting for candles…")

        hdr = ttk.Frame(root)
        hdr.pack(fill=tk.X, expand=False)
        ttk.Label(hdr, text="Signals & Greeks", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Label(hdr, textvariable=self._sig_updated_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Label(hdr, textvariable=self._sig_health_var).pack(side=tk.LEFT, padx=(14, 0))

        ttk.Button(hdr, text="Refresh now", command=lambda: self._render_signals_and_greeks(force=True)).pack(side=tk.RIGHT)

        ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 12))

        # Split tab into left (signals) and right (greeks table)
        pw = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=3)
        pw.add(right, weight=4)

        # ---- Latest candle / signals ----
        self._sig_symbol_var = tk.StringVar(value=(os.getenv("MSTOCK_UNDERLYING") or os.getenv("MSTOCK_SYMBOL") or "NIFTY").strip() or "NIFTY")
        self._sig_timeframe_var = tk.StringVar(value=(os.getenv("MSTOCK_TIMEFRAME") or "1m").strip() or "1m")
        self._sig_last_candle_var = tk.StringVar(value="n/a")
        self._sig_last_close_var = tk.StringVar(value="n/a")

        top = ttk.LabelFrame(left, text="Market")
        top.pack(fill=tk.X, expand=False, pady=(0, 8))
        ttk.Label(top, text="Symbol:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(top, textvariable=self._sig_symbol_var).grid(row=0, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(top, text="Timeframe:").grid(row=0, column=2, sticky="w", padx=8, pady=4)
        ttk.Label(top, textvariable=self._sig_timeframe_var).grid(row=0, column=3, sticky="w", padx=8, pady=4)
        ttk.Label(top, text="Last candle:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(top, textvariable=self._sig_last_candle_var).grid(row=1, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(top, text="Spot (close):").grid(row=1, column=2, sticky="w", padx=8, pady=4)
        ttk.Label(top, textvariable=self._sig_last_close_var).grid(row=1, column=3, sticky="w", padx=8, pady=4)
        for c in range(4):
            top.grid_columnconfigure(c, weight=1)

        self._ind_vars: dict[str, tk.StringVar] = {
            "EMA Fast": tk.StringVar(value="n/a"),
            "EMA Slow": tk.StringVar(value="n/a"),
            "RSI": tk.StringVar(value="n/a"),
            "ATR": tk.StringVar(value="n/a"),
            "ADX": tk.StringVar(value="n/a"),
            "Supertrend": tk.StringVar(value="n/a"),
            "ROC%": tk.StringVar(value="n/a"),
            "Choppiness": tk.StringVar(value="n/a"),
            "Pivot PP": tk.StringVar(value="n/a"),
        }

        ind = ttk.LabelFrame(left, text="Indicators")
        ind.pack(fill=tk.X, expand=False, pady=(0, 8))
        r = 0
        for k in ("EMA Fast", "EMA Slow", "RSI", "ATR", "ADX", "Supertrend", "ROC%", "Choppiness", "Pivot PP"):
            ttk.Label(ind, text=k + ":").grid(row=r, column=0, sticky="w", padx=8, pady=2)
            ttk.Label(ind, textvariable=self._ind_vars[k]).grid(row=r, column=1, sticky="w", padx=8, pady=2)
            r += 1
        ind.grid_columnconfigure(1, weight=1)

        self._pat_vars: dict[str, tk.StringVar] = {
            "Doji": tk.StringVar(value="n/a"),
            "Hammer": tk.StringVar(value="n/a"),
            "Inverted Hammer": tk.StringVar(value="n/a"),
            "Shooting Star": tk.StringVar(value="n/a"),
            "Hanging Man": tk.StringVar(value="n/a"),
            "Bullish Engulf": tk.StringVar(value="n/a"),
            "Bearish Engulf": tk.StringVar(value="n/a"),
            "Piercing Line": tk.StringVar(value="n/a"),
            "Dark Cloud Cover": tk.StringVar(value="n/a"),
            "Morning Star": tk.StringVar(value="n/a"),
            "Evening Star": tk.StringVar(value="n/a"),
            "Three White Soldiers": tk.StringVar(value="n/a"),
            "Three Black Crows": tk.StringVar(value="n/a"),
            "Rising Three Methods": tk.StringVar(value="n/a"),
            "Falling Three Methods": tk.StringVar(value="n/a"),
        }
        pat = ttk.LabelFrame(left, text="Candlestick Patterns")
        pat.pack(fill=tk.X, expand=False)
        pat_scroll_area = ttk.Frame(pat)
        pat_scroll_area.pack(fill=tk.BOTH, expand=True)

        pat_canvas = tk.Canvas(pat_scroll_area, height=170, highlightthickness=0, bd=0)
        pat_vsb = ttk.Scrollbar(pat_scroll_area, orient="vertical", command=pat_canvas.yview)
        pat_inner = ttk.Frame(pat_canvas)

        pat_canvas.configure(yscrollcommand=pat_vsb.set)
        pat_canvas.grid(row=0, column=0, sticky="nsew")
        pat_vsb.grid(row=0, column=1, sticky="ns")
        pat_scroll_area.grid_rowconfigure(0, weight=1)
        pat_scroll_area.grid_columnconfigure(0, weight=1)

        pat_window = pat_canvas.create_window((0, 0), window=pat_inner, anchor="nw")

        def _on_pat_inner_configure(_e: object) -> None:
            try:
                pat_canvas.configure(scrollregion=pat_canvas.bbox("all"))
            except Exception:
                pass

        def _on_pat_canvas_configure(e: object) -> None:
            try:
                width = int(getattr(e, "width", 0) or 0)
                if width > 0:
                    pat_canvas.itemconfigure(pat_window, width=width)
            except Exception:
                pass

        pat_inner.bind("<Configure>", _on_pat_inner_configure)
        pat_canvas.bind("<Configure>", _on_pat_canvas_configure)

        r = 0
        for k in (
            "Doji",
            "Hammer",
            "Inverted Hammer",
            "Shooting Star",
            "Hanging Man",
            "Bullish Engulf",
            "Bearish Engulf",
            "Piercing Line",
            "Dark Cloud Cover",
            "Morning Star",
            "Evening Star",
            "Three White Soldiers",
            "Three Black Crows",
            "Rising Three Methods",
            "Falling Three Methods",
        ):
            ttk.Label(pat_inner, text=k + ":").grid(row=r, column=0, sticky="w", padx=8, pady=2)
            ttk.Label(pat_inner, textvariable=self._pat_vars[k]).grid(row=r, column=1, sticky="w", padx=8, pady=2)
            r += 1
        pat_inner.grid_columnconfigure(1, weight=1)

        # ---- Greeks table ----
        self._greeks_summary_var = tk.StringVar(value="Net Δ: n/a | Avg IV: n/a")
        ttk.Label(right, textvariable=self._greeks_summary_var).pack(anchor="w", pady=(0, 6))

        cols = (
            "trade_id",
            "symbol",
            "side",
            "qty",
            "strike",
            "type",
            "expiry",
            "price",
            "iv",
            "delta",
            "gamma",
            "vega",
            "theta",
            "net_delta",
        )
        tree_area = ttk.Frame(right)
        tree_area.pack(fill=tk.BOTH, expand=True)
        self.greeks_tree = ttk.Treeview(tree_area, columns=cols, show="headings", height=14)
        for c in cols:
            self.greeks_tree.heading(c, text=c)

        self.greeks_tree.column("trade_id", width=70, stretch=False, anchor="w")
        self.greeks_tree.column("symbol", width=170, stretch=False, anchor="w")
        self.greeks_tree.column("side", width=55, stretch=False, anchor="w")
        self.greeks_tree.column("qty", width=55, stretch=False, anchor="e")
        self.greeks_tree.column("strike", width=70, stretch=False, anchor="e")
        self.greeks_tree.column("type", width=45, stretch=False, anchor="w")
        self.greeks_tree.column("expiry", width=95, stretch=False, anchor="w")
        self.greeks_tree.column("price", width=70, stretch=False, anchor="e")
        self.greeks_tree.column("iv", width=60, stretch=False, anchor="e")
        self.greeks_tree.column("delta", width=60, stretch=False, anchor="e")
        self.greeks_tree.column("gamma", width=70, stretch=False, anchor="e")
        self.greeks_tree.column("vega", width=70, stretch=False, anchor="e")
        self.greeks_tree.column("theta", width=70, stretch=False, anchor="e")
        self.greeks_tree.column("net_delta", width=80, stretch=False, anchor="e")

        vsb = ttk.Scrollbar(tree_area, orient="vertical", command=self.greeks_tree.yview)
        hsb = ttk.Scrollbar(tree_area, orient="horizontal", command=self.greeks_tree.xview)
        self.greeks_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.greeks_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_area.grid_rowconfigure(0, weight=1)
        tree_area.grid_columnconfigure(0, weight=1)

    def _build_gpt_tab(self) -> None:
        root = ttk.Frame(self.gpt_frame)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Configuration Section ---
        cfg = ttk.LabelFrame(root, text="GPT Configuration")
        cfg.pack(fill=tk.X, expand=False, pady=(0, 8))

        self.gpt_api_key_var = tk.StringVar(value=(os.getenv("MSTOCK_GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip())
        self.gpt_base_url_var = tk.StringVar(value=(os.getenv("MSTOCK_GPT_API_BASE_URL") or "").strip() or "https://api.aicredits.in/v1")
        self.gpt_model_var = tk.StringVar(value=(os.getenv("MSTOCK_GPT_MODEL") or "").strip() or "gpt-4o-mini")

        def add_tooltip(widget, text):
            def on_enter(e):
                self._gpt_tooltip = tk.Toplevel(widget)
                self._gpt_tooltip.wm_overrideredirect(True)
                x = widget.winfo_rootx() + 20
                y = widget.winfo_rooty() + 20
                self._gpt_tooltip.wm_geometry(f"+{x}+{y}")
                label = tk.Label(self._gpt_tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1, font=("Segoe UI", 9))
                label.pack()
            def on_leave(e):
                if hasattr(self, '_gpt_tooltip') and self._gpt_tooltip:
                    self._gpt_tooltip.destroy()
                    self._gpt_tooltip = None
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        # API Key
        lbl_key = ttk.Label(cfg, text="API key:")
        lbl_key.grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ent_key = ttk.Entry(cfg, textvariable=self.gpt_api_key_var, show="*", width=58)
        ent_key.grid(row=0, column=1, sticky="we", padx=8, pady=4)
        add_tooltip(lbl_key, "Your AICredits API key (OpenAI-compatible). Required.")

        # Model
        lbl_model = ttk.Label(cfg, text="Model:")
        lbl_model.grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ent_model = ttk.Entry(cfg, textvariable=self.gpt_model_var, width=30)
        ent_model.grid(row=1, column=1, sticky="w", padx=8, pady=4)
        add_tooltip(lbl_model, "Model name, e.g. gpt-4o-mini.")

        # Base URL
        lbl_url = ttk.Label(cfg, text="Base URL (optional):")
        lbl_url.grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ent_url = ttk.Entry(cfg, textvariable=self.gpt_base_url_var, width=58)
        ent_url.grid(row=2, column=1, sticky="we", padx=8, pady=4)
        add_tooltip(lbl_url, "API base URL. Default: https://api.aicredits.in/v1")

        cfg.grid_columnconfigure(1, weight=1)

        # --- Actions Section ---
        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, expand=False, pady=(0, 8))
        ttk.Button(btns, text="Save Config", command=self._on_gpt_save_config).pack(side=tk.LEFT)
        ttk.Button(btns, text="Test Config", command=self._on_gpt_health_check).pack(side=tk.LEFT)
        ttk.Button(btns, text="Analyze Market Snapshot", command=self._on_gpt_analyze_snapshot).pack(side=tk.LEFT, padx=(8, 0))
        self._gpt_status_var = tk.StringVar(value="Idle")
        ttk.Label(btns, textvariable=self._gpt_status_var, foreground="#444").pack(side=tk.LEFT, padx=(12, 0))

        # --- Output Section ---
        out = ttk.LabelFrame(root, text="Advisor Output & Status")
        out.pack(fill=tk.BOTH, expand=True)
        self.gpt_out = ScrolledText(out, height=10)
        self.gpt_out.pack(fill=tk.BOTH, expand=True)
        self.gpt_out.insert(tk.END, "Ready. Use Test Config or Analyze Market Snapshot.\n")
        self.gpt_out.configure(state=tk.DISABLED)

        # --- Summary Section ---
        self.gpt_summary_var = tk.StringVar(value="")
        summary = ttk.Label(out, textvariable=self.gpt_summary_var, font=("Segoe UI", 10, "bold"), foreground="#225522")
        summary.pack(fill=tk.X, padx=6, pady=4)

    def _append_gpt_output(self, text: str, summary: str = "") -> None:
        try:
            self.gpt_out.configure(state=tk.NORMAL)
            self.gpt_out.insert(tk.END, str(text) + ("\n" if not str(text).endswith("\n") else ""))
            self.gpt_out.see(tk.END)
            self.gpt_out.configure(state=tk.DISABLED)
            if summary:
                self.gpt_summary_var.set(summary)
        except Exception:
            pass

    def _sync_gpt_config_from_ui(self, *, persist_non_secret: bool = False) -> dict[str, str]:
        """Apply GPT tab values to the current process and optionally persist safe fields."""

        applied: dict[str, str] = {}
        api_key_now = str(self.gpt_api_key_var.get() or "").strip()
        base_url_now = str(self.gpt_base_url_var.get() or "").strip()
        model_now = str(self.gpt_model_var.get() or "").strip()

        if api_key_now:
            os.environ["MSTOCK_GPT_API_KEY"] = api_key_now
            applied["MSTOCK_GPT_API_KEY"] = "set"

        if base_url_now:
            os.environ["MSTOCK_GPT_API_BASE_URL"] = base_url_now
            applied["MSTOCK_GPT_API_BASE_URL"] = base_url_now
        else:
            os.environ.pop("MSTOCK_GPT_API_BASE_URL", None)

        if model_now:
            os.environ["MSTOCK_GPT_MODEL"] = model_now
            applied["MSTOCK_GPT_MODEL"] = model_now
        else:
            os.environ.pop("MSTOCK_GPT_MODEL", None)

        if persist_non_secret:
            to_persist: dict[str, str] = {}
            to_unset: list[str] = []

            if base_url_now:
                to_persist["MSTOCK_GPT_API_BASE_URL"] = base_url_now
            else:
                to_unset.append("MSTOCK_GPT_API_BASE_URL")

            if model_now:
                to_persist["MSTOCK_GPT_MODEL"] = model_now
            else:
                to_unset.append("MSTOCK_GPT_MODEL")

            try:
                persist_settings_env(to_persist, to_unset)
                applied["persisted"] = "true"
            except Exception:
                applied["persisted"] = "false"

        return applied

    def _on_gpt_save_config(self) -> None:
        try:
            applied = self._sync_gpt_config_from_ui(persist_non_secret=True)
            saved_model = applied.get("MSTOCK_GPT_MODEL") or "(default)"
            saved_url = applied.get("MSTOCK_GPT_API_BASE_URL") or "(default AICredits URL)"
            self._gpt_status_var.set("Saved")
            self._append_gpt_output(
                json.dumps(
                    {
                        "saved": True,
                        "persisted_fields": {
                            "MSTOCK_GPT_MODEL": saved_model,
                            "MSTOCK_GPT_API_BASE_URL": saved_url,
                        },
                        "api_key_persisted": False,
                    },
                    indent=2,
                ),
                summary="Saved GPT model/base URL for next startup.",
            )
        except Exception as exc:
            self._gpt_status_var.set("Save failed")
            self._append_gpt_output(json.dumps({"saved": False, "detail": str(exc)}, indent=2), summary="Failed to save GPT config.")

    def _on_gpt_health_check(self) -> None:
        if self._gpt_inflight:
            return
        self._gpt_inflight = True
        self._gpt_status_var.set("Checking...")

        try:
            self._sync_gpt_config_from_ui(persist_non_secret=True)
        except Exception:
            pass

        def worker() -> None:
            try:
                api_key = str(self.gpt_api_key_var.get() or "").strip()
                if not api_key:
                    raise ValueError("API key is required.")
                res = health_check(
                    api_key=api_key,
                    base_url=str(self.gpt_base_url_var.get() or "").strip() or None,
                    model=str(self.gpt_model_var.get() or "").strip() or None,
                )
            except Exception as exc:
                res = {"ok": False, "detail": str(exc)}

            def _done() -> None:
                try:
                    ok = bool(res.get("ok"))
                    status = 0
                    try:
                        status = int(res.get("status") or 0)
                    except Exception:
                        status = 0
                    detail = str(res.get("detail") or "Unknown error")
                    detail_l = detail.lower()
                    if (not ok) and (status == 401 or "invalid api key" in detail_l):
                        msg = "Config Error: Invalid API Key. Update GPT API key and retry."
                    else:
                        msg = "Config OK" if ok else f"Config Error: {detail}"
                    self._gpt_status_var.set("OK" if ok else "Failed")
                    self._append_gpt_output(json.dumps(res, indent=2), summary=msg)
                finally:
                    self._gpt_inflight = False

            self.after(0, _done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_gpt_analyze_snapshot(self) -> None:
        if self._gpt_inflight:
            return
        self._gpt_inflight = True
        self._gpt_status_var.set("Analyzing...")

        try:
            self._sync_gpt_config_from_ui(persist_non_secret=True)
        except Exception:
            pass

        # Build snapshot on UI thread.
        snapshot = self._build_market_snapshot()
        model = str(self.gpt_model_var.get() or "").strip() or "gpt-4o-mini"
        api_key = str(self.gpt_api_key_var.get() or "").strip()
        base_url = str(self.gpt_base_url_var.get() or "").strip() or None

        def worker() -> None:
            is_error = False
            try:
                if not api_key:
                    raise ValueError("API key is required.")
                analysis = analyze_market(
                    snapshot=snapshot,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )
                payload = {
                    "ce_pe_bias": getattr(analysis, "ce_pe_bias", None),
                    "recommended_strategy": getattr(analysis, "recommended_strategy", None),
                    "directional_strike_offset_steps": getattr(analysis, "directional_strike_offset_steps", None),
                    "greeks_interpretation": getattr(analysis, "greeks_interpretation", None),
                    "option_chain_analysis": getattr(analysis, "option_chain_analysis", None),
                    "reason": getattr(analysis, "reason", None),
                    "confidence": getattr(analysis, "confidence", None),
                    "strategy_parameters": getattr(analysis, "strategy_parameters", None),
                }
                reason_text = str(payload.get("reason") or "")
                reason_l = reason_text.lower()
                if "http 401" in reason_l or "invalid api key" in reason_l:
                    is_error = True
                    payload = {
                        "error": "Invalid API Key",
                        "detail": reason_text,
                    }
                    summary = "Error: Invalid API Key. Update GPT API key and retry."
                else:
                    summary = f"Bias: {payload['ce_pe_bias']} | Strategy: {payload['recommended_strategy']} | Confidence: {payload['confidence']}"
                    if payload.get("reason"):
                        summary += f"\nReason: {payload['reason']}"
            except Exception as exc:
                is_error = True
                payload = {"error": str(exc)}
                summary = f"Error: {exc}"

            def _done() -> None:
                try:
                    self._gpt_status_var.set("Done" if not is_error else "Error")
                    self._append_gpt_output(json.dumps(payload, indent=2), summary=summary)
                finally:
                    self._gpt_inflight = False

            self.after(0, _done)

        threading.Thread(target=worker, daemon=True).start()

    def _build_market_snapshot(self) -> dict[str, object]:
        # Snapshot is intentionally compact (helps token usage).
        candles = list(self._latest_candles or [])
        last = candles[-1] if candles else None
        spot = float(last.close) if last is not None else None
        indicators_row = self._compute_indicators(candles)
        patterns_row = self._compute_patterns(candles)
        bullish_patterns = [
            "hammer",
            "inverted_hammer",
            "bullish_engulfing",
            "piercing_line",
            "morning_star",
            "three_white_soldiers",
            "rising_three_methods",
        ]
        bearish_patterns = [
            "hanging_man",
            "shooting_star",
            "bearish_engulfing",
            "dark_cloud_cover",
            "evening_star",
            "three_black_crows",
            "falling_three_methods",
        ]
        neutral_patterns = ["doji"]
        pattern_groups = {
            "bullish": [k for k in bullish_patterns if bool(patterns_row.get(k))],
            "bearish": [k for k in bearish_patterns if bool(patterns_row.get(k))],
            "neutral": [k for k in neutral_patterns if bool(patterns_row.get(k))],
        }
        greeks_rows, greeks_summary = self._compute_open_leg_greeks(spot)

        open_positions: list[dict[str, object]] = []
        for tid, st in (self._trade_state or {}).items():
            if self._is_closed_trade_state(tid, st):
                continue
            legs = st.get("legs")
            if not isinstance(legs, list):
                legs = []

            legs_out: list[dict[str, object]] = []
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                is_hedge_leg = bool(leg.get("is_hedge"))
                sym = self._format_leg_symbol_ui(leg, include_hedge_tag=is_hedge_leg)
                legs_out.append(
                    {
                        "symbol": sym,
                        "side": leg.get("side"),
                        "quantity": leg.get("quantity"),
                        "strike": leg.get("strike"),
                        "option_type": leg.get("option_type"),
                        "expiry": (str(leg.get("expiry"))[:10] if leg.get("expiry") is not None else None),
                        "entry": leg.get("entry_price"),
                        "exit": leg.get("exit_price"),
                        "ltp": leg.get("ltp"),
                        "is_hedge": is_hedge_leg,
                    }
                )

            open_positions.append(
                {
                    "trade_id": str(tid),
                    "type": st.get("pos_type"),
                    "strategy": st.get("strategy"),
                    "status": st.get("status"),
                    "legs": legs_out,
                }
            )

        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "symbol": str(self._sig_symbol_var.get() or ""),
            "timeframe": str(self._sig_timeframe_var.get() or ""),
            "spot": spot,
            "last_candle": (
                {
                    "time": (last.time.isoformat(timespec="seconds") if last is not None and hasattr(last.time, "isoformat") else None),
                    "open": (float(last.open) if last is not None else None),
                    "high": (float(last.high) if last is not None else None),
                    "low": (float(last.low) if last is not None else None),
                    "close": (float(last.close) if last is not None else None),
                }
                if last is not None
                else None
            ),
            "indicators": indicators_row,
            "patterns": patterns_row,
            "pattern_groups": pattern_groups,
            "greeks_summary": greeks_summary,
            "greeks_legs": greeks_rows[:40],
            "open_positions": open_positions[:20],
        }

    def _compute_indicators(self, candles: list[Candle]) -> dict[str, object]:
        if not candles:
            return {}
        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]

        # Best-effort config alignment with strategy.
        try:
            cfg = load_strategy_config()
            ema_fast = int(getattr(cfg, "ema_fast", 9) or 9)
            ema_slow = int(getattr(cfg, "ema_slow", 21) or 21)
            atr_p = int(getattr(cfg, "atr_period", 14) or 14)
        except Exception:
            ema_fast, ema_slow, atr_p = 9, 21, 14

        out: dict[str, object] = {}
        try:
            out["ema_fast"] = ema(closes, ema_fast)
            out["ema_slow"] = ema(closes, ema_slow)
        except Exception:
            pass
        try:
            out["rsi"] = rsi(closes, 14)
        except Exception:
            pass
        try:
            out["atr"] = atr(highs, lows, closes, atr_p)
        except Exception:
            pass
        try:
            out["adx"] = adx(highs, lows, closes, 14)
        except Exception:
            pass
        try:
            out["supertrend"] = supertrend(highs, lows, closes, 10, 3.0)
        except Exception:
            pass
        try:
            out["roc_pct"] = roc(closes, 14)
        except Exception:
            pass
        try:
            out["choppiness"] = choppiness_index(highs, lows, closes, 14)
        except Exception:
            pass
        try:
            piv = pivot_points(float(candles[-1].high), float(candles[-1].low), float(candles[-1].close))
            out["pivot_pp"] = piv.get("pp")
        except Exception:
            pass
        return out

    def _compute_patterns(self, candles: list[Candle]) -> dict[str, object]:
        if not candles:
            return {}
        out: dict[str, object] = {}
        try:
            out["doji"] = bool(is_doji(candles))
        except Exception:
            pass
        try:
            out["hammer"] = bool(is_hammer(candles))
        except Exception:
            pass
        try:
            out["inverted_hammer"] = bool(is_inverted_hammer(candles))
        except Exception:
            pass
        try:
            out["shooting_star"] = bool(is_shooting_star(candles))
        except Exception:
            pass
        try:
            out["hanging_man"] = bool(is_hanging_man(candles))
        except Exception:
            pass
        try:
            out["bullish_engulfing"] = bool(is_bullish_engulfing(candles))
        except Exception:
            pass
        try:
            out["bearish_engulfing"] = bool(is_bearish_engulfing(candles))
        except Exception:
            pass
        try:
            out["piercing_line"] = bool(is_piercing_line(candles))
        except Exception:
            pass
        try:
            out["dark_cloud_cover"] = bool(is_dark_cloud_cover(candles))
        except Exception:
            pass
        try:
            out["morning_star"] = bool(is_morning_star(candles))
        except Exception:
            pass
        try:
            out["evening_star"] = bool(is_evening_star(candles))
        except Exception:
            pass
        try:
            out["three_white_soldiers"] = bool(is_three_white_soldiers(candles))
        except Exception:
            pass
        try:
            out["three_black_crows"] = bool(is_three_black_crows(candles))
        except Exception:
            pass
        try:
            out["rising_three_methods"] = bool(is_rising_three_methods(candles))
        except Exception:
            pass
        try:
            out["falling_three_methods"] = bool(is_falling_three_methods(candles))
        except Exception:
            pass
        return out

    def _normalize_candles(self, candles_list: object) -> list[Candle]:
        """Normalize various candle shapes into a list of Candle objects.

        The bot/chart integrations may emit Candle objects, dicts, or tuples.
        Signals/Greeks rendering requires Candle dataclass instances.
        """

        if candles_list is None:
            return []

        try:
            raw = list(candles_list)  # type: ignore[arg-type]
        except Exception:
            raw = []

        out: list[Candle] = []
        for it in raw:
            if isinstance(it, Candle):
                out.append(it)
                continue

            # Dict shape: {time/open/high/low/close}
            if isinstance(it, dict):
                try:
                    t = it.get("time") or it.get("ts") or it.get("timestamp") or it.get("datetime")
                    o = it.get("open") or it.get("o")
                    h = it.get("high") or it.get("h")
                    l = it.get("low") or it.get("l")
                    c = it.get("close") or it.get("c")
                except Exception:
                    continue

                dt: datetime | None = None
                try:
                    if isinstance(t, datetime):
                        dt = t
                    elif isinstance(t, (int, float)):
                        # Heuristic: if too large, assume ms.
                        ts = float(t)
                        if ts > 10_000_000_000:
                            ts = ts / 1000.0
                        dt = datetime.fromtimestamp(ts)
                    elif t is not None:
                        s = str(t).strip()
                        if s:
                            # Accept ISO strings; fall back to a couple common formats.
                            try:
                                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                            except Exception:
                                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S"):
                                    try:
                                        dt = datetime.strptime(s[:19], fmt)
                                        break
                                    except Exception:
                                        continue
                except Exception:
                    dt = None

                try:
                    if dt is None:
                        continue
                    out.append(
                        Candle(
                            time=dt,
                            open=float(o),
                            high=float(h),
                            low=float(l),
                            close=float(c),
                            volume=(float(it.get("volume")) if it.get("volume") is not None else None),
                        )
                    )
                except Exception:
                    continue

                continue

            # Tuple/list shape: (time, open, high, low, close, ...)
            if isinstance(it, (list, tuple)) and len(it) >= 5:
                t, o, h, l, c = it[0], it[1], it[2], it[3], it[4]
                dt: datetime | None = None
                try:
                    if isinstance(t, datetime):
                        dt = t
                    elif isinstance(t, (int, float)):
                        ts = float(t)
                        if ts > 10_000_000_000:
                            ts = ts / 1000.0
                        dt = datetime.fromtimestamp(ts)
                    else:
                        s = str(t).strip()
                        if s:
                            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                except Exception:
                    dt = None

                try:
                    if dt is None:
                        continue
                    out.append(Candle(time=dt, open=float(o), high=float(h), low=float(l), close=float(c)))
                except Exception:
                    continue

        # Keep candles in time order (best-effort).
        try:
            out.sort(key=lambda x: x.time)
        except Exception:
            pass
        return out

    def _parse_expiry(self, expiry_val: object) -> datetime | None:
        if expiry_val is None:
            return None
        if hasattr(expiry_val, "year") and hasattr(expiry_val, "month") and hasattr(expiry_val, "day"):
            try:
                # If a date/datetime is provided, normalize to 15:30 local time.
                if isinstance(expiry_val, datetime):
                    base = expiry_val
                else:
                    base = datetime(int(expiry_val.year), int(expiry_val.month), int(expiry_val.day))
                return base.replace(hour=15, minute=30, second=0, microsecond=0)
            except Exception:
                pass

        s = str(expiry_val).strip()
        if not s:
            return None
        # Common UI input: DD-MM-YYYY
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                d = datetime.strptime(s[:10], fmt)
                return d.replace(hour=15, minute=30, second=0, microsecond=0)
            except Exception:
                continue
        return None

    def _compute_open_leg_greeks(
        self,
        spot: float | None,
        *,
        trade_state: dict[str, dict[str, object]] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if spot is None or spot <= 0:
            return [], {"net_delta": None, "avg_iv": None, "avg_theta": None, "legs": 0}

        rows: list[dict[str, object]] = []
        ivs: list[float] = []
        thetas: list[float] = []
        net_delta_total = 0.0
        now = datetime.now()
        client = getattr(self, "_client", None)

        state_src = trade_state if isinstance(trade_state, dict) else (self._trade_state or {})
        for tid, st in state_src.items():
            if self._is_closed_trade_state(tid, st):
                continue
            legs = st.get("legs")
            if not isinstance(legs, list):
                continue

            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                is_hedge = bool(leg.get("is_hedge"))

                sym = str(leg.get("symbol") or "").strip()
                if not sym:
                    continue
                side = str(leg.get("side") or leg.get("transactionType") or leg.get("transactiontype") or "").strip().upper()
                try:
                    qty = int(leg.get("quantity") or leg.get("qty") or leg.get("netQty") or leg.get("netqty") or 0)
                except Exception:
                    qty = 0
                if qty <= 0:
                    continue

                opt_type = str(leg.get("option_type") or leg.get("type") or leg.get("opt_type") or "").strip().upper()
                display_sym = f"HEDGE {sym}" if is_hedge else sym

                # Show hedge-underlying legs in the greeks tab as visibility-only rows.
                if is_hedge and opt_type not in {"CE", "PE"}:
                    px_val = leg.get("ltp")
                    if px_val is None:
                        px_val = leg.get("exit_price")
                    if px_val is None:
                        px_val = leg.get("entry_price")
                    price: float | None
                    try:
                        price = float(px_val)
                        if price <= 0:
                            price = None
                    except Exception:
                        price = None
                    if price is None and client is not None:
                        try:
                            live_ltp = self._try_get_live_ltp_for_leg(client, leg)
                            if live_ltp is not None and float(live_ltp) > 0:
                                price = float(live_ltp)
                                try:
                                    leg["ltp"] = float(live_ltp)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    rows.append(
                        {
                            "trade_id": str(tid),
                            "symbol": display_sym,
                            "side": side,
                            "qty": qty,
                            "strike": price,
                            "type": "HEDGE",
                            "expiry": None,
                            "price": price,
                            "iv": 0.0,
                            "delta": 1.0 if side != "SELL" else -1.0,
                            "gamma": 0.0,
                            "vega": 0.0,
                            "theta": 0.0,
                            "net_delta": (1.0 if side != "SELL" else -1.0) * float(qty),
                            "is_hedge": True,
                        }
                    )
                    continue

                if opt_type not in {"CE", "PE"}:
                    continue

                strike = leg.get("strike") if leg.get("strike") is not None else leg.get("Strike")
                try:
                    strike_f = float(strike)
                except Exception:
                    continue
                if strike_f <= 0:
                    continue

                expiry_dt = self._parse_expiry(leg.get("expiry") if leg.get("expiry") is not None else leg.get("Expiry"))
                if expiry_dt is None:
                    continue
                t_sec = (expiry_dt - now).total_seconds()
                if t_sec <= 60:
                    continue
                t_years = float(t_sec) / float(365.0 * 24.0 * 3600.0)

                # Prefer current ltp, else exit_price, else entry_price.
                px_val = leg.get("ltp")
                if px_val is None:
                    px_val = leg.get("exit_price")
                if px_val is None:
                    px_val = leg.get("entry_price")

                # If we still don't have a usable price (common in live mode right after order placement),
                # try a best-effort live quote fetch.
                if px_val is None and client is not None:
                    try:
                        live_ltp = self._try_get_live_ltp_for_leg(client, leg)
                        if live_ltp is not None and float(live_ltp) > 0:
                            px_val = float(live_ltp)
                            # Cache into the leg dict so subsequent renders don't need another call.
                            try:
                                leg["ltp"] = float(live_ltp)
                            except Exception:
                                pass
                    except Exception:
                        pass

                price: float | None
                try:
                    price = float(px_val)
                    if price <= 0:
                        price = None
                except Exception:
                    price = None

                # Heuristic normalization: some feeds return option premium in paise.
                # If premium is wildly larger than spot, scale down.
                if price is not None:
                    try:
                        s_f = float(spot)
                        if s_f > 0 and float(price) > (1.5 * s_f) and float(price) > 1000.0:
                            price = float(price) / 100.0
                    except Exception:
                        pass

                # Clip price into theoretical no-arbitrage bounds to keep IV solver stable.
                # This avoids returning None for small feed glitches or bid/ask noise.
                if price is not None:
                    try:
                        s_f = float(spot)
                        k_f = float(strike_f)
                        t_f = float(t_years)
                        r_f = 0.0
                        df = math.exp(-r_f * t_f)
                        if opt_type == "CE":
                            lower = max(0.0, s_f - k_f * df)
                            upper = s_f
                        else:
                            lower = max(0.0, k_f * df - s_f)
                            upper = k_f * df
                        # Keep a tiny epsilon away from exact bounds.
                        eps = 1e-9
                        if price < lower:
                            price = float(lower) + eps
                        elif price > upper:
                            price = float(upper) - eps
                    except Exception:
                        pass

                # If the IV solver fails (common for illiquid/very low premium),
                # still show the leg row with n/a greeks instead of skipping it.
                iv: float | None = None
                try:
                    iv_raw = leg.get("iv")
                    if iv_raw is not None:
                        iv_guess = float(iv_raw)
                        # Allow user-provided percents like 18.5
                        if iv_guess > 3.0:
                            iv_guess = iv_guess / 100.0
                        if 0.0001 < iv_guess < 10.0:
                            iv = float(iv_guess)
                except Exception:
                    iv = None

                if iv is None and price is not None:
                    try:
                        iv_solved = implied_volatility(price, float(spot), strike_f, t_years, 0.0, opt_type)
                        if iv_solved is not None and float(iv_solved) > 0:
                            iv = float(iv_solved)
                    except Exception:
                        iv = None

                # Last-resort: use a conservative default IV guess so greeks populate.
                if iv is None:
                    iv = 0.20

                d: float | None = None
                g: float | None = None
                v: float | None = None
                t: float | None = None
                if iv is not None and float(iv) > 0:
                    try:
                        d = float(bs_delta(float(spot), strike_f, t_years, 0.0, float(iv), opt_type))
                        g = float(bs_gamma(float(spot), strike_f, t_years, 0.0, float(iv)))
                        v = float(bs_vega(float(spot), strike_f, t_years, 0.0, float(iv)))
                        t = float(bs_theta(float(spot), strike_f, t_years, 0.0, float(iv), opt_type) / 365.0)
                    except Exception:
                        d = g = v = t = None

                sign = 1.0
                if side == "SELL":
                    sign = -1.0

                net_d: float | None = None
                if d is not None:
                    try:
                        net_d = float(d) * float(qty) * float(sign)
                        net_delta_total += float(net_d)
                    except Exception:
                        net_d = None
                if iv is not None:
                    ivs.append(float(iv))
                if t is not None:
                    thetas.append(float(t))

                rows.append(
                    {
                        "trade_id": str(tid),
                        "symbol": display_sym,
                        "side": side,
                        "qty": qty,
                        "strike": strike_f,
                        "type": opt_type,
                        "expiry": expiry_dt,
                        "price": price,
                        "iv": iv,
                        "delta": d,
                        "gamma": g,
                        "vega": v,
                        "theta": t,
                        "net_delta": net_d,
                        "is_hedge": is_hedge,
                    }
                )

        avg_iv = (sum(ivs) / len(ivs)) if ivs else None
        avg_theta = (sum(thetas) / len(thetas)) if thetas else None
        return rows, {"net_delta": net_delta_total if rows else None, "avg_iv": avg_iv, "avg_theta": avg_theta, "legs": len(rows)}

    def _snapshot_trade_state_for_greeks(self) -> dict[str, dict[str, object]]:
        snapshot: dict[str, dict[str, object]] = {}
        for tid, st in (self._trade_state or {}).items():
            if not isinstance(st, dict):
                continue
            legs = st.get("legs")
            if not isinstance(legs, list):
                continue
            legs_copy: list[dict[str, object]] = []
            for leg in legs:
                if isinstance(leg, dict):
                    legs_copy.append(dict(leg))
            snapshot[str(tid)] = {
                "status": st.get("status"),
                "legs": legs_copy,
            }
        return snapshot

    def _render_greeks_rows(self, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
        try:
            for item in self.greeks_tree.get_children(""):
                self.greeks_tree.delete(item)
        except Exception:
            pass

        def _fmt_num(val: object, *, digits: int = 2) -> str:
            try:
                if val is None:
                    return "n/a"
                return f"{float(val):.{digits}f}"
            except Exception:
                return "n/a"

        def _fmt_pct(val: object, *, digits: int = 1) -> str:
            try:
                if val is None:
                    return "n/a"
                return f"{float(val) * 100.0:.{digits}f}%"
            except Exception:
                return "n/a"

        for r in rows[:400]:
            try:
                exp = r.get("expiry")
                if exp is None:
                    exp_s = ""
                else:
                    exp_s = exp.strftime("%d-%m-%Y") if hasattr(exp, "strftime") else str(exp)
                self.greeks_tree.insert(
                    "",
                    tk.END,
                    values=(
                        r.get("trade_id"),
                        r.get("symbol"),
                        r.get("side"),
                        r.get("qty"),
                        _fmt_num(r.get("strike"), digits=0),
                        r.get("type"),
                        exp_s,
                        _fmt_num(r.get("price"), digits=2),
                        _fmt_pct(r.get("iv"), digits=1),
                        _fmt_num(r.get("delta"), digits=3),
                        _fmt_num(r.get("gamma"), digits=6),
                        _fmt_num(r.get("vega"), digits=3),
                        _fmt_num(r.get("theta"), digits=3),
                        _fmt_num(r.get("net_delta"), digits=2),
                    ),
                )
            except Exception:
                continue

        try:
            nd = summary.get("net_delta")
            av = summary.get("avg_iv")
            ath = summary.get("avg_theta")
            if nd is None:
                nd_s = "n/a"
            else:
                nd_s = f"{float(nd):.2f}"
            if av is None:
                av_s = "n/a"
            else:
                av_s = f"{float(av)*100.0:.1f}%"
            if ath is None:
                ath_s = "n/a"
            else:
                ath_s = f"{float(ath):.3f}/day"
            self._greeks_summary_var.set(
                f"Net Δ: {nd_s} | Avg IV: {av_s} | Avg Θ: {ath_s} | Legs: {int(summary.get('legs') or 0)}"
            )
        except Exception:
            pass

    def _queue_greeks_render(self, spot: float | None) -> None:
        now = time.time()
        if self._greeks_inflight:
            self._greeks_pending = True
            self._greeks_pending_spot = spot
            return
        if (now - float(self._greeks_last_render_ts or 0.0)) < float(self._greeks_render_throttle_sec):
            self._greeks_pending = True
            self._greeks_pending_spot = spot
            return

        if spot is None or spot <= 0:
            self._render_greeks_rows([], {"net_delta": None, "avg_iv": None, "avg_theta": None, "legs": 0})
            self._greeks_last_render_ts = time.time()
            return

        snapshot = self._snapshot_trade_state_for_greeks()
        self._greeks_inflight = True

        def _worker(p_spot: float, p_state: dict[str, dict[str, object]]) -> None:
            try:
                rows, summary = self._compute_open_leg_greeks(p_spot, trade_state=p_state)
            except Exception:
                rows, summary = [], {"net_delta": None, "avg_iv": None, "avg_theta": None, "legs": 0}

            def _apply() -> None:
                try:
                    self._render_greeks_rows(rows, summary)
                finally:
                    self._greeks_last_render_ts = time.time()
                    self._greeks_inflight = False
                    if self._greeks_pending:
                        self._greeks_pending = False
                        pending_spot = self._greeks_pending_spot
                        self._greeks_pending_spot = None
                        self._queue_greeks_render(pending_spot)

            try:
                self.after(0, _apply)
            except Exception:
                self._greeks_inflight = False

        threading.Thread(target=_worker, args=(float(spot), snapshot), daemon=True).start()

    def _render_signals_and_greeks(self, *, force: bool = False) -> None:
        # Throttle UI rendering to keep Tk responsive.
        now_ts = time.time()
        if (not force) and (now_ts - float(self._signals_last_render_ts or 0.0)) < float(self._signals_render_throttle_sec):
            return
        self._signals_last_render_ts = now_ts

        candles = list(self._latest_candles or [])
        if not candles:
            try:
                if hasattr(self, "_sig_health_var"):
                    self._sig_health_var.set("Waiting for candles…")
            except Exception:
                pass
            return
        last = candles[-1]

        try:
            if hasattr(self, "_sig_health_var"):
                self._sig_health_var.set("OK")
        except Exception:
            pass

        try:
            if hasattr(self, "_sig_updated_var"):
                self._sig_updated_var.set(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
        except Exception:
            pass

        try:
            self._sig_last_candle_var.set(last.time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            self._sig_last_candle_var.set("n/a")
        try:
            self._sig_last_close_var.set(f"{float(last.close):.2f}")
        except Exception:
            self._sig_last_close_var.set("n/a")

        ind = self._compute_indicators(candles)
        def _fmt(v: object, *, digits: int = 3) -> str:
            try:
                if v is None:
                    return "n/a"
                return f"{float(v):.{digits}f}"
            except Exception:
                return "n/a"

        self._ind_vars["EMA Fast"].set(_fmt(ind.get("ema_fast"), digits=3))
        self._ind_vars["EMA Slow"].set(_fmt(ind.get("ema_slow"), digits=3))
        self._ind_vars["RSI"].set(_fmt(ind.get("rsi"), digits=2))
        self._ind_vars["ATR"].set(_fmt(ind.get("atr"), digits=3))
        self._ind_vars["ADX"].set(_fmt(ind.get("adx"), digits=2))
        self._ind_vars["Supertrend"].set(_fmt(ind.get("supertrend"), digits=2))
        self._ind_vars["ROC%"].set(_fmt(ind.get("roc_pct"), digits=2))
        self._ind_vars["Choppiness"].set(_fmt(ind.get("choppiness"), digits=2))
        self._ind_vars["Pivot PP"].set(_fmt(ind.get("pivot_pp"), digits=2))

        pat = self._compute_patterns(candles)
        def _yn(x: object) -> str:
            if x is None:
                return "n/a"
            return "YES" if bool(x) else "NO"

        self._pat_vars["Doji"].set(_yn(pat.get("doji")))
        self._pat_vars["Hammer"].set(_yn(pat.get("hammer")))
        self._pat_vars["Inverted Hammer"].set(_yn(pat.get("inverted_hammer")))
        self._pat_vars["Shooting Star"].set(_yn(pat.get("shooting_star")))
        self._pat_vars["Hanging Man"].set(_yn(pat.get("hanging_man")))
        self._pat_vars["Bullish Engulf"].set(_yn(pat.get("bullish_engulfing")))
        self._pat_vars["Bearish Engulf"].set(_yn(pat.get("bearish_engulfing")))
        self._pat_vars["Piercing Line"].set(_yn(pat.get("piercing_line")))
        self._pat_vars["Dark Cloud Cover"].set(_yn(pat.get("dark_cloud_cover")))
        self._pat_vars["Morning Star"].set(_yn(pat.get("morning_star")))
        self._pat_vars["Evening Star"].set(_yn(pat.get("evening_star")))
        self._pat_vars["Three White Soldiers"].set(_yn(pat.get("three_white_soldiers")))
        self._pat_vars["Three Black Crows"].set(_yn(pat.get("three_black_crows")))
        self._pat_vars["Rising Three Methods"].set(_yn(pat.get("rising_three_methods")))
        self._pat_vars["Falling Three Methods"].set(_yn(pat.get("falling_three_methods")))

        # Greeks from open legs (async to avoid blocking Tk).
        spot = None
        try:
            spot = float(last.close)
        except Exception:
            spot = None
        self._queue_greeks_render(spot)

    def _sync_trade_log_visibility(self) -> None:
        # Notebook/tabs are always visible.
        # Trade Logs remain visible in both paper and live modes.
        return

    def _trade_tree_right_click(self, evt: object) -> None:
        try:
            y = int(getattr(evt, "y"))
            x_root = int(getattr(evt, "x_root"))
            y_root = int(getattr(evt, "y_root"))
        except Exception:
            return

        row = self.trade_tree.identify_row(y)
        if not row:
            return

        try:
            self.trade_tree.selection_set(row)
        except Exception:
            pass

        try:
            values = self.trade_tree.item(row, "values")
            if values and len(values) >= 2:
                self._trade_ctx_trade_id = str(values[1])
            else:
                self._trade_ctx_trade_id = None
        except Exception:
            self._trade_ctx_trade_id = None

        try:
            self._trade_ctx_menu.tk_popup(x_root, y_root)
        except Exception:
            return

    def _trade_ctx_manual_exit(self) -> None:
        tid = str(self._trade_ctx_trade_id or "").strip()
        if not tid or tid.endswith("-H"):
            return

        if self._client is None:
            messagebox.showinfo("Not available", "Manual exit requires an active session (client not connected yet).")
            return

        st = self._trade_state.get(tid)
        if not isinstance(st, dict):
            messagebox.showinfo("Not available", "No trade details available for this row yet.")
            return
        legs = st.get("legs")
        if not isinstance(legs, list) or not legs:
            messagebox.showinfo("Not available", "No legs available for this trade.")
            return

        # Build a simple selection dialog.
        win = tk.Toplevel(self)
        win.title(f"Manual Exit Leg — {tid}")
        win.geometry("520x240")
        win.transient(self)

        tk.Label(win, text="Select a leg to exit:").pack(anchor="w", padx=10, pady=(10, 6))

        leg_var = tk.StringVar(value="")
        options: list[str] = []
        legs_by_label: dict[str, dict] = {}

        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if bool(leg.get("is_hedge")):
                continue
            sym = str(leg.get("symbol") or "").strip()
            if not sym:
                continue
            side = str(leg.get("side") or "").upper().strip()
            qty = int(leg.get("quantity") or 0)
            label = f"{side} {sym} x{qty}"
            options.append(label)
            legs_by_label[label] = leg

        if not options:
            win.destroy()
            messagebox.showinfo("Not available", "No non-hedge legs available to exit for this trade.")
            return

        leg_var.set(options[0])
        tk.OptionMenu(win, leg_var, *options).pack(fill=tk.X, padx=10)

        def _do_exit() -> None:
            label = str(leg_var.get() or "")
            leg = legs_by_label.get(label)
            if not isinstance(leg, dict):
                return

            sym = str(leg.get("symbol") or "").strip()
            token = str(leg.get("token") or "").strip() or None
            exch = str(leg.get("exchange") or "").strip() or None
            side = str(leg.get("side") or "").upper().strip()
            qty = int(leg.get("quantity") or 0)
            if not sym or side not in {"BUY", "SELL"} or qty <= 0:
                messagebox.showerror("Invalid leg", "Selected leg is missing symbol/side/qty.")
                return
            close_side = "SELL" if side == "BUY" else "BUY"

            if not messagebox.askyesno(
                "Confirm",
                f"Place manual exit order?\n\n{close_side} {sym} x{qty}",
            ):
                return

            try:
                self._client.place_order(
                    symbol=sym,
                    side=close_side,
                    quantity=qty,
                    exchange=exch,
                    symbol_token=token,
                )
                print(f"[UI] Manual exit order placed: {close_side} {sym} x{qty} (trade_id={tid})")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Order failed", f"Failed to place order: {exc}")
                return
            finally:
                try:
                    win.destroy()
                except Exception:
                    pass

        tk.Button(win, text="Exit selected leg", command=_do_exit).pack(pady=10)

    def _trade_log_row_values_from_state(self, trade_id: str, state: dict[str, object]) -> tuple[str, ...] | None:
        try:
            opened_ts = float(state.get("opened_ts") or 0.0)
        except Exception:
            opened_ts = 0.0
        if opened_ts <= 0.0:
            return None

        try:
            time_text = datetime.fromtimestamp(opened_ts).strftime("%H:%M:%S")
        except Exception:
            time_text = ""

        def _fmt_amount(value: object) -> str:
            if value is None:
                return ""
            try:
                return f"{float(value):.2f}"
            except Exception:
                return str(value)

        legs_all = state.get("legs") if isinstance(state.get("legs"), list) else []
        if str(trade_id).endswith("-H"):
            hedge_legs = [leg for leg in legs_all if isinstance(leg, dict)]
            legs_text = self._format_legs(hedge_legs, show_prices=True, hedge_prefix=False)
        else:
            main_legs, hedge_opt_legs, _hedge_under_legs = self._split_legs_for_display(legs_all)
            display_main_legs: list[dict] = []
            for leg in list(main_legs) + list(hedge_opt_legs):
                if not isinstance(leg, dict):
                    continue
                try:
                    q = int(leg.get("quantity") or 0)
                except Exception:
                    q = 0
                if q <= 0:
                    continue
                display_main_legs.append(leg)
            legs_text = self._format_legs(display_main_legs, show_prices=True, hedge_prefix=True)

        pnl_breakdown = self._compute_parent_display_pnl_breakdown(str(trade_id), state)
        mtm_display = pnl_breakdown.get("parent_mtm")
        realized_display = pnl_breakdown.get("parent_realized")

        def _fmt_breakdown(total: object, base: object, hedge: object) -> str:
            total_s = _fmt_amount(total)
            base_s = _fmt_amount(base)
            hedge_s = _fmt_amount(hedge)
            hedge_is_zero = False
            try:
                hedge_is_zero = float(hedge or 0.0) == 0.0
            except Exception:
                hedge_is_zero = not bool(hedge_s)
            if base_s and (not hedge_s or hedge_is_zero) and total_s == base_s:
                return total_s
            if base_s and hedge_s:
                return f"{total_s} (base {base_s} + hedge {hedge_s})"
            if base_s:
                return f"{total_s} (base {base_s})"
            if hedge_s:
                return f"{total_s} (hedge {hedge_s})"
            return total_s

        return (
            time_text,
            str(trade_id),
            str(state.get("pos_type") or ""),
            str(state.get("strategy") or ""),
            str(legs_text or ""),
            _fmt_breakdown(mtm_display, pnl_breakdown.get("base_mtm"), pnl_breakdown.get("hedge_mtm")),
            _fmt_breakdown(realized_display, pnl_breakdown.get("base_realized"), pnl_breakdown.get("hedge_realized")),
            str(state.get("status") or ""),
        )

    def _collect_today_trade_log_pdf_rows(self) -> list[tuple[str, ...]]:
        try:
            today = datetime.now().date()
        except Exception:
            return []

        items: list[tuple[str, dict[str, object]]] = []
        for trade_id, state in (self._trade_state or {}).items():
            if isinstance(state, dict):
                items.append((str(trade_id), state))

        def _opened_sort_key(item: tuple[str, dict[str, object]]) -> tuple[float, str]:
            trade_id, state = item
            try:
                opened_ts = float(state.get("opened_ts") or 0.0)
            except Exception:
                opened_ts = 0.0
            return (opened_ts, trade_id)

        rows: list[tuple[str, ...]] = []
        for trade_id, state in sorted(items, key=_opened_sort_key):
            try:
                opened_ts = float(state.get("opened_ts") or 0.0)
            except Exception:
                continue
            if opened_ts <= 0.0:
                continue
            try:
                if datetime.fromtimestamp(opened_ts).date() != today:
                    continue
            except Exception:
                continue
            row = self._trade_log_row_values_from_state(trade_id, state)
            if row is not None:
                rows.append(row)
        return rows

    def _export_today_trade_log_pdf(self) -> None:
        rows = self._collect_today_trade_log_pdf_rows()
        if not rows:
            messagebox.showinfo("No rows", "No trade log rows found for today.")
            return

        try:
            today = datetime.now().date()
            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            today = None
            generated_at = ""

        default_name = f"trade-log-{today.isoformat() if today is not None else 'today'}.pdf"
        path = filedialog.asksaveasfilename(
            title="Export Today's Trade Log PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf")],
        )
        if not path:
            return

        try:
            import html
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except Exception as exc:
            messagebox.showerror("PDF unavailable", f"PDF export requires reportlab.\n\nDetails: {exc}")
            return

        headers = ["Time", "ID", "Type", "Position", "Legs", "Live MTM", "P&L", "Status"]
        styles = getSampleStyleSheet()
        header_style = ParagraphStyle(
            "TradeLogHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        )
        cell_style = ParagraphStyle(
            "TradeLogCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
        )

        table_data = [[Paragraph(html.escape(text), header_style) for text in headers]]
        for row in rows:
            table_data.append([Paragraph(html.escape(str(value or "")), cell_style) for value in row])

        doc = SimpleDocTemplate(
            str(path),
            pagesize=landscape(A4),
            leftMargin=18,
            rightMargin=18,
            topMargin=18,
            bottomMargin=18,
        )
        table = Table(table_data, colWidths=[42, 44, 58, 88, 320, 54, 54, 110], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        story = [
            Paragraph("Today's Trade Log", styles["Title"]),
            Spacer(1, 6),
            Paragraph(f"Generated: {html.escape(generated_at)}", styles["BodyText"]),
            Paragraph(f"Rows exported: {len(rows)}", styles["BodyText"]),
            Spacer(1, 10),
            table,
        ]
        try:
            doc.build(story)
        except Exception as exc:
            messagebox.showerror("Export failed", f"Could not generate PDF: {exc}")
            return

        try:
            self._app_status_var.set(f"Trade log PDF saved to {path}")
        except Exception:
            pass
        try:
            print(f"[UI] Trade log PDF saved: {path}")
        except Exception:
            pass
        messagebox.showinfo("PDF saved", f"Today's trade log was exported to:\n{path}")

    def _sort_trade_log(self, col: str) -> None:
        # Toggle sort direction per column.
        desc = bool(self._trade_sort_desc.get(col, False))
        self._trade_sort_desc[col] = not desc

        cols = ("time", "trade_id", "pos_type", "strategy", "legs", "mtm", "realized", "status")
        try:
            idx = cols.index(col)
        except ValueError:
            return

        def _key(item_id: str):
            try:
                vals = self.trade_tree.item(item_id, "values")
            except Exception:
                vals = None
            v = "" if not vals or idx >= len(vals) else vals[idx]
            if col in {"mtm", "realized"}:
                try:
                    return float(v)
                except Exception:
                    return float("-inf")
            if col == "time":
                try:
                    # HH:MM:SS
                    parts = str(v).split(":")
                    if len(parts) == 3:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                except Exception:
                    pass
                return 0
            return str(v)

        items = list(self.trade_tree.get_children(""))
        items.sort(key=_key, reverse=not desc)
        for i, item in enumerate(items):
            try:
                self.trade_tree.move(item, "", i)
            except Exception:
                pass

    def _reset_trade_log(self) -> None:
        self._trade_rows.clear()
        self._trade_state.clear()
        self._closed_option_legs_day = datetime.now().date()
        self._closed_option_legs = []
        self._qty_traded_day = datetime.now().date()
        self._qty_traded_total = 0
        self._open_qty_by_trade_symbol = {}
        self._recent_qty_reduction_by_track_key = {}
        self._hedge_qty_traded_day = datetime.now().date()
        self._hedge_qty_traded_total = 0
        self._open_hedge_qty_by_trade_symbol = {}
        for item in self.trade_tree.get_children():
            self.trade_tree.delete(item)
        try:
            self._refresh_pnl_totals()
        except Exception:
            pass

    def _prune_stale_open_rows_when_idle(self, force: bool = False) -> None:
        """Remove non-closed trade rows when the bot is fully idle.

        This prevents stale OPEN/PARTIAL rows from lingering in the Trade Log
        after stop/crash paths where a CLOSE event was never emitted.
        """

        try:
            status_txt = str(self.status_var.get() or "").strip().lower()
        except Exception:
            status_txt = ""

        try:
            thread_alive = bool(self._bot_thread and self._bot_thread.is_alive())
        except Exception:
            thread_alive = False

        try:
            scalper_alive = getattr(self, "_scalper", None) is not None
        except Exception:
            scalper_alive = False

        # Only prune when bot is truly idle unless explicitly forced.
        if not force and (status_txt != "idle" or thread_alive or scalper_alive):
            return

        stale_ids: list[str] = []
        for trade_id, state in list(self._trade_state.items()):
            try:
                if str(trade_id).endswith("-H"):
                    continue
                st = str((state or {}).get("status") or "").strip().upper()
                if not st.startswith("CLOSED"):
                    stale_ids.append(str(trade_id))
            except Exception:
                continue

        if not stale_ids:
            return

        for trade_id in stale_ids:
            try:
                row = self._trade_rows.pop(trade_id, None)
                if row is not None:
                    self.trade_tree.delete(row)
            except Exception:
                pass
            try:
                self._trade_state.pop(trade_id, None)
            except Exception:
                pass

            hedge_trade_id = f"{trade_id}-H"
            try:
                hedge_row = self._trade_rows.pop(hedge_trade_id, None)
                if hedge_row is not None:
                    self.trade_tree.delete(hedge_row)
            except Exception:
                pass
            try:
                self._trade_state.pop(hedge_trade_id, None)
            except Exception:
                pass

            # Remove open-qty trackers for this trade.
            try:
                for key in list(self._open_qty_by_trade_symbol.keys()):
                    if isinstance(key, tuple) and key and str(key[0]) == trade_id:
                        self._open_qty_by_trade_symbol.pop(key, None)
            except Exception:
                pass
            try:
                for key in list(self._open_hedge_qty_by_trade_symbol.keys()):
                    if isinstance(key, tuple) and key and str(key[0]) == trade_id:
                        self._open_hedge_qty_by_trade_symbol.pop(key, None)
            except Exception:
                pass

        try:
            self._refresh_pnl_totals()
        except Exception:
            pass

    def _format_legs(
        self,
        legs: list[dict],
        *,
        show_prices: bool = True,
        hedge_prefix: bool = True,
    ) -> str:
        parts: list[str] = []
        for leg in legs:
            try:
                side = str(leg.get("side") or "").upper()
                sym = str(leg.get("symbol") or "")
                qty = self._get_leg_qty(leg)
                if not sym:
                    continue

                is_hedge = False
                try:
                    is_hedge = bool(leg.get("is_hedge"))
                except Exception:
                    is_hedge = False
                is_option_leg = self._is_option_leg(leg)

                # Try to simplify option legs like NIFTY2610626200CE -> 26200 CE.
                label = sym
                strike = leg.get("strike")
                opt_type = str(leg.get("option_type") or "").upper()
                expiry = leg.get("expiry")

                if is_hedge and not is_option_leg:
                    # Render hedge legs in a distinct, unambiguous format.
                    # Example: "HEDGE: SELL NIFTY x65 (...)".
                    label = sym
                    strike = None
                    opt_type = ""
                    expiry = None

                # Prefer explicit strike/option_type when present from the option chain.
                if strike is not None or opt_type:
                    try:
                        if strike is not None:
                            s_val = float(strike)
                            # Render integers without .0
                            s_str = f"{int(s_val)}" if s_val.is_integer() else f"{s_val:g}"
                        else:
                            s_str = ""
                    except Exception:
                        s_str = str(strike)
                    pieces: list[str] = []
                    # Expiry (if present) first for clarity.
                    exp_str = ""
                    if expiry is not None:
                        try:
                            if hasattr(expiry, "strftime"):
                                exp_str = expiry.strftime("%d-%b-%y")
                            else:
                                exp_str = str(expiry).split()[0]
                        except Exception:
                            exp_str = str(expiry)
                    if exp_str:
                        pieces.append(exp_str)
                    if s_str:
                        pieces.append(f"{s_str}")
                    if opt_type:
                        pieces.append(opt_type)
                    if pieces:
                        label = " ".join(pieces)
                else:
                    # Fallback: parse from trading symbol suffix when it ends with CE/PE.
                    up = sym.upper()
                    if up.endswith("CE") or up.endswith("PE"):
                        opt = up[-2:]
                        body = up[:-2]
                        j = len(body) - 1
                        while j >= 0 and body[j].isdigit():
                            j -= 1
                        digits = body[j + 1 :]
                        if digits:
                            label = f"{digits} {opt}"

                # Attach price info where available.
                entry = leg.get("entry_price")
                exit_p = leg.get("exit_price")
                last_p = leg.get("ltp")
                price_suffix = ""
                try:
                    entry_f = float(entry) if entry is not None else None
                except Exception:
                    entry_f = None
                try:
                    exit_f = float(exit_p) if exit_p is not None else None
                except Exception:
                    exit_f = None
                try:
                    last_f = float(last_p) if last_p is not None else None
                except Exception:
                    last_f = None

                if show_prices and (is_option_leg or entry_f is not None or exit_f is not None or last_f is not None):
                    parts_price: list[str] = []
                    if is_option_leg:
                        # Keep option fields explicit to avoid confusion.
                        # Show entry price when available; show exit/last only if present.
                        if entry_f is not None:
                            parts_price.append(f"entry {entry_f:.2f}")
                        if exit_f is not None:
                            parts_price.append(f"exit {exit_f:.2f}")
                        elif last_f is not None:
                            parts_price.append(f"last {last_f:.2f}")
                    else:
                        if entry_f is not None:
                            parts_price.append(f"entry {entry_f:.2f}")
                        if exit_f is not None:
                            parts_price.append(f"exit {exit_f:.2f}")
                        elif last_f is not None:
                            parts_price.append(f"last {last_f:.2f}")
                    if parts_price:
                        price_suffix = " (" + ", ".join(parts_price) + ")"

                if is_hedge and hedge_prefix:
                    # Use unified 'HEDGE:' prefix for hedged legs in the main display.
                    prefix = "HEDGE: "
                else:
                    prefix = ""
                if qty > 0:
                    parts.append(f"{prefix}{side} {label} x{qty}{price_suffix}")
                else:
                    parts.append(f"{prefix}{side} {label}{price_suffix}")
            except Exception:
                continue
        return "; ".join(parts)[:500]

    def _note_traded_qty(self, evt: TradeLogEvent) -> None:
        """Accumulate traded quantity (turnover) for today (option legs only, excludes hedges).

        Rules:
        - OPEN: counts full entry qty.
        - UPDATE: counts net qty change (adds and partial-exit reductions).
        - PARTIAL_CLOSE: counts exit qty, but is idempotent with UPDATE snapshots.
        - CLOSE: unwinds remaining open qty to zero (idempotent).

        This number is *turnover* (entry + exits), not current open quantity.
        """

        try:
            evt_day = datetime.fromtimestamp(float(evt.ts)).date()
        except Exception:
            evt_day = datetime.now().date()

        if evt_day != self._qty_traded_day:
            self._qty_traded_day = evt_day
            self._qty_traded_total = 0
            # Reset open-qty trackers on day rollover to avoid carrying stale
            # intra-day state into a new turnover bucket.
            self._open_qty_by_trade_symbol = {}
            self._recent_qty_reduction_by_track_key = {}

        legs = evt.legs or []
        if not isinstance(legs, list):
            return

        add_qty = 0
        trade_id = str(getattr(evt, "trade_id", "") or "")
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if bool(leg.get("is_hedge")):
                continue
            if not self._is_option_leg(leg):
                continue

            primary_key, keys_all = self._qty_track_keys(trade_id, leg)
            if primary_key is None:
                continue
            try:
                q = int(leg.get("quantity") or 0)
            except Exception:
                q = 0

            prev_q = int(self._open_qty_by_trade_symbol.get(primary_key, 0) or 0)

            if evt.event == "OPEN":
                # Treat OPEN as 0 -> q.
                if q > 0:
                    add_qty += int(abs(q - 0))
                    for k in keys_all:
                        self._open_qty_by_trade_symbol[k] = int(q)
                        self._recent_qty_reduction_by_track_key.pop(k, None)

            elif evt.event == "UPDATE":
                # Treat UPDATE as net qty change. This captures partial exits even
                # when the strategy/broker reports them only via reduced qty snapshots.
                if q < 0:
                    q = 0
                delta = int(q - prev_q)
                if delta != 0:
                    add_qty += int(abs(delta))
                for k in keys_all:
                    if q > 0:
                        self._open_qty_by_trade_symbol[k] = int(q)
                    else:
                        self._open_qty_by_trade_symbol.pop(k, None)
                    if delta < 0:
                        try:
                            tsf = float(evt.ts)
                        except Exception:
                            tsf = time.time()
                        self._recent_qty_reduction_by_track_key[k] = (int(abs(delta)), tsf)
                    else:
                        self._recent_qty_reduction_by_track_key.pop(k, None)

            elif evt.event == "PARTIAL_CLOSE":
                # PARTIAL_CLOSE legs carry exited qty, not remaining qty.
                closed_qty = max(0, int(q))
                if closed_qty <= 0:
                    continue

                # If UPDATE already reduced qty by exactly this amount moments
                # earlier, this PARTIAL_CLOSE is a duplicate accounting signal.
                skip_as_duplicate = False
                try:
                    evt_ts = float(evt.ts)
                except Exception:
                    evt_ts = time.time()
                seen = self._recent_qty_reduction_by_track_key.get(primary_key)
                if seen is not None:
                    seen_qty, seen_ts = seen
                    if int(seen_qty) == int(closed_qty) and abs(evt_ts - float(seen_ts)) <= 5.0:
                        skip_as_duplicate = True

                if skip_as_duplicate:
                    new_q = max(0, int(prev_q))
                elif prev_q > 0:
                    applied = min(int(prev_q), int(closed_qty))
                    if applied > 0:
                        add_qty += int(applied)
                    new_q = int(prev_q) - int(applied)
                else:
                    # Fallback when we have no tracker snapshot for this leg.
                    add_qty += int(closed_qty)
                    new_q = 0

                for k in keys_all:
                    if new_q > 0:
                        self._open_qty_by_trade_symbol[k] = int(new_q)
                    else:
                        self._open_qty_by_trade_symbol.pop(k, None)
                    self._recent_qty_reduction_by_track_key.pop(k, None)

            elif evt.event == "CLOSE":
                # Unwind any remaining open qty to zero. If we don't have tracker
                # (e.g., UI attached late), fall back to event qty.
                if prev_q > 0:
                    add_qty += int(abs(prev_q))
                else:
                    if q > 0:
                        add_qty += int(q)
                for k in keys_all:
                    self._open_qty_by_trade_symbol.pop(k, None)
                    self._recent_qty_reduction_by_track_key.pop(k, None)

        # On CLOSE, clear any remaining tracked open qty for this trade_id.
        if evt.event == "CLOSE" and trade_id:
            try:
                for (tid, sym), _v in list(self._open_qty_by_trade_symbol.items()):
                    if tid == trade_id:
                        self._open_qty_by_trade_symbol.pop((tid, sym), None)
            except Exception:
                pass
            try:
                for (tid, sym), _v in list(self._recent_qty_reduction_by_track_key.items()):
                    if tid == trade_id:
                        self._recent_qty_reduction_by_track_key.pop((tid, sym), None)
            except Exception:
                pass

        if add_qty > 0:
            self._qty_traded_total += int(add_qty)

    def _note_hedge_traded_qty(self, evt: TradeLogEvent) -> None:
        """Accumulate traded quantity for today (underlying hedge legs only).

        This tracks hedge *turnover* as absolute changes in the net hedge position.

        Rules (net hedge qty per symbol, signed by BUY/SELL):
        - OPEN: counts |net_qty| from 0.
        - UPDATE: counts |net_qty - prev_net_qty| (rebalance adds/removes).
        - PARTIAL_CLOSE/CLOSE: also counts |delta| if the snapshot includes hedges.
          If CLOSE has no hedge legs snapshot, we assume hedge is exited and count
          |prev_net_qty| to unwind.
        """

        try:
            evt_day = datetime.fromtimestamp(float(evt.ts)).date()
        except Exception:
            evt_day = datetime.now().date()

        if evt_day != self._hedge_qty_traded_day:
            self._hedge_qty_traded_day = evt_day
            self._hedge_qty_traded_total = 0

        legs = evt.legs or []
        if not isinstance(legs, list):
            legs = []

        trade_id = str(getattr(evt, "trade_id", "") or "")
        if not trade_id:
            return

        # Special-case CLOSE: the strategy emits hedge legs in the CLOSE snapshot
        # with the *original* hedge side/qty (not the closing order). For turnover
        # accounting, treat CLOSE as an unwind to zero for all hedge symbols.
        if evt.event == "CLOSE":
            add_qty = 0

            # Use the last known net from trackers; if missing (e.g., UI attached
            # late), fall back to the snapshot net.
            snap_net: dict[str, int] = {}
            legs = evt.legs or []
            if not isinstance(legs, list):
                legs = []

            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                if not bool(leg.get("is_hedge")):
                    continue
                sym = str(leg.get("symbol") or "").strip()
                if not sym:
                    continue
                try:
                    qty = int(leg.get("quantity") or 0)
                except Exception:
                    qty = 0
                if qty < 0:
                    qty = abs(qty)

                side = str(leg.get("side") or "").upper().strip()
                if side.startswith("B"):
                    sign = 1
                elif side.startswith("S"):
                    sign = -1
                else:
                    continue

                snap_net[sym] = int(sign * qty)

            # Unwind tracked hedge positions for this trade.
            for (tid, sym), prev_net in list(self._open_hedge_qty_by_trade_symbol.items()):
                if tid != trade_id:
                    continue
                try:
                    add_qty += abs(int(prev_net))
                except Exception:
                    pass
                self._open_hedge_qty_by_trade_symbol.pop((tid, sym), None)

            # If nothing was tracked, unwind snapshot net (best-effort).
            if add_qty == 0 and snap_net:
                for _sym, net in snap_net.items():
                    try:
                        add_qty += abs(int(net))
                    except Exception:
                        pass

            if add_qty > 0:
                self._hedge_qty_traded_total += int(add_qty)
            return

        add_qty = 0
        processed_any_hedge = False
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if not bool(leg.get("is_hedge")):
                continue
            # Only underlying hedges here (exclude option-like hedges).
            if self._is_option_leg(leg):
                continue

            sym = str(leg.get("symbol") or "").strip()
            if not sym:
                continue
            try:
                qty = int(leg.get("quantity") or 0)
            except Exception:
                qty = 0
            if qty < 0:
                qty = abs(qty)

            side = str(leg.get("side") or "").upper().strip()
            sign = 0
            if side.startswith("B"):
                sign = 1
            elif side.startswith("S"):
                sign = -1
            else:
                # Unknown direction -> skip to avoid corrupt accounting.
                continue

            processed_any_hedge = True
            net_qty = int(sign * qty)
            key = (trade_id, sym)
            prev_net = int(self._open_hedge_qty_by_trade_symbol.get(key, 0) or 0)

            if evt.event == "OPEN":
                add_qty += abs(net_qty)
            else:
                delta = int(net_qty - prev_net)
                if delta != 0:
                    add_qty += abs(delta)

            if net_qty != 0:
                self._open_hedge_qty_by_trade_symbol[key] = int(net_qty)
            else:
                self._open_hedge_qty_by_trade_symbol.pop(key, None)

        # CLOSE is handled above (treated as unwind-to-zero).

        if add_qty > 0:
            self._hedge_qty_traded_total += int(add_qty)

    def _pump_trades(self) -> None:
        try:
            processed = 0
            # Do not drain the entire queue in one tick; large bursts can freeze the UI.
            max_items = 250
            while True:
                evt = self._trade_q.get_nowait()
                processed += 1
                try:
                    trade_id = str(evt.trade_id)
                    state = self._trade_state.setdefault(trade_id, {})

                    if evt.event == "OPEN":
                        state.setdefault("opened_ts", float(evt.ts))
                        state["pos_type"] = evt.position_type
                        state["strategy"] = evt.name
                        state["legs"] = evt.legs
                        state["status"] = "OPEN"
                        self._note_traded_qty(evt)
                        self._note_hedge_traded_qty(evt)
                    elif evt.event == "PARTIAL_CLOSE":
                        state["status"] = "PARTIAL"
                        self._note_closed_option_legs(evt)
                        self._note_traded_qty(evt)
                        self._note_hedge_traded_qty(evt)
                        # IMPORTANT: strategy emits PARTIAL_CLOSE events with only the closed leg(s)
                        # (qty exited + exit_price). Do NOT replace the open legs snapshot with that
                        # payload, or the UI will show the exited qty as if it's the remaining position.
                        legs_state = state.get("legs")
                        if isinstance(legs_state, list) and evt.legs:
                            for closed_leg in evt.legs:
                                if not isinstance(closed_leg, dict):
                                    continue
                                sym_c = str(closed_leg.get("symbol") or "").strip()
                                if not sym_c:
                                    continue
                                try:
                                    closed_qty = int(closed_leg.get("quantity") or 0)
                                except Exception:
                                    closed_qty = 0
                                if closed_qty <= 0:
                                    continue

                                for open_leg in legs_state:
                                    if not isinstance(open_leg, dict):
                                        continue
                                    if bool(open_leg.get("is_hedge")):
                                        continue
                                    sym_o = str(open_leg.get("symbol") or "").strip()
                                    if sym_o != sym_c:
                                        continue
                                    # Sync display qty from tracker (idempotent even if UPDATE arrived first).
                                    try:
                                        primary_key, keys_all = self._qty_track_keys(trade_id, open_leg)
                                        if primary_key is not None:
                                            tracked_qty = int(self._open_qty_by_trade_symbol.get(primary_key, open_leg.get("quantity") or 0) or 0)
                                            if tracked_qty < 0:
                                                tracked_qty = 0
                                            open_leg["quantity"] = int(tracked_qty)
                                            # Keep aliases aligned.
                                            for k in keys_all:
                                                if tracked_qty > 0:
                                                    self._open_qty_by_trade_symbol[k] = int(tracked_qty)
                                                else:
                                                    self._open_qty_by_trade_symbol.pop(k, None)
                                    except Exception:
                                        # Fallback: best-effort subtract.
                                        try:
                                            open_qty = int(open_leg.get("quantity") or 0)
                                        except Exception:
                                            open_qty = 0
                                        open_leg["quantity"] = max(0, int(open_qty) - int(closed_qty))
                                    break
                        # Canonical re-sync: align every main leg qty from tracker so stale
                        # per-symbol matching cannot leave phantom open qty in UI state.
                        if isinstance(legs_state, list):
                            for open_leg in legs_state:
                                if not isinstance(open_leg, dict) or bool(open_leg.get("is_hedge")):
                                    continue
                                try:
                                    primary_key, keys_all = self._qty_track_keys(trade_id, open_leg)
                                    if primary_key is None:
                                        continue
                                    tracked_qty = int(
                                        self._open_qty_by_trade_symbol.get(
                                            primary_key,
                                            open_leg.get("quantity") or 0,
                                        )
                                        or 0
                                    )
                                    if tracked_qty < 0:
                                        tracked_qty = 0
                                    open_leg["quantity"] = int(tracked_qty)
                                    for k in keys_all:
                                        if tracked_qty > 0:
                                            self._open_qty_by_trade_symbol[k] = int(tracked_qty)
                                        else:
                                            self._open_qty_by_trade_symbol.pop(k, None)
                                except Exception:
                                    continue
                        # If all non-hedge legs are now zero qty, close state eagerly.
                        try:
                            any_open_main = False
                            if isinstance(legs_state, list):
                                for lg in legs_state:
                                    if not isinstance(lg, dict) or bool(lg.get("is_hedge")):
                                        continue
                                    try:
                                        if int(lg.get("quantity") or 0) > 0:
                                            any_open_main = True
                                            break
                                    except Exception:
                                        continue
                            if not any_open_main:
                                state["status"] = "CLOSED (all_legs_closed)"
                        except Exception:
                            pass
                    elif evt.event == "CLOSE":
                        state["status"] = "CLOSED" if not evt.reason else f"CLOSED ({evt.reason})"
                        # Preserve the pre-close leg snapshot and merge in any close-time
                        # exit prices so closed-leg logs keep quantities and strikes.
                        try:
                            if evt.legs:
                                merged_legs: list[dict[str, object]] = []
                                previous_legs = state.get("legs") if isinstance(state.get("legs"), list) else []
                                previous_by_symbol: dict[str, dict[str, object]] = {}
                                for prev_leg in previous_legs:
                                    if not isinstance(prev_leg, dict):
                                        continue
                                    prev_symbol = str(prev_leg.get("symbol") or "").strip()
                                    if prev_symbol and prev_symbol not in previous_by_symbol:
                                        previous_by_symbol[prev_symbol] = prev_leg

                                for leg in evt.legs:
                                    if not isinstance(leg, dict):
                                        continue
                                    leg_copy = dict(leg)
                                    symbol = str(leg_copy.get("symbol") or "").strip()
                                    prev_leg = previous_by_symbol.get(symbol) if symbol else None
                                    if isinstance(prev_leg, dict):
                                        try:
                                            leg_qty = int(leg_copy.get("quantity") or 0)
                                        except Exception:
                                            leg_qty = 0
                                        if leg_qty <= 0:
                                            try:
                                                prev_qty = int(prev_leg.get("quantity") or 0)
                                            except Exception:
                                                prev_qty = 0
                                            if prev_qty > 0:
                                                leg_copy["quantity"] = prev_qty
                                        for key in ("strike", "option_type", "expiry", "side", "entry_price", "token", "exchange"):
                                            if leg_copy.get(key) in (None, "") and prev_leg.get(key) not in (None, ""):
                                                leg_copy[key] = prev_leg.get(key)
                                    merged_legs.append(leg_copy)

                                if merged_legs:
                                    state["legs"] = merged_legs
                        except Exception:
                            pass
                        close_evt = evt
                        try:
                            if state.get("legs"):
                                close_evt = TradeLogEvent(
                                    ts=evt.ts,
                                    event=evt.event,
                                    trade_id=evt.trade_id,
                                    position_type=evt.position_type,
                                    name=evt.name,
                                    legs=list(state.get("legs") or []),
                                    mtm=evt.mtm,
                                    realized=evt.realized,
                                    reason=evt.reason,
                                    margin_required=evt.margin_required,
                                )
                        except Exception:
                            close_evt = evt
                        self._note_closed_option_legs(close_evt)
                        self._note_traded_qty(evt)
                        self._note_hedge_traded_qty(evt)
                    elif evt.event == "UPDATE":
                        # Keep status as-is.
                        state.setdefault("status", "OPEN")
                        # Count any incremental adds to open qty on UPDATE.
                        self._note_traded_qty(evt)
                        self._note_hedge_traded_qty(evt)

                    if evt.legs and evt.event not in {"PARTIAL_CLOSE", "CLOSE"}:
                        state["legs"] = evt.legs
                    if evt.mtm is not None:
                        state["mtm"] = float(evt.mtm)
                    if evt.realized is not None:
                        # Accumulate realized P&L for partial bookings
                        cur_realized = float(state.get("realized") or 0.0)
                        state["realized"] = cur_realized + float(evt.realized)

                    # Margin requirement (best-effort; present only when strategy computed it).
                    try:
                        mr = getattr(evt, "margin_required", None)
                        if mr is not None:
                            state["margin_required"] = float(mr)
                    except Exception:
                        pass

                    opened_ts = float(state.get("opened_ts") or evt.ts)
                    t_str = datetime.fromtimestamp(opened_ts).strftime("%H:%M:%S")
                    mtm, realized = self._compute_parent_display_pnl(trade_id, state)
                    mtm_str = "" if mtm is None else f"{float(mtm):.2f}"
                    realized_str = "" if realized is None else f"{float(realized):.2f}"

                    legs_all = state.get("legs") or []
                    main_legs, hedge_opt_legs, hedge_under_legs = self._split_legs_for_display(legs_all)
                    # Keep option hedges in Legs; underlying hedges get their own row.
                    display_main_legs: list[dict] = []
                    for leg in list(main_legs) + list(hedge_opt_legs):
                        if not isinstance(leg, dict):
                            continue
                        if self._get_leg_qty(leg) <= 0:
                            continue
                        display_main_legs.append(leg)
                    legs_str = self._format_legs(display_main_legs, show_prices=True, hedge_prefix=True)
                    values = (
                        t_str,
                        trade_id,
                        str(state.get("pos_type") or ""),
                        str(state.get("strategy") or ""),
                        legs_str,
                        mtm_str,
                        realized_str,
                        str(state.get("status") or ""),
                    )

                    row = self._trade_rows.get(trade_id)
                    if row is None:
                        row = self.trade_tree.insert("", tk.END, values=values)
                        self._trade_rows[trade_id] = row
                        # First time this trade shows up in the table.
                        print(
                            f"[UI] Paper Trade Log row created: {trade_id}"
                            f" ({state.get('pos_type','')} {state.get('strategy','')})"
                        )
                    else:
                        self.trade_tree.item(row, values=values)

                    # Apply row color-coding (profit/loss).
                    try:
                        tag = "neutral"
                        if str(trade_id).endswith("-H"):
                            tag = "hedge"
                        else:
                            status_s = str(state.get("status") or "").strip().upper()
                            mtm_f = None
                            r_f = None
                            try:
                                if mtm is not None:
                                    mtm_f = float(mtm)
                            except Exception:
                                mtm_f = None
                            try:
                                if realized is not None:
                                    r_f = float(realized)
                            except Exception:
                                r_f = None

                            # Prefer realized P&L when the trade is closed/partial.
                            ref = None
                            if status_s.startswith("CLOSED") or status_s.startswith("PARTIAL"):
                                ref = r_f
                            else:
                                ref = mtm_f

                            if ref is not None:
                                if float(ref) > 0:
                                    tag = "profit"
                                elif float(ref) < 0:
                                    tag = "loss"
                        if tag == "neutral":
                            self.trade_tree.item(row, tags=())
                        else:
                            self.trade_tree.item(row, tags=(tag,))
                    except Exception:
                        pass

                    # --- Underlying hedge row (separate position type) ---
                    hedge_trade_id = f"{trade_id}-H"
                    if hedge_under_legs:
                        hedge_state = self._trade_state.setdefault(hedge_trade_id, {})
                        hedge_state.setdefault("opened_ts", opened_ts)
                        hedge_state["pos_type"] = "delta_hedge"
                        base_name = str(state.get("strategy") or "")
                        hedge_state["strategy"] = f"{base_name} Δ-hedge".strip() if base_name else "Δ-hedge"
                        hedge_legs_enriched: list[dict[str, object]] = []
                        for lg in hedge_under_legs:
                            if not isinstance(lg, dict):
                                continue
                            leg_copy = dict(lg)
                            # Prefer live last price for MTM; if unavailable, use
                            # explicit exit_price (for closed snapshots) as fallback.
                            try:
                                has_ltp = leg_copy.get("ltp") is not None
                            except Exception:
                                has_ltp = False
                            if not has_ltp:
                                try:
                                    live_ltp = self._try_get_live_ltp_for_leg(client, leg_copy) if client is not None else None
                                except Exception:
                                    live_ltp = None
                                if live_ltp is not None:
                                    leg_copy["ltp"] = float(live_ltp)
                                elif leg_copy.get("exit_price") is not None:
                                    try:
                                        leg_copy["ltp"] = float(leg_copy.get("exit_price"))
                                    except Exception:
                                        pass
                            hedge_legs_enriched.append(leg_copy)

                        hedge_state["legs"] = list(hedge_legs_enriched)
                        hedge_state["status"] = str(state.get("status") or "").strip() or "OPEN"

                        hedge_mtm = self._compute_cached_trade_mtm(hedge_legs_enriched)
                        if hedge_mtm is not None:
                            hedge_state["mtm"] = float(hedge_mtm)
                        else:
                            hedge_state.pop("mtm", None)

                        # Hedge row is already a separate position: render legs like normal legs
                        # (no redundant HEDGE prefix) and without prices (less clutter).
                        hedge_legs_str = self._format_legs(
                            list(hedge_legs_enriched),
                            show_prices=True,
                            hedge_prefix=False,
                        )
                        hedge_mtm_val = hedge_state.get("mtm")
                        hedge_mtm_str = ""
                        try:
                            if hedge_mtm_val is not None:
                                hedge_mtm_str = f"{float(hedge_mtm_val):.2f}"
                        except Exception:
                            hedge_mtm_str = ""
                        if hedge_legs_str:
                            hedge_values = (
                                t_str,
                                hedge_trade_id,
                                str(hedge_state.get("pos_type") or ""),
                                str(hedge_state.get("strategy") or ""),
                                hedge_legs_str,
                                hedge_mtm_str,
                                "",  # keep realized blank for hedge row
                                str(hedge_state.get("status") or ""),
                            )

                            hedge_row = self._trade_rows.get(hedge_trade_id)
                            if hedge_row is None:
                                hedge_row = self.trade_tree.insert("", tk.END, values=hedge_values)
                                self._trade_rows[hedge_trade_id] = hedge_row
                            else:
                                self.trade_tree.item(hedge_row, values=hedge_values)

                            try:
                                self.trade_tree.item(hedge_row, tags=("hedge",))
                            except Exception:
                                pass
                        else:
                            hedge_row = self._trade_rows.pop(hedge_trade_id, None)
                            if hedge_row is not None:
                                try:
                                    self.trade_tree.delete(hedge_row)
                                except Exception:
                                    pass
                            try:
                                self._trade_state.pop(hedge_trade_id, None)
                            except Exception:
                                pass
                    else:
                        # If hedge is fully removed, hide the hedge row.
                        hedge_row = self._trade_rows.pop(hedge_trade_id, None)
                        if hedge_row is not None:
                            try:
                                self.trade_tree.delete(hedge_row)
                            except Exception:
                                pass
                        if hedge_trade_id in self._trade_state:
                            try:
                                self._trade_state.pop(hedge_trade_id, None)
                            except Exception:
                                pass

                    if evt.event == "CLOSE":
                        print(f"[UI] Paper Trade Log row closed: {trade_id} ({state.get('status','')})")
                except Exception as exc:  # noqa: BLE001
                    # Never let a UI rendering issue stop updates.
                    print(f"[UI] Failed to render trade event: {exc}")
                    continue

                if processed >= max_items:
                    break

        except queue.Empty:
            pass
        finally:
            if "processed" in locals() and processed:
                try:
                    self._refresh_pnl_totals()
                except Exception:
                    pass
                try:
                    self._dash_portfolio_ts = 0.0
                    self.after(0, self._pump_dashboard_portfolio)
                except Exception:
                    pass
                try:
                    self._render_signals_and_greeks(force=True)
                except Exception:
                    pass
                try:
                    print(
                        f"[UI] Paper Trade Log processed {processed} events"
                        f" (rows={len(self._trade_rows)})"
                    )
                except Exception:
                    pass
            # Also prune on the regular pump loop so any late queue events do not
            # leave stale OPEN/PARTIAL rows once the bot is idle.
            try:
                self._prune_stale_open_rows_when_idle()
            except Exception:
                pass
            self.after(150, self._pump_trades)

    def _on_show_settings(self) -> None:
        """Show a small window with current strategy settings and allow tweaks.

        This reads the effective StrategyConfig (including env overrides)
        so you can see what the bot will actually use, and lets you
        change a few key values via environment variables.
        """

        try:
            cfg = load_strategy_config()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load strategy config: {exc}")
            return

        win = tk.Toplevel(self)
        win.title("Strategy Settings")
        win.transient(self)
        try:
            sw = int(win.winfo_screenwidth())
            sh = int(win.winfo_screenheight())
            w = min(900, max(560, sw - 80))
            h = min(920, max(650, sh - 120))
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)
            win.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            win.geometry("560x750")
        win.minsize(560, 500)
        win.resizable(True, True)

        canvas = tk.Canvas(win, highlightthickness=0)
        vsb = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        content = tk.Frame(canvas)
        content_id = canvas.create_window((0, 0), window=content, anchor="nw")
        content.columnconfigure(1, weight=1)

        def _normalize_settings_timeframe(value: object, default: str = "1m") -> str:
            raw = str(value or "").strip().lower()
            aliases = {
                "1m": "1m",
                "m1": "1m",
                "one_minute": "1m",
                "3m": "3m",
                "m3": "3m",
                "three_minute": "3m",
                "5m": "5m",
                "m5": "5m",
                "five_minute": "5m",
                "10m": "10m",
                "m10": "10m",
                "ten_minute": "10m",
                "15m": "15m",
                "m15": "15m",
                "fifteen_minute": "15m",
                "30m": "30m",
                "m30": "30m",
                "thirty_minute": "30m",
            }
            return aliases.get(raw, default)

        def _on_content_configure(_evt: object = None) -> None:
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                return

        def _on_canvas_configure(evt: object) -> None:
            try:
                width = int(getattr(evt, "width"))
            except Exception:
                return
            try:
                canvas.itemconfigure(content_id, width=width)
            except Exception:
                return

        content.bind("<Configure>", lambda e: _on_content_configure(e))
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling (Windows/macOS/Linux best-effort).
        def _on_mousewheel(evt: object) -> None:
            try:
                delta = int(getattr(evt, "delta"))
            except Exception:
                delta = 0
            if delta:
                step = int(-delta / 120)
                if step == 0:
                    step = -1 if delta > 0 else 1
                canvas.yview_scroll(step, "units")
            return "break"

        def _on_linux_scroll(evt: object) -> None:
            num = getattr(evt, "num", None)
            if num == 4:
                canvas.yview_scroll(-1, "units")
            elif num == 5:
                canvas.yview_scroll(1, "units")
            return "break"

        def _on_key_scroll(evt: object) -> None:
            key = str(getattr(evt, "keysym", "")).lower()
            if key == "prior":
                canvas.yview_scroll(-1, "pages")
            elif key == "next":
                canvas.yview_scroll(1, "pages")
            elif key == "home":
                canvas.yview_moveto(0.0)
            elif key == "end":
                canvas.yview_moveto(1.0)
            else:
                return
            return "break"

        win.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        win.bind_all("<Button-4>", _on_linux_scroll, add="+")
        win.bind_all("<Button-5>", _on_linux_scroll, add="+")
        win.bind("<Prior>", _on_key_scroll, add="+")
        win.bind("<Next>", _on_key_scroll, add="+")
        win.bind("<Home>", _on_key_scroll, add="+")
        win.bind("<End>", _on_key_scroll, add="+")

        def _unbind_settings_scroll(_evt: object = None) -> None:
            try:
                win.unbind_all("<MouseWheel>")
            except Exception:
                pass
            try:
                win.unbind_all("<Button-4>")
            except Exception:
                pass
            try:
                win.unbind_all("<Button-5>")
            except Exception:
                pass

        win.bind("<Destroy>", _unbind_settings_scroll, add="+")

        row = 0
        tk.Label(content, text="Read-only summary (current values)", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 4), padx=8
        )

        def add_ro(label: str, value: object) -> None:
            nonlocal row
            row += 1
            tk.Label(content, text=label + ":").grid(row=row, column=0, sticky="w", padx=8)
            tk.Label(content, text=str(value)).grid(row=row, column=1, sticky="w", padx=8)

        add_ro("Strategy", cfg.strategy_name)
        add_ro("Delta Hedge Scope", getattr(cfg, "delta_hedge_scope", "strategy_only"))
        add_ro("Hedge Symbol (NIFTY)", getattr(cfg, "delta_hedge_symbol_nifty", "") or "(unset)")
        add_ro("Hedge Symbol (BANKNIFTY)", getattr(cfg, "delta_hedge_symbol_banknifty", "") or "(unset)")
        add_ro(
            "Delta hedge tol (entry/exit)",
            f"{cfg.delta_hedge_entry_tolerance} / {cfg.delta_hedge_exit_tolerance}",
        )
        add_ro(
            "Delta hedge adj/min move",
            f"{cfg.delta_hedge_adjustment_factor} / {cfg.delta_hedge_min_spot_move}+{cfg.delta_hedge_min_spot_move_atr_mult}xATR",
        )
        add_ro(
            "Delta hedge guardrails",
            f"max adj {cfg.delta_hedge_max_adjust_abs_qty}, max/day {cfg.delta_hedge_max_orders_per_day}, mkt hours {cfg.delta_hedge_market_hours_only}",
        )
        add_ro(
            "Delta hedge spread guard",
            f"req={cfg.delta_hedge_require_bid_ask}, {cfg.delta_hedge_max_spread_pct}%/{cfg.delta_hedge_max_spread_abs}",
        )
        add_ro("Symbol / Underlying", f"{cfg.symbol} / {cfg.underlying}")
        add_ro("Timeframe / Lot Size", f"{os.getenv('MSTOCK_TIMEFRAME', '1m')} / {cfg.lot_size}")
        add_ro("Max Open / Trades/Day", f"{cfg.max_open_positions} / {cfg.max_trades_per_day}")
        add_ro("Max Daily Loss", cfg.max_daily_loss)
        add_ro(
            "Risk scaling",
            f"{cfg.risk_scale_enabled} (min/max/step {cfg.risk_scale_min_qty}/{cfg.risk_scale_max_qty}/{cfg.risk_scale_step_qty})",
        )
        add_ro("Cooldown", f"{cfg.cooldown_sec}s")
        add_ro("Cooldown after stopout / ATR", f"{cfg.cooldown_after_stopout_sec}s / {cfg.cooldown_atr_mult}")
        add_ro(
            "Equity GPT / watchlist",
            f"{getattr(cfg, 'equity_trade_engine', 'INDICATORS')} / GPT list={getattr(cfg, 'equity_trade_gpt_watchlist_enable', False)} / {getattr(cfg, 'equity_trade_symbols', '') or '(holdings only)'}",
        )
        add_ro(
            "Equity horizon / products",
            f"{getattr(cfg, 'equity_trade_horizon', 'INTRADAY')} / {getattr(cfg, 'equity_trade_product_type', 'INTRADAY')} / {getattr(cfg, 'equity_trade_longterm_product_type', 'CNC')}",
        )
        add_ro(
            "Equity capital / max notional",
            f"{getattr(cfg, 'equity_trade_capital_rupees', 0.0)} / {getattr(cfg, 'equity_trade_max_notional', 0.0)}",
        )
        add_ro(
            "Equity daily GPT manage",
            f"{getattr(cfg, 'equity_trade_gpt_manage_daily', True)}",
        )
        add_ro(
            "Equity GPT refresh / candidates",
            f"{getattr(cfg, 'equity_trade_watchlist_refresh_sec', 900.0)}s / {getattr(cfg, 'equity_trade_candidate_limit', 20)}",
        )
        add_ro(
            "Entry candle guard (age/range/gap)",
            f"{cfg.entry_candle_max_age_sec}s / {cfg.entry_max_candle_range_atr_mult} / {cfg.entry_gap_atr_mult}",
        )
        add_ro("Short dist / Straddle ATR", f"{cfg.short_strike_distance} / {cfg.straddle_atr_threshold}")
        add_ro("Wing width / Hold min/days", f"{cfg.wing_width} / {cfg.max_hold_minutes}/{getattr(cfg, 'max_hold_days', 0)}")
        add_ro("Trend Filter (max/enabled)", f"{cfg.max_trend_strength} / {cfg.enable_trend_filter}")
        add_ro("Supertrend Filter", f"{cfg.enable_supertrend_filter}")
        add_ro("EMA Fast / Slow", f"{cfg.ema_fast} / {cfg.ema_slow}")
        add_ro("Premium RSI Filter", f"{cfg.enable_premium_rsi_filter} ({cfg.premium_rsi_low}-{cfg.premium_rsi_high})")
        add_ro(
            "Entry premium bounds (leg/total)",
            f"{cfg.entry_min_option_premium}-{cfg.entry_max_option_premium} / {cfg.entry_min_total_premium}-{cfg.entry_max_total_premium}",
        )
        add_ro(
            "Entry liquidity guard",
            f"req={cfg.entry_require_bid_ask}, spread {cfg.entry_max_bid_ask_spread_pct}%/{cfg.entry_max_bid_ask_spread_abs}",
        )
        add_ro("IV regime filter (expand/contract)", f"{cfg.iv_expand_threshold_pct}% / {cfg.iv_contract_threshold_pct}%")

        add_ro("ATR Period / Spike x", f"{cfg.atr_period} / {cfg.atr_spike_mult}")
        add_ro("Min / Max ATR", f"{cfg.min_atr} / {cfg.max_atr}")
        add_ro("MTM stop / target %", f"{cfg.premium_mtm_stop_pct*100:.0f}% / {cfg.premium_mtm_target_pct*100:.0f}%")
        add_ro("MTM Trail (Start/Stop)", f"Start={cfg.premium_mtm_trail_start_pct*100:.1f}%, Stop={cfg.premium_mtm_trail_stop_pct*100:.1f}%")
        add_ro(
            "GPT leg MTM manage / refresh",
            f"{getattr(cfg, 'gpt_leg_manage', False)} / {getattr(cfg, 'gpt_leg_manage_refresh_sec', 45.0)}s",
        )
        add_ro(
            "GPT leg min visible age",
            f"{getattr(cfg, 'gpt_leg_manage_min_age_sec', 90.0)}s",
        )
        add_ro(
            "GPT leg grace / min dist",
            (
                f"{getattr(cfg, 'gpt_leg_manage_post_update_grace_sec', 20.0)}s / "
                f"S:{getattr(cfg, 'gpt_leg_manage_min_stop_distance_pct', 0.08):.2f}, "
                f"T:{getattr(cfg, 'gpt_leg_manage_min_target_distance_pct', 0.10):.2f}"
            ),
        )
        add_ro("Hard exit / Cutoff", f"{cfg.premium_force_exit_hhmm} / {cfg.premium_entry_cutoff_hhmm or '(none)'}")
        add_ro("VWAP max dev / enabled", f"{cfg.vwap_max_dev_pct}% / {cfg.enable_vwap_filter}")
        add_ro("Opening max move / window", f"{cfg.opening_max_move_pct}% / {cfg.opening_filter_minutes}m ({cfg.enable_opening_filter})")
        add_ro("Max stopouts / consec", f"{cfg.max_stopouts_per_day} / {cfg.max_consecutive_stopouts}")
        add_ro("Directional Trail Stop x", f"BE={cfg.dir_breakeven_atr_mult}, Trail={cfg.dir_trail_atr_mult}, Prem={cfg.dir_premium_trail_pct*100:.0f}%")
        add_ro(
            "Directional post-partial BE/Trail",
            f"{cfg.dir_post_partial_be_atr_mult} / {cfg.dir_post_partial_trail_atr_mult}",
        )
        add_ro("Directional EMA slope xATR", cfg.dir_ema_slope_atr_mult)
        add_ro("Directional exposure cap", cfg.entry_max_same_direction_qty)
        add_ro("Directional Partials", f"{cfg.dir_partial_qty_pct*100:.0f}% @ {cfg.dir_partial_target_mult}x")
        add_ro("Auto range trend x", cfg.auto_range_trend_mult)
        add_ro("Auto RSI", f"B={cfg.auto_dir_rsi_buy}, S={cfg.auto_dir_rsi_sell}, slop={getattr(cfg, 'auto_dir_rsi_slop', 1.0)}")
        add_ro("Router mode / diagnostics", f"{getattr(cfg, 'strategy_router_mode', 'balanced')} / {getattr(cfg, 'diagnostics_enabled', True)}")
        add_ro(
            "Portfolio caps (notional/|delta|)",
            f"{getattr(cfg, 'max_portfolio_option_notional', 0.0)} / {getattr(cfg, 'max_portfolio_delta_abs', 0.0)}",
        )
        add_ro(
            "Order retries (attempts/delay)",
            f"{getattr(cfg, 'order_retry_attempts', 2)} / {getattr(cfg, 'order_retry_delay_sec', 0.25)}s",
        )
        add_ro("Weekly / Target Expiry", f"{getattr(cfg, 'nifty_weekly_only', True)} / {cfg.target_expiry or '(auto)'}")
        warn_rows = list(getattr(cfg, "validation_warnings", []) or [])
        add_ro("Config warnings", " | ".join(str(w) for w in warn_rows) if warn_rows else "(none)")

        # Signal Quality
        add_ro("MTF Confirmation", f"{cfg.enable_mtf_confirmation} ({cfg.mtf_timeframe})")
        add_ro("ROC Filter", f"{cfg.enable_roc_filter} (p={cfg.roc_period}, thr={cfg.roc_min_threshold})")
        add_ro("Choppiness Filter", f"{cfg.enable_choppiness_filter} (p={cfg.choppiness_period}, thr={cfg.choppiness_threshold})")

        # Risk Management
        add_ro("Stagnation Exit", f"{cfg.stagnation_exit_minutes}m (thr={cfg.stagnation_pnl_threshold})")
        add_ro("Chandelier Exit", f"{cfg.enable_chandelier_exit} (p={cfg.chandelier_period}, x={cfg.chandelier_multiplier})")
        add_ro("Pivot Point Targets", f"{cfg.enable_pivot_targets}")

        # Execution
        add_ro("Limit Orders", f"{cfg.enable_limit_orders} (buf={cfg.limit_price_buffer_pct}%)")
        add_ro("Max Pyramiding", f"{cfg.max_pyramid_levels}")

        row += 1
        tk.Label(content, text="Quick tweaks (apply then restart bot)", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 4), padx=8
        )

        # Editable entries for a few high-impact knobs.
        row += 1
        tk.Label(content, text="Strategy").grid(row=row, column=0, sticky="w", padx=8)
        strategy_var = tk.StringVar(value=str(cfg.strategy_name or "directional"))
        tk.OptionMenu(
            content,
            strategy_var,
            *_STRATEGY_CHOICES,
        ).grid(row=row, column=1, sticky="w", padx=8)

        delta_scope_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_scope", "strategy_only") or "strategy_only"))
        tk.OptionMenu(
            content,
            delta_scope_var,
            "strategy_only",
            "multi_only",
            "all_options",
        ).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Hedge symbol (NIFTY)").grid(row=row, column=0, sticky="w", padx=8)
        hedge_nifty_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_symbol_nifty", "") or ""))
        tk.Entry(content, textvariable=hedge_nifty_var, width=28).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Hedge symbol (BANKNIFTY)").grid(row=row, column=0, sticky="w", padx=8)
        hedge_bank_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_symbol_banknifty", "") or ""))
        tk.Entry(content, textvariable=hedge_bank_var, width=28).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Delta hedge base tolerance").grid(row=row, column=0, sticky="w", padx=8)
        dh_base_tol_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_delta_tolerance", 5.0)))
        tk.Entry(content, textvariable=dh_base_tol_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Delta hedge entry/exit tol").grid(row=row, column=0, sticky="w", padx=8)
        dh_tol_frame = tk.Frame(content)
        dh_tol_frame.grid(row=row, column=1, sticky="w", padx=8)
        dh_entry_tol_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_entry_tolerance", 0.0)))
        tk.Entry(dh_tol_frame, textvariable=dh_entry_tol_var, width=10).pack(side=tk.LEFT)
        dh_exit_tol_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_exit_tolerance", 0.0)))
        tk.Entry(dh_tol_frame, textvariable=dh_exit_tol_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Delta hedge adj factor / max adj qty").grid(row=row, column=0, sticky="w", padx=8)
        dh_adj_frame = tk.Frame(content)
        dh_adj_frame.grid(row=row, column=1, sticky="w", padx=8)
        dh_adjust_factor_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_adjustment_factor", 1.0)))
        tk.Entry(dh_adj_frame, textvariable=dh_adjust_factor_var, width=10).pack(side=tk.LEFT)
        dh_max_adjust_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_max_adjust_abs_qty", 0.0)))
        tk.Entry(dh_adj_frame, textvariable=dh_max_adjust_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Delta hedge min spot move / ATR x").grid(row=row, column=0, sticky="w", padx=8)
        dh_move_frame = tk.Frame(content)
        dh_move_frame.grid(row=row, column=1, sticky="w", padx=8)
        dh_min_spot_move_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_min_spot_move", 0.0)))
        tk.Entry(dh_move_frame, textvariable=dh_min_spot_move_var, width=10).pack(side=tk.LEFT)
        dh_min_spot_atr_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_min_spot_move_atr_mult", 0.0)))
        tk.Entry(dh_move_frame, textvariable=dh_min_spot_atr_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Delta hedge max orders/day").grid(row=row, column=0, sticky="w", padx=8)
        dh_max_orders_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_max_orders_per_day", 0)))
        tk.Entry(content, textvariable=dh_max_orders_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        dh_hours_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_market_hours_only", False)))
        ttk.Checkbutton(content, text="Delta hedge: market hours only", variable=dh_hours_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        dh_req_ba_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_require_bid_ask", False)))
        ttk.Checkbutton(content, text="Delta hedge: require bid/ask", variable=dh_req_ba_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        dh_inc_pos_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_include_broker_positions", False)))
        ttk.Checkbutton(content, text="Delta hedge: include broker positions", variable=dh_inc_pos_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        dh_inc_hold_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_include_holdings", False)))
        ttk.Checkbutton(content, text="Delta hedge: include holdings (ETF shares)", variable=dh_inc_hold_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        dh_beta_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_include_equity_portfolio_beta", False)))
        ttk.Checkbutton(content, text="Delta hedge: beta-hedge equity holdings", variable=dh_beta_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Beta hedge lookback days / max symbols").grid(row=row, column=0, sticky="w", padx=8)
        dh_beta_frame = tk.Frame(content)
        dh_beta_frame.grid(row=row, column=1, sticky="w", padx=8)
        dh_beta_lookback_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_beta_lookback_days", 90)))
        tk.Entry(dh_beta_frame, textvariable=dh_beta_lookback_var, width=10).pack(side=tk.LEFT)
        dh_beta_maxsym_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_beta_max_symbols", 12)))
        tk.Entry(dh_beta_frame, textvariable=dh_beta_maxsym_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Beta hedge refresh (sec)").grid(row=row, column=0, sticky="w", padx=8)
        dh_beta_refresh_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_beta_refresh_sec", 300.0)))
        tk.Entry(content, textvariable=dh_beta_refresh_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        dh_portfolio_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_portfolio_hedge_enable", False)))
        ttk.Checkbutton(
            content,
            text="Delta hedge: run portfolio beta hedge (even with no option trades)",
            variable=dh_portfolio_var,
        ).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Equity trading (holdings)", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 2), padx=8
        )

        row += 1
        eq_enable_var = tk.BooleanVar(value=bool(getattr(cfg, "equity_trade_enable", False)))
        ttk.Checkbutton(content, text="Enable equity trading on holdings", variable=eq_enable_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        eq_short_var = tk.BooleanVar(value=bool(getattr(cfg, "equity_trade_allow_short", False)))
        ttk.Checkbutton(content, text="Allow short (INTRADAY)", variable=eq_short_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Equity engine / horizon").grid(row=row, column=0, sticky="w", padx=8)
        eq_eh_frame = tk.Frame(content)
        eq_eh_frame.grid(row=row, column=1, sticky="w", padx=8)
        eq_engine_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_engine", "INDICATORS") or "INDICATORS").strip().upper())
        tk.OptionMenu(eq_eh_frame, eq_engine_var, "INDICATORS", "GPT").pack(side=tk.LEFT)
        eq_horizon_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_horizon", "INTRADAY") or "INTRADAY").strip().upper())
        tk.OptionMenu(eq_eh_frame, eq_horizon_var, "INTRADAY", "LONGTERM", "BOTH").pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        eq_churn_var = tk.BooleanVar(value=bool(getattr(cfg, "equity_trade_churn_enable", False)))
        ttk.Checkbutton(content, text="Equity: allow portfolio churn (GPT engine)", variable=eq_churn_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Equity watchlist symbols").grid(row=row, column=0, sticky="w", padx=8)
        eq_symbols_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_symbols", "") or ""))
        tk.Entry(content, textvariable=eq_symbols_var, width=42).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        eq_gpt_watchlist_var = tk.BooleanVar(value=bool(getattr(cfg, "equity_trade_gpt_watchlist_enable", False)))
        ttk.Checkbutton(content, text="GPT creates watchlist from equity universe", variable=eq_gpt_watchlist_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="GPT refresh sec / candidate limit").grid(row=row, column=0, sticky="w", padx=8)
        eq_watch_frame = tk.Frame(content)
        eq_watch_frame.grid(row=row, column=1, sticky="w", padx=8)
        eq_watch_refresh_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_watchlist_refresh_sec", 900.0) or 900.0))
        tk.Entry(eq_watch_frame, textvariable=eq_watch_refresh_var, width=10).pack(side=tk.LEFT)
        eq_candidate_limit_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_candidate_limit", 20) or 20))
        tk.Entry(eq_watch_frame, textvariable=eq_candidate_limit_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Longterm target % / horizon days").grid(row=row, column=0, sticky="w", padx=8)
        eq_lt_frame = tk.Frame(content)
        eq_lt_frame.grid(row=row, column=1, sticky="w", padx=8)
        eq_lt_target_var = tk.StringVar(value=str(getattr(cfg, "equity_longterm_target_pct", 0.0) or 0.0))
        tk.Entry(eq_lt_frame, textvariable=eq_lt_target_var, width=8).pack(side=tk.LEFT)
        eq_lt_days_var = tk.StringVar(value=str(getattr(cfg, "equity_longterm_horizon_days", 90) or 90))
        tk.Entry(eq_lt_frame, textvariable=eq_lt_days_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Equity product type / longterm").grid(row=row, column=0, sticky="w", padx=8)
        eq_prod_frame = tk.Frame(content)
        eq_prod_frame.grid(row=row, column=1, sticky="w", padx=8)
        eq_prod_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_product_type", "INTRADAY") or "INTRADAY").strip().upper())
        tk.OptionMenu(eq_prod_frame, eq_prod_var, "INTRADAY", "CNC", "DELIVERY").pack(side=tk.LEFT)
        eq_prod_long_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_longterm_product_type", "CNC") or "CNC").strip().upper())
        tk.OptionMenu(eq_prod_frame, eq_prod_long_var, "CNC", "DELIVERY", "INTRADAY").pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Equity timeframe").grid(row=row, column=0, sticky="w", padx=8)
        eq_tf_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_timeframe", "5m") or "5m"))
        tk.OptionMenu(content, eq_tf_var, "1m", "3m", "5m", "10m", "15m", "30m", "1d").grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Max symbols / rebalance sec").grid(row=row, column=0, sticky="w", padx=8)
        eq_frame1 = tk.Frame(content)
        eq_frame1.grid(row=row, column=1, sticky="w", padx=8)
        eq_maxsym_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_max_symbols", 10)))
        tk.Entry(eq_frame1, textvariable=eq_maxsym_var, width=6).pack(side=tk.LEFT)
        eq_reb_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_rebalance_interval_sec", 60.0)))
        tk.Entry(eq_frame1, textvariable=eq_reb_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="ATR period / SLx / TPx").grid(row=row, column=0, sticky="w", padx=8)
        eq_frame2 = tk.Frame(content)
        eq_frame2.grid(row=row, column=1, sticky="w", padx=8)
        eq_atr_p_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_atr_period", 14)))
        tk.Entry(eq_frame2, textvariable=eq_atr_p_var, width=6).pack(side=tk.LEFT)
        eq_sl_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_sl_atr_mult", 2.0)))
        tk.Entry(eq_frame2, textvariable=eq_sl_var, width=6).pack(side=tk.LEFT, padx=(6, 0))
        eq_tp_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_tp_atr_mult", 3.0)))
        tk.Entry(eq_frame2, textvariable=eq_tp_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Risk ₹ / max notional ₹").grid(row=row, column=0, sticky="w", padx=8)
        eq_frame3 = tk.Frame(content)
        eq_frame3.grid(row=row, column=1, sticky="w", padx=8)
        eq_risk_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_risk_rupees", 500.0)))
        tk.Entry(eq_frame3, textvariable=eq_risk_var, width=10).pack(side=tk.LEFT)
        eq_maxnot_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_max_notional", 20000.0)))
        tk.Entry(eq_frame3, textvariable=eq_maxnot_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Equity capital ₹ / GPT daily manage").grid(row=row, column=0, sticky="w", padx=8)
        eq_frame4 = tk.Frame(content)
        eq_frame4.grid(row=row, column=1, sticky="w", padx=8)
        eq_capital_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_capital_rupees", 0.0) or 0.0))
        tk.Entry(eq_frame4, textvariable=eq_capital_var, width=10).pack(side=tk.LEFT)
        eq_gpt_daily_var = tk.BooleanVar(value=bool(getattr(cfg, "equity_trade_gpt_manage_daily", True)))
        ttk.Checkbutton(eq_frame4, text="Once/day", variable=eq_gpt_daily_var).pack(side=tk.LEFT, padx=(8, 0))

        row += 1
        tk.Label(content, text="Equity cooldown sec").grid(row=row, column=0, sticky="w", padx=8)
        eq_cd_var = tk.StringVar(value=str(getattr(cfg, "equity_trade_cooldown_sec", 300.0)))
        tk.Entry(content, textvariable=eq_cd_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Delta hedge max spread % / abs").grid(row=row, column=0, sticky="w", padx=8)
        dh_spread_frame = tk.Frame(content)
        dh_spread_frame.grid(row=row, column=1, sticky="w", padx=8)
        dh_max_spread_pct_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_max_spread_pct", 0.0)))
        tk.Entry(dh_spread_frame, textvariable=dh_max_spread_pct_var, width=10).pack(side=tk.LEFT)
        dh_max_spread_abs_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_max_spread_abs", 0.0)))
        tk.Entry(dh_spread_frame, textvariable=dh_max_spread_abs_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Symbol / Underlying").grid(row=row, column=0, sticky="w", padx=8)
        sym_frame = tk.Frame(content)
        sym_frame.grid(row=row, column=1, sticky="w", padx=8)
        symbol_var = tk.StringVar(value=str(cfg.symbol))
        tk.Entry(sym_frame, textvariable=symbol_var, width=12).pack(side=tk.LEFT)
        underlying_var = tk.StringVar(value=str(cfg.underlying))
        tk.Entry(sym_frame, textvariable=underlying_var, width=12).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Timeframe").grid(row=row, column=0, sticky="w", padx=8)
        timeframe_var = tk.StringVar(value=_normalize_settings_timeframe(os.getenv("MSTOCK_TIMEFRAME", "1m"), "1m"))
        tk.OptionMenu(
            content,
            timeframe_var,
            "1m", "3m", "5m", "10m", "15m", "30m"
        ).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Preset").grid(row=row, column=0, sticky="w", padx=8)
        preset_frame = tk.Frame(content)
        preset_frame.grid(row=row, column=1, sticky="w", padx=8)
        _preset_env = (os.getenv("MSTOCK_PRESET", "") or "").strip().lower()
        _preset_ui = "(none)"
        if _preset_env in {"aggressive", "aggresive"}:
            _preset_ui = "Aggressive"
        elif _preset_env == "conservative":
            _preset_ui = "Conservative"
        preset_var = tk.StringVar(value=_preset_ui)
        tk.OptionMenu(
            preset_frame,
            preset_var,
            "(none)",
            "Aggressive",
            "Conservative",
        ).pack(side=tk.LEFT)

        row += 1
        tk.Label(content, text="Modified entries").grid(row=row, column=0, sticky="nw", padx=8)
        preset_changes_text = tk.Text(content, width=44, height=6, wrap="word")
        preset_changes_text.grid(row=row, column=1, sticky="w", padx=8)
        preset_changes_text.insert("1.0", "(apply a preset to preview changes)")
        preset_changes_text.configure(state=tk.DISABLED)

        def _set_preset_changes(lines: list[str]) -> None:
            try:
                preset_changes_text.configure(state=tk.NORMAL)
                preset_changes_text.delete("1.0", tk.END)
                preset_changes_text.insert("1.0", "\n".join(lines) if lines else "(no changes)")
                preset_changes_text.configure(state=tk.DISABLED)
            except Exception:
                pass

        def _snapshot_preset_fields() -> dict[str, str]:
            snap: dict[str, str] = {}
            # List of all variables to track for preset diffs
            vars_to_track = {
                "Strategy": strategy_var,
                "Timeframe": timeframe_var,
                "Delta Hedge Scope": delta_scope_var,
                "Hedge Symbol (NIFTY)": hedge_nifty_var,
                "Hedge Symbol (BANKNIFTY)": hedge_bank_var,
                "Cooldown": cooldown_var,
                "Max Hold": max_hold_var,
                "Max Hold Days": max_hold_days_var,
                "Max Same Type Streak": max_same_type_var,
                "Max Same Dir Qty": max_same_dir_qty_var,
                "Cooldown Stopout": cooldown_stopout_var,
                "Cooldown ATR Mult": cooldown_atr_mult_var,
                "Dir Prem Trail": dir_prem_trail_var,
                "Dir BE": dir_be_var,
                "Dir Trail ATR": dir_trail_atr_var,
                "Dir EMA Slope": dir_ema_slope_var,
                "Dir Partial Target": dir_partial_tgt_var,
                "Dir Partial Qty": dir_partial_qty_var,
                "Dir Post BE": dir_post_be_var,
                "Dir Post Trail": dir_post_trail_var,
                "Req Spot LTP": req_spot_var,
                "Supertrend Filter": supertrend_enabled_var,
                "Supertrend Mode": supertrend_mode_var,
                "Trend Filter": trend_enabled_var,
                "ADX Filter": adx_enabled_var,
                "Vol Filter": vol_enabled_var,
                "Opening Filter": opening_enabled_var,
                "Max Pyramiding": pyramid_var,
                "MTF Confirmation": mtf_enabled_var,
                "ROC Filter": roc_enabled_var,
                "CHOP Filter": chop_enabled_var,
                "Chandelier Exit": chan_enabled_var,
                "Pivot Targets": pivot_enabled_var,
                "Limit Orders": limit_enabled_var,
                "VWAP Filter": vwap_enabled_var,
                "Premium RSI Filter": premium_rsi_enabled_var,
                "Premium RSI Low": premium_rsi_low_var,
                "Premium RSI High": premium_rsi_high_var,
                "Entry Candle Age": entry_candle_age_var,
                "Entry Candle Range": entry_candle_range_var,
                "Entry Gap": entry_gap_var,
                "Entry Min Prem": entry_min_prem_var,
                "Entry Max Prem": entry_max_prem_var,
                "Entry Min Total": entry_min_total_prem_var,
                "Entry Max Total": entry_max_total_prem_var,
                "Entry Require BA": entry_req_ba_var,
                "Entry Spread %": entry_spread_pct_var,
                "Entry Spread Abs": entry_spread_abs_var,
                "IV Expand %": iv_expand_var,
                "IV Contract %": iv_contract_var,
                "GPT Leg Manage": gpt_leg_manage_var,
                "GPT Leg Refresh": gpt_leg_refresh_var,
                "GPT Leg Min Age": gpt_leg_min_age_var,
                "GPT Leg Grace": gpt_leg_grace_var,
                "GPT Leg Stop Dist": gpt_leg_stop_dist_var,
                "GPT Leg Target Dist": gpt_leg_target_dist_var,
                "Risk Scale": risk_scale_enabled_var,
                "Risk Stopout Factor": risk_scale_stopout_var,
                "Risk Recovery Wins": risk_scale_recovery_var,
                "Risk ATR High": risk_scale_atr_high_var,
                "Risk ATR Factor": risk_scale_atr_factor_var,
                "Risk Min Qty": risk_scale_min_qty_var,
                "Risk Max Qty": risk_scale_max_qty_var,
                "Risk Step Qty": risk_scale_step_qty_var,
                "Max Open Positions": max_pos_var,
                "DH Base Tol": dh_base_tol_var,
                "DH Entry Tol": dh_entry_tol_var,
                "DH Exit Tol": dh_exit_tol_var,
                "DH Adjust Factor": dh_adjust_factor_var,
                "DH Max Adjust": dh_max_adjust_var,
                "DH Min Move": dh_min_spot_move_var,
                "DH Min Move ATR": dh_min_spot_atr_var,
                "DH Max Orders": dh_max_orders_var,
                "DH Market Hours": dh_hours_var,
                "DH Require BA": dh_req_ba_var,
                "DH Max Spread %": dh_max_spread_pct_var,
                "DH Max Spread Abs": dh_max_spread_abs_var,
                "Dir Min Confirm": dir_min_confirm_var,
                "Dir Min Diff": dir_min_diff_var,
                "Dir Mom xATR": dir_mom_atr_mult_var,
                "Dir Allow Tie": dir_allow_tie_var,
                "Exit Short Touch": exit_short_var,
                "Exit Wing Touch": exit_wing_var,
            }
            for label, var in vars_to_track.items():
                try:
                    val = var.get()
                    snap[label] = str(val).lower() if isinstance(val, bool) else str(val)
                except Exception:
                    pass
            return snap

        def apply_preset() -> None:
            name = str(preset_var.get() or "").strip()
            before = _snapshot_preset_fields()

            cur_strategy = str(strategy_var.get() or "").strip().lower()
            allow_strategy_override = cur_strategy != "auto"
            try:
                preset_lot_size = int(float(str(lot_var.get() or "").strip()))
            except Exception:
                try:
                    preset_lot_size = int(getattr(cfg, "lot_size", 65) or 65)
                except Exception:
                    preset_lot_size = 65
            preset_position_limit = 2
            preset_max_qty = str(preset_position_limit * preset_lot_size)

            if name == "Aggressive":
                if allow_strategy_override:
                    strategy_var.set("auto")
                timeframe_var.set("1m")
                delta_scope_var.set("all_options")
                if not str(hedge_nifty_var.get() or "").strip():
                    hedge_nifty_var.set("NSE:NIFTYBEES")
                if not str(hedge_bank_var.get() or "").strip():
                    hedge_bank_var.set("NSE:BANKBEES")

                cooldown_var.set("20")
                cooldown_stopout_var.set("75")
                cooldown_atr_mult_var.set("0.25")
                max_hold_var.set("35")
                max_same_type_var.set("0")
                max_same_dir_qty_var.set(preset_max_qty)
                dir_prem_trail_var.set("0.09")
                dir_be_var.set("0.65")
                dir_trail_atr_var.set("1.05")
                dir_ema_slope_var.set("0.035")
                dir_min_confirm_var.set("3")
                dir_min_diff_var.set("0")
                dir_mom_atr_mult_var.set("0.07")
                dir_allow_tie_var.set(True)
                dir_partial_tgt_var.set("1.4")
                dir_partial_qty_var.set("0.35")
                dir_post_be_var.set("0.35")
                dir_post_trail_var.set("1.0")
                req_spot_var.set(True)
                supertrend_enabled_var.set(True)
                supertrend_mode_var.set("counter")
                trend_enabled_var.set(True)
                adx_enabled_var.set(True)
                vol_enabled_var.set(True)
                opening_enabled_var.set(False)
                pyramid_var.set(str(preset_position_limit))
                mtf_enabled_var.set(False)
                roc_enabled_var.set(False)
                chop_enabled_var.set(False)
                chan_enabled_var.set(False)
                pivot_enabled_var.set(False)
                limit_enabled_var.set(False)
                vwap_enabled_var.set(True)
                premium_rsi_enabled_var.set(True)
                premium_rsi_low_var.set("42")
                premium_rsi_high_var.set("58")
                entry_candle_age_var.set("150")
                entry_candle_range_var.set("3.5")
                entry_gap_var.set("2.0")
                entry_req_ba_var.set(False)
                iv_expand_var.set("4.5")
                iv_contract_var.set("4.0")
                gpt_leg_refresh_var.set("30")
                gpt_leg_min_age_var.set("60")
                gpt_leg_grace_var.set("12")
                gpt_leg_stop_dist_var.set("0.06")
                gpt_leg_target_dist_var.set("0.10")
                risk_scale_enabled_var.set(True)
                risk_scale_stopout_var.set("0.9")
                risk_scale_recovery_var.set("1")
                risk_scale_atr_high_var.set("85")
                risk_scale_atr_factor_var.set("0.9")
                risk_scale_min_qty_var.set(str(preset_lot_size))
                risk_scale_max_qty_var.set(preset_max_qty)
                risk_scale_step_qty_var.set(str(preset_lot_size))
                max_pos_var.set(str(preset_position_limit))
                exit_short_var.set(True)
                exit_wing_var.set(True)

            elif name == "Conservative":
                if allow_strategy_override:
                    strategy_var.set("auto")
                timeframe_var.set("3m")
                delta_scope_var.set("all_options")
                if not str(hedge_nifty_var.get() or "").strip():
                    hedge_nifty_var.set("NSE:NIFTYBEES")
                if not str(hedge_bank_var.get() or "").strip():
                    hedge_bank_var.set("NSE:BANKBEES")

                cooldown_var.set("45")
                cooldown_stopout_var.set("180")
                cooldown_atr_mult_var.set("0.45")
                max_hold_var.set("60")
                max_same_type_var.set("0")
                max_same_dir_qty_var.set(preset_max_qty)
                dir_prem_trail_var.set("0.12")
                dir_be_var.set("0.60")
                dir_trail_atr_var.set("1.10")
                dir_ema_slope_var.set("0.06")
                dir_min_confirm_var.set("4")
                dir_min_diff_var.set("1")
                dir_mom_atr_mult_var.set("0.12")
                dir_allow_tie_var.set(False)
                dir_partial_tgt_var.set("1.15")
                dir_partial_qty_var.set("0.25")
                dir_post_be_var.set("0.30")
                dir_post_trail_var.set("1.0")
                req_spot_var.set(True)
                supertrend_enabled_var.set(True)
                supertrend_mode_var.set("trend")
                trend_enabled_var.set(True)
                adx_enabled_var.set(True)
                vol_enabled_var.set(True)
                opening_enabled_var.set(True)
                pyramid_var.set(str(preset_position_limit))
                mtf_enabled_var.set(False)
                roc_enabled_var.set(False)
                chop_enabled_var.set(False)
                chan_enabled_var.set(False)
                pivot_enabled_var.set(False)
                limit_enabled_var.set(False)
                vwap_enabled_var.set(True)
                premium_rsi_enabled_var.set(True)
                premium_rsi_low_var.set("43")
                premium_rsi_high_var.set("57")
                entry_candle_age_var.set("150")
                entry_candle_range_var.set("2.3")
                entry_gap_var.set("1.4")
                entry_req_ba_var.set(True)
                entry_spread_pct_var.set("0.45")
                iv_expand_var.set("3.0")
                iv_contract_var.set("3.2")
                gpt_leg_refresh_var.set("60")
                gpt_leg_min_age_var.set("120")
                gpt_leg_grace_var.set("25")
                gpt_leg_stop_dist_var.set("0.10")
                gpt_leg_target_dist_var.set("0.14")
                risk_scale_enabled_var.set(True)
                risk_scale_stopout_var.set("0.75")
                risk_scale_recovery_var.set("2")
                risk_scale_atr_high_var.set("70")
                risk_scale_atr_factor_var.set("0.8")
                risk_scale_min_qty_var.set(str(preset_lot_size))
                risk_scale_max_qty_var.set(preset_max_qty)
                risk_scale_step_qty_var.set(str(preset_lot_size))
                max_pos_var.set(str(preset_position_limit))
                exit_short_var.set(True)
                exit_wing_var.set(False)

            after = _snapshot_preset_fields()
            keys = list({*before.keys(), *after.keys()})
            keys.sort()
            diffs = [f"{k}: {before.get(k, '')} -> {after.get(k, '')}" for k in keys if before.get(k) != after.get(k)]
            _set_preset_changes(diffs)

        tk.Button(preset_frame, text="Apply preset", command=apply_preset).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Short strike distance").grid(row=row, column=0, sticky="w", padx=8)
        dist_var = tk.StringVar(value=str(cfg.short_strike_distance))
        tk.Entry(content, textvariable=dist_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Straddle ATR threshold").grid(row=row, column=0, sticky="w", padx=8)
        atr_thr_var = tk.StringVar(value=str(cfg.straddle_atr_threshold))
        tk.Entry(content, textvariable=atr_thr_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Wing width (condor)").grid(row=row, column=0, sticky="w", padx=8)
        wing_var = tk.StringVar(value=str(cfg.wing_width))
        tk.Entry(content, textvariable=wing_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Lot size").grid(row=row, column=0, sticky="w", padx=8)
        lot_var = tk.StringVar(value=str(cfg.lot_size))
        tk.Entry(content, textvariable=lot_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max trades/day").grid(row=row, column=0, sticky="w", padx=8)
        max_trades_var = tk.StringVar(value=str(cfg.max_trades_per_day))
        tk.Entry(content, textvariable=max_trades_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max open positions").grid(row=row, column=0, sticky="w", padx=8)
        max_pos_var = tk.StringVar(value=str(cfg.max_open_positions))
        tk.Entry(content, textvariable=max_pos_var, width=10).grid(row=row, column=1, sticky="w", padx=8)



        row += 1
        tk.Label(content, text="Min ATR").grid(row=row, column=0, sticky="w", padx=8)
        min_atr_var = tk.StringVar(value=str(cfg.min_atr))
        tk.Entry(content, textvariable=min_atr_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max ATR").grid(row=row, column=0, sticky="w", padx=8)
        max_atr_var = tk.StringVar(value=str(cfg.max_atr))
        tk.Entry(content, textvariable=max_atr_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="ATR period").grid(row=row, column=0, sticky="w", padx=8)
        atr_period_var = tk.StringVar(value=str(getattr(cfg, "atr_period", 14)))
        tk.Entry(content, textvariable=atr_period_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Cooldown (sec)").grid(row=row, column=0, sticky="w", padx=8)
        cooldown_var = tk.StringVar(value=str(getattr(cfg, "cooldown_sec", 30.0)))
        tk.Entry(content, textvariable=cooldown_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Cooldown after stopout (sec)").grid(row=row, column=0, sticky="w", padx=8)
        cooldown_stopout_var = tk.StringVar(value=str(getattr(cfg, "cooldown_after_stopout_sec", 180.0)))
        tk.Entry(content, textvariable=cooldown_stopout_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Cooldown ATR mult (sec/ATR)").grid(row=row, column=0, sticky="w", padx=8)
        cooldown_atr_mult_var = tk.StringVar(value=str(getattr(cfg, "cooldown_atr_mult", 0.0)))
        tk.Entry(content, textvariable=cooldown_atr_mult_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max stopouts/day").grid(row=row, column=0, sticky="w", padx=8)
        stopouts_var = tk.StringVar(value=str(cfg.max_stopouts_per_day))
        tk.Entry(content, textvariable=stopouts_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max consecutive stopouts").grid(row=row, column=0, sticky="w", padx=8)
        consec_var = tk.StringVar(value=str(cfg.max_consecutive_stopouts))
        tk.Entry(content, textvariable=consec_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max same trade-type streak (0=off)").grid(row=row, column=0, sticky="w", padx=8)
        max_same_type_var = tk.StringVar(value=str(getattr(cfg, "max_consecutive_same_trade_type", 0)))
        tk.Entry(content, textvariable=max_same_type_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max same direction qty (0=off)").grid(row=row, column=0, sticky="w", padx=8)
        max_same_dir_qty_var = tk.StringVar(value=str(getattr(cfg, "entry_max_same_direction_qty", 0)))
        tk.Entry(content, textvariable=max_same_dir_qty_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max Pyramiding Levels (0=off)").grid(row=row, column=0, sticky="w", padx=8)
        pyramid_var = tk.StringVar(value=str(getattr(cfg, "max_pyramid_levels", 0)))
        tk.Entry(content, textvariable=pyramid_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max daily loss (₹)").grid(row=row, column=0, sticky="w", padx=8)
        daily_loss_var = tk.StringVar(value=f"{cfg.max_daily_loss:.2f}")
        tk.Entry(content, textvariable=daily_loss_var, width=10).grid(row=row, column=1, sticky="w", padx=8)


        row += 1
        tk.Label(content, text="Risk scaling", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 0)
        )

        row += 1
        risk_scale_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "risk_scale_enabled", False)))
        ttk.Checkbutton(content, text="Enable risk scaling", variable=risk_scale_enabled_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Risk stopout factor / recovery wins").grid(row=row, column=0, sticky="w", padx=8)
        risk_stop_frame = tk.Frame(content)
        risk_stop_frame.grid(row=row, column=1, sticky="w", padx=8)
        risk_scale_stopout_var = tk.StringVar(value=str(getattr(cfg, "risk_scale_stopout_factor", 0.7)))
        tk.Entry(risk_stop_frame, textvariable=risk_scale_stopout_var, width=10).pack(side=tk.LEFT)
        risk_scale_recovery_var = tk.StringVar(value=str(getattr(cfg, "risk_scale_recovery_wins", 2)))
        tk.Entry(risk_stop_frame, textvariable=risk_scale_recovery_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Risk ATR high / factor").grid(row=row, column=0, sticky="w", padx=8)
        risk_atr_frame = tk.Frame(content)
        risk_atr_frame.grid(row=row, column=1, sticky="w", padx=8)
        risk_scale_atr_high_var = tk.StringVar(value=str(getattr(cfg, "risk_scale_atr_high", 0.0)))
        tk.Entry(risk_atr_frame, textvariable=risk_scale_atr_high_var, width=10).pack(side=tk.LEFT)
        risk_scale_atr_factor_var = tk.StringVar(value=str(getattr(cfg, "risk_scale_atr_high_factor", 0.7)))
        tk.Entry(risk_atr_frame, textvariable=risk_scale_atr_factor_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Risk qty min / max / step").grid(row=row, column=0, sticky="w", padx=8)
        risk_qty_frame = tk.Frame(content)
        risk_qty_frame.grid(row=row, column=1, sticky="w", padx=8)
        risk_scale_min_qty_var = tk.StringVar(value=str(getattr(cfg, "risk_scale_min_qty", 0)))
        tk.Entry(risk_qty_frame, textvariable=risk_scale_min_qty_var, width=8).pack(side=tk.LEFT)
        risk_scale_max_qty_var = tk.StringVar(value=str(getattr(cfg, "risk_scale_max_qty", 0)))
        tk.Entry(risk_qty_frame, textvariable=risk_scale_max_qty_var, width=8).pack(side=tk.LEFT, padx=(6, 0))
        risk_scale_step_qty_var = tk.StringVar(value=str(getattr(cfg, "risk_scale_step_qty", 0)))
        tk.Entry(risk_qty_frame, textvariable=risk_scale_step_qty_var, width=8).pack(side=tk.LEFT, padx=(6, 0))


        row += 1
        tk.Label(content, text="Max hold (min)").grid(row=row, column=0, sticky="w", padx=8)
        max_hold_var = tk.StringVar(value=str(cfg.max_hold_minutes))
        tk.Entry(content, textvariable=max_hold_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max hold (days)").grid(row=row, column=0, sticky="w", padx=8)
        max_hold_days_var = tk.StringVar(value=str(getattr(cfg, "max_hold_days", 0) or 0))
        tk.Entry(content, textvariable=max_hold_days_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Max trend strength").grid(row=row, column=0, sticky="w", padx=8)
        trend_strength_var = tk.StringVar(value=str(getattr(cfg, "max_trend_strength", 0.0015)))
        tk.Entry(content, textvariable=trend_strength_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="EMA Fast / Slow").grid(row=row, column=0, sticky="w", padx=8)
        ema_frame = tk.Frame(content)
        ema_frame.grid(row=row, column=1, sticky="w", padx=8)
        ema_fast_var = tk.StringVar(value=str(cfg.ema_fast))
        tk.Entry(ema_frame, textvariable=ema_fast_var, width=10).pack(side=tk.LEFT)
        ema_slow_var = tk.StringVar(value=str(cfg.ema_slow))
        tk.Entry(ema_frame, textvariable=ema_slow_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        premium_rsi_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_premium_rsi_filter", True)))
        ttk.Checkbutton(content, text="Enable premium RSI filter", variable=premium_rsi_enabled_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Premium RSI low").grid(row=row, column=0, sticky="w", padx=8)
        premium_rsi_low_var = tk.StringVar(value=str(getattr(cfg, "premium_rsi_low", 45.0)))
        tk.Entry(content, textvariable=premium_rsi_low_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Premium RSI high").grid(row=row, column=0, sticky="w", padx=8)
        premium_rsi_high_var = tk.StringVar(value=str(getattr(cfg, "premium_rsi_high", 55.0)))
        tk.Entry(content, textvariable=premium_rsi_high_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Entry guardrails", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 0)
        )

        row += 1
        tk.Label(content, text="Entry candle age / range / gap").grid(row=row, column=0, sticky="w", padx=8)
        entry_candle_frame = tk.Frame(content)
        entry_candle_frame.grid(row=row, column=1, sticky="w", padx=8)
        entry_candle_age_var = tk.StringVar(value=str(getattr(cfg, "entry_candle_max_age_sec", 120.0)))
        tk.Entry(entry_candle_frame, textvariable=entry_candle_age_var, width=6).pack(side=tk.LEFT)
        entry_candle_range_var = tk.StringVar(value=str(getattr(cfg, "entry_max_candle_range_atr_mult", 2.5)))
        tk.Entry(entry_candle_frame, textvariable=entry_candle_range_var, width=6).pack(side=tk.LEFT, padx=(6, 0))
        entry_gap_var = tk.StringVar(value=str(getattr(cfg, "entry_gap_atr_mult", 1.5)))
        tk.Entry(entry_candle_frame, textvariable=entry_gap_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Entry option premium (min/max)").grid(row=row, column=0, sticky="w", padx=8)
        entry_prem_frame = tk.Frame(content)
        entry_prem_frame.grid(row=row, column=1, sticky="w", padx=8)
        entry_min_prem_var = tk.StringVar(value=str(getattr(cfg, "entry_min_option_premium", 0.0)))
        tk.Entry(entry_prem_frame, textvariable=entry_min_prem_var, width=8).pack(side=tk.LEFT)
        entry_max_prem_var = tk.StringVar(value=str(getattr(cfg, "entry_max_option_premium", 0.0)))
        tk.Entry(entry_prem_frame, textvariable=entry_max_prem_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Entry total premium (min/max)").grid(row=row, column=0, sticky="w", padx=8)
        entry_total_frame = tk.Frame(content)
        entry_total_frame.grid(row=row, column=1, sticky="w", padx=8)
        entry_min_total_prem_var = tk.StringVar(value=str(getattr(cfg, "entry_min_total_premium", 0.0)))
        tk.Entry(entry_total_frame, textvariable=entry_min_total_prem_var, width=8).pack(side=tk.LEFT)
        entry_max_total_prem_var = tk.StringVar(value=str(getattr(cfg, "entry_max_total_premium", 0.0)))
        tk.Entry(entry_total_frame, textvariable=entry_max_total_prem_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        entry_req_ba_var = tk.BooleanVar(value=bool(getattr(cfg, "entry_require_bid_ask", False)))
        ttk.Checkbutton(content, text="Entry require bid/ask", variable=entry_req_ba_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Entry max spread % / abs").grid(row=row, column=0, sticky="w", padx=8)
        entry_spread_frame = tk.Frame(content)
        entry_spread_frame.grid(row=row, column=1, sticky="w", padx=8)
        entry_spread_pct_var = tk.StringVar(value=str(getattr(cfg, "entry_max_bid_ask_spread_pct", 0.0)))
        tk.Entry(entry_spread_frame, textvariable=entry_spread_pct_var, width=8).pack(side=tk.LEFT)
        entry_spread_abs_var = tk.StringVar(value=str(getattr(cfg, "entry_max_bid_ask_spread_abs", 0.0)))
        tk.Entry(entry_spread_frame, textvariable=entry_spread_abs_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="IV expand / contract % (short/long)").grid(row=row, column=0, sticky="w", padx=8)
        iv_frame = tk.Frame(content)
        iv_frame.grid(row=row, column=1, sticky="w", padx=8)
        iv_expand_var = tk.StringVar(value=str(getattr(cfg, "iv_expand_threshold_pct", 3.0)))
        tk.Entry(iv_frame, textvariable=iv_expand_var, width=8).pack(side=tk.LEFT)
        iv_contract_var = tk.StringVar(value=str(getattr(cfg, "iv_contract_threshold_pct", 3.0)))
        tk.Entry(iv_frame, textvariable=iv_contract_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        weekly_only_var = tk.BooleanVar(value=bool(getattr(cfg, "nifty_weekly_only", True)))
        ttk.Checkbutton(content, text="NIFTY weekly only", variable=weekly_only_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        bypass_weekly_var = tk.BooleanVar(value=bool(getattr(cfg, "bypass_weekly_filter", False)))
        ttk.Checkbutton(content, text="Bypass weekly filter (DEBUG)", variable=bypass_weekly_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Target expiry (DD-MM-YYYY)").grid(row=row, column=0, sticky="w", padx=8)
        target_expiry_var = tk.StringVar(value=str(getattr(cfg, "target_expiry", "")))
        expiry_combo = ttk.Combobox(content, textvariable=target_expiry_var, width=14)
        expiry_combo.grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="ScripMaster CSV (options)").grid(row=row, column=0, sticky="w", padx=8)
        scripmaster_path_var = tk.StringVar(
            value=str(
                (
                    getattr(self, "scripmaster_path_var", tk.StringVar()).get()
                    or os.getenv("MSTOCK_SCRIPMASTER_PATH", "")
                )
            )
        )
        sm_row = tk.Frame(content)
        sm_row.grid(row=row, column=1, sticky="we", padx=8)
        tk.Entry(sm_row, textvariable=scripmaster_path_var, width=38).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _browse_sm_dialog() -> None:
            p = filedialog.askopenfilename(
                title="Select ScripMaster CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*")],
            )
            if p:
                scripmaster_path_var.set(p)

        tk.Button(sm_row, text="Browse...", command=_browse_sm_dialog).pack(side=tk.LEFT, padx=(6, 0))

        _expiry_refresh_after_id: str | None = None
        _expiry_refresh_seq = 0
        _expiry_cache: dict[tuple[str, str], list[str]] = {}

        def _apply_settings_expiry_values(values: list[str], seq: int) -> None:
            nonlocal _expiry_refresh_seq
            if seq != _expiry_refresh_seq:
                return
            try:
                expiry_combo["values"] = values
            except Exception:
                pass

        def _refresh_settings_expiries_now() -> None:
            nonlocal _expiry_refresh_seq
            path = (
                str(scripmaster_path_var.get() or "").strip()
                or getattr(self, "scripmaster_path_var", tk.StringVar()).get()
                or os.getenv("MSTOCK_SCRIPMASTER_PATH", "")
            )
            root = str(underlying_var.get() or "").strip()
            if not path or not Path(path).exists() or not root:
                try:
                    expiry_combo["values"] = []
                except Exception:
                    pass
                return

            cache_key = (path, root.upper())
            cached = _expiry_cache.get(cache_key)
            if cached is not None:
                _apply_settings_expiry_values(cached, _expiry_refresh_seq)
                return

            _expiry_refresh_seq += 1
            seq = _expiry_refresh_seq

            def _worker() -> None:
                values: list[str] = []
                try:
                    from scripmaster import ScripMaster

                    sm = ScripMaster(path)
                    exps = sm.get_available_expiries(root)
                    values = [d.strftime("%d-%m-%Y") for d in exps] if exps else []
                except Exception:
                    values = []
                _expiry_cache[cache_key] = values
                try:
                    self.after(0, lambda: _apply_settings_expiry_values(values, seq))
                except Exception:
                    pass

            threading.Thread(target=_worker, daemon=True).start()

        def _refresh_settings_expiries(*_args: object) -> None:
            nonlocal _expiry_refresh_after_id
            try:
                if _expiry_refresh_after_id is not None:
                    self.after_cancel(_expiry_refresh_after_id)
            except Exception:
                pass
            _expiry_refresh_after_id = self.after(180, _refresh_settings_expiries_now)

        # Refresh once on load
        _refresh_settings_expiries()
        # And when underlying changes
        underlying_var.trace_add("write", _refresh_settings_expiries)
        # And when CSV path changes
        try:
            scripmaster_path_var.trace_add("write", _refresh_settings_expiries)
        except Exception:
            pass


        row += 1
        tk.Label(content, text="Auto range trend x").grid(row=row, column=0, sticky="w", padx=8)
        auto_trend_mult_var = tk.StringVar(value=str(getattr(cfg, "auto_range_trend_mult", 2.0)))
        tk.Entry(content, textvariable=auto_trend_mult_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Auto RSI B/S/slop").grid(row=row, column=0, sticky="w", padx=8)
        auto_rsi_frame = tk.Frame(content)
        auto_rsi_frame.grid(row=row, column=1, sticky="w", padx=8)
        auto_rsi_buy_var = tk.StringVar(value=str(getattr(cfg, "auto_dir_rsi_buy", 52.0)))
        tk.Entry(auto_rsi_frame, textvariable=auto_rsi_buy_var, width=6).pack(side=tk.LEFT)
        auto_rsi_sell_var = tk.StringVar(value=str(getattr(cfg, "auto_dir_rsi_sell", 48.0)))
        tk.Entry(auto_rsi_frame, textvariable=auto_rsi_sell_var, width=6).pack(side=tk.LEFT, padx=(4, 0))
        auto_rsi_slop_var = tk.StringVar(value=str(getattr(cfg, "auto_dir_rsi_slop", 1.0)))
        tk.Entry(auto_rsi_frame, textvariable=auto_rsi_slop_var, width=6).pack(side=tk.LEFT, padx=(4, 0))

        row += 1
        tk.Label(content, text="MTM stop / target %").grid(row=row, column=0, sticky="w", padx=8)
        mtm_frame = tk.Frame(content)
        mtm_frame.grid(row=row, column=1, sticky="w", padx=8)
        mtm_stop_var = tk.StringVar(value=str(getattr(cfg, "premium_mtm_stop_pct", 0.30)))
        tk.Entry(mtm_frame, textvariable=mtm_stop_var, width=10).pack(side=tk.LEFT)
        mtm_tgt_var = tk.StringVar(value=str(getattr(cfg, "premium_mtm_target_pct", 0.18)))
        tk.Entry(mtm_frame, textvariable=mtm_tgt_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="MTM Trail (Start/Stop %)").grid(row=row, column=0, sticky="w", padx=8)
        mtm_trail_frame = tk.Frame(content)
        mtm_trail_frame.grid(row=row, column=1, sticky="w", padx=8)
        mtm_trail_start_var = tk.StringVar(value=str(getattr(cfg, "premium_mtm_trail_start_pct", 0.05)))
        tk.Entry(mtm_trail_frame, textvariable=mtm_trail_start_var, width=10).pack(side=tk.LEFT)
        mtm_trail_stop_var = tk.StringVar(value=str(getattr(cfg, "premium_mtm_trail_stop_pct", 0.05)))
        tk.Entry(mtm_trail_frame, textvariable=mtm_trail_stop_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="GPT leg MTM manage / refresh / min age sec").grid(row=row, column=0, sticky="w", padx=8)
        gpt_leg_frame = tk.Frame(content)
        gpt_leg_frame.grid(row=row, column=1, sticky="w", padx=8)
        gpt_leg_manage_var = tk.BooleanVar(value=bool(getattr(cfg, "gpt_leg_manage", False)))
        ttk.Checkbutton(gpt_leg_frame, text="Enable", variable=gpt_leg_manage_var).pack(side=tk.LEFT)
        gpt_leg_refresh_var = tk.StringVar(value=str(getattr(cfg, "gpt_leg_manage_refresh_sec", 45.0)))
        tk.Entry(gpt_leg_frame, textvariable=gpt_leg_refresh_var, width=10).pack(side=tk.LEFT, padx=(8, 0))
        gpt_leg_min_age_var = tk.StringVar(value=str(getattr(cfg, "gpt_leg_manage_min_age_sec", 90.0)))
        tk.Entry(gpt_leg_frame, textvariable=gpt_leg_min_age_var, width=10).pack(side=tk.LEFT, padx=(8, 0))

        row += 1
        tk.Label(content, text="GPT leg grace sec / min stop% / min target%").grid(row=row, column=0, sticky="w", padx=8)
        gpt_leg_guard_frame = tk.Frame(content)
        gpt_leg_guard_frame.grid(row=row, column=1, sticky="w", padx=8)
        gpt_leg_grace_var = tk.StringVar(value=str(getattr(cfg, "gpt_leg_manage_post_update_grace_sec", 20.0)))
        tk.Entry(gpt_leg_guard_frame, textvariable=gpt_leg_grace_var, width=10).pack(side=tk.LEFT)
        gpt_leg_stop_dist_var = tk.StringVar(value=str(getattr(cfg, "gpt_leg_manage_min_stop_distance_pct", 0.08)))
        tk.Entry(gpt_leg_guard_frame, textvariable=gpt_leg_stop_dist_var, width=10).pack(side=tk.LEFT, padx=(8, 0))
        gpt_leg_target_dist_var = tk.StringVar(value=str(getattr(cfg, "gpt_leg_manage_min_target_distance_pct", 0.10)))
        tk.Entry(gpt_leg_guard_frame, textvariable=gpt_leg_target_dist_var, width=10).pack(side=tk.LEFT, padx=(8, 0))


        row += 1
        exit_short_var = tk.BooleanVar(value=bool(cfg.premium_exit_on_short_strike_touch))
        ttk.Checkbutton(content, text="Exit on short strike touch", variable=exit_short_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        exit_wing_var = tk.BooleanVar(value=bool(cfg.premium_exit_on_wing_touch))
        ttk.Checkbutton(content, text="Exit on wing touch", variable=exit_wing_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Hard exit (HH:MM)").grid(row=row, column=0, sticky="w", padx=8)
        force_exit_var = tk.StringVar(value=str(getattr(cfg, "premium_force_exit_hhmm", "14:20")))
        tk.Entry(content, textvariable=force_exit_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Entry cutoff (HH:MM)").grid(row=row, column=0, sticky="w", padx=8)
        cutoff_var = tk.StringVar(value=str(getattr(cfg, "premium_entry_cutoff_hhmm", "")))
        tk.Entry(content, textvariable=cutoff_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="VWAP max dev (%)").grid(row=row, column=0, sticky="w", padx=8)
        vwap_dev_var = tk.StringVar(value=str(getattr(cfg, "vwap_max_dev_pct", 0.15)))
        tk.Entry(content, textvariable=vwap_dev_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        vwap_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_vwap_filter", True)))
        ttk.Checkbutton(content, text="Enable VWAP filter", variable=vwap_enabled_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Opening max move (%)").grid(row=row, column=0, sticky="w", padx=8)
        open_move_var = tk.StringVar(value=str(getattr(cfg, "opening_max_move_pct", 0.40)))
        tk.Entry(content, textvariable=open_move_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Opening window (min)").grid(row=row, column=0, sticky="w", padx=8)
        open_min_var = tk.StringVar(value=str(getattr(cfg, "opening_filter_minutes", 45)))
        tk.Entry(content, textvariable=open_min_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        opening_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_opening_filter", True)))
        ttk.Checkbutton(content, text="Enable opening filter", variable=opening_enabled_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        trend_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_trend_filter", True)))
        ttk.Checkbutton(content, text="Enable trend filter", variable=trend_enabled_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        supertrend_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_supertrend_filter", True)))
        ttk.Checkbutton(content, text="Enable Supertrend filter", variable=supertrend_enabled_var).grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Supertrend mode").grid(row=row, column=0, sticky="w", padx=8)
        supertrend_mode_var = tk.StringVar(
            value=str(getattr(cfg, "supertrend_mode", "counter") or "counter").strip().lower()
        )
        tk.OptionMenu(content, supertrend_mode_var, "trend", "counter").grid(
            row=row, column=1, sticky="w", padx=8
        )

        row += 1
        tk.Label(content, text="Dir BE / Trail x ATR (BE can be negative)").grid(row=row, column=0, sticky="w", padx=8)
        dir_trail_frame = tk.Frame(content)
        dir_trail_frame.grid(row=row, column=1, sticky="w", padx=8)
        dir_be_var = tk.StringVar(value=str(cfg.dir_breakeven_atr_mult))
        tk.Entry(dir_trail_frame, textvariable=dir_be_var, width=10).pack(side=tk.LEFT)
        dir_trail_atr_var = tk.StringVar(value=str(cfg.dir_trail_atr_mult))
        tk.Entry(dir_trail_frame, textvariable=dir_trail_atr_var, width=10).pack(side=tk.LEFT, padx=(6, 0))



        row += 1
        tk.Label(content, text="Dir Prem Trail / Require Spot").grid(row=row, column=0, sticky="w", padx=8)
        dir_opt_frame = tk.Frame(content)
        dir_opt_frame.grid(row=row, column=1, sticky="w", padx=8)
        dir_prem_trail_var = tk.StringVar(value=str(cfg.dir_premium_trail_pct))
        tk.Entry(dir_opt_frame, textvariable=dir_prem_trail_var, width=10).pack(side=tk.LEFT)
        req_spot_var = tk.BooleanVar(value=bool(cfg.require_spot_ltp_for_entry))
        ttk.Checkbutton(dir_opt_frame, text="Spot LTP Req", variable=req_spot_var).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Dir Quality (Min/Δ/Mom xATR)").grid(row=row, column=0, sticky="w", padx=8)
        dir_q_frame = tk.Frame(content)
        dir_q_frame.grid(row=row, column=1, sticky="w", padx=8)
        dir_min_confirm_var = tk.StringVar(value=str(getattr(cfg, "dir_min_confirmations", 4)))
        tk.Entry(dir_q_frame, textvariable=dir_min_confirm_var, width=4).pack(side=tk.LEFT)
        dir_min_diff_var = tk.StringVar(value=str(getattr(cfg, "dir_min_score_diff", 1)))
        tk.Entry(dir_q_frame, textvariable=dir_min_diff_var, width=4).pack(side=tk.LEFT, padx=(6, 0))
        dir_mom_atr_mult_var = tk.StringVar(value=str(getattr(cfg, "dir_momentum_atr_mult", 0.10)))
        tk.Entry(dir_q_frame, textvariable=dir_mom_atr_mult_var, width=6).pack(side=tk.LEFT, padx=(6, 0))
        dir_allow_tie_var = tk.BooleanVar(value=bool(getattr(cfg, "dir_allow_tie_break_entries", False)))
        ttk.Checkbutton(dir_q_frame, text="Tie", variable=dir_allow_tie_var).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Dir EMA slope xATR").grid(row=row, column=0, sticky="w", padx=8)
        dir_ema_slope_var = tk.StringVar(value=str(getattr(cfg, "dir_ema_slope_atr_mult", 0.05)))
        tk.Entry(content, textvariable=dir_ema_slope_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        tk.Label(content, text="Dir Partial xATR / %Qty").grid(row=row, column=0, sticky="w", padx=8)
        dir_partial_frame = tk.Frame(content)
        dir_partial_frame.grid(row=row, column=1, sticky="w", padx=8)
        dir_partial_tgt_var = tk.StringVar(value=str(cfg.dir_partial_target_mult))
        tk.Entry(dir_partial_frame, textvariable=dir_partial_tgt_var, width=10).pack(side=tk.LEFT)
        dir_partial_qty_var = tk.StringVar(value=str(cfg.dir_partial_qty_pct))
        tk.Entry(dir_partial_frame, textvariable=dir_partial_qty_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="Dir post-partial BE/Trail xATR").grid(row=row, column=0, sticky="w", padx=8)
        dir_post_frame = tk.Frame(content)
        dir_post_frame.grid(row=row, column=1, sticky="w", padx=8)
        dir_post_be_var = tk.StringVar(value=str(getattr(cfg, "dir_post_partial_be_atr_mult", 0.0)))
        tk.Entry(dir_post_frame, textvariable=dir_post_be_var, width=10).pack(side=tk.LEFT)
        dir_post_trail_var = tk.StringVar(value=str(getattr(cfg, "dir_post_partial_trail_atr_mult", 0.0)))
        tk.Entry(dir_post_frame, textvariable=dir_post_trail_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="ATR spike x / debug log").grid(row=row, column=0, sticky="w", padx=8)
        dbg_frame = tk.Frame(content)
        dbg_frame.grid(row=row, column=1, sticky="w", padx=8)
        spike_var = tk.StringVar(value=str(getattr(cfg, "atr_spike_mult", 2.0)))
        tk.Entry(dbg_frame, textvariable=spike_var, width=10).pack(side=tk.LEFT)
        no_signal_var = tk.BooleanVar(value=bool(getattr(cfg, "debug_log_no_signal", False)))
        ttk.Checkbutton(dbg_frame, text="No signal log", variable=no_signal_var).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        tk.Label(content, text="ADX Filter (Enable/Min)").grid(row=row, column=0, sticky="w", padx=8)
        adx_frame = tk.Frame(content)
        adx_frame.grid(row=row, column=1, sticky="w", padx=8)
        adx_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_adx_filter", True)))
        ttk.Checkbutton(adx_frame, text="On", variable=adx_enabled_var).pack(side=tk.LEFT)
        adx_min_var = tk.StringVar(value=str(getattr(cfg, "adx_min_strength", 25.0)))
        tk.Entry(adx_frame, textvariable=adx_min_var, width=6).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(adx_frame, text="Per").pack(side=tk.LEFT, padx=(4, 0))
        adx_period_var = tk.StringVar(value=str(getattr(cfg, "adx_period", 14)))
        tk.Entry(adx_frame, textvariable=adx_period_var, width=4).pack(side=tk.LEFT)

        row += 1
        tk.Label(content, text="Volume Filter (Enable/SMA)").grid(row=row, column=0, sticky="w", padx=8)
        vol_frame = tk.Frame(content)
        vol_frame.grid(row=row, column=1, sticky="w", padx=8)
        vol_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_volume_filter", True)))
        ttk.Checkbutton(vol_frame, text="On", variable=vol_enabled_var).pack(side=tk.LEFT)
        tk.Label(vol_frame, text="SMA").pack(side=tk.LEFT, padx=(6, 0))
        vol_sma_var = tk.StringVar(value=str(getattr(cfg, "volume_sma_period", 20)))
        tk.Entry(vol_frame, textvariable=vol_sma_var, width=4).pack(side=tk.LEFT)

        row += 1
        tk.Label(content, text="Limit Order Buffer (%)").grid(row=row, column=0, sticky="w", padx=8)
        limit_buf_var = tk.StringVar(value=str(getattr(cfg, "limit_price_buffer_pct", 0.05)))
        tk.Entry(content, textvariable=limit_buf_var, width=10).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        enh_row1 = tk.Frame(content)
        enh_row1.grid(row=row, column=1, sticky="w", padx=8)
        mtf_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_mtf_confirmation", False)))
        ttk.Checkbutton(enh_row1, text="MTF Confirmation", variable=mtf_enabled_var).pack(side=tk.LEFT)
        roc_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_roc_filter", False)))
        ttk.Checkbutton(enh_row1, text="ROC Filter", variable=roc_enabled_var).pack(side=tk.LEFT)
        chop_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_choppiness_filter", False)))
        ttk.Checkbutton(enh_row1, text="Choppiness Filter", variable=chop_enabled_var).pack(side=tk.LEFT)

        row += 1
        enh_row2 = tk.Frame(content)
        enh_row2.grid(row=row, column=1, sticky="w", padx=8)
        chan_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_chandelier_exit", False)))
        ttk.Checkbutton(enh_row2, text="Chandelier Exit", variable=chan_enabled_var).pack(side=tk.LEFT)
        pivot_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_pivot_targets", False)))
        ttk.Checkbutton(enh_row2, text="Pivot Targets", variable=pivot_enabled_var).pack(side=tk.LEFT)
        limit_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_limit_orders", False)))
        ttk.Checkbutton(enh_row2, text="Use Limit Orders", variable=limit_enabled_var).pack(side=tk.LEFT)

        def apply_changes() -> None:
            try:
                to_persist: dict[str, str] = {}
                to_unset: list[str] = []

                preset_name = str(preset_var.get() or "").strip().lower()
                if preset_name == "aggressive":
                    os.environ["MSTOCK_PRESET"] = "aggressive"
                    to_persist["MSTOCK_PRESET"] = "aggressive"
                    try:
                        if hasattr(self, "preset_var"):
                            self.preset_var.set("Aggressive")
                    except Exception:
                        pass
                elif preset_name == "conservative":
                    os.environ["MSTOCK_PRESET"] = "conservative"
                    to_persist["MSTOCK_PRESET"] = "conservative"
                    try:
                        if hasattr(self, "preset_var"):
                            self.preset_var.set("Conservative")
                    except Exception:
                        pass
                else:
                    os.environ.pop("MSTOCK_PRESET", None)
                    to_unset.append("MSTOCK_PRESET")

                strat = str(strategy_var.get() or "").strip().lower()
                if strat:
                    os.environ["MSTOCK_STRATEGY"] = strat
                    to_persist["MSTOCK_STRATEGY"] = strat
                    try:
                        self.strategy_var.set(strat)
                    except Exception:
                        pass

                tf = str(timeframe_var.get() or "").strip().lower()
                if tf:
                    os.environ["MSTOCK_TIMEFRAME"] = tf
                    to_persist["MSTOCK_TIMEFRAME"] = tf
                    try:
                        if hasattr(self, "timeframe_var"):
                            self.timeframe_var.set(tf)
                    except Exception:
                        pass

                dscope = str(delta_scope_var.get() or "").strip().lower() or "strategy_only"
                os.environ["MSTOCK_DELTA_HEDGE_SCOPE"] = dscope
                to_persist["MSTOCK_DELTA_HEDGE_SCOPE"] = dscope
                try:
                    if hasattr(self, "delta_hedge_scope_var"):
                        self.delta_hedge_scope_var.set(dscope)
                except Exception:
                    pass

                hn = str(hedge_nifty_var.get() or "").strip()
                hb = str(hedge_bank_var.get() or "").strip()

                # Guardrail: a spot index symbol like "NIFTY" is not tradable.
                # Delta-hedging places orders; in live mode this will fail if the hedge
                # symbol is not tradable (use an ETF or a future).
                try:
                    live_mode = bool(getattr(cfg, "enable_live_trading", False))
                except Exception:
                    live_mode = False
                def _looks_like_spot_index(sym: str) -> bool:
                    s = str(sym or "").strip().upper()
                    if not s:
                        return False
                    s = s.split(":", 1)[-1].strip()
                    return s in {"NIFTY", "NIFTY50", "NIFTY 50", "BANKNIFTY"}

                if live_mode and (_looks_like_spot_index(hn) or _looks_like_spot_index(hb)):
                    messagebox.showwarning(
                        "Hedge symbol not tradable",
                        "Delta hedging places orders. A spot index symbol like 'NIFTY' is not tradable.\n\n"
                        "Use a tradable hedge instrument, e.g.:\n"
                        "- NSE:NIFTYBEES (ETF)\n"
                        "- NFO:NIFTY (auto-picks nearest FUT)\n"
                        "- NFO:<NIFTY...FUT> (explicit index future tradingsymbol)\n\n"
                        "You can still save settings now; just update the hedge symbol before running live.",
                    )
                if hn:
                    os.environ["MSTOCK_DELTA_HEDGE_SYMBOL_NIFTY"] = hn
                    to_persist["MSTOCK_DELTA_HEDGE_SYMBOL_NIFTY"] = hn
                else:
                    os.environ.pop("MSTOCK_DELTA_HEDGE_SYMBOL_NIFTY", None)
                    to_unset.append("MSTOCK_DELTA_HEDGE_SYMBOL_NIFTY")

                if hb:
                    os.environ["MSTOCK_DELTA_HEDGE_SYMBOL_BANKNIFTY"] = hb
                    to_persist["MSTOCK_DELTA_HEDGE_SYMBOL_BANKNIFTY"] = hb
                else:
                    os.environ.pop("MSTOCK_DELTA_HEDGE_SYMBOL_BANKNIFTY", None)
                    to_unset.append("MSTOCK_DELTA_HEDGE_SYMBOL_BANKNIFTY")

                if dh_base_tol_var.get().strip():
                    v = str(float(dh_base_tol_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_TOLERANCE"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_TOLERANCE"] = v
                if dh_entry_tol_var.get().strip():
                    v = str(float(dh_entry_tol_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_ENTRY_TOLERANCE"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_ENTRY_TOLERANCE"] = v
                if dh_exit_tol_var.get().strip():
                    v = str(float(dh_exit_tol_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_EXIT_TOLERANCE"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_EXIT_TOLERANCE"] = v
                if dh_adjust_factor_var.get().strip():
                    v = str(float(dh_adjust_factor_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_ADJUSTMENT_FACTOR"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_ADJUSTMENT_FACTOR"] = v
                if dh_max_adjust_var.get().strip():
                    v = str(float(dh_max_adjust_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_MAX_ADJUST_ABS_QTY"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_MAX_ADJUST_ABS_QTY"] = v
                if dh_min_spot_move_var.get().strip():
                    v = str(float(dh_min_spot_move_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE"] = v
                if dh_min_spot_atr_var.get().strip():
                    v = str(float(dh_min_spot_atr_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE_ATR_MULT"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_MIN_SPOT_MOVE_ATR_MULT"] = v
                if dh_max_orders_var.get().strip():
                    v = str(int(dh_max_orders_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_MAX_ORDERS_PER_DAY"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_MAX_ORDERS_PER_DAY"] = v
                os.environ["MSTOCK_DELTA_HEDGE_MARKET_HOURS_ONLY"] = "true" if dh_hours_var.get() else "false"
                to_persist["MSTOCK_DELTA_HEDGE_MARKET_HOURS_ONLY"] = os.environ["MSTOCK_DELTA_HEDGE_MARKET_HOURS_ONLY"]
                os.environ["MSTOCK_DELTA_HEDGE_REQUIRE_BID_ASK"] = "true" if dh_req_ba_var.get() else "false"
                to_persist["MSTOCK_DELTA_HEDGE_REQUIRE_BID_ASK"] = os.environ["MSTOCK_DELTA_HEDGE_REQUIRE_BID_ASK"]

                os.environ["MSTOCK_DELTA_HEDGE_INCLUDE_POSITIONS"] = "true" if dh_inc_pos_var.get() else "false"
                to_persist["MSTOCK_DELTA_HEDGE_INCLUDE_POSITIONS"] = os.environ["MSTOCK_DELTA_HEDGE_INCLUDE_POSITIONS"]
                os.environ["MSTOCK_DELTA_HEDGE_INCLUDE_HOLDINGS"] = "true" if dh_inc_hold_var.get() else "false"
                to_persist["MSTOCK_DELTA_HEDGE_INCLUDE_HOLDINGS"] = os.environ["MSTOCK_DELTA_HEDGE_INCLUDE_HOLDINGS"]

                os.environ["MSTOCK_DELTA_HEDGE_INCLUDE_EQUITY_BETA"] = "true" if dh_beta_var.get() else "false"
                to_persist["MSTOCK_DELTA_HEDGE_INCLUDE_EQUITY_BETA"] = os.environ["MSTOCK_DELTA_HEDGE_INCLUDE_EQUITY_BETA"]
                os.environ["MSTOCK_DELTA_HEDGE_PORTFOLIO_HEDGE_ENABLE"] = "true" if dh_portfolio_var.get() else "false"
                to_persist["MSTOCK_DELTA_HEDGE_PORTFOLIO_HEDGE_ENABLE"] = os.environ[
                    "MSTOCK_DELTA_HEDGE_PORTFOLIO_HEDGE_ENABLE"
                ]
                if dh_beta_lookback_var.get().strip():
                    v = str(int(float(dh_beta_lookback_var.get().strip())))
                    os.environ["MSTOCK_DELTA_HEDGE_BETA_LOOKBACK_DAYS"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_BETA_LOOKBACK_DAYS"] = v
                if dh_beta_maxsym_var.get().strip():
                    v = str(int(float(dh_beta_maxsym_var.get().strip())))
                    os.environ["MSTOCK_DELTA_HEDGE_BETA_MAX_SYMBOLS"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_BETA_MAX_SYMBOLS"] = v
                if dh_beta_refresh_var.get().strip():
                    v = str(float(dh_beta_refresh_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_BETA_REFRESH_SEC"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_BETA_REFRESH_SEC"] = v
                if dh_max_spread_pct_var.get().strip():
                    v = str(float(dh_max_spread_pct_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_MAX_SPREAD_PCT"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_MAX_SPREAD_PCT"] = v
                if dh_max_spread_abs_var.get().strip():
                    v = str(float(dh_max_spread_abs_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_MAX_SPREAD_ABS"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_MAX_SPREAD_ABS"] = v

                os.environ["MSTOCK_EQUITY_TRADE_ENABLE"] = "true" if eq_enable_var.get() else "false"
                to_persist["MSTOCK_EQUITY_TRADE_ENABLE"] = os.environ["MSTOCK_EQUITY_TRADE_ENABLE"]
                os.environ["MSTOCK_EQUITY_TRADE_ALLOW_SHORT"] = "true" if eq_short_var.get() else "false"
                to_persist["MSTOCK_EQUITY_TRADE_ALLOW_SHORT"] = os.environ["MSTOCK_EQUITY_TRADE_ALLOW_SHORT"]

                if eq_engine_var.get().strip():
                    v = str(eq_engine_var.get().strip().upper())
                    os.environ["MSTOCK_EQUITY_TRADE_ENGINE"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_ENGINE"] = v
                if eq_horizon_var.get().strip():
                    v = str(eq_horizon_var.get().strip().upper())
                    os.environ["MSTOCK_EQUITY_TRADE_HORIZON"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_HORIZON"] = v
                os.environ["MSTOCK_EQUITY_TRADE_CHURN_ENABLE"] = "true" if eq_churn_var.get() else "false"
                to_persist["MSTOCK_EQUITY_TRADE_CHURN_ENABLE"] = os.environ["MSTOCK_EQUITY_TRADE_CHURN_ENABLE"]
                os.environ["MSTOCK_EQUITY_TRADE_GPT_MANAGE_DAILY"] = "true" if eq_gpt_daily_var.get() else "false"
                to_persist["MSTOCK_EQUITY_TRADE_GPT_MANAGE_DAILY"] = os.environ["MSTOCK_EQUITY_TRADE_GPT_MANAGE_DAILY"]
                os.environ["MSTOCK_EQUITY_TRADE_GPT_WATCHLIST_ENABLE"] = "true" if eq_gpt_watchlist_var.get() else "false"
                to_persist["MSTOCK_EQUITY_TRADE_GPT_WATCHLIST_ENABLE"] = os.environ["MSTOCK_EQUITY_TRADE_GPT_WATCHLIST_ENABLE"]
                v = str(eq_symbols_var.get() or "")
                os.environ["MSTOCK_EQUITY_TRADE_SYMBOLS"] = v
                to_persist["MSTOCK_EQUITY_TRADE_SYMBOLS"] = v
                if eq_watch_refresh_var.get().strip():
                    v = str(float(eq_watch_refresh_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_WATCHLIST_REFRESH_SEC"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_WATCHLIST_REFRESH_SEC"] = v
                if eq_candidate_limit_var.get().strip():
                    v = str(int(float(eq_candidate_limit_var.get().strip())))
                    os.environ["MSTOCK_EQUITY_TRADE_CANDIDATE_LIMIT"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_CANDIDATE_LIMIT"] = v
                if eq_lt_target_var.get().strip():
                    v = str(float(eq_lt_target_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_LONGTERM_TARGET_PCT"] = v
                    to_persist["MSTOCK_EQUITY_LONGTERM_TARGET_PCT"] = v
                if eq_lt_days_var.get().strip():
                    v = str(int(float(eq_lt_days_var.get().strip())))
                    os.environ["MSTOCK_EQUITY_LONGTERM_HORIZON_DAYS"] = v
                    to_persist["MSTOCK_EQUITY_LONGTERM_HORIZON_DAYS"] = v
                if eq_prod_var.get().strip():
                    v = str(eq_prod_var.get().strip().upper())
                    os.environ["MSTOCK_EQUITY_TRADE_PRODUCT_TYPE"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_PRODUCT_TYPE"] = v
                if eq_prod_long_var.get().strip():
                    v = str(eq_prod_long_var.get().strip().upper())
                    os.environ["MSTOCK_EQUITY_TRADE_LONGTERM_PRODUCT_TYPE"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_LONGTERM_PRODUCT_TYPE"] = v
                if eq_tf_var.get().strip():
                    v = str(eq_tf_var.get().strip().lower())
                    os.environ["MSTOCK_EQUITY_TRADE_TIMEFRAME"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_TIMEFRAME"] = v
                if eq_maxsym_var.get().strip():
                    v = str(int(float(eq_maxsym_var.get().strip())))
                    os.environ["MSTOCK_EQUITY_TRADE_MAX_SYMBOLS"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_MAX_SYMBOLS"] = v
                if eq_reb_var.get().strip():
                    v = str(float(eq_reb_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_REBALANCE_SEC"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_REBALANCE_SEC"] = v
                if eq_atr_p_var.get().strip():
                    v = str(int(float(eq_atr_p_var.get().strip())))
                    os.environ["MSTOCK_EQUITY_TRADE_ATR_PERIOD"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_ATR_PERIOD"] = v
                if eq_sl_var.get().strip():
                    v = str(float(eq_sl_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_SL_ATR_MULT"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_SL_ATR_MULT"] = v
                if eq_tp_var.get().strip():
                    v = str(float(eq_tp_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_TP_ATR_MULT"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_TP_ATR_MULT"] = v
                if eq_risk_var.get().strip():
                    v = str(float(eq_risk_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_RISK_RUPEES"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_RISK_RUPEES"] = v
                if eq_maxnot_var.get().strip():
                    v = str(float(eq_maxnot_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_MAX_NOTIONAL"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_MAX_NOTIONAL"] = v
                if eq_capital_var.get().strip():
                    v = str(float(eq_capital_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_CAPITAL_RUPEES"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_CAPITAL_RUPEES"] = v
                if eq_cd_var.get().strip():
                    v = str(float(eq_cd_var.get().strip()))
                    os.environ["MSTOCK_EQUITY_TRADE_COOLDOWN_SEC"] = v
                    to_persist["MSTOCK_EQUITY_TRADE_COOLDOWN_SEC"] = v
                if symbol_var.get().strip():
                    v = symbol_var.get().strip().upper()
                    os.environ["MSTOCK_SYMBOL"] = v
                    to_persist["MSTOCK_SYMBOL"] = v
                if underlying_var.get().strip():
                    v = underlying_var.get().strip().upper()
                    os.environ["MSTOCK_UNDERLYING"] = v
                    to_persist["MSTOCK_UNDERLYING"] = v
                if lot_var.get().strip():
                    v = str(int(lot_var.get().strip()))
                    os.environ["MSTOCK_LOT_SIZE"] = v
                    to_persist["MSTOCK_LOT_SIZE"] = v
                if dist_var.get().strip():
                    v = str(float(dist_var.get().strip()))
                    os.environ["MSTOCK_SHORT_STRIKE_DISTANCE"] = v
                    to_persist["MSTOCK_SHORT_STRIKE_DISTANCE"] = v
                    try:
                        if hasattr(self, "distance_var"):
                            self.distance_var.set(v)
                    except Exception:
                        pass
                if atr_thr_var.get().strip():
                    v = str(float(atr_thr_var.get().strip()))
                    os.environ["MSTOCK_STRADDLE_ATR_THRESHOLD"] = v
                    to_persist["MSTOCK_STRADDLE_ATR_THRESHOLD"] = v
                if wing_var.get().strip():
                    v = str(float(wing_var.get().strip()))
                    os.environ["MSTOCK_WING_WIDTH"] = v
                    to_persist["MSTOCK_WING_WIDTH"] = v
                if max_trades_var.get().strip():
                    v = str(int(max_trades_var.get().strip()))
                    os.environ["MSTOCK_MAX_TRADES_PER_DAY"] = v
                    to_persist["MSTOCK_MAX_TRADES_PER_DAY"] = v
                if max_pos_var.get().strip():
                    v = str(int(max_pos_var.get().strip()))
                    os.environ["MSTOCK_MAX_OPEN_POSITIONS"] = v
                    to_persist["MSTOCK_MAX_OPEN_POSITIONS"] = v

                if min_atr_var.get().strip():
                    v = str(float(min_atr_var.get().strip()))
                    os.environ["MSTOCK_MIN_ATR"] = v
                    to_persist["MSTOCK_MIN_ATR"] = v
                if max_atr_var.get().strip():
                    v = str(float(max_atr_var.get().strip()))
                    os.environ["MSTOCK_MAX_ATR"] = v
                    to_persist["MSTOCK_MAX_ATR"] = v
                if atr_period_var.get().strip():
                    v = str(int(atr_period_var.get().strip()))
                    os.environ["MSTOCK_ATR_PERIOD"] = v
                    to_persist["MSTOCK_ATR_PERIOD"] = v
                if cooldown_var.get().strip():
                    v = str(float(cooldown_var.get().strip()))
                    os.environ["MSTOCK_COOLDOWN_SEC"] = v
                    to_persist["MSTOCK_COOLDOWN_SEC"] = v
                if cooldown_stopout_var.get().strip():
                    v = str(float(cooldown_stopout_var.get().strip()))
                    os.environ["MSTOCK_COOLDOWN_AFTER_STOPOUT_SEC"] = v
                    to_persist["MSTOCK_COOLDOWN_AFTER_STOPOUT_SEC"] = v
                if cooldown_atr_mult_var.get().strip():
                    v = str(float(cooldown_atr_mult_var.get().strip()))
                    os.environ["MSTOCK_COOLDOWN_ATR_MULT"] = v
                    to_persist["MSTOCK_COOLDOWN_ATR_MULT"] = v

                # Directional entry quality (reduce churn)
                if dir_min_confirm_var.get().strip():
                    v = str(int(dir_min_confirm_var.get().strip()))
                    os.environ["MSTOCK_DIR_MIN_CONFIRMATIONS"] = v
                    to_persist["MSTOCK_DIR_MIN_CONFIRMATIONS"] = v
                if dir_min_diff_var.get().strip():
                    v = str(int(dir_min_diff_var.get().strip()))
                    os.environ["MSTOCK_DIR_MIN_SCORE_DIFF"] = v
                    to_persist["MSTOCK_DIR_MIN_SCORE_DIFF"] = v
                if dir_mom_atr_mult_var.get().strip():
                    v = str(float(dir_mom_atr_mult_var.get().strip()))
                    os.environ["MSTOCK_DIR_MOMENTUM_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_MOMENTUM_ATR_MULT"] = v
                os.environ["MSTOCK_DIR_ALLOW_TIEBREAK"] = "true" if dir_allow_tie_var.get() else "false"
                to_persist["MSTOCK_DIR_ALLOW_TIEBREAK"] = "true" if dir_allow_tie_var.get() else "false"
                if dir_ema_slope_var.get().strip():
                    v = str(float(dir_ema_slope_var.get().strip()))
                    os.environ["MSTOCK_DIR_EMA_SLOPE_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_EMA_SLOPE_ATR_MULT"] = v
                if stopouts_var.get().strip():
                    v = str(int(stopouts_var.get().strip()))
                    os.environ["MSTOCK_MAX_STOPOUTS_PER_DAY"] = v
                    to_persist["MSTOCK_MAX_STOPOUTS_PER_DAY"] = v
                if consec_var.get().strip():
                    v = str(int(consec_var.get().strip()))
                    os.environ["MSTOCK_MAX_CONSECUTIVE_STOPOUTS"] = v
                    to_persist["MSTOCK_MAX_CONSECUTIVE_STOPOUTS"] = v

                if max_same_type_var.get().strip():
                    v = str(int(max_same_type_var.get().strip()))
                    os.environ["MSTOCK_MAX_CONSECUTIVE_SAME_TRADE_TYPE"] = v
                    to_persist["MSTOCK_MAX_CONSECUTIVE_SAME_TRADE_TYPE"] = v
                if max_same_dir_qty_var.get().strip():
                    v = str(int(max_same_dir_qty_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MAX_SAME_DIRECTION_QTY"] = v
                    to_persist["MSTOCK_ENTRY_MAX_SAME_DIRECTION_QTY"] = v

                # Persist ADX & Volume
                os.environ["MSTOCK_ENABLE_ADX_FILTER"] = "true" if adx_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_ADX_FILTER"] = "true" if adx_enabled_var.get() else "false"
                
                if adx_min_var.get().strip():
                    v = str(float(adx_min_var.get().strip()))
                    os.environ["MSTOCK_ADX_MIN_STRENGTH"] = v
                    to_persist["MSTOCK_ADX_MIN_STRENGTH"] = v
                
                if adx_period_var.get().strip():
                    v = str(int(adx_period_var.get().strip()))
                    os.environ["MSTOCK_ADX_PERIOD"] = v
                    to_persist["MSTOCK_ADX_PERIOD"] = v

                os.environ["MSTOCK_ENABLE_VOLUME_FILTER"] = "true" if vol_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_VOLUME_FILTER"] = "true" if vol_enabled_var.get() else "false"

                if vol_sma_var.get().strip():
                    v = str(int(vol_sma_var.get().strip()))
                    os.environ["MSTOCK_VOLUME_SMA_PERIOD"] = v
                    to_persist["MSTOCK_VOLUME_SMA_PERIOD"] = v
                if daily_loss_var.get().strip():
                    v = str(float(daily_loss_var.get().strip()))
                    os.environ["MSTOCK_MAX_DAILY_LOSS"] = v
                    to_persist["MSTOCK_MAX_DAILY_LOSS"] = v
                os.environ["MSTOCK_RISK_SCALE_ENABLED"] = "true" if risk_scale_enabled_var.get() else "false"
                to_persist["MSTOCK_RISK_SCALE_ENABLED"] = os.environ["MSTOCK_RISK_SCALE_ENABLED"]
                if risk_scale_stopout_var.get().strip():
                    v = str(float(risk_scale_stopout_var.get().strip()))
                    os.environ["MSTOCK_RISK_SCALE_STOPOUT_FACTOR"] = v
                    to_persist["MSTOCK_RISK_SCALE_STOPOUT_FACTOR"] = v
                if risk_scale_recovery_var.get().strip():
                    v = str(int(risk_scale_recovery_var.get().strip()))
                    os.environ["MSTOCK_RISK_SCALE_RECOVERY_WINS"] = v
                    to_persist["MSTOCK_RISK_SCALE_RECOVERY_WINS"] = v
                if risk_scale_atr_high_var.get().strip():
                    v = str(float(risk_scale_atr_high_var.get().strip()))
                    os.environ["MSTOCK_RISK_SCALE_ATR_HIGH"] = v
                    to_persist["MSTOCK_RISK_SCALE_ATR_HIGH"] = v
                if risk_scale_atr_factor_var.get().strip():
                    v = str(float(risk_scale_atr_factor_var.get().strip()))
                    os.environ["MSTOCK_RISK_SCALE_ATR_HIGH_FACTOR"] = v
                    to_persist["MSTOCK_RISK_SCALE_ATR_HIGH_FACTOR"] = v
                if risk_scale_min_qty_var.get().strip():
                    v = str(int(risk_scale_min_qty_var.get().strip()))
                    os.environ["MSTOCK_RISK_SCALE_MIN_QTY"] = v
                    to_persist["MSTOCK_RISK_SCALE_MIN_QTY"] = v
                if risk_scale_max_qty_var.get().strip():
                    v = str(int(risk_scale_max_qty_var.get().strip()))
                    os.environ["MSTOCK_RISK_SCALE_MAX_QTY"] = v
                    to_persist["MSTOCK_RISK_SCALE_MAX_QTY"] = v
                if risk_scale_step_qty_var.get().strip():
                    v = str(int(risk_scale_step_qty_var.get().strip()))
                    os.environ["MSTOCK_RISK_SCALE_STEP_QTY"] = v
                    to_persist["MSTOCK_RISK_SCALE_STEP_QTY"] = v
                if max_hold_var.get().strip():
                    v = str(int(max_hold_var.get().strip()))
                    os.environ["MSTOCK_MAX_HOLD_MINUTES"] = v
                    to_persist["MSTOCK_MAX_HOLD_MINUTES"] = v
                if max_hold_days_var.get().strip():
                    v = str(int(max_hold_days_var.get().strip()))
                    os.environ["MSTOCK_MAX_HOLD_DAYS"] = v
                    to_persist["MSTOCK_MAX_HOLD_DAYS"] = v
                if trend_strength_var.get().strip():
                    v = str(float(trend_strength_var.get().strip()))
                    os.environ["MSTOCK_MAX_TREND_STRENGTH"] = v
                    to_persist["MSTOCK_MAX_TREND_STRENGTH"] = v
                if ema_fast_var.get().strip():
                    v = str(int(ema_fast_var.get().strip()))
                    os.environ["MSTOCK_EMA_FAST"] = v
                    to_persist["MSTOCK_EMA_FAST"] = v
                if ema_slow_var.get().strip():
                    v = str(int(ema_slow_var.get().strip()))
                    os.environ["MSTOCK_EMA_SLOW"] = v
                    to_persist["MSTOCK_EMA_SLOW"] = v

                v = "true" if bool(premium_rsi_enabled_var.get()) else "false"
                os.environ["MSTOCK_ENABLE_PREMIUM_RSI_FILTER"] = v
                to_persist["MSTOCK_ENABLE_PREMIUM_RSI_FILTER"] = v
                if premium_rsi_low_var.get().strip():
                    v = str(float(premium_rsi_low_var.get().strip()))
                    os.environ["MSTOCK_PREMIUM_RSI_LOW"] = v
                    to_persist["MSTOCK_PREMIUM_RSI_LOW"] = v
                if premium_rsi_high_var.get().strip():
                    v = str(float(premium_rsi_high_var.get().strip()))
                    os.environ["MSTOCK_PREMIUM_RSI_HIGH"] = v
                    to_persist["MSTOCK_PREMIUM_RSI_HIGH"] = v

                if entry_candle_age_var.get().strip():
                    v = str(float(entry_candle_age_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_CANDLE_MAX_AGE_SEC"] = v
                    to_persist["MSTOCK_ENTRY_CANDLE_MAX_AGE_SEC"] = v
                if entry_candle_range_var.get().strip():
                    v = str(float(entry_candle_range_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MAX_CANDLE_RANGE_ATR_MULT"] = v
                    to_persist["MSTOCK_ENTRY_MAX_CANDLE_RANGE_ATR_MULT"] = v
                if entry_gap_var.get().strip():
                    v = str(float(entry_gap_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_GAP_ATR_MULT"] = v
                    to_persist["MSTOCK_ENTRY_GAP_ATR_MULT"] = v
                if entry_min_prem_var.get().strip():
                    v = str(float(entry_min_prem_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MIN_OPTION_PREMIUM"] = v
                    to_persist["MSTOCK_ENTRY_MIN_OPTION_PREMIUM"] = v
                if entry_max_prem_var.get().strip():
                    v = str(float(entry_max_prem_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MAX_OPTION_PREMIUM"] = v
                    to_persist["MSTOCK_ENTRY_MAX_OPTION_PREMIUM"] = v
                if entry_min_total_prem_var.get().strip():
                    v = str(float(entry_min_total_prem_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MIN_TOTAL_PREMIUM"] = v
                    to_persist["MSTOCK_ENTRY_MIN_TOTAL_PREMIUM"] = v
                if entry_max_total_prem_var.get().strip():
                    v = str(float(entry_max_total_prem_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MAX_TOTAL_PREMIUM"] = v
                    to_persist["MSTOCK_ENTRY_MAX_TOTAL_PREMIUM"] = v
                os.environ["MSTOCK_ENTRY_REQUIRE_BID_ASK"] = "true" if entry_req_ba_var.get() else "false"
                to_persist["MSTOCK_ENTRY_REQUIRE_BID_ASK"] = os.environ["MSTOCK_ENTRY_REQUIRE_BID_ASK"]
                if entry_spread_pct_var.get().strip():
                    v = str(float(entry_spread_pct_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_PCT"] = v
                    to_persist["MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_PCT"] = v
                if entry_spread_abs_var.get().strip():
                    v = str(float(entry_spread_abs_var.get().strip()))
                    os.environ["MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_ABS"] = v
                    to_persist["MSTOCK_ENTRY_MAX_BID_ASK_SPREAD_ABS"] = v
                if iv_expand_var.get().strip():
                    v = str(float(iv_expand_var.get().strip()))
                    os.environ["MSTOCK_IV_EXPAND_THRESHOLD_PCT"] = v
                    to_persist["MSTOCK_IV_EXPAND_THRESHOLD_PCT"] = v
                if iv_contract_var.get().strip():
                    v = str(float(iv_contract_var.get().strip()))
                    os.environ["MSTOCK_IV_CONTRACT_THRESHOLD_PCT"] = v
                    to_persist["MSTOCK_IV_CONTRACT_THRESHOLD_PCT"] = v

                v = "true" if bool(weekly_only_var.get()) else "false"
                os.environ["MSTOCK_NIFTY_WEEKLY_ONLY"] = v
                to_persist["MSTOCK_NIFTY_WEEKLY_ONLY"] = v

                v = "true" if bool(bypass_weekly_var.get()) else "false"
                os.environ["MSTOCK_BYPASS_WEEKLY_FILTER"] = v
                to_persist["MSTOCK_BYPASS_WEEKLY_FILTER"] = v

                sm_path = str(scripmaster_path_var.get() or "").strip()
                if sm_path:
                    os.environ["MSTOCK_SCRIPMASTER_PATH"] = sm_path
                    to_persist["MSTOCK_SCRIPMASTER_PATH"] = sm_path
                    try:
                        if hasattr(self, "scripmaster_path_var"):
                            self.scripmaster_path_var.set(sm_path)
                    except Exception:
                        pass
                else:
                    os.environ.pop("MSTOCK_SCRIPMASTER_PATH", None)
                    to_unset.append("MSTOCK_SCRIPMASTER_PATH")
                    try:
                        if hasattr(self, "scripmaster_path_var"):
                            self.scripmaster_path_var.set("")
                    except Exception:
                        pass

                # Prefer the combobox widget's current text so typed values are captured.
                _tgt_raw = ""
                try:
                    _tgt_raw = str(expiry_combo.get() or "").strip()
                except Exception:
                    _tgt_raw = ""
                if not _tgt_raw:
                    _tgt_raw = str(target_expiry_var.get() or "").strip()

                if _tgt_raw:
                    v = _tgt_raw
                    os.environ["MSTOCK_TARGET_EXPIRY"] = v
                    to_persist["MSTOCK_TARGET_EXPIRY"] = v
                    try:
                        if hasattr(self, "target_expiry_var"):
                            self.target_expiry_var.set(v)
                    except Exception:
                        pass
                else:
                    os.environ.pop("MSTOCK_TARGET_EXPIRY", None)
                    to_unset.append("MSTOCK_TARGET_EXPIRY")
                    try:
                        if hasattr(self, "target_expiry_var"):
                            self.target_expiry_var.set("")
                    except Exception:
                        pass

                if auto_trend_mult_var.get().strip():
                    v = str(float(auto_trend_mult_var.get().strip()))
                    os.environ["MSTOCK_AUTO_RANGE_TREND_MULT"] = v
                    to_persist["MSTOCK_AUTO_RANGE_TREND_MULT"] = v
                if auto_rsi_buy_var.get().strip():
                    v = str(float(auto_rsi_buy_var.get().strip()))
                    os.environ["MSTOCK_AUTO_DIR_RSI_BUY"] = v
                    to_persist["MSTOCK_AUTO_DIR_RSI_BUY"] = v
                if auto_rsi_sell_var.get().strip():
                    v = str(float(auto_rsi_sell_var.get().strip()))
                    os.environ["MSTOCK_AUTO_DIR_RSI_SELL"] = v
                    to_persist["MSTOCK_AUTO_DIR_RSI_SELL"] = v
                if auto_rsi_slop_var.get().strip():
                    v = str(float(auto_rsi_slop_var.get().strip()))
                    os.environ["MSTOCK_AUTO_DIR_RSI_SLOP"] = v
                    to_persist["MSTOCK_AUTO_DIR_RSI_SLOP"] = v

                if mtm_stop_var.get().strip():
                    v = str(float(mtm_stop_var.get().strip()))
                    os.environ["MSTOCK_PREMIUM_MTM_STOP_PCT"] = v
                    to_persist["MSTOCK_PREMIUM_MTM_STOP_PCT"] = v
                if mtm_tgt_var.get().strip():
                    v = str(float(mtm_tgt_var.get().strip()))
                    os.environ["MSTOCK_PREMIUM_MTM_TARGET_PCT"] = v
                    to_persist["MSTOCK_PREMIUM_MTM_TARGET_PCT"] = v
                if mtm_trail_start_var.get().strip():
                    v = str(float(mtm_trail_start_var.get().strip()))
                    os.environ["MSTOCK_PREMIUM_MTM_TRAIL_START_PCT"] = v
                    to_persist["MSTOCK_PREMIUM_MTM_TRAIL_START_PCT"] = v
                if mtm_trail_stop_var.get().strip():
                    v = str(float(mtm_trail_stop_var.get().strip()))
                    os.environ["MSTOCK_PREMIUM_MTM_TRAIL_STOP_PCT"] = v
                    to_persist["MSTOCK_PREMIUM_MTM_TRAIL_STOP_PCT"] = v

                os.environ["MSTOCK_GPT_LEG_MANAGE"] = "true" if gpt_leg_manage_var.get() else "false"
                to_persist["MSTOCK_GPT_LEG_MANAGE"] = os.environ["MSTOCK_GPT_LEG_MANAGE"]
                if gpt_leg_refresh_var.get().strip():
                    v = str(float(gpt_leg_refresh_var.get().strip()))
                    os.environ["MSTOCK_GPT_LEG_MANAGE_REFRESH_SEC"] = v
                    to_persist["MSTOCK_GPT_LEG_MANAGE_REFRESH_SEC"] = v
                if gpt_leg_min_age_var.get().strip():
                    v = str(float(gpt_leg_min_age_var.get().strip()))
                    os.environ["MSTOCK_GPT_LEG_MANAGE_MIN_AGE_SEC"] = v
                    to_persist["MSTOCK_GPT_LEG_MANAGE_MIN_AGE_SEC"] = v
                if gpt_leg_grace_var.get().strip():
                    v = str(float(gpt_leg_grace_var.get().strip()))
                    os.environ["MSTOCK_GPT_LEG_MANAGE_POST_UPDATE_GRACE_SEC"] = v
                    to_persist["MSTOCK_GPT_LEG_MANAGE_POST_UPDATE_GRACE_SEC"] = v
                if gpt_leg_stop_dist_var.get().strip():
                    v = str(float(gpt_leg_stop_dist_var.get().strip()))
                    os.environ["MSTOCK_GPT_LEG_MANAGE_MIN_STOP_DISTANCE_PCT"] = v
                    to_persist["MSTOCK_GPT_LEG_MANAGE_MIN_STOP_DISTANCE_PCT"] = v
                if gpt_leg_target_dist_var.get().strip():
                    v = str(float(gpt_leg_target_dist_var.get().strip()))
                    os.environ["MSTOCK_GPT_LEG_MANAGE_MIN_TARGET_DISTANCE_PCT"] = v
                    to_persist["MSTOCK_GPT_LEG_MANAGE_MIN_TARGET_DISTANCE_PCT"] = v

                v = "true" if bool(exit_short_var.get()) else "false"
                os.environ["MSTOCK_PREMIUM_EXIT_ON_SHORT_TOUCH"] = v
                to_persist["MSTOCK_PREMIUM_EXIT_ON_SHORT_TOUCH"] = v

                v = "true" if bool(exit_wing_var.get()) else "false"
                os.environ["MSTOCK_PREMIUM_EXIT_ON_WING_TOUCH"] = v
                to_persist["MSTOCK_PREMIUM_EXIT_ON_WING_TOUCH"] = v

                if force_exit_var.get().strip():
                    v = force_exit_var.get().strip()
                    os.environ["MSTOCK_PREMIUM_FORCE_EXIT"] = v
                    to_persist["MSTOCK_PREMIUM_FORCE_EXIT"] = v
                if cutoff_var.get().strip():
                    v = cutoff_var.get().strip()
                    os.environ["MSTOCK_PREMIUM_ENTRY_CUTOFF"] = v
                    to_persist["MSTOCK_PREMIUM_ENTRY_CUTOFF"] = v
                else:
                    os.environ.pop("MSTOCK_PREMIUM_ENTRY_CUTOFF", None)
                    to_unset.append("MSTOCK_PREMIUM_ENTRY_CUTOFF")

                if vwap_dev_var.get().strip():
                    v = str(float(vwap_dev_var.get().strip()))
                    os.environ["MSTOCK_VWAP_MAX_DEV_PCT"] = v
                    to_persist["MSTOCK_VWAP_MAX_DEV_PCT"] = v
                v = "true" if bool(vwap_enabled_var.get()) else "false"
                os.environ["MSTOCK_ENABLE_VWAP_FILTER"] = v
                to_persist["MSTOCK_ENABLE_VWAP_FILTER"] = v

                if open_move_var.get().strip():
                    v = str(float(open_move_var.get().strip()))
                    os.environ["MSTOCK_MAX_OPEN_MOVE_PCT"] = v
                    to_persist["MSTOCK_MAX_OPEN_MOVE_PCT"] = v
                if open_min_var.get().strip():
                    v = str(int(open_min_var.get().strip()))
                    os.environ["MSTOCK_OPENING_FILTER_MINUTES"] = v
                    to_persist["MSTOCK_OPENING_FILTER_MINUTES"] = v
                v = "true" if bool(opening_enabled_var.get()) else "false"
                os.environ["MSTOCK_ENABLE_OPENING_FILTER"] = v
                to_persist["MSTOCK_ENABLE_OPENING_FILTER"] = v

                v = "true" if bool(trend_enabled_var.get()) else "false"
                os.environ["MSTOCK_ENABLE_TREND_FILTER"] = v
                to_persist["MSTOCK_ENABLE_TREND_FILTER"] = v

                v = "true" if bool(supertrend_enabled_var.get()) else "false"
                os.environ["MSTOCK_ENABLE_SUPERTREND_FILTER"] = v
                to_persist["MSTOCK_ENABLE_SUPERTREND_FILTER"] = v

                if supertrend_mode_var.get().strip():
                    v = str(supertrend_mode_var.get().strip().lower())
                    os.environ["MSTOCK_SUPERTREND_MODE"] = v
                    to_persist["MSTOCK_SUPERTREND_MODE"] = v

                if dir_be_var.get().strip():
                    v = str(float(dir_be_var.get().strip()))
                    os.environ["MSTOCK_DIR_BREAKEVEN_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_BREAKEVEN_ATR_MULT"] = v
                if dir_trail_atr_var.get().strip():
                    v = str(float(dir_trail_atr_var.get().strip()))
                    os.environ["MSTOCK_DIR_TRAIL_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_TRAIL_ATR_MULT"] = v
                if dir_prem_trail_var.get().strip():
                    v = str(float(dir_prem_trail_var.get().strip()))
                    os.environ["MSTOCK_DIR_PREMIUM_TRAIL_PCT"] = v
                    to_persist["MSTOCK_DIR_PREMIUM_TRAIL_PCT"] = v
                v = "true" if bool(req_spot_var.get()) else "false"
                os.environ["MSTOCK_REQUIRE_SPOT_LTP_FOR_ENTRY"] = v
                to_persist["MSTOCK_REQUIRE_SPOT_LTP_FOR_ENTRY"] = v

                if dir_partial_tgt_var.get().strip():
                    v = str(float(dir_partial_tgt_var.get().strip()))
                    os.environ["MSTOCK_DIR_PARTIAL_TARGET_MULT"] = v
                    to_persist["MSTOCK_DIR_PARTIAL_TARGET_MULT"] = v
                if dir_partial_qty_var.get().strip():
                    v = str(float(dir_partial_qty_var.get().strip()))
                    os.environ["MSTOCK_DIR_PARTIAL_QTY_PCT"] = v
                    to_persist["MSTOCK_DIR_PARTIAL_QTY_PCT"] = v
                if dir_post_be_var.get().strip():
                    v = str(float(dir_post_be_var.get().strip()))
                    os.environ["MSTOCK_DIR_POST_PARTIAL_BE_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_POST_PARTIAL_BE_ATR_MULT"] = v
                if dir_post_trail_var.get().strip():
                    v = str(float(dir_post_trail_var.get().strip()))
                    os.environ["MSTOCK_DIR_POST_PARTIAL_TRAIL_ATR_MULT"] = v
                    to_persist["MSTOCK_DIR_POST_PARTIAL_TRAIL_ATR_MULT"] = v

                if spike_var.get().strip():
                    v = str(float(spike_var.get().strip()))
                    os.environ["MSTOCK_ATR_SPIKE_MULT"] = v
                    to_persist["MSTOCK_ATR_SPIKE_MULT"] = v
                
                # Persist New Enhancements
                if limit_buf_var.get().strip():
                    v = str(float(limit_buf_var.get().strip()))
                    os.environ["MSTOCK_LIMIT_PRICE_BUFFER_PCT"] = v
                    to_persist["MSTOCK_LIMIT_PRICE_BUFFER_PCT"] = v
                
                os.environ["MSTOCK_ENABLE_MTF_CONFIRMATION"] = "true" if mtf_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_MTF_CONFIRMATION"] = os.environ["MSTOCK_ENABLE_MTF_CONFIRMATION"]
                
                os.environ["MSTOCK_ENABLE_ROC_FILTER"] = "true" if roc_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_ROC_FILTER"] = os.environ["MSTOCK_ENABLE_ROC_FILTER"]
                
                os.environ["MSTOCK_ENABLE_CHOPPINESS_FILTER"] = "true" if chop_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_CHOPPINESS_FILTER"] = os.environ["MSTOCK_ENABLE_CHOPPINESS_FILTER"]
                
                os.environ["MSTOCK_ENABLE_CHANDELIER_EXIT"] = "true" if chan_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_CHANDELIER_EXIT"] = os.environ["MSTOCK_ENABLE_CHANDELIER_EXIT"]
                
                os.environ["MSTOCK_ENABLE_PIVOT_TARGETS"] = "true" if pivot_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_PIVOT_TARGETS"] = os.environ["MSTOCK_ENABLE_PIVOT_TARGETS"]
                
                os.environ["MSTOCK_ENABLE_LIMIT_ORDERS"] = "true" if limit_enabled_var.get() else "false"
                to_persist["MSTOCK_ENABLE_LIMIT_ORDERS"] = os.environ["MSTOCK_ENABLE_LIMIT_ORDERS"]

                if pyramid_var.get().strip():
                    v = str(int(pyramid_var.get().strip()))
                    os.environ["MSTOCK_MAX_PYRAMID_LEVELS"] = v
                    to_persist["MSTOCK_MAX_PYRAMID_LEVELS"] = v

                v = "true" if bool(no_signal_var.get()) else "false"
                os.environ["MSTOCK_DEBUG_NO_SIGNAL"] = v
                to_persist["MSTOCK_DEBUG_NO_SIGNAL"] = v


                persist_settings_env(to_persist, to_unset)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Error", f"Failed to apply settings: {exc}")
                return

            # If the bot is running, offer an immediate restart so changes apply.
            try:
                running = bool(self._bot_thread and self._bot_thread.is_alive())
            except Exception:
                running = False

            if running:
                do_restart = messagebox.askyesno(
                    "Settings updated",
                    "Settings saved. Restart bot now to apply changes?",
                )
                if do_restart:
                    self._stop_and_restart_bot()
                    return

            messagebox.showinfo(
                "Settings updated",
                "Settings saved. Stop the bot (if running) and click Start Bot again for changes to take effect.",
            )

        row += 1
        tk.Button(content, text="Apply", command=apply_changes).grid(row=row, column=0, sticky="w", padx=8, pady=(10, 8))
        tk.Button(content, text="Close", command=win.destroy).grid(row=row, column=1, sticky="e", padx=8, pady=(10, 8))

        # Ensure initial scroll region is correct.
        _on_content_configure()
        canvas.yview_moveto(0.0)

    def _set_env_from_fields(self) -> None:
        os.environ["MSTOCK_API_KEY"] = self.api_key_var.get().strip()
        os.environ["MSTOCK_USERNAME"] = self.username_var.get().strip()
        os.environ["MSTOCK_PASSWORD"] = self.password_var.get()
        token = self.access_token_var.get().strip()
        if token:
            os.environ["MSTOCK_ACCESS_TOKEN"] = token

        # Underlying token/exchange for candle fetching.
        if hasattr(self, "underlying_token_var"):
            u_tok = self.underlying_token_var.get().strip()
            if u_tok:
                os.environ["MSTOCK_UNDERLYING_TOKEN"] = u_tok
            else:
                os.environ.pop("MSTOCK_UNDERLYING_TOKEN", None)
        if hasattr(self, "underlying_exchange_var"):
            u_exch = self.underlying_exchange_var.get().strip()
            if u_exch:
                os.environ["MSTOCK_UNDERLYING_EXCHANGE"] = u_exch
            else:
                os.environ.pop("MSTOCK_UNDERLYING_EXCHANGE", None)

        # Target options expiry for contract selection.
        # Prefer the widget's current text so typed values are captured even if the
        # StringVar hasn't updated yet.
        tgt = ""
        try:
            if hasattr(self, "target_expiry_combo"):
                tgt = str(self.target_expiry_combo.get() or "").strip()
        except Exception:
            tgt = ""

        if not tgt and hasattr(self, "target_expiry_var"):
            try:
                tgt = str(self.target_expiry_var.get() or "").strip()
            except Exception:
                tgt = ""

        if tgt:
            os.environ["MSTOCK_TARGET_EXPIRY"] = tgt
            if hasattr(self, "target_expiry_var"):
                try:
                    self.target_expiry_var.set(tgt)
                except Exception:
                    pass
        else:
            os.environ.pop("MSTOCK_TARGET_EXPIRY", None)

        if hasattr(self, "use_intraday_chart_var"):
            os.environ["MSTOCK_USE_INTRADAY_CHART"] = "true" if bool(self.use_intraday_chart_var.get()) else "false"
        if hasattr(self, "startup_use_historical_candles_var"):
            os.environ["MSTOCK_STARTUP_USE_HISTORICAL_CANDLES"] = (
                "true" if bool(self.startup_use_historical_candles_var.get()) else "false"
            )

        # ScripMaster CSV mapping.
        if hasattr(self, "scripmaster_path_var"):
            sm_path = self.scripmaster_path_var.get().strip()
            if sm_path:
                os.environ["MSTOCK_SCRIPMASTER_PATH"] = sm_path
            else:
                os.environ.pop("MSTOCK_SCRIPMASTER_PATH", None)
        if hasattr(self, "csv_only_var"):
            os.environ["MSTOCK_USE_CSV_ONLY"] = "true" if bool(self.csv_only_var.get()) else "false"
        if hasattr(self, "chain_fallback_var"):
            os.environ["MSTOCK_USE_CHAIN_FALLBACK"] = "true" if bool(self.chain_fallback_var.get()) else "false"

        if hasattr(self, "auto_fetch_scripmaster_var"):
            os.environ["MSTOCK_AUTO_FETCH_SCRIPMASTER"] = "true" if bool(self.auto_fetch_scripmaster_var.get()) else "false"

        if hasattr(self, "preset_var"):
            p = str(self.preset_var.get() or "").strip().lower()
            if p == "aggressive":
                os.environ["MSTOCK_PRESET"] = "aggressive"
            elif p == "conservative":
                os.environ["MSTOCK_PRESET"] = "conservative"
            else:
                os.environ.pop("MSTOCK_PRESET", None)

        os.environ["MSTOCK_STRATEGY"] = self.strategy_var.get().strip().lower() or "directional"
        os.environ["MSTOCK_TIMEFRAME"] = self.timeframe_var.get().strip() or "1m"
        os.environ["MSTOCK_SHORT_STRIKE_DISTANCE"] = self.distance_var.get().strip() or "50"
        os.environ["MSTOCK_ENABLE_LIVE_TRADING"] = "true" if self.live_var.get() else "false"

        if hasattr(self, "delta_hedge_scope_var"):
            os.environ["MSTOCK_DELTA_HEDGE_SCOPE"] = (
                self.delta_hedge_scope_var.get().strip().lower() or "strategy_only"
            )



    def _default_credential_path(self) -> Path:
        # Store in user profile (not in repo) to keep defaults "preinstalled" locally.
        base = os.getenv("APPDATA")
        if base:
            return Path(base) / "scalper" / "credentials.json"
        return Path.home() / ".scalper" / "credentials.json"

    def _load_prefilled_credentials(self) -> None:
        # Priority: saved file -> env vars -> blanks.
        saved = {}
        try:
            if self._cred_path.exists():
                saved = json.loads(self._cred_path.read_text(encoding="utf-8"))
        except Exception:
            saved = {}

        api_key = str(saved.get("api_key") or os.getenv("MSTOCK_API_KEY", ""))
        username = str(saved.get("username") or os.getenv("MSTOCK_USERNAME", ""))
        password = str(saved.get("password") or os.getenv("MSTOCK_PASSWORD", ""))
        underlying_token = str(saved.get("underlying_token") or os.getenv("MSTOCK_UNDERLYING_TOKEN", ""))
        underlying_exchange = str(
            saved.get("underlying_exchange")
            or os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE"))
        )
        use_intraday_chart = bool(
            saved.get("use_intraday_chart")
            if "use_intraday_chart" in saved
            else os.getenv("MSTOCK_USE_INTRADAY_CHART", "false").lower() in {"1", "true", "yes", "y"}
        )
        startup_use_historical_candles = bool(
            saved.get("startup_use_historical_candles")
            if "startup_use_historical_candles" in saved
            else os.getenv("MSTOCK_STARTUP_USE_HISTORICAL_CANDLES", "true").lower() in {"1", "true", "yes", "y"}
        )
        scripmaster_path = str(saved.get("scripmaster_path") or os.getenv("MSTOCK_SCRIPMASTER_PATH", ""))
        csv_only = bool(
            saved.get("use_csv_only")
            if "use_csv_only" in saved
            else os.getenv("MSTOCK_USE_CSV_ONLY", "false").lower() in {"1", "true", "yes", "y"}
        )
        chain_fallback = bool(
            saved.get("use_chain_fallback")
            if "use_chain_fallback" in saved
            else os.getenv("MSTOCK_USE_CHAIN_FALLBACK", "true").lower() in {"1", "true", "yes", "y"}
        )
        auto_fetch_sm = bool(
            saved.get("auto_fetch_scripmaster")
            if "auto_fetch_scripmaster" in saved
            else os.getenv("MSTOCK_AUTO_FETCH_SCRIPMASTER", "true").lower() in {"1", "true", "yes", "y"}
        )
        # Target expiry is a bot setting rather than a credential; prefer env (including
        # .scalper.env / Settings dialog) over any previously saved credentials value.
        target_expiry = str(os.getenv("MSTOCK_TARGET_EXPIRY", "") or saved.get("target_expiry") or "")
        short_strike_distance = str(
            saved.get("short_strike_distance")
            or os.getenv("MSTOCK_SHORT_STRIKE_DISTANCE", "50")
        )

        self.api_key_var.set(api_key)
        self.username_var.set(username)
        self.password_var.set(password)
        if hasattr(self, "underlying_token_var"):
            self.underlying_token_var.set(underlying_token)
        if hasattr(self, "underlying_exchange_var"):
            self.underlying_exchange_var.set(underlying_exchange)
        if hasattr(self, "use_intraday_chart_var"):
            self.use_intraday_chart_var.set(bool(use_intraday_chart))
        if hasattr(self, "startup_use_historical_candles_var"):
            self.startup_use_historical_candles_var.set(bool(startup_use_historical_candles))
        if hasattr(self, "scripmaster_path_var"):
            self.scripmaster_path_var.set(scripmaster_path)
        if hasattr(self, "csv_only_var"):
            self.csv_only_var.set(bool(csv_only))
        if hasattr(self, "chain_fallback_var"):
            self.chain_fallback_var.set(bool(chain_fallback))
        if hasattr(self, "auto_fetch_scripmaster_var"):
            self.auto_fetch_scripmaster_var.set(bool(auto_fetch_sm))
        if hasattr(self, "target_expiry_var"):
            self.target_expiry_var.set(target_expiry)
        if hasattr(self, "distance_var"):
            self.distance_var.set(short_strike_distance)

        if saved:
            self.remember_var.set(True)
            print(f"Loaded saved credentials from {self._cred_path}.\n")

        # If we don't have complete credentials yet, default to editable so
        # the UI doesn't appear "broken" on first run.
        if not (api_key and username and password):
            self.edit_creds_var.set(True)

    def _sync_credential_editability(self) -> None:
        # Use readonly instead of disabled so users can still select/copy text.
        editing = self.edit_creds_var.get()
        state = tk.NORMAL if editing else "readonly"
        self.api_key_entry.configure(state=state)
        self.username_entry.configure(state=state)
        self.password_entry.configure(state=state)

    def _on_save_credentials(self) -> None:
        if not self.remember_var.get():
            messagebox.showinfo(
                "Not saved",
                "Enable 'Remember API key/user/password' to save locally.",
            )
            return

        data = {
            "api_key": self.api_key_var.get().strip(),
            "username": self.username_var.get().strip(),
            "password": self.password_var.get(),
        }
        if hasattr(self, "underlying_token_var"):
            u_tok = self.underlying_token_var.get().strip()
            if u_tok:
                data["underlying_token"] = u_tok
        if hasattr(self, "underlying_exchange_var"):
            u_exch = self.underlying_exchange_var.get().strip()
            if u_exch:
                data["underlying_exchange"] = u_exch
        if hasattr(self, "use_intraday_chart_var"):
            data["use_intraday_chart"] = bool(self.use_intraday_chart_var.get())
        if hasattr(self, "startup_use_historical_candles_var"):
            data["startup_use_historical_candles"] = bool(self.startup_use_historical_candles_var.get())
        if hasattr(self, "scripmaster_path_var"):
            sm_path = self.scripmaster_path_var.get().strip()
            if sm_path:
                data["scripmaster_path"] = sm_path
        if hasattr(self, "csv_only_var"):
            data["use_csv_only"] = bool(self.csv_only_var.get())
        if hasattr(self, "chain_fallback_var"):
            data["use_chain_fallback"] = bool(self.chain_fallback_var.get())
        if hasattr(self, "auto_fetch_scripmaster_var"):
            data["auto_fetch_scripmaster"] = bool(self.auto_fetch_scripmaster_var.get())
        if hasattr(self, "target_expiry_var"):
            tgt = self.target_expiry_var.get().strip()
            if tgt:
                data["target_expiry"] = tgt
        if hasattr(self, "distance_var"):
            dist = self.distance_var.get().strip()
            if dist:
                data["short_strike_distance"] = dist
        if not data["api_key"] or not data["username"] or not data["password"]:
            messagebox.showerror("Missing info", "API key, username, and password are required to save.")
            return

        try:
            self._cred_path.parent.mkdir(parents=True, exist_ok=True)
            self._cred_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Saved credentials to {self._cred_path}.\n")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", f"Could not save credentials: {exc}")

    def _copy_token(self) -> None:
        token = self.access_token_var.get().strip()
        if not token:
            return
        self.clipboard_clear()
        self.clipboard_append(token)
        print("Access token copied to clipboard.\n")

    def _clear_token(self) -> None:
        self.access_token_var.set("")
        os.environ.pop("MSTOCK_ACCESS_TOKEN", None)
        print("Access token cleared.\n")

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

        try:
            if self._ui_log_fp is not None and text:
                self._ui_log_fp.write(text)
                self._ui_log_fp.flush()
        except Exception:
            # Logging should never break the UI.
            pass

    def _pump_logs(self) -> None:
        try:
            # Do not drain the entire queue in one tick; large bursts of logs
            # can freeze the UI and cause Windows to show "Not responding".
            max_items = 250
            processed = 0
            while True:
                chunk = self._log_q.get_nowait()
                self._append_log(chunk)
                processed += 1
                if processed >= max_items:
                    break
        except queue.Empty:
            pass
        self.after(100, self._pump_logs)

    # --- Login flow ---

    def _on_request_otp(self) -> None:
        # Always push any prefilled creds into env for the rest of the app.
        self._set_env_from_fields()
        username = self.username_var.get().strip()
        password = self.password_var.get()
        if not username or not password:
            messagebox.showerror("Missing info", "Enter username and password first.")
            return

        self.btn_request_otp.configure(state=tk.DISABLED)
        self.btn_verify_otp.configure(state=tk.DISABLED)

        def worker() -> None:
            try:
                refresh_token, message = request_sms_otp(username, password)
                self._login_state.refresh_token = refresh_token
                print(message + "\n")
                self.after(0, lambda: self.otp_entry.focus_set())
            except Exception as exc:  # noqa: BLE001
                print(f"Request OTP failed: {exc}\n")
                self._login_state.refresh_token = None
            finally:
                self.after(0, lambda: self.btn_request_otp.configure(state=tk.NORMAL))
                self.after(0, lambda: self.btn_verify_otp.configure(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    def _on_verify_otp(self) -> None:
        self._set_env_from_fields()
        api_key = self.api_key_var.get().strip()
        otp = self.otp_var.get().strip()
        refresh_token = self._login_state.refresh_token

        if not api_key:
            messagebox.showerror("Missing info", "Enter API key first.")
            return
        if not refresh_token:
            messagebox.showerror("Missing step", "Click 'Request OTP' first.")
            return
        if not otp:
            messagebox.showerror("Missing info", "Enter OTP first.")
            return

        self.btn_verify_otp.configure(state=tk.DISABLED)

        def worker() -> None:
            try:
                access_token = verify_sms_otp(api_key, refresh_token, otp)
                os.environ["MSTOCK_ACCESS_TOKEN"] = access_token
                self.after(0, lambda: self.access_token_var.set(access_token))
                self.after(0, lambda: self.otp_var.set(""))
                print("Access token generated and set in MSTOCK_ACCESS_TOKEN.\n")

                if self.remember_var.get():
                    # Save credentials on successful login if user opted in.
                    self.after(0, self._on_save_credentials)
            except Exception as exc:  # noqa: BLE001
                print(f"Verify OTP failed: {exc}\n")
            finally:
                self.after(0, lambda: self.btn_verify_otp.configure(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    # --- Bot control ---

    def _on_start(self) -> None:
        if self._bot_thread and self._bot_thread.is_alive():
            messagebox.showinfo("Already running", "Bot is already running.")
            return

        # Load persisted settings first so current UI fields can override them.
        # This avoids stale .scalper.env values clobbering what the user sees/entered.
        try:
            import config as _cfgmod

            _cfgmod.load_persisted_env(override_existing=True)
            _cfgmod._PERSISTED_ENV_LOADED = True  # type: ignore[attr-defined]
        except Exception:
            pass

        self._set_env_from_fields()
        if not os.getenv("MSTOCK_ACCESS_TOKEN"):
            messagebox.showerror("Not logged in", "Generate/set an access token first.")
            return

        self._bot_stop.clear()
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        if hasattr(self, "status_var"):
            self.status_var.set("Running...")

        def worker() -> None:
            try:
                # Reload persisted settings every time Start Bot is pressed.
                # The UI process can retain old os.environ values from a prior run; without
                # overriding, `load_strategy_config()` will keep using stale knobs.
                try:
                    import config as _cfgmod

                    # Do not override values just set from the UI fields above.
                    _cfgmod.load_persisted_env(override_existing=False)
                    _cfgmod._PERSISTED_ENV_LOADED = True  # type: ignore[attr-defined]
                except Exception:
                    pass

                api_cfg = load_api_config()
                strat_cfg = load_strategy_config()

                # Keep Signals/Greeks tab aligned to the effective config.
                try:
                    if hasattr(self, "_sig_symbol_var") and hasattr(self, "_sig_timeframe_var"):
                        sym_now = str(getattr(strat_cfg, "underlying", "") or "").strip()
                        tf_now = str(getattr(strat_cfg, "timeframe", "") or "").strip()
                        if sym_now:
                            self.after(0, lambda v=sym_now: self._sig_symbol_var.set(v))
                        if tf_now:
                            self.after(0, lambda v=tf_now: self._sig_timeframe_var.set(v))
                except Exception:
                    pass

                try:
                    _tgt = str(getattr(strat_cfg, "target_expiry", "") or "").strip()
                    print(f"[UI] Effective target expiry: {_tgt or '(auto)'}")
                except Exception:
                    pass

                try:
                    _cutoff = str(getattr(strat_cfg, "premium_entry_cutoff_hhmm", "") or "").strip()
                    print(f"[UI] Effective entry cutoff (HH:MM): {_cutoff or '(disabled)'}")
                except Exception:
                    pass

                try:
                    print(
                        "[UI] Effective auto-dir RSI: "
                        f"buy={getattr(strat_cfg, 'auto_dir_rsi_buy', None)} "
                        f"sell={getattr(strat_cfg, 'auto_dir_rsi_sell', None)} "
                        f"slop={getattr(strat_cfg, 'auto_dir_rsi_slop', None)}"
                    )
                except Exception:
                    pass

                client = MStockTypeBClient(api_cfg)
                client.login()  # no-op if token is already set
                try:
                    self.after(0, lambda c=client: setattr(self, "_client", c))
                except Exception:
                    self._client = client

                # m.Stock candles require an underlying token for the selected exchange.
                # Auto-resolve it if missing so candle fetch doesn't block trading.
                u_tok = os.getenv("MSTOCK_UNDERLYING_TOKEN", "").strip()
                if not u_tok:
                    try:
                        exch_hint = os.getenv(
                            "MSTOCK_UNDERLYING_EXCHANGE",
                            os.getenv("MSTOCK_EXCHANGE", "NSE"),
                        )

                        # For index underlyings, candles should generally come from spot (NSE),
                        # even if orders are placed on NFO. Prevent ambiguous NFO:NIFTY matches.
                        underlying_key = str(strat_cfg.underlying or "").strip().upper()
                        if ":" in underlying_key:
                            underlying_key = underlying_key.split(":", 1)[-1].strip().upper()
                        if underlying_key in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}:
                            if str(exch_hint or "").strip().upper() in {"NFO", "2"}:
                                print("[UI] Underlying exchange is NFO but underlying is an index; using NSE for candle token resolve")
                                exch_hint = "NSE"
                                if hasattr(self, "underlying_exchange_var"):
                                    self.after(0, lambda v="NSE": self.underlying_exchange_var.set(v))

                        resolved = client._resolve_token_from_instruments(strat_cfg.underlying, str(exch_hint or "NSE"))
                        resolved = str(resolved or "").strip()
                        if resolved:
                            os.environ["MSTOCK_UNDERLYING_TOKEN"] = resolved
                            print(f"[UI] Auto-resolved MSTOCK_UNDERLYING_TOKEN -> {resolved}")
                            if hasattr(self, "underlying_token_var"):
                                self.after(0, lambda: self.underlying_token_var.set(resolved))
                        else:
                            self.after(
                                0,
                                lambda: messagebox.showwarning(
                                    "Underlying token needed",
                                    "m.Stock candles need an underlying token.\n\n"
                                    "Please set Underlying Token (MSTOCK_UNDERLYING_TOKEN).",
                                ),
                            )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[UI] Underlying token auto-resolve failed: {exc}")
                        # If this is an index underlying and the instruments master is ambiguous
                        # (often only NFO tokens exist), fall back to known NSE spot index tokens.
                        underlying_key = str(strat_cfg.underlying or "").strip().upper()
                        if ":" in underlying_key:
                            underlying_key = underlying_key.split(":", 1)[-1].strip().upper()

                        known_map = {
                            "NIFTY": "26000",
                            "BANKNIFTY": "26009",
                            "FINNIFTY": "26037",
                            "MIDCPNIFTY": "26074",
                        }
                        fallback_tok = known_map.get(underlying_key, "")
                        if fallback_tok:
                            os.environ["MSTOCK_UNDERLYING_TOKEN"] = fallback_tok
                            print(f"[UI] Falling back to known spot index token for {underlying_key}: {fallback_tok}")
                            if hasattr(self, "underlying_token_var"):
                                self.after(0, lambda v=fallback_tok: self.underlying_token_var.set(v))
                            if hasattr(self, "underlying_exchange_var"):
                                self.after(0, lambda v="NSE": self.underlying_exchange_var.set(v))
                        else:
                            exc_msg = str(exc)
                            self.after(
                                0,
                                lambda msg=exc_msg: messagebox.showwarning(
                                    "Underlying token needed",
                                    "m.Stock candles need an underlying token.\n\n"
                                    f"Auto-resolve failed: {msg}\n\n"
                                    "Please set Underlying Token (MSTOCK_UNDERLYING_TOKEN).",
                                ),
                            )

                # Always attach an event sink so the UI can show live MTM/P&L
                # for both paper and live trading modes.
                self.after(0, self._reset_trade_log)

                def event_sink(evt: TradeLogEvent) -> None:  # type: ignore[no-redef]
                    try:
                        self._trade_q.put(evt)
                        print(
                            "[UI] Trade event queued: "
                            f"{evt.event} {evt.trade_id} "
                            f"type={evt.position_type} name={evt.name} "
                            f"legs={len(evt.legs)} mtm={evt.mtm} realized={evt.realized}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[UI] Trade event sink failed: {exc}")

                def on_tick(candles_list) -> None:
                    # Strategy callbacks run in the bot thread; push all UI work
                    # onto the Tk main thread.
                    try:
                        cl = list(candles_list) if candles_list else []
                    except Exception:
                        cl = []

                    # Normalize to Candle objects (renderer expects Candle dataclass).
                    try:
                        norm = self._normalize_candles(cl)
                    except Exception:
                        norm = []

                    def _ui_tick() -> None:
                        try:
                            if norm:
                                self._latest_candles = norm
                                self._latest_candles_ts = float(time.time())
                                try:
                                    last = norm[-1]
                                    self._dash_last_tick_var.set(last.time.strftime("%H:%M:%S"))
                                    self._dash_candles_var.set(str(len(norm)))
                                    # Spot in header should reflect live LTP when available.
                                    # Fall back to candle close only when live spot is missing/stale.
                                    try:
                                        age = float(time.time() - float(self._spot_ltp_live_ts or 0.0))
                                    except Exception:
                                        age = float(self._spot_live_stale_sec) + 1.0
                                    try:
                                        if self._spot_ltp_live is None or age > float(self._spot_live_stale_sec):
                                            self._dash_spot_var.set(f"{float(last.close):.2f}")
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                self._render_signals_and_greeks()
                        except Exception:
                            pass

                    try:
                        self.after(0, _ui_tick)
                    except Exception:
                        return

                scalper = NiftyScalper(client, strat_cfg, event_sink=event_sink, on_tick=on_tick)
                try:
                    self.after(0, lambda s=scalper: setattr(self, "_scalper", s))
                except Exception:
                    self._scalper = scalper
                print("Starting scalper loop...\n")
                scalper.run_forever(stop_event=self._bot_stop)
                print("Scalper loop ended.\n")
            except Exception as exc:  # noqa: BLE001
                print(f"Bot crashed: {exc}\n")
            finally:
                try:
                    self.after(0, lambda: setattr(self, "_scalper", None))
                except Exception:
                    self._scalper = None
                self.after(0, lambda: self.btn_start.configure(state=tk.NORMAL))
                self.after(0, lambda: self.btn_stop.configure(state=tk.DISABLED))
                if hasattr(self, "status_var"):
                    # Reflect final state after the worker stops.
                    self.after(0, lambda: self.status_var.set("Idle"))
                # Force one cleanup on shutdown to avoid timing races with status updates.
                self.after(0, lambda: self._prune_stale_open_rows_when_idle(force=True))

        self._bot_thread = threading.Thread(target=worker, daemon=True)
        self._bot_thread.start()

    def _on_stop(self) -> None:
        self._bot_stop.set()
        self.btn_stop.configure(state=tk.DISABLED)
        if hasattr(self, "status_var"):
            self.status_var.set("Stopping...")

    def _stop_and_restart_bot(self) -> None:
        """Stop the current bot thread and restart once it exits."""

        try:
            if hasattr(self, "status_var"):
                self.status_var.set("Restarting...")
        except Exception:
            pass

        # Signal stop first.
        try:
            self._bot_stop.set()
        except Exception:
            pass

        def _poll() -> None:
            try:
                alive = bool(self._bot_thread and self._bot_thread.is_alive())
            except Exception:
                alive = False

            if alive:
                self.after(200, _poll)
                return

            # Thread is stopped; start a fresh run.
            try:
                self._bot_stop.clear()
            except Exception:
                pass
            try:
                self._on_start()
            except Exception:
                pass

        self.after(200, _poll)

    def _on_close(self) -> None:
        try:
            self._bot_stop.set()
            try:
                if self._ui_log_fp is not None:
                    self._ui_log_fp.write("\n--- UI session ended ---\n")
                    self._ui_log_fp.flush()
                    self._ui_log_fp.close()
            except Exception:
                pass
        finally:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self.destroy()


def main() -> None:
    print("Starting Scalper UI...")
    print("Creating ScalperUI instance...")
    app = ScalperUI()
    print("Scalper UI created, entering mainloop...")
    app.mainloop()
    print("Mainloop exited.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Scalper UI interrupted by user.")
    except Exception as e:
        import traceback
        print("Error launching GUI:", e)
        traceback.print_exc()
