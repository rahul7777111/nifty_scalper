"""
Live Chart tab panel methods — NiftyScalper UI.

Implements a world-class, modern, institutional-grade dark-themed charting dashboard
fully compatible with Tkinter + Matplotlib.
"""

from __future__ import annotations

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import matplotlib.dates as mdates

from live_chart_snapshot import (
    LiveChartSnapshot,
    build_live_chart_snapshot,
    option_chain_to_summary,
    compute_shadow_metrics,
    assess_data_health,
    SignalMarker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# INSTITUTIONAL DARK THEME PALETTE  (Phase 1 - premium terminal spec)
# ---------------------------------------------------------------------------}

# Backgrounds / Borders
BG_DARK   = "#020617"   # App background (near-black)
BG_PANEL  = "#0f172a"   # Panel background
BG_CARD   = "#111827"   # Card / secondary panel
BORDER    = "#1e293b"   # Border
GRID      = "#334155"   # Chart grid

# Text
FG_PRIMARY  = "#e5e7eb"  # High-contrast text
FG_SECONDARY= "#94a3b8"  # Secondary text (alias for FG_MUTED)
FG_MUTED    = "#94a3b8"  # Muted / secondary text
FG_DIM      = "#64748b"  # Very muted / labels
FG_WHITE    = "#ffffff"  # Pure white

# Market / Price colors
C_BULL   = "#22c55e"   # Bull candle / positive
C_BEAR   = "#ef4444"   # Bear candle / negative
C_NEUT   = "#64748b"   # Neutral
C_PRICE  = "#ffffff"   # Current price
C_VWAP   = "#38bdf8"   # VWAP
C_EMA9   = "#facc15"   # EMA 9
C_EMA21  = "#a78bfa"   # EMA 21
C_EMA50  = "#fb923c"   # EMA 50
C_EMA200 = "#e879f9"   # EMA 200
C_ATM    = "#fbbf24"   # ATM strike
C_OPENR  = "#f97316"   # Opening range
C_PDH    = "#eab308"   # Prev day high
C_PDL    = "#eab308"   # Prev day low
C_SESH   = "#10b981"   # Session high
C_SESL   = "#f43f5e"   # Session low

# Signal marker colors
C_BUY     = "#22c55e"  # Executed buy
C_SELL    = "#ef4444"  # Executed sell
C_BLOCKED = "#f59e0b"  # Blocked signal
C_SHADOW  = "#60a5fa"  # Shadow-only signal
C_EXIT    = "#facc15"  # Exit signal
C_DANGER  = "#f43f5e"  # Danger

# Badge status colors
STATUS_OK       = "#22c55e"  # CONNECTED / LOADED / OPEN / HEALTHY
STATUS_WARN     = "#f59e0b"  # WARNING / STALE / DEGRADED
STATUS_CRIT     = "#ef4444"  # CRITICAL / DISCONNECTED / ERROR / NOT_LOADED
STATUS_UNKNOWN  = "#64748b"  # CLOSED / UNKNOWN / N/A
STATUS_SHADOW   = "#3b82f6"  # Shadow mode badge
STATUS_LIVE     = "#22c55e"  # Live mode badge

# ---------------------------------------------------------------------------
# Top-level wire function
# ---------------------------------------------------------------------------

def wire_live_chart_panels(app: Any) -> None:
    """Import this function and call it after live_chart_frame is created.

    Wires up the custom dashboard, and injects methods to app.
    """
    _build_live_chart_tab(app)
    
    # Expose the safe refresh method on ScalperUI
    app.refresh_live_chart_snapshot = lambda snapshot, a=app: refresh_live_chart_snapshot(a, snapshot)


# ---------------------------------------------------------------------------
# Tab construction
# ---------------------------------------------------------------------------

def _build_live_chart_tab(app: Any) -> None:
    """Build the complete Live Chart tab layout."""
    # Clear any default children
    for w in app.live_chart_frame.pack_slaves():
        w.pack_forget()

    # Apply global BG to live_chart_frame using ttk Style for compatibility
    try:
        style = ttk.Style()
        style.configure("LiveChartFrame.TFrame", background=BG_DARK)
        app.live_chart_frame.configure(style="LiveChartFrame.TFrame")
    except Exception:
        try:
            app.live_chart_frame.configure(background=BG_DARK)
        except Exception:
            pass

    # --- TOP STATUS BAR ---
    status_bar = _build_top_status_bar(app)
    status_bar.pack(fill=tk.X, padx=4, pady=(2, 2))

    # --- BOTTOM AREA: Notebook with tabs ---
    try:
        style = ttk.Style()
        style.configure("LiveChart.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("LiveChart.TNotebook.Tab", background=BG_CARD, foreground=FG_SECONDARY, font=("Segoe UI", 8, "bold"), padding=[10, 4])
        style.map("LiveChart.TNotebook.Tab", background=[("selected", BG_DARK)], foreground=[("selected", FG_PRIMARY)])
    except Exception:
        pass

    bottom_notebook = ttk.Notebook(app.live_chart_frame, style="LiveChart.TNotebook")
    bottom_notebook.pack(fill=tk.BOTH, expand=False, padx=4, pady=(0, 4), side=tk.BOTTOM)

    # Bottom Tab 1: Signal Review (contains event timeline)
    signal_review_tab = tk.Frame(bottom_notebook, bg=BG_DARK)
    bottom_notebook.add(signal_review_tab, text="📋 Signal Review")
    timeline_frame = _build_event_timeline(app, signal_review_tab)

    # Bottom Tab 2: Feature Explainability
    explainability_tab = tk.Frame(bottom_notebook, bg=BG_DARK)
    bottom_notebook.add(explainability_tab, text="🧠 Feature Explainability")
    _build_model_explainability_tab(app, explainability_tab)

    # Bottom Tab 3: Shadow Performance Metrics
    shadow_tab = tk.Frame(bottom_notebook, bg=BG_DARK)
    bottom_notebook.add(shadow_tab, text="🔵 Shadow Metrics")
    _build_shadow_performance_tab(app, shadow_tab)

    # Bottom Tab 4: Alerts
    alerts_tab = tk.Frame(bottom_notebook, bg=BG_DARK)
    bottom_notebook.add(alerts_tab, text="🔔 Alerts")
    app._lc_alert_card = _build_alert_card(app, alerts_tab)

    # --- MAIN AREA: chart (left) + right cards (right) ---
    main_pw = tk.PanedWindow(app.live_chart_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4, bg=BG_DARK, bd=0)
    main_pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2), side=tk.TOP)

    # Left: chart area
    chart_area = tk.Frame(main_pw, bg=BG_DARK)
    main_pw.add(chart_area, minsize=400, stretch="always")

    # Right: card stack
    right_cards = tk.Frame(main_pw, bg=BG_DARK)
    main_pw.add(right_cards, minsize=320, stretch="never")

    # Build chart toolbar and re-pack the live_chart_plugin canvas into chart_area below toolbar
    if hasattr(app, "live_chart_plugin") and app.live_chart_plugin:
        toolbar_frame = tk.Frame(chart_area, bg=BG_CARD, bd=1, relief="solid")
        toolbar_frame.pack(fill=tk.X, side=tk.TOP, padx=2, pady=2)
        
        _build_chart_toolbar(app, toolbar_frame, main_pw, right_cards, bottom_notebook)
        
        canvas = app.live_chart_plugin.canvas.get_tk_widget()
        canvas.configure(bg=BG_DARK)
        canvas.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        # Click binding
        canvas.bind("<Button-1>", lambda event: _on_chart_canvas_click(app, event))

        # Keyboard shortcuts — require focus on canvas
        canvas.bind("<Key-R>", lambda e: _on_key_reset_view(app))
        canvas.bind("<Key-r>", lambda e: _on_key_reset_view(app))
        canvas.bind("<Key-L>", lambda e: _on_key_toggle_lock(app))
        canvas.bind("<Key-l>", lambda e: _on_key_toggle_lock(app))
        canvas.bind("<Key-S>", lambda e: _on_key_snapshot(app))
        canvas.bind("<Key-s>", lambda e: _on_key_snapshot(app))
        canvas.bind("<Key-F>", lambda e: _on_key_focus_signal(app))
        canvas.bind("<Key-f>", lambda e: _on_key_focus_signal(app))
        canvas.focus_set()

        # Crosshair overlay — suppress built-in Matplotlib crosshair
        if hasattr(app, "live_chart_plugin"):
            app.live_chart_plugin.disable_crosshair()
        app._lc_crosshair = _CrosshairOverlay(app, canvas)

    # Right panel: single scrollable Market Intelligence column
    app._lc_mi_panel = _build_market_intelligence_panel(right_cards)

    # Store reference
    app._lc_status_bar = status_bar

    # Initial "collecting data" state
    _set_all_cards_collecting(app)


# ---------------------------------------------------------------------------
# Controls / Configuration Cards
# ---------------------------------------------------------------------------

def _build_chart_options_card(app: Any, parent: tk.Frame) -> tk.LabelFrame:
    card = tk.LabelFrame(parent, text="⚙️ Chart Display Options", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    card.pack(fill=tk.X, padx=4, pady=(0, 4))

    # Timeframe selection
    tf_frame = tk.Frame(card, bg=BG_CARD)
    tf_frame.pack(fill=tk.X, pady=2)
    tk.Label(tf_frame, text="Timeframe:", bg=BG_CARD, fg=FG_SECONDARY, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 5))
    app._lc_timeframe_var = tk.StringVar(value="1m")
    tf_combo = ttk.Combobox(tf_frame, textvariable=app._lc_timeframe_var, values=["1m", "3m", "5m", "15m"], width=8, state="readonly")
    tf_combo.pack(side=tk.LEFT)
    
    def on_tf_change(event=None):
        tf = app._lc_timeframe_var.get()
        if hasattr(app, "live_chart_plugin") and app.live_chart_plugin:
            import os
            underlying = "NIFTY"
            scalper = getattr(app, "_scalper", None)
            if scalper is not None and hasattr(scalper, "cfg") and getattr(scalper.cfg, "underlying", None):
                underlying = scalper.cfg.underlying
            else:
                underlying = (os.getenv("MSTOCK_UNDERLYING") or os.getenv("MSTOCK_SYMBOL") or "NIFTY").strip()
            
            app.live_chart_plugin.set_symbol(underlying, tf)
            _refresh_live_chart_tab(app, force=True)
            
    tf_combo.bind("<<ComboboxSelected>>", on_tf_change)

    # Signal Filter
    sf_frame = tk.Frame(card, bg=BG_CARD)
    sf_frame.pack(fill=tk.X, pady=2)
    tk.Label(sf_frame, text="Signal Filter:", bg=BG_CARD, fg=FG_SECONDARY, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 5))
    app._lc_sig_filter_var = tk.StringVar(value="all")
    sf_combo = ttk.Combobox(sf_frame, textvariable=app._lc_sig_filter_var, values=["all", "executed", "blocked", "shadow"], width=10, state="readonly")
    sf_combo.pack(side=tk.LEFT)
    
    def on_sf_change(event=None):
        sf = app._lc_sig_filter_var.get()
        if hasattr(app, "live_chart_plugin") and app.live_chart_plugin:
            app.live_chart_plugin.set_signal_filter(sf)
            _refresh_live_chart_tab(app, force=True)
            
    sf_combo.bind("<<ComboboxSelected>>", on_sf_change)

    # Auto scroll Checkbutton
    app._lc_lock_var = tk.BooleanVar(value=True)
    cb_lock = tk.Checkbutton(card, text="Lock to Live (Auto-scroll)", variable=app._lc_lock_var,
                             command=lambda: app.live_chart_plugin.set_lock_to_live(app._lc_lock_var.get()) if hasattr(app, "live_chart_plugin") else None,
                             bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_DARK, activebackground=BG_CARD, activeforeground=FG_PRIMARY, bd=0)
    cb_lock.pack(anchor="w", pady=2)

    # Overlays Title
    tk.Label(card, text="Overlays & Indicators:", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=FG_PRIMARY).pack(anchor="w", pady=(4, 2))

    # Grid of indicators
    grid = tk.Frame(card, bg=BG_CARD)
    grid.pack(fill=tk.X, pady=2)

    indicators = [
        ("VWAP", "vwap"),
        ("Bollinger Bands", "bollinger"),
        ("Session High/Low", "session_hl"),
        ("Prev Day High/Low", "prevday_hl"),
        ("ATM Strike Line", "atm"),
        ("Volume Panel", "volume"),
        ("EMA Ultra (50)", "ema_ultra"),
        ("EMA Super (200)", "ema_super"),
    ]

    app._lc_overlay_vars = {}
    for idx, (label_text, key) in enumerate(indicators):
        row = idx // 2
        col = idx % 2
        
        var = tk.BooleanVar(value=True)
        app._lc_overlay_vars[key] = var
        
        def toggle_overlay(k=key, v=var):
            if hasattr(app, "live_chart_plugin") and app.live_chart_plugin:
                app.live_chart_plugin.set_overlay_visible(k, v.get())
                _refresh_live_chart_tab(app, force=True)
                
        cb = tk.Checkbutton(grid, text=label_text, variable=var, command=toggle_overlay,
                            bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_DARK, activebackground=BG_CARD, activeforeground=FG_PRIMARY, bd=0)
        cb.grid(row=row, column=col, sticky="w", padx=2, pady=1)

    return card


def _build_interactive_actions_card(app: Any, parent: tk.Frame) -> tk.LabelFrame:
    card = tk.LabelFrame(parent, text="⚡ Manual Actions", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    card.pack(fill=tk.X, padx=4, pady=(0, 4))

    # Force Refresh Button
    btn_refresh = ttk.Button(card, text="🔄 Force UI Redraw", command=lambda: _refresh_live_chart_tab(app, force=True))
    btn_refresh.pack(fill=tk.X, pady=2)

    # Export PNG Button
    def export_chart():
        if not hasattr(app, "live_chart_plugin") or not app.live_chart_plugin:
            import tkinter.messagebox as messagebox
            messagebox.showwarning("Export Chart", "No active chart plugin found!")
            return
        try:
            import os
            os.makedirs("reports", exist_ok=True)
            filename = f"reports/chart_export_{int(time.time())}.png"
            app.live_chart_plugin.export_png(filename)
            import tkinter.messagebox as messagebox
            messagebox.showinfo("Export Chart", f"Chart successfully exported to:\n{filename}")
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Export Chart", f"Failed to export chart: {e}")

    btn_export = ttk.Button(card, text="📸 Export Chart to PNG", command=export_chart)
    btn_export.pack(fill=tk.X, pady=2)

    # Inject Mock Signal Button
    def inject_mock_signal():
        db = getattr(app, "_db_manager", None)
        if db is None:
            scalper = getattr(app, "_scalper", None)
            if scalper is not None:
                db = getattr(scalper, "db_manager", None)
        
        if db is None:
            import tkinter.messagebox as messagebox
            messagebox.showwarning("Inject Signal", "SQLite database manager is not available!")
            return
            
        try:
            import random
            spot = app._spot_ltp_live if hasattr(app, "_spot_ltp_live") else None
            if spot is None or spot <= 0:
                spot = 23450.0
                if hasattr(app, "live_chart_plugin") and app.live_chart_plugin._candles:
                    try:
                        spot = float(app.live_chart_plugin._candles[-1].close)
                    except Exception:
                        pass

            strike = round(spot / 50) * 50
            direction = random.choice(["LONG_CE", "SHORT_PE"])
            side = "BUY" if "LONG" in direction else "SELL"
            prob = random.uniform(0.65, 0.95)
            pred_class = 1 if side == "BUY" else 0
            ts = time.time()
            pred_id = f"mock_{int(ts)}_{random.randint(100, 999)}"

            record = {
                "prediction_id": pred_id,
                "ts": ts,
                "symbol": "NIFTY",
                "exchange": "NFO",
                "underlying_price": spot,
                "direction": direction,
                "regime": "VOLATILITY_BREAKOUT",
                "model_version": "Mock_Ensemble_V2",
                "threshold": 0.50,
                "probability": prob,
                "prediction": 1.0,
                "predicted_class": pred_class,
                "confidence": abs(prob - 0.5) * 2,
                "confidence_bucket": "HIGH",
                "strategy_signal": side,
                "source": "MOCK_INJECT",
                "reason": "UI Manual Signal Injection",
                "trade_candidate": 1,
                "trade_taken": random.choice([0, 1]),
                "no_trade_reason": "" if random.choice([0, 1]) else "Spread threshold exceeded",
            }
            
            db.insert_prediction(record)
            
            if record["trade_taken"] == 1:
                trade_id = f"T_{pred_id}"
                db.link_prediction_to_trade(pred_id, trade_id, trade_taken=True)
                
                legs = [{
                    "symbol": f"NIFTY26JUN{strike}{'CE' if pred_class == 1 else 'PE'}",
                    "side": "BUY",
                    "quantity": 50,
                    "entry_price": 120.0,
                    "ltp": 120.0,
                    "token": "12345",
                    "exchange": "NFO",
                }]
                
                event = {
                    "trade_id": trade_id,
                    "event": "OPEN",
                    "position_type": "options",
                    "name": "Directional Scalping",
                    "ts": ts,
                    "mtm": 0.0,
                    "realized": 0.0,
                    "margin_required": 12000.0,
                    "legs": legs,
                    "symbol": "NIFTY",
                    "strike": strike,
                    "side": "BUY",
                    "quantity": 50,
                    "entry_price": 120.0,
                }
                db.insert_trade_event(event)

            _refresh_live_chart_tab(app, force=True)
            
            import tkinter.messagebox as messagebox
            messagebox.showinfo("Inject Signal", f"Mock signal successfully injected!\nID: {pred_id}\nSide: {side}\nProbability: {prob*100:.1f}%\nDecision: {'ACCEPT' if record['trade_taken'] == 1 else 'BLOCK'}")
            
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Inject Signal", f"Failed to inject mock signal: {e}")

    btn_inject = ttk.Button(card, text="🧪 Inject Mock ML Signal", command=inject_mock_signal)
    btn_inject.pack(fill=tk.X, pady=2)

    return card


# ---------------------------------------------------------------------------
# TOP STATUS BAR
# ---------------------------------------------------------------------------

def _build_top_status_bar(app: Any) -> tk.Frame:
    """Build a premium badge-style institutional status header at the top of the chart.

    Layout: [LEFT: symbol/tf/spot/fut/atm/change%] [MIDDLE: regime/badges] [RIGHT: last candle/stale/api/refresh]
    Badge color logic:
      GREEN  = OK / CONNECTED / LOADED / OPEN / HEALTHY
      AMBER  = WARNING / STALE / DEGRADED
      RED    = CRITICAL / DISCONNECTED / ERROR / NOT_LOADED
      GRAY   = CLOSED / UNKNOWN / N/A
    """
    frame = tk.Frame(app.live_chart_frame, bg=BG_PANEL, height=42, bd=1, relief="solid", highlightthickness=0)
    frame.pack(fill=tk.X, padx=0, pady=0)
    frame.pack_propagate(False)

    # Three sections
    left_sec  = tk.Frame(frame, bg=BG_PANEL)
    mid_sec   = tk.Frame(frame, bg=BG_PANEL)
    right_sec = tk.Frame(frame, bg=BG_PANEL)
    left_sec.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
    mid_sec.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=3)
    right_sec.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=3)

    # ── Variables ──────────────────────────────────────────────
    app._lc_spot_var        = tk.StringVar(value="--")
    app._lc_fut_var         = tk.StringVar(value="--")
    app._lc_atm_var         = tk.StringVar(value="--")
    app._lc_change_var      = tk.StringVar(value="--")
    app._lc_regime_var      = tk.StringVar(value="--")
    app._lc_last_candle_var = tk.StringVar(value="--")
    app._lc_stale_var       = tk.StringVar(value="")
    app._lc_api_err_var     = tk.StringVar(value="API: 0")
    app._lc_refresh_var     = tk.StringVar(value="--")

    # Backward-compat fallbacks (read by existing refresh code)
    app._lc_market_status_var  = tk.StringVar(value="MARKET: --")
    app._lc_broker_var         = tk.StringVar(value="Broker: --")
    app._lc_model_var          = tk.StringVar(value="Model: --")
    app._lc_shadow_var         = tk.StringVar(value="🔵 SHADOW")
    app._lc_last_update_var    = tk.StringVar(value="Updated: --")
    app._lc_pcr_var            = tk.StringVar(value="PCR: --")
    app._lc_iv_var             = tk.StringVar(value="IV: --")
    app._lc_signal_var         = tk.StringVar(value="Signal: --")

    # ── Helper: make one badge label ────────────────────────────
    def _badge(parent, text, key) -> tk.Label:
        lbl = tk.Label(parent, text=text, font=("Segoe UI", 8, "bold"),
                       fg=FG_WHITE, bg=STATUS_UNKNOWN,
                       padx=6, pady=1, relief="solid", bd=1)
        lbl.pack(side=tk.LEFT, padx=2)
        setattr(app, f"_lc_badge_{key}", lbl)
        return lbl

    def _info_cell(parent, label_text, text_var, accent: str = "") -> None:
        """Compact label pair: muted 'LABEL' above bold value."""
        cell = tk.Frame(parent, bg=BG_PANEL)
        cell.pack(side=tk.LEFT, padx=5)
        tk.Label(cell, text=label_text, font=("Segoe UI", 7), fg=FG_DIM, bg=BG_PANEL).pack(anchor="w")
        tk.Label(cell, textvariable=text_var, font=("Segoe UI", 9, "bold"),
                 fg=accent if accent else FG_PRIMARY, bg=BG_PANEL).pack(anchor="w")

    # ── LEFT: symbol + price info ────────────────────────────────
    tk.Label(left_sec, text="NIFTY · 1m", font=("Segoe UI", 9, "bold"),
             fg=FG_PRIMARY, bg=BORDER, padx=7, pady=2, relief="solid", bd=1).pack(side=tk.LEFT, padx=(0, 6))
    _info_cell(left_sec, "SPOT",   app._lc_spot_var)
    _info_cell(left_sec, "FUT",    app._lc_fut_var)
    _info_cell(left_sec, "ATM",    app._lc_atm_var,   C_ATM)
    _info_cell(left_sec, "CHG",    app._lc_change_var)

    # ── MIDDLE: regime + status badges ──────────────────────────
    _info_cell(mid_sec, "REGIME", app._lc_regime_var, C_EMA21)
    _badge(mid_sec, "MARKET: --",    "market")
    _badge(mid_sec, "BROKER: --",    "broker")
    _badge(mid_sec, "MODEL: --",     "model")
    _badge(mid_sec, "HEALTH: --",    "health")
    _badge(mid_sec, "SHADOW",        "shadow")

    # ── RIGHT: data freshness + errors ──────────────────────────
    _info_cell(right_sec, "LAST CANDLE", app._lc_last_candle_var)
    app._lbl_stale = tk.Label(right_sec, textvariable=app._lc_stale_var,
                              font=("Segoe UI", 8, "bold"), fg=STATUS_WARN, bg=BG_PANEL)
    app._lbl_stale.pack(side=tk.LEFT, padx=4)
    _info_cell(right_sec, "API ERR", app._lc_api_err_var)
    _info_cell(right_sec, "REFRESH", app._lc_refresh_var)

    return frame


def _update_badge_colors(app: Any, mkt: str, broker_status: str, model_status: str,
                          health_status: str, shadow_mode: bool) -> None:
    """Update badge backgrounds using the institutional status color palette."""
    try:
        # Market status
        mkt_bg = {"OPEN": STATUS_OK, "PRE_OPEN": STATUS_WARN}.get(mkt, STATUS_UNKNOWN)
        app._lc_badge_market.configure(text=f"MARKET: {mkt}", bg=mkt_bg)

        # Broker
        broker_bg = STATUS_OK if broker_status == "CONNECTED" else STATUS_CRIT
        app._lc_badge_broker.configure(text=f"BROKER: {broker_status}", bg=broker_bg)

        # Model
        model_bg = STATUS_OK if model_status == "LOADED" else STATUS_CRIT
        app._lc_badge_model.configure(text=f"MODEL: {model_status}", bg=model_bg)

        # Health
        health_bg = {"OK": STATUS_OK, "WARNING": STATUS_WARN, "CRITICAL": STATUS_CRIT}.get(
            health_status, STATUS_UNKNOWN)
        app._lc_badge_health.configure(text=f"HEALTH: {health_status}", bg=health_bg)

        # Shadow mode
        shadow_bg  = STATUS_SHADOW if shadow_mode else STATUS_LIVE
        shadow_txt = "SHADOW" if shadow_mode else "LIVE"
        app._lc_badge_shadow.configure(text=shadow_txt, bg=shadow_bg)

    except Exception:
        pass


# ---------------------------------------------------------------------------
# CURRENT PREDICTION CARD
# ---------------------------------------------------------------------------

def _build_current_prediction_card(app: Any, parent: tk.Frame) -> tk.LabelFrame:
    card = tk.LabelFrame(parent, text="📊 Current Prediction", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    card.pack(fill=tk.X, padx=4, pady=(0, 4))

    app._lc_pred_direction_var  = tk.StringVar(value="--")
    app._lc_pred_probability_var = tk.StringVar(value="Prob: --")
    app._lc_pred_confidence_var  = tk.StringVar(value="Conf: --")
    app._lc_pred_threshold_var   = tk.StringVar(value="Thresh: --")
    app._lc_pred_model_var       = tk.StringVar(value="Model: --")
    app._lc_pred_time_var        = tk.StringVar(value="Time: --")
    app._lc_pred_edge_var        = tk.StringVar(value="Edge: --")
    app._lc_pred_regime_var      = tk.StringVar(value="Regime: --")
    app._lc_pred_decision_var    = tk.StringVar(value="Decision: --")
    app._lc_pred_liq_var         = tk.StringVar(value="Liquidity: --")
    app._lc_pred_spread_var      = tk.StringVar(value="Spread: --")

    for text_var in (
        app._lc_pred_direction_var,
        app._lc_pred_probability_var,
        app._lc_pred_confidence_var,
        app._lc_pred_threshold_var,
        app._lc_pred_model_var,
        app._lc_pred_time_var,
        app._lc_pred_edge_var,
        app._lc_pred_regime_var,
        app._lc_pred_liq_var,
        app._lc_pred_spread_var,
        app._lc_pred_decision_var,
    ):
        lbl = tk.Label(card, textvariable=text_var, font=("Segoe UI", 8), bg=BG_CARD, fg=FG_SECONDARY, anchor="w")
        lbl.pack(anchor="w", padx=2, pady=0)

    # Decision badge (ACCEPT / BLOCK) — coloured label
    app._lc_pred_decision_lbl = tk.Label(card, text="", font=("Segoe UI", 9, "bold"),
                                         anchor="center", relief=tk.RIDGE, bd=1, padx=4, pady=2, bg=BG_CARD)
    app._lc_pred_decision_lbl.pack(fill=tk.X, pady=(4, 0))

    return card


# ---------------------------------------------------------------------------
# SHADOW METRICS CARD (horizontal layout in Bottom Notebook)
# ---------------------------------------------------------------------------

def _build_shadow_performance_tab(app: Any, parent: tk.Frame) -> None:
    """Premium Shadow Performance tab: metrics cards + mini equity curve."""
    frame = tk.Frame(parent, bg=BG_DARK)
    frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Metrics grid (top) ─────────────────────────────────────────────
    grid_frame = tk.Frame(frame, bg=BG_CARD)
    grid_frame.pack(fill=tk.X, pady=(0, 4))

    app._lc_shad_preds_var    = tk.StringVar(value="--")
    app._lc_shad_accepted_var = tk.StringVar(value="--")
    app._lc_shad_blocked_var  = tk.StringVar(value="--")
    app._lc_shad_resolved_var = tk.StringVar(value="--")
    app._lc_shad_winrate_var  = tk.StringVar(value="--")
    app._lc_shad_pf_var       = tk.StringVar(value="--")
    app._lc_shad_exp_var      = tk.StringVar(value="--")
    app._lc_shad_mdd_var      = tk.StringVar(value="--")
    app._lc_shad_pnl_var      = tk.StringVar(value="--")
    app._lc_shad_drift_var    = tk.StringVar(value="--")
    app._lc_shad_calib_var    = tk.StringVar(value="--")
    app._lc_shad_edge_var     = tk.StringVar(value="--")

    metrics = [
        ("Predictions",    app._lc_shad_preds_var,    0, 0),
        ("Accepted",       app._lc_shad_accepted_var, 0, 1),
        ("Blocked",        app._lc_shad_blocked_var,  0, 2),
        ("Resolved",       app._lc_shad_resolved_var, 0, 3),
        ("Win Rate",       app._lc_shad_winrate_var,  0, 4),
        ("Profit Factor",  app._lc_shad_pf_var,       1, 0),
        ("Expectancy",     app._lc_shad_exp_var,      1, 1),
        ("Max Drawdown",   app._lc_shad_mdd_var,      1, 2),
        ("Daily PnL",      app._lc_shad_pnl_var,      1, 3),
        ("Drift",          app._lc_shad_drift_var,    1, 4),
    ]
    for label_text, text_var, r, c in metrics:
        cell = tk.Frame(grid_frame, bg=BG_DARK, bd=1, relief="solid", highlightthickness=0)
        cell.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
        tk.Label(cell, text=label_text, font=("Segoe UI", 7), bg=BG_DARK,
                 fg=FG_SECONDARY).pack(anchor="center", pady=(3, 1))
        tk.Label(cell, textvariable=text_var, font=("Segoe UI", 9, "bold"),
                 bg=BG_DARK, fg=FG_PRIMARY).pack(anchor="center", pady=(0, 3))
    for col in range(5):
        grid_frame.columnconfigure(col, weight=1)
    for row in range(2):
        grid_frame.rowconfigure(row, weight=1)

    # ── Mini equity curve canvas (bottom) ──────────────────────────────
    eq_card = tk.LabelFrame(frame, text="Equity Curve", bg=BG_CARD, fg=FG_PRIMARY,
                            font=("Segoe UI", 8, "bold"), bd=1, relief="solid", padx=4, pady=4)
    eq_card.pack(fill=tk.X, pady=(0, 4))
    app._lc_equity_canvas = tk.Canvas(eq_card, width=700, height=80,
                                       bg=BG_DARK, highlightthickness=0)
    app._lc_equity_canvas.pack(fill=tk.X)
    app._lc_equity_canvas_refs: list = []   # stored line ids for fast redraw


def _refresh_shadow_performance(app: Any, snap: LiveChartSnapshot) -> None:
    """Refresh Shadow Performance tab from snapshot."""
    try:
        sm = snap.shadow_metrics
        if sm:
            def s(v, fmt="--"):
                return fmt.format(v) if v is not None else "--"
            app._lc_shad_preds_var.set(s(sm.total_predictions, "{}"))
            app._lc_shad_accepted_var.set(s(sm.accepted_predictions, "{}"))
            app._lc_shad_blocked_var.set(s(sm.blocked_predictions, "{}"))
            app._lc_shad_resolved_var.set(s(sm.resolved_count, "{}/{}").format(
                sm.resolved_count or 0, (sm.pending_count or 0)))
            wr = sm.win_rate
            app._lc_shad_winrate_var.set(f"{wr:.1%}" if wr is not None else "--")
            pf = sm.profit_factor
            pf_fg = C_BULL if (pf is not None and pf >= 1.0) else (STATUS_WARN if pf else FG_PRIMARY)
            app._lc_shad_pf_var.set(f"{pf:.2f}" if pf is not None else "--")
            exp = sm.expectancy
            app._lc_shad_exp_var.set(f"{exp:.3f}" if exp is not None else "--")
            mdd = sm.max_drawdown
            app._lc_shad_mdd_var.set(f"{mdd:.2%}" if mdd is not None else "--")
            pnl = sm.total_pnl
            app._lc_shad_pnl_var.set(f"{pnl:+.2f}" if pnl is not None else "--")
            drift = sm.drift_detected
            drift_txt = "⚠ DRIFT" if drift else "✓ OK"
            app._lc_shad_drift_var.set(drift_txt)
            calib = sm.calibration_error
            app._lc_shad_calib_var.set(f"{calib:.3f}" if calib else "--")
            edge = sm.avg_edge
            app._lc_shad_edge_var.set(f"{edge:+.3f}" if edge else "--")

        # ── Equity curve ──────────────────────────────────────────────
        canvas = getattr(app, "_lc_equity_canvas", None)
        if canvas:
            canvas.delete("all")
            rows = getattr(app, "_lc_all_timeline_rows", []) or []
            resolved = [r for r in rows if r.get("trade_taken") and r.get("pnl") is not None]
            if len(resolved) < 2:
                canvas.create_text(350, 40, text="Not enough resolved trades for equity curve",
                                   font=("Segoe UI", 8), fill=FG_DIM, anchor="center")
                return

            pnls = [float(r["pnl"]) for r in resolved]
            equity = []
            cumulative = 0.0
            for p in pnls:
                cumulative += p
                equity.append(cumulative)

            W, H = 700, 80
            pad_x, pad_y = 8, 8
            chart_w = W - 2 * pad_x
            chart_h = H - 2 * pad_y

            eq_min, eq_max = min(equity), max(equity)
            eq_range = eq_max - eq_min if eq_max != eq_min else 1

            def to_xy(i, val):
                x = pad_x + (i / (len(equity) - 1)) * chart_w
                y = pad_y + (1 - (val - eq_min) / eq_range) * chart_h
                return x, y

            # Zero line
            if eq_min < 0 < eq_max:
                zero_y = pad_y + (1 - (0 - eq_min) / eq_range) * chart_h
                canvas.create_line(pad_x, zero_y, pad_x + chart_w, zero_y,
                                   fill=FG_DIM, dash=(3, 3), width=1)

            # Curve color
            curve_color = C_BULL if equity[-1] >= 0 else C_BEAR

            coords = []
            for i, val in enumerate(equity):
                x, y = to_xy(i, val)
                coords.extend([x, y])
            canvas.create_line(*coords, fill=curve_color, width=1.5, smooth=True)

            # Current PnL label
            lbl_x, lbl_y = to_xy(len(equity) - 1, equity[-1])
            canvas.create_text(lbl_x + 4, lbl_y, text=f"{equity[-1]:+.1f}",
                               font=("Segoe UI", 7, "bold"), fill=curve_color, anchor="w")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OPTION CHAIN SUMMARY CARD (mini market internals panel)
# ---------------------------------------------------------------------------

def _build_option_chain_card(app: Any, parent: tk.Frame) -> tk.LabelFrame:
    card = tk.LabelFrame(parent, text="📈 Option Chain Internals", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    card.pack(fill=tk.X, padx=4, pady=(0, 4))

    app._lc_oc_atm_var       = tk.StringVar(value="ATM: --")
    app._lc_oc_ce_ltp_var    = tk.StringVar(value="CE LTP: --")
    app._lc_oc_pe_ltp_var    = tk.StringVar(value="PE LTP: --")
    app._lc_oc_ce_iv_var     = tk.StringVar(value="CE IV: --")
    app._lc_oc_pe_iv_var     = tk.StringVar(value="PE IV: --")
    app._lc_oc_ce_oi_var     = tk.StringVar(value="CE OI: --")
    app._lc_oc_pe_oi_var     = tk.StringVar(value="PE OI: --")
    app._lc_oc_pcr_var       = tk.StringVar(value="PCR: --")
    app._lc_oc_spread_var    = tk.StringVar(value="Spread: --")
    app._lc_oc_liq_var       = tk.StringVar(value="Liquidity: --")
    app._lc_oc_warning_var   = tk.StringVar(value="")

    for text_var in (
        app._lc_oc_atm_var,
        app._lc_oc_ce_ltp_var,
        app._lc_oc_pe_ltp_var,
        app._lc_oc_ce_iv_var,
        app._lc_oc_pe_iv_var,
        app._lc_oc_ce_oi_var,
        app._lc_oc_pe_oi_var,
        app._lc_oc_pcr_var,
        app._lc_oc_spread_var,
        app._lc_oc_liq_var,
    ):
        lbl = tk.Label(card, textvariable=text_var, font=("Segoe UI", 8), bg=BG_CARD, fg=FG_SECONDARY, anchor="w")
        lbl.pack(anchor="w", padx=2, pady=0)

    # Highlighted warning row
    app._lbl_oc_warning = tk.Label(card, textvariable=app._lc_oc_warning_var, font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=FG_SECONDARY, anchor="w")
    app._lbl_oc_warning.pack(anchor="w", padx=2, pady=0)

    return card


# ---------------------------------------------------------------------------
# DATA HEALTH CARD
# ---------------------------------------------------------------------------

def _build_data_health_card(app: Any, parent: tk.Frame) -> tk.LabelFrame:
    card = tk.LabelFrame(parent, text="💠 Data Health", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    card.pack(fill=tk.X, padx=4, pady=(0, 4))

    app._lc_dh_broker_var    = tk.StringVar(value="Broker: --")
    app._lc_dh_candle_age_var = tk.StringVar(value="Last Candle: --")
    app._lc_dh_tick_age_var  = tk.StringVar(value="Last Tick: --")
    app._lc_dh_stale_sec_var = tk.StringVar(value="Stale Seconds: --")
    app._lc_dh_model_var     = tk.StringVar(value="Model: --")
    app._lc_dh_shadow_var    = tk.StringVar(value="Shadow Log: --")
    app._lc_dh_db_var        = tk.StringVar(value="DB Write: --")
    app._lc_dh_missing_var   = tk.StringVar(value="Missing Candles: --")
    app._lc_dh_api_err_var   = tk.StringVar(value="API Errors: --")
    app._lc_dh_reason_var    = tk.StringVar(value="Reason: --")
    app._lc_dh_status_var    = tk.StringVar(value="Status: --")

    for text_var in (
        app._lc_dh_broker_var,
        app._lc_dh_candle_age_var,
        app._lc_dh_tick_age_var,
        app._lc_dh_stale_sec_var,
        app._lc_dh_model_var,
        app._lc_dh_shadow_var,
        app._lc_dh_db_var,
        app._lc_dh_missing_var,
        app._lc_dh_api_err_var,
        app._lc_dh_reason_var,
        app._lc_dh_status_var,
    ):
        lbl = tk.Label(card, textvariable=text_var, font=("Segoe UI", 8), bg=BG_CARD, fg=FG_SECONDARY, anchor="w")
        lbl.pack(anchor="w", padx=2, pady=0)

    # Status badge
    app._lc_dh_status_badge = tk.Label(card, text="", font=("Segoe UI", 8, "bold"),
                                        anchor="center", relief=tk.RIDGE, bd=1, padx=4, pady=2)
    app._lc_dh_status_badge.pack(fill=tk.X, pady=(4, 0))

    # Trading safety guidance
    safety_frame = tk.Frame(card, bg=BG_DARK, bd=1, relief="solid", padx=6, pady=4)
    safety_frame.pack(fill=tk.X, pady=(6, 0))
    app._lc_trading_safety_var = tk.StringVar(value="Trading Safety: --")
    app._lc_trading_safety_lbl = tk.Label(safety_frame, textvariable=app._lc_trading_safety_var,
                                           font=("Segoe UI", 9, "bold"), bg=BG_DARK,
                                           fg=FG_PRIMARY, anchor="center")
    app._lc_trading_safety_lbl.pack(fill=tk.X)

    return card


# ---------------------------------------------------------------------------
# ALERT CARD
# ---------------------------------------------------------------------------

def _build_alert_card(app: Any, parent: tk.Frame) -> tk.LabelFrame:
    card = tk.LabelFrame(parent, text="🔔 Active Alerts", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    app._lc_alerts_container = tk.Frame(card, bg=BG_CARD)
    app._lc_alerts_container.pack(fill=tk.BOTH, expand=True)
    app._lc_active_alerts: list[dict[str, Any]] = []

    app._lc_no_alerts_var = tk.StringVar(value="No active alerts")
    lbl = tk.Label(app._lc_alerts_container, textvariable=app._lc_no_alerts_var,
                   font=("Segoe UI", 9), foreground="gray", bg=BG_CARD, anchor="w")
    lbl.pack(anchor="w", padx=10, pady=10)

    return card


# ---------------------------------------------------------------------------
# EVENT TIMELINE
# ---------------------------------------------------------------------------

def _build_event_timeline(app: Any, parent: tk.Frame) -> tk.Frame:
    """Build the event timeline container within the Signal Review tab.

    Wraps the Signal Tape in a labelled frame and returns the container.
    """
    container = tk.LabelFrame(
        parent,
        text="📋 Event Timeline",
        bg=BG_CARD,
        fg=FG_PRIMARY,
        font=("Segoe UI", 9, "bold"),
        bd=1,
        relief="solid",
        padx=4,
        pady=4,
    )
    container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    inner = tk.Frame(container, bg=BG_CARD)
    inner.pack(fill=tk.BOTH, expand=True)

    _build_signal_tape(app, inner)

    return container


# ---------------------------------------------------------------------------
# EVENT TIMELINE TABLE
# ---------------------------------------------------------------------------

def _build_signal_tape(app: Any, parent: tk.Frame) -> None:
    """Premium Signal Tape: sortable table of signals with color-tagged rows."""
    frame = tk.Frame(parent, bg=BG_DARK)
    frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Filter bar
    filter_bar = tk.Frame(frame, bg=BG_DARK)
    filter_bar.pack(fill=tk.X, pady=(0, 4))

    app._lc_timeline_filter_var = tk.StringVar(value="all")
    filter_options = [("All", "all"), ("Signals", "signals"), ("Blocks", "blocks"),
                      ("Exits", "exits"), ("Shadow", "shadow")]
    for label_text, value in filter_options:
        b = tk.Radiobutton(filter_bar, text=label_text, variable=app._lc_timeline_filter_var,
                           value=value, font=("Segoe UI", 8),
                           bg=BG_DARK, fg=FG_PRIMARY, selectcolor=BG_CARD,
                           activebackground=BG_DARK, activeforeground=FG_PRIMARY, bd=0,
                           command=lambda v=value: _on_timeline_filter(app, v))
        b.pack(side=tk.LEFT, padx=(0, 6))

    # Count label
    app._lc_tape_count_var = tk.StringVar(value="0 signals")
    tk.Label(filter_bar, textvariable=app._lc_tape_count_var, font=("Segoe UI", 8),
             fg=FG_DIM, bg=BG_DARK).pack(side=tk.RIGHT)

    # Treeview
    tree_frame = tk.Frame(frame, bg=BG_DARK)
    tree_frame.pack(fill=tk.BOTH, expand=True)

    style = ttk.Style()
    style.configure("SignalTape.Treeview", background=BG_CARD, fieldbackground=BG_CARD,
                   foreground=FG_PRIMARY, borderwidth=0, font=("Segoe UI", 8))
    style.configure("SignalTape.Treeview.Heading", background=BG_PANEL, foreground=FG_MUTED,
                   borderwidth=1, font=("Segoe UI", 8, "bold"))
    style.map("SignalTape.Treeview", background=[("selected", BORDER)])

    cols = ("time", "side", "cp", "strike", "prob", "edge", "status", "reason", "pnl")
    app._lc_timeline_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                         height=6, selectmode="browse", style="SignalTape.Treeview")
    col_config = {
        "time":   ("Time",   70),
        "side":   ("Side",   45),
        "cp":     ("CE/PE",  45),
        "strike": ("Strike", 60),
        "prob":   ("Prob",   45),
        "edge":   ("Edge",   45),
        "status": ("Status", 65),
        "reason": ("Reason", 200),
        "pnl":    ("PnL",    55),
    }
    for col, (text, width) in col_config.items():
        app._lc_timeline_tree.heading(col, text=text,
                                      command=lambda c=col: _sort_timeline(app, c))
        app._lc_timeline_tree.column(col, width=width, stretch=True, anchor="center")

    vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=app._lc_timeline_tree.yview)
    app._lc_timeline_tree.configure(yscrollcommand=vsb.set)
    app._lc_timeline_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.LEFT, fill=tk.Y)

    # Click-to-focus
    app._lc_timeline_tree.bind("<Button-1>", lambda e: _on_timeline_row_click(app))

    # Tag colors matching institutional spec
    tree = app._lc_timeline_tree
    tree.tag_configure("buy",    background="", foreground=C_BUY)
    tree.tag_configure("sell",   background="", foreground=C_SELL)
    tree.tag_configure("exit",   background="", foreground=C_EXIT)
    tree.tag_configure("blocked",background="", foreground=C_BLOCKED)
    tree.tag_configure("shadow", background="", foreground=C_SHADOW)

    app._lc_tape_sort_col = None
    app._lc_tape_sort_rev = False


def _sort_timeline(app: Any, col: str) -> None:
    """Sort Signal Tape by column (asc/desc toggle)."""
    rows = getattr(app, "_lc_all_timeline_rows", []) or []
    if app._lc_tape_sort_col == col:
        app._lc_tape_sort_rev = not app._lc_tape_sort_rev
    else:
        app._lc_tape_sort_col = col
        app._lc_tape_sort_rev = False
    # Rebuild with sorted rows
    _refresh_timeline_from_rows(app, rows)


def _on_timeline_row_click(app: Any) -> None:
    """Focus the selected signal marker on the chart."""
    tree = getattr(app, "_lc_timeline_tree", None)
    if not tree:
        return
    sel = tree.selection()
    if not sel:
        return
    row_id = tree.identify_row(tree.winfo_y())
    # Get the stored row data from tag
    rows = getattr(app, "_lc_all_timeline_rows", []) or []
    idx = tree.index(sel[0])
    if idx < len(rows):
        row = rows[idx]
        plugin = getattr(app, "live_chart_plugin", None)
        if plugin:
            tid = str(row.get("prediction_id", "") or row.get("trade_id", "") or "")
            if tid:
                plugin.focus_trade(tid)


def _on_timeline_filter(app: Any, filter_value: str) -> None:
    app._lc_timeline_filter_var.set(filter_value)
    _refresh_timeline_from_rows(app, getattr(app, "_lc_all_timeline_rows", []))


# ---------------------------------------------------------------------------
# MODEL EXPLAINABILITY TAB
# ---------------------------------------------------------------------------

def _build_model_explainability_tab(app: Any, parent: tk.Frame) -> None:
    """Premium Model Explainability panel with probability gauge, direction, confidence, contributors."""
    frame = tk.Frame(parent, bg=BG_DARK)
    frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Row 1: Probability gauge + Direction + Confidence ────────────
    top = tk.Frame(frame, bg=BG_DARK)
    top.pack(fill=tk.X, pady=(0, 6))

    # Probability gauge canvas (left)
    app._lc_prob_gauge = tk.Canvas(top, width=200, height=80, bg=BG_DARK, highlightthickness=0)
    app._lc_prob_gauge.pack(side=tk.LEFT, padx=(0, 8))

    # Direction badge + Confidence (middle)
    dir_frame = tk.Frame(top, bg=BG_CARD, padx=10, pady=6)
    dir_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

    app._lc_prob_val_var  = tk.StringVar(value="--")
    app._lc_prob_dir_var  = tk.StringVar(value="Direction: --")
    app._lc_prob_conf_var = tk.StringVar(value="Conf: --")
    app._lc_prob_thresh_var = tk.StringVar(value="Thresh: --")
    app._lc_prob_model_var = tk.StringVar(value="Model: --")
    app._lc_prob_edge_var  = tk.StringVar(value="Edge: --")
    app._lc_prob_decision_var = tk.StringVar(value="Decision: --")

    app._lc_prob_dir_lbl = tk.Label(dir_frame, textvariable=app._lc_prob_dir_var,
                                   font=("Segoe UI", 11, "bold"), bg=BG_CARD, fg=FG_PRIMARY)
    app._lc_prob_dir_lbl.pack(anchor="w", pady=(0, 4))
    app._lc_prob_decision_lbl = tk.Label(dir_frame, textvariable=app._lc_prob_decision_var,
                                         font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_PRIMARY,
                                         relief=tk.RIDGE, padx=6, pady=2)
    app._lc_prob_decision_lbl.pack(anchor="w", pady=(0, 4))
    for var in [app._lc_prob_val_var, app._lc_prob_conf_var,
                app._lc_prob_thresh_var, app._lc_prob_model_var, app._lc_prob_edge_var]:
        tk.Label(dir_frame, textvariable=var, font=("Segoe UI", 8),
                 fg=FG_MUTED, bg=BG_CARD, anchor="w").pack(anchor="w", pady=1)

    # Confidence bucket
    conf_frame = tk.Frame(top, bg=BG_CARD, padx=10, pady=6)
    conf_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

    tk.Label(conf_frame, text="Confidence Bucket", font=("Segoe UI", 8, "bold"),
             fg=FG_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0, 4))
    app._lc_prob_bucket_var = tk.StringVar(value="--")
    app._lc_prob_bucket_lbl = tk.Label(conf_frame, textvariable=app._lc_prob_bucket_var,
                                      font=("Segoe UI", 13, "bold"), bg=BG_CARD, fg=FG_PRIMARY)
    app._lc_prob_bucket_lbl.pack(anchor="w")

    # Warnings (right)
    warn_frame = tk.Frame(top, bg=BG_CARD, padx=10, pady=6)
    warn_frame.pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(warn_frame, text="Alerts", font=("Segoe UI", 8, "bold"),
             fg=FG_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0, 4))
    app._lc_prob_calib_var = tk.StringVar(value="Calibration: --")
    app._lc_prob_drift_var = tk.StringVar(value="Drift: --")
    app._lc_prob_warn_lbl = tk.Label(warn_frame, textvariable=app._lc_prob_calib_var,
                                    font=("Segoe UI", 8), bg=BG_CARD, fg=FG_MUTED)
    app._lc_prob_warn_lbl.pack(anchor="w", pady=1)
    app._lc_prob_drift_lbl = tk.Label(warn_frame, textvariable=app._lc_prob_drift_var,
                                     font=("Segoe UI", 8), bg=BG_CARD, fg=FG_MUTED)
    app._lc_prob_drift_lbl.pack(anchor="w", pady=1)

    # ── Row 2: Top contributors ──────────────────────────────────────
    contrib_frame = tk.Frame(frame, bg=BG_DARK)
    contrib_frame.pack(fill=tk.BOTH, expand=True)

    # Positive contributors (left)
    pos_card = tk.LabelFrame(contrib_frame, text="▲ Top Positive Contributors",
                             bg=BG_CARD, fg=C_BULL, font=("Segoe UI", 8, "bold"),
                             bd=1, relief="solid", padx=6, pady=4)
    pos_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
    app._lc_pos_contrib_tree = ttk.Treeview(pos_card, columns=("feat", "val"),
                                            show="headings", height=6, selectmode="none")
    app._lc_pos_contrib_tree.heading("feat", text="Feature")
    app._lc_pos_contrib_tree.column("feat", width=200, anchor="w")
    app._lc_pos_contrib_tree.heading("val", text="Contribution")
    app._lc_pos_contrib_tree.column("val", width=100, anchor="e")
    app._lc_pos_contrib_tree.pack(fill=tk.BOTH, expand=True)

    # Negative contributors (right)
    neg_card = tk.LabelFrame(contrib_frame, text="▼ Top Negative Contributors",
                             bg=BG_CARD, fg=C_BEAR, font=("Segoe UI", 8, "bold"),
                             bd=1, relief="solid", padx=6, pady=4)
    neg_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
    app._lc_neg_contrib_tree = ttk.Treeview(neg_card, columns=("feat", "val"),
                                            show="headings", height=6, selectmode="none")
    app._lc_neg_contrib_tree.heading("feat", text="Feature")
    app._lc_neg_contrib_tree.column("feat", width=200, anchor="w")
    app._lc_neg_contrib_tree.heading("val", text="Contribution")
    app._lc_neg_contrib_tree.column("val", width=100, anchor="e")
    app._lc_neg_contrib_tree.pack(fill=tk.BOTH, expand=True)


def _draw_probability_gauge(app: Any, prob: float, direction: str) -> None:
    """Draw a horizontal probability bar gauge on the gauge canvas."""
    canvas = getattr(app, "_lc_prob_gauge", None)
    if canvas is None:
        return
    canvas.delete("all")
    w, h = 200, 80
    bar_h = 18
    bar_y = 50
    bar_x = 10
    bar_w = w - 20

    # Background track
    canvas.create_rectangle(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h,
                            fill=BG_DARK, outline=BORDER, width=1)

    # Filled portion (prob * bar_w)
    fill_w = max(0, min(bar_w, int(prob * bar_w)))
    color = C_BULL if direction in ("BUY", "LONG") else (C_BEAR if direction in ("SELL", "SHORT") else FG_DIM)
    if fill_w > 0:
        canvas.create_rectangle(bar_x, bar_y, bar_x + fill_w, bar_y + bar_h,
                                fill=color, outline="")

    # Threshold line at 0.5
    thresh_x = bar_x + int(0.5 * bar_w)
    canvas.create_line(thresh_x, bar_y - 2, thresh_x, bar_y + bar_h + 2,
                       fill=FG_DIM, width=1, dash=(3, 2))

    # Labels
    canvas.create_text(bar_x, bar_y - 8, text="0", font=("Segoe UI", 7), fill=FG_DIM, anchor="w")
    canvas.create_text(bar_x + bar_w, bar_y - 8, text="1", font=("Segoe UI", 7), fill=FG_DIM, anchor="e")
    canvas.create_text(bar_x + bar_w // 2, bar_y - 8, text="0.5", font=("Segoe UI", 7), fill=FG_DIM, anchor="n")

    # Probability text
    canvas.create_text(w // 2, bar_y - 22, text=f"{prob:.2%}",
                       font=("Segoe UI", 14, "bold"), fill=color, anchor="center")


def _update_model_explainability(app: Any, row: dict[str, Any]) -> None:
    """Update Model Explainability panel from a prediction row."""
    try:
        prob = float(row.get("probability", 0))
        thresh = float(row.get("threshold", 0.5))
        conf = float(row.get("confidence") or abs(prob - 0.5) * 2)
        pred_class = int(row.get("predicted_class", -1))
        trade_candidate = str(row.get("trade_candidate", "")).upper()

        if pred_class == 1 or "LONG" in trade_candidate or "BUY" in trade_candidate:
            direction, dir_fg = "BUY", C_BULL
        elif pred_class == 0 or "SHORT" in trade_candidate or "SELL" in trade_candidate:
            direction, dir_fg = "SELL", C_BEAR
        else:
            direction, dir_fg = "NO_TRADE", FG_DIM

        # Decision
        taken = row.get("trade_taken") in (True, 1, "1", "true", "True")
        decision = "EXECUTED" if taken else ("SHADOW" if prob >= thresh else "BLOCKED")
        dec_fg = C_BULL if decision == "EXECUTED" else (C_SHADOW if decision == "SHADOW" else C_BLOCKED)

        # Confidence bucket
        if conf >= 0.6:
            bucket, bucket_fg = "HIGH", C_BULL
        elif conf >= 0.3:
            bucket, bucket_fg = "MEDIUM", STATUS_WARN
        else:
            bucket, bucket_fg = "LOW", C_BEAR

        # Update vars
        app._lc_prob_val_var.set(f"{prob:.2%}")
        app._lc_prob_dir_var.set(f"Direction: {direction}")
        app._lc_prob_conf_var.set(f"Conf: {conf:.2%}")
        app._lc_prob_thresh_var.set(f"Thresh: {thresh:.2f}")
        app._lc_prob_model_var.set(f"Model: {row.get('model_name', 'N/A')}")
        edge = row.get("expected_edge")
        app._lc_prob_edge_var.set(f"Edge: {edge:+.3f}" if edge is not None else "Edge: --")
        app._lc_prob_decision_var.set(f"  {decision}  ")
        app._lc_prob_bucket_var.set(bucket)

        # Colors
        app._lc_prob_dir_lbl.configure(fg=dir_fg)
        app._lc_prob_decision_lbl.configure(fg=dec_fg)
        app._lc_prob_bucket_lbl.configure(fg=bucket_fg)

        # Warnings
        sm = app._lc_shadow_metrics_var.get() if hasattr(app, "_lc_shadow_metrics_var") else ""
        calib_err = row.get("calibration_error")
        drift = row.get("drift_warning", False)
        app._lc_prob_calib_var.set(f"Calibration: {calib_err:.3f}" if calib_err else "Calibration: OK")
        app._lc_prob_drift_var.set(f"Drift: {'⚠ YES' if drift else '✗ No'}")
        calib_fg = STATUS_WARN if calib_err and abs(calib_err) > 0.05 else C_BULL
        drift_fg = STATUS_WARN if drift else C_BULL
        app._lc_prob_warn_lbl.configure(fg=calib_fg)
        app._lc_prob_drift_lbl.configure(fg=drift_fg)

        # Draw gauge
        _draw_probability_gauge(app, prob, direction)

        # Contributors from SHAP/LIME if available
        for tree in [app._lc_pos_contrib_tree, app._lc_neg_contrib_tree]:
            for iid in tree.get_children():
                tree.delete(iid)

        shap_vals = row.get("shap_values") or row.get("feature_importances")
        if not shap_vals and "shap_json" in row:
            try:
                shap_vals = json.loads(row["shap_json"])
            except Exception:
                pass

        if shap_vals:
            try:
                if isinstance(shap_vals, dict):
                    items = sorted(shap_vals.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0)
                else:
                    items = []
                negs = [(k, v) for k, v in items if isinstance(v, (int, float)) and v < 0][:5]
                pos = [(k, v) for k, v in items if isinstance(v, (int, float)) and v >= 0][-5:]
                pos.reverse()
                for feat, val in pos:
                    app._lc_pos_contrib_tree.insert("", tk.END,
                                                   values=(feat, f"{float(val):+.4f}"))
                for feat, val in negs:
                    app._lc_neg_contrib_tree.insert("", tk.END,
                                                   values=(feat, f"{float(val):+.4f}"))
            except Exception:
                pass
        else:
            app._lc_pos_contrib_tree.insert("", tk.END,
                                           values=("No SHAP/LIME data", "--"))
            app._lc_neg_contrib_tree.insert("", tk.END,
                                           values=("No SHAP/LIME data", "--"))

    except Exception:
        pass


def _update_explainability_tab(app: Any, row: dict[str, Any]) -> None:
    """Updates the Feature Explainability treeview from feature json snap."""
    try:
        tree = getattr(app, "_lc_explain_tree", None)
        if tree is None:
            return
        
        # Clear existing
        for iid in tree.get_children():
            tree.delete(iid)

        features = row.get("features") or row.get("feature_snapshot")
        if not features and "features_json" in row:
            try:
                features = json.loads(row["features_json"])
            except Exception:
                pass
        if not features and "feature_snapshot_json" in row:
            try:
                features = json.loads(row["feature_snapshot_json"])
            except Exception:
                pass

        if not features:
            tree.insert("", tk.END, values=("No features logged for this prediction", "--"))
            return

        for k, v in sorted(features.items()):
            try:
                val_str = f"{float(v):.6g}" if isinstance(v, (int, float)) else str(v)
            except Exception:
                val_str = str(v)
            tree.insert("", tk.END, values=(k, val_str))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SAFE REFRESH LOGIC (Required API)
# ---------------------------------------------------------------------------

def refresh_live_chart_snapshot(app: Any, snapshot: LiveChartSnapshot) -> None:
    """Safely refreshes the Matplotlib chart plugin and all UI panels from the snapshot.

    Guarantees no GUI crashes due to missing broker/model/DB data.
    """
    try:
        if snapshot is None:
            return

        # 1. Update Matplotlib Chart Plugin
        plugin = getattr(app, "live_chart_plugin", None)
        has_candles = bool(snapshot.candles)
        if plugin is not None:
            try:
                # Push candles
                if has_candles:
                    plugin.push_candles(snapshot.candles)

                # Push market info
                plugin.set_market_info(
                    spot=snapshot.spot_price,
                    futures=snapshot.futures_price,
                    atm_strike=snapshot.atm_strike,
                    iv=snapshot.option_chain_summary.ce_iv if snapshot.option_chain_summary else None,
                    pcr=snapshot.option_chain_summary.pcr if snapshot.option_chain_summary else None
                )

                # Set session levels
                plugin.set_session_levels(snapshot.session_high, snapshot.session_low)

                # Set prevday levels
                plugin.set_prevday_levels(snapshot.previous_day_high, snapshot.previous_day_low)

                # Set ATM strike
                if snapshot.atm_strike is not None:
                    plugin.set_atm_strike(snapshot.atm_strike)

                # Set prediction markers
                if hasattr(snapshot, "shadow_rows") and snapshot.shadow_rows is not None:
                    plugin.load_prediction_rows(snapshot.shadow_rows)

                # Set trade events
                if hasattr(snapshot, "trade_events") and snapshot.trade_events is not None:
                    plugin.load_trade_event_rows(snapshot.trade_events)

            except Exception as e:
                logger.debug("Failed to refresh chart plugin variables: %s", e)

        # 1b. Empty state — show overlay when no candles, hide when data arrives
        if has_candles:
            _hide_empty_state(app)
        else:
            _show_empty_state(app)
            return  # Skip remaining card updates when waiting for data

        # 2. Update Status Bar

        # 2. Update Status Bar
        try:
            mkt = (snapshot.market_status or "CLOSED").upper()
            app._lc_market_status_var.set(f"MARKET: {mkt}")
            
            broker_status = snapshot.broker_connection_status or "DISCONNECTED"
            app._lc_broker_var.set(f"Broker: {broker_status}")
            
            model_status = snapshot.model_status or "NOT_LOADED"
            app._lc_model_var.set(f"Model: {model_status}")
            
            app._lc_shadow_var.set("🔵 SHADOW" if snapshot.shadow_mode_active else "💰 LIVE")
            app._lc_last_update_var.set(f"Updated: {snapshot.timestamp.strftime('%H:%M:%S')}")

            # Spot
            spot_str = f"{snapshot.spot_price:.2f}" if snapshot.spot_price is not None else "--"
            app._lc_spot_var.set(spot_str)

            # Futures
            fut_str = f"{snapshot.futures_price:.2f}" if snapshot.futures_price is not None else "--"
            app._lc_fut_var.set(fut_str)

            # ATM
            atm_str = f"{snapshot.atm_strike:.0f}" if snapshot.atm_strike is not None else "--"
            app._lc_atm_var.set(atm_str)

            # Regime
            reg_str = str(snapshot.regime_label or "N/A")
            app._lc_regime_var.set(reg_str)

            # Backwards compatibility fields
            sig = snapshot.current_signal or "NO_TRADE"
            app._lc_signal_var.set(f"Signal: {sig}")

            pcr_val = snapshot.option_chain_summary.pcr if snapshot.option_chain_summary else None
            pcr_str = f"PCR: {pcr_val:.2f}" if pcr_val is not None else "PCR: --"
            app._lc_pcr_var.set(pcr_str)

            iv_val = snapshot.option_chain_summary.ce_iv if snapshot.option_chain_summary else None
            iv_str = f"IV: {iv_val:.1f}%" if iv_val is not None else "IV: --"
            app._lc_iv_var.set(iv_str)

            # Update badge colors
            _update_badge_colors(app, mkt, broker_status, model_status,
                                 snapshot.data_health.status if snapshot.data_health else "UNKNOWN",
                                 snapshot.shadow_mode_active)

            # New badge vars: last candle time
            dh = snapshot.data_health
            if dh and dh.last_candle_time:
                age = (datetime.now() - dh.last_candle_time).total_seconds()
                app._lc_last_candle_var.set(f"{dh.last_candle_time.strftime('%H:%M:%S')} ({int(age)}s)")
            else:
                app._lc_last_candle_var.set("--")

            # Stale warning (right-section display)
            if dh:
                if dh.status == "CRITICAL":
                    app._lc_stale_var.set(f"⚠ CRIT: {dh.stale_data_seconds}s")
                elif dh.status == "WARNING":
                    app._lc_stale_var.set(f"⚡ WARN: {dh.stale_data_seconds}s")
                else:
                    app._lc_stale_var.set("")
            else:
                app._lc_stale_var.set("")

            # API errors today
            api_err = dh.api_error_count_today if dh else 0
            app._lc_api_err_var.set(f"API: {api_err}")

            # Refresh timestamp
            app._lc_refresh_var.set(snapshot.timestamp.strftime('%H:%M:%S'))

            # Change % (requires previous close — show -- if unavailable)
            app._lc_change_var.set("--")

        except Exception as e:
            logger.debug("Failed status bar refresh: %s", e)

        # 3. Update current prediction card
        try:
            _refresh_current_prediction_card(app, snapshot)
        except Exception:
            pass

        # 4. Update shadow performance tab
        try:
            _refresh_shadow_performance(app, snapshot)
        except Exception:
            pass

        # 4b. Update model explainability tab
        try:
            lp = snapshot.latest_prediction
            if lp:
                _update_model_explainability(app, lp)
        except Exception:
            pass

        # 5. Update option chain card
        try:
            _refresh_option_chain_card(app, snapshot)
        except Exception:
            pass

        # 6. Update data health card
        try:
            _refresh_data_health_card(app, snapshot)
        except Exception:
            pass

        # 6b. Update Market Intelligence panel
        try:
            _refresh_market_intelligence(app, snapshot)
        except Exception:
            pass

        # 7. Update alert card
        try:
            _refresh_alert_card(app, snapshot)
        except Exception:
            pass

        # 8. Update event timeline
        try:
            _refresh_timeline(app, snapshot)
        except Exception:
            pass

        # 9. Update event timeline
        try:
            _refresh_timeline(app, snapshot)
        except Exception:
            pass
        try:
            _update_greeks_gauges(app)
        except Exception:
            pass

    except Exception as e:
        logger.error("Error in refresh_live_chart_snapshot: %s", e)


# ---------------------------------------------------------------------------
# Orchestrated refresh
# ---------------------------------------------------------------------------

def _refresh_live_chart_tab(app: Any, force: bool = False) -> None:
    """Main refresh dispatcher — called every ~3s via _safe_after."""
    try:
        now_ts = time.time()
        last_ts = float(getattr(app, "_lc_refresh_ts", 0.0) or 0.0)
        refresh_sec = float(getattr(app, "_lc_refresh_sec", 3.0) or 3.0)
        if not force and (now_ts - last_ts) < refresh_sec:
            app._safe_after(int(refresh_sec * 1000), _refresh_live_chart_tab, app, False)
            return

        # Build the snapshot
        snapshot = _build_snapshot(app)

        # Delegate refresh to the public API method
        refresh_live_chart_snapshot(app, snapshot)

        app._lc_refresh_ts = now_ts

    except Exception as e:
        logger.debug("Error in _refresh_live_chart_tab: %s", e)
    finally:
        app._safe_after(3000, _refresh_live_chart_tab, app, False)


def _build_snapshot(app: Any) -> LiveChartSnapshot:
    """Gather all available data and build a LiveChartSnapshot."""
    try:
        scalper = getattr(app, "_scalper", None)
        client = getattr(app, "_client", None)
        db = getattr(app, "_db_manager", None)
        if db is None and scalper is not None:
            db = getattr(scalper, "db_manager", None)

        # Spot / ATM / IV / PCR from scalper or market info
        spot = None
        iv = None
        pcr = None
        atm_strike = None
        futures_price = None
        spot_source = "none"
        futures_source = "none"
        atm_source = "none"

        if scalper is not None:
            spot = getattr(scalper, "_spot_ltp", None)
            if spot is not None:
                spot_source = "scalper._spot_ltp"
            iv = getattr(scalper, "_iv", None)
            pcr = getattr(scalper, "_pcr", None)
            atm_strike = getattr(scalper, "_atm_strike", None)
            if atm_strike is not None:
                atm_source = "scalper._atm_strike"
            futures_price = getattr(scalper, "_futures_price", None)
            if futures_price is not None:
                futures_source = "scalper._futures_price"

        # Fallback: app._market_info
        if spot is None and hasattr(app, "_market_info") and app._market_info:
            spot = app._market_info.get("spot")
            if spot is not None:
                spot_source = "app._market_info.spot"
        if futures_price is None and hasattr(app, "_market_info") and app._market_info:
            futures_price = app._market_info.get("futures")
            if futures_price is not None:
                futures_source = "app._market_info.futures"
        if atm_strike is None and hasattr(app, "_market_info") and app._market_info:
            atm_strike = app._market_info.get("atm_strike")
            if atm_strike is not None:
                atm_source = "app._market_info.atm_strike"
        if iv is None and hasattr(app, "_market_info") and app._market_info:
            iv = app._market_info.get("iv")
        if pcr is None and hasattr(app, "_market_info") and app._market_info:
            pcr = app._market_info.get("pcr")

        # Fallback: app.live_chart_plugin for ATM
        if atm_strike is None and hasattr(app, "live_chart_plugin"):
            atm_strike = getattr(app.live_chart_plugin, "_atm_strike", None)
            if atm_strike is not None:
                atm_source = "live_chart_plugin._atm_strike"

        # Option chain payload
        option_chain_payload = getattr(app, "_option_chain_data", None)
        oc_source = "app._option_chain_data"
        if option_chain_payload is None and scalper is not None:
            option_chain_payload = getattr(scalper, "_option_chain_cache", None)
            oc_source = "scalper._option_chain_cache"

        # Candles — multi-source fallback priority (commit bf351fb / 57d00c5)
        raw_candles = None
        candle_source = "none"
        if hasattr(app, "live_chart_plugin") and getattr(app.live_chart_plugin, "_candles", None):
            raw_candles = app.live_chart_plugin._candles
            candle_source = "live_chart_plugin"
        elif hasattr(app, "_latest_candles") and app._latest_candles:
            raw_candles = app._latest_candles
            candle_source = "app._latest_candles"
        elif hasattr(app, "_candles") and app._candles:
            raw_candles = app._candles
            candle_source = "app._candles"
        elif scalper is not None and getattr(scalper, "_candles", None):
            raw_candles = scalper._candles
            candle_source = "scalper._candles"
        elif scalper is not None and getattr(scalper, "_spot_candles", None):
            raw_candles = scalper._spot_candles
            candle_source = "scalper._spot_candles"

        # Normalize: resolve time/timestamp mismatch, sort, dedupe (inline, cf. normalize_live_chart_candles)
        normalized = []
        if raw_candles:
            try:
                from pandas import to_datetime as _pd_tt  # noqa: N811
                # Normalize each candle to a sortable time field
                norm_candles = []
                for c in raw_candles:
                    t_raw = c.time if hasattr(c, "time") else (getattr(c, "timestamp", None) if hasattr(c, "timestamp") else None)
                    if t_raw is None:
                        continue
                    # Ensure comparable datetime
                    if isinstance(t_raw, (int, float)):
                        t_norm = _pd_tt(t_raw, unit="s", utc=True).tz_localize(None).to_pydatetime()
                    elif isinstance(t_raw, str):
                        t_norm = _pd_tt(t_raw).to_pydatetime()
                    else:
                        t_norm = t_raw  # already datetime-like
                    norm_candles.append((t_norm, c))

                # Sort by normalized time
                norm_candles.sort(key=lambda x: x[0])

                # Deduplicate by time
                seen_times = set()
                for t_norm, c in norm_candles:
                    if t_norm not in seen_times:
                        seen_times.add(t_norm)
                        normalized.append(c)
            except Exception as e_norm:
                logger.warning("[LIVE_CHART] candle normalization failed: %s", e_norm)
                normalized = list(raw_candles) if raw_candles else []

        # Push normalized candles back into plugin so it stays in sync
        if hasattr(app, "live_chart_plugin") and normalized:
            try:
                app.live_chart_plugin.push_candles(normalized)
            except Exception as e_push:
                logger.debug("[LIVE_CHART] push_candles failed: %s", e_push)

        candles = normalized
        logger.info("[LIVE_CHART] candle_source=%s raw=%d normalized=%d",
                    candle_source,
                    len(raw_candles) if raw_candles else 0,
                    len(normalized))

        # VWAP + EMAs from plugin (latest computed values)
        vwap = _get_latest_vwap(app)
        ema_9 = _get_latest_ema(app, 9)
        ema_21 = _get_latest_ema(app, 21)
        ema_50 = _get_latest_ema(app, 50)
        ema_200 = _get_latest_ema(app, 200)

        # Session / prevday levels from plugin
        session_high = getattr(app.live_chart_plugin, "_session_high", None) if hasattr(app, "live_chart_plugin") else None
        session_low  = getattr(app.live_chart_plugin, "_session_low", None)  if hasattr(app, "live_chart_plugin") else None
        pdh = getattr(app.live_chart_plugin, "_prevday_high", None)          if hasattr(app, "live_chart_plugin") else None
        pdl = getattr(app.live_chart_plugin, "_prevday_low", None)            if hasattr(app, "live_chart_plugin") else None
        or_high = getattr(app.live_chart_plugin, "_opening_range_high", None) if hasattr(app, "live_chart_plugin") else None
        or_low  = getattr(app.live_chart_plugin, "_opening_range_low", None)  if hasattr(app, "live_chart_plugin") else None

        # DB rows
        prediction_rows: list[dict] = []
        trade_events: list[dict] = []
        if db is not None:
            if hasattr(db, "list_recent_prediction_markers"):
                prediction_rows = db.list_recent_prediction_markers(limit=500)
            if hasattr(db, "list_recent_trade_events"):
                trade_events = db.list_recent_trade_events(limit=200)

        # Latest prediction → SignalMarker
        latest_pred = None
        if prediction_rows:
            latest_row = prediction_rows[0]
            latest_pred = _row_to_signal_marker(latest_row, spot)

        # Broker / model status
        broker_connected = client is not None
        model_loaded = scalper is not None and getattr(scalper, "_model_loaded", False)

        # Shadow logger / DB write
        shadow_logger_ok = db is not None
        db_write_ok = db is not None

        # Market status
        market_status = _guess_market_status()
        broker_status = "CONNECTED" if broker_connected else "DISCONNECTED"

        # Last candle / tick timestamps
        last_candle_ts = None
        if candles:
            try:
                c = candles[-1]
                t = c.time if hasattr(c, "time") else None
                if t is not None:
                    last_candle_ts = float(t) if not isinstance(t, datetime) else t.timestamp()
            except Exception:
                pass

        # Regime
        regime_label = getattr(scalper, "_regime", "N/A") if scalper else "N/A"

        # ── INFO-level data-source audit logs ─────────────────────────────────
        logger.info("[LIVE_CHART] spot_source=%s value=%s", spot_source, spot)
        logger.info("[LIVE_CHART] futures_source=%s value=%s", futures_source, futures_price)
        logger.info("[LIVE_CHART] atm_source=%s value=%s", atm_source, atm_strike)
        logger.info("[LIVE_CHART] option_chain_source=%s has_data=%s", oc_source, option_chain_payload is not None)
        logger.info("[LIVE_CHART] prediction_rows=%d trade_events=%d", len(prediction_rows), len(trade_events))
        logger.info("[LIVE_CHART] broker_connected=%s model_loaded=%s", broker_connected, model_loaded)

        snapshot = build_live_chart_snapshot(
            timestamp=datetime.now(),
            spot_price=spot,
            futures_price=futures_price,
            atm_strike=atm_strike,
            candles=candles,
            vwap=vwap,
            ema_9=ema_9, ema_21=ema_21, ema_50=ema_50, ema_200=ema_200,
            previous_day_high=pdh,
            previous_day_low=pdl,
            session_high=session_high,
            session_low=session_low,
            opening_range_high=or_high,
            opening_range_low=or_low,
            latest_prediction=latest_pred,
            option_chain_payload=option_chain_payload,
            shadow_rows=prediction_rows,
            trade_events=trade_events,
            last_candle_ts=last_candle_ts,
            last_tick_ts=None,
            broker_connected=broker_connected,
            model_loaded=model_loaded,
            shadow_logger_ok=shadow_logger_ok,
            db_write_ok=db_write_ok,
            missing_candles=0,
            api_errors_today=0,
            market_status=market_status,
            broker_connection_status=broker_status,
            model_status="LOADED" if model_loaded else "NOT_LOADED",
            shadow_mode_active=True,
            regime_label=str(regime_label) if regime_label else "N/A",
        )

        app._lc_all_timeline_rows = prediction_rows
        return snapshot

    except Exception as e:
        logger.debug("Failed build snapshot: %s", e)
        return build_live_chart_snapshot()


# ---------------------------------------------------------------------------
# Individual card refreshers
# ---------------------------------------------------------------------------

def _refresh_status_bar(app: Any, snap: LiveChartSnapshot) -> None:
    # Retained for backward compatibility. Direct refreshes now go through refresh_live_chart_snapshot.
    pass


def _refresh_current_prediction_card(app: Any, snap: LiveChartSnapshot) -> None:
    try:
        pred = snap.latest_prediction
        if pred is None:
            _set_prediction_card_collecting(app)
            return

        app._lc_pred_direction_var.set(f"Direction: {pred.side} {pred.instrument}")

        prob_pct = f"{pred.probability * 100:.1f}%" if pred.probability else "--"
        app._lc_pred_probability_var.set(f"Probability: {prob_pct}")

        conf_pct = f"{pred.confidence * 100:.1f}%" if pred.confidence else "--"
        app._lc_pred_confidence_var.set(f"Confidence: {conf_pct}")

        app._lc_pred_threshold_var.set(f"Threshold: {pred.threshold:.2f}")
        app._lc_pred_model_var.set(f"Model: {pred.model_name}")
        ts_str = pred.timestamp.strftime("%H:%M:%S") if pred.timestamp else "--"
        app._lc_pred_time_var.set(f"Time: {ts_str}")
        edge_str = f"{pred.expected_edge:.4f}" if pred.expected_edge else "n/a"
        app._lc_pred_edge_var.set(f"Expected Edge: {edge_str}")
        app._lc_pred_regime_var.set(f"Regime: {snap.regime_label}")

        liq = snap.option_chain_summary.liquidity_score
        app._lc_pred_liq_var.set(f"Liquidity Score: {liq}")

        ce_sp = snap.option_chain_summary.ce_bid_ask_spread
        pe_sp = snap.option_chain_summary.pe_bid_ask_spread
        spread_str = "n/a"
        if ce_sp is not None or pe_sp is not None:
            best = min(filter(None, [ce_sp, pe_sp]), default=None)
            spread_str = f"{best:.2f}" if best else "n/a"
        app._lc_pred_spread_var.set(f"Bid-Ask Spread: {spread_str}")

        # Final decision
        decision = "ACCEPT" if pred.probability >= pred.threshold else "BLOCK"
        app._lc_pred_decision_var.set(f"Decision: {decision}")

        badge_text = f"  {decision}  "
        if decision == "ACCEPT":
            badge_color = "#16a34a"  # green
            fg = "white"
        else:
            badge_color = "#dc2626"  # red
            fg = "white"
        app._lc_pred_decision_lbl.configure(text=badge_text, background=badge_color, foreground=fg)

        # Update explainability for the latest prediction
        if pred and hasattr(pred, "timestamp") and snap.shadow_rows:
            # Match by nearest ts
            pred_ts = pred.timestamp.timestamp()
            for row in snap.shadow_rows:
                if abs(float(row.get("ts", 0) or 0) - pred_ts) < 2:
                    _update_explainability_tab(app, row)
                    break

    except Exception:
        _set_prediction_card_collecting(app)


def _refresh_shadow_metrics_card(app: Any, snap: LiveChartSnapshot) -> None:
    try:
        m = snap.shadow_metrics
        if m.predictions_today == 0:
            _set_shadow_card_collecting(app)
            return

        app._lc_shad_preds_var.set(str(m.predictions_today))
        app._lc_shad_accepted_var.set(str(m.accepted_signals))
        app._lc_shad_blocked_var.set(str(m.blocked_signals))
        app._lc_shad_resolved_var.set(f"{m.resolved_trades} / {m.pending_trades}")
        app._lc_shad_winrate_var.set(f"{m.win_rate:.1f}%")
        app._lc_shad_pf_var.set(f"{m.profit_factor:.2f}")
        app._lc_shad_exp_var.set(f"₹{m.expectancy:.2f}")
        app._lc_shad_mdd_var.set(f"₹{m.max_drawdown:.2f}")
        app._lc_shad_pnl_var.set(f"₹{m.daily_pnl:.2f}")

        if m.drift_warning:
            app._lc_shad_drift_var.set("⚠️ DRIFT WARNING")
        else:
            app._lc_shad_drift_var.set("NORMAL")

    except Exception:
        _set_shadow_card_collecting(app)


def _refresh_option_chain_card(app: Any, snap: LiveChartSnapshot) -> None:
    try:
        oc = snap.option_chain_summary
        app._lc_oc_atm_var.set(f"ATM Strike: {oc.atm_strike:.0f}" if oc.atm_strike else "ATM Strike: --")
        app._lc_oc_ce_ltp_var.set(f"CE LTP: {oc.ce_ltp:.2f}" if oc.ce_ltp else "CE LTP: --")
        app._lc_oc_pe_ltp_var.set(f"PE LTP: {oc.pe_ltp:.2f}" if oc.pe_ltp else "PE LTP: --")
        app._lc_oc_ce_iv_var.set(f"CE IV: {oc.ce_iv:.1f}%" if oc.ce_iv else "CE IV: --")
        app._lc_oc_pe_iv_var.set(f"PE IV: {oc.pe_iv:.1f}%" if oc.pe_iv else "PE IV: --")
        app._lc_oc_ce_oi_var.set(f"CE OI: {oc.ce_oi:,.0f}" if oc.ce_oi else "CE OI: --")
        app._lc_oc_pe_oi_var.set(f"PE OI: {oc.pe_oi:,.0f}" if oc.pe_oi else "PE OI: --")
        app._lc_oc_pcr_var.set(f"PCR: {oc.pcr:.2f}" if oc.pcr else "PCR: --")

        ce_sp = oc.ce_bid_ask_spread
        pe_sp = oc.pe_bid_ask_spread
        sp_str = "n/a"
        if ce_sp is not None or pe_sp is not None:
            best = min(filter(None, [ce_sp, pe_sp]), default=None)
            sp_str = f"{best:.2f}" if best else "n/a"
        app._lc_oc_spread_var.set(f"ATM Spread: {sp_str}")
        app._lc_oc_liq_var.set(f"Liquidity Score: {oc.liquidity_score}")

        warnings = []
        if oc.wide_spread_warning:
            warnings.append("Wide Spread")
        if oc.iv_spike_warning:
            warnings.append("IV Spike")
        if oc.is_stale:
            warnings.append("Stale")
        
        warn_str = " | ".join(warnings) if warnings else ""
        if warn_str:
            app._lc_oc_warning_var.set(f"Warnings: {warn_str}")
            app._lbl_oc_warning.configure(fg="#f59e0b")
        else:
            app._lc_oc_warning_var.set("Warnings: None")
            app._lbl_oc_warning.configure(fg=FG_SECONDARY)

    except Exception:
        pass


def _refresh_data_health_card(app: Any, snap: LiveChartSnapshot) -> None:
    try:
        dh = snap.data_health
        broker_str = "✓ Connected" if dh.broker_connected else "✗ Disconnected"
        app._lc_dh_broker_var.set(f"Broker Connection: {broker_str}")

        if dh.last_candle_time:
            age_s = (datetime.now() - dh.last_candle_time).total_seconds()
            app._lc_dh_candle_age_var.set(f"Last Candle: {dh.last_candle_time.strftime('%H:%M:%S')} ({int(age_s)}s ago)")
        else:
            app._lc_dh_candle_age_var.set("Last Candle: --")

        if dh.last_tick_time:
            tick_age_s = (datetime.now() - dh.last_tick_time).total_seconds()
            app._lc_dh_tick_age_var.set(f"Last Tick: {dh.last_tick_time.strftime('%H:%M:%S.%f')[:-3]} ({int(tick_age_s)}s ago)")
        else:
            app._lc_dh_tick_age_var.set("Last Tick: --")

        app._lc_dh_stale_sec_var.set(f"Stale Seconds: {dh.stale_data_seconds}s")

        model_str = "✓ Loaded" if dh.model_loaded else "✗ Not Loaded"
        app._lc_dh_model_var.set(f"Model Registry: {model_str}")

        shadow_str = "✓ OK" if dh.shadow_logger_ok else "✗ Inactive"
        app._lc_dh_shadow_var.set(f"Shadow Log Queue: {shadow_str}")

        db_str = "✓ OK" if dh.db_write_ok else "✗ Error"
        app._lc_dh_db_var.set(f"DB Write Health: {db_str}")

        app._lc_dh_missing_var.set(f"Missing Candles: {dh.missing_candles}")
        app._lc_dh_api_err_var.set(f"API Error Count: {dh.api_error_count_today}")
        app._lc_dh_reason_var.set(f"Status Reason: {dh.status_reason or 'None'}")

        status_text = dh.status
        app._lc_dh_status_var.set(f"Overall Status: {status_text}")

        # Badge
        badge_colors = {"OK": "#16a34a", "WARNING": "#f59e0b", "CRITICAL": "#dc2626", "UNKNOWN": "#6b7280"}
        bg = badge_colors.get(dh.status, "#6b7280")
        app._lc_dh_status_badge.configure(text=f"  {dh.status}  ", background=bg, foreground="white")

        # ── Trading safety guidance ─────────────────────────────────
        # SAFE TO OBSERVE: data health OK, broker connected, model loaded, stale < 30s
        # DEGRADED: data health WARNING or stale 30-120s
        # DO NOT TRADE: data health CRITICAL or stale > 120s or broker disconnected
        # DATA UNSAFE: model not loaded or DB write errors
        is_ok      = dh.status == "OK"
        is_warn    = dh.status == "WARNING"
        is_crit    = dh.status == "CRITICAL"
        connected  = dh.broker_connected
        model_ok   = dh.model_loaded
        stale_ok   = dh.stale_data_seconds < 30
        stale_warn = 30 <= dh.stale_data_seconds < 120
        stale_bad  = dh.stale_data_seconds >= 120
        db_ok      = dh.db_write_ok

        if is_crit or stale_bad or not connected:
            safety_txt, safety_bg, safety_fg = "⛔  DO NOT TRADE  ⛔", C_BEAR, "white"
        elif is_warn or stale_warn:
            safety_txt, safety_bg, safety_fg = "⚠  DEGRADED — CAUTION  ⚠", STATUS_WARN, "white"
        elif not model_ok or not db_ok:
            safety_txt, safety_bg, safety_fg = "⚠  DATA UNSAFE  ⚠", STATUS_WARN, "white"
        elif is_ok and connected and model_ok and stale_ok and db_ok:
            safety_txt, safety_bg, safety_fg = "✓  SAFE TO OBSERVE  ✓", "#16a34a", "white"
        else:
            safety_txt, safety_bg, safety_fg = "?  UNCERTAIN  ?", "#6b7280", "white"

        app._lc_trading_safety_var.set(safety_txt)
        app._lc_trading_safety_lbl.configure(background=safety_bg, foreground=safety_fg)

    except Exception:
        pass


def _refresh_alert_card(app: Any, snap: LiveChartSnapshot) -> None:
    """Refresh Alerts tab: deduplicated, severity-badge list, last 100."""
    try:
        container = getattr(app, "_lc_alerts_container", None)
        if container is None:
            return

        for w in container.pack_slaves():
            w.destroy()

        alerts: list[dict[str, Any]] = getattr(app, "_lc_active_alerts", [])
        now = time.time()
        # Keep alerts alive for 5 minutes; trim to 100
        alerts = [a for a in alerts if (now - a.get("_added_ts", now)) < 300]

        # Generate new alerts from snapshot (deduplicate by type within window)
        recent_types = {a.get("type") for a in alerts if (now - a.get("_added_ts", now)) < 300}
        m = snap.shadow_metrics
        dh = snap.data_health
        oc = snap.option_chain_summary

        if m.drift_warning and "DRIFT" not in recent_types:
            alerts.append({"type": "DRIFT", "severity": "WARNING",
                           "message": "Prediction drift detected", "source": "Shadow",
                           "_added_ts": now})
        if oc.wide_spread_warning and "SPREAD" not in recent_types:
            alerts.append({"type": "SPREAD", "severity": "WARNING",
                           "message": f"Wide spread: ₹{oc.ce_bid_ask_spread:.2f}", "source": "OptionChain",
                           "_added_ts": now})
        if oc.iv_spike_warning and "IV" not in recent_types:
            alerts.append({"type": "IV", "severity": "WARNING",
                           "message": f"IV spike CE={oc.ce_iv:.1f}% PE={oc.pe_iv:.1f}%", "source": "OptionChain",
                           "_added_ts": now})
        if dh.status == "CRITICAL" and "CRITICAL" not in recent_types:
            alerts.append({"type": "CRITICAL", "severity": "ERROR",
                           "message": dh.status_reason or "Data health critical", "source": "DataHealth",
                           "_added_ts": now})
        if not dh.broker_connected and "BROKER_DISC" not in recent_types:
            alerts.append({"type": "BROKER_DISC", "severity": "ERROR",
                           "message": "Broker disconnected", "source": "DataHealth",
                           "_added_ts": now})
        if m.drift_detected and "DRIFT_DET" not in recent_types:
            alerts.append({"type": "DRIFT_DET", "severity": "WARNING",
                           "message": "Statistical drift in feature distribution", "source": "Shadow",
                           "_added_ts": now})

        # Trim to last 100
        app._lc_active_alerts = alerts[-100:]

        if not app._lc_active_alerts:
            tk.Label(container, text="No active alerts", font=("Segoe UI", 9),
                     foreground=FG_DIM, bg=BG_CARD, anchor="w").pack(anchor="w", padx=8, pady=10)
            return

        sev_colors = {"INFO": "#3b82f6", "WARNING": "#f59e0b", "ERROR": "#dc2626"}
        sev_order  = {"ERROR": 0, "WARNING": 1, "INFO": 2}

        # Sort by severity then time (most severe/recent first)
        sorted_alerts = sorted(app._lc_active_alerts,
                               key=lambda a: (sev_order.get(a.get("severity", "INFO"), 2),
                                              a.get("_added_ts", 0)), reverse=True)

        for alert in sorted_alerts[:50]:  # show last 50 in UI (last 100 stored)
            sev  = alert.get("severity", "INFO")
            msg  = alert.get("message", "")
            src  = alert.get("source", "System")
            age  = now - alert.get("_added_ts", now)
            age_str = f"{int(age)}s ago" if age < 60 else f"{int(age // 60)}m ago"
            color = sev_colors.get(sev, "#6b7280")

            row = tk.Frame(container, bg=BG_CARD, padx=0, pady=1)
            row.pack(fill=tk.X, padx=0, pady=1)

            # Severity badge
            sev_badge = tk.Frame(row, background=color, padx=4, pady=0)
            sev_badge.pack(side=tk.LEFT, fill=tk.Y)
            tk.Label(sev_badge, text=sev[:3].upper(), background=color, foreground="white",
                     font=("Segoe UI", 6, "bold"), padx=2).pack(fill=tk.Y)

            # Content
            content = tk.Frame(row, bg=BG_CARD)
            content.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            tk.Label(content, text=f"{msg}", font=("Segoe UI", 8), bg=BG_CARD,
                     fg=FG_PRIMARY, anchor="w").pack(anchor="w")
            tk.Label(content, text=f"{src} · {age_str}", font=("Segoe UI", 7),
                     bg=BG_CARD, fg=FG_DIM, anchor="w").pack(anchor="w")

            if len(app._lc_active_alerts) > 50:
                tk.Label(container, text=f"+ {len(app._lc_active_alerts) - 50} older alerts",
                         font=("Segoe UI", 7), fg=FG_DIM, bg=BG_CARD,
                         anchor="center").pack(fill=tk.X)

    except Exception:
        pass


def _refresh_timeline(app: Any, snap: LiveChartSnapshot) -> None:
    _refresh_timeline_from_rows(app, getattr(app, "_lc_all_timeline_rows", []))


def _refresh_timeline_from_rows(app: Any, rows: list[dict[str, Any]]) -> None:
    try:
        tree = getattr(app, "_lc_timeline_tree", None)
        if tree is None:
            return

        for iid in tree.get_children():
            tree.delete(iid)

        # Sort if active
        sort_col = getattr(app, "_lc_tape_sort_col", None)
        sort_rev = getattr(app, "_lc_tape_sort_rev", False)
        if sort_col and rows:
            col_map = {"time": 0, "side": 1, "cp": 2, "strike": 3, "prob": 4,
                       "edge": 5, "status": 6, "reason": 7, "pnl": 8}
            idx = col_map.get(sort_col, 0)
            rows = sorted(rows, key=lambda r: _timeline_sort_key(r, idx), reverse=sort_rev)

        # Store filtered rows for click-to-focus
        app._lc_all_timeline_rows = rows

        filter_mode = app._lc_timeline_filter_var.get()
        count = 0

        for row in rows[:50]:
            try:
                ts_val = row.get("ts") or row.get("timestamp")
                if ts_val is None:
                    continue
                try:
                    ts_dt = datetime.fromtimestamp(float(ts_val))
                except Exception:
                    ts_dt = datetime.now()
                ts_str = ts_dt.strftime("%H:%M:%S")

                trade_candidate = str(row.get("trade_candidate", "")).upper()
                prob_class = int(row.get("predicted_class", -1))
                taken = row.get("trade_taken") in (True, 1, "1", "true", "True")

                if prob_class == 1 or "LONG" in trade_candidate or "BUY" in trade_candidate:
                    side, cp, tag = "BUY", "CE", "buy"
                elif prob_class == 0 or "SHORT" in trade_candidate or "SELL" in trade_candidate:
                    side, cp, tag = "SELL", "PE", "sell"
                else:
                    side, cp, tag = "EXIT", "", "exit"

                # Filters
                if filter_mode == "signals" and not taken:
                    continue
                if filter_mode == "blocks" and taken:
                    continue
                if filter_mode == "shadow" and taken:
                    continue
                if filter_mode == "exits" and tag != "exit":
                    continue

                prob = float(row.get("probability", 0))
                prob_str = f"{prob:.2f}"
                edge = row.get("expected_edge")
                edge_str = f"{edge:+.3f}" if edge is not None else "--"
                strike = row.get("strike", "")
                status = "EXECUTED" if taken else ("SHADOW" if not taken else "BLOCKED")
                if status == "SHADOW":
                    tag = "shadow"
                elif status == "BLOCKED":
                    tag = "blocked"
                reason = str(row.get("reason", "") or row.get("regime", "") or "")
                pnl = row.get("pnl")
                pnl_str = f"{pnl:+.2f}" if pnl is not None else "--"

                tree.insert("", tk.END,
                           values=(ts_str, side, cp, strike, prob_str, edge_str,
                                   status, reason[:50], pnl_str),
                           tags=(tag,))
                count += 1
            except Exception:
                continue

        app._lc_tape_count_var.set(f"{count} signals")

    except Exception:
        pass


def _timeline_sort_key(row: dict[str, Any], col_idx: int):
    """Return sort key for timeline row by column index."""
    try:
        if col_idx == 0:  # time
            return row.get("ts") or row.get("timestamp") or 0
        if col_idx == 3:  # strike
            return float(row.get("strike") or 0)
        if col_idx == 4:  # prob
            return float(row.get("probability") or 0)
        if col_idx == 5:  # edge
            return float(row.get("expected_edge") or 0)
        if col_idx == 8:  # pnl
            return float(row.get("pnl") or 0)
        return str(row.get("trade_candidate", ""))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Coordinate Conversion & Marker Selection Clicks (Fix Click Handler Bug)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KEYBOARD SHORTCUT HANDLERS
# ---------------------------------------------------------------------------

def _on_key_reset_view(app: Any) -> None:
    """R key — reset chart zoom to show all candles."""
    try:
        plugin = getattr(app, "live_chart_plugin", None)
        if plugin and hasattr(plugin, "reset_view"):
            plugin.reset_view()
    except Exception:
        pass


def _on_key_toggle_lock(app: Any) -> None:
    """L key — toggle lock-to-live."""
    try:
        plugin = getattr(app, "live_chart_plugin", None)
        if plugin:
            current = getattr(plugin, "_lock_to_live", False)
            plugin.set_lock_to_live(not current)
            # Sync toolbar checkbox if it exists
            lock_var = getattr(app, "_lc_lock_var", None)
            if lock_var:
                lock_var.set(not current)
    except Exception:
        pass


class _CrosshairOverlay:
    """Tkinter crosshair + OHLC HUD overlay on the chart canvas.

    Shows dashed vertical + horizontal crosshair lines and a floating HUD panel
    with OHLC data when the mouse hovers over the chart area. Hidden when the
    mouse leaves. Suppresses the built-in Matplotlib crosshair via the plugin flag.
    """
    def __init__(self, app: Any, canvas_widget: tk.Widget) -> None:
        self.app = app
        self.canvas = canvas_widget
        self.visible = False

        # Outer frame — full-canvas overlay, transparent background
        self.frame = tk.Frame(canvas_widget, bg="", bd=0, relief="flat",
                              highlightthickness=0, cursor="cross")
        # Place over entire canvas
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.frame.lower()  # below all widgets

        # Crosshair lines (Canvas.create_line)
        self.line_h = None   # horizontal line
        self.line_v = None   # vertical line
        self.line_x = 0
        self.line_y = 0

        # HUD panel
        self._hud: Optional[tk.Widget] = None
        self._hud_labels: dict[str, tk.StringVar] = {}
        self._build_hud()
        self._hide()

        # Mouse bindings on the chart canvas widget
        canvas_widget.bind("<Motion>", self._on_motion)
        canvas_widget.bind("<Leave>", lambda e: self._hide())
        # Also hide when canvas widget is unmapped
        canvas_widget.bind("<Unmap>", lambda e: self._hide())

    def _build_hud(self) -> None:
        """Build the HUD canvas with OHLC text items."""
        self._hud_canvas = tk.Canvas(self.frame, bg=BG_CARD, bd=1,
                                     relief="solid", highlightthickness=0,
                                     insertwidth=0, takefocus=False)
        self._hud_canvas.bind("<Motion>", lambda e: "break")
        self._hud_canvas.bind("<Leave>", lambda e: "break")

        rows = ["DateTime", "O", "H", "L", "C", "Volume",
                "VWAP", "EMA9", "EMA21", "RSI"]
        for label_text in rows:
            var = tk.StringVar(value="--")
            self._hud_labels[label_text] = var

        self._hud_row_keys = rows  # ordered list for positioning

    def _ensure_hud_positioned(self, x: int, y: int) -> None:
        """Position HUD panel near cursor, flipping sides to stay on screen."""
        cv_w = self.canvas.winfo_width() or 800
        cv_h = self.canvas.winfo_height() or 600
        hud_w = 130
        hud_h = 150

        pad = 10
        if x + hud_w + pad < cv_w:
            hx = x + pad
        else:
            hx = max(0, x - hud_w - pad)
        hy = max(0, min(y - hud_h // 2, cv_h - hud_h))

        self._hud_canvas.place(x=hx, y=hy, width=hud_w, height=hud_h)

        # Redraw all text items at correct y positions
        self._hud_canvas.delete("all")
        row_h = 14
        y_off = 8
        for label_text in self._hud_row_keys:
            val = self._hud_labels.get(label_text, tk.StringVar(value="--"))
            self._hud_canvas.create_text(6, y_off, text=f"{label_text}:",
                                         font=("Segoe UI", 7, "bold"),
                                         fill=FG_MUTED, anchor="nw")
            self._hud_canvas.create_text(72, y_off, text=val.get(),
                                         font=("Segoe UI", 7),
                                         fill=FG_PRIMARY, anchor="nw")
            y_off += row_h

    def _on_motion(self, event: tk.Event) -> None:
        """Handle mouse motion on the chart canvas."""
        plugin = getattr(self.app, "live_chart_plugin", None)
        if plugin is None:
            return

        try:
            x, y = event.x, event.y
            # Convert pixel → data coords via Matplotlib transData
            ax = self._get_main_axes(plugin)
            if ax is None:
                return

            data_x, data_y = self._to_data_coords(ax, x, y)
            self.line_x, self.line_y = data_x, data_y

            # Update crosshair lines
            self._draw_crosshair(x, y)

            # Find nearest candle
            candle = self._nearest_candle(plugin, data_x)
            self._update_hud(candle, plugin, data_x, data_y)
            self._ensure_hud_positioned(x, y)
            self._show()
        except Exception:
            self._hide()
            return

    def _get_main_axes(self, plugin: Any):
        """Get the main price axes from the plugin's figure (fig.axes[0] = ax_price)."""
        try:
            return plugin.fig.axes[0] if plugin.fig.axes else None
        except Exception:
            return None

    def _to_data_coords(self, ax: Any, px: int, py: int) -> tuple:
        """Convert pixel (x, y) to data coordinates using transData.inverted()."""
        try:
            trans = ax.transData.inverted()
            dx, dy = trans.transform((px, py))
            return float(dx), float(dy)
        except Exception:
            return 0.0, 0.0

    def _nearest_candle(self, plugin: Any, data_x: float) -> Optional[Any]:
        """Return the nearest candle object to data_x (float timestamp)."""
        candles = getattr(plugin, "_candles", []) or []
        if not candles:
            return None
        try:
            return min(candles, key=lambda c: abs(float(getattr(c, "ts", 0)) - data_x))
        except Exception:
            return None

    def _draw_crosshair(self, px: int, py: int) -> None:
        """Draw/update dashed crosshair lines at pixel position."""
        cv = self.canvas
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 2 or h < 2:
            return

        # Delete old lines
        for tag in ("_ch_h", "_ch_v"):
            cv.delete(tag)

        # Horizontal line
        cv.create_line(0, py, w, py, dash=(4, 3), fill=FG_DIM,
                       tags=("_ch_h",), width=1)
        # Vertical line
        cv.create_line(px, 0, px, h, dash=(4, 3), fill=FG_DIM,
                       tags=("_ch_v",), width=1)

    def _update_hud(self, candle: Optional[Any], plugin: Any,
                    data_x: float, data_y: float) -> None:
        """Update HUD labels from candle data and plugin overlays."""
        labels = self._hud_labels

        def set_val(key: str, default: str = "--", fmt: str = "{}") -> None:
            val = labels.get(key)
            if val is None:
                return
            v = None
            if candle is not None:
                v = getattr(candle, key.lower(), None)  # open, high, low, close, volume
            if v is None:
                val.set(default)
            else:
                try:
                    val.set(fmt.format(v))
                except Exception:
                    val.set(default)

        # DateTime from data_x (timestamp)
        dt_var = labels.get("DateTime")
        if dt_var:
            try:
                from datetime import datetime
                dt = datetime.fromtimestamp(data_x)
                dt_var.set(dt.strftime("%H:%M:%S"))
            except Exception:
                dt_var.set("--")

        # OHLCV from candle
        set_val("O", "--", "{:.2f}")
        set_val("H", "--", "{:.2f}")
        set_val("L", "--", "{:.2f}")
        set_val("C", "--", "{:.2f}")
        set_val("Volume", "--", "{:,.0f}")

        # Overlay values from plugin cache
        vw = getattr(plugin, "_vwap_cache", [])
        vw_val = labels.get("VWAP")
        if vw_val:
            vw_val.set(f"{vw[-1].vwap:.2f}" if vw else "--")

        ema9_val = labels.get("EMA9")
        if ema9_val:
            ema9 = getattr(plugin, "_ema9_cache", [])
            ema9_val.set(f"{ema9[-1]:.2f}" if ema9 else "--")

        ema21_val = labels.get("EMA21")
        if ema21_val:
            ema21 = getattr(plugin, "_ema21_cache", [])
            ema21_val.set(f"{ema21[-1]:.2f}" if ema21 else "--")

        rsi_val = labels.get("RSI")
        if rsi_val:
            rsi = getattr(plugin, "_rsi_cache", [])
            rsi_val.set(f"{rsi[-1]:.1f}" if rsi else "--")

    def _show(self) -> None:
        if not self.visible:
            self.visible = True

    def _hide(self) -> None:
        """Hide crosshair and HUD."""
        self.visible = False
        try:
            self._hud_canvas.place_forget()
        except Exception:
            pass
        cv = self.canvas
        for tag in ("_ch_h", "_ch_v"):
            cv.delete(tag)

    def destroy(self) -> None:
        self._hide()
        try:
            self.frame.destroy()
        except Exception:
            pass


def _on_key_snapshot(app: Any) -> None:
    """S key — export chart PNG snapshot."""
    try:
        plugin = getattr(app, "live_chart_plugin", None)
        if plugin:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nifty_chart_snapshot_{ts}.png"
            plugin.export_png(filename)
            logger.info("Chart snapshot saved: %s", filename)
    except Exception:
        pass


def _on_key_focus_signal(app: Any) -> None:
    """F key — focus the latest signal marker on chart."""
    try:
        plugin = getattr(app, "live_chart_plugin", None)
        if plugin:
            rows = getattr(plugin, "_prediction_rows", [])
            if rows:
                latest = rows[-1]
                tid = latest.get("prediction_id", "") or latest.get("trade_id", "")
                if tid:
                    plugin.focus_trade(str(tid))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# EMPTY STATE
# ---------------------------------------------------------------------------

def _show_empty_state(app: Any) -> None:
    """Show 'waiting for live market data' overlay on the chart canvas."""
    try:
        plugin = getattr(app, "live_chart_plugin", None)
        if plugin is None:
            return

        # Remove old overlay if present
        _hide_empty_state(app)

        canvas_widget = plugin.canvas.get_tk_widget()
        w, h = canvas_widget.winfo_width() or 800, canvas_widget.winfo_height() or 400

        # Dark overlay frame
        overlay = tk.Frame(canvas_widget, bg=BG_DARK, width=w, height=h)
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        overlay.lift()
        app._lc_empty_overlay = overlay

        # Main message
        broker_status = "DISCONNECTED"
        model_status  = "NOT_LOADED"
        shadow_status = "ON"
        if hasattr(app, "_lc_badge_broker"):
            t = app._lc_badge_broker.cget("text") or ""
            broker_status = t.replace("BROKER: ", "").strip()
        if hasattr(app, "_lc_badge_model"):
            t = app._lc_badge_model.cget("text") or ""
            model_status = t.replace("MODEL: ", "").strip()
        if hasattr(app, "_lc_badge_shadow"):
            shadow_status = app._lc_badge_shadow.cget("text").strip()

        msg = (
            f"\n\n\n  ⏳  Waiting for live market data...  \n\n"
            f"  Broker: {broker_status}   ·   "
            f"Model: {model_status}   ·   "
            f"Shadow: {shadow_status}\n\n"
            f"  Press R to reset view   ·   "
            f"L to toggle lock   ·   "
            f"S for snapshot\n"
        )
        tk.Label(
            overlay, text=msg,
            font=("Segoe UI", 11), fg=FG_MUTED, bg=BG_DARK,
            justify=tk.CENTER, anchor="center"
        ).place(x=w // 2, y=h // 2, anchor="center")

    except Exception as e:
        logger.debug("Failed to show empty state: %s", e)


def _hide_empty_state(app: Any) -> None:
    """Remove the empty state overlay."""
    overlay = getattr(app, "_lc_empty_overlay", None)
    if overlay:
        try:
            overlay.destroy()
        except Exception:
            pass
        app._lc_empty_overlay = None


# ---------------------------------------------------------------------------
# CLICK HANDLER
# ---------------------------------------------------------------------------

def _on_chart_canvas_click(app: Any, event: Any) -> None:
    """Handle click on chart canvas to select a trade/prediction for review.

    Translates pixel event.x and event.y into Matplotlib data coordinate timestamps correctly.
    """
    try:
        plugin = getattr(app, "live_chart_plugin", None)
        if plugin is None:
            return

        # Correctly flip the y coordinate for Matplotlib display height
        canvas_widget = plugin.canvas.get_tk_widget()
        canvas_height = canvas_widget.winfo_height()
        mpl_y = canvas_height - event.y
        
        try:
            data_x, data_y = plugin.ax_price.transData.inverted().transform((int(event.x), int(mpl_y)))
        except Exception:
            # Never fall back to raw event.x — pixel coords are not comparable to date numbers
            logger.debug("Could not invert canvas coordinates, ignoring click")
            return

        candles = getattr(plugin, "_candles", [])
        if not candles:
            return

        # Find nearest candle to data_x
        try:
            xs = mdates.date2num([plugin._candle_dt(c) for c in candles if plugin._candle_dt(c) is not None])
            if len(xs) == 0:
                return
            idx = int(round(np.argmin(np.abs(xs - data_x))))
            clicked_candle = candles[idx]
            clicked_ts = float(getattr(clicked_candle, "time", 0) or 0)
        except Exception:
            return

        # Find nearest prediction row
        prediction_rows = getattr(plugin, "_prediction_rows", [])
        if not prediction_rows:
            return

        nearest_row = min(
            prediction_rows,
            key=lambda r: abs(float(r.get("ts", 0) or r.get("timestamp", 0) or 0) - clicked_ts)
        )

        # Check if within 5 minutes (300 seconds)
        row_ts = float(nearest_row.get("ts", 0) or nearest_row.get("timestamp", 0) or 0)
        if abs(row_ts - clicked_ts) > 300:
            return

        trade_id = nearest_row.get("prediction_id", "") or nearest_row.get("trade_id", "")
        if trade_id:
            plugin.focus_trade(str(trade_id))
            _on_chart_marker_clicked(app, nearest_row)

    except Exception as e:
        logger.debug("Failed canvas click handler: %s", e)


def _on_chart_marker_clicked(app: Any, row: dict[str, Any]) -> None:
    """Populate prediction details when a chart marker is clicked."""
    try:
        # Update current prediction card
        spot = row.get("underlying_price") or row.get("spot")
        latest_pred = _row_to_signal_marker(row, spot)
        if latest_pred:
            snap = build_live_chart_snapshot(latest_prediction=latest_pred)
            _refresh_current_prediction_card(app, snap)
        
        # Update Feature Explainability tab
        _update_explainability_tab(app, row)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Indicator Tuner Tab Card
# ---------------------------------------------------------------------------

def _build_indicator_tuner_tab(app: Any, parent: tk.Frame) -> tk.Frame:
    # Outer frame
    frame = tk.Frame(parent, bg=BG_DARK, padx=6, pady=6)
    frame.pack(fill=tk.BOTH, expand=True)

    # MA & Bands
    lf_ma = tk.LabelFrame(frame, text="📈 Moving Averages & Bands", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    lf_ma.pack(fill=tk.X, pady=(0, 4))

    grid_ma = tk.Frame(lf_ma, bg=BG_CARD)
    grid_ma.pack(fill=tk.X, pady=2)

    app._lc_ema_fast_var = tk.IntVar(value=9)
    app._lc_ema_slow_var = tk.IntVar(value=21)
    app._lc_ema_ultra_var = tk.IntVar(value=50)
    app._lc_ema_super_var = tk.IntVar(value=200)
    app._lc_bb_per_var = tk.IntVar(value=20)
    app._lc_bb_std_var = tk.DoubleVar(value=2.0)

    ma_inputs = [
        ("EMA Fast:", app._lc_ema_fast_var, 2, 100, 1, 0, 0),
        ("EMA Slow:", app._lc_ema_slow_var, 2, 200, 1, 0, 1),
        ("EMA Ultra:", app._lc_ema_ultra_var, 5, 300, 1, 1, 0),
        ("EMA Super:", app._lc_ema_super_var, 10, 500, 1, 1, 1),
        ("BB Period:", app._lc_bb_per_var, 5, 100, 1, 2, 0),
        ("BB Std Dev:", app._lc_bb_std_var, 0.5, 5.0, 0.1, 2, 1),
    ]

    for label_text, var, from_val, to_val, inc, r, c in ma_inputs:
        f = tk.Frame(grid_ma, bg=BG_CARD)
        f.grid(row=r, column=c, sticky="ew", padx=4, pady=2)
        tk.Label(f, text=label_text, width=11, anchor="w", bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        sb = ttk.Spinbox(f, from_=from_val, to=to_val, increment=inc, width=5, textvariable=var)
        sb.pack(side=tk.RIGHT)

    grid_ma.columnconfigure(0, weight=1)
    grid_ma.columnconfigure(1, weight=1)

    # Oscillators & Volatility
    lf_osc = tk.LabelFrame(frame, text="📊 Oscillators & Volatility", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    lf_osc.pack(fill=tk.X, pady=(0, 4))

    grid_osc = tk.Frame(lf_osc, bg=BG_CARD)
    grid_osc.pack(fill=tk.X, pady=2)

    app._lc_rsi_var = tk.IntVar(value=14)
    app._lc_st_per_var = tk.IntVar(value=10)
    app._lc_st_mult_var = tk.DoubleVar(value=3.0)
    app._lc_or_var = tk.IntVar(value=15)

    osc_inputs = [
        ("RSI Period:", app._lc_rsi_var, 2, 50, 1, 0, 0),
        ("OR Period:", app._lc_or_var, 5, 60, 1, 0, 1),
        ("ST Period:", app._lc_st_per_var, 2, 50, 1, 1, 0),
        ("ST Mult:", app._lc_st_mult_var, 0.5, 10.0, 0.1, 1, 1),
    ]

    for label_text, var, from_val, to_val, inc, r, c in osc_inputs:
        f = tk.Frame(grid_osc, bg=BG_CARD)
        f.grid(row=r, column=c, sticky="ew", padx=4, pady=2)
        tk.Label(f, text=label_text, width=11, anchor="w", bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        sb = ttk.Spinbox(f, from_=from_val, to=to_val, increment=inc, width=5, textvariable=var)
        sb.pack(side=tk.RIGHT)

    grid_osc.columnconfigure(0, weight=1)
    grid_osc.columnconfigure(1, weight=1)

    # Action Button
    def apply_tuner_settings():
        if not hasattr(app, "live_chart_plugin") or not app.live_chart_plugin:
            import tkinter.messagebox as messagebox
            messagebox.showwarning("Tuner", "No active chart plugin found!")
            return
        try:
            p = app.live_chart_plugin
            p.ema_fast = int(app._lc_ema_fast_var.get())
            p.ema_slow = int(app._lc_ema_slow_var.get())
            p.ema_ultra = int(app._lc_ema_ultra_var.get())
            p.ema_super = int(app._lc_ema_super_var.get())
            p.bb_period = int(app._lc_bb_per_var.get())
            p.bb_std = float(app._lc_bb_std_var.get())
            p.rsi_period = int(app._lc_rsi_var.get())
            p.supertrend_period = int(app._lc_st_per_var.get())
            p.supertrend_mult = float(app._lc_st_mult_var.get())
            p.opening_range_period = int(app._lc_or_var.get())

            p._ema_fast_line.set_label(f"EMA({p.ema_fast})")
            p._ema_slow_line.set_label(f"EMA({p.ema_slow})")
            p._ema_ultra_line.set_label(f"EMA({p.ema_ultra})")
            p._ema_super_line.set_label(f"EMA({p.ema_super})")
            p._supertrend_line.set_label(f"Supertrend({p.supertrend_period},{p.supertrend_mult:g})")
            p._rsi_line.set_label(f"RSI({p.rsi_period})")
            p._bb_upper_line.set_label(f"BB({p.bb_period},{p.bb_std})")
            
            try:
                p.ax_price.legend(loc="upper left", fontsize=8)
            except Exception:
                pass

            p._indicator_cache.clear()
            p._dirty = True
            p._render()
            
            import tkinter.messagebox as messagebox
            messagebox.showinfo("Tuner", "Indicator parameters applied successfully!")
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Tuner", f"Failed to apply indicator parameters: {e}")

    btn_apply = ttk.Button(frame, text="✅ Apply Indicator Tunings", command=apply_tuner_settings)
    btn_apply.pack(fill=tk.X, pady=10)

    return frame


# ---------------------------------------------------------------------------
# GREEKS PANEL CARD
# ---------------------------------------------------------------------------

def _build_greeks_card(app: Any, parent: tk.Frame) -> tk.LabelFrame:
    card = tk.LabelFrame(parent, text="⚖️ Portfolio Greeks Exposure", bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"), bd=1, relief="solid", padx=6, pady=6)
    card.pack(fill=tk.X, padx=4, pady=(0, 4))

    app._lc_greeks_delta_var = tk.StringVar(value="Delta: --")
    app._lc_greeks_gamma_var = tk.StringVar(value="Gamma: --")
    app._lc_greeks_theta_var = tk.StringVar(value="Theta: --")
    app._lc_greeks_vega_var  = tk.StringVar(value="Vega: --")

    greeks = [
        ("Delta", app._lc_greeks_delta_var, "delta"),
        ("Gamma", app._lc_greeks_gamma_var, "gamma"),
        ("Theta", app._lc_greeks_theta_var, "theta"),
        ("Vega", app._lc_greeks_vega_var, "vega"),
    ]

    app._lc_greeks_canvases = {}
    for label, text_var, key in greeks:
        f = tk.Frame(card, bg=BG_CARD)
        f.pack(fill=tk.X, pady=2)
        
        lbl = tk.Label(f, textvariable=text_var, width=16, anchor="w", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=FG_SECONDARY)
        lbl.pack(side=tk.LEFT, padx=2)
        
        c = tk.Canvas(f, width=100, height=12, bg=BG_DARK, highlightthickness=0)
        c.pack(side=tk.RIGHT, padx=2)
        
        c.create_rectangle(0, 0, 100, 12, fill="#1e293b", outline="")
        c.create_line(50, 0, 50, 12, fill="#64748b", width=1)
        
        app._lc_greeks_canvases[key] = c

    return card


def _update_greeks_gauges(app: Any) -> None:
    try:
        raw_delta = app._greeks_delta_var.get() if hasattr(app, "_greeks_delta_var") else ""
        delta_val = 0.0
        if "Delta:" in raw_delta:
            try:
                delta_val = float(raw_delta.split(":")[-1].strip())
            except Exception:
                pass
        app._lc_greeks_delta_var.set(f"Delta: {delta_val:.2f}")
        _draw_gauge(app._lc_greeks_canvases.get("delta"), delta_val, max_val=100.0)

        raw_gamma = app._greeks_gamma_var.get() if hasattr(app, "_greeks_gamma_var") else ""
        gamma_val = 0.0
        if "Gamma:" in raw_gamma:
            try:
                gamma_val = float(raw_gamma.split(":")[-1].strip())
            except Exception:
                pass
        app._lc_greeks_gamma_var.set(f"Gamma: {gamma_val:.5f}")
        _draw_gauge(app._lc_greeks_canvases.get("gamma"), gamma_val, max_val=0.1)

        raw_theta = app._greeks_theta_var.get() if hasattr(app, "_greeks_theta_var") else ""
        theta_val = 0.0
        if "Theta:" in raw_theta:
            try:
                theta_val = float(raw_theta.split(":")[-1].split("/")[0].strip())
            except Exception:
                pass
        app._lc_greeks_theta_var.set(f"Theta: {theta_val:.2f}/d")
        _draw_gauge(app._lc_greeks_canvases.get("theta"), theta_val, max_val=500.0)

        raw_vega = app._greeks_vega_var.get() if hasattr(app, "_greeks_vega_var") else ""
        vega_val = 0.0
        if "Vega:" in raw_vega:
            try:
                vega_val = float(raw_vega.split(":")[-1].split("/")[0].strip())
            except Exception:
                pass
        app._lc_greeks_vega_var.set(f"Vega: {vega_val:.2f}/1%")
        _draw_gauge(app._lc_greeks_canvases.get("vega"), vega_val, max_val=500.0)

    except Exception:
        pass


def _draw_gauge(canvas: Optional[tk.Canvas], val: float, max_val: float) -> None:
    if canvas is None:
        return
    try:
        canvas.delete("gauge")
        val_clamped = max(-max_val, min(val, max_val))
        pct = val_clamped / max_val
        bar_width = pct * 50
        
        color = "#16a34a" if val >= 0 else "#dc2626"
        
        if bar_width >= 0:
            canvas.create_rectangle(50, 1, 50 + bar_width, 11, fill=color, outline="", tags="gauge")
        else:
            canvas.create_rectangle(50 + bar_width, 1, 50, 11, fill=color, outline="", tags="gauge")
            
        canvas.create_line(50, 0, 50, 12, fill="#ffffff", width=1, tags="gauge")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# MARKET INTELLIGENCE RIGHT PANEL
# ---------------------------------------------------------------------------

def _build_market_intelligence_panel(parent: tk.Frame) -> tk.Frame:
    """Build the scrollable Market Intelligence right panel with 4 sections.

    Sections:
      A. ATM Snapshot        — ATM strike, CE/PE LTP, IV, OI, PCR, spreads, liquidity
      B. Liquidity & Risk    — spread/IV/stale warnings, execution risk badge
      C. Market Structure    — session/PRE day levels, spot vs VWAP
      D. Model Bias          — latest signal, prob, regime, model, confidence, decision
    """
    container = tk.Frame(parent, bg=BG_PANEL)
    container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Scrollable canvas
    canvas = tk.Canvas(container, bg=BG_PANEL, highlightthickness=0,
                       xscrollcommand=lambda *a: None, yscrollcommand=lambda *a: None)
    scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scroll_frame = tk.Frame(canvas, bg=BG_PANEL)
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    # Each section builder
    _build_mi_atm_snapshot(scroll_frame)
    _build_mi_liquidity_risk(scroll_frame)
    _build_mi_market_structure(scroll_frame)
    _build_mi_model_bias(scroll_frame)

    return container


# ── Section A: ATM Snapshot ────────────────────────────────────────────────

def _build_mi_atm_snapshot(parent: tk.Frame) -> None:
    card = tk.LabelFrame(parent, text="A  ·  ATM Option Snapshot",
                         bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"),
                         bd=1, relief="solid", padx=6, pady=4)
    card.pack(fill=tk.X, pady=(0, 6))

    # ATM strike display
    atm_row = tk.Frame(card, bg=BG_CARD)
    atm_row.pack(fill=tk.X, pady=(0, 4))
    tk.Label(atm_row, text="ATM Strike", font=("Segoe UI", 8), fg=FG_DIM, bg=BG_CARD).pack(side=tk.LEFT)
    lbl = tk.Label(atm_row, text="--", font=("Segoe UI", 14, "bold"),
                   fg=C_ATM, bg=BG_CARD)
    lbl.pack(side=tk.RIGHT)
    lbl._app_var = None  # will be set by refresh

    # CE / PE rows
    def _row(label: str, ltp_var: tk.StringVar, iv_var: tk.StringVar,
             oi_var: tk.StringVar) -> None:
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text=label, font=("Segoe UI", 8, "bold"),
                 fg=FG_MUTED, bg=BG_CARD, width=6, anchor="w").pack(side=tk.LEFT)
        tk.Label(row, textvariable=ltp_var, font=("Segoe UI", 9, "bold"),
                 fg=FG_PRIMARY, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, textvariable=iv_var, font=("Segoe UI", 8),
                 fg=FG_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(row, textvariable=oi_var, font=("Segoe UI", 8),
                 fg=FG_DIM, bg=BG_CARD).pack(side=tk.RIGHT)

    app = None
    # We'll get app from the first parent that has it — use a module-level store
    # Store references on module level for refresh
    import sys as _sys
    mod = _sys.modules[__name__]

    mod._lc_mi_atm_var   = tk.StringVar(value="--")
    mod._lc_mi_ce_ltp    = tk.StringVar(value="LTP: --")
    mod._lc_mi_pe_ltp    = tk.StringVar(value="LTP: --")
    mod._lc_mi_ce_iv     = tk.StringVar(value="IV: --")
    mod._lc_mi_pe_iv     = tk.StringVar(value="IV: --")
    mod._lc_mi_ce_oi     = tk.StringVar(value="OI: --")
    mod._lc_mi_pe_oi     = tk.StringVar(value="OI: --")
    mod._lc_mi_pcr       = tk.StringVar(value="PCR: --")
    mod._lc_mi_spread    = tk.StringVar(value="Spread: --")
    mod._lc_mi_liq       = tk.StringVar(value="Liquidity: --")

    _row("CE", mod._lc_mi_ce_ltp, mod._lc_mi_ce_iv, mod._lc_mi_ce_oi)
    _row("PE", mod._lc_mi_pe_ltp, mod._lc_mi_pe_iv, mod._lc_mi_pe_oi)

    # PCR, Spread, Liquidity
    for var in [mod._lc_mi_pcr, mod._lc_mi_spread, mod._lc_mi_liq]:
        tk.Label(card, textvariable=var, font=("Segoe UI", 8),
                 fg=FG_MUTED, bg=BG_CARD, anchor="w").pack(fill=tk.X, pady=1)


# ── Section B: Liquidity & Execution Risk ──────────────────────────────────

def _build_mi_liquidity_risk(parent: tk.Frame) -> None:
    card = tk.LabelFrame(parent, text="B  ·  Liquidity & Execution",
                         bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"),
                         bd=1, relief="solid", padx=6, pady=4)
    card.pack(fill=tk.X, pady=(0, 6))

    import sys as _sys
    mod = _sys.modules[__name__]

    mod._lc_mi_wide_spread  = tk.StringVar(value="  Wide Spread: --")
    mod._lc_mi_stale_data   = tk.StringVar(value="  Stale Data:   --")
    mod._lc_mi_iv_spike     = tk.StringVar(value="  IV Spike:     --")
    mod._lc_mi_exec_risk    = tk.StringVar(value="  Exec Risk:    --")

    warn_color_map = {"WIDE": C_BLOCKED, "STALE": C_BLOCKED, "IV": C_BLOCKED,
                      "OK": C_BULL, "WARNING": STATUS_WARN, "DANGER": STATUS_CRIT}

    for var in [mod._lc_mi_wide_spread, mod._lc_mi_stale_data,
                mod._lc_mi_iv_spike, mod._lc_mi_exec_risk]:
        lbl = tk.Label(card, textvariable=var, font=("Segoe UI", 8),
                       fg=FG_MUTED, bg=BG_CARD, anchor="w")
        lbl.pack(fill=tk.X, pady=1)


# ── Section C: Market Structure ────────────────────────────────────────────

def _build_mi_market_structure(parent: tk.Frame) -> None:
    card = tk.LabelFrame(parent, text="C  ·  Market Structure",
                         bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"),
                         bd=1, relief="solid", padx=6, pady=4)
    card.pack(fill=tk.X, pady=(0, 6))

    import sys as _sys
    mod = _sys.modules[__name__]

    mod._lc_mi_sess_high   = tk.StringVar(value="Sess High:   --")
    mod._lc_mi_sess_low    = tk.StringVar(value="Sess Low:    --")
    mod._lc_mi_pdh         = tk.StringVar(value="PRE Day H:   --")
    mod._lc_mi_pdl         = tk.StringVar(value="PRE Day L:   --")
    mod._lc_mi_or_high     = tk.StringVar(value="OR High:     --")
    mod._lc_mi_or_low      = tk.StringVar(value="OR Low:      --")
    mod._lc_mi_vwap_diff   = tk.StringVar(value="Spot vs VWAP: --")

    for var in [mod._lc_mi_sess_high, mod._lc_mi_sess_low, mod._lc_mi_pdh,
                mod._lc_mi_pdl, mod._lc_mi_or_high, mod._lc_mi_or_low, mod._lc_mi_vwap_diff]:
        tk.Label(card, textvariable=var, font=("Segoe UI", 8),
                 fg=FG_MUTED, bg=BG_CARD, anchor="w").pack(fill=tk.X, pady=1)


# ── Section D: Current Model Bias ─────────────────────────────────────────

def _build_mi_model_bias(parent: tk.Frame) -> None:
    card = tk.LabelFrame(parent, text="D  ·  Model Bias",
                         bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 9, "bold"),
                         bd=1, relief="solid", padx=6, pady=4)
    card.pack(fill=tk.X, pady=(0, 6))

    import sys as _sys
    mod = _sys.modules[__name__]

    mod._lc_mi_signal      = tk.StringVar(value="Signal:    --")
    mod._lc_mi_prob        = tk.StringVar(value="Prob:      --")
    mod._lc_mi_conf        = tk.StringVar(value="Conf:      --")
    mod._lc_mi_regime      = tk.StringVar(value="Regime:    --")
    mod._lc_mi_model       = tk.StringVar(value="Model:     --")
    mod._lc_mi_thresh      = tk.StringVar(value="Thresh:    --")
    mod._lc_mi_decision    = tk.StringVar(value="Decision:  --")

    for var in [mod._lc_mi_signal, mod._lc_mi_prob, mod._lc_mi_conf,
                mod._lc_mi_regime, mod._lc_mi_model, mod._lc_mi_thresh]:
        tk.Label(card, textvariable=var, font=("Segoe UI", 8),
                 fg=FG_MUTED, bg=BG_CARD, anchor="w").pack(fill=tk.X, pady=1)

    # Decision badge
    mod._lc_mi_decision_lbl = tk.Label(card, textvariable=mod._lc_mi_decision,
                                       font=("Segoe UI", 9, "bold"),
                                       anchor="center", relief=tk.RIDGE, bd=1,
                                       padx=4, pady=2, bg=BG_CARD, fg=FG_PRIMARY)
    mod._lc_mi_decision_lbl.pack(fill=tk.X, pady=(4, 0))


# ---------------------------------------------------------------------------
# REFRESH: Market Intelligence
# ---------------------------------------------------------------------------

def _refresh_market_intelligence(app: Any, snap: LiveChartSnapshot) -> None:
    """Refresh all Market Intelligence panel vars from the snapshot."""
    import sys as _sys
    mod = _sys.modules[__name__]

    def _v(name: str):
        return getattr(mod, name, None)

    def _safe(name: str, default="--") -> str:
        var = _v(name)
        return var.get() if var else default

    def _set(name: str, value: str) -> None:
        var = _v(name)
        if var:
            var.set(value)

    oc = snap.option_chain_summary
    dh = snap.data_health
    sp = snap.shadow_metrics

    # Section A: ATM Snapshot
    atm = snap.atm_strike
    _set("_lc_mi_atm_var", f"{atm:.0f}" if atm else "--")
    _set("_lc_mi_ce_ltp",  f"LTP: {oc.ce_ltp:.2f}" if oc and oc.ce_ltp else "LTP: --")
    _set("_lc_mi_pe_ltp",  f"LTP: {oc.pe_ltp:.2f}" if oc and oc.pe_ltp else "LTP: --")
    _set("_lc_mi_ce_iv",   f"IV: {oc.ce_iv:.1f}%" if oc and oc.ce_iv else "IV: --")
    _set("_lc_mi_pe_iv",   f"IV: {oc.pe_iv:.1f}%" if oc and oc.pe_iv else "IV: --")
    _set("_lc_mi_ce_oi",   f"OI: {oc.ce_oi:,.0f}" if oc and oc.ce_oi else "OI: --")
    _set("_lc_mi_pe_oi",   f"OI: {oc.pe_oi:,.0f}" if oc and oc.pe_oi else "OI: --")
    _set("_lc_mi_pcr",     f"PCR: {oc.pcr:.2f}" if oc and oc.pcr else "PCR: --")
    ce_sp = oc.ce_bid_ask_spread if oc else None
    pe_sp = oc.pe_bid_ask_spread if oc else None
    best_sp = min(filter(None, [ce_sp, pe_sp]), default=None)
    _set("_lc_mi_spread",  f"Spread: {best_sp:.2f}" if best_sp else "Spread: --")
    liq = oc.liquidity_score if oc else "N/A"
    _set("_lc_mi_liq",     f"Liquidity: {liq}")

    # Section B: Liquidity & Risk
    ws = oc.wide_spread_warning if oc else False
    st = (dh.stale_data_seconds >= 30) if dh else False
    iv = oc.iv_spike_warning if oc else False
    liq_ok = (oc.liquidity_score == "OK") if oc else False

    _set("_lc_mi_wide_spread", f"  Wide Spread:  {'⚠ YES' if ws else '✗ No'}")
    stale_s = dh.stale_data_seconds if dh else 0
    _set("_lc_mi_stale_data",   f"  Stale Data:   {'⚠ ' + str(stale_s) + 's' if stale_s >= 30 else '✗ OK'}")
    _set("_lc_mi_iv_spike",     f"  IV Spike:     {'⚠ YES' if iv else '✗ No'}")

    if ws or stale_s >= 60 or iv or (not liq_ok and liq == "DANGER"):
        risk = "DANGER"
        risk_fg = STATUS_CRIT
    elif st or liq == "WARNING":
        risk = "WARNING"
        risk_fg = STATUS_WARN
    else:
        risk = "OK"
        risk_fg = STATUS_OK
    _set("_lc_mi_exec_risk", f"  Exec Risk:    {risk}")
    # Color the exec risk label if possible
    try:
        pass  # color applied in the label via fg update below
    except Exception:
        pass

    # Section C: Market Structure
    _set("_lc_mi_sess_high", f"Sess High:   {snap.session_high:.2f}" if snap.session_high else "Sess High:   --")
    _set("_lc_mi_sess_low",  f"Sess Low:    {snap.session_low:.2f}"  if snap.session_low  else "Sess Low:    --")
    _set("_lc_mi_pdh",       f"PRE Day H:   {snap.previous_day_high:.2f}" if snap.previous_day_high else "PRE Day H:   --")
    _set("_lc_mi_pdl",       f"PRE Day L:   {snap.previous_day_low:.2f}"  if snap.previous_day_low else "PRE Day L:   --")
    _set("_lc_mi_or_high",   f"OR High:     {snap.opening_range_high:.2f}" if snap.opening_range_high else "OR High:     --")
    _set("_lc_mi_or_low",    f"OR Low:      {snap.opening_range_low:.2f}"  if snap.opening_range_low else "OR Low:      --")
    vwap_val = getattr(snap, "vwap", None) or 0.0
    spot_val = snap.spot_price or 0.0
    if vwap_val and spot_val:
        diff = spot_val - vwap_val
        sign = "▲" if diff >= 0 else "▼"
        _set("_lc_mi_vwap_diff", f"Spot vs VWAP: {sign} {abs(diff):.2f}")
    else:
        _set("_lc_mi_vwap_diff", "Spot vs VWAP: --")

    # Section D: Model Bias
    lp = snap.latest_prediction
    if lp:
        _set("_lc_mi_signal",  f"Signal:    {lp.side} {lp.instrument or ''}")
        _set("_lc_mi_prob",    f"Prob:      {lp.probability * 100:.1f}%")
        _set("_lc_mi_conf",    f"Conf:      {lp.confidence * 100:.1f}%")
        _set("_lc_mi_regime",  f"Regime:    {snap.regime_label or '--'}")
        _set("_lc_mi_model",   f"Model:     {lp.model_name}")
        _set("_lc_mi_thresh",  f"Thresh:    {lp.threshold:.2f}")
        decision = "ACCEPT" if lp.status == "executed" else ("SHADOW" if lp.status == "shadow" else "BLOCKED")
        _set("_lc_mi_decision", f"  {decision}  ")
        dec_fg = C_BULL if decision == "ACCEPT" else (C_SHADOW if decision == "SHADOW" else C_BLOCKED)
        dec_lbl = _v("_lc_mi_decision_lbl")
        if dec_lbl:
            dec_lbl.configure(fg=dec_fg)
    else:
        for name in ["_lc_mi_signal", "_lc_mi_prob", "_lc_mi_conf",
                     "_lc_mi_regime", "_lc_mi_model", "_lc_mi_thresh"]:
            label = name.split("_")[-1].capitalize()
            _set(name, f"{label:8s}    --")
        _set("_lc_mi_decision", "  --  ")
        dec_lbl = _v("_lc_mi_decision_lbl")
        if dec_lbl:
            dec_lbl.configure(fg=FG_DIM)


# ---------------------------------------------------------------------------
# Chart Toolbar
# ---------------------------------------------------------------------------

def _build_chart_toolbar(app: Any, parent: tk.Frame, main_pw: tk.PanedWindow,
                             right_cards: tk.Frame, bottom_notebook: ttk.Notebook) -> None:
    """Build the premium institutional toolbar for the Live Chart.

    Sections: [TF + Candle Count] | [Overlays] | [Signal Filter] | [Actions] | [Maximize]
    """
    parent.configure(bg=BG_CARD)

    def _sep() -> None:
        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

    plugin = getattr(app, "live_chart_plugin", None)

    # ── Section 1: Timeframe + Candle Count ────────────────────────
    if not hasattr(app, "_lc_tf_var"):
        app._lc_tf_var = tk.StringVar(value=getattr(plugin, "timeframe", "1m") if plugin else "1m")
    if not hasattr(app, "_lc_candles_var"):
        app._lc_candles_var = tk.StringVar(value="Full")

    def _on_tf_changed() -> None:
        tf = app._lc_tf_var.get()
        if hasattr(app, "_lc_timeframe_var"):
            app._lc_timeframe_var.set(tf)
        if plugin:
            import os
            scalper = getattr(app, "_scalper", None)
            underlying = (
                getattr(scalper.cfg, "underlying", None)
                if scalper and hasattr(scalper, "cfg")
                else None
            ) or os.getenv("MSTOCK_UNDERLYING") or os.getenv("MSTOCK_SYMBOL") or "NIFTY"
            plugin.set_symbol(underlying.strip(), tf)
            _refresh_live_chart_tab(app, force=True)

    def _on_candles_changed(val: str) -> None:
        if plugin:
            count = None if val == "Full" else int(val)
            plugin.set_max_candles(count) if count else plugin.set_max_candles(None)
            _refresh_live_chart_tab(app, force=True)

    tk.Label(parent, text="TF:", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.LEFT, padx=(4, 2))
    for tf in ["1m", "3m", "5m", "15m"]:
        ttk.Radiobutton(parent, text=tf.upper(), value=tf, variable=app._lc_tf_var,
                        style="Toolbutton", command=_on_tf_changed).pack(side=tk.LEFT, padx=1)

    tk.Label(parent, text="Bars:", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.LEFT, padx=(8, 2))
    for opt, val in [("100", "100"), ("250", "250"), ("500", "500"), ("Full", "Full")]:
        ttk.Radiobutton(parent, text=opt, value=val, variable=app._lc_candles_var,
                        style="Toolbutton",
                        command=lambda v=val: _on_candles_changed(v)).pack(side=tk.LEFT, padx=1)

    _sep()

    # ── Section 2: Overlay Toggles ─────────────────────────────────
    if not hasattr(app, "_lc_overlay_vars"):
        app._lc_overlay_vars = {}

    overlay_defs = [
        # (label,  key,          default)
        ("VWAP",  "vwap",       True),
        ("EMA9",  "ema_fast",   True),
        ("EMA21", "ema_slow",   True),
        ("EMA50", "ema_ultra",  True),
        ("EMA200","ema_super",  True),
        ("BB",    "bollinger",  True),
        ("OR",    "opening_range", True),
        ("PDH/L", "prevday_hl", True),
        ("Sess H/L","session_hl",True),
        ("ATM",   "atm",        True),
        ("Vol",   "volume",     True),
        ("Sig",   "signals",    True),
    ]

    def _toggle_overlay(key: str, var: tk.BooleanVar) -> None:
        if plugin:
            plugin.set_overlay_visible(key, var.get())
            _refresh_live_chart_tab(app, force=True)

    tk.Label(parent, text="Overlays:", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.LEFT, padx=(2, 2))

    for label, key, default in overlay_defs:
        if key not in app._lc_overlay_vars:
            app._lc_overlay_vars[key] = tk.BooleanVar(value=default)
        var = app._lc_overlay_vars[key]
        ttk.Checkbutton(parent, text=label, variable=var, style="Toolbutton",
                        command=lambda k=key, v=var: _toggle_overlay(k, v)).pack(side=tk.LEFT, padx=1)

    _sep()

    # ── Section 3: Signal Filter ───────────────────────────────────
    if not hasattr(app, "_lc_signal_filter_var"):
        app._lc_signal_filter_var = tk.StringVar(value="All|all")

    tk.Label(parent, text="Filter:", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.LEFT, padx=(2, 2))

    sig_options = ["all", "executed", "blocked", "shadow", "buy_only", "sell_only"]
    sig_labels  = {"all": "All", "executed": "Exec", "blocked": "Block",
                   "shadow": "Shadow", "buy_only": "Buy", "sell_only": "Sell"}

    def _on_sig_filter_changed(display_val: str) -> None:
        # display_val is "All|all" format — extract key after |
        if plugin:
            key = display_val.split("|")[-1] if "|" in display_val else display_val
            plugin.set_signal_filter(key)

    sig_menu = ttk.OptionMenu(parent, app._lc_signal_filter_var, "All|all",
                              *[f"{sig_labels[k]}|{k}" for k in sig_options],
                              command=_on_sig_filter_changed)
    sig_menu.pack(side=tk.LEFT, padx=2)

    _sep()

    # ── Section 4: Action Buttons ──────────────────────────────────
    def _btn(text: str, command, side=tk.LEFT) -> None:
        ttk.Button(parent, text=text, style="Toolbutton", command=command).pack(side=side, padx=2)

    def _snapshot() -> None:
        if plugin:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            plugin.export_png(f"nifty_chart_snapshot_{ts}.png")
            logger.info("Chart snapshot saved.")

    def _export_csv() -> None:
        """Export recent signals to CSV."""
        try:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV", "*.csv")],
                                                initialfile="nifty_signals.csv")
            if not path:
                return
            rows = getattr(plugin, "_prediction_rows", []) or []
            if not rows:
                logger.debug("No prediction rows to export.")
                return
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Signals exported to %s", path)
        except Exception as e:
            logger.debug("Export CSV failed: %s", e)

    def _focus_latest() -> None:
        if plugin and hasattr(plugin, "focus_latest_signal"):
            plugin.focus_latest_signal()

    def _clear_focus() -> None:
        if plugin:
            plugin.focus_trade(None)

    _btn("📸 PNG",   _snapshot)
    _btn("📤 CSV",  _export_csv)
    _btn("🔄 Reset", lambda: plugin.reset_view() if plugin and hasattr(plugin, "reset_view") else None)
    _btn("🎯 Signal", _focus_latest)
    _btn("✕ Clear",  _clear_focus)

    # Lock Live
    if not hasattr(app, "_lc_lock_var"):
        app._lc_lock_var = tk.BooleanVar(value=True)
    cb_lock = ttk.Checkbutton(parent, text="🔒 Live", variable=app._lc_lock_var, style="Toolbutton",
                              command=lambda: plugin.set_lock_to_live(app._lc_lock_var.get()) if plugin else None)
    cb_lock.pack(side=tk.LEFT, padx=2)

    # Maximize (right-aligned)
    app._lc_is_maximized = False

    def _toggle_maximize() -> None:
        if app._lc_is_maximized:
            main_pw.add(right_cards)
            bottom_notebook.pack(fill=tk.BOTH, expand=False, padx=4, pady=(0, 4), side=tk.BOTTOM)
            btn_max.configure(text="🖥 Max")
            app._lc_is_maximized = False
        else:
            main_pw.forget(right_cards)
            bottom_notebook.pack_forget()
            btn_max.configure(text="🗖 Rest")
            app._lc_is_maximized = True

    btn_max = ttk.Button(parent, text="🖥 Max", style="Toolbutton", command=_toggle_maximize)
    btn_max.pack(side=tk.RIGHT, padx=(4, 4))


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _get_latest_vwap(app: Any) -> Optional[float]:
    try:
        cache = getattr(app.live_chart_plugin, "_vwap_cache", [])
        if cache:
            for v in reversed(cache):
                if v is not None:
                    return float(v)
    except Exception:
        pass
    return None


def _get_latest_ema(app: Any, period: int) -> Optional[float]:
    try:
        cache_key = f"ema_{period}"
        cache = getattr(app.live_chart_plugin, "_indicator_cache", {})
        series = cache.get(cache_key, {}).get("series", [])
        if series:
            for v in reversed(series):
                if v is not None:
                    return float(v)
    except Exception:
        pass
    return None


def _row_to_signal_marker(row: dict[str, Any], spot: Optional[float]) -> Optional[SignalMarker]:
    try:
        ts_val = row.get("ts") or row.get("timestamp")
        if ts_val is None:
            return None
        ts_dt = datetime.fromtimestamp(float(ts_val)) if isinstance(ts_val, (int, float)) else datetime.now()

        prob = float(row.get("probability", 0))
        pred_class = int(row.get("predicted_class", -1))
        trade_candidate = str(row.get("trade_candidate", "")).upper()
        trade_taken = row.get("trade_taken") in (True, 1, "1", "true", "True")

        if pred_class == 1 or "LONG" in trade_candidate or "BUY" in trade_candidate:
            side = "BUY"
            instrument = "CE"
        elif pred_class == 0 or "SHORT" in trade_candidate or "SELL" in trade_candidate:
            side = "SELL"
            instrument = "PE"
        else:
            side = "EXIT"
            instrument = ""

        confidence = min(abs(prob - 0.5) * 2, 1.0)  # 0–1 scale

        return SignalMarker(
            timestamp=ts_dt,
            price=spot or 0.0,
            side=side,
            instrument=instrument,
            strike=row.get("strike"),
            probability=prob,
            confidence=confidence,
            model_name=str(row.get("model_name", "N/A")),
            threshold=row.get("threshold", 0.5),
            expected_edge=row.get("expected_edge"),
            reason=str(row.get("reason", "") or row.get("regime", "")),
            status="executed" if trade_taken else "shadow",
            trade_id=str(row.get("trade_id", "") or ""),
        )
    except Exception:
        return None


def _guess_market_status() -> str:
    """Heuristic market status based on current time (Asia/Kolkata)."""
    try:
        now = datetime.now()
        # Asia/Kolkata is UTC+5:30 — use hour in IST
        ist_hour = (now.hour + 5) % 24
        ist_min = now.minute

        if ist_hour < 9 or (ist_hour == 9 and ist_min < 15):
            return "PRE_OPEN"
        elif ist_hour >= 15 and ist_min >= 30:
            return "POST_CLOSE"
        elif 9 <= ist_hour < 15:
            return "OPEN"
        else:
            return "CLOSED"
    except Exception:
        return "CLOSED"


def _set_all_cards_collecting(app: Any) -> None:
    """Show collecting-data state in all cards when no data is available yet."""
    try:
        _set_prediction_card_collecting(app)
        _set_shadow_card_collecting(app)

        app._lc_oc_atm_var.set("ATM Strike: --")
        app._lc_oc_ce_ltp_var.set("CE LTP: --")
        app._lc_oc_pe_ltp_var.set("PE LTP: --")
        app._lc_oc_ce_iv_var.set("CE IV: --")
        app._lc_oc_pe_iv_var.set("PE IV: --")
        app._lc_oc_ce_oi_var.set("CE OI: --")
        app._lc_oc_pe_oi_var.set("PE OI: --")
        app._lc_oc_pcr_var.set("PCR: --")
        app._lc_oc_spread_var.set("ATM Spread: --")
        app._lc_oc_liq_var.set("Liquidity Score: --")
        app._lc_oc_warning_var.set("Warnings: None")

        app._lc_dh_broker_var.set("Broker Connection: --")
        app._lc_dh_candle_age_var.set("Last Candle: --")
        app._lc_dh_tick_age_var.set("Last Tick: --")
        app._lc_dh_stale_sec_var.set("Stale Seconds: --")
        app._lc_dh_model_var.set("Model Registry: --")
        app._lc_dh_shadow_var.set("Shadow Log Queue: --")
        app._lc_dh_db_var.set("DB Write Health: --")
        app._lc_dh_missing_var.set("Missing Candles: 0")
        app._lc_dh_api_err_var.set("API Error Count: 0")
        app._lc_dh_reason_var.set("Status Reason: None")
        app._lc_dh_status_var.set("Overall Status: --")
        app._lc_dh_status_badge.configure(text="  --  ", background="#374151", foreground="white")
    except Exception:
        pass


def _set_prediction_card_collecting(app: Any) -> None:
    try:
        app._lc_pred_direction_var.set("Direction: --")
        app._lc_pred_probability_var.set("Probability: --")
        app._lc_pred_confidence_var.set("Confidence: --")
        app._lc_pred_threshold_var.set("Threshold: --")
        app._lc_pred_model_var.set("Model: --")
        app._lc_pred_time_var.set("Time: --")
        app._lc_pred_edge_var.set("Expected Edge: --")
        app._lc_pred_regime_var.set("Regime: --")
        app._lc_pred_liq_var.set("Liquidity Score: --")
        app._lc_pred_spread_var.set("Bid-Ask Spread: --")
        app._lc_pred_decision_var.set("Decision: --")
        app._lc_pred_decision_lbl.configure(text="  Collecting...  ", background="#374151", foreground="white")
    except Exception:
        pass


def _set_shadow_card_collecting(app: Any) -> None:
    try:
        app._lc_shad_preds_var.set("--")
        app._lc_shad_accepted_var.set("--")
        app._lc_shad_blocked_var.set("--")
        app._lc_shad_resolved_var.set("--")
        app._lc_shad_winrate_var.set("--")
        app._lc_shad_pf_var.set("--")
        app._lc_shad_exp_var.set("--")
        app._lc_shad_mdd_var.set("--")
        app._lc_shad_pnl_var.set("--")
        app._lc_shad_drift_var.set("Collecting...")
    except Exception:
        pass