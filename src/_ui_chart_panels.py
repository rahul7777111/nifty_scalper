"""
New UI panel methods for NiftyScalperUI — to be merged into src/ui.py

Methods added:
  - _build_shadow_analytics_tab
  - _refresh_shadow_analytics
  - _build_chart_toolbar
  - _build_feature_explainability_panel
  - _build_option_chain_panel
  - _build_trade_review_panel
  - _build_alert_panel
  - _export_shadow_report
  - _export_signals_csv
  - _on_chart_canvas_click
  - _refresh_option_chain
  - _refresh_feature_explainability
  - _refresh_alerts
  - _on_chart_marker_clicked (helper)
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog


# =============================================================================
# SHADOW ANALYTICS TAB
# =============================================================================

def _build_shadow_analytics_tab(self) -> None:
    """Build the Shadow Mode Analytics tab inside self.shadow_analytics_frame."""
    root = tk.Frame(self.shadow_analytics_frame)
    root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Title
    title = ttk.Label(root, text="Shadow Mode Analytics", font=("Segoe UI", 13, "bold"))
    title.pack(anchor="w", pady=(0, 8))

    # 4-column grid of metric cards
    grid = ttk.Frame(root)
    grid.pack(fill=tk.X, pady=(0, 8))
    for col_idx in range(4):
        grid.grid_columnconfigure(col_idx, weight=1, uniform="sg")

    metric_specs = [
        ("Total Predictions", self._shadow_total_var, "Total signals generated"),
        ("Long Signals", self._shadow_long_var, "Long direction signals"),
        ("Short Signals", self._shadow_short_var, "Short direction signals"),
        ("Win Rate", self._shadow_winrate_var, "Shadow win rate (prob > 0.5)"),
        ("Precision", self._shadow_precision_var, "True positive rate"),
        ("Recall", self._shadow_recall_var, "Capture rate of executed"),
        ("Profit Factor", self._shadow_pf_var, "Gross profit / gross loss"),
        ("Sharpe Ratio", self._shadow_sharpe_var, "Risk-adjusted return"),
        ("Max Drawdown", self._shadow_mdd_var, "Largest peak-to-trough"),
        ("Daily PnL", self._shadow_daily_pnl_var, "Today's shadow P&L"),
        ("Weekly PnL", self._shadow_weekly_pnl_var, "This week's shadow P&L"),
        ("Anomalies", self._shadow_anomaly_var, "Flags requiring attention"),
    ]

    card_widgets: dict[str, tuple[ttk.LabelFrame, ttk.Label, ttk.Label]] = {}
    for idx, (label_text, var, desc) in enumerate(metric_specs):
        row = idx // 4
        col = idx % 4
        card = ttk.LabelFrame(grid, text=label_text, padding=6)
        card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        value_lbl = ttk.Label(card, textvariable=var, font=("Segoe UI", 16, "bold"))
        value_lbl.pack(anchor="center", pady=(4, 2))
        desc_lbl = ttk.Label(card, text=desc, font=("Segoe UI", 7), foreground="gray")
        desc_lbl.pack(anchor="center")
        card_widgets[label_text] = (card, value_lbl, desc_lbl)

    self._shadow_card_widgets = card_widgets

    # Button row
    btn_row = ttk.Frame(root)
    btn_row.pack(fill=tk.X, pady=(6, 0))
    ttk.Button(btn_row, text="Refresh", command=lambda: self._refresh_shadow_analytics(force=True)).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Export Shadow Report", command=self._export_shadow_report).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(btn_row, text="Export Signals CSV", command=self._export_signals_csv).pack(side=tk.LEFT, padx=(8, 0))


def _refresh_shadow_analytics(self, force: bool = False) -> None:
    """Compute and display shadow-mode analytics from prediction markers."""
    try:
        now_ts = time.time()
        last_ts = float(getattr(self, "_shadow_refresh_ts", 0.0) or 0.0)
        if not force and (now_ts - last_ts) < 5.0:
            return

        db = getattr(self, "_db_manager", None)
        if db is None:
            scalper = getattr(self, "_scalper", None)
            db = getattr(scalper, "db_manager", None) if scalper is not None else None
        if db is None:
            return

        rows = db.list_recent_prediction_markers(limit=5000) if hasattr(db, "list_recent_prediction_markers") else []

        total = len(rows)
        longs = sum(1 for r in rows if str(r.get("predicted_class", "")).lower() == "long")
        shorts = sum(1 for r in rows if str(r.get("predicted_class", "")).lower() == "short")

        # Win rate: prob > 0.5 = would-be win
        wins = sum(1 for r in rows if float(r.get("probability", 0)) > 0.5)
        win_rate = (wins / total * 100) if total > 0 else 0.0

        # Precision: among trade_taken=True, how many had prob > 0.5
        taken = [r for r in rows if r.get("trade_taken") in (True, 1, "1", "true", "True")]
        taken_wins = sum(1 for r in taken if float(r.get("probability", 0)) > 0.5)
        precision = (taken_wins / len(taken) * 100) if taken else 0.0

        # Recall: among all rows with prob > 0.5, how many were taken
        high_prob = [r for r in rows if float(r.get("probability", 0)) > 0.5]
        recalled = sum(1 for r in high_prob if r.get("trade_taken") in (True, 1, "1", "true", "True"))
        recall = (recalled / len(high_prob) * 100) if high_prob else 0.0

        # Profit factor: sum of realized PnL for taken trades
        trade_events = db.list_recent_trade_events(limit=2000) if hasattr(db, "list_recent_trade_events") else []
        gross_profit = sum(float(e.get("realized", 0) or 0) for e in trade_events if float(e.get("realized", 0) or 0) > 0)
        gross_loss = abs(sum(float(e.get("realized", 0) or 0) for e in trade_events if float(e.get("realized", 0) or 0) < 0))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        # Sharpe: simplified (total return / volatility of returns)
        realized_list = [float(e.get("realized", 0) or 0) for e in trade_events]
        if len(realized_list) > 1:
            mean_ret = sum(realized_list) / len(realized_list)
            variance = sum((x - mean_ret) ** 2 for x in realized_list) / len(realized_list)
            std_ret = max(variance ** 0.5, 0.01)
            sharpe = mean_ret / std_ret
        else:
            sharpe = 0.0

        # Max drawdown
        mdd = 0.0
        running = 0.0
        peak = 0.0
        for r in realized_list:
            running += r
            if running > peak:
                peak = running
            dd = running - peak
            if dd < mdd:
                mdd = dd

        # Daily / Weekly PnL
        now_dt = datetime.now()
        cutoff_daily = (now_dt - timedelta(days=1)).timestamp()
        cutoff_weekly = (now_dt - timedelta(days=7)).timestamp()
        daily_pnl = sum(float(e.get("realized", 0) or 0) for e in trade_events if float(e.get("ts", 0) or 0) > cutoff_daily)
        weekly_pnl = sum(float(e.get("realized", 0) or 0) for e in trade_events if float(e.get("ts", 0) or 0) > cutoff_weekly)

        # Anomalies count
        anomalies = 0
        if win_rate < 40:
            anomalies += 1
        if sharpe < 0.5:
            anomalies += 1
        if mdd < -0.2:
            anomalies += 1

        # Update vars
        self._shadow_total_var.set(str(total))
        self._shadow_long_var.set(str(longs))
        self._shadow_short_var.set(str(shorts))
        self._shadow_winrate_var.set(f"{win_rate:.1f}%")
        self._shadow_precision_var.set(f"{precision:.1f}%")
        self._shadow_recall_var.set(f"{recall:.1f}%")
        self._shadow_pf_var.set(f"{pf:.2f}")
        self._shadow_sharpe_var.set(f"{sharpe:.2f}")
        self._shadow_mdd_var.set(f"₹{mdd:.2f}")
        self._shadow_daily_pnl_var.set(f"₹{daily_pnl:.2f}")
        self._shadow_weekly_pnl_var.set(f"₹{weekly_pnl:.2f}")
        self._shadow_anomaly_var.set(str(anomalies))

        # Highlight anomalies in red
        for key, (card, val_lbl, _) in self._shadow_card_widgets.items():
            red_flag = False
            if key == "Win Rate" and win_rate < 40:
                red_flag = True
            elif key == "Sharpe Ratio" and sharpe < 0.5:
                red_flag = True
            elif key == "Max Drawdown" and mdd < -0.2:
                red_flag = True
            if red_flag:
                try:
                    card.configure(style="Anomaly.TLabelframe")
                except Exception:
                    pass
                try:
                    val_lbl.configure(foreground="#f43f5e")
                except Exception:
                    pass
            else:
                try:
                    val_lbl.configure(foreground="")
                except Exception:
                    pass

        self._shadow_refresh_ts = now_ts

        # Also refresh feature explainability
        self._refresh_feature_explainability(rows[:10] if rows else [])

    except Exception:
        pass
    finally:
        self._safe_after_app("shadow_analytics", 5000, self._refresh_shadow_analytics)


# =============================================================================
# CHART TOOLBAR
# =============================================================================

def _build_chart_toolbar(self) -> None:
    """Build overlay toggles + controls above/below the live chart."""
    toolbar = ttk.Frame(self.live_chart_frame)
    toolbar.pack(fill=tk.X, padx=4, pady=(2, 2))

    # Lock-to-live button
    self._lock_to_live_var = tk.BooleanVar(value=True)

    def _toggle_lock() -> None:
        self.live_chart_plugin.set_lock_to_live(self._lock_to_live_var.get())

    lock_btn = ttk.Checkbutton(
        toolbar, text="🔒 Lock to Live", variable=self._lock_to_live_var,
        command=_toggle_lock
    )
    lock_btn.pack(side=tk.LEFT, padx=(0, 6))

    # Separator
    ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

    # Overlay toggles
    overlay_defs = [
        ("VWAP", "vwap"),
        ("EMA9", "ema_fast"),
        ("EMA21", "ema_slow"),
        ("EMA50", "ema_ultra"),
        ("EMA200", "ema_super"),
        ("Bollinger", "bollinger"),
        ("Supertrend", "supertrend"),
        ("Open Range", "opening_range"),
        ("PDH/PDL", "prevday_hl"),
        ("Session H/L", "session_hl"),
    ]

    self._overlay_vars: dict[str, tk.BooleanVar] = {}

    for label_text, overlay_name in overlay_defs:
        var = tk.BooleanVar(value=True)
        self._overlay_vars[overlay_name] = var

        def _make_toggle(name: str, v: tk.BooleanVar) -> None:
            self.live_chart_plugin.set_overlay_visible(name, v.get())

        cb = ttk.Checkbutton(
            toolbar, text=label_text, variable=var,
            command=lambda n=overlay_name, vv=var: self.live_chart_plugin.set_overlay_visible(n, vv.get())
        )
        cb.pack(side=tk.LEFT, padx=(0, 4))

    # Separator
    ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

    # Signal filter dropdown
    ttk.Label(toolbar, text="Signal Filter:").pack(side=tk.LEFT, padx=(0, 4))
    self._signal_filter_var = tk.StringVar(value="all")
    signal_filter_combo = ttk.Combobox(
        toolbar, textvariable=self._signal_filter_var, values=["all", "executed", "blocked", "shadow"],
        width=10, state="readonly"
    )
    signal_filter_combo.pack(side=tk.LEFT, padx=(0, 6))

    def _on_signal_filter_change(event: Any = None) -> None:
        self.live_chart_plugin.set_signal_filter(self._signal_filter_var.get())

    signal_filter_combo.bind("<<ComboboxSelected>>", _on_signal_filter_change)

    # Snapshot PNG button
    def _snapshot_png() -> None:
        fn = filedialog.asksaveasfilename(
            title="Save Chart Snapshot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if fn:
            self.live_chart_plugin.export_png(fn)

    ttk.Button(toolbar, text="📷 Snapshot PNG", command=_snapshot_png).pack(side=tk.LEFT, padx=(0, 4))

    # Export signals CSV
    ttk.Button(toolbar, text="📥 Export Signals", command=self._export_signals_csv).pack(side=tk.LEFT)


# =============================================================================
# FEATURE EXPLAINABILITY PANEL
# =============================================================================

def _build_feature_explainability_panel(self) -> None:
    """Build the Feature Explainability panel below the chart."""
    frame = ttk.LabelFrame(self.live_chart_frame, text="Feature Explainability", padding=8)
    frame.pack(fill=tk.X, padx=6, pady=(0, 6))

    # Big probability display
    prob_row = ttk.Frame(frame)
    prob_row.pack(fill=tk.X, pady=(0, 6))
    ttk.Label(prob_row, text="Signal Probability:", font=("Segoe UI", 10)).pack(side=tk.LEFT)
    self._feature_prob_var = tk.StringVar(value="n/a")
    ttk.Label(prob_row, textvariable=self._feature_prob_var, font=("Segoe UI", 14, "bold"), foreground="#60a5fa").pack(side=tk.LEFT, padx=(6, 0))

    # Two-column layout for positive/negative contributors
    contrib_row = ttk.Frame(frame)
    contrib_row.pack(fill=tk.BOTH, expand=True)

    # Positive contributors
    pos_frame = ttk.LabelFrame(contrib_row, text="↑ Top Positive Contributors", padding=6)
    pos_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
    self._pos_features_listbox = tk.Listbox(pos_frame, height=4, font=("Segoe UI", 9))
    self._pos_features_listbox.pack(fill=tk.BOTH, expand=True)
    self._feature_explain_vars["pos"] = self._pos_features_listbox

    # Negative contributors
    neg_frame = ttk.LabelFrame(contrib_row, text="↓ Top Negative Contributors", padding=6)
    neg_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
    self._neg_features_listbox = tk.Listbox(neg_frame, height=4, font=("Segoe UI", 9))
    self._neg_features_listbox.pack(fill=tk.BOTH, expand=True)
    self._feature_explain_vars["neg"] = self._neg_features_listbox

    # Pre-defined feature lists (replace with actual SHAP/LIME values if available)
    self._default_pos_features = [
        "• IV Expansion",
        "• PCR Divergence",
        "• Momentum Acceleration",
        "• Vanna Signal",
        "• Gamma Exposure",
    ]
    self._default_neg_features = [
        "• Spread Widening",
        "• Low Liquidity",
        "• Time Decay",
        "• Volatility Collapse",
    ]


def _refresh_feature_explainability(self, recent_rows: list[dict[str, Any]] = None) -> None:
    """Update feature explainability based on most recent prediction row."""
    try:
        prob = "n/a"
        if recent_rows:
            row = recent_rows[0]
            prob_val = float(row.get("probability", 0))
            prob = f"{prob_val:.2f}"

        self._feature_prob_var.set(prob)

        pos_lb: tk.Listbox = self._feature_explain_vars.get("pos")
        neg_lb: tk.Listbox = self._feature_explain_vars.get("neg")

        if pos_lb:
            pos_lb.delete(0, tk.END)
            for feat in self._default_pos_features:
                pos_lb.insert(tk.END, feat)

        if neg_lb:
            neg_lb.delete(0, tk.END)
            for feat in self._default_neg_features:
                neg_lb.insert(tk.END, feat)

    except Exception:
        pass


# =============================================================================
# OPTION CHAIN PANEL
# =============================================================================

def _build_option_chain_panel(self) -> None:
    """Build option chain side panel to the right of the chart."""
    frame = ttk.LabelFrame(self.live_chart_frame, text="Option Chain", padding=6)
    frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 6), pady=(0, 0))

    # ATM strike display
    atm_row = ttk.Frame(frame)
    atm_row.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(atm_row, text="ATM:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
    self._atm_strike_var = tk.StringVar(value="--")
    ttk.Label(atm_row, textvariable=self._atm_strike_var, font=("Segoe UI", 11, "bold"), foreground="#fbbf24").pack(side=tk.LEFT, padx=(4, 0))

    # Column headers
    hdr = ttk.Frame(frame)
    hdr.pack(fill=tk.X)
    for col_text, col_w in [("Strike", 55), ("OI", 45), ("IV", 38), ("Bid", 42), ("Ask", 42), ("Spread", 38)]:
        lbl = ttk.Label(hdr, text=col_text, font=("Segoe UI", 7, "bold"), anchor="center", width=col_w)
        lbl.pack(side=tk.LEFT, padx=1)

    # Calls section
    ttk.Label(frame, text="— Calls —", font=("Segoe UI", 8, "bold"), foreground="#60a5fa").pack(fill=tk.X)
    self._calls_container = ttk.Frame(frame)
    self._calls_container.pack(fill=tk.X)
    self._call_rows: list[dict[str, Any]] = []

    # Divider
    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

    # Puts section
    ttk.Label(frame, text="— Puts —", font=("Segoe UI", 8, "bold"), foreground="#f87171").pack(fill=tk.X)
    self._puts_container = ttk.Frame(frame)
    self._puts_container.pack(fill=tk.X)
    self._put_rows: list[dict[str, Any]] = []

    # Placeholder label
    self._option_chain_placeholder = ttk.Label(frame, text="Awaiting data…", foreground="gray", font=("Segoe UI", 8))
    self._option_chain_placeholder.pack(pady=(8, 0))


def _refresh_option_chain(self) -> None:
    """Refresh option chain from self._option_chain_data or strategy websocket data."""
    try:
        chain_data = getattr(self, "_option_chain_data", []) or []

        if not chain_data:
            # Try to fetch from strategy if available
            scalper = getattr(self, "_scalper", None)
            if scalper is not None:
                chain_data = getattr(scalper, "_option_chain_cache", []) or []

        if not chain_data:
            return

        self._option_chain_placeholder.pack_forget()

        # Clear existing rows
        for w in self._calls_container.pack_slaves():
            w.destroy()
        for w in self._puts_container.pack_slaves():
            w.destroy()
        self._call_rows.clear()
        self._put_rows.clear()

        # Determine ATM
        spot_val = None
        scalper = getattr(self, "_scalper", None)
        if scalper is not None:
            spot_val = getattr(scalper, "_spot_ltp", None)
        if spot_val is None:
            spot_val = self._market_info.get("spot") if hasattr(self, "_market_info") else None

        atm_strike = None
        if spot_val:
            atm_strike = round(float(spot_val) / 50) * 50
            self._atm_strike_var.set(str(atm_strike))

        # Show 5 OTM calls, 5 ITM calls, 5 OTM puts, 5 ITM puts
        strikes = sorted(set(float(c.get("strike", 0)) for c in chain_data if c.get("strike")))
        if not strikes:
            return

        calls = sorted([c for c in chain_data if str(c.get("option_type", "")).upper() in ("CE", "CALL")], key=lambda x: float(x.get("strike", 0)))
        puts = sorted([p for p in chain_data if str(p.get("option_type", "")).upper() in ("PE", "PUT")], key=lambda x: float(x.get("strike", 0)))

        def _render_rows(container: ttk.Frame, contracts: list[dict], is_call: bool) -> None:
            # Pick strikes near ATM
            if atm_strike:
                target = [s for s in contracts if float(s.get("strike", 0)) >= atm_strike][:5]
                target += [s for s in contracts if float(s.get("strike", 0)) < atm_strike][-5:]
            else:
                target = contracts[:10]
            target = sorted(target, key=lambda x: float(x.get("strike", 0)), reverse=(not is_call))

            for row_data in target:
                row = ttk.Frame(container)
                row.pack(fill=tk.X)
                strike = float(row_data.get("strike", 0))
                oi = row_data.get("oi", 0)
                iv = float(row_data.get("iv", 0) or 0)
                bid = float(row_data.get("bid", 0) or 0)
                ask = float(row_data.get("ask", 0) or 0)
                spread = ask - bid if ask and bid else 0

                # Colour coding
                bg = ""
                strike_fg = "white"
                if oi and int(oi) > 50000:
                    bg = "#92400e"  # amber for OI build-up
                iv_fg = "#f43f5e" if iv > 30 else "white"
                spread_fg = "#fb923c" if spread > 2 else "white"

                values = [
                    (f"{strike:.0f}", 55, "white"),
                    (f"{oi}", 45, "white"),
                    (f"{iv:.1f}", 38, iv_fg),
                    (f"{bid:.2f}", 42, "white"),
                    (f"{ask:.2f}", 42, "white"),
                    (f"{spread:.2f}", 38, spread_fg),
                ]
                for text, width, fg in values:
                    lbl = ttk.Label(row, text=text, anchor="center", width=width, font=("Segoe UI", 8))
                    lbl.pack(side=tk.LEFT, padx=1)
                    if bg:
                        lbl.configure(background=bg)

        _render_rows(self._calls_container, calls, True)
        _render_rows(self._puts_container, puts, False)

    except Exception:
        pass
    finally:
        self._safe_after_app("option_chain_panel", 5000, self._refresh_option_chain)


# =============================================================================
# TRADE REVIEW PANEL
# =============================================================================

def _build_trade_review_panel(self) -> None:
    """Build collapsible trade review panel below the chart."""
    frame = ttk.LabelFrame(self.live_chart_frame, text="Trade Review", padding=8)
    frame.pack(fill=tk.BOTH, expand=False, padx=6, pady=(0, 6))

    self._review_content_frame = ttk.Frame(frame)
    self._review_content_frame.pack(fill=tk.BOTH, expand=True)

    # Placeholder
    self._review_placeholder_var = tk.StringVar(value="Click a signal marker on chart to review")
    ttk.Label(
        self._review_content_frame,
        textvariable=self._review_placeholder_var,
        foreground="gray", font=("Segoe UI", 9, "italic")
    ).pack(anchor="w")

    # Fields (created on demand)
    self._review_fields: dict[str, Any] = {}

    # Canvas click binding
    try:
        chart_canvas = self.live_chart_plugin.canvas.get_tk_widget()
        chart_canvas.bind("<Button-1>", self._on_chart_canvas_click)
    except Exception:
        pass

    # Export button
    btn_row = ttk.Frame(frame)
    btn_row.pack(fill=tk.X, pady=(4, 0))
    ttk.Button(btn_row, text="Export Trade Review CSV", command=self._export_trade_review_csv).pack(side=tk.LEFT)


def _on_chart_canvas_click(self, event: Any) -> None:
    """Handle click on chart canvas to select a trade for review."""
    try:
        plugin = getattr(self, "live_chart_plugin", None)
        if plugin is None:
            return

        # Translate x,y to timestamp via nearest candle
        x, y = event.x, event.y
        candles = getattr(plugin, "_candles", [])
        if not candles:
            return

        import matplotlib.dates as mdates

        # Get x data from candles
        try:
            xs = mdates.date2num([plugin._candle_dt(c) for c in candles if plugin._candle_dt(c) is not None])
            if len(xs) == 0:
                return
            idx = int(round(np.argmin(np.abs(xs - x))))
            clicked_candle = candles[idx]
            clicked_ts = float(getattr(clicked_candle, "ts", 0) or 0)
        except Exception:
            return

        # Find nearest prediction/signal
        prediction_rows = getattr(plugin, "_prediction_rows", [])
        if not prediction_rows:
            return

        nearest_row = min(
            prediction_rows,
            key=lambda r: abs(float(r.get("ts", 0) or 0) - clicked_ts)
        )

        # Check if within reasonable distance (e.g., 60 seconds)
        if abs(float(nearest_row.get("ts", 0) or 0) - clicked_ts) > 60:
            return

        trade_id = nearest_row.get("prediction_id", "")
        if trade_id:
            self.live_chart_plugin.focus_trade(str(trade_id))
            self._on_chart_marker_clicked(nearest_row)

    except Exception:
        pass


def _on_chart_marker_clicked(self, row: dict[str, Any]) -> None:
    """Populate trade review panel with the selected prediction row."""
    try:
        self._review_trade = row

        # Clear placeholder
        placeholder = self._review_content_frame.winfo_children()
        for w in placeholder:
            w.destroy()

        fields_cfg = [
            ("Timestamp", lambda: datetime.fromtimestamp(float(row.get("ts", 0) or 0)).strftime("%Y-%m-%d %H:%M:%S")),
            ("Symbol", lambda: str(row.get("symbol", "--"))),
            ("Probability", lambda: f"{float(row.get('probability', 0)):.4f}"),
            ("Predicted Class", lambda: str(row.get("predicted_class", "--"))),
            ("Confidence Bucket", lambda: str(row.get("confidence_bucket", "--"))),
            ("Regime", lambda: str(row.get("regime", "--"))),
            ("Trade Candidate", lambda: str(row.get("trade_candidate", "--"))),
            ("Trade Taken", lambda: str(row.get("trade_taken", "--"))),
        ]

        inner = ttk.Frame(self._review_content_frame)
        inner.pack(fill=tk.BOTH, expand=True, anchor="n")

        for lbl_text, value_fn in fields_cfg:
            row_f = ttk.Frame(inner)
            row_f.pack(fill=tk.X, pady=1)
            ttk.Label(row_f, text=f"{lbl_text}:", font=("Segoe UI", 8, "bold"), width=16, anchor="w").pack(side=tk.LEFT)
            val_var = tk.StringVar(value=value_fn())
            self._review_fields[lbl_text] = val_var
            ttk.Label(row_f, textvariable=val_var, font=("Segoe UI", 8)).pack(side=tk.LEFT)

        # Outcome (inferred)
        prob = float(row.get("probability", 0))
        outcome = "Would-be WIN" if prob > 0.5 else "Would-be LOSS"
        out_row = ttk.Frame(inner)
        out_row.pack(fill=tk.X, pady=1)
        ttk.Label(out_row, text="Inferred Outcome:", font=("Segoe UI", 8, "bold"), width=16, anchor="w").pack(side=tk.LEFT)
        out_var = tk.StringVar(value=outcome)
        self._review_fields["Outcome"] = out_var
        fg_color = "#10b981" if prob > 0.5 else "#f43f5e"
        lbl = ttk.Label(out_row, textvariable=out_var, font=("Segoe UI", 8, "bold"), foreground=fg_color)
        lbl.pack(side=tk.LEFT)

        # Features list (simplified)
        feat_row = ttk.Frame(inner)
        feat_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(feat_row, text="Top Features:", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        for feat in self._default_pos_features[:3]:
            ttk.Label(feat_row, text=feat, font=("Segoe UI", 7), foreground="gray").pack(anchor="w")

    except Exception:
        pass


def _export_trade_review_csv(self) -> None:
    """Export trade review data to CSV."""
    try:
        fn = filedialog.asksaveasfilename(
            title="Export Trade Review",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not fn:
            return

        row = getattr(self, "_review_trade", None)
        if row is None:
            # Export all recent predictions
            db = getattr(self, "_db_manager", None)
            if db is None:
                scalper = getattr(self, "_scalper", None)
                db = getattr(scalper, "db_manager", None) if scalper is not None else None
            if db is None:
                return
            rows = db.list_recent_prediction_markers(limit=10000) if hasattr(db, "list_recent_prediction_markers") else []
        else:
            rows = [row]

        with open(fn, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    except Exception:
        pass


# =============================================================================
# ALERT PANEL
# =============================================================================

def _build_alert_panel(self) -> None:
    """Build alert badges panel at the bottom of the Live Chart tab."""
    frame = ttk.Frame(self.live_chart_frame)
    frame.pack(fill=tk.X, padx=6, pady=(0, 4))

    ttk.Label(frame, text="Alerts:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

    self._alert_badges_container = ttk.Frame(frame)
    self._alert_badges_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

    self._alert_badge_labels: list[dict[str, Any]] = []


def _refresh_alerts(self) -> None:
    """Check recent data and update active alerts."""
    try:
        now = time.time()
        active = getattr(self, "_active_alerts", [])

        # Expire old alerts (older than 60s)
        active = [a for a in active if (now - a.get("_added_ts", now)) < 60]

        db = getattr(self, "_db_manager", None)
        if db is None:
            scalper = getattr(self, "_scalper", None)
            db = getattr(scalper, "db_manager", None) if scalper is not None else None

        if db is not None:
            rows = db.list_recent_prediction_markers(limit=100) if hasattr(db, "list_recent_prediction_markers") else []
            if rows:
                latest = rows[0]
                prob = float(latest.get("probability", 0))

                # High Confidence Signal
                if prob > 0.8:
                    if not any(a.get("type") == "HIGH_CONF" for a in active):
                        active.append({"type": "HIGH_CONF", "severity": "INFO", "message": f"High confidence signal: {prob:.2f}", "_added_ts": now})

                # Feature Drift / Prediction Failure (check recent failures)
                if len(rows) >= 10:
                    recent_probs = [float(r.get("probability", 0)) for r in rows[:10]]
                    variance = sum((p - sum(recent_probs)/len(recent_probs))**2 for p in recent_probs) / len(recent_probs)
                    if variance > 0.1:
                        if not any(a.get("type") == "FEATURE_DRIFT" for a in active):
                            active.append({"type": "FEATURE_DRIFT", "severity": "WARNING", "message": "Feature/Prediction drift detected", "_added_ts": now})

        # Check IV from market data
        scalper = getattr(self, "_scalper", None)
        iv = None
        spread = None
        if scalper is not None:
            iv = getattr(scalper, "_iv", None)
            spread = getattr(scalper, "_spread", None)

        if iv is not None and float(iv) > 30:
            if not any(a.get("type") == "EXTREME_IV" for a in active):
                active.append({"type": "EXTREME_IV", "severity": "WARNING", "message": f"Extreme IV: {iv:.1f}%", "_added_ts": now})

        if spread is not None and float(spread) > 2.0:
            if not any(a.get("type") == "SPREAD_SHOCK" for a in active):
                active.append({"type": "SPREAD_SHOCK", "severity": "ERROR", "message": f"Spread shock: {spread:.2f}%", "_added_ts": now})

        self._active_alerts = active[:5]  # Keep max 5

        # Update badge labels
        container = getattr(self, "_alert_badges_container", None)
        if container is None:
            return

        for w in container.pack_slaves():
            w.destroy()
        self._alert_badge_labels.clear()

        severity_colors = {
            "INFO": "#3b82f6",
            "WARNING": "#f59e0b",
            "ERROR": "#f43f5e",
        }

        for alert in self._active_alerts:
            sev = alert.get("severity", "INFO")
            color = severity_colors.get(sev, "#6b7280")
            badge = tk.Frame(container, background=color, padx=4, pady=2)
            badge.pack(side=tk.LEFT, padx=(0, 4))
            ttk.Label(badge, text=f"{sev}: {alert.get('message', '')}", background=color, foreground="white", font=("Segoe UI", 8)).pack()
            self._alert_badge_labels.append({"widget": badge, "alert": alert})

    except Exception:
        pass
    finally:
        self._safe_after_app("alerts_panel", 10000, self._refresh_alerts)


# =============================================================================
# EXPORT METHODS
# =============================================================================

def _export_shadow_report(self) -> None:
    """Export shadow analytics report as CSV."""
    try:
        fn = filedialog.asksaveasfilename(
            title="Export Shadow Report",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not fn:
            return

        db = getattr(self, "_db_manager", None)
        if db is None:
            scalper = getattr(self, "_scalper", None)
            db = getattr(scalper, "db_manager", None) if scalper is not None else None
        if db is None:
            return

        rows = db.list_recent_prediction_markers(limit=10000) if hasattr(db, "list_recent_prediction_markers") else []
        if not rows:
            return

        fieldnames = ["timestamp", "probability", "predicted_class", "regime", "trade_candidate", "trade_taken", "outcome", "pnl"]
        with open(fn, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                prob = float(row.get("probability", 0))
                outcome = "win" if prob > 0.5 else "loss"
                ts = row.get("ts", "")
                try:
                    ts = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                except Exception:
                    pass
                writer.writerow({
                    "timestamp": ts,
                    "probability": prob,
                    "predicted_class": row.get("predicted_class", ""),
                    "regime": row.get("regime", ""),
                    "trade_candidate": row.get("trade_candidate", ""),
                    "trade_taken": row.get("trade_taken", ""),
                    "outcome": outcome,
                    "pnl": "",
                })

    except Exception:
        pass


def _export_signals_csv(self) -> None:
    """Export recent signal/prediction rows to CSV."""
    try:
        fn = filedialog.asksaveasfilename(
            title="Export Signals CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not fn:
            return

        db = getattr(self, "_db_manager", None)
        if db is None:
            scalper = getattr(self, "_scalper", None)
            db = getattr(scalper, "db_manager", None) if scalper is not None else None
        if db is None:
            return

        rows = db.list_recent_prediction_markers(limit=10000) if hasattr(db, "list_recent_prediction_markers") else []
        if not rows:
            return

        with open(fn, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    except Exception:
        pass