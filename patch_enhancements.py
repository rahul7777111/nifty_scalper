#!/usr/bin/env python3
"""Apply all remaining Phase 2 config + Phase 3-5 new features."""

import re

# ============================================================
# 1. FIX config.py - Add new feature params
# ============================================================

with open('src/config.py', 'r', encoding='utf-8') as f:
    cfg = f.read()

# Insert new feature params after the max_portfolio_delta_abs line
old_cfg_end_block = """    max_portfolio_delta_abs: float = 500.0


def _default_credential_path() -> Path:"""

new_cfg_end_block = """    max_portfolio_delta_abs: float = 500.0

    # ---- IV Percentile Position Sizing ----
    # When enabled, position sizes are scaled inversely to IV percentile.
    # High IV => smaller size, Low IV => full size.
    iv_sizing_enabled: bool = True
    # Number of past days to use for IV percentile calculation.
    iv_percentile_period: int = 252
    # Below this IV percentile, use full configured size.
    iv_sizing_min_pct: float = 50.0
    # Above this IV percentile, scale down to iv_sizing_factor_at_max.
    iv_sizing_max_pct: float = 80.0
    # At max IV percentile, multiply size by this factor.
    iv_sizing_factor_at_max: float = 0.5
    # IV percentile at which scaling starts (1.0 = full size at min_pct).
    iv_sizing_factor_min: float = 1.0

    # ---- Session-Specific Exit Adjustments ----
    # When enabled, exit parameters (SL, trail, target) vary by market session.
    session_exit_enabled: bool = True
    # Opening session (first 45 min): tighter stops to avoid whipsaw
    session_exit_sl_atr_mult_open: float = 1.0
    session_exit_trail_atr_mult_open: float = 0.5
    session_exit_tp_atr_mult_open: float = 2.0
    # Mid session (core hours): standard parameters
    session_exit_sl_atr_mult_mid: float = 1.5
    session_exit_trail_atr_mult_mid: float = 0.8
    session_exit_tp_atr_mult_mid: float = 2.5
    # Closing session (last 90 min): moderate to avoid late-day reversals
    session_exit_sl_atr_mult_close: float = 1.2
    session_exit_trail_atr_mult_close: float = 0.6
    session_exit_tp_atr_mult_close: float = 2.0

    # ---- Strategy Win-Rate Tracker ----
    # When enabled, tracks win/loss per strategy and auto-disables losers.
    winrate_tracker_enabled: bool = True
    # Max consecutive losses before auto-disabling a strategy.
    winrate_tracker_max_losses: int = 3
    # After disabling, re-enable after this many consecutive wins on other strategies.
    winrate_tracker_recovery_wins: int = 1
    # Rolling lookback for win-rate display (not for disable logic).
    winrate_tracker_lookback: int = 20


def _default_credential_path() -> Path:"""

if old_cfg_end_block in cfg:
    cfg = cfg.replace(old_cfg_end_block, new_cfg_end_block)
    print("[config.py] Added new feature params (IV sizing, session exit, winrate tracker)")
else:
    print("[config.py] ERROR: Could not find insertion point")
    print("  Old block first 120 chars:", repr(old_cfg_end_block[:120]))

with open('src/config.py', 'w', encoding='utf-8') as f:
    f.write(cfg)

# ============================================================
# 2. PATCH strategy.py - Add IV percentile sizing hook
#    Hook into existing position sizing at the risk scaling section (~line 3740)
# ============================================================

with open('src/strategy.py', 'r', encoding='utf-8') as f:
    strat = f.read()

# ---- Feature 1: IV Percentile Position Sizing ----
# Find the risk scaling section and add IV percentile adjustment
old_risk_scale_section = """        qty = base_qty
        if getattr(self.cfg, "risk_scale_enabled", False):
            factor = 1.0
            stopout_count = int(getattr(self.state, "consecutive_stopouts", 0) or 0)
            recovery_wins = int(getattr(self.cfg, "risk_scale_recovery_wins", 2) or 2)"""

new_risk_scale_section = """        qty = base_qty
        
        # ---- IV Percentile Position Sizing ----
        # Scale position size inversely to IV percentile.
        # When IV is high (expensive options), trade smaller.
        # When IV is low (cheap options), trade full size.
        iv_factor = 1.0
        if bool(getattr(self.cfg, "iv_sizing_enabled", False)):
            try:
                iv = float(self._iv_snapshot.get("iv", 0.0) or 0.0)
                if iv > 0:
                    iv_hist = list(getattr(self, "_iv_history", []) or [])
                    if iv_hist:
                        iv_sorted = sorted(float(x) for x in iv_hist if float(x) > 0)
                        if iv_sorted:
                            rank = sum(1 for x in iv_sorted if x <= iv)
                            pct = float(rank) / float(len(iv_sorted)) * 100.0
                            min_pct = float(getattr(self.cfg, "iv_sizing_min_pct", 50.0))
                            max_pct = float(getattr(self.cfg, "iv_sizing_max_pct", 80.0))
                            factor_min = float(getattr(self.cfg, "iv_sizing_factor_min", 1.0))
                            factor_max = float(getattr(self.cfg, "iv_sizing_factor_at_max", 0.5))
                            if pct <= min_pct:
                                iv_factor = factor_min
                            elif pct >= max_pct:
                                iv_factor = factor_max
                            else:
                                t = (pct - min_pct) / (max_pct - min_pct)
                                iv_factor = factor_min + t * (factor_max - factor_min)
            except Exception:
                pass
        
        if getattr(self.cfg, "risk_scale_enabled", False):
            factor = 1.0 * iv_factor
            stopout_count = int(getattr(self.state, "consecutive_stopouts", 0) or 0)
            recovery_wins = int(getattr(self.cfg, "risk_scale_recovery_wins", 2) or 2)"""

if old_risk_scale_section in strat:
    strat = strat.replace(old_risk_scale_section, new_risk_scale_section)
    print("[strategy.py] Added IV percentile position sizing")
else:
    print("[strategy.py] ERROR: Could not find risk scaling section for IV sizing")
    # Try with slight variations
    idx = strat.find('if getattr(self.cfg, "risk_scale_enabled", False):')
    if idx >= 0:
        print(f"  risk_scale_enabled found at index {idx}")
        print(f"  Context: {repr(strat[idx-50:idx+100])}")
    else:
        print("  risk_scale_enabled NOT FOUND in file")

# ============================================================
# 3. PATCH strategy.py - Add session-specific exit adjustments
#    Hook into _manage_open_trades where SL/trail/target params are read
# ============================================================

# Find the section where exit params are read (dir_sl_atr_mult, dir_trail_atr_mult, dir_tp_atr_mult)
# This appears at the point where bear/bull exit params are read for directional trades
old_exit_params_bear = """                            # Explicit ATR Stop Loss
                            sl_mult = float(getattr(self.cfg, "dir_sl_atr_mult", 1.5))
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)"""

new_exit_params_bear = """                            # Session-specific exit adjustments
                            _sl_m, _trail_m, _tp_m = self._session_exit_adjust()
                            # Explicit ATR Stop Loss
                            sl_mult = float(getattr(self.cfg, "dir_sl_atr_mult", 1.5))
                            if _sl_m is not None:
                                sl_mult = float(_sl_m)
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)"""

if old_exit_params_bear in strat:
    strat = strat.replace(old_exit_params_bear, new_exit_params_bear)
    print("[strategy.py] Added session exit adjustments (bear)")
else:
    print("[strategy.py] ERROR: bear exit params section not found")

# Also apply to the bear side (second occurrence for puts)
# Check if there are two occurrences
bear_count = strat.count(old_exit_params_bear)
if bear_count > 0:
    print(f"[strategy.py] Session exit bear: {bear_count} occurrences patched")
else:
    # Try alternate format
    alt_bear = """                            loss_mult = float(be_mult_local) if float(be_mult_local) < 0 else -abs_trail"""
    idx = strat.find(alt_bear)
    if idx >= 0:
        print(f"  Found alternate bear format at index {idx}")

# Now add the _session_exit_adjust helper method
# Find a good insertion point - after _try_get_ltp_for_leg or similar
old_method_marker = "    def _risk_scale_pos_size(self"
new_method_prefix = '''    def _session_exit_adjust(self) -> tuple:
        """Return (sl_mult, trail_mult, tp_mult) adjusted for current market session, or (None, None, None)."""
        if not bool(getattr(self.cfg, "session_exit_enabled", False)):
            return (None, None, None)
        try:
            now = datetime.now()
            # IST offset (5h30m)
            try:
                import pytz
                ist = pytz.timezone("Asia/Kolkata")
                now_ist = datetime.now(ist)
            except Exception:
                now_ist = now
            h = now_ist.hour
            m = now_ist.minute
            mins = h * 60 + m
            
            open_start_str = str(getattr(self.cfg, "session_open_start_hhmm", "09:15") or "09:15")
            open_end_str = str(getattr(self.cfg, "session_open_end_hhmm", "10:00") or "10:00")
            close_start_str = str(getattr(self.cfg, "session_close_start_hhmm", "14:30") or "14:30")
            close_end_str = str(getattr(self.cfg, "session_close_end_hhmm", "15:30") or "15:30")
            
            def _to_mins(s: str) -> int:
                parts = s.strip().split(":")
                return int(parts[0]) * 60 + int(parts[1])
            
            open_start = _to_mins(open_start_str)
            open_end = _to_mins(open_end_str)
            close_start = _to_mins(close_start_str)
            close_end = _to_mins(close_end_str)
            
            if open_start <= mins <= open_end:
                sl = float(getattr(self.cfg, "session_exit_sl_atr_mult_open", 1.0))
                trail = float(getattr(self.cfg, "session_exit_trail_atr_mult_open", 0.5))
                tp = float(getattr(self.cfg, "session_exit_tp_atr_mult_open", 2.0))
                return (sl, trail, tp)
            elif close_start <= mins <= close_end:
                sl = float(getattr(self.cfg, "session_exit_sl_atr_mult_close", 1.2))
                trail = float(getattr(self.cfg, "session_exit_trail_atr_mult_close", 0.6))
                tp = float(getattr(self.cfg, "session_exit_tp_atr_mult_close", 2.0))
                return (sl, trail, tp)
            else:
                sl = float(getattr(self.cfg, "session_exit_sl_atr_mult_mid", 1.5))
                trail = float(getattr(self.cfg, "session_exit_trail_atr_mult_mid", 0.8))
                tp = float(getattr(self.cfg, "session_exit_tp_atr_mult_mid", 2.5))
                return (sl, trail, tp)
        except Exception:
            return (None, None, None)

    def _risk_scale_pos_size(self'''

if old_method_marker in strat:
    strat = strat.replace(old_method_marker, new_method_prefix)
    print("[strategy.py] Added _session_exit_adjust helper method")
else:
    print("[strategy.py] ERROR: Could not find _risk_scale_pos_size insertion point")
    # Search for alternative
    idx = strat.find("def _risk_scale")
    if idx >= 0:
        print(f"  Found at index {idx}: {repr(strat[idx:idx+50])}")

# ============================================================
# 4. PATCH strategy.py - Add win-rate tracker
#    Hook into trade closing/result recording
# ============================================================

# Add win-rate state tracking to the class __init__
# Find the __init__ method and add tracking dicts
old_init_marker = """        # Throttle entry evaluation to candle boundaries"""
new_init_marker = """        # Win-rate tracker state
        self._strategy_winrates: Dict[str, Dict[str, object]] = {}
        self._disabled_strategies: Dict[str, float] = {}
        
        # Throttle entry evaluation to candle boundaries"""

if old_init_marker in strat:
    strat = strat.replace(old_init_marker, new_init_marker)
    print("[strategy.py] Added win-rate tracker state to __init__")
else:
    print("[strategy.py] ERROR: Could not find __init__ insertion point for winrate tracker")

# Add _record_trade_outcome method - insert before _manage_open_trades
old_manage_marker = "    def _manage_open_trades(self) -> None:"
new_method_block = '''    def _record_trade_outcome(self, trade_id: str, realized_pnl: float, strategy_name: str) -> None:
        """Record a trade outcome and auto-disable losing strategies."""
        if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):
            return
        try:
            key = str(strategy_name or "unknown").strip().lower()
            if not key:
                return
            win = float(realized_pnl or 0.0) > 0.0
            
            # Get or create tracker for this strategy
            tracker = dict(self._strategy_winrates.get(key, {}) or {})
            streak = int(tracker.get("streak", 0) or 0)
            wins = int(tracker.get("wins", 0) or 0)
            losses = int(tracker.get("losses", 0) or 0)
            total = int(tracker.get("total", 0) or 0)
            
            if win:
                streak = streak + 1 if streak > 0 else 1
                wins += 1
            else:
                streak = streak - 1 if streak < 0 else -1
                losses += 1
            total += 1
            
            tracker["streak"] = streak
            tracker["wins"] = wins
            tracker["losses"] = losses
            tracker["total"] = total
            tracker["last_ts"] = float(time.time())
            self._strategy_winrates[key] = tracker
            
            # Auto-disable on consecutive losses
            max_losses = int(getattr(self.cfg, "winrate_tracker_max_losses", 3) or 3)
            if not win and abs(streak) >= max_losses:
                self._disabled_strategies[key] = float(time.time())
                print(f"[WINRATE] Disabled {key}: {abs(streak)} consecutive losses")
            
            # Re-enable on recovery wins
            recovery_wins = int(getattr(self.cfg, "winrate_tracker_recovery_wins", 1) or 1)
            if win and streak >= recovery_wins:
                disabled = dict(self._disabled_strategies)
                if key in disabled:
                    del self._disabled_strategies[key]
                    print(f"[WINRATE] Re-enabled {key}: {streak} consecutive wins")
        except Exception:
            pass
    
    def _is_strategy_disabled(self, strategy_name: str) -> bool:
        """Check if a strategy is currently disabled by the win-rate tracker."""
        if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):
            return False
        try:
            key = str(strategy_name or "unknown").strip().lower()
            if not key:
                return False
            disabled = dict(getattr(self, "_disabled_strategies", {}) or {})
            if key in disabled:
                elapsed = float(time.time()) - float(disabled.get(key, 0.0) or 0.0)
                # Auto-re-enable after 1 hour as a safety measure
                if elapsed > 3600.0:
                    del self._disabled_strategies[key]
                    return False
                return True
            return False
        except Exception:
            return False

    def _manage_open_trades(self) -> None:'''

if old_manage_marker in strat:
    strat = strat.replace(old_manage_marker, new_method_block)
    print("[strategy.py] Added _record_trade_outcome and _is_strategy_disabled methods")
else:
    print("[strategy.py] ERROR: Could not find _manage_open_trades for winrate methods")

# ============================================================
# 5. Add win-rate check call to entry logic
#    Hook into the main entry decision where strategies are evaluated
# ============================================================

# Find where _note_strategy_decision is called and add disabled check
old_strat_decision = """        self._note_strategy_decision(name)"""
new_strat_decision = """        self._note_strategy_decision(name)
        if self._is_strategy_disabled(name):
            print(f"[ENTRY] Strategy {name} disabled by win-rate tracker, skipping")
            return"""

# Check if this pattern exists and apply
if old_strat_decision in strat:
    count = strat.count(old_strat_decision)
    if count <= 3:
        strat = strat.replace(old_strat_decision, new_strat_decision)
        print(f"[strategy.py] Added win-rate check to {count} strategy decision sites")
    else:
        # Too many matches, be more specific
        print(f"[strategy.py] {count} matches for _note_strategy_decision, too many for blind replace")
else:
    print("[strategy.py] _note_strategy_decision call not found with exact match")

# ============================================================
# 6. Inject win-rate recording into trade closure
#    Find where realized PnL is recorded on trade close
# ============================================================

# Find trade close PnL recording
old_trade_close = """            self.state.realized_pnl = float(self.state.realized_pnl or 0.0) + float(total_realized or 0.0)"""
new_trade_close = """            self.state.realized_pnl = float(self.state.realized_pnl or 0.0) + float(total_realized or 0.0)
            try:
                self._record_trade_outcome(
                    trade_id=str(tr.get("id") or ""),
                    realized_pnl=float(total_realized or 0.0),
                    strategy_name=str(tr.get("name") or ""),
                )
            except Exception:
                pass"""

if old_trade_close in strat:
    strat = strat.replace(old_trade_close, new_trade_close)
    print("[strategy.py] Added win-rate recording to trade closure")
else:
    print("[strategy.py] ERROR: Could not find trade close PnL recording")

# ============================================================
# 7. Add IV history tracking to _refresh chain or market data poll
# ============================================================

# Find where IV snapshot is updated and add history tracking
old_iv_snapshot = """            self._iv_snapshot = {}"""
new_iv_snapshot = """            self._iv_snapshot = {}
        # Track IV history for percentile sizing
        try:
            iv_val = float(self._iv_snapshot.get("iv", 0.0) or 0.0)
            if iv_val > 0:
                iv_hist = list(getattr(self, "_iv_history", []) or [])
                iv_hist.append(float(iv_val))
                # Keep max 500 entries
                if len(iv_hist) > 500:
                    iv_hist = iv_hist[-500:]
                self._iv_history = iv_hist
        except Exception:
            pass"""

if old_iv_snapshot in strat:
    strat = strat.replace(old_iv_snapshot, new_iv_snapshot)
    print("[strategy.py] Added IV history tracking for percentile sizing")
else:
    print("[strategy.py] ERROR: Could not find IV snapshot for history tracking")

# Write the final strategy.py
with open('src/strategy.py', 'w', encoding='utf-8') as f:
    f.write(strat)

print("\n=== ALL PATCHES APPLIED ===")
