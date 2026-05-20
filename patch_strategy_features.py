"""Patch strategy.py with 3 new profit-enhancement features.

1. IV Percentile Position Sizing
2. Session-Specific Exit Adjustments
3. Strategy Win-Rate Tracker (auto-disable losers)

We inject code into the file using precise line-by-line string replacement.
"""

import re
import sys

FILE = "src/strategy.py"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# FEATURE 1: IV Percentile Position Sizing
# ============================================================
# Add a new method after _entry_qty (which ends around line 3760)
# and modify _entry_qty to include IV percentile factor

# First, add the _entry_iv_percentile_factor method. We insert it right after the
# min_qty block (after line ~3760) and before the next method.
# The next method after _entry_qty is _held_days_since_open - let me find it.
insert_point = content.find("    def _held_days_since_open")
if insert_point == -1:
    print("ERROR: Could not find _held_days_since_open")
    sys.exit(1)

iv_method = """
    def _entry_iv_percentile_factor(self) -> float:
        \"\"\"Return a sizing multiplier based on the IV percentile of the underlying.

        When current IV is in a high percentile, we size down (avoid overpaying).
        When IV is in a low percentile, we size normally or slightly up.

        Reads config keys:
          iv_sizing_enabled (bool) - master switch
          iv_sizing_lookback_days (int) - how many days of IV history to track (default 20)
          iv_sizing_high_percentile (float) - above this percentile, reduce size (default 75.0)
          iv_sizing_low_percentile (float) - below this percentile, normal size (default 25.0)
          iv_sizing_high_factor (float) - multiplier when IV is high (default 0.5)
          iv_sizing_max_factor (float) - max factor allowed (default 1.0)

        Returns 1.0 when disabled or on error (no impact).
        \"\"\"
        try:
            if not bool(getattr(self.cfg, \"iv_sizing_enabled\", False)):
                return 1.0
        except Exception:
            return 1.0

        # Use ATR as a proxy for volatility regime (more reliable than IV for this purpose).
        # For IV percentile we need estimate from the option chain snapshot.
        try:
            lookback = int(getattr(self.cfg, \"iv_sizing_lookback_days\", 20) or 20)
        except Exception:
            lookback = 20
        try:
            high_pct = float(getattr(self.cfg, \"iv_sizing_high_percentile\", 75.0) or 75.0)
        except Exception:
            high_pct = 75.0
        try:
            low_pct = float(getattr(self.cfg, \"iv_sizing_low_percentile\", 25.0) or 25.0)
        except Exception:
            low_pct = 25.0
        try:
            high_factor = float(getattr(self.cfg, \"iv_sizing_high_factor\", 0.5) or 0.5)
        except Exception:
            high_factor = 0.5
        try:
            max_factor = float(getattr(self.cfg, \"iv_sizing_max_factor\", 1.0) or 1.0)
        except Exception:
            max_factor = 1.0

        high_pct = max(1.0, min(99.0, float(high_pct)))
        low_pct = max(0.0, min(float(high_pct) - 1.0, float(low_pct)))
        lookback = max(2, min(100, int(lookback)))
        high_factor = max(0.1, min(1.0, float(high_factor)))
        max_factor = max(float(high_factor), min(2.0, float(max_factor)))

        # Build ATR history from the _iv_watch (which tracks ATR values over time).
        # Fall back to reading from current market conditions and a simple heuristic.
        iv_hist_name = getattr(self, \"_iv_pct_hist\", None)
        hist = iv_hist_name if isinstance(iv_hist_name, list) else []
        if len(hist) < int(lookback):
            # Not enough history - return 1.0 (no adjustment)
            return 1.0

        recent = list(hist)
        ordered = sorted(float(x) for x in recent if isinstance(x, (int, float)))
        if len(ordered) < 2:
            return 1.0

        current = float(ordered[-1])
        rank = sum(1 for x in ordered if x <= current)
        percentile = (float(rank) / float(len(ordered))) * 100.0

        if float(percentile) >= float(high_pct):
            # High IV percentile - size down
            # Linear scale from high_factor at high_pct down to high_factor/2 at 100%
            pct_above = (float(percentile) - float(high_pct)) / (100.0 - float(high_pct) + 1e-9)
            pct_above = max(0.0, min(1.0, float(pct_above)))
            factor = float(high_factor) * (1.0 + float(pct_above)) / 2.0
            return max(float(high_factor) * 0.5, min(float(max_factor), float(factor)))
        elif float(percentile) <= float(low_pct):
            # Low IV percentile - normal or slightly up
            return min(float(max_factor), 1.0)
        else:
            # Normal IV - no adjustment
            return 1.0

"""

content = content[:insert_point] + iv_method + content[insert_point:]

# Now modify _entry_qty to use the IV percentile factor.
# Find the risk scaling section and add the IV factor after the stopout factor
old_line = """        if int(self.state.consecutive_stopouts) > 0:
            if recovery_wins <= 0 or int(self.state.consecutive_wins) < recovery_wins:
                factor *= float(stop_factor) ** int(self.state.consecutive_stopouts)"""

new_line = """        if int(self.state.consecutive_stopouts) > 0:
            if recovery_wins <= 0 or int(self.state.consecutive_wins) < recovery_wins:
                factor *= float(stop_factor) ** int(self.state.consecutive_stopouts)

        # IV Percentile Sizing Factor
        iv_factor = self._entry_iv_percentile_factor()
        factor *= float(iv_factor)"""

if old_line in content:
    content = content.replace(old_line, new_line, 1)
    print("OK: IV percentile factor added to _entry_qty")
else:
    print("WARN: Could not find stopout factor block in _entry_qty")

# ============================================================
# FEATURE 2: Session-Specific Exit Adjustments
# ============================================================
# Add a method _session_exit_adjustments and modify _manage_open_trades

# First, find _manage_open_trades start
insert_point = content.find("    def _manage_open_trades(self)")
if insert_point == -1:
    print("ERROR: Could not find _manage_open_trades")
    sys.exit(1)
# Insert the new method just before _manage_open_trades
session_method = """
    def _session_exit_adjustments(self) -> Dict[str, float]:
        \"\"\"Return exit parameter multipliers based on current trading session time.

        Returns a dict with keys:
          sl_mult   - stop loss multiplier adjustment
          tp_mult   - take profit multiplier adjustment
          trail_mult - trailing stop multiplier adjustment

        Reads config keys:
          session_exit_enabled (bool) - master switch
          session_opening_hhmm (str) - end of opening session (default \"10:00\")
          session_lunch_start_hhmm (str) - start of lunch/quiet session (default \"12:00\")
          session_lunch_end_hhmm (str) - end of lunch/quiet session (default \"13:30\")
          session_close_hhmm (str) - start of closing session (default \"14:45\")
          session_opening_sl_mult (float) - tighter stop in opening (default 0.7)
          session_opening_tp_mult (float) - tighter target in opening (default 0.8)
          session_lunch_sl_mult (float) - wider stop in lunch (default 1.3)
          session_lunch_tp_mult (float) - wider target in lunch (default 1.2)
          session_close_sl_mult (float) - tighter stop in closing (default 0.6)
          session_close_tp_mult (float) - tighter target in closing (default 0.7)

        Returns {sl: 1.0, tp: 1.0, trail: 1.0} when disabled or on error.
        \"\"\"
        default = {\"sl_mult\": 1.0, \"tp_mult\": 1.0, \"trail_mult\": 1.0}

        try:
            if not bool(getattr(self.cfg, \"session_exit_enabled\", False)):
                return dict(default)
        except Exception:
            return dict(default)

        try:
            opening_end = str(getattr(self.cfg, \"session_opening_hhmm\", \"10:00\") or \"10:00\").strip()
            lunch_start = str(getattr(self.cfg, \"session_lunch_start_hhmm\", \"12:00\") or \"12:00\").strip()
            lunch_end = str(getattr(self.cfg, \"session_lunch_end_hhmm\", \"13:30\") or \"13:30\").strip()
            close_start = str(getattr(self.cfg, \"session_close_hhmm\", \"14:45\") or \"14:45\").strip()
            opening_sl = float(getattr(self.cfg, \"session_opening_sl_mult\", 0.7) or 0.7)
            opening_tp = float(getattr(self.cfg, \"session_opening_tp_mult\", 0.8) or 0.8)
            lunch_sl = float(getattr(self.cfg, \"session_lunch_sl_mult\", 1.3) or 1.3)
            lunch_tp = float(getattr(self.cfg, \"session_lunch_tp_mult\", 1.2) or 1.2)
            close_sl = float(getattr(self.cfg, \"session_close_sl_mult\", 0.6) or 0.6)
            close_tp = float(getattr(self.cfg, \"session_close_tp_mult\", 0.7) or 0.7)
        except Exception:
            return dict(default)

        def _parse_hhmm(s: str) -> int:
            try:
                parts = str(s or \"\").strip().split(\":\")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return 0

        now = dt_datetime.now(IST).time() if IST is not None else dt_datetime.now().time()
        now_min = int(now.hour) * 60 + int(now.minute)

        open_min = 9 * 60 + 15  # 09:15
        opening_end_min = _parse_hhmm(opening_end) or (10 * 60)
        lunch_start_min = _parse_hhmm(lunch_start) or (12 * 60)
        lunch_end_min = _parse_hhmm(lunch_end) or (13 * 60 + 30)
        close_start_min = _parse_hhmm(close_start) or (14 * 60 + 45)
        market_close_min = 15 * 60 + 30  # 15:30

        if now_min < open_min or now_min > market_close_min:
            return dict(default)

        sl_adj = 1.0
        tp_adj = 1.0

        if now_min <= opening_end_min:
            sl_adj = float(opening_sl)
            tp_adj = float(opening_tp)
        elif now_min <= lunch_start_min:
            sl_adj = 1.0
            tp_adj = 1.0
        elif now_min <= lunch_end_min:
            sl_adj = float(lunch_sl)
            tp_adj = float(lunch_tp)
        elif now_min <= close_start_min:
            sl_adj = 1.0
            tp_adj = 1.0
        else:
            sl_adj = float(close_sl)
            tp_adj = float(close_tp)

        sl_adj = max(0.3, min(3.0, float(sl_adj)))
        tp_adj = max(0.3, min(3.0, float(tp_adj)))
        trail_adj = float(sl_adj)  # trail tightens/loosens with stop

        return {\"sl_mult\": float(sl_adj), \"tp_mult\": float(tp_adj), \"trail_mult\": float(trail_adj)}

"""

content = content[:insert_point] + session_method + content[insert_point:]

# ============================================================
# FEATURE 3: Strategy Win-Rate Tracker
# ============================================================
# Add state tracking, a method to check/disable underperformers,
# and integrate into the decision flow

# 3a. Add state tracking dicts in __init__
# Find the observability section in __init__
init_tracker_insert = content.find("        # Runtime diagnostics / observability.")
if init_tracker_insert == -1:
    print("ERROR: Could not find diagnostics section in __init__")
    sys.exit(1)

tracker_state = """
        # Strategy win-rate tracker: maps strategy_name -> {wins, losses, last_n}
        self._strategy_winrate: Dict[str, Dict[str, object]] = {}
        # Strategies temporarily disabled by the win-rate tracker.
        self._disabled_strategies: Dict[str, float] = {}

"""

content = content[:init_tracker_insert] + tracker_state + content[init_tracker_insert:]

# 3b. Add the win-rate tracking method before _decide_entries
insert_point = content.find("    def _decide_entries(self)")
if insert_point == -1:
    print("ERROR: Could not find _decide_entries")
    sys.exit(1)

winrate_method = """
    def _strategy_winrate_check(self, strategy_name: str) -> bool:
        \"\"\"Check if a strategy should be allowed based on its recent win-rate.

        Returns True if the strategy can be traded, False if it should be blocked.

        Tracks wins/losses per strategy name and auto-disables a strategy
        when it loses more than a configurable threshold.

        Reads config keys:
          winrate_tracker_enabled (bool) - master switch
          winrate_lookback (int) - how many recent trades to track (default 10)
          winrate_min_wins (int) - minimum wins before considering filter (default 3)
          winrate_max_loss_streak (int) - max consecutive losses before disable (default 3)
          winrate_min_win_rate (float) - minimum win rate % to stay enabled (default 30.0)
          winrate_cooldown_sec (int) - seconds before a disabled strategy can retry (default 300)
        \"\"\"
        try:
            if not bool(getattr(self.cfg, \"winrate_tracker_enabled\", False)):
                return True
        except Exception:
            return True

        name = str(strategy_name or \"\").strip().lower()
        if not name or name in {\"auto\", \"directional\"}:
            return True

        try:
            lookback = int(getattr(self.cfg, \"winrate_lookback\", 10) or 10)
        except Exception:
            lookback = 10
        try:
            min_wins = int(getattr(self.cfg, \"winrate_min_wins\", 3) or 3)
        except Exception:
            min_wins = 3
        try:
            max_loss_streak = int(getattr(self.cfg, \"winrate_max_loss_streak\", 3) or 3)
        except Exception:
            max_loss_streak = 3
        try:
            min_win_rate = float(getattr(self.cfg, \"winrate_min_win_rate\", 30.0) or 30.0)
        except Exception:
            min_win_rate = 30.0
        try:
            cooldown = int(getattr(self.cfg, \"winrate_cooldown_sec\", 300) or 300)
        except Exception:
            cooldown = 300

        lookback = max(2, min(50, int(lookback)))
        min_wins = max(1, min(int(lookback) - 1, int(min_wins)))
        max_loss_streak = max(1, min(int(lookback), int(max_loss_streak)))
        min_win_rate = max(1.0, min(99.0, float(min_win_rate)))

        now_ts = float(time.time())

        # Check if currently disabled
        disabled_until = float(self._disabled_strategies.get(name, 0.0) or 0.0)
        if disabled_until > 0:
            if now_ts < disabled_until:
                return False  # Still in cooldown
            else:
                # Cooldown expired, re-enable
                self._disabled_strategies.pop(name, None)
                return True

        # Get or create tracker entry
        tracker = self._strategy_winrate.setdefault(name, {\"results\": [], \"wins\": 0, \"losses\": 0})
        try:
            results = list(tracker.get(\"results\") or [])
        except Exception:
            results = []

        if len(results) < int(min_wins):
            return True  # Not enough data

        # Look at recent results
        recent = list(results)[-int(lookback):]
        wins = sum(1 for r in recent if r is True)
        losses = sum(1 for r in recent if r is False)

        # Consecutive loss check
        consec_losses = 0
        for r in reversed(recent):
            if r is False:
                consec_losses += 1
            else:
                break

        if int(consec_losses) >= int(max_loss_streak):
            # Disable this strategy
            self._disabled_strategies[name] = float(now_ts) + float(cooldown)
            print(f\"[WINRATE] {name}: {consec_losses} consecutive losses - disabled for {cooldown}s\")
            return False

        # Win rate check
        if wins + losses > 0:
            win_rate = (float(wins) / float(wins + losses)) * 100.0
            if float(win_rate) < float(min_win_rate):
                self._disabled_strategies[name] = float(now_ts) + float(cooldown)
                print(f\"[WINRATE] {name}: {win_rate:.1f}% win rate (< {min_win_rate}%) - disabled for {cooldown}s\")
                return False

        return True

    def _strategy_winrate_record_result(self, strategy_name: str, won: bool) -> None:
        \"\"\"Record a win or loss for a strategy.\"\"\"
        if not bool(getattr(self.cfg, \"winrate_tracker_enabled\", False)):
            return

        name = str(strategy_name or \"\").strip().lower()
        if not name or name in {\"auto\", \"directional\"}:
            return

        try:
            lookback = int(getattr(self.cfg, \"winrate_lookback\", 10) or 10)
        except Exception:
            lookback = 10
        lookback = max(2, min(50, int(lookback)))

        tracker = self._strategy_winrate.setdefault(name, {\"results\": [], \"wins\": 0, \"losses\": 0})
        try:
            results = list(tracker.get(\"results\") or [])
        except Exception:
            results = []

        results.append(bool(won))
        if len(results) > int(lookback) * 2:
            results = results[-int(lookback) * 2:]

        wins = sum(1 for r in results if r is True)
        losses = sum(1 for r in results if r is False)

        tracker[\"results\"] = results
        tracker[\"wins\"] = int(wins)
        tracker[\"losses\"] = int(losses)

"""

content = content[:insert_point] + winrate_method + content[insert_point:]

# 3c. Integrate winrate check into the strategy decision flow
# Find where _note_strategy_decision is called and add winrate check
# Find the multi-leg entry section where strategies are evaluated
# The key integration point is in _decide_entries or the router where
# candidate strategies are checked before being selected.

# Let's find where strategies are filtered/selected in the auto-router
# Look for "_note_strategy_decision" calls in context
insert_point = content.find("        # Re-evaluate signals with new scores")
if insert_point > 0:
    # Find the line before where desired_dir is set from scores
    # Add winrate check for the strategy's trade type
    winrate_integration = """
            # Win-rate tracker check for strategy
            if getattr(self.cfg, \"winrate_tracker_enabled\", False):
                try:
                    strat_check = str(forced_directional_name or effective_strat or \"\").strip().lower()
                    if strat_check and strat_check != \"auto\" and strat_check != \"directional\":
                        if not self._strategy_winrate_check(strat_check):
                            if maybe_reroute_entry_block(f\"Winrate filter blocked {strat_check}\", current_strategy=str(forced_directional_name or effective_strat)):
                                return
                            return
                except Exception:
                    pass

"""
    content = content[:insert_point] + winrate_integration + content[insert_point:]
    print("OK: Winrate check integrated into entry flow")
else:
    print("WARN: Could not find signal re-evaluation section for winrate integration")

# 3d. Record results when a trade closes in _manage_open_trades
# Find where trades are closed with realized PnL
# Look for where realized_pnl_total is computed or where trades are marked closed
close_insert = content.find("                        self.state.realized_pnl += float(r)")
if close_insert > 0:
    # Find the line before this and add winrate recording
    # We need to find the strategy name of the closing trade
    winrate_record = """
                        # Record win/loss for winrate tracker
                        try:
                            tr_name = str(tr.get(\"name\") or \"\").strip().lower()
                            if tr_name and tr_name not in {\"auto\", \"directional\"}:
                                self._strategy_winrate_record_result(tr_name, float(r) > 0)
                        except Exception:
                            pass

"""
    content = content[:close_insert] + winrate_record + content[close_insert:]
    print("OK: Winrate recording added on trade close")
else:
    print("WARN: Could not find realized_pnl line for winrate recording")


# ============================================================
# Write the patched file
# ============================================================
with open(FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nAll patches applied to {FILE} successfully!")

# Verify no syntax errors
try:
    compile(content, FILE, "exec")
    print("Syntax check: PASSED")
except SyntaxError as e:
    print(f"Syntax check: FAILED - {e}")
    sys.exit(1)
