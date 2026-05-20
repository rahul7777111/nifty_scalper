"""Patch src/strategy.py with 3 new profit features using line-number-based insertion.

Features:
1. IV Percentile Position Sizing - size positions inversely to IV
2. Session-Specific Exit Adjustments - tighten/loosen exits by time of day
3. Strategy Win-Rate Tracker - auto-disable underperforming strategies

All insertions are by known line numbers (1-indexed).
"""

FILE = "src/strategy.py"

with open(FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"File loaded: {len(lines)} lines")

# ============================================================
# FEATURE 1: IV Percentile Position Sizing
# ============================================================

# 1a. Insert _entry_iv_percentile_factor method before _held_days_since_open (line 6349, 0-indexed: 6348)
# We insert it at line 6349 (before the method at 6349)
iv_method_lines = [
    "    def _entry_iv_percentile_factor(self) -> float:\n",
    '        """Return a sizing multiplier based on the IV percentile of the underlying.\n',
    "\n",
    "        When current IV is in a high percentile, we size down (avoid overpaying).\n",
    "        When IV is in a low percentile, we size normally or slightly up.\n",
    "\n",
    "        Reads config keys:\n",
    "          iv_sizing_enabled (bool) - master switch\n",
    "          iv_sizing_lookback_days (int) - days of IV history to track (default 20)\n",
    "          iv_sizing_high_percentile (float) - above this, reduce size (default 75.0)\n",
    "          iv_sizing_low_percentile (float) - below this, normal size (default 25.0)\n",
    "          iv_sizing_high_factor (float) - multiplier when IV is high (default 0.5)\n",
    "          iv_sizing_max_factor (float) - max factor allowed (default 1.0)\n",
    "\n",
    "        Returns 1.0 when disabled or on error (no impact).\n",
    '        """\n',
    "        try:\n",
    '            if not bool(getattr(self.cfg, "iv_sizing_enabled", False)):\n',
    "                return 1.0\n",
    "        except Exception:\n",
    "            return 1.0\n",
    "\n",
    "        try:\n",
    '            lookback = int(getattr(self.cfg, "iv_sizing_lookback_days", 20) or 20)\n',
    "        except Exception:\n",
    "            lookback = 20\n",
    "        try:\n",
    '            high_pct = float(getattr(self.cfg, "iv_sizing_high_percentile", 75.0) or 75.0)\n',
    "        except Exception:\n",
    "            high_pct = 75.0\n",
    "        try:\n",
    '            low_pct = float(getattr(self.cfg, "iv_sizing_low_percentile", 25.0) or 25.0)\n',
    "        except Exception:\n",
    "            low_pct = 25.0\n",
    "        try:\n",
    '            high_factor = float(getattr(self.cfg, "iv_sizing_high_factor", 0.5) or 0.5)\n',
    "        except Exception:\n",
    "            high_factor = 0.5\n",
    "        try:\n",
    '            max_factor = float(getattr(self.cfg, "iv_sizing_max_factor", 1.0) or 1.0)\n',
    "        except Exception:\n",
    "            max_factor = 1.0\n",
    "\n",
    "        high_pct = max(1.0, min(99.0, float(high_pct)))\n",
    "        low_pct = max(0.0, min(float(high_pct) - 1.0, float(low_pct)))\n",
    "        lookback = max(2, min(100, int(lookback)))\n",
    "        high_factor = max(0.1, min(1.0, float(high_factor)))\n",
    "        max_factor = max(float(high_factor), min(2.0, float(max_factor)))\n",
    "\n",
    "        # Build IV/ATR history from tracking attribute\n",
    '        iv_hist = list(getattr(self, "_iv_pct_hist", []) or [])\n',
    "        if len(iv_hist) < int(lookback):\n",
    "            return 1.0\n",
    "\n",
    "        recent = list(iv_hist)\n",
    "        ordered = sorted(float(x) for x in recent if isinstance(x, (int, float)))\n",
    "        if len(ordered) < 2:\n",
    "            return 1.0\n",
    "\n",
    "        current = float(ordered[-1])\n",
    "        rank = sum(1 for x in ordered if x <= current)\n",
    "        percentile = (float(rank) / float(len(ordered))) * 100.0\n",
    "\n",
    "        if float(percentile) >= float(high_pct):\n",
    "            pct_above = (float(percentile) - float(high_pct)) / (100.0 - float(high_pct) + 1e-9)\n",
    "            pct_above = max(0.0, min(1.0, float(pct_above)))\n",
    "            factor = float(high_factor) * (1.0 + float(pct_above)) / 2.0\n",
    "            return max(float(high_factor) * 0.5, min(float(max_factor), float(factor)))\n",
    "        elif float(percentile) <= float(low_pct):\n",
    "            return min(float(max_factor), 1.0)\n",
    "        else:\n",
    "            return 1.0\n",
    "\n",
]

# Insert at line 6349 (0-indexed: 6348, before _held_days_since_open)
insert_idx = 6348  # 0-indexed, before line 6349 (1-indexed)
lines = lines[:insert_idx] + iv_method_lines + lines[insert_idx:]
print("OK: IV percentile factor method inserted")

# 1b. Modify _entry_qty to use IV percentile factor
# Find the line with "factor *= float(stop_factor)" (around line 3727)
for i, line in enumerate(lines):
    if 'factor *= float(stop_factor)' in line:
        # Insert the IV factor right after this line
        indent = " " * 8  # 8 spaces for indentation
        iv_lines = [
            f"\n",
            f"{indent}# IV Percentile Sizing Factor\n",
            f"{indent}iv_factor = self._entry_iv_percentile_factor()\n",
            f"{indent}factor *= float(iv_factor)\n",
        ]
        lines = lines[:i+1] + iv_lines + lines[i+1:]
        print(f"OK: IV factor added to _entry_qty at line {i+1}")
        break

# ============================================================
# FEATURE 2: Session-Specific Exit Adjustments
# ============================================================
# Insert _session_exit_adjustments before _manage_open_trades (line 11005, 0-indexed: 11004)

session_method_lines = [
    "    def _session_exit_adjustments(self) -> Dict[str, float]:\n",
    '        """Return exit parameter multipliers based on current trading session time.\n',
    "\n",
    "        Returns a dict with keys: sl_mult, tp_mult, trail_mult.\n",
    "\n",
    "        Reads config keys:\n",
    "          session_exit_enabled (bool) - master switch\n",
    "          session_opening_hhmm (str) - end of opening session (default \"10:00\")\n",
    "          session_lunch_start_hhmm (str) - start of lunch session (default \"12:00\")\n",
    "          session_lunch_end_hhmm (str) - end of lunch session (default \"13:30\")\n",
    "          session_close_hhmm (str) - start of closing session (default \"14:45\")\n",
    "          session_opening_sl_mult (float) - tighter stop in opening (default 0.7)\n",
    "          session_opening_tp_mult (float) - tighter target in opening (default 0.8)\n",
    "          session_lunch_sl_mult (float) - wider stop in lunch (default 1.3)\n",
    "          session_lunch_tp_mult (float) - wider target in lunch (default 1.2)\n",
    "          session_close_sl_mult (float) - tighter stop in closing (default 0.6)\n",
    "          session_close_tp_mult (float) - tighter target in closing (default 0.7)\n",
    "\n",
    "        Returns {sl: 1.0, tp: 1.0, trail: 1.0} when disabled or on error.\n",
    '        """\n',
    '        default = {"sl_mult": 1.0, "tp_mult": 1.0, "trail_mult": 1.0}\n',
    "\n",
    "        try:\n",
    '            if not bool(getattr(self.cfg, "session_exit_enabled", False)):\n',
    "                return dict(default)\n",
    "        except Exception:\n",
    "            return dict(default)\n",
    "\n",
    "        try:\n",
    '            opening_end = str(getattr(self.cfg, "session_opening_hhmm", "10:00") or "10:00").strip()\n',
    '            lunch_start = str(getattr(self.cfg, "session_lunch_start_hhmm", "12:00") or "12:00").strip()\n',
    '            lunch_end = str(getattr(self.cfg, "session_lunch_end_hhmm", "13:30") or "13:30").strip()\n',
    '            close_start = str(getattr(self.cfg, "session_close_hhmm", "14:45") or "14:45").strip()\n',
    '            opening_sl = float(getattr(self.cfg, "session_opening_sl_mult", 0.7) or 0.7)\n',
    '            opening_tp = float(getattr(self.cfg, "session_opening_tp_mult", 0.8) or 0.8)\n',
    '            lunch_sl = float(getattr(self.cfg, "session_lunch_sl_mult", 1.3) or 1.3)\n',
    '            lunch_tp = float(getattr(self.cfg, "session_lunch_tp_mult", 1.2) or 1.2)\n',
    '            close_sl = float(getattr(self.cfg, "session_close_sl_mult", 0.6) or 0.6)\n',
    '            close_tp = float(getattr(self.cfg, "session_close_tp_mult", 0.7) or 0.7)\n',
    "        except Exception:\n",
    "            return dict(default)\n",
    "\n",
    "        def _parse_hhmm(s: str) -> int:\n",
    "            try:\n",
    "                parts = str(s or \"\").strip().split(\":\")\n",
    "                return int(parts[0]) * 60 + int(parts[1])\n",
    "            except Exception:\n",
    "                return 0\n",
    "\n",
    "        now = dt_datetime.now(IST).time() if IST is not None else dt_datetime.now().time()\n",
    "        now_min = int(now.hour) * 60 + int(now.minute)\n",
    "\n",
    "        open_min = 9 * 60 + 15  # 09:15\n",
    "        opening_end_min = _parse_hhmm(opening_end) or (10 * 60)\n",
    "        lunch_start_min = _parse_hhmm(lunch_start) or (12 * 60)\n",
    "        lunch_end_min = _parse_hhmm(lunch_end) or (13 * 60 + 30)\n",
    "        close_start_min = _parse_hhmm(close_start) or (14 * 60 + 45)\n",
    "        market_close_min = 15 * 60 + 30  # 15:30\n",
    "\n",
    "        if now_min < open_min or now_min > market_close_min:\n",
    "            return dict(default)\n",
    "\n",
    "        sl_adj = 1.0\n",
    "        tp_adj = 1.0\n",
    "\n",
    "        if now_min <= opening_end_min:\n",
    "            sl_adj = float(opening_sl)\n",
    "            tp_adj = float(opening_tp)\n",
    "        elif now_min <= lunch_start_min:\n",
    "            sl_adj = 1.0\n",
    "            tp_adj = 1.0\n",
    "        elif now_min <= lunch_end_min:\n",
    "            sl_adj = float(lunch_sl)\n",
    "            tp_adj = float(lunch_tp)\n",
    "        elif now_min <= close_start_min:\n",
    "            sl_adj = 1.0\n",
    "            tp_adj = 1.0\n",
    "        else:\n",
    "            sl_adj = float(close_sl)\n",
    "            tp_adj = float(close_tp)\n",
    "\n",
    "        sl_adj = max(0.3, min(3.0, float(sl_adj)))\n",
    "        tp_adj = max(0.3, min(3.0, float(tp_adj)))\n",
    '        return {"sl_mult": float(sl_adj), "tp_mult": float(tp_adj), "trail_mult": float(sl_adj)}\n',
    "\n",
]

# Insert before _manage_open_trades (line 11005, 0-indexed: 11004)
# But offset might have changed due to earlier insertions. Let me recalculate.
# We inserted ~60 lines for IV method + 4 lines for IV factor in _entry_qty
# So the new line for _manage_open_trades is approximately 11005 + 64 = 11069
# Actually, let me find it dynamically:
manage_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith("def _manage_open_trades("):
        manage_idx = i
        break

if manage_idx is not None:
    lines = lines[:manage_idx] + session_method_lines + lines[manage_idx:]
    print(f"OK: Session exit adjustments inserted before _manage_open_trades at line {manage_idx+1}")
else:
    print("ERROR: Could not find _manage_open_trades")

# ============================================================
# FEATURE 3: Strategy Win-Rate Tracker
# ============================================================

# 3a. Add state tracking dicts after "# Runtime diagnostics / observability."
# Find this line dynamically (offset may have changed from prior insertions)
for i, line in enumerate(lines):
    if "# Runtime diagnostics / observability." in line:
        tracker_state = [
            "        # Strategy win-rate tracker: maps strategy_name -> {wins, losses, results}\n",
            "        self._strategy_winrate: Dict[str, Dict[str, object]] = {}\n",
            "        # Strategies temporarily disabled by the win-rate tracker.\n",
            "        self._disabled_strategies: Dict[str, float] = {}\n",
            "\n",
        ]
        lines = lines[:i+1] + tracker_state + lines[i+1:]
        print(f"OK: Winrate state tracking added at line {i+1}")
        break

# 3b. Insert winrate methods before _decide_entries
# Find _decide_entries dynamically
decide_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith("def _decide_entries("):
        decide_idx = i
        break

winrate_method_lines = [
    "    def _strategy_winrate_check(self, strategy_name: str) -> bool:\n",
    '        """Check if a strategy should be allowed based on its recent win-rate.\n',
    "\n",
    "        Returns True if allowed, False if blocked.\n",
    "\n",
    "        Reads config keys:\n",
    "          winrate_tracker_enabled (bool) - master switch\n",
    "          winrate_lookback (int) - how many recent trades to track (default 10)\n",
    "          winrate_min_wins (int) - minimum wins before filter activates (default 3)\n",
    "          winrate_max_loss_streak (int) - consecutive losses before disable (default 3)\n",
    "          winrate_min_win_rate (float) - min win rate % to stay enabled (default 30.0)\n",
    "          winrate_cooldown_sec (int) - seconds before retry (default 300)\n",
    '        """\n',
    "        try:\n",
    '            if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):\n',
    "                return True\n",
    "        except Exception:\n",
    "            return True\n",
    "\n",
    '        name = str(strategy_name or "").strip().lower()\n',
    '        if not name or name in {"auto", "directional"}:\n',
    "            return True\n",
    "\n",
    "        try:\n",
    '            lookback = int(getattr(self.cfg, "winrate_lookback", 10) or 10)\n',
    "        except Exception:\n",
    "            lookback = 10\n",
    "        try:\n",
    '            min_wins = int(getattr(self.cfg, "winrate_min_wins", 3) or 3)\n',
    "        except Exception:\n",
    "            min_wins = 3\n",
    "        try:\n",
    '            max_loss_streak = int(getattr(self.cfg, "winrate_max_loss_streak", 3) or 3)\n',
    "        except Exception:\n",
    "            max_loss_streak = 3\n",
    "        try:\n",
    '            min_win_rate = float(getattr(self.cfg, "winrate_min_win_rate", 30.0) or 30.0)\n',
    "        except Exception:\n",
    "            min_win_rate = 30.0\n",
    "        try:\n",
    '            cooldown = int(getattr(self.cfg, "winrate_cooldown_sec", 300) or 300)\n',
    "        except Exception:\n",
    "            cooldown = 300\n",
    "\n",
    "        lookback = max(2, min(50, int(lookback)))\n",
    "        min_wins = max(1, min(int(lookback) - 1, int(min_wins)))\n",
    "        max_loss_streak = max(1, min(int(lookback), int(max_loss_streak)))\n",
    "        min_win_rate = max(1.0, min(99.0, float(min_win_rate)))\n",
    "\n",
    "        now_ts = float(time.time())\n",
    "\n",
    "        # Check if currently disabled\n",
    '        disabled_until = float(self._disabled_strategies.get(name, 0.0) or 0.0)\n',
    "        if disabled_until > 0:\n",
    "            if now_ts < disabled_until:\n",
    "                return False\n",
    "            else:\n",
    "                self._disabled_strategies.pop(name, None)\n",
    "                return True\n",
    "\n",
    "        # Get or create tracker entry\n",
    '        tracker = self._strategy_winrate.setdefault(name, {"results": [], "wins": 0, "losses": 0})\n',
    "        try:\n",
    '            results = list(tracker.get("results") or [])\n',
    "        except Exception:\n",
    "            results = []\n",
    "\n",
    "        if len(results) < int(min_wins):\n",
    "            return True\n",
    "\n",
    "        recent = list(results)[-int(lookback):]\n",
    "        wins = sum(1 for r in recent if r is True)\n",
    "        losses = sum(1 for r in recent if r is False)\n",
    "\n",
    "        # Consecutive loss check\n",
    "        consec_losses = 0\n",
    "        for r in reversed(recent):\n",
    "            if r is False:\n",
    "                consec_losses += 1\n",
    "            else:\n",
    "                break\n",
    "\n",
    "        if int(consec_losses) >= int(max_loss_streak):\n",
    "            self._disabled_strategies[name] = float(now_ts) + float(cooldown)\n",
    '            print(f"[WINRATE] {name}: {consec_losses} consecutive losses - disabled for {cooldown}s")\n',
    "            return False\n",
    "\n",
    "        # Win rate check\n",
    "        if wins + losses > 0:\n",
    "            win_rate = (float(wins) / float(wins + losses)) * 100.0\n",
    "            if float(win_rate) < float(min_win_rate):\n",
    "                self._disabled_strategies[name] = float(now_ts) + float(cooldown)\n",
    '                print(f"[WINRATE] {name}: {win_rate:.1f}% win rate (< {min_win_rate}%) - disabled for {cooldown}s")\n',
    "                return False\n",
    "\n",
    "        return True\n",
    "\n",
    "    def _strategy_winrate_record_result(self, strategy_name: str, won: bool) -> None:\n",
    '        """Record a win or loss for a strategy."""\n',
    '        if not bool(getattr(self.cfg, "winrate_tracker_enabled", False)):\n',
    "            return\n",
    "\n",
    '        name = str(strategy_name or "").strip().lower()\n',
    '        if not name or name in {"auto", "directional"}:\n',
    "            return\n",
    "\n",
    "        try:\n",
    '            lookback = int(getattr(self.cfg, "winrate_lookback", 10) or 10)\n',
    "        except Exception:\n",
    "            lookback = 10\n",
    "        lookback = max(2, min(50, int(lookback)))\n",
    "\n",
    '        tracker = self._strategy_winrate.setdefault(name, {"results": [], "wins": 0, "losses": 0})\n',
    "        try:\n",
    '            results = list(tracker.get("results") or [])\n',
    "        except Exception:\n",
    "            results = []\n",
    "\n",
    "        results.append(bool(won))\n",
    "        if len(results) > int(lookback) * 2:\n",
    "            results = results[-int(lookback) * 2:]\n",
    "\n",
    '        wins = sum(1 for r in results if r is True)\n',
    '        losses = sum(1 for r in results if r is False)\n',
    "\n",
    '        tracker["results"] = results\n',
    '        tracker["wins"] = int(wins)\n',
    '        tracker["losses"] = int(losses)\n',
    "\n",
]

if decide_idx is not None:
    lines = lines[:decide_idx] + winrate_method_lines + lines[decide_idx:]
    print(f"OK: Winrate methods inserted before _decide_entries at line {decide_idx+1}")
else:
    print("ERROR: Could not find _decide_entries")

# 3c. Integrate winrate check into entry flow
# Find "# Re-evaluate signals with new scores"
for i, line in enumerate(lines):
    if "# Re-evaluate signals with new scores" in line:
        # Find the next blank line after this block, or the line before the winrate integration
        # Actually, we need to find where forced_directional_name/effective_strat are used
        # Insert the check right after the comment line
        winrate_integration = [
            "            # Win-rate tracker check for strategy\n",
            '            if getattr(self.cfg, "winrate_tracker_enabled", False):\n',
            "                try:\n",
            '                    strat_check = str(forced_directional_name or effective_strat or "").strip().lower()\n',
            '                    if strat_check and strat_check not in {"auto", "directional"}:\n',
            "                        if not self._strategy_winrate_check(strat_check):\n",
            '                            if maybe_reroute_entry_block(f"Winrate filter blocked {strat_check}", current_strategy=str(forced_directional_name or effective_strat)):\n',
            "                                return\n",
            "                            return\n",
            "                except Exception:\n",
            "                    pass\n",
            "\n",
        ]
        lines = lines[:i+1] + winrate_integration + lines[i+1:]
        print(f"OK: Winrate integration added at line {i+1}")
        break

# 3d. Record results when a trade closes
# Find "self.state.realized_pnl += float(r)" - there are multiple occurrences
# We want the one in _manage_open_trades where trades are closed
# The relevant one is around the directional exit logic
for i, line in enumerate(lines):
    if "self.state.realized_pnl += float(r)" in line:
        # Check if we're in the trade closing context (look for "tr" variable name nearby)
        if i > 11000:  # Only the ones in the exit/close section
            winrate_record = [
                "                        # Record win/loss for winrate tracker\n",
                "                        try:\n",
                '                            tr_name = str(tr.get("name") or "").strip().lower()\n',
                '                            if tr_name and tr_name not in {"auto", "directional"}:\n',
                "                                self._strategy_winrate_record_result(tr_name, float(r) > 0)\n",
                "                        except Exception:\n",
                "                            pass\n",
                "\n",
            ]
            lines = lines[:i] + winrate_record + lines[i:]
            print(f"OK: Winrate recording added at line {i+1}")
            break

# ============================================================
# Write the patched file
# ============================================================
with open(FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\nAll patches applied to {FILE}!")
print(f"Total lines: {len(lines)}")

# Verify syntax
content_out = "".join(lines)
try:
    compile(content_out, FILE, "exec")
    print("Syntax check: PASSED")
except SyntaxError as e:
    print(f"Syntax check: FAILED - {e}")
