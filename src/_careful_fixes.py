"""Apply all remaining fixes with verification after each step."""

import subprocess
import sys

def compile_check(path):
    result = subprocess.run(
        ["python", "-m", "py_compile", path],
        capture_output=True, text=True, cwd=work_dir
    )
    return result.returncode == 0, result.stderr

work_dir = r"C:\Users\rahul\Downloads\NiftyScalper"

with open('src/strategy.py', 'r') as f:
    content = f.read()

changes = []
steps = []

# ============================================================
# Fix 1: P3 - GPT sizing factor caching
# ============================================================
# This doesn't need anchor matching, it's in the _gpt_sizing_factor method
# which was already added in the initial insertion

# ============================================================
# Fix 2: P4 - Entry pause check in _decide_entries
# ============================================================
old_decide = '''    def _decide_entries(self) -> None:
        """Evaluate all eligible strategies and attempt entry once per cycle."""'''

new_decide = '''    def _decide_entries(self) -> None:
        """Evaluate all eligible strategies and attempt entry once per cycle."""

        # P4: Check if entries are paused by regime monitor
        try:
            if bool(self._gpt_regime_actions.get("_entries_paused", False)):
                return
        except Exception:
            pass'''

if old_decide in content:
    content = content.replace(old_decide, new_decide, 1)
    with open('src/strategy.py', 'w') as f:
        f.write(content)
    ok, err = compile_check('src/strategy.py')
    if ok:
        changes.append("P4: Entry pause check in _decide_entries ✓")
    else:
        print(f"ERROR after Fix 2: {err}")
        sys.exit(1)
else:
    print("WARNING: Fix 2 anchor not found")

# ============================================================
# Fix 3: P4 - Regime sizing_mult in _entry_qty
# ============================================================
# Find the section where gpt_factor is applied and add regime sizing_mult after it
old_qty = '''        # P3: GPT-based position sizing multiplier
        try:
            gpt_factor = self._gpt_sizing_factor()
        except Exception:
            gpt_factor = 1.0
        if 0.25 <= float(gpt_factor) <= 2.0 and float(gpt_factor) != 1.0:
            qty = max(base_qty, int(round(float(qty) * float(gpt_factor))))'''

new_qty = '''        # P3: GPT-based position sizing multiplier
        try:
            gpt_factor = self._gpt_sizing_factor()
        except Exception:
            gpt_factor = 1.0
        if 0.25 <= float(gpt_factor) <= 2.0 and float(gpt_factor) != 1.0:
            qty = max(base_qty, int(round(float(qty) * float(gpt_factor))))

        # P4: Apply regime-based sizing multiplier on top of GPT factor
        try:
            regime_mult = float(self._gpt_regime_actions.get("sizing_mult", 1.0) or 1.0)
            if 0.25 <= float(regime_mult) <= 2.0 and float(regime_mult) != 1.0:
                qty = max(base_qty, int(round(float(qty) * float(regime_mult))))
        except Exception:
            pass'''

if old_qty in content:
    content = content.replace(old_qty, new_qty, 1)
    with open('src/strategy.py', 'w') as f:
        f.write(content)
    ok, err = compile_check('src/strategy.py')
    if ok:
        changes.append("P4: Regime sizing_mult in _entry_qty ✓")
    else:
        print(f"ERROR after Fix 3: {err}")
        sys.exit(1)
else:
    print("WARNING: Fix 3 anchor not found")

# ============================================================
# Fix 4: P4 - Regime stops_mult in directional exit logic (BOTH bull and bear)
# ============================================================
# Find both occurrences of dir_sl_atr_mult assignments
old_sl_bull = '''                            # Explicit ATR Stop Loss
                            sl_mult = float(getattr(self.cfg, "dir_sl_atr_mult", 1.5))
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)

                            eff_stop = float(entry_spot) + (loss_mult * float(atr_val))'''

new_sl_bull = '''                            # Explicit ATR Stop Loss
                            sl_mult = float(getattr(self.cfg, "dir_sl_atr_mult", 1.5))
                            # P4: Apply regime-based stop multiplier
                            try:
                                regime_stops = float(self._gpt_regime_actions.get("stops_mult", 1.0) or 1.0)
                                if float(regime_stops) > 0 and float(regime_stops) != 1.0:
                                    sl_mult = max(0.5, float(sl_mult) * float(regime_stops))
                            except Exception:
                                pass
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)

                            eff_stop = float(entry_spot) + (loss_mult * float(atr_val))'''

if old_sl_bull in content:
    content = content.replace(old_sl_bull, new_sl_bull, 1)
    changes.append("P4: Regime stops_mult in bull exit ✓")
else:
    print("WARNING: Fix 4 (bull) anchor not found")

# Find the bear equivalent
old_sl_bear = '''                            # Explicit ATR Stop Loss
                            sl_mult = float(getattr(self.cfg, "dir_sl_atr_mult", 1.5))
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)

                            eff_stop = float(entry_spot) - (loss_mult * float(atr_val))'''

new_sl_bear = '''                            # Explicit ATR Stop Loss
                            sl_mult = float(getattr(self.cfg, "dir_sl_atr_mult", 1.5))
                            # P4: Apply regime-based stop multiplier
                            try:
                                regime_stops = float(self._gpt_regime_actions.get("stops_mult", 1.0) or 1.0)
                                if float(regime_stops) > 0 and float(regime_stops) != 1.0:
                                    sl_mult = max(0.5, float(sl_mult) * float(regime_stops))
                            except Exception:
                                pass
                            loss_mult = -sl_mult
                            if float(be_mult_local) < 0:
                                loss_mult = float(be_mult_local)

                            eff_stop = float(entry_spot) - (loss_mult * float(atr_val))'''

if old_sl_bear in content:
    content = content.replace(old_sl_bear, new_sl_bear, 1)
    changes.append("P4: Regime stops_mult in bear exit ✓")
else:
    print("WARNING: Fix 4 (bear) anchor not found")

# Write and compile after Fix 4
with open('src/strategy.py', 'w') as f:
    f.write(content)
ok, err = compile_check('src/strategy.py')
if ok:
    pass  # Already counted changes
else:
    print(f"ERROR after Fix 4: {err}")
    sys.exit(1)

# ============================================================
# Fix 5: P1 - TRAIL_TIGHTER override application in _manage_open_trades
# ============================================================
old_trail = '''            try:
                trail_mult = float(getattr(self.cfg, "dir_trail_atr_mult", 1.0) or 1.0)
            except Exception:
                trail_mult = 1.0'''

new_trail = '''            try:
                trail_mult = float(getattr(self.cfg, "dir_trail_atr_mult", 1.0) or 1.0)
            except Exception:
                trail_mult = 1.0
            # P1: Allow GPT exit management to override trail multiplier
            try:
                if isinstance(tr, dict):
                    gpt_trail = tr.get("_gpt_trail_override")
                    gpt_trail_ts = tr.get("_gpt_trail_override_ts")
                    if isinstance(gpt_trail, (int, float)) and float(gpt_trail) > 0:
                        # Only apply for 5 minutes after GPT recommendation
                        if gpt_trail_ts is None or (time.time() - float(gpt_trail_ts)) < 300.0:
                            trail_mult = float(gpt_trail)
            except Exception:
                pass'''

if old_trail in content:
    content = content.replace(old_trail, new_trail, 1)
    with open('src/strategy.py', 'w') as f:
        f.write(content)
    ok, err = compile_check('src/strategy.py')
    if ok:
        changes.append("P1: TRAIL_TIGHTER override applied ✓")
    else:
        print(f"ERROR after Fix 5: {err}")
        sys.exit(1)
else:
    print("WARNING: Fix 5 anchor not found")

# ============================================================
# Fix 6: P5 - Wire _gpt_strike_selection into _set_auto_gpt_strike_context
# ============================================================
old_strike_body = '''        self._auto_gpt_strike_context = {
            "active": True,
            "ts": float(time.time()),
            "spot": float(spot),
            "expiry": "",
            "strike_step": 50.0,
            "ce": {},
            "put": {},
        }'''

new_strike_body = '''        self._auto_gpt_strike_context = {
            "active": True,
            "ts": float(time.time()),
            "spot": float(spot),
            "expiry": "",
            "strike_step": 50.0,
            "ce": {},
            "put": {},
        }

        # P5: Attempt enhanced GPT-based strike selection for CE and PE
        try:
            ce_candidates = [r for r in chain if str(r.get("option_type") or "").upper() == "CE"]
            pe_candidates = [r for r in chain if str(r.get("option_type") or "").upper() == "PE"]
            if ce_candidates:
                ce_selected = self._gpt_strike_selection(ce_candidates, "CE", float(spot))
                if ce_selected is not None:
                    self._auto_gpt_strike_context["ce"]["gpt_selected"] = True
                    self._auto_gpt_strike_context["ce"]["gpt_strike"] = float(ce_selected.get("strike") or 0)
            if pe_candidates:
                pe_selected = self._gpt_strike_selection(pe_candidates, "PE", float(spot))
                if pe_selected is not None:
                    self._auto_gpt_strike_context["put"]["gpt_selected"] = True
                    self._auto_gpt_strike_context["put"]["gpt_strike"] = float(pe_selected.get("strike") or 0)
        except Exception:
            pass'''

if old_strike_body in content:
    content = content.replace(old_strike_body, new_strike_body, 1)
    with open('src/strategy.py', 'w') as f:
        f.write(content)
    ok, err = compile_check('src/strategy.py')
    if ok:
        changes.append("P5: Wired into _set_auto_gpt_strike_context ✓")
    else:
        print(f"ERROR after Fix 6: {err}")
        sys.exit(1)
else:
    print("WARNING: Fix 6 anchor not found")

# ============================================================
# Fix 7: P3 - GPT sizing factor caching (within _gpt_sizing_factor method)
# ============================================================
# Add caching logic at the beginning of the method
old_sizing_start = '''    def _gpt_sizing_factor(self) -> float:
        """Get a GPT-recommended sizing multiplier based on market conditions and recent performance.

        Returns a float multiplier (0.25 to 2.0) to apply on top of existing position sizing.
        """
        try:
            if not bool(getattr(self.cfg, "gpt_position_sizing_enabled", False)):
                return 1.0
        except Exception:
            return 1.0'''

new_sizing_start = '''    def _gpt_sizing_factor(self) -> float:
        """Get a GPT-recommended sizing multiplier based on market conditions and recent performance.

        Returns a float multiplier (0.25 to 2.0) to apply on top of existing position sizing.
        Cached for 60 seconds to avoid calling GPT on every entry evaluation cycle.
        """
        try:
            if not bool(getattr(self.cfg, "gpt_position_sizing_enabled", False)):
                return 1.0
        except Exception:
            return 1.0

        # Caching with TTL
        try:
            cache_ttl = 60.0
            now = float(time.time())
            cached_mult = getattr(self, "_gpt_sizing_factor_cache", None)
            cached_ts = getattr(self, "_gpt_sizing_factor_cache_ts", 0.0)
            if isinstance(cached_mult, (int, float)) and float(cached_ts) > 0 and (now - float(cached_ts)) < cache_ttl:
                return float(cached_mult)
        except Exception:
            pass'''

if old_sizing_start in content:
    content = content.replace(old_sizing_start, new_sizing_start, 1)
    with open('src/strategy.py', 'w') as f:
        f.write(content)
    ok, err = compile_check('src/strategy.py')
    if ok:
        changes.append("P3: GPT sizing factor caching ✓")
    else:
        print(f"ERROR after Fix 7: {err}")
        sys.exit(1)
else:
    print("WARNING: Fix 7 anchor not found")

# Also add cache storage after the return value computation
old_sizing_end = '''        return max(0.25, min(2.0, mult))'''

new_sizing_end = '''        mult = max(0.25, min(2.0, mult))

        # Store in cache
        try:
            self._gpt_sizing_factor_cache = float(mult)
            self._gpt_sizing_factor_cache_ts = float(time.time())
        except Exception:
            pass

        return float(mult)'''

if old_sizing_end in content:
    content = content.replace(old_sizing_end, new_sizing_end, 1)
    with open('src/strategy.py', 'w') as f:
        f.write(content)
    ok, err = compile_check('src/strategy.py')
    if ok:
        changes.append("P3: GPT sizing factor cache storage ✓")
    else:
        print(f"ERROR after Fix 7b: {err}")
        sys.exit(1)
else:
    print("WARNING: Fix 7b anchor not found")

# ============================================================
# Fix 8: P6 - Improve delta estimate in what-if analysis
# ============================================================
old_delta = '''                    # Rough estimate: assume delta ~0.5 for ATM options
                    est_new_ltp = float(entry_px) + (float(scenario_spot) - float(spot)) * 0.5'''

new_delta = '''                    # Estimate delta from leg data or default to 0.5 for ATM options
                    try:
                        leg_delta = float(leg.get("delta") or 0.5)
                    except Exception:
                        leg_delta = 0.5
                    est_new_ltp = float(entry_px) + (float(scenario_spot) - float(spot)) * float(leg_delta)'''

if old_delta in content:
    content = content.replace(old_delta, new_delta, 1)
    with open('src/strategy.py', 'w') as f:
        f.write(content)
    ok, err = compile_check('src/strategy.py')
    if ok:
        changes.append("P6: Improved delta estimate ✓")
    else:
        print(f"ERROR after Fix 8: {err}")
        sys.exit(1)
else:
    print("WARNING: Fix 8 anchor not found")

# ============================================================
# Final verification
# ============================================================
ok, err = compile_check('src/strategy.py')
if ok:
    print(f"\nAll fixes applied successfully! ({len(changes)} changes)")
    for c in changes:
        print(f"  - {c}")
else:
    print(f"\nFINAL COMPILATION ERROR: {err}")
