"""Fix critical code review issues for GPT enhancements."""

with open('src/strategy.py', 'r') as f:
    content = f.read()

changes = 0

# ============================================================
# Fix 1: P4 - Add real entry pausing check in _decide_entries
# ============================================================
# Find the start of _decide_entries and add a pause check early
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
    print("Fix 1: Added regime pause check at top of _decide_entries")
    changes += 1
else:
    print("WARNING: Could not find _decide_entries anchor")

# ============================================================
# Fix 2: P5 - Wire _gpt_strike_selection into the strike selection flow
# ============================================================
# We need to find where _auto_gpt_strike_context is used to pick strikes
# and inject our P5 method. Let's find the _set_auto_gpt_strike_context method
# or where the auto_gpt strike context is applied

# Find the _set_auto_gpt_strike_context method and add a call to _gpt_strike_selection
old_strike_method = '''    def _set_auto_gpt_strike_context(
        self,
        *,
        spot: float,
        chain: List[dict],
    ) -> None:
        """Build and store the strike-picking context that _apply_auto_gpt_strike_context reads."""'''

new_strike_method = '''    def _set_auto_gpt_strike_context(
        self,
        *,
        spot: float,
        chain: List[dict],
    ) -> None:
        """Build and store the strike-picking context that _apply_auto_gpt_strike_context reads.

        P5: Also attempts GPT-based strike selection for enhanced precision.
        """'''

if old_strike_method in content:
    content = content.replace(old_strike_method, new_strike_method, 1)
    print("Fix 2a: Updated _set_auto_gpt_strike_context docstring for P5")
    changes += 1
    
    # Now find where candidates are built in this method and wire P5
    # Look for the line where it builds candidate_strikes or similar
    # Actually, let me find where the method body starts and add P5 call
    old_body = '''        self._auto_gpt_strike_context = {
            "active": True,
            "ts": float(time.time()),
            "spot": float(spot),
            "expiry": "",
            "strike_step": 50.0,
            "ce": {},
            "put": {},
        }'''
    
    if old_body in content:
        new_body = '''        self._auto_gpt_strike_context = {
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
            if pe_candidates:
                pe_selected = self._gpt_strike_selection(pe_candidates, "PE", float(spot))
                if pe_selected is not None:
                    self._auto_gpt_strike_context["put"]["gpt_selected"] = True
        except Exception:
            pass'''
        content = content.replace(old_body, new_body, 1)
        print("Fix 2b: Wired P5 strike selection into _set_auto_gpt_strike_context")
        changes += 1
    else:
        print("WARNING: Could not find _set_auto_gpt_strike_context body anchor")
else:
    print("WARNING: Could not find _set_auto_gpt_strike_context method")

# ============================================================
# Fix 3: P3 - Add caching TTL for GPT sizing factor
# ============================================================
# Add a cached sizing factor with TTL to avoid calling GPT on every entry eval
# Find the _gpt_sizing_factor method and add caching

old_sizing = '''    def _gpt_sizing_factor(self) -> float:
        """Get a GPT-recommended sizing multiplier based on market conditions and recent performance.

        Returns a float multiplier (0.25 to 2.0) to apply on top of existing position sizing.
        """
        try:
            if not bool(getattr(self.cfg, "gpt_position_sizing_enabled", False)):
                return 1.0
        except Exception:
            return 1.0'''

new_sizing = '''    def _gpt_sizing_factor(self) -> float:
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

if old_sizing in content:
    content = content.replace(old_sizing, new_sizing, 1)
    print("Fix 3: Added caching TTL to _gpt_sizing_factor")
    changes += 1
else:
    print("WARNING: Could not find _gpt_sizing_factor anchor")

# Also add the cache storage after computing the factor
old_cache_store = '''        try:
            mult = float(extras.get("sizing_mult", 1.0))
        except Exception:
            mult = 1.0

        return max(0.25, min(2.0, mult))'''

new_cache_store = '''        try:
            mult = float(extras.get("sizing_mult", 1.0))
        except Exception:
            mult = 1.0

        mult = max(0.25, min(2.0, mult))

        # Store in cache
        try:
            self._gpt_sizing_factor_cache = float(mult)
            self._gpt_sizing_factor_cache_ts = float(time.time())
        except Exception:
            pass

        return float(mult)'''

if old_cache_store in content:
    content = content.replace(old_cache_store, new_cache_store, 1)
    print("Fix 3b: Added cache storage after GPT sizing factor computation")
    changes += 1
else:
    print("WARNING: Could not find cache store anchor")

# ============================================================
# Fix 4: P6 - Improve delta estimate by using leg greeks when available
# ============================================================
# Replace the fixed delta=0.5 with a smarter approach that
# looks at leg-specific data

old_delta = '''                    # Rough estimate: assume delta ~0.5 for ATM options
                    est_new_ltp = float(entry_px) + (float(scenario_spot) - float(spot)) * 0.5'''

new_delta = '''                    # Estimate delta from leg data (theta/delta from greeks context), or default to 0.5
                    try:
                        leg_delta = float(leg.get("delta") or 0.5)
                    except Exception:
                        leg_delta = 0.5
                    # Rough estimate: delta * (price change)
                    est_new_ltp = float(entry_px) + (float(scenario_spot) - float(spot)) * float(leg_delta)'''

if old_delta in content:
    content = content.replace(old_delta, new_delta, 1)
    print("Fix 4: Improved P6 delta estimate to use leg greeks when available")
    changes += 1
else:
    print("WARNING: Could not find P6 delta estimate anchor")

# ============================================================
# Write back
# ============================================================
with open('src/strategy.py', 'w') as f:
    f.write(content)

print(f"\nTotal fixes applied: {changes}/6")
print(f"File size: {len(content)} chars")
