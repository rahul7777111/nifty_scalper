"""Fix remaining critical issues from code review."""

with open('src/strategy.py', 'r') as f:
    lines = f.readlines()

changes = []

# ============================================================
# Fix 1: Wire P5 into _set_auto_gpt_strike_context method body
# ============================================================
# Find the method body - look for the closing of the context dict initialization
for i, line in enumerate(lines):
    if '    def _set_auto_gpt_strike_context(' in line:
        # Find where the dict initialization ends
        for j in range(i, min(i+50, len(lines))):
            if 'self._auto_gpt_strike_context = {' in lines[j]:
                # Find closing brace
                for k in range(j, min(j+30, len(lines))):
                    if lines[k].strip() == '}' and k > j:
                        # Check if P5 wire already exists
                        p5_exists = any('gpt_strike_selection' in lines[m] for m in range(k, min(k+20, len(lines))))
                        if not p5_exists:
                            p5_lines = [
                                "\n",
                                "        # P5: Attempt enhanced GPT-based strike selection for CE and PE\n",
                                "        try:\n",
                                '            ce_candidates = [r for r in chain if str(r.get("option_type") or "").upper() == "CE"]\n',
                                '            pe_candidates = [r for r in chain if str(r.get("option_type") or "").upper() == "PE"]\n',
                                "            if ce_candidates:\n",
                                '                ce_selected = self._gpt_strike_selection(ce_candidates, "CE", float(spot))\n',
                                "                if ce_selected is not None:\n",
                                '                    self._auto_gpt_strike_context["ce"]["gpt_selected"] = True\n',
                                "            if pe_candidates:\n",
                                '                pe_selected = self._gpt_strike_selection(pe_candidates, "PE", float(spot))\n',
                                "                if pe_selected is not None:\n",
                                '                    self._auto_gpt_strike_context["put"]["gpt_selected"] = True\n',
                                "        except Exception:\n",
                                "            pass\n",
                            ]
                            for m, pl in enumerate(p5_lines):
                                lines.insert(k + 1 + m, pl)
                            changes.append(f"P5 wired into _set_auto_gpt_strike_context (line {k+1})")
                        break
                break
        break

# ============================================================
# Fix 2: Apply TRAIL_TIGHTER override in _manage_open_trades directional exit logic
# ============================================================
# Find where dir_trail_atr_mult is read for directional trades and add override check
for i, line in enumerate(lines):
    if 'trail_mult = float(getattr(self.cfg, "dir_trail_atr_mult", 1.0) or 1.0)' in line:
        # Check if override already applied
        has_override = False
        for j in range(i, min(i+10, len(lines))):
            if '_gpt_trail_override' in lines[j]:
                has_override = True
                break
        if not has_override:
            # Add override check after the trail_mult assignment
            override_lines = [
                "        # P1: Allow GPT exit management to override trail multiplier\n",
                "        try:\n",
                '            if isinstance(tr, dict):\n',
                '                gpt_trail = tr.get("_gpt_trail_override")\n',
                '                gpt_trail_ts = tr.get("_gpt_trail_override_ts")\n',
                "                if isinstance(gpt_trail, (int, float)) and float(gpt_trail) > 0:\n",
                "                    # Only apply for 5 minutes after GPT recommendation\n",
                '                    if gpt_trail_ts is None or (time.time() - float(gpt_trail_ts)) < 300.0:\n',
                "                        trail_mult = float(gpt_trail)\n",
                "        except Exception:\n",
                "            pass\n",
            ]
            # Insert after the trail_mult assignment line
            for j, pl in enumerate(override_lines):
                lines.insert(i + 1 + j, pl)
            changes.append(f"P1 TRAIL_TIGHTER override applied (line {i+1})")
        break

# ============================================================
# Fix 3: Apply P4 regime sizing_mult in _entry_qty
# ============================================================
# Find where gpt_factor is applied in _entry_qty
for i, line in enumerate(lines):
    if 'gpt_factor = self._gpt_sizing_factor()' in line:
        # After the gpt_factor application, add regime sizing_mult
        for j in range(i, min(i+15, len(lines))):
            if 'if 0.25 <= float(gpt_factor) <= 2.0 and float(gpt_factor) != 1.0:' in lines[j]:
                has_regime = False
                for k in range(j, min(j+10, len(lines))):
                    if 'regime_actions' in lines[k] or 'sizing_mult' in lines[k]:
                        has_regime = True
                        break
                if not has_regime:
                    # After the gpt_factor application, add regime sizing multiplier
                    regime_lines = [
                        "        # P4: Apply regime-based sizing multiplier on top of GPT factor\n",
                        "        try:\n",
                        '            regime_mult = float(self._gpt_regime_actions.get("sizing_mult", 1.0) or 1.0)\n',
                        "            if 0.25 <= float(regime_mult) <= 2.0 and float(regime_mult) != 1.0:\n",
                        "                qty = max(base_qty, int(round(float(qty) * float(regime_mult))))\n",
                        "        except Exception:\n",
                        "            pass\n",
                    ]
                    for k, pl in enumerate(regime_lines):
                        lines.insert(j + 7 + k, pl)
                    changes.append(f"P4 regime sizing_mult applied in _entry_qty (line {j+1})")
                break
        break

# ============================================================
# Fix 4: Apply P4 regime stops_mult in directional exit logic
# ============================================================
# Find where dir_sl_atr_mult is read and add regime override
for i, line in enumerate(lines):
    if 'sl_mult = float(getattr(self.cfg, "dir_sl_atr_mult", 1.5))' in line:
        has_regime_stop = False
        for j in range(i, min(i+10, len(lines))):
            if '_gpt_regime_actions' in lines[j]:
                has_regime_stop = True
                break
        if not has_regime_stop:
            stop_lines = [
                "                        # P4: Apply regime-based stop multiplier\n",
                "                        try:\n",
                '                            regime_stops = float(self._gpt_regime_actions.get("stops_mult", 1.0) or 1.0)\n',
                "                            if float(regime_stops) > 0 and float(regime_stops) != 1.0:\n",
                "                                sl_mult = max(0.5, float(sl_mult) * float(regime_stops))\n",
                "                        except Exception:\n",
                "                            pass\n",
            ]
            for j, pl in enumerate(stop_lines):
                lines.insert(i + 1 + j, pl)
            changes.append(f"P4 regime stops_mult applied (line {i+1})")
        break

# ============================================================
# Write back
# ============================================================
with open('src/strategy.py', 'w') as f:
    f.writelines(lines)

print("Fixes applied:")
for c in changes:
    print(f"  - {c}")
print(f"\nTotal: {len(changes)} fixes")
