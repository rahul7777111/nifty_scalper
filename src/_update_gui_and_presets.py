"""
Standalone script: update ui.py settings popup and config.py presets
for the 5 new features (IV filter, MTF hard filter, theta stop widen,
exit ladder, DH vol adjust).
"""

import re
import os


def patch_ui_py(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        original = content

    changes = []

    # --- 1. vars_to_track dict: add new entries before the closing brace ---
    # Find the closing brace of vars_to_track dict
    marker_vars_end = '"Exit Wing Touch": exit_wing_var,'
    new_vars_entries = '''
                "IV Filter": iv_filter_enabled_var,
                "MTF Hard Filter": mtf_hard_filter_var,
                "Theta Stop Widen": theta_stop_widen_var,
                "Theta Stop Max Mult": theta_stop_max_mult_var,
                "Theta Stop Threshold": theta_stop_threshold_var,
                "Exit Ladder": exit_ladder_enabled_var,
                "Exit Ladder Steps": exit_ladder_steps_var,
                "DH Vol Adjust": dh_vol_adjust_var,
                "DH Vol Low Factor": dh_vol_low_factor_var,
                "DH Vol High Factor": dh_vol_high_factor_var,
            '''
    if marker_vars_end in content:
        old = marker_vars_end + ',\n            }'
        new = marker_vars_end + ',\n' + new_vars_entries + '        }'
        content = content.replace(old, new)
        changes.append("vars_to_track: added 10 new entries")
    else:
        changes.append("WARNING: vars_to_track marker not found")

    # --- 2. Variable declarations: add before they are used (before the interactive section starts) ---
    # We'll inject them right before the Delta/Theta filter controls section
    marker_vars_decl = 'delta_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_delta_strike_selection", False)))'
    new_vars_decl = '''        iv_filter_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "iv_filter_enabled", False)))
        mtf_hard_filter_var = tk.BooleanVar(value=bool(getattr(cfg, "mtf_hard_filter", False)))
        theta_stop_widen_var = tk.BooleanVar(value=bool(getattr(cfg, "theta_stop_widen_enabled", False)))
        theta_stop_max_mult_var = tk.StringVar(value=str(getattr(cfg, "theta_stop_widen_max_mult", 2.0)))
        theta_stop_threshold_var = tk.StringVar(value=str(getattr(cfg, "theta_stop_widen_threshold", -5.0)))
        exit_ladder_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "exit_ladder_enabled", False)))
        exit_ladder_steps_var = tk.StringVar(value=str(getattr(cfg, "exit_ladder_steps", "0.25:1.0,0.25:1.5,0.25:2.0")))
        dh_vol_adjust_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_vol_adjust_enabled", False)))
        dh_vol_low_factor_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_vol_low_tol_factor", 0.7)))
        dh_vol_high_factor_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_vol_high_tol_factor", 1.5)))
'''
    if marker_vars_decl in content:
        content = content.replace(marker_vars_decl, new_vars_decl + '\n        ' + marker_vars_decl)
        changes.append("variable declarations: added 10 variables before delta_enabled_var")
    else:
        changes.append("WARNING: variable declaration marker not found")

    # --- 3. Interactive controls: add filter checkboxes and entry fields ---
    # We'll add them right after the Hard CHOP / RSI Confluence line (the closing of filter_frame)
    marker_ctrl = 'rsi_conf_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_rsi_confluence", True)))'
    new_ctrl = '''rsi_conf_var = tk.BooleanVar(value=bool(getattr(cfg, "enable_rsi_confluence", True)))
        iv_filter_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "iv_filter_enabled", False)))
        mtf_hard_filter_var = tk.BooleanVar(value=bool(getattr(cfg, "mtf_hard_filter", False)))
        theta_stop_widen_var = tk.BooleanVar(value=bool(getattr(cfg, "theta_stop_widen_enabled", False)))
        theta_stop_max_mult_var = tk.StringVar(value=str(getattr(cfg, "theta_stop_widen_max_mult", 2.0)))
        theta_stop_threshold_var = tk.StringVar(value=str(getattr(cfg, "theta_stop_widen_threshold", -5.0)))
        exit_ladder_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, "exit_ladder_enabled", False)))
        exit_ladder_steps_var = tk.StringVar(value=str(getattr(cfg, "exit_ladder_steps", "0.25:1.0,0.25:1.5,0.25:2.0")))
        dh_vol_adjust_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_vol_adjust_enabled", False)))
        dh_vol_low_factor_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_vol_low_tol_factor", 0.7)))
        dh_vol_high_factor_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_vol_high_tol_factor", 1.5)))
'''
    # Actually, the above will create duplicates since we already injected them. Let me take a different approach.
    # Instead, I'll add the interactive UI widgets AFTER the rsi_confluence checkbox grid placement.
    # Let me find where the widgets are actually placed.

    # Actually, the variable declarations and the grid() calls are mixed. Let me find the widget grid placement
    # after rsi_conf_var's grid call.
    # Look for the pattern where rsi_conf Checkbutton is gridded
    marker_widget = 'ttk.Checkbutton(filter_frame, text="RSI Conf", variable=rsi_conf_var).pack(side=tk.LEFT, padx=(2, 0))'
    new_widgets = '''ttk.Checkbutton(filter_frame, text="RSI Conf", variable=rsi_conf_var).pack(side=tk.LEFT, padx=(2, 0))

        row += 1
        adv_frame = tk.Frame(content)
        adv_frame.grid(row=row, column=1, sticky="w", padx=8)
        ttk.Checkbutton(adv_frame, text="IV Filter", variable=iv_filter_enabled_var).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(adv_frame, text="MTF Hard", variable=mtf_hard_filter_var).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Checkbutton(adv_frame, text="Theta Stop", variable=theta_stop_widen_var).pack(side=tk.LEFT, padx=(4, 4))
        tk.Entry(adv_frame, textvariable=theta_stop_max_mult_var, width=5).pack(side=tk.LEFT, padx=(2, 2))
        tk.Entry(adv_frame, textvariable=theta_stop_threshold_var, width=5).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Checkbutton(adv_frame, text="Exit Ladder", variable=exit_ladder_enabled_var).pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(adv_frame, textvariable=exit_ladder_steps_var, width=14).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Checkbutton(adv_frame, text="DH Vol Adj", variable=dh_vol_adjust_var).pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(adv_frame, textvariable=dh_vol_low_factor_var, width=4).pack(side=tk.LEFT, padx=(2, 2))
        tk.Entry(adv_frame, textvariable=dh_vol_high_factor_var, width=4).pack(side=tk.LEFT, padx=(2, 0))
'''
    if marker_widget in content:
        content = content.replace(marker_widget, new_widgets)
        changes.append("interactive widgets: added IV Filter, MTF Hard, Theta Stop, Exit Ladder, DH Vol Adj controls")
    else:
        changes.append("WARNING: rsi_conf widget marker not found")

    # --- 4. apply_preset: add aggressive preset values ---
    # Find the aggressive section - look for a pattern like mtf_enabled_var.set(False)
    marker_agg = 'mtf_enabled_var.set(False)'
    agg_new = '''mtf_enabled_var.set(False)
            iv_filter_enabled_var.set(True)
            mtf_hard_filter_var.set(True)
            theta_stop_widen_var.set(True)
            theta_stop_max_mult_var.set("2.5")
            theta_stop_threshold_var.set("-8.0")
            exit_ladder_enabled_var.set(True)
            exit_ladder_steps_var.set("0.25:0.8,0.25:1.2,0.30:1.8")
            dh_vol_adjust_var.set(True)
            dh_vol_low_factor_var.set("0.6")
            dh_vol_high_factor_var.set("1.6")
'''
    # But we need to target the aggressive branch specifically. Let me find a unique anchor in aggressive.
    marker_agg_branch = 'roi_target_var.set("0.15")'
    agg_new_branch = '''roi_target_var.set("0.15")
            iv_filter_enabled_var.set(True)
            mtf_hard_filter_var.set(True)
            theta_stop_widen_var.set(True)
            theta_stop_max_mult_var.set("2.5")
            theta_stop_threshold_var.set("-8.0")
            exit_ladder_enabled_var.set(True)
            exit_ladder_steps_var.set("0.25:0.8,0.25:1.2,0.30:1.8")
            dh_vol_adjust_var.set(True)
            dh_vol_low_factor_var.set("0.6")
            dh_vol_high_factor_var.set("1.6")
'''
    if marker_agg_branch in content:
        # This appears in aggressive, but might also appear in conservative. Let me be more specific.
        # Find the aggressive block
        agg_start = content.find('if preset_name == "Aggressive":')
        if agg_start >= 0:
            # Find the roi_target_var line within the aggressive block
            agg_roi = content.find('roi_target_var.set("0.15")', agg_start)
            if agg_roi >= 0 and agg_roi < agg_start + 2000:
                content = content.replace(marker_agg_branch, agg_new_branch, 1)
                changes.append("apply_preset aggressive: added IV Filter, MTF Hard, Theta Stop, Exit Ladder, DH Vol Adj")
            else:
                changes.append("WARNING: aggressive roi_target_var not found in aggressive block")
        else:
            changes.append("WARNING: aggressive block not found")
    else:
        changes.append("WARNING: roi_target_var marker not found")

    # --- 5. apply_preset: add conservative preset values ---
    marker_cons = 'roi_target_var.set("0.12")'
    cons_new = '''roi_target_var.set("0.12")
            iv_filter_enabled_var.set(True)
            mtf_hard_filter_var.set(False)
            theta_stop_widen_var.set(True)
            theta_stop_max_mult_var.set("1.8")
            theta_stop_threshold_var.set("-4.0")
            exit_ladder_enabled_var.set(True)
            exit_ladder_steps_var.set("0.20:1.0,0.20:1.5,0.20:2.0")
            dh_vol_adjust_var.set(True)
            dh_vol_low_factor_var.set("0.8")
            dh_vol_high_factor_var.set("1.3")
'''
    cons_start = content.find('if preset_name == "Conservative":')
    if cons_start >= 0:
        cons_roi = content.find('roi_target_var.set("0.12")', cons_start)
        if cons_roi >= 0 and cons_roi < cons_start + 2000:
            content = content.replace(marker_cons, cons_new, 1)
            changes.append("apply_preset conservative: added IV Filter, MTF Hard, Theta Stop, Exit Ladder, DH Vol Adj")
        else:
            changes.append("WARNING: conservative roi_target_var not found in conservative block")
    else:
        changes.append("WARNING: conservative block not found")

    # --- 6. apply_changes: add env var persistence for new features ---
    # Find a good insertion point - after the MTF/ROC/CHOP env saving section
    # Look for MSTOCK_ENABLE_MTF_CONFIRMATION
    marker_env = 'os.environ["MSTOCK_ENABLE_MTF_CONFIRMATION"] = "true" if mtf_enabled_var.get() else "false"'
    env_new = '''os.environ["MSTOCK_ENABLE_MTF_CONFIRMATION"] = "true" if mtf_enabled_var.get() else "false"
            to_persist["MSTOCK_ENABLE_MTF_CONFIRMATION"] = os.environ["MSTOCK_ENABLE_MTF_CONFIRMATION"]
            os.environ["MSTOCK_IV_FILTER_ENABLED"] = "true" if iv_filter_enabled_var.get() else "false"
            to_persist["MSTOCK_IV_FILTER_ENABLED"] = os.environ["MSTOCK_IV_FILTER_ENABLED"]
            os.environ["MSTOCK_MTF_HARD_FILTER"] = "true" if mtf_hard_filter_var.get() else "false"
            to_persist["MSTOCK_MTF_HARD_FILTER"] = os.environ["MSTOCK_MTF_HARD_FILTER"]
            os.environ["MSTOCK_THETA_STOP_WIDEN_ENABLED"] = "true" if theta_stop_widen_var.get() else "false"
            to_persist["MSTOCK_THETA_STOP_WIDEN_ENABLED"] = os.environ["MSTOCK_THETA_STOP_WIDEN_ENABLED"]
            if theta_stop_max_mult_var.get().strip():
                try:
                    v = str(float(theta_stop_max_mult_var.get().strip()))
                    os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v
                    to_persist["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v
                except Exception:
                    pass
            if theta_stop_threshold_var.get().strip():
                try:
                    v = str(float(theta_stop_threshold_var.get().strip()))
                    os.environ["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v
                    to_persist["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v
                except Exception:
                    pass
            os.environ["MSTOCK_EXIT_LADDER_ENABLED"] = "true" if exit_ladder_enabled_var.get() else "false"
            to_persist["MSTOCK_EXIT_LADDER_ENABLED"] = os.environ["MSTOCK_EXIT_LADDER_ENABLED"]
            if exit_ladder_steps_var.get().strip():
                os.environ["MSTOCK_EXIT_LADDER_STEPS"] = exit_ladder_steps_var.get().strip()
                to_persist["MSTOCK_EXIT_LADDER_STEPS"] = os.environ["MSTOCK_EXIT_LADDER_STEPS"]
            os.environ["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"] = "true" if dh_vol_adjust_var.get() else "false"
            to_persist["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"] = os.environ["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"]
            if dh_vol_low_factor_var.get().strip():
                try:
                    v = str(float(dh_vol_low_factor_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"] = v
                except Exception:
                    pass
            if dh_vol_high_factor_var.get().strip():
                try:
                    v = str(float(dh_vol_high_factor_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"] = v
                except Exception:
                    pass
'''
    # We need to find the MTF confirmation line that has the to_persist pattern already
    marker_mtf_env = marker_env + '\n            to_persist["MSTOCK_ENABLE_MTF_CONFIRMATION"]'
    if marker_env in content:
        # Find the line with to_persist for MTF
        mtf_env_idx = content.find(marker_env)
        if mtf_env_idx >= 0:
            # Find the end of this env saving block (look for next os.environ or the block end)
            next_section = content.find('\n            os.environ["', mtf_env_idx + len(marker_env) + 50)
            if next_section < 0:
                next_section = mtf_env_idx + 500  # fallback
            before = content[mtf_env_idx:next_section]
            # Check if to_persist is already there
            if 'to_persist["MSTOCK_ENABLE_MTF_CONFIRMATION"]' in before:
                # Replace the MTF confirmation block with one that includes the new features
                old_block = content[mtf_env_idx:next_section]
                new_block = env_new
                content = content.replace(old_block, new_block, 1)
                changes.append("apply_changes: added env var persistence for all 5 new features")
            else:
                # Try finding just the os.environ line
                old_line = content[mtf_env_idx:content.find('\n', mtf_env_idx)]
                content = content.replace(old_line, env_new, 1)
                changes.append("apply_changes: added env var persistence for all 5 new features")
    else:
        changes.append("WARNING: MTF confirmation env marker not found")

    # --- 7. _set_env_from_fields: add new feature env reads ---
    # Find a spot after the existing preset/backtest env vars
    marker_setenv = 'os.environ["MSTOCK_PRESET"] = str(preset_var.get() or "").strip().lower()'
    setenv_new = '''os.environ["MSTOCK_PRESET"] = str(preset_var.get() or "").strip().lower()
        try:
            os.environ["MSTOCK_IV_FILTER_ENABLED"] = "true" if iv_filter_enabled_var.get() else "false"
        except Exception:
            pass
        try:
            os.environ["MSTOCK_MTF_HARD_FILTER"] = "true" if mtf_hard_filter_var.get() else "false"
        except Exception:
            pass
        try:
            os.environ["MSTOCK_THETA_STOP_WIDEN_ENABLED"] = "true" if theta_stop_widen_var.get() else "false"
        except Exception:
            pass
        try:
            os.environ["MSTOCK_EXIT_LADDER_ENABLED"] = "true" if exit_ladder_enabled_var.get() else "false"
        except Exception:
            pass
        try:
            os.environ["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"] = "true" if dh_vol_adjust_var.get() else "false"
        except Exception:
            pass
'''
    if marker_setenv in content:
        content = content.replace(marker_setenv, setenv_new, 1)
        changes.append("_set_env_from_fields: added new feature env reads")
    else:
        changes.append("WARNING: _set_env_from_fields marker not found")

    # Write if changed
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        changes.append(f"Written {path}")
    else:
        changes.append(f"No changes to {path}")

    return changes


def patch_config_py(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        original = content

    changes = []

    # --- Add aggressive preset values ---
    # Find the aggressive preset section where we add new values
    # Look for a unique aggressive-only marker
    marker_agg_risk = 'cfg.risk_scale_atr_high_factor = 0.90  # aggressive'
    agg_new = '''cfg.risk_scale_atr_high_factor = 0.90  # aggressive
            if not _env_present("MSTOCK_IV_FILTER_ENABLED"):
                cfg.iv_filter_enabled = True
            if not _env_present("MSTOCK_MTF_HARD_FILTER"):
                cfg.mtf_hard_filter = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_ENABLED"):
                cfg.theta_stop_widen_enabled = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_MAX_MULT"):
                cfg.theta_stop_widen_max_mult = 2.5
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_THRESHOLD"):
                cfg.theta_stop_widen_threshold = -8.0
            if not _env_present("MSTOCK_EXIT_LADDER_ENABLED"):
                cfg.exit_ladder_enabled = True
            if not _env_present("MSTOCK_EXIT_LADDER_STEPS"):
                cfg.exit_ladder_steps = "0.25:0.8,0.25:1.2,0.30:1.8"
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"):
                cfg.delta_hedge_vol_adjust_enabled = True
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"):
                cfg.delta_hedge_vol_low_tol_factor = 0.6
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"):
                cfg.delta_hedge_vol_high_tol_factor = 1.6
'''
    if marker_agg_risk in content:
        content = content.replace(marker_agg_risk, agg_new, 1)
        changes.append("aggressive preset: added 10 new config fields")
    else:
        changes.append("WARNING: aggressive risk_scale marker not found - trying alternative anchor")
        # Try alternative anchor
        marker_agg2 = 'cfg.strategy_router_mode = "aggressive"'
        agg_new2 = '''cfg.strategy_router_mode = "aggressive"
            if not _env_present("MSTOCK_IV_FILTER_ENABLED"):
                cfg.iv_filter_enabled = True
            if not _env_present("MSTOCK_MTF_HARD_FILTER"):
                cfg.mtf_hard_filter = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_ENABLED"):
                cfg.theta_stop_widen_enabled = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_MAX_MULT"):
                cfg.theta_stop_widen_max_mult = 2.5
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_THRESHOLD"):
                cfg.theta_stop_widen_threshold = -8.0
            if not _env_present("MSTOCK_EXIT_LADDER_ENABLED"):
                cfg.exit_ladder_enabled = True
            if not _env_present("MSTOCK_EXIT_LADDER_STEPS"):
                cfg.exit_ladder_steps = "0.25:0.8,0.25:1.2,0.30:1.8"
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"):
                cfg.delta_hedge_vol_adjust_enabled = True
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"):
                cfg.delta_hedge_vol_low_tol_factor = 0.6
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"):
                cfg.delta_hedge_vol_high_tol_factor = 1.6
'''
        if marker_agg2 in content:
            content = content.replace(marker_agg2, agg_new2, 1)
            changes.append("aggressive preset: added 10 new config fields (alt anchor)")
        else:
            changes.append("WARNING: aggressive preset section not found")

    # --- Add conservative preset values ---
    # Find the conservative section
    marker_cons_risk = 'cfg.enable_dynamic_pyramiding = False'
    cons_new = '''cfg.enable_dynamic_pyramiding = False
            if not _env_present("MSTOCK_IV_FILTER_ENABLED"):
                cfg.iv_filter_enabled = True
            if not _env_present("MSTOCK_MTF_HARD_FILTER"):
                cfg.mtf_hard_filter = False
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_ENABLED"):
                cfg.theta_stop_widen_enabled = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_MAX_MULT"):
                cfg.theta_stop_widen_max_mult = 1.8
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_THRESHOLD"):
                cfg.theta_stop_widen_threshold = -4.0
            if not _env_present("MSTOCK_EXIT_LADDER_ENABLED"):
                cfg.exit_ladder_enabled = True
            if not _env_present("MSTOCK_EXIT_LADDER_STEPS"):
                cfg.exit_ladder_steps = "0.20:1.0,0.20:1.5,0.20:2.0"
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"):
                cfg.delta_hedge_vol_adjust_enabled = True
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"):
                cfg.delta_hedge_vol_low_tol_factor = 0.8
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"):
                cfg.delta_hedge_vol_high_tol_factor = 1.3
'''
    if marker_cons_risk in content:
        content = content.replace(marker_cons_risk, cons_new, 1)
        changes.append("conservative preset: added 10 new config fields")
    else:
        changes.append("WARNING: conservative enable_dynamic_pyramiding marker not found - trying alternative")
        marker_cons2 = 'cfg.strategy_router_mode = "balanced"'
        cons_new2 = '''cfg.strategy_router_mode = "balanced"
            if not _env_present("MSTOCK_IV_FILTER_ENABLED"):
                cfg.iv_filter_enabled = True
            if not _env_present("MSTOCK_MTF_HARD_FILTER"):
                cfg.mtf_hard_filter = False
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_ENABLED"):
                cfg.theta_stop_widen_enabled = True
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_MAX_MULT"):
                cfg.theta_stop_widen_max_mult = 1.8
            if not _env_present("MSTOCK_THETA_STOP_WIDEN_THRESHOLD"):
                cfg.theta_stop_widen_threshold = -4.0
            if not _env_present("MSTOCK_EXIT_LADDER_ENABLED"):
                cfg.exit_ladder_enabled = True
            if not _env_present("MSTOCK_EXIT_LADDER_STEPS"):
                cfg.exit_ladder_steps = "0.20:1.0,0.20:1.5,0.20:2.0"
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"):
                cfg.delta_hedge_vol_adjust_enabled = True
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"):
                cfg.delta_hedge_vol_low_tol_factor = 0.8
            if not _env_present("MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"):
                cfg.delta_hedge_vol_high_tol_factor = 1.3
'''
        if marker_cons2 in content:
            content = content.replace(marker_cons2, cons_new2, 1)
            changes.append("conservative preset: added 10 new config fields (alt anchor)")
        else:
            changes.append("WARNING: conservative preset section not found")

    # Write if changed
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        changes.append(f"Written {path}")
    else:
        changes.append(f"No changes to {path}")

    return changes


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"
    config_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\config.py"

    print("=== Patching ui.py ===")
    ui_changes = patch_ui_py(ui_path)
    for c in ui_changes:
        print(f"  {c}")

    print()
    print("=== Patching config.py ===")
    cfg_changes = patch_config_py(config_path)
    for c in cfg_changes:
        print(f"  {c}")

    print()
    print("Done.")
