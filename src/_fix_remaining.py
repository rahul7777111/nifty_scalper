"""
Fix script for remaining missing pieces in ui.py and config.py.
Adds: vars_to_track entries, interactive widgets, apply_preset sets, config.py conservative preset.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"
config_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\config.py"

changes_made = []

# ===== FIX 1: vars_to_track entries =====
with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()
    original = content

marker_vt = '''                "Exit Wing Touch": exit_wing_var,
            }'''
new_vt = '''                "Exit Wing Touch": exit_wing_var,
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
            }'''
if marker_vt in content:
    content = content.replace(marker_vt, new_vt, 1)
    changes_made.append("vars_to_track: added 10 new entries")
else:
    changes_made.append("WARNING: vars_to_track marker not found")

if content != original:
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] vars_to_track updated")

# ===== FIX 2: Interactive widgets =====
with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()
    original = content

marker_widget = '''        ttk.Checkbutton(filter_frame, text="RSI Conf", variable=rsi_conf_var).pack(side=tk.LEFT, padx=(6, 0))

        dir_be_var = tk.StringVar'''
new_widget = '''        ttk.Checkbutton(filter_frame, text="RSI Conf", variable=rsi_conf_var).pack(side=tk.LEFT, padx=(6, 0))

        row += 1
        adv_frame = tk.Frame(content)
        adv_frame.grid(row=row, column=1, sticky="w", padx=8)
        iv_filter_var = tk.BooleanVar(value=bool(getattr(cfg, "iv_filter_enabled", False)))
        mtf_hard_var = tk.BooleanVar(value=bool(getattr(cfg, "mtf_hard_filter", False)))
        theta_stop_var = tk.BooleanVar(value=bool(getattr(cfg, "theta_stop_widen_enabled", False)))
        theta_max_mult_var = tk.StringVar(value=str(getattr(cfg, "theta_stop_widen_max_mult", 2.0)))
        theta_threshold_var = tk.StringVar(value=str(getattr(cfg, "theta_stop_widen_threshold", -5.0)))
        exit_ladder_var = tk.BooleanVar(value=bool(getattr(cfg, "exit_ladder_enabled", False)))
        exit_ladder_steps_var2 = tk.StringVar(value=str(getattr(cfg, "exit_ladder_steps", "0.25:1.0,0.25:1.5,0.25:2.0")))
        dh_vol_var = tk.BooleanVar(value=bool(getattr(cfg, "delta_hedge_vol_adjust_enabled", False)))
        dh_vol_low_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_vol_low_tol_factor", 0.7)))
        dh_vol_high_var = tk.StringVar(value=str(getattr(cfg, "delta_hedge_vol_high_tol_factor", 1.5)))
        ttk.Checkbutton(adv_frame, text="IV Filter", variable=iv_filter_var).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(adv_frame, text="MTF Hard", variable=mtf_hard_var).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Checkbutton(adv_frame, text="Theta Stop", variable=theta_stop_var).pack(side=tk.LEFT, padx=(4, 4))
        tk.Entry(adv_frame, textvariable=theta_max_mult_var, width=5).pack(side=tk.LEFT, padx=(2, 2))
        tk.Entry(adv_frame, textvariable=theta_threshold_var, width=5).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Checkbutton(adv_frame, text="Exit Ladder", variable=exit_ladder_var).pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(adv_frame, textvariable=exit_ladder_steps_var2, width=14).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Checkbutton(adv_frame, text="DH Vol Adj", variable=dh_vol_var).pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(adv_frame, textvariable=dh_vol_low_var, width=4).pack(side=tk.LEFT, padx=(2, 2))
        tk.Entry(adv_frame, textvariable=dh_vol_high_var, width=4).pack(side=tk.LEFT, padx=(2, 0))

        dir_be_var = tk.StringVar'''

if marker_widget in content:
    content = content.replace(marker_widget, new_widget, 1)
    changes_made.append("interactive widgets: added IV Filter, MTF Hard, Theta Stop, Exit Ladder, DH Vol Adj")
else:
    changes_made.append("WARNING: widget marker not found (trying alternative without padx)")
    # Try alternative without padx on rsi conf
    marker_widget2 = '''        ttk.Checkbutton(filter_frame, text="RSI Conf", variable=rsi_conf_var).pack(side=tk.LEFT)

        dir_be_var = tk.StringVar'''
    if marker_widget2 in content:
        content = content.replace(marker_widget2, new_widget.replace('padx=(6, 0)', ''), 1)
        changes_made.append("interactive widgets: added via alt marker")
    else:
        changes_made.append("WARNING: widget marker not found (both attempts)")

if content != original:
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] interactive widgets updated")

# ===== FIX 3: apply_preset aggressive branch set() calls =====
with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()
    original = content

marker_agg_end = '''                exit_wing_var.set(True)

            elif name == "Conservative":'''
new_agg_end = '''                exit_wing_var.set(True)
                # New features: aggressive defaults
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

            elif name == "Conservative":'''

if marker_agg_end in content:
    content = content.replace(marker_agg_end, new_agg_end, 1)
    changes_made.append("apply_preset aggressive: added 10 new set() calls")
else:
    changes_made.append("WARNING: aggressive branch end marker not found")
    # Try without blank line
    marker_agg_end2 = '''                exit_wing_var.set(True)
            elif name == "Conservative":'''
    if marker_agg_end2 in content:
        new_agg_end2 = '''                exit_wing_var.set(True)
                # New features: aggressive defaults
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
            elif name == "Conservative":'''
        content = content.replace(marker_agg_end2, new_agg_end2, 1)
        changes_made.append("apply_preset aggressive: added 10 new set() calls (alt anchor)")

if content != original:
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)

# ===== FIX 4: apply_preset conservative branch set() calls =====
with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()
    original = content

marker_cons_end = '''                exit_wing_var.set(False)

            after = _snapshot_preset_fields()'''
new_cons_end = '''                exit_wing_var.set(False)
                # New features: conservative defaults
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

            after = _snapshot_preset_fields()'''

if marker_cons_end in content:
    content = content.replace(marker_cons_end, new_cons_end, 1)
    changes_made.append("apply_preset conservative: added 10 new set() calls")
else:
    changes_made.append("WARNING: conservative branch end marker not found")
    marker_cons_end2 = '''                exit_wing_var.set(False)
            after = _snapshot_preset_fields()'''
    if marker_cons_end2 in content:
        new_cons_end2 = '''                exit_wing_var.set(False)
                # New features: conservative defaults
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
            after = _snapshot_preset_fields()'''
        content = content.replace(marker_cons_end2, new_cons_end2, 1)
        changes_made.append("apply_preset conservative: added 10 new set() calls (alt anchor)")

if content != original:
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)

# ===== FIX 5: config.py conservative preset =====
with open(config_path, 'r', encoding='utf-8') as f:
    content = f.read()
    original = content

# Check if conservative values were already added
if 'cfg.iv_filter_enabled = True' in content and 'cfg.mtf_hard_filter = True' in content:
    pass  # already done
else:
    # Try to find the conservative section - look for enable_dynamic_pyramiding
    marker_cons_cfg = '''            cfg.enable_dynamic_pyramiding = False
            if not _env_present("MSTOCK_ENTRY_SPREAD_SHOCK_MIN_SAMPLES"):'''
    new_cons_cfg = '''            cfg.enable_dynamic_pyramiding = False
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
            if not _env_present("MSTOCK_ENTRY_SPREAD_SHOCK_MIN_SAMPLES"):'''
    if marker_cons_cfg in content:
        content = content.replace(marker_cons_cfg, new_cons_cfg, 1)
        changes_made.append("config.py conservative preset: added 10 new fields")
    else:
        changes_made.append("WARNING: config.py conservative marker not found")

if content != original:
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] config.py conservative preset updated")

print()
print("=== SUMMARY ===")
for c in changes_made:
    print(f"  {c}")
print("Done.")
