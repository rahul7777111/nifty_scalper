"""
Fix script: 
1. Replace widget variable declarations to reuse the already-declared variables
2. Fix indentation error on line 7588
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"

with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()
    original = content

# The problem: the interactive widgets section creates NEW variables like iv_filter_var
# instead of reusing already-declared iv_filter_enabled_var.
# Fix: Replace the widget variable declarations to use the existing variables.

old_widget = '''        row += 1
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
        tk.Entry(adv_frame, textvariable=dh_vol_high_var, width=4).pack(side=tk.LEFT, padx=(2, 0))'''

new_widget = '''        row += 1
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
        tk.Entry(adv_frame, textvariable=dh_vol_high_factor_var, width=4).pack(side=tk.LEFT, padx=(2, 0))'''

if old_widget in content:
    content = content.replace(old_widget, new_widget, 1)
    print("[OK] Replaced widget variable declarations to reuse existing vars")
else:
    print("[WARNING] Old widget pattern not found - checking current state")
    # Check what's there now
    pos = content.find('adv_frame = tk.Frame(content)')
    if pos >= 0:
        print(content[pos:pos+1200])

# Now fix the indentation error - find lines with 16-space indentation (non-standard)
# The lines like "                iv_filter_enabled_var = tk.BooleanVar..."
# should be 12 spaces (matching the surrounding code)
import re
# Find patterns with 16 spaces that should be 12
# These are the lines injected by the first script before delta_enabled_var
# They have indentation like "                " when they should be "            "
fixed_content = content

# Fix specific known indentation issue - the first script injected vars with wrong indentation
# before the delta_enabled_var line
old_indent_block = """        # ---- #8: Smart Partial Exit Ladder ----
                iv_filter_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, \"iv_filter_enabled\", False)))
                mtf_hard_filter_var = tk.BooleanVar(value=bool(getattr(cfg, \"mtf_hard_filter\", False)))
                theta_stop_widen_var = tk.BooleanVar(value=bool(getattr(cfg, \"theta_stop_widen_enabled\", False)))
                theta_stop_max_mult_var = tk.StringVar(value=str(getattr(cfg, \"theta_stop_widen_max_mult\", 2.0)))
                theta_stop_threshold_var = tk.StringVar(value=str(getattr(cfg, \"theta_stop_widen_threshold\", -5.0)))
                exit_ladder_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, \"exit_ladder_enabled\", False)))
                exit_ladder_steps_var = tk.StringVar(value=str(getattr(cfg, \"exit_ladder_steps\", \"0.25:1.0,0.25:1.5,0.25:2.0\")))
                dh_vol_adjust_var = tk.BooleanVar(value=bool(getattr(cfg, \"delta_hedge_vol_adjust_enabled\", False)))
                dh_vol_low_factor_var = tk.StringVar(value=str(getattr(cfg, \"delta_hedge_vol_low_tol_factor\", 0.7)))
                dh_vol_high_factor_var = tk.StringVar(value=str(getattr(cfg, \"delta_hedge_vol_high_tol_factor\", 1.5)))
        delta_enabled_var = tk.BooleanVar"""

new_indent_block = """        # ---- #8: Smart Partial Exit Ladder ----
            iv_filter_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, \"iv_filter_enabled\", False)))
            mtf_hard_filter_var = tk.BooleanVar(value=bool(getattr(cfg, \"mtf_hard_filter\", False)))
            theta_stop_widen_var = tk.BooleanVar(value=bool(getattr(cfg, \"theta_stop_widen_enabled\", False)))
            theta_stop_max_mult_var = tk.StringVar(value=str(getattr(cfg, \"theta_stop_widen_max_mult\", 2.0)))
            theta_stop_threshold_var = tk.StringVar(value=str(getattr(cfg, \"theta_stop_widen_threshold\", -5.0)))
            exit_ladder_enabled_var = tk.BooleanVar(value=bool(getattr(cfg, \"exit_ladder_enabled\", False)))
            exit_ladder_steps_var = tk.StringVar(value=str(getattr(cfg, \"exit_ladder_steps\", \"0.25:1.0,0.25:1.5,0.25:2.0\")))
            dh_vol_adjust_var = tk.BooleanVar(value=bool(getattr(cfg, \"delta_hedge_vol_adjust_enabled\", False)))
            dh_vol_low_factor_var = tk.StringVar(value=str(getattr(cfg, \"delta_hedge_vol_low_tol_factor\", 0.7)))
            dh_vol_high_factor_var = tk.StringVar(value=str(getattr(cfg, \"delta_hedge_vol_high_tol_factor\", 1.5)))
        delta_enabled_var = tk.BooleanVar"""

if old_indent_block in fixed_content:
    fixed_content = fixed_content.replace(old_indent_block, new_indent_block, 1)
    print("[OK] Fixed indentation for injected variable declarations (16->12 spaces)")
else:
    print("[WARNING] Old indentation block not found")
    # Try searching for the issue
    if 'iv_filter_enabled_var' in fixed_content:
        pos = fixed_content.find('iv_filter_enabled_var')
        snippet = fixed_content[max(0,pos-60):pos+100]
        print(f"Found at position {pos}:")
        print(repr(snippet))

if fixed_content != content:
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print("[OK] Written ui.py")
else:
    print("[INFO] No changes needed")

print()
print("Now checking for compilation...")
