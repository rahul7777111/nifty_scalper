"""
Fix remaining indentation issues in the env persistence block.
Lines after the correctly-indented theta_stop_threshold block 
have 12-space indentation but should be at 16 spaces.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"

with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the problematic boundary: after correctly-indented theta_stop_threshold block
# the to_persist["MSTOCK_EXIT_LADDER_ENABLED"] line has 12-space indent instead of 16
old_boundary = '''                os.environ["MSTOCK_EXIT_LADDER_ENABLED"] = "true" if exit_ladder_enabled_var.get() else "false"
            to_persist["MSTOCK_EXIT_LADDER_ENABLED"] = os.environ["MSTOCK_EXIT_LADDER_ENABLED"]'''

new_boundary = '''                os.environ["MSTOCK_EXIT_LADDER_ENABLED"] = "true" if exit_ladder_enabled_var.get() else "false"
                to_persist["MSTOCK_EXIT_LADDER_ENABLED"] = os.environ["MSTOCK_EXIT_LADDER_ENABLED"]'''

if old_boundary in content:
    content = content.replace(old_boundary, new_boundary, 1)
    print("[OK] Fixed exit ladder indent (12->16 spaces)")
else:
    print("[WARNING] Exit ladder boundary not found")

# Fix the exit ladder steps block
old_steps = '''                if exit_ladder_steps_var.get().strip():
                os.environ["MSTOCK_EXIT_LADDER_STEPS"] = exit_ladder_steps_var.get().strip()
                to_persist["MSTOCK_EXIT_LADDER_STEPS"] = os.environ["MSTOCK_EXIT_LADDER_STEPS"]'''

new_steps = '''                if exit_ladder_steps_var.get().strip():
                    os.environ["MSTOCK_EXIT_LADDER_STEPS"] = exit_ladder_steps_var.get().strip()
                    to_persist["MSTOCK_EXIT_LADDER_STEPS"] = os.environ["MSTOCK_EXIT_LADDER_STEPS"]'''

if old_steps in content:
    content = content.replace(old_steps, new_steps, 1)
    print("[OK] Fixed exit ladder steps indent (12->16/20 spaces)")
else:
    print("[WARNING] Exit ladder steps pattern not found")

# Fix DH vol adj lines
old_dh1 = '''            os.environ["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"] = "true" if dh_vol_adjust_var.get() else "false"
            to_persist["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"] = os.environ["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"]'''

new_dh1 = '''                os.environ["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"] = "true" if dh_vol_adjust_var.get() else "false"
                to_persist["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"] = os.environ["MSTOCK_DELTA_HEDGE_VOL_ADJUST_ENABLED"]'''

if old_dh1 in content:
    content = content.replace(old_dh1, new_dh1, 1)
    print("[OK] Fixed DH vol adjust indent (12->16 spaces)")
else:
    print("[WARNING] DH vol adjust pattern not found")

# Fix DH vol low factor
old_dh2 = '''            if dh_vol_low_factor_var.get().strip():
                try:'''
new_dh2 = '''                if dh_vol_low_factor_var.get().strip():
                    try:'''

if old_dh2 in content:
    content = content.replace(old_dh2, new_dh2, 1)
    print("[OK] Fixed DH vol low factor indent")
else:
    print("[WARNING] DH vol low factor pattern not found")

# Fix DH vol high factor
old_dh3 = '''            if dh_vol_high_factor_var.get().strip():
                try:'''
new_dh3 = '''                if dh_vol_high_factor_var.get().strip():
                    try:'''

if old_dh3 in content:
    content = content.replace(old_dh3, new_dh3, 1)
    print("[OK] Fixed DH vol high factor indent")
else:
    print("[WARNING] DH vol high factor pattern not found")

with open(ui_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone.")
