"""
Fix nested indentation: try blocks inside if statements need 20 spaces, not 16.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"

with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()

# The problematic section looks like this pattern between theta stop and exit ladder
# We need to fix the try: lines that are at same indent as their if

# Pattern 1: theta_stop_max_mult_var
old1 = '''                if theta_stop_max_mult_var.get().strip():
                try:
                    v = str(float(theta_stop_max_mult_var.get().strip()))
                    os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v
                    to_persist["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v
                except Exception:
                    pass'''
new1 = '''                if theta_stop_max_mult_var.get().strip():
                    try:
                        v = str(float(theta_stop_max_mult_var.get().strip()))
                        os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v
                        to_persist["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v
                    except Exception:
                        pass'''

# Pattern 2: theta_stop_threshold_var
old2 = '''                if theta_stop_threshold_var.get().strip():
                try:
                    v = str(float(theta_stop_threshold_var.get().strip()))
                    os.environ["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v
                    to_persist["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v
                except Exception:
                    pass'''
new2 = '''                if theta_stop_threshold_var.get().strip():
                    try:
                        v = str(float(theta_stop_threshold_var.get().strip()))
                        os.environ["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v
                        to_persist["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v
                    except Exception:
                        pass'''

# Pattern 3: dh_vol_low_factor_var (similar pattern)
old3 = '''                if dh_vol_low_factor_var.get().strip():
                try:
                    v = str(float(dh_vol_low_factor_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"] = v
                except Exception:
                    pass'''
new3 = '''                if dh_vol_low_factor_var.get().strip():
                    try:
                        v = str(float(dh_vol_low_factor_var.get().strip()))
                        os.environ["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"] = v
                        to_persist["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"] = v
                    except Exception:
                        pass'''

# Pattern 4: dh_vol_high_factor_var
old4 = '''                if dh_vol_high_factor_var.get().strip():
                try:
                    v = str(float(dh_vol_high_factor_var.get().strip()))
                    os.environ["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"] = v
                    to_persist["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"] = v
                except Exception:
                    pass'''
new4 = '''                if dh_vol_high_factor_var.get().strip():
                    try:
                        v = str(float(dh_vol_high_factor_var.get().strip()))
                        os.environ["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"] = v
                        to_persist["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"] = v
                    except Exception:
                        pass'''

pairs = [(old1, new1, "theta_stop_max_mult"), (old2, new2, "theta_stop_threshold"), 
         (old3, new3, "dh_vol_low"), (old4, new4, "dh_vol_high")]

for old, new, label in pairs:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"[OK] Fixed {label}")
    else:
        print(f"[WARNING] {label} pattern not found")

with open(ui_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone.")
