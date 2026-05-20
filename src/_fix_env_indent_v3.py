"""Fix indentation in _set_env_from_fields method of ui.py.

The current indentation has try/except at the same level as if, and
os.environ/to_persist lines at the wrong level.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('src/ui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The anchor is the exact line we want to start replacing from.
# Find the first problematic line.
old_marker = 'os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v\n                to_persist["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v\n                except Exception:\n                    pass\n                if theta_stop_threshold_var.get().strip():\n                try:\n                    v = str(float(theta_stop_threshold_var.get().strip()))\n                os.environ["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v\n                to_persist["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"] = v\n                except Exception:\n                    pass\n                os.environ["MSTOCK_EXIT_LADDER_ENABLED"]'

pos = content.find(old_marker)
if pos < 0:
    # Try without the "pass\n" part - maybe newlines differ
    old_marker2 = 'os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v\n                to_persist["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v\n                except Exception:\n                    pass'
    pos = content.find(old_marker2)

if pos < 0:
    print("Could not find anchor 1")
    sys.exit(1)

# Walk forward to find the ending anchor
end_anchor = '\n\n                os.environ["MSTOCK_ACCESS_TOKEN"] = token'
end_pos = content.find(end_anchor, pos)
if end_pos < 0:
    print("Could not find ending anchor")
    sys.exit(1)

old_section = content[pos:end_pos]
print(f"Found section at pos {pos} to {end_pos}")
print(f"Old section repr:\n{repr(old_section)}\n")

# Build the corrected section
new_section_lines = []
for line in old_section.split('\n'):
    original = line
    stripped = line.lstrip()
    leading = len(line) - len(stripped)
    
    # Fix try/except blocks where they're at the wrong level
    if stripped == 'try:' and leading == 16:
        new_line = '                    try:'
        new_section_lines.append(new_line)
    elif stripped == 'except Exception:' and leading == 16:
        new_line = '                    except Exception:'
        new_section_lines.append(new_line)
    elif stripped.startswith('os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('to_persist["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('os.environ["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('to_persist["MSTOCK_THETA_STOP_WIDEN_THRESHOLD"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('v = str(float(theta_stop_max_mult') and leading == 20:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('v = str(float(theta_stop_threshold') and leading == 20:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped == 'try:' and leading == 16:
        new_line = '                    try:'
        new_section_lines.append(new_line)
    elif stripped == 'except Exception:' and leading == 16:
        new_line = '                    except Exception:'
        new_section_lines.append(new_line)
    elif stripped.startswith('os.environ["MSTOCK_EXIT_LADDER_STEPS"') and leading == 16:
        new_line = '                    ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('to_persist["MSTOCK_EXIT_LADDER_STEPS"') and leading == 16:
        new_line = '                    ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('os.environ["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('to_persist["MSTOCK_DELTA_HEDGE_VOL_LOW_TOL_FACTOR"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('os.environ["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('to_persist["MSTOCK_DELTA_HEDGE_VOL_HIGH_TOL_FACTOR"') and leading == 16:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('v = str(float(dh_vol_low_factor') and leading == 20:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    elif stripped.startswith('v = str(float(dh_vol_high_factor') and leading == 20:
        new_line = '                        ' + stripped
        new_section_lines.append(new_line)
    else:
        new_section_lines.append(line)

new_section = '\n'.join(new_section_lines)
print(f"New section repr:\n{repr(new_section)}\n")

# Replace
new_content = content[:pos] + new_section + content[end_pos:]
with open('src/ui.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Replacement done. Verifying...")
# Verify
with open('src/ui.py', 'r', encoding='utf-8') as f:
    verify = f.read()
check_pos = verify.find('os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"]')
if check_pos >= 0:
    snippet = verify[check_pos:check_pos+200]
    print(f"New section at find pos:\n{repr(snippet)}")

print("\nDone.")
