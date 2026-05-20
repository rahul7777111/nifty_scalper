"""
Fix ALL remaining indentation issues in the env persistence block.
This script reads the file as lines, analyzes indentation, and fixes
inconsistencies by ensuring the injected env persistence block has
consistent 16-space indentation (matching the surrounding code).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"

with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the env persistence block boundaries
# Look for MSTOCK_ENABLE_MTF_CONFIRMATION line (injected by first script)
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'MSTOCK_ENABLE_MTF_CONFIRMATION' in line and 'to_persist' not in line:
        if start_idx is None:
            start_idx = i
    if start_idx is not None and i > start_idx + 40:
        # Look for the next section after our injected block
        if 'try:' in line and i > start_idx + 20:
            end_idx = i
            break
        if i > start_idx + 80:
            end_idx = i
            break

if end_idx is None:
    end_idx = min(start_idx + 100, len(lines)) if start_idx else len(lines)

print(f"Block found: lines {start_idx+1} to {end_idx}")

# Now fix all lines in this block to have consistent 16-space indentation
# Lines inside if/try blocks need 20 or 24 spaces
fixed_count = 0
for i in range(start_idx, min(end_idx, len(lines))):
    original = lines[i]
    stripped = original.lstrip()
    leading = len(original) - len(stripped)
    
    # Determine expected indentation based on content
    if stripped.startswith('if ') or stripped.startswith('try:') or stripped.startswith('except ') or stripped.startswith('elif '):
        expected = 16
    elif stripped.startswith('os.environ[') or stripped.startswith('to_persist['):
        expected = 16
    elif stripped.startswith('v = ') or stripped.startswith('pass'):
        expected = 20
    elif stripped.startswith('#') or stripped.startswith('"') or stripped.startswith("'"):
        expected = 16
    elif stripped.strip() == '':
        expected = 0  # don't fix blank lines
    else:
        expected = 16
    
    if expected > 0 and leading != expected and stripped.strip():
        lines[i] = ' ' * expected + stripped
        print(f"  Fixed line {i+1}: spaces {leading}->{expected}: {stripped.rstrip()[:60]}")
        fixed_count += 1

if fixed_count > 0:
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nFixed {fixed_count} lines total")
else:
    print("No fixes needed")
