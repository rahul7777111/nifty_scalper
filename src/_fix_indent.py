"""
Fix indentation on lines 7588-7597: change 16 spaces to 8 spaces.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"

with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines 7588-7597 (0-indexed: 7587-7596) have 16-space indentation, fix to 8 spaces
for i in range(7587, 7597):
    if i < len(lines):
        original = lines[i]
        # Count leading spaces
        stripped = original.lstrip()
        leading = len(original) - len(stripped)
        if leading == 16:
            lines[i] = '        ' + stripped  # 8 spaces
            print(f"  Fixed line {i+1}: {repr(original)} -> {repr(lines[i])}")

with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\nDone fixing indentation.")
