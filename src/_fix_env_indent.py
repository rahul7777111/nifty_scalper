"""
Fix indentation mismatch: lines 8489-8510 have 12 spaces but should have 16 spaces.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

ui_path = r"C:\Users\rahul\Downloads\NiftyScalper\src\ui.py"

with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = 0
for i in range(8488, min(len(lines), 8510)):  # 0-indexed: line 8489 is index 8488
    if i < len(lines):
        original = lines[i]
        stripped = original.lstrip()
        leading = len(original) - len(stripped)
        if leading == 12:  # wrong indentation
            lines[i] = '                ' + stripped  # 16 spaces
            print(f"  Fixed line {i+1}: {repr(original)} -> {repr(lines[i])}")
            fixed += 1

if fixed > 0:
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nFixed {fixed} lines. Now checking compilation...")
else:
    print("No lines needed fixing.")
