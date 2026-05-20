"""Fix remaining indentation issues in ui.py lines 8496-8530.

Only fixes specific lines that are at wrong indentation levels.
"""

with open('src/ui.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Line numbers (0-indexed from 1)
fixes = {
    8496: '                if theta_stop_max_mult_var.get().strip():\n',
    8497: '                    try:\n',
    8498: '                        v = str(float(theta_stop_max_mult_var.get().strip()))\n',
    8499: '                        os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v\n',
    8502: '                        pass\n',
    8509: '                        pass\n',
    8523: '                        pass\n',
    8530: '                        pass\n',
}

changed = []
for line_num, fixed_line in fixes.items():
    idx = line_num - 1  # 0-indexed
    if idx < len(lines):
        if lines[idx] != fixed_line:
            old_repr = repr(lines[idx])
            lines[idx] = fixed_line
            changed.append((line_num, old_repr, repr(fixed_line)))

with open('src/ui.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed {len(changed)} lines:")
for line_num, old, new in changed:
    print(f"  Line {line_num}: {old} -> {new}")
if not changed:
    print("No changes needed (all lines already correct)")
