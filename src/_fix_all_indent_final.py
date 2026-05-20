"""Fix ALL remaining indentation issues in the _set_env_from_fields section of ui.py.

Fixes specific lines that are at wrong indentation levels throughout
the injected env persistence block (lines 8496-8546).
"""

with open('src/ui.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Line numbers (1-indexed) and their correct content
fixes = {
    # First theta_stop_max_mult try block
    8497: '                    try:\n',
    8498: '                        v = str(float(theta_stop_max_mult_var.get().strip()))\n',
    8499: '                        os.environ["MSTOCK_THETA_STOP_WIDEN_MAX_MULT"] = v\n',
    8502: '                        pass\n',
    # First theta_stop_threshold pass
    8509: '                        pass\n',
    # dh_vol_low pass
    8523: '                        pass\n',
    # dh_vol_high pass
    8530: '                        pass\n',
    # underlying_token_var if block
    8536: '                    u_tok = self.underlying_token_var.get().strip()\n',
    8537: '                    if u_tok:\n',
    8538: '                        os.environ["MSTOCK_UNDERLYING_TOKEN"] = u_tok\n',
    8539: '                    else:\n',
    8540: '                        os.environ.pop("MSTOCK_UNDERLYING_TOKEN", None)\n',
    # underlying_exchange_var if block
    8541: '                if hasattr(self, "underlying_exchange_var"):\n',
    8542: '                    u_exch = self.underlying_exchange_var.get().strip()\n',
    8543: '                    if u_exch:\n',
    8544: '                        os.environ["MSTOCK_UNDERLYING_EXCHANGE"] = u_exch\n',
    8545: '                    else:\n',
    8546: '                        os.environ.pop("MSTOCK_UNDERLYING_EXCHANGE", None)\n',
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
    print(f"  L{line_num}: {old} -> {new}")
if not changed:
    print("No changes needed (all lines already correct)")
