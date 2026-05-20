"""Fix indentation of if/else block where else at 20 can't pair with if at 24."""

import re

with open("src/strategy.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# The issue: line 12064 (index 12063) has `if do_flip...` at 24 spaces,
# but `else:` at line 12079 (index 12078) is at 20 spaces.
# Fix: change `if` from 24 to 20 spaces, and its body from 28-32 to 24-28.

# Lines to fix: 12064-12078 (1-indexed)
START = 12063  # 0-indexed (line 12064)
END = 12078    # 0-indexed (line 12079-1)

for i in range(START, END + 1):
    line = lines[i]
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if indent >= 24:  # Indent by 24-32 spaces
        new_indent = indent - 4
        lines[i] = ' ' * new_indent + stripped
        print(f"Line {i+1}: indent {indent} -> {new_indent}: {stripped[:80]}")

with open("src/strategy.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("\nFixed! Checking for syntax errors...")
import ast
with open("src/strategy.py", "r", encoding="utf-8") as f:
    content = f.read()
try:
    ast.parse(content)
    print("No syntax errors!")
except SyntaxError as e:
    print(f"SyntaxError at line {e.lineno}: {e.msg}")
    if e.lineno:
        context = content.split('\n')[e.lineno-1]
        print(f"  {context[:150]}")
