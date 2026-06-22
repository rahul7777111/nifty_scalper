#!/usr/bin/env python3
"""Static audit for GUI refresh-loop stability patterns in ui.py and related files."""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    REPO / "src" / "ui.py",
    REPO / "src" / "ui_live_chart_panels.py",
    REPO / "src" / "_ui_chart_panels.py",
]

WARNING = 0


def warn(msg: str) -> None:
    global WARNING
    WARNING += 1
    print(f"WARNING: {msg}")


def _line_of(source: str, node: ast.AST) -> int:
    return getattr(node, "lineno", 0) or 0


def _func_name(node: ast.AST) -> str:
    while node is not None:
        if isinstance(node, ast.FunctionDef):
            return node.name
        node = getattr(node, "parent", None)
    return "<module>"


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def audit_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        warn(f"{path}:{exc.lineno}: syntax error {exc.msg}")
        return
    _attach_parents(tree)

    after_calls: list[tuple[int, str, str]] = []
    after_cancel_names: set[str] = set()
    recurring_funcs: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            src = ast.get_source_segment(text, node) or ""
            if "self.after(" in src or "self._safe_after" in src or "self._gui_schedule_tab_after" in src:
                recurring_funcs.add(node.name)
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            attr = fn.attr
            if attr in {"after", "_safe_after", "_safe_after_app", "_safe_after_bot", "_gui_schedule_tab_after"}:
                name = ""
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    name = str(node.args[0].value)
                after_calls.append((_line_of(source=text, node=node), _func_name(node), name or attr))
            if attr == "after_cancel":
                after_cancel_names.add(_func_name(node))

    # Raw-text heuristics for destructive tree refresh inside recurring updaters
    delete_re = re.compile(r"\.delete\s*\(")
    get_children_re = re.compile(r"get_children\s*\(")
    insert_re = re.compile(r"\.insert\s*\(")
    widget_touch_re = re.compile(
        r"(tree|pf_tree|trade_tree|greeks_tree|journal_tree|option_legs_tree|managed_positions_tree|cp_tree|p[1-7]_tree)\.(item|insert|delete|set|heading|column)\("
    )
    thread_current_re = re.compile(r"threading\.current_thread\(\)")

    lines = text.splitlines()
    for func in sorted(recurring_funcs):
        # locate function block roughly
        m = re.search(rf"^\s*def\s+{re.escape(func)}\s*\(", text, re.M)
        if not m:
            continue
        start = text[: m.start()].count("\n")
        # scan until next top-level def at same indent (4 spaces)
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if lines[i].startswith("    def ") and not lines[i].startswith("        "):
                end = i
                break
        block = "\n".join(lines[start:end])
        if delete_re.search(block) and get_children_re.search(block) and insert_re.search(block):
            if any(k in func for k in ("refresh", "render", "pump", "update", "schedule")):
                warn(f"{path}:{start + 1}: function `{func}` may delete+reinsert tree rows on each refresh")
        if thread_current_re.search(block) and widget_touch_re.search(block):
            warn(f"{path}:{start + 1}: function `{func}` references threading and direct widget updates")

    safe_after_helpers = {"_safe_after_app", "_safe_after_bot", "_safe_after", "_gui_schedule_tab_after"}
    # after() calls in recurring functions without dedup helper
    for lineno, func, name in after_calls:
        if func in recurring_funcs and "cancel" not in func and "stop" not in func and "close" not in func:
            line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            if any(h in line for h in safe_after_helpers):
                continue
            if name in {"", "after"}:
                warn(f"{path}:{lineno}: `{func}` schedules raw `.after` without stored handle")
            elif name and "_gui_schedule_tab_after" not in line and "_safe_after" not in line:
                warn(f"{path}:{lineno}: `{func}` schedules after job `{name}` without dedup helper")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", dest="files", default=[], help="Additional file to audit")
    args = parser.parse_args(argv)
    files = [Path(p) for p in args.files] if args.files else []
    targets = files or DEFAULT_FILES
    print("GUI refresh loop static audit")
    print("=" * 60)
    for path in targets:
        if not path.exists():
            warn(f"{path}: file not found")
            continue
        print(f"\n-- {path.relative_to(REPO)} --")
        audit_file(path)
    print("\n" + "=" * 60)
    print(f"Warnings: {WARNING}")
    return 1 if WARNING else 0


if __name__ == "__main__":
    sys.exit(main())