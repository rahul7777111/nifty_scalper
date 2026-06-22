#!/usr/bin/env python3
"""Static audit for Tkinter notebook/tab wiring in src/ui.py."""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
UI_PATH = REPO / "src" / "ui.py"


def _read() -> str:
    try:
        return UI_PATH.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"FAIL: cannot read {UI_PATH}: {exc}")
        sys.exit(1)


def _class_methods(tree: ast.AST, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


def _notebook_tab_labels(source: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(r"\.add\([^)]*text\s*=\s*['\"]([^'\"]+)['\"]", source, flags=re.S):
        labels.append(match.group(1))
    return labels


def _callback_refs(source: str) -> set[str]:
    refs: set[str] = set()
    patterns = (
        r"command\s*=\s*self\.(\w+)\b(?!\s*\.)",
        r"\.bind\([^)\n]*self\.(\w+)\b(?!\s*\.)",
    )
    for pattern in patterns:
        refs.update(re.findall(pattern, source, flags=re.S))
    for line in source.splitlines():
        if "_safe_after_" not in line and "_schedule_after_job" not in line:
            continue
        refs.update(
            re.findall(
                r"_(?:safe_after_app|safe_after_bot|schedule_after_job)\([^,]+,[^,]+,\s*self\.(\w+)\b",
                line,
            )
        )
    return {ref for ref in refs if ref.startswith("_")}


def _after_job_names(source: str) -> Counter[str]:
    names: list[str] = []
    for pattern in (r"_safe_after_(?:app|bot)\(\s*['\"]([^'\"]+)['\"]", r"_schedule_after_job\(\s*['\"]([^'\"]+)['\"]"):
        names.extend(re.findall(pattern, source))
    return Counter(names)


def main() -> int:
    source = _read()
    try:
        tree = ast.parse(source, filename=str(UI_PATH))
    except SyntaxError as exc:
        print(f"FAIL: syntax error in {UI_PATH}: {exc}")
        return 1

    methods = _class_methods(tree, "ScalperUI")
    if not methods:
        print("FAIL: ScalperUI class not found")
        return 1

    required = {
        "_safe_call",
        "_append_gui_log",
        "_on_tab_changed",
        "_on_sub_notebook_tab_changed",
        "_refresh_active_tab_now",
        "_gui_tab_diag",
        "_gui_widget_alive",
        "_cancel_all_after_jobs",
    }
    missing_required = sorted(required - methods)

    refs = _callback_refs(source)
    missing_refs = sorted(ref for ref in refs if ref not in methods)

    labels = _notebook_tab_labels(source)
    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    after_dupes = {name: count for name, count in _after_job_names(source).items() if count > 6}

    direct_thread_ui = []
    for line_no, line in enumerate(source.splitlines(), 1):
        if ".start()" in line and "thread" in line.lower():
            direct_thread_ui.append(line_no)

    errors: list[str] = []
    if missing_required:
        errors.append(f"missing required safety methods: {', '.join(missing_required)}")
    if missing_refs:
        errors.append(f"callback methods referenced but not defined: {', '.join(missing_refs[:30])}")
    if not labels:
        errors.append("no notebook tab labels found")

    print("GUI TAB AUDIT")
    print(f"  file={UI_PATH}")
    print(f"  methods={len(methods)}")
    print(f"  notebook_tabs={len(labels)}")
    print(f"  tab_labels={', '.join(labels)}")
    if duplicate_labels:
        print(f"  duplicate_tab_labels={', '.join(duplicate_labels)}")
    if after_dupes:
        print(f"  repeated_after_job_names={after_dupes}")
    if direct_thread_ui:
        print(f"  thread_start_lines={direct_thread_ui[:20]}")

    if errors:
        print("AUDIT FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
