#!/usr/bin/env python3
"""Static audit for Tk after()/after_cancel() usage patterns."""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TARGETS = [
    SRC_DIR / "ui.py",
    SRC_DIR / "ui_live_chart_panels.py",
]

SCHEDULER_NAMES = {
    "_safe_after",
    "_safe_after_app",
    "_safe_after_bot",
    "_gui_schedule_tab_after",
}


def _attr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _attr_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):
        return _attr_name(node.value)
    return ""


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class AfterAudit(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.function_stack: list[str] = []
        self.raw_after_calls: list[tuple[int, str, str]] = []
        self.after_cancel_calls: list[tuple[int, str, str]] = []
        self.scheduler_name_sites: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.after_assignment_sites: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.direct_thread_targets: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _attr_name(node.func)
        current_fn = self.function_stack[-1] if self.function_stack else "<module>"

        if func_name.endswith(".after") or func_name == "after":
            self.raw_after_calls.append((node.lineno, current_fn, ast.get_source_segment(self.source, node) or "after(...)"))

        if func_name.endswith(".after_cancel") or func_name == "after_cancel":
            self.after_cancel_calls.append(
                (node.lineno, current_fn, ast.get_source_segment(self.source, node) or "after_cancel(...)")
            )

        base_name = func_name.rsplit(".", 1)[-1]
        if base_name in SCHEDULER_NAMES and node.args:
            label = _literal_str(node.args[0])
            if label:
                self.scheduler_name_sites[label].append((node.lineno, current_fn))

        if func_name.endswith(".Thread") or func_name == "Thread":
            for kw in node.keywords:
                if kw.arg == "target":
                    target_name = _attr_name(kw.value)
                    if target_name:
                        self.direct_thread_targets.append((node.lineno, target_name))
                    break

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        current_fn = self.function_stack[-1] if self.function_stack else "<module>"
        value = node.value
        if isinstance(value, ast.Call):
            func_name = _attr_name(value.func)
            if func_name.endswith(".after") or func_name == "after":
                for target in node.targets:
                    target_name = _attr_name(target)
                    if target_name:
                        self.after_assignment_sites[target_name].append((node.lineno, current_fn))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        current_fn = self.function_stack[-1] if self.function_stack else "<module>"
        value = node.value
        if isinstance(value, ast.Call):
            func_name = _attr_name(value.func)
            if func_name.endswith(".after") or func_name == "after":
                target_name = _attr_name(node.target)
                if target_name:
                    self.after_assignment_sites[target_name].append((node.lineno, current_fn))
        self.generic_visit(node)


def main() -> int:
    problems: list[str] = []
    warnings: list[str] = []

    print("GUI AFTER LOOP AUDIT")
    print(f"repo={REPO_ROOT}")

    for path in TARGETS:
        if not path.exists():
            problems.append(f"missing target: {path}")
            continue

        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path}: syntax error: {exc}")
            continue

        audit = AfterAudit(path, source)
        audit.visit(tree)

        wrapper_lines: set[int] = set()
        for lineno, fn_name, _snippet in audit.raw_after_calls:
            if fn_name in {"_safe_after", "_safe_after_app", "_safe_after_bot", "_gui_schedule_tab_after"}:
                wrapper_lines.add(lineno)

        raw_after_nonwrapper = [
            (lineno, fn_name, snippet)
            for lineno, fn_name, snippet in audit.raw_after_calls
            if lineno not in wrapper_lines
        ]

        repeated_scheduler_labels = {
            label: sites
            for label, sites in audit.scheduler_name_sites.items()
            if len({(lineno, fn_name) for lineno, fn_name in sites}) > 1
        }
        repeated_after_assignments = {
            name: sites
            for name, sites in audit.after_assignment_sites.items()
            if len({(lineno, fn_name) for lineno, fn_name in sites}) > 1 and re.search(r"after", name, re.I)
        }

        print(
            f"{path.relative_to(REPO_ROOT)} "
            f"raw_after_calls={len(audit.raw_after_calls)} "
            f"after_cancel_calls={len(audit.after_cancel_calls)} "
            f"scheduler_labels={len(audit.scheduler_name_sites)}"
        )

        for lineno, fn_name, snippet in raw_after_nonwrapper[:20]:
            warnings.append(
                f"{path}:{lineno}: raw after() outside wrapper in {fn_name}: {snippet}"
            )

        for label, sites in sorted(repeated_scheduler_labels.items()):
            site_summary = ", ".join(f"{fn_name}@{lineno}" for lineno, fn_name in sites[:6])
            warnings.append(
                f"{path}: scheduler label {label!r} is scheduled from multiple sites: {site_summary}"
            )

        for name, sites in sorted(repeated_after_assignments.items()):
            site_summary = ", ".join(f"{fn_name}@{lineno}" for lineno, fn_name in sites[:6])
            warnings.append(
                f"{path}: after-id storage {name!r} assigned from multiple sites: {site_summary}"
            )

        thread_target_counts = Counter(target for _lineno, target in audit.direct_thread_targets)
        noisy_targets = {target: count for target, count in thread_target_counts.items() if count > 2}
        if noisy_targets:
            warnings.append(f"{path}: repeated thread targets: {noisy_targets}")

    if warnings:
        print("warnings:")
        for item in warnings:
            print(f"  - {item}")

    if problems:
        print("AUDIT FAILED")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
