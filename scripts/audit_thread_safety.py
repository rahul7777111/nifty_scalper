#!/usr/bin/env python3
"""Heuristic audit for Tkinter updates performed from non-UI worker contexts."""
from __future__ import annotations

import ast
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    REPO_ROOT / "src" / "ui.py",
    REPO_ROOT / "src" / "ui_live_chart_panels.py",
]

TK_MUTATOR_ATTRS = {
    "config",
    "configure",
    "delete",
    "destroy",
    "focus_force",
    "geometry",
    "iconify",
    "insert",
    "item",
    "itemconfigure",
    "selection_set",
    "set",
    "state",
    "tab",
    "title",
    "update",
    "update_idletasks",
    "withdraw",
    "yview_moveto",
}
SAFE_DISPATCH_NAMES = {
    "after",
    "_safe_after",
    "_safe_after_app",
    "_safe_after_bot",
    "_safe_ui_call",
    "_dispatch_ui",
    "_schedule_ui",
    "ui_call",
}
SAFE_QUEUE_NAMES = {"put", "put_nowait"}


def _attr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _attr_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _is_self_thread_call(node: ast.Call) -> bool:
    func_name = _attr_name(node.func)
    return func_name.endswith(".Thread") or func_name == "Thread"


class ThreadSafetyAudit(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.function_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.function_nodes_by_key: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.class_stack: list[str] = []
        self.func_stack: list[str] = []
        self.thread_targets: list[tuple[int, str, str]] = []
        self.inline_thread_warnings: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        key = ".".join(self.class_stack + [node.name]) if self.class_stack else node.name
        self.function_defs[key] = node
        self.function_nodes_by_key[key] = node
        self.func_stack.append(key)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if _is_self_thread_call(node):
            current_fn = self.func_stack[-1] if self.func_stack else "<module>"
            target_name = ""
            for kw in node.keywords:
                if kw.arg == "target":
                    target_name = _attr_name(kw.value)
                    break
            if target_name:
                self.thread_targets.append((node.lineno, current_fn, target_name))
            elif node.args:
                self.inline_thread_warnings.append(
                    f"line {node.lineno}: Thread target is positional; unable to resolve statically"
                )
        self.generic_visit(node)


class SuspiciousUiCallFinder(ast.NodeVisitor):
    def __init__(self, source: str, allowed_nested_callbacks: set[str] | None = None) -> None:
        self.source = source
        self.allowed_nested_callbacks = allowed_nested_callbacks or set()
        self.scope_stack: list[str] = []
        self.warnings: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _attr_name(node.func)
        attr = func_name.rsplit(".", 1)[-1]

        if attr in SAFE_DISPATCH_NAMES:
            return

        if isinstance(node.func, ast.Attribute):
            owner = _attr_name(node.func.value)
            if attr in SAFE_QUEUE_NAMES and owner.endswith("_ui_queue"):
                return
            if attr in TK_MUTATOR_ATTRS and owner.startswith("self"):
                line = ast.get_source_segment(self.source, node) or func_name
                scope = " > ".join(self.scope_stack) if self.scope_stack else "<scope>"
                self.warnings.append(f"{scope} line {node.lineno}: direct UI call {line}")
            if owner.startswith("messagebox") and attr.startswith("show"):
                line = ast.get_source_segment(self.source, node) or func_name
                scope = " > ".join(self.scope_stack) if self.scope_stack else "<scope>"
                self.warnings.append(f"{scope} line {node.lineno}: messagebox from worker {line}")

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.scope_stack and node.name in self.allowed_nested_callbacks:
            return
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def _resolve_target_node(
    target_name: str,
    function_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    function_nodes_by_key: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    current_fn: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current_node = function_nodes_by_key.get(current_fn)
    if current_node is not None:
        for node in ast.walk(current_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                return node

    if target_name.startswith("self."):
        class_name = current_fn.split(".", 1)[0] if "." in current_fn else ""
        key = f"{class_name}.{target_name.split('.', 1)[1]}" if class_name else target_name.split(".", 1)[1]
        return function_defs.get(key)

    direct = function_defs.get(target_name)
    if direct is not None:
        return direct

    if "." in current_fn:
        class_name = current_fn.split(".", 1)[0]
        scoped = function_defs.get(f"{class_name}.{target_name}")
        if scoped is not None:
            return scoped

    return None


def _allowed_nested_callbacks(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    allowed: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func_name = _attr_name(sub.func)
        attr = func_name.rsplit(".", 1)[-1]
        if attr not in SAFE_DISPATCH_NAMES:
            continue
        for arg in sub.args:
            if isinstance(arg, ast.Name):
                allowed.add(arg.id)
    return allowed


def main() -> int:
    problems: list[str] = []

    print("THREAD SAFETY AUDIT")
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

        audit = ThreadSafetyAudit(source)
        audit.visit(tree)

        print(
            f"{path.relative_to(REPO_ROOT)} "
            f"thread_starts={len(audit.thread_targets)} "
            f"resolvable_functions={len(audit.function_defs)}"
        )

        for line, current_fn, target_name in audit.thread_targets:
            node = _resolve_target_node(target_name, audit.function_defs, audit.function_nodes_by_key, current_fn)
            if node is None:
                continue
            finder = SuspiciousUiCallFinder(source, allowed_nested_callbacks=_allowed_nested_callbacks(node))
            finder.visit(node)
            for warning in finder.warnings:
                problems.append(f"{path}:{line}: thread target {target_name}: {warning}")

        for warning in audit.inline_thread_warnings:
            print(f"warning: {path}: {warning}")

    if problems:
        print("AUDIT FAILED")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
