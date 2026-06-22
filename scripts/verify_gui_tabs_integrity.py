#!/usr/bin/env python3
"""Static integrity audit for NiftyScalper Tkinter GUI tabs and refresh loops."""
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
    REPO / "src" / "chart.py",
]

WARNINGS = 0
INFOS = 0

SAFE_AFTER = {
    "_safe_after_app",
    "_safe_after_bot",
    "_safe_after",
    "_gui_schedule_tab_after",
    "_schedule_after_job",
    "_run_on_ui_thread",
    "ui_call",
    "_safe_after_app",
}


def warn(path: Path, func: str, lineno: int, issue: str, fix: str) -> None:
    global WARNINGS
    WARNINGS += 1
    print(f"[WARN] {path}:{lineno} function={func} issue={issue} suggested_fix={fix}")


def info(msg: str) -> None:
    global INFOS
    INFOS += 1
    print(f"[PASS] {msg}")


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def _func_name(node: ast.AST) -> str:
    while node is not None:
        if isinstance(node, ast.FunctionDef):
            return node.name
        node = getattr(node, "parent", None)
    return "<module>"


def _function_block(text: str, func: str) -> tuple[int, str]:
    m = re.search(rf"^\s*def\s+{re.escape(func)}\s*\(", text, re.M)
    if not m:
        return 0, ""
    start = text[: m.start()].count("\n") + 1
    lines = text.splitlines()
    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].startswith("    def ") and not lines[i].startswith("        "):
            end = i
            break
    return start, "\n".join(lines[start - 1 : end])


def audit_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        warn(path, "<module>", exc.lineno or 0, f"syntax error {exc.msg}", "fix syntax before running GUI")
        return
    _attach_parents(tree)

    recurring: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            src = ast.get_source_segment(text, node) or ""
            if any(tok in src for tok in (".after(", "_safe_after", "_gui_schedule_tab_after", "while not")):
                recurring.add(node.name)

    for func in sorted(recurring):
        start, block = _function_block(text, func)
        if not block:
            continue
        if re.search(r"get_children\s*\(", block) and re.search(r"\.delete\s*\(", block) and re.search(r"\.insert\s*\(", block):
            if any(k in func for k in ("refresh", "render", "pump", "update", "schedule", "_cp_", "_apply_validation")):
                if "_upsert_tree_rows" not in block and "_pf_refresh_table" not in func and "callable(upsert)" not in block:
                    warn(
                        path,
                        func,
                        start,
                        "recurring refresh uses tree delete+mass insert",
                        "use _upsert_tree_rows with stable iid map",
                    )
        if "threading.current_thread()" in block or "Thread(" in block or "def _worker" in block or "_worker" in func:
            if re.search(r"(tree|label|text|canvas|StringVar|\.var)\.(insert|delete|item|config|set|draw)", block, re.I):
                if "_run_on_ui_thread" not in block and "ui_call" not in block and "after(0" not in block and "_safe_after_app" not in block:
                    warn(
                        path,
                        func,
                        start,
                        "worker/thread function may touch Tk widgets directly",
                        "build payload in worker, apply via _run_on_ui_thread/root.after(0)",
                    )
        if "canvas.draw(" in block and "draw_idle" not in block:
            if "threading" in block or "Thread" in block:
                warn(path, func, start, "canvas.draw from possible worker thread", "use _safe_canvas_draw_idle on UI thread")
        if re.search(r"\.column\s*\(.*width", block) and any(k in func for k in ("refresh", "render", "update")):
            warn(path, func, start, "column width configured inside refresh loop", "configure columns once at build time")

    if path.name == "ui.py":
        if "_on_close" not in text:
            warn(path, "_on_close", 0, "missing _on_close handler", "implement shutdown cleanup")
        elif "_ui_closing" not in text.split("def _on_close")[1].split("def ", 1)[0]:
            warn(path, "_on_close", 0, "_on_close should set _ui_closing", "set self._ui_closing = True at start")
        if "_cancel_all_after_jobs" not in text:
            warn(path, "<module>", 0, "missing _cancel_all_after_jobs helper", "add after-job registry cleanup")
        if "_upsert_tree_rows" not in text:
            warn(path, "<module>", 0, "missing _upsert_tree_rows helper", "add stable tree upsert helper")
        if "_register_all_tab_audits" not in text:
            warn(path, "<module>", 0, "missing tab audit registry", "add _register_all_tab_audits")
        else:
            info(f"{path.name}: tab audit registry present")
        if "_log_gui_error" not in text:
            warn(path, "<module>", 0, "missing _log_gui_error helper", "log GUI exceptions with context")
        else:
            info(f"{path.name}: _log_gui_error present")
        if "already_running=true skip_start_duplicate" in text:
            info(f"{path.name}: paper-forward duplicate-start guard present")

    # Silent except/pass in high-risk GUI handlers (capped)
    silent_hits = 0
    lines = text.splitlines()
    gui_func_re = re.compile(r"(refresh|render|pump|worker|_on_|_bt_|_pf_|_cp_|_trigger_|_apply_|_build_)")
    for idx, line in enumerate(lines):
        if silent_hits >= 25:
            break
        if re.match(r"\s*except\s+Exception(?:\s+as\s+\w+)?:\s*$", line):
            nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
            if re.match(r"\s*pass\s*$", nxt):
                fn = _func_name_from_lines(lines, idx)
                if gui_func_re.search(fn):
                    silent_hits += 1
                    warn(
                        path,
                        fn,
                        idx + 1,
                        "silent except Exception: pass in GUI handler",
                        "use self._log_gui_error(context, exc, tab=...)",
                    )

    for lineno, line in enumerate(text.splitlines(), start=1):
        if ".after(" in line and not any(s in line for s in SAFE_AFTER):
            if "lambda" in line and any(k in line for k in ("tree.", "label.", "text.", "canvas.")):
                warn(path, "<lambda>", lineno, "raw after() lambda may touch widgets", "route via ui_call/_run_on_ui_thread")


def _func_name_from_lines(lines: list[str], idx: int) -> str:
    for j in range(idx, -1, -1):
        m = re.match(r"\s*def\s+(\w+)\s*\(", lines[j])
        if m:
            return m.group(1)
    return "<module>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", dest="files", default=[])
    args = parser.parse_args(argv)
    targets = [Path(p) for p in args.files] if args.files else DEFAULT_FILES
    print("GUI tabs integrity audit")
    print("=" * 60)
    for path in targets:
        if not path.exists():
            warn(path, "<module>", 0, "file not found", "check path")
            continue
        print(f"\n-- {path.relative_to(REPO)} --")
        audit_file(path)
    print("\n" + "=" * 60)
    print(f"Warnings: {WARNINGS} | Info: {INFOS}")
    return 1 if WARNINGS else 0


if __name__ == "__main__":
    sys.exit(main())