#!/usr/bin/env python3
"""Runtime smoke test that opens every Tk notebook tab without broker login."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MSTOCK_ENABLE_LIVE_ORDERS", "false")
os.environ.setdefault("MSTOCK_UI_STARTUP_POPUP", "false")
os.environ.setdefault("MSTOCK_AUTO_START", "false")
os.environ.setdefault("MSTOCK_AUTO_START_PAPER_FORWARD", "false")
os.environ.setdefault("MSTOCK_PAPER_FORWARD_AUTOSTART", "false")


def _select_tabs(app, notebook, label: str, errors: list[str]) -> None:
    import tkinter as tk

    try:
        tabs = list(notebook.tabs())
    except Exception as exc:
        errors.append(f"{label}: cannot list tabs: {exc}")
        return

    for idx, tab_id in enumerate(tabs):
        try:
            tab_text = str(notebook.tab(tab_id, "text") or f"#{idx}")
        except Exception:
            tab_text = f"#{idx}"
        try:
            notebook.select(tab_id)
            notebook.event_generate("<<NotebookTabChanged>>")
            for _ in range(5):
                app.update()
                time.sleep(0.02)
            print(f"PASS tab {label}/{tab_text}")
        except tk.TclError as exc:
            errors.append(f"{label}/{tab_text}: TclError: {exc}")
        except Exception as exc:
            errors.append(f"{label}/{tab_text}: {type(exc).__name__}: {exc}")


def _notebooks(app) -> list[tuple[str, object]]:
    import tkinter.ttk as ttk

    found: list[tuple[str, object]] = []
    seen: set[int] = set()
    for name, value in sorted(getattr(app, "__dict__", {}).items()):
        try:
            if isinstance(value, ttk.Notebook) and id(value) not in seen:
                found.append((name, value))
                seen.add(id(value))
        except Exception:
            continue
    return found


def main() -> int:
    import tkinter as tk

    errors: list[str] = []
    app = None
    try:
        from ui import ScalperUI
    except Exception:
        print("GUI_SMOKE_IMPORT_FAIL")
        print(traceback.format_exc())
        return 1

    try:
        app = ScalperUI()
        app.withdraw()
        app.update()
    except tk.TclError:
        print("GUI_SMOKE_STARTUP_FAIL")
        print(traceback.format_exc())
        return 1
    except Exception:
        print("GUI_SMOKE_STARTUP_FAIL")
        print(traceback.format_exc())
        return 1

    try:
        notebooks = _notebooks(app)
        if not notebooks:
            errors.append("no ttk.Notebook widgets found")
        for name, notebook in notebooks:
            _select_tabs(app, notebook, name, errors)

        try:
            if hasattr(app, "_cancel_all_after_jobs"):
                app._cancel_all_after_jobs()
        except Exception as exc:
            errors.append(f"cancel after jobs failed: {exc}")
        try:
            if hasattr(app, "_on_close"):
                app._on_close()
            else:
                app.destroy()
        except Exception as exc:
            errors.append(f"close failed: {exc}")
    finally:
        try:
            if app is not None and app.winfo_exists():
                app.destroy()
        except Exception:
            pass

    if errors:
        print("GUI_SMOKE_FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("GUI_SMOKE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
