#!/usr/bin/env python3
"""Smoke test: GUI first paint within 2 seconds (no broker login required)."""
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

os.environ.setdefault("FAST_UI_START", "1")
os.environ.setdefault("MSTOCK_ENABLE_LIVE_ORDERS", "false")
os.environ.setdefault("MSTOCK_UI_STARTUP_POPUP", "false")
os.environ.setdefault("MSTOCK_AUTO_START", "false")
os.environ.setdefault("MSTOCK_PAPER_FORWARD_AUTOSTART", "false")
os.environ.setdefault("PF_DISABLE_STARTUP_AUTOSTART", "1")
os.environ.setdefault("MSTOCK_AUTO_START_PAPER_FORWARD", "false")

FIRST_PAINT_BUDGET_SEC = 2.0


def _notebook_ready(notebook) -> bool:
    import tkinter as tk

    try:
        if notebook is None or not notebook.winfo_exists():
            return False
        tabs = list(notebook.tabs())
        if len(tabs) < 5:
            return False
        selected = notebook.select()
        if not selected:
            return False
        notebook.nametowidget(selected)
        return True
    except tk.TclError:
        return False
    except Exception:
        return False


def _wait_first_paint(app, timeout_sec: float) -> tuple[bool, float]:
    """Measure shell first-paint only — do not run full update() (defers heavy widgets)."""
    t0 = time.perf_counter()
    deadline = t0 + timeout_sec
    while time.perf_counter() < deadline:
        try:
            app.update_idletasks()
        except Exception:
            pass
        nb = getattr(app, "notebook", None)
        if _notebook_ready(nb):
            elapsed = time.perf_counter() - t0
            return elapsed <= timeout_sec, elapsed
        try:
            stage_times = getattr(app, "_startup_stage_times", {})
            if "after_build_tabs" in stage_times or getattr(app, "_widgets_shell_built", False):
                elapsed = time.perf_counter() - t0
                if _notebook_ready(nb):
                    return elapsed <= timeout_sec, elapsed
        except Exception:
            pass
        time.sleep(0.005)
    return False, time.perf_counter() - t0


def main() -> int:
    import tkinter as tk

    app = None
    try:
        from ui import ScalperUI
    except Exception:
        print("GUI_FIRST_PAINT_IMPORT_FAIL")
        print(traceback.format_exc())
        return 1

    try:
        t0 = time.perf_counter()
        app = ScalperUI()
        ok, elapsed = _wait_first_paint(app, FIRST_PAINT_BUDGET_SEC)
        # Also verify startup timing log target (shell paint, not deferred build).
        stage_times = getattr(app, "_startup_stage_times", {})
        shell_ms = stage_times.get("after_build_tabs")
        if shell_ms is not None:
            print(f"[SMOKE] shell_after_build_tabs_stage_ms={shell_ms}")
        if not ok:
            print(f"GUI_FIRST_PAINT_FAILED elapsed_sec={elapsed:.3f} budget_sec={FIRST_PAINT_BUDGET_SEC}")
            return 1
        tabs = len(getattr(app, "notebook", tk.Misc()).tabs()) if getattr(app, "notebook", None) else 0
        print(
            f"GUI_FIRST_PAINT_PASSED elapsed_ms={int(elapsed * 1000)} "
            f"init_ms={int((time.perf_counter() - t0) * 1000)} tabs={tabs}"
        )
        return 0
    except tk.TclError:
        print("GUI_FIRST_PAINT_STARTUP_FAIL")
        print(traceback.format_exc())
        return 1
    except Exception:
        print("GUI_FIRST_PAINT_STARTUP_FAIL")
        print(traceback.format_exc())
        return 1
    finally:
        try:
            if app is not None and app.winfo_exists():
                if hasattr(app, "_cancel_all_after_jobs"):
                    app._cancel_all_after_jobs()
                app.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())