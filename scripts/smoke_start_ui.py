#!/usr/bin/env python3
"""Smoke-start the Tk UI in an isolated subprocess with autostart disabled."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TIMEOUT_SEC = 25


CHILD_CODE = textwrap.dedent(
    f"""
    import os
    import sys
    import time
    import traceback
    from pathlib import Path

    REPO_ROOT = Path(r"{REPO_ROOT}")
    SRC_DIR = Path(r"{SRC_DIR}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    os.environ.setdefault("FAST_UI_START", "1")
    os.environ.setdefault("MSTOCK_ENABLE_LIVE_ORDERS", "false")
    os.environ.setdefault("MSTOCK_UI_STARTUP_POPUP", "false")
    os.environ.setdefault("MSTOCK_UI_BRING_TO_FRONT", "false")
    os.environ.setdefault("MSTOCK_AUTO_START", "false")
    os.environ.setdefault("MSTOCK_AUTO_START_PAPER_FORWARD", "false")
    os.environ.setdefault("MSTOCK_PAPER_FORWARD_AUTOSTART", "false")
    os.environ.setdefault("PF_DISABLE_STARTUP_AUTOSTART", "1")
    os.environ.setdefault("SCALPER_BROKER", "mstock")

    try:
        import tkinter as tk
        from ui import ScalperUI
    except Exception:
        print("UI_SMOKE_IMPORT_FAIL")
        print(traceback.format_exc())
        raise SystemExit(1)

    app = None
    try:
        started = time.perf_counter()
        app = ScalperUI()
        try:
            app.withdraw()
        except Exception:
            pass
        deadline = started + 8.0
        notebook = None
        while time.perf_counter() < deadline:
            app.update_idletasks()
            try:
                app.update()
            except tk.TclError:
                pass
            notebook = getattr(app, "notebook", None)
            if notebook is not None:
                try:
                    if notebook.winfo_exists() and len(notebook.tabs()) >= 1:
                        break
                except Exception:
                    pass
            if getattr(app, "_widgets_shell_built", False):
                break
            time.sleep(0.02)

        tabs = 0
        try:
            if notebook is not None and notebook.winfo_exists():
                tabs = len(notebook.tabs())
        except Exception:
            tabs = 0

        print(
            "UI_SMOKE_OK "
            f"elapsed_ms={{int((time.perf_counter() - started) * 1000)}} "
            f"shell_built={{bool(getattr(app, '_widgets_shell_built', False))}} "
            f"deferred_built={{bool(getattr(app, '_widgets_deferred_built', False))}} "
            f"tabs={{tabs}}"
        )
        raise SystemExit(0)
    except Exception:
        print("UI_SMOKE_START_FAIL")
        print(traceback.format_exc())
        raise SystemExit(1)
    finally:
        try:
            if app is not None and hasattr(app, "_cancel_all_after_jobs"):
                app._cancel_all_after_jobs()
        except Exception:
            pass
        try:
            if app is not None and hasattr(app, "_on_close") and app.winfo_exists():
                app._on_close()
        except Exception:
            pass
        try:
            if app is not None and app.winfo_exists():
                app.destroy()
        except Exception:
            pass
    """
).strip()


def main() -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [sys.executable, "-c", CHILD_CODE]
    print("UI SMOKE START")
    print(f"python={sys.executable}")
    print(f"repo={REPO_ROOT}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"UI_SMOKE_TIMEOUT timeout_sec={TIMEOUT_SEC}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        return 1

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())

    if proc.returncode != 0:
        print(f"UI_SMOKE_FAILED returncode={proc.returncode}")
        return proc.returncode

    print("UI_SMOKE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
