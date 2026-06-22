#!/usr/bin/env python3
"""Fail if GUI code casts Tk after() job IDs to int."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGETS = [
    REPO / "src" / "ui.py",
    REPO / "src" / "ui_live_chart_panels.py",
]

BAD_PATTERNS = [
    (re.compile(r"int\s*\(\s*job_id\s*\)"), "int(job_id)"),
    (re.compile(r"int\s*\(\s*after_id\s*\)"), "int(after_id)"),
    (re.compile(r"int\s*\(\s*self\.after\s*\("), "int(self.after(...))"),
    (re.compile(r"_after_jobs\s*\[\s*[^\]]+\s*\]\s*=\s*int\s*\("), "_after_jobs[...] = int(...)"),
]

ALLOW_LINE = re.compile(r"_normalize_after_job_id|Never parse|opaque string")


def main() -> int:
    errors: list[str] = []
    for path in TARGETS:
        if not path.exists():
            errors.append(f"{path}: missing")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if ALLOW_LINE.search(line):
                continue
            for rx, label in BAD_PATTERNS:
                if rx.search(line):
                    errors.append(f"{path}:{lineno}: forbidden {label}")

    if errors:
        print("verify_gui_after_jobs FAILED")
        for err in errors:
            print(f"  [FAIL] {err}")
        return 1

    print("verify_gui_after_jobs PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())