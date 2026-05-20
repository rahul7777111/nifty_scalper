from __future__ import annotations

import os
import sys


def pytest_configure() -> None:
    """Ensure the project-style imports work under pytest.

    The codebase imports modules like `candlestick_patterns` from `src/` without
    a package prefix, so we add `src/` to `sys.path` for test runs.
    """

    repo_root = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(repo_root, "src"),
        os.path.join(repo_root, "pytradingapi-typeB-main"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
