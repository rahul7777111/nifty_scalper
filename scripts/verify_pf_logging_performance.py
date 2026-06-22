#!/usr/bin/env python3
"""Static scan: Paper Forward loop must not flood console with heavy debug I/O."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SCAN_FILES = [
    REPO / "src" / "paper_forward_engine.py",
    REPO / "src" / "candidate_router.py",
    REPO / "src" / "ui.py",
    REPO / "src" / "chart.py",
    REPO / "src" / "pf_logging.py",
]

# Patterns that indicate console flooding inside PF hot paths.
BAD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"print\s*\(\s*decision\b"), "print(decision)"),
    (re.compile(r"print\s*\(\s*feature_contract\b"), "print(feature_contract)"),
    (re.compile(r"print\s*\([^)]*available_features"), "printing available_features"),
    (re.compile(r"print\s*\([^)]*feature_order"), "printing feature_order"),
    (re.compile(r"\bpprint\b"), "pprint in PF module"),
    (re.compile(r"json\.dumps\s*\(\s*decision\b"), "json.dumps(decision) to console"),
]

# Allowed if routed through pf_logging compact helpers.
ALLOW_SUBSTR = (
    "pf_log(",
    "log_paper_fwd_",
    "log_feature_x(",
    "log_pf_data(",
    "log_gui_perf(",
    "rate_key=",
    "verify_pf_logging_performance",
    "# ok:",
)

# Model / CSV reload inside obvious loop bodies (heuristic).
LOOP_RELOAD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"pickle\.load\s*\("), "pickle.load inside hot path (use cache)"),
    (re.compile(r"open\s*\([^)]*model\.pkl"), "model.pkl open inside hot path"),
    (re.compile(r"DictReader\s*\("), "instrument CSV DictReader inside hot path"),
]

REQUIRED_HELPERS = [
    (REPO / "src" / "pf_logging.py", "def should_log"),
    (REPO / "src" / "pf_logging.py", "PF_LOG_LEVEL"),
    (REPO / "src" / "candidate_router.py", "_PF_MODEL_CACHE"),
    (REPO / "src" / "paper_forward_engine.py", "_option_chain_cache"),
    (REPO / "src" / "ui.py", "_log_flush_interval_ms"),
]


def _in_hot_loop_context(lines: list[str], idx: int) -> bool:
    """Heuristic: line is inside on_market_snapshot / _predict / poller loop."""
    window = "\n".join(lines[max(0, idx - 40): idx + 1])
    markers = (
        "on_market_snapshot",
        "_predict_confidence_from_artifact",
        "paper_forward_poll",
        "_pf_data_poller",
        "for cand in self.candidates",
        "for candidate in",
    )
    return any(m in window for m in markers)


def main() -> int:
    errors: list[str] = []

    for path, needle in REQUIRED_HELPERS:
        if not path.exists():
            errors.append(f"{path}: missing file")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if needle not in text:
            errors.append(f"{path}: missing required symbol {needle!r}")

    for path in SCAN_FILES:
        if not path.exists():
            errors.append(f"{path}: missing")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if any(a in line for a in ALLOW_SUBSTR):
                continue
            for rx, label in BAD_PATTERNS:
                if rx.search(line):
                    errors.append(f"{path}:{lineno}: forbidden {label}")
            if path.name in {"candidate_router.py", "paper_forward_engine.py"}:
                if _in_hot_loop_context(lines, lineno - 1):
                    for rx, label in LOOP_RELOAD_PATTERNS:
                        if rx.search(line) and "cache" not in line.lower():
                            # candidate_router may pickle.load in cached loader only
                            if "_load_model_bundle_cached" in "\n".join(lines[max(0, lineno - 15): lineno + 5]):
                                continue
                            if path.name == "scripmaster.py" and "_load(" in "\n".join(lines[max(0, lineno - 20): lineno]):
                                continue
                            errors.append(f"{path}:{lineno}: {label}")

    if errors:
        print("verify_pf_logging_performance FAILED")
        for err in errors:
            print(f"  [FAIL] {err}")
        return 1

    print("verify_pf_logging_performance PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())