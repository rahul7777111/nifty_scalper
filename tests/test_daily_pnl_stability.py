from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_daily_pnl_gate_fails_on_single_outlier_day() -> None:
    rows = [
        {"timestamp": "2026-06-01T10:00:00+05:30", "selected_return": 10.0},
        {"timestamp": "2026-06-02T10:00:00+05:30", "selected_return": -0.1},
        {"timestamp": "2026-06-03T10:00:00+05:30", "selected_return": -0.1},
    ]
    report = retrain._daily_pnl_stability_report(rows, max_trades_per_day=5)
    assert report["gate"] == "FAIL"
