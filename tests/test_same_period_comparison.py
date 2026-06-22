from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import retrain_all_edge_models as retrain


def test_same_period_comparison_restricts_overlap() -> None:
    baseline = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-06-01T09:15:00+05:30",
            "2026-06-01T09:16:00+05:30",
            "2026-06-01T09:17:00+05:30",
            "2026-06-01T09:18:00+05:30",
        ])
    })
    enriched = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-06-01T09:17:00+05:30",
            "2026-06-01T09:18:00+05:30",
            "2026-06-01T09:19:00+05:30",
        ])
    })
    baseline_aligned, enriched_aligned, report = retrain._same_period_alignment_report(baseline, enriched, enabled=True)
    assert report["same_period_gate"] == "PASS"
    assert len(baseline_aligned) == 2
    assert len(enriched_aligned) == 2
    assert report["baseline_rows_dropped"] == 2
    assert report["bs_rows_dropped"] == 1


def test_same_period_gate_fails_when_not_enabled_and_ranges_differ() -> None:
    baseline = pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-01T09:15:00+05:30"])})
    enriched = pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-02T09:15:00+05:30"])})
    _, _, report = retrain._same_period_alignment_report(baseline, enriched, enabled=False)
    assert report["same_period_gate"] == "FAIL"
