"""tests/test_daily_paper_report.py"""
import json, tempfile, os, sys
from pathlib import Path
from unittest.mock import patch

# Ensure the script is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_daily_paper_report import generate_daily_report, collect_journal_entries


def test_generates_report_without_error():
    """generate_daily_report runs without error even with empty journal."""
    with patch("generate_daily_paper_report.PAPER_FWD", Path(tempfile.mkdtemp())):
        result = generate_daily_report("20250608")
    assert "total_decisions" in result
    assert result["total_decisions"] == 0


def test_all_skip_triggers_alert():
    """All SKIP decisions → ALL_SKIP alert."""
    entry = {
        "final_action": "SKIP",
        "blocking_reasons": ["no_live_data"],
        "probability": 0.55,
        "spread_pct": 0.0,
        "premium": 0.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        pdir = Path(tmpdir) / "paper_forward_test"
        pdir.mkdir()
        jl = pdir / "paper_decisions_20250608_001.jsonl"
        jl.write_text(json.dumps(entry) + "\n")
        with patch("generate_daily_paper_report.PAPER_FWD", pdir):
            result = generate_daily_report("20250608")
    assert "ALL_SKIP" in result["alerts"]


def test_no_decisions_triggers_alert():
    """No decisions → NO_DECISIONS alert."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdir = Path(tmpdir) / "paper_forward_test"
        pdir.mkdir()
        with patch("generate_daily_paper_report.PAPER_FWD", pdir):
            result = generate_daily_report("20250608")
    assert "NO_DECISIONS" in result["alerts"]


def test_low_signal_triggers_alert():
    """Mean probability < 0.4 → LOW_SIGNAL alert."""
    entries = [
        {"final_action": "TRADE", "probability": 0.30, "spread_pct": 0.001, "premium": 100.0},
        {"final_action": "TRADE", "probability": 0.35, "spread_pct": 0.001, "premium": 100.0},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        pdir = Path(tmpdir) / "paper_forward_test"
        pdir.mkdir()
        jl = pdir / "paper_decisions_20250608_001.jsonl"
        jl.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        with patch("generate_daily_paper_report.PAPER_FWD", pdir):
            result = generate_daily_report("20250608")
    assert "LOW_SIGNAL" in result["alerts"]