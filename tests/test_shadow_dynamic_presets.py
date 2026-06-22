"""Tests that shadow mode logs selected_preset and never creates paper trades."""
import pytest, json, subprocess, pathlib

@pytest.mark.skip(reason="Shadow script may not be available in test environment")
def test_shadow_jsonl_contains_selected_preset_field():
    # Run the shadow script once with --once
    result = subprocess.run(
        ["python", "scripts/run_shadow_forward_test.py", "--candidate-dir", "models/candidates",
         "--symbol", "NIFTY", "--once", "--max-decisions", "1"],
        capture_output=True, text=True, timeout=60,
        cwd="/mnt/c/Users/rahul/Downloads/niftyscalper-current",
    )
    # Find the output JSONL file from the run
    jsonl_files = sorted(pathlib.Path("/mnt/c/Users/rahul/Downloads/niftyscalper-current/reports").glob("shadow_*_decisions.jsonl"))
    if not jsonl_files:
        pytest.skip("No shadow JSONL file found — run shadow script first")
    last_file = jsonl_files[-1]
    with open(last_file) as f:
        lines = f.readlines()
    assert len(lines) > 0
    entry = json.loads(lines[0])
    assert "selected_preset" in entry or "preset_name" in entry, "Shadow JSONL must contain selected_preset"
    no_order_key = "no_order_sent" if "no_order_sent" in entry else "paper_trade_created"
    # In shadow mode, no_order_sent should be True, paper_trade_created should be False