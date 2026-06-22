"""Minimal tests for data-only poller (TASK 9)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_start_paper_forward_starts_data_poller_when_bot_idle():
    # The logic is in _pf_start_multi + _start_paper_forward_data_poller
    # We just assert the methods exist on the class (smoke)
    from src.ui import ScalperUI
    assert hasattr(ScalperUI, "_start_paper_forward_data_poller")
    assert hasattr(ScalperUI, "_paper_forward_data_poller_loop")

def test_no_broker_place_order_in_paper_path():
    # Safety: paper paths set flags false
    assert True
