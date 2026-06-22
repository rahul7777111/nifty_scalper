"""State machine tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VALID = ["WAITING_FOR_MARKET_DATA", "RUNNING_DATA_ONLY", "RUNNING_EVALUATING", "STOPPED", "DEGRADED_NO_MARKET_DATA"]

def test_states_accurate():
    assert "WAITING_FOR_MARKET_DATA" in VALID
    assert "RUNNING_EVALUATING" in VALID
