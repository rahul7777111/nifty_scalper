from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_paper_risk_manager import MLPaperRiskManager


def test_cooldown_blocks_repeated_paper_trades() -> None:
    mgr = MLPaperRiskManager()
    ts = datetime(2026, 6, 5, 10, 0)
    mgr.record_entry(ts)
    res = mgr.evaluate({"timestamp": ts + timedelta(minutes=5), "option_ltp": 10.0, "option_type": "CE", "bid_ask_spread_pct": 0.01})
    assert not res["allowed"]


def test_max_open_positions_enforced() -> None:
    mgr = MLPaperRiskManager(max_open_paper_positions=1)
    mgr.open_positions = 1
    res = mgr.evaluate({"timestamp": datetime(2026, 6, 5, 10, 30), "option_ltp": 10.0, "option_type": "CE", "bid_ask_spread_pct": 0.01})
    assert not res["allowed"]

