from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db import DatabaseManager


def test_paper_trade_links_to_prediction_id(tmp_path):
    db_path = tmp_path / "paper.db"
    db = DatabaseManager(str(db_path))
    db.insert_prediction(
        {
            "prediction_id": "pred_link",
            "ts": 11.0,
            "symbol": "NIFTY",
            "probability": 0.72,
            "prediction": 0.72,
            "trade_candidate": True,
        }
    )
    db.link_prediction_to_trade("pred_link", "T001")
    db.insert_trade_outcome(
        {
            "trade_id": "T001",
            "prediction_id": "pred_link",
            "strategy": "directional",
            "pnl": 250.0,
            "gross_pnl": 260.0,
            "estimated_costs": 10.0,
            "estimated_slippage": 0.0,
        }
    )

    conn = sqlite3.connect(db_path)
    pred_row = conn.execute("SELECT trade_id, trade_taken FROM predictions WHERE prediction_id='pred_link'").fetchone()
    outcome_row = conn.execute("SELECT prediction_id, pnl FROM trade_outcomes WHERE trade_id='T001'").fetchone()
    conn.close()

    assert pred_row == ("T001", 1)
    assert outcome_row == ("pred_link", 250.0)
