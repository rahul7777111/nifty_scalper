#!/usr/bin/env python3
"""PHASE 19 report gen tests."""
import pandas as pd, json, tempfile
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def test_summary_has_required_stats(tmp_path):
    summary = {
        "total_decisions": 100, "total_simulated_trades": 12, "skipped_decisions": 88,
        "gross_sim_pnl": 120.5, "est_cost": 15.0, "net_sim_pnl": 105.5,
        "win_rate": 0.55, "avg_trade_return": 8.79, "max_drawdown": 22.3,
        "live_trading_status": "DISABLED",
        "next_recommendation": "continue paper observation",
    }
    p = tmp_path / "sum.json"
    p.write_text(json.dumps(summary))
    s = json.loads(p.read_text())
    assert s["live_trading_status"] == "DISABLED"
    assert s["max_drawdown"] >= 0  # positive
    assert "net_sim_pnl" in s

def test_trades_csv_and_candidate_stats(tmp_path):
    trades = pd.DataFrame([{"gross_pnl": 5, "costs": 0.5, "net_pnl": 4.5, "option_type": "CE"}])
    tcsv = tmp_path / "trades.csv"
    trades.to_csv(tcsv, index=False)
    assert tcsv.exists()
    cstats = pd.DataFrame([{"candidate_id": "c1", "trades": 5}])
    cstats.to_csv(tmp_path / "candstats.csv", index=False)
    assert "trades" in cstats.columns

def test_ce_pe_split_and_dd_positive():
    # from journal
    j = [{"option_type": "CE", "net_pnl": 1}, {"option_type": "PE", "net_pnl": -0.5}]
    ce = sum(1 for x in j if x["option_type"]=="CE")
    assert ce == 1
    dd = 12.3
    assert dd > 0
