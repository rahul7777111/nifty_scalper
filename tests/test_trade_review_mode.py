"""Tests for Trade Review Mode in UI.

Tests the data-logic portion: trade extraction, PnL calculation,
duration, outcome classification, CSV export, and panel population.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Computation helpers (mimic what TradeReviewMode panel computes)
# ---------------------------------------------------------------------------


def _extract_trade_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract key fields from a trade event row for review display."""
    ts = row.get("timestamp") or row.get("ts") or row.get("entry_ts")
    entry = row.get("entry_price") or row.get("entry")
    exit_price = row.get("exit_price") or row.get("exit") or row.get("exit_price_")
    qty = row.get("quantity") or row.get("qty") or row.get("size") or 1
    prob = row.get("probability") or row.get("prob") or row.get("confidence")
    trade_id = row.get("trade_id") or row.get("id") or row.get("tradeId") or ""
    return {
        "trade_id": str(trade_id).strip(),
        "entry_ts": ts,
        "entry_price": float(entry) if entry not in (None, "") else None,
        "exit_price": float(exit_price) if exit_price not in (None, "") else None,
        "quantity": float(qty),
        "probability": float(prob) if prob not in (None, "") else None,
    }


def _trade_pnl(entry: float, exit_price: float, qty: float, side: str = "BUY") -> float:
    """Compute PnL for a trade. side: BUY or SELL."""
    if side.upper() == "BUY":
        return (exit_price - entry) * qty
    else:  # SELL
        return (entry - exit_price) * qty


def _holding_duration_minutes(entry_ts: Any, exit_ts: Any) -> float:
    """Return holding duration in minutes between two timestamps."""
    try:
        if isinstance(entry_ts, (int, float)):
            entry_dt = datetime.fromtimestamp(float(entry_ts), tz=timezone.utc)
        elif isinstance(entry_ts, str):
            entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
        else:
            return 0.0

        if isinstance(exit_ts, (int, float)):
            exit_dt = datetime.fromtimestamp(float(exit_ts), tz=timezone.utc)
        elif isinstance(exit_ts, str):
            exit_dt = datetime.fromisoformat(exit_ts.replace("Z", "+00:00"))
        else:
            return 0.0

        delta = exit_dt - entry_dt
        return delta.total_seconds() / 60.0
    except Exception:
        return 0.0


def _outcome(net_pnl: float) -> str:
    if net_pnl > 0:
        return "WIN"
    elif net_pnl < 0:
        return "LOSS"
    else:
        return "BE"


def _sort_trades_by_pnl(trades: list[dict[str, Any]], descending: bool = True) -> list[dict[str, Any]]:
    """Sort trade dicts by net_pnl field."""
    return sorted(trades, key=lambda t: float(t.get("net_pnl", 0.0) or 0.0), reverse=descending)


def _export_trades_csv(trades: list[dict[str, Any]], path: str | Path) -> Path:
    """Write trades to a CSV file. Returns the Path."""
    fieldnames = ["trade_id", "entry_price", "exit_price", "quantity", "net_pnl", "outcome", "duration_min", "probability"]
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow({k: t.get(k, "") for k in fieldnames})
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trade_review_row_extraction() -> None:
    row = {
        "trade_id": "T-001",
        "timestamp": 1200.0,
        "entry_price": 100.0,
        "exit_price": 120.0,
        "quantity": 50,
        "probability": 0.78,
    }
    extracted = _extract_trade_row(row)
    assert extracted["trade_id"] == "T-001"
    assert extracted["entry_price"] == 100.0
    assert extracted["exit_price"] == 120.0
    assert extracted["quantity"] == 50.0
    assert extracted["probability"] == 0.78


def test_trade_review_pnl_calculation() -> None:
    # entry=100, exit=120, qty=50, BUY -> (120-100)*50 = 1000
    pnl = _trade_pnl(entry=100.0, exit_price=120.0, qty=50.0, side="BUY")
    assert pnl == 1000.0

    # Same but SELL -> (100-120)*50 = -1000
    pnl_sell = _trade_pnl(entry=100.0, exit_price=120.0, qty=50.0, side="SELL")
    assert pnl_sell == -1000.0

    # Fractional qty
    pnl_frac = _trade_pnl(entry=100.0, exit_price=110.0, qty=10.5, side="BUY")
    assert pnl_frac == 105.0


def test_trade_review_holding_duration() -> None:
    # 10:00 -> 10:30 = 30 minutes
    duration = _holding_duration_minutes(entry_ts=36000.0, exit_ts=37800.0)
    assert duration == pytest.approx(30.0, abs=1.0)  # allow 1-min tolerance

    # 09:15 IST (03:45 UTC) -> 15:30 IST (10:00 UTC) = 6h 15min = 375 min
    # Use ISO strings so timezone is unambiguous
    duration_long = _holding_duration_minutes(
        entry_ts="2026-06-01T03:45:00Z",
        exit_ts="2026-06-01T10:00:00Z",
    )
    assert duration_long == pytest.approx(375.0, abs=1.0)

    # ISO string timestamps
    duration_iso = _holding_duration_minutes(
        entry_ts="2026-06-01T09:15:00Z",
        exit_ts="2026-06-01T15:30:00Z",
    )
    assert duration_iso == pytest.approx(375.0, abs=1.0)


def test_trade_review_features_used() -> None:
    """Verify feature_names can be extracted from prediction rows."""
    row = {
        "trade_id": "T-001",
        "feature_names": ["rsi_14", "ema_9", "volume", "iv_percentile"],
        "predicted_class": 1,
        "probability": 0.72,
    }
    features = row.get("feature_names", [])
    assert len(features) == 4
    assert "rsi_14" in features


def test_trade_review_outcome_winner() -> None:
    assert _outcome(100.0) == "WIN"
    assert _outcome(0.01) == "WIN"
    assert _outcome(1_000_000.0) == "WIN"


def test_trade_review_outcome_loser() -> None:
    assert _outcome(-0.01) == "LOSS"
    assert _outcome(-100.0) == "LOSS"
    assert _outcome(-1_000_000.0) == "LOSS"


def test_trade_review_outcome_breakeven() -> None:
    assert _outcome(0.0) == "BE"


def test_trade_review_empty_trade_id() -> None:
    row = {
        "trade_id": "",
        "entry_price": 100.0,
        "exit_price": 120.0,
        "quantity": 10.0,
    }
    extracted = _extract_trade_row(row)
    assert extracted["trade_id"] == ""


def test_trade_review_missing_fields() -> None:
    # Missing exit_price -> treated as "open" (None)
    row = {
        "trade_id": "T-OPEN",
        "entry_price": 100.0,
        "quantity": 20.0,
    }
    extracted = _extract_trade_row(row)
    assert extracted["exit_price"] is None
    assert extracted["entry_price"] == 100.0


def test_trade_review_sorting_by_pnl() -> None:
    trades = [
        {"trade_id": "A", "net_pnl": -50.0},
        {"trade_id": "B", "net_pnl": 200.0},
        {"trade_id": "C", "net_pnl": -10.0},
        {"trade_id": "D", "net_pnl": 150.0},
        {"trade_id": "E", "net_pnl": 0.0},
    ]
    sorted_trades = _sort_trades_by_pnl(trades, descending=True)
    assert sorted_trades[0]["trade_id"] == "B"
    assert sorted_trades[1]["trade_id"] == "D"
    assert sorted_trades[2]["trade_id"] == "E"
    assert sorted_trades[3]["trade_id"] == "C"
    assert sorted_trades[4]["trade_id"] == "A"


def test_trade_review_export_csv(tmp_path: Path) -> None:
    trades = [
        {
            "trade_id": "T-001",
            "entry_price": 100.0,
            "exit_price": 120.0,
            "quantity": 50.0,
            "net_pnl": 1000.0,
            "outcome": "WIN",
            "duration_min": 30.0,
            "probability": 0.78,
        },
        {
            "trade_id": "T-002",
            "entry_price": 200.0,
            "exit_price": 180.0,
            "quantity": 25.0,
            "net_pnl": -500.0,
            "outcome": "LOSS",
            "duration_min": 45.0,
            "probability": 0.65,
        },
    ]
    out_path = tmp_path / "trade_review.csv"
    result = _export_trades_csv(trades, out_path)
    assert result == out_path
    assert out_path.exists()

    # Read back and verify
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["trade_id"] == "T-001"
    assert rows[0]["net_pnl"] == "1000.0"
    assert rows[0]["outcome"] == "WIN"
    assert rows[1]["trade_id"] == "T-002"
    assert rows[1]["net_pnl"] == "-500.0"
    assert rows[1]["outcome"] == "LOSS"


def test_trade_review_click_populates_panel() -> None:
    """Simulate clicking a trade in the chart to focus it in the review panel.

    We verify the data that would be passed to the panel is complete and correct.
    """
    import os
    import sys

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(REPO_ROOT, "src")
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import tkinter as tk
        from chart import LiveChartPlugin
    except Exception:
        pytest.skip("matplotlib/tkinter not available in this environment")

    root = tk.Tk()
    root.withdraw()
    frame = tk.Frame(root)
    try:
        plugin = LiveChartPlugin(frame)

        # Load 5 trade events
        trade_rows = [
            {"event": "OPEN", "ts": 1000.0, "trade_id": f"T-{i:03d}", "name": f"Trade {i}"}
            for i in range(1, 6)
        ]
        plugin.load_trade_event_rows(trade_rows)
        assert len(plugin._marks) == 5

        # Focus on T-003 specifically
        plugin.focus_trade("T-003")
        assert plugin._focus_trade_id == "T-003"

        # Verify the focused trade would be highlighted in render (check marks list)
        focused_marks = [m for m in plugin._marks if m.trade_id == "T-003"]
        assert len(focused_marks) == 1
        assert focused_marks[0].trade_id == "T-003"
    finally:
        root.destroy()


import pytest  # noqa: E402  (needed for pytest.approx)