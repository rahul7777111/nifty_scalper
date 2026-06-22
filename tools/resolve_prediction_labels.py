from __future__ import annotations

import json
import sqlite3
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
DB_PATH = REPO_ROOT / "trades.db"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cost_model import CostModel
from db import DatabaseManager
from market_data import Candle


def dict_to_candle(raw: Dict[str, Any]) -> Optional[Candle]:
    from datetime import datetime

    raw_time = str(raw.get("time", "")).split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw_time, fmt)
            return Candle(
                time=dt,
                open=float(raw.get("open", 0.0) or 0.0),
                high=float(raw.get("high", 0.0) or 0.0),
                low=float(raw.get("low", 0.0) or 0.0),
                close=float(raw.get("close", 0.0) or 0.0),
                volume=float(raw.get("volume", 0.0) or 0.0),
            )
        except Exception:
            continue
    return None


def load_candles() -> List[Candle]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("candles") or [])
        except Exception:
            continue
    candles = [c for c in (dict_to_candle(row) for row in rows) if c is not None]
    candles.sort(key=lambda candle: candle.time)
    return candles[::10]


def resolve_predictions() -> Dict[str, Any]:
    DatabaseManager(str(DB_PATH))
    candles = load_candles()
    candle_ts = [float(c.time.timestamp()) for c in candles]
    cost_model = CostModel()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.prediction_id, p.ts, p.horizon_bars, p.trade_id, o.pnl
        FROM predictions p
        LEFT JOIN trade_outcomes o ON o.prediction_id = p.prediction_id
        WHERE COALESCE(p.future_label_status, 'pending') != 'resolved'
        ORDER BY p.id ASC
        """
    )
    rows = [dict(row) for row in cur.fetchall()]

    resolved = 0
    pending = 0
    for row in rows:
        ts = float(row.get("ts") or 0.0)
        horizon_bars = int(row.get("horizon_bars") or 45)
        idx = bisect_left(candle_ts, ts)
        if idx >= len(candles) or idx + horizon_bars >= len(candles):
            pending += 1
            continue
        now_close = float(candles[idx].close or 0.0)
        fut_close = float(candles[idx + horizon_bars].close or now_close)
        forward_return = ((fut_close - now_close) / now_close) if now_close else 0.0
        resolved_label = 1 if cost_model.trade_is_viable(forward_return) else 0
        cur.execute(
            """
            UPDATE predictions
            SET future_label_status = 'resolved',
                resolved_label = ?,
                realized_forward_return = ?,
                realized_trade_pnl = COALESCE(realized_trade_pnl, ?)
            WHERE prediction_id = ?
            """,
            (resolved_label, forward_return, row.get("pnl"), str(row.get("prediction_id") or "")),
        )
        resolved += 1
    conn.commit()
    conn.close()
    return {"resolved": resolved, "pending": pending, "cost_round_trip_pct": cost_model.assumptions.round_trip_cost_pct}


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = resolve_predictions()
    (REPORTS_DIR / "live_prediction_logging_fix.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Live Prediction Logging Fix",
        "",
        f"- Resolved predictions: `{report['resolved']}`",
        f"- Pending predictions: `{report['pending']}`",
        f"- Cost round-trip pct: `{report['cost_round_trip_pct']:.6f}`",
    ]
    (REPORTS_DIR / "live_prediction_logging_fix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
