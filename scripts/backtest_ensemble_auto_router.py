from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.ensemble_auto_router import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    EnsembleAutoRouter,
    EnsemblePaperTrader,
    _normalise_side,
    decision_to_dict,
    load_config,
    summarize_trades,
)

DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "historical_unified_nifty_options_single" / "nifty_option_chain_historical_cost_aware_upstox_2024_2026_single.csv"
DEFAULT_TRADES_OUT = REPO_ROOT / "reports" / "ensemble_backtest_trades.csv"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "reports" / "ensemble_backtest_summary.json"


def _coerce_row(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = value
            continue
        try:
            out[key] = float(value)
        except Exception:
            out[key] = value
    if "ltp" not in out and "close" in out:
        out["ltp"] = out.get("close")
    if "strike_price" not in out and "strike" in out:
        out["strike_price"] = out.get("strike")
    if "option_type" in out:
        side = _normalise_side(out.get("option_type"))
        out["option_type"] = side
        out["option_type_ce"] = 1.0 if side == "CE" else 0.0
        out["option_type_pe"] = 1.0 if side == "PE" else 0.0
    return out


def _timestamp_key(row: Dict[str, Any]) -> str:
    for key in ("timestamp", "datetime", "time", "date"):
        if row.get(key):
            return str(row.get(key))
    return ""


def run_backtest(dataset: Path, config_path: Path, trades_out: Path, summary_out: Path, limit: int = 0) -> Dict[str, Any]:
    cfg = load_config(config_path)
    cfg["enabled"] = True
    cfg.setdefault("paper_trading", {})
    cfg["paper_trading"]["trade_log_path"] = str(trades_out)
    cfg["paper_trading"]["summary_path"] = str(summary_out)
    router = EnsembleAutoRouter(cfg, config_path=config_path)
    paper = EnsemblePaperTrader(router)
    paper.start()
    decisions: List[Dict[str, Any]] = []
    model_disagreements: List[float] = []

    with dataset.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = [_coerce_row(row) for row in reader]
    rows.sort(key=_timestamp_key)
    if limit > 0:
        rows = rows[:limit]

    for idx, row in enumerate(rows, start=1):
        if idx % 10000 == 0:
            print(f"[ENSEMBLE-BACKTEST] processed={idx}", flush=True)
        decision = paper.on_snapshot(row)
        decisions.append(decision_to_dict(decision))
        scores = [float(m.direction_score or 0.0) for m in decision.model_outputs if m.valid and m.direction_score is not None]
        if len(scores) >= 2:
            model_disagreements.append(max(abs(a - b) for i, a in enumerate(scores) for b in scores[i + 1 :]))

    if paper.open_position is not None:
        last_price = paper._price_for_position(rows[-1]) if rows else None
        if last_price is not None:
            paper._close_position(float(last_price), "BACKTEST_END")

    tpath, spath = paper.export(trades_path=trades_out, summary_path=summary_out)
    summary = summarize_trades(paper.trades, router.skipped_reason_counts, model_disagreements)
    summary.update(
        {
            "dataset": str(dataset),
            "rows_processed": len(rows),
            "trades_path": str(tpath),
            "summary_path": str(spath),
            "last_decision": decisions[-1] if decisions else None,
        }
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[ENSEMBLE-BACKTEST] trades={tpath}")
    print(f"[ENSEMBLE-BACKTEST] summary={summary_out}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the optional ensemble auto router.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Historical CSV path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Router config path")
    parser.add_argument("--trades-out", default=str(DEFAULT_TRADES_OUT), help="Trade CSV output path")
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT), help="Summary JSON output path")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests")
    args = parser.parse_args()
    dataset = Path(args.data)
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")
    run_backtest(dataset, Path(args.config), Path(args.trades_out), Path(args.summary_out), int(args.limit or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

