#!/usr/bin/env python3
"""Backtest a single ensemble artifact with vectorized scoring and candidate-style metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_ml_models_from_csv import run_historical_ml_backtest  # noqa: E402


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _artifact_candidate_id(path: Path) -> str:
    if path.is_dir():
        return path.name
    if path.suffix.lower() in {".pkl", ".joblib"}:
        return path.stem
    return path.name


def _infer_candidate_id(summary: Dict[str, Any], artifact_path: Path) -> str:
    sources = list(summary.get("signal_sources_considered") or [])
    candidates = list(summary.get("candidates_used") or [])
    if len(candidates) == 1:
        return str(candidates[0])
    if len(sources) == 1:
        return str(sources[0])
    best = str(summary.get("best_candidate") or "").strip()
    if best and best.lower() != "n/a":
        return best
    return _artifact_candidate_id(artifact_path)


def _trade_days(trades: Iterable[Dict[str, Any]]) -> int:
    days = set()
    for trade in trades:
        stamp = str(trade.get("entry_ts") or "")
        if stamp:
            days.add(stamp[:10])
    return len(days)


def build_candidate_metrics(
    *,
    result_summary: Dict[str, Any],
    trades: List[Dict[str, Any]],
    artifact_path: Path,
) -> Dict[str, Any]:
    total_trades = int(result_summary.get("total_trades") or 0)
    trade_days = _trade_days(trades)
    candidate_id = _infer_candidate_id(result_summary, artifact_path)
    threshold = _safe_float((result_summary.get("parameters") or {}).get("fallback_threshold"), 0.0)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path.resolve()),
        "candidate_id": candidate_id,
        "model_family": "xgb_rf_ensemble",
        "status": str(result_summary.get("status") or ""),
        "selected_threshold": threshold,
        "total_trades": total_trades,
        "wins": int(result_summary.get("wins") or 0),
        "losses": int(result_summary.get("losses") or 0),
        "win_rate": _safe_float(result_summary.get("win_rate"), 0.0),
        "profit_factor": result_summary.get("profit_factor"),
        "net_pnl": _safe_float(result_summary.get("net_pnl"), 0.0),
        "gross_pnl": _safe_float(result_summary.get("gross_pnl"), 0.0),
        "avg_trade": _safe_float(result_summary.get("avg_trade"), 0.0),
        "max_drawdown": _safe_float(result_summary.get("max_drawdown"), 0.0),
        "max_drawdown_pct": _safe_float(result_summary.get("max_drawdown_pct"), 0.0),
        "sharpe_like": _safe_float(result_summary.get("sharpe_like"), 0.0),
        "rows_processed": int(result_summary.get("rows_processed") or 0),
        "rows_above_threshold": int(result_summary.get("rows_above_threshold") or 0),
        "prediction_prefilter_rows": int(result_summary.get("prediction_prefilter_rows") or 0),
        "trade_days": trade_days,
        "trades_per_day": round(total_trades / trade_days, 4) if trade_days > 0 else 0.0,
        "timing_seconds": dict(result_summary.get("timing_seconds") or {}),
        "artifact_loading_failures": list(result_summary.get("artifact_loading_failures") or []),
        "feature_alignment": list(result_summary.get("feature_alignment") or []),
        "backtest_outputs": {
            "trades_csv": (result_summary.get("output_paths") or {}).get("trades_csv"),
            "summary_json": (result_summary.get("output_paths") or {}).get("summary_json"),
            "report_md": (result_summary.get("output_paths") or {}).get("report_md"),
        },
    }
    return metrics


def render_candidate_metrics_md(metrics: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Ensemble Backtest Candidate Summary",
            "",
            f"- artifact_path: `{metrics['artifact_path']}`",
            f"- candidate_id: `{metrics['candidate_id']}`",
            f"- model_family: `{metrics['model_family']}`",
            f"- status: `{metrics['status']}`",
            f"- selected_threshold: {metrics['selected_threshold']:.4f}",
            f"- total_trades: {metrics['total_trades']}",
            f"- wins: {metrics['wins']}",
            f"- losses: {metrics['losses']}",
            f"- win_rate: {metrics['win_rate']:.4f}",
            f"- profit_factor: {metrics['profit_factor']}",
            f"- net_pnl: {metrics['net_pnl']:.2f}",
            f"- gross_pnl: {metrics['gross_pnl']:.2f}",
            f"- avg_trade: {metrics['avg_trade']:.2f}",
            f"- max_drawdown: {metrics['max_drawdown']:.2f}",
            f"- max_drawdown_pct: {metrics['max_drawdown_pct']:.4f}",
            f"- sharpe_like: {metrics['sharpe_like']:.4f}",
            f"- trade_days: {metrics['trade_days']}",
            f"- trades_per_day: {metrics['trades_per_day']:.4f}",
            f"- rows_processed: {metrics['rows_processed']}",
            f"- rows_above_threshold: {metrics['rows_above_threshold']}",
            f"- prediction_prefilter_rows: {metrics['prediction_prefilter_rows']}",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest a single xgb_rf_ensemble artifact over historical options data.")
    parser.add_argument("--dataset", dest="csv", default=None, help="Historical options CSV or Parquet path")
    parser.add_argument("--csv", dest="csv_legacy", default=None, help="Historical options CSV or Parquet path")
    parser.add_argument("--model-dir", dest="artifact", default=None, help="Artifact .pkl/.joblib path or candidate artifact directory")
    parser.add_argument("--artifact", dest="artifact_legacy", default=None, help="Artifact .pkl/.joblib path or candidate artifact directory")
    parser.add_argument("--target", default="profitable_trade_label")
    parser.add_argument("--output-dir", default="reports/ensemble_backtests", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fallback threshold when artifact metadata does not provide one")
    parser.add_argument("--xgb-min-prob", type=float, default=0.58)
    parser.add_argument("--rf-min-prob", type=float, default=0.52)
    parser.add_argument("--max-model-disagreement", type=float, default=0.25)
    parser.add_argument("--vectorized-backtest", action="store_true", default=True)
    parser.add_argument("--max-backtest-rows", type=int, default=None)
    parser.add_argument("--cost-per-trade", type=float, default=0.0)
    parser.add_argument("--lot-size", type=int, default=65)
    parser.add_argument("--target-pct", type=float, default=0.20)
    parser.add_argument("--stoploss-pct", type=float, default=0.10)
    parser.add_argument("--max-hold-bars", type=int, default=5)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--fast-mode", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--multiprocessing", action="store_true", default=False)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args(argv)

    csv_path = args.csv or args.csv_legacy
    artifact_value = args.artifact or args.artifact_legacy
    if not csv_path:
        raise SystemExit("Missing required --dataset")
    if not artifact_value:
        raise SystemExit("Missing required --model-dir")

    artifact_path = Path(artifact_value).resolve()
    if not artifact_path.exists():
        raise SystemExit(f"Artifact not found: {artifact_path}")

    result = run_historical_ml_backtest(
        csv_path=csv_path,
        candidate_config_path=str(artifact_path),
        output_dir=args.output_dir,
        threshold=args.threshold,
        target_pct=args.target_pct,
        stoploss_pct=args.stoploss_pct,
        max_hold_bars=args.max_hold_bars,
        max_trades_per_day=args.max_trades_per_day,
        debug=args.debug,
        use_candidate_thresholds=True,
        strict_artifacts=False,
        allow_artifact_fallback=True,
        ml_artifact_scoring=True,
        allow_embedded_score_fallback=False,
        fast_mode=args.fast_mode,
        date_from=args.date_from,
        date_to=args.date_to,
        max_rows=args.max_backtest_rows if args.max_backtest_rows is not None else args.max_rows,
        max_candidates=1,
        multiprocessing=args.multiprocessing,
        progress_every=args.progress_every,
    )

    result.summary["output_paths"] = dict(result.output_paths or {})
    candidate_metrics = build_candidate_metrics(
        result_summary=result.summary,
        trades=result.trades,
        artifact_path=artifact_path,
    )

    output_dir = Path(result.output_paths.get("output_dir") or args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics_json = output_dir / f"ensemble_candidate_metrics_{stamp}.json"
    metrics_md = output_dir / f"ensemble_candidate_metrics_{stamp}.md"
    metrics_json.write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")
    metrics_md.write_text(render_candidate_metrics_md(candidate_metrics), encoding="utf-8")

    print(json.dumps(
        {
            "status": candidate_metrics["status"],
            "candidate_id": candidate_metrics["candidate_id"],
            "artifact_path": candidate_metrics["artifact_path"],
            "summary_json": result.output_paths.get("summary_json"),
            "trades_csv": result.output_paths.get("trades_csv"),
            "report_md": result.output_paths.get("report_md"),
            "candidate_metrics_json": str(metrics_json),
            "candidate_metrics_md": str(metrics_md),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
