from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "forward_edge_observer"
DB_PATH = DATA_DIR / "forward_edge_observer.db"
MODELS_DIR = REPO_ROOT / "models" / "forward_edge_reports"


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ensure_serializable(payload), indent=2), encoding="utf-8")


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB_PATH)
    try:
        signals = pd.read_sql_query("SELECT * FROM forward_signals ORDER BY timestamp ASC", conn)
        outcomes = pd.read_sql_query("SELECT * FROM forward_signal_outcomes ORDER BY observer_signal_id, horizon_minutes ASC", conn)
    finally:
        conn.close()
    if not signals.empty:
        signals["timestamp"] = pd.to_datetime(signals["timestamp"], errors="coerce")
        signals["trading_date"] = pd.to_datetime(signals["trading_date"], errors="coerce")
        signals["predicted_probability"] = pd.to_numeric(signals["predicted_probability"], errors="coerce")
        signals["simulated_net_pnl_after_costs"] = pd.to_numeric(signals["simulated_net_pnl_after_costs"], errors="coerce")
        signals["risk_gate_would_pass"] = pd.to_numeric(signals["risk_gate_would_pass"], errors="coerce").fillna(0)
    if not outcomes.empty:
        outcomes["net_pnl_after_costs"] = pd.to_numeric(outcomes["net_pnl_after_costs"], errors="coerce")
        outcomes["gross_pnl"] = pd.to_numeric(outcomes["gross_pnl"], errors="coerce")
    return signals, outcomes


def probability_bucket(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    lower = (vals * 10).floordiv(1) / 10.0
    upper = (lower + 0.1).clip(upper=1.0)
    return lower.map(lambda x: f"{x:.1f}-{min(1.0, x+0.1):.1f}")


def classify(signals: pd.DataFrame, outcomes: pd.DataFrame) -> str:
    if signals.empty or len(signals) < 30:
        return "INSUFFICIENT_DATA"
    weeks = signals["timestamp"].dt.to_period("W").astype(str)
    if weeks.nunique() < 3:
        return "INSUFFICIENT_DATA"
    merged = outcomes.merge(signals[["observer_signal_id", "predicted_probability"]], on="observer_signal_id", how="left")
    complete = merged[merged["status"] == "complete"].copy()
    if complete.empty:
        return "INSUFFICIENT_DATA"
    pf = complete.loc[complete["net_pnl_after_costs"] > 0, "net_pnl_after_costs"].sum() / max(
        abs(complete.loc[complete["net_pnl_after_costs"] < 0, "net_pnl_after_costs"].sum()),
        1e-9,
    )
    avg_net = float(complete["net_pnl_after_costs"].mean())
    complete["prob_bucket"] = probability_bucket(complete["predicted_probability"])
    bucket_perf = complete.groupby("prob_bucket")["net_pnl_after_costs"].mean().to_dict()
    low_bucket = min(bucket_perf.values()) if bucket_perf else -1.0
    high_bucket = max(bucket_perf.values()) if bucket_perf else -1.0
    coverage = float(signals["selected_option_symbol"].fillna("").astype(str).ne("").mean())
    stale_ok = float(pd.to_numeric(signals["quote_staleness_seconds"], errors="coerce").fillna(9999).le(60).mean())
    complete["week"] = pd.to_datetime(complete["observed_at"], errors="coerce").dt.to_period("W").astype(str)
    weekly_wins = complete.loc[complete["net_pnl_after_costs"] > 0].groupby("week").size()
    if not weekly_wins.empty and weekly_wins.max() / max(weekly_wins.sum(), 1) > 0.5:
        return "WATCHLIST_ONLY"
    if pf > 1.15 and avg_net > 0 and high_bucket > low_bucket and coverage >= 0.5 and stale_ok >= 0.7:
        return "READY_FOR_MANUAL_REVIEW"
    return "NO_EDGE_DETECTED"


def markdown_recommendation(classification: str, signals: pd.DataFrame, outcomes: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Final Forward Edge Recommendation",
            "",
            f"Classification: `{classification}`",
            "",
            "This report must not be treated as a live-trading promotion.",
            "",
            f"- Signals captured: `{len(signals)}`",
            f"- Outcome rows: `{len(outcomes)}`",
            f"- Separate trading weeks with signals: `{signals['timestamp'].dt.to_period('W').nunique() if not signals.empty else 0}`",
            "",
            "Allowed interpretations:",
            "",
            "- `INSUFFICIENT_DATA`",
            "- `NO_EDGE_DETECTED`",
            "- `WATCHLIST_ONLY`",
            "- `READY_FOR_MANUAL_REVIEW`",
        ]
    )


def main() -> None:
    out_dir = MODELS_DIR / f"forward_edge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    signals, outcomes = load_frames()

    signal_count_summary = {
        "signals_total": int(len(signals)),
        "observer_runs": int(signals["observer_run_id"].nunique()) if not signals.empty else 0,
        "days_with_signals": int(signals["trading_date"].dt.date.nunique()) if not signals.empty else 0,
        "weeks_with_signals": int(signals["timestamp"].dt.to_period("W").nunique()) if not signals.empty else 0,
    }
    option_feature_coverage = {}
    for field in ["selected_option_symbol", "option_ltp_at_signal", "bid", "ask", "option_volume", "option_oi", "option_iv"]:
        option_feature_coverage[field] = float(signals[field].notna().mean()) if (not signals.empty and field in signals.columns) else 0.0

    outcome_label_distribution = {}
    if not outcomes.empty:
        outcome_label_distribution = Counter(outcomes["result_label"].fillna("NO_DATA").astype(str)).copy()

    merged = outcomes.merge(
        signals[["observer_signal_id", "predicted_probability", "session_bucket", "regime_features_json", "quote_staleness_seconds"]],
        on="observer_signal_id",
        how="left",
    ) if not outcomes.empty and not signals.empty else pd.DataFrame()

    if not merged.empty:
        merged["prob_bucket"] = probability_bucket(merged["predicted_probability"])
        precision_by_probability_bucket = (
            merged.assign(win=merged["net_pnl_after_costs"].fillna(0).gt(0).astype(int))
            .groupby("prob_bucket")["win"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "precision"})
            .reset_index()
            .to_dict(orient="records")
        )
        pnl_by_probability_bucket = (
            merged.groupby("prob_bucket")["net_pnl_after_costs"]
            .agg(["count", "mean", "sum"])
            .reset_index()
            .rename(columns={"mean": "avg_net_pnl", "sum": "total_net_pnl"})
            .to_dict(orient="records")
        )
        precision_by_session_bucket = (
            merged.assign(win=merged["net_pnl_after_costs"].fillna(0).gt(0).astype(int))
            .groupby("session_bucket")["win"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "precision"})
            .reset_index()
            .to_dict(orient="records")
        )
        pnl_by_session_bucket = (
            merged.groupby("session_bucket")["net_pnl_after_costs"]
            .agg(["count", "mean", "sum"])
            .reset_index()
            .rename(columns={"mean": "avg_net_pnl", "sum": "total_net_pnl"})
            .to_dict(orient="records")
        )
    else:
        precision_by_probability_bucket = []
        pnl_by_probability_bucket = []
        precision_by_session_bucket = []
        pnl_by_session_bucket = []

    regime_rows: Dict[str, List[float]] = defaultdict(list)
    if not signals.empty and "regime_features_json" in signals.columns:
        for row in signals.itertuples(index=False):
            try:
                regime = json.loads(getattr(row, "regime_features_json") or "{}")
            except Exception:
                regime = {}
            label = "quiet"
            if regime.get("regime_volatile"):
                label = "volatile"
            elif regime.get("regime_trending"):
                label = "trending"
            elif regime.get("regime_mean_reverting"):
                label = "mean_reverting"
            regime_rows[label].append(float(getattr(row, "predicted_probability") or 0.0))
    precision_by_regime = [{"regime": k, "signal_count": len(v), "avg_probability": float(sum(v) / max(len(v), 1))} for k, v in regime_rows.items()]
    pnl_by_regime = [{"regime": row["regime"], "total_net_pnl": None, "avg_net_pnl": None} for row in precision_by_regime]

    weekly_stability_report = {}
    if not signals.empty:
        weekly = signals.copy()
        weekly["week"] = weekly["timestamp"].dt.to_period("W").astype(str)
        weekly_stability_report = (
            weekly.groupby("week")["predicted_probability"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .rename(columns={"count": "signal_count", "mean": "avg_probability", "std": "prob_std"})
            .to_dict(orient="records")
        )

    cost_adjusted_paper_pf_report = {}
    if not outcomes.empty:
        positive = outcomes.loc[pd.to_numeric(outcomes["net_pnl_after_costs"], errors="coerce") > 0, "net_pnl_after_costs"].sum()
        negative = outcomes.loc[pd.to_numeric(outcomes["net_pnl_after_costs"], errors="coerce") < 0, "net_pnl_after_costs"].sum()
        cost_adjusted_paper_pf_report = {
            "profit_factor": float(positive / max(abs(negative), 1e-9)),
            "average_net_pnl_after_costs": float(pd.to_numeric(outcomes["net_pnl_after_costs"], errors="coerce").mean()) if not outcomes.empty else None,
        }

    data_quality_report = {
        "quote_staleness_acceptable_share": float(pd.to_numeric(signals.get("quote_staleness_seconds"), errors="coerce").fillna(9999).le(60).mean()) if not signals.empty else 0.0,
        "missing_option_symbol_share": float(signals["selected_option_symbol"].fillna("").astype(str).eq("").mean()) if not signals.empty else 1.0,
        "observer_error_rows": int(signals["observer_error"].fillna("").astype(str).ne("").sum()) if not signals.empty else 0,
    }

    classification = classify(signals, outcomes)
    final_md = markdown_recommendation(classification, signals, outcomes)

    write_json(out_dir / "signal_count_summary.json", signal_count_summary)
    write_json(out_dir / "option_feature_coverage.json", option_feature_coverage)
    write_json(out_dir / "outcome_label_distribution.json", outcome_label_distribution)
    write_json(out_dir / "precision_by_probability_bucket.json", precision_by_probability_bucket)
    write_json(out_dir / "pnl_by_probability_bucket.json", pnl_by_probability_bucket)
    write_json(out_dir / "precision_by_regime.json", precision_by_regime)
    write_json(out_dir / "pnl_by_regime.json", pnl_by_regime)
    write_json(out_dir / "precision_by_session_bucket.json", precision_by_session_bucket)
    write_json(out_dir / "pnl_by_session_bucket.json", pnl_by_session_bucket)
    write_json(out_dir / "weekly_stability_report.json", weekly_stability_report)
    write_json(out_dir / "cost_adjusted_paper_pf_report.json", cost_adjusted_paper_pf_report)
    write_json(out_dir / "data_quality_report.json", data_quality_report)
    (out_dir / "final_forward_edge_recommendation.md").write_text(final_md, encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir), "classification": classification, "signals": int(len(signals))}, indent=2))


if __name__ == "__main__":
    main()
