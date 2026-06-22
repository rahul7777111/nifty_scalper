from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
MODELS_DIR = REPO_ROOT / "models"
TRADES_DB = REPO_ROOT / "trades.db"
OBSERVER_DB = REPO_ROOT / "data" / "forward_edge_observer" / "forward_edge_observer.db"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cost_model import CostModel
from retrain_nifty_1year import write_json


MIN_ROWS = 300
MIN_POSITIVE = 50
MIN_POSITIVE_SHARE = 0.03
MIN_COVERAGE = 0.80


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def make_output_dir() -> Path:
    out_dir = MODELS_DIR / f"edge_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_sqlite_frame(db_path: Path, query: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def parse_json_column(value: Any) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def compute_session_bucket(ts: pd.Series) -> pd.Series:
    hours = ts.dt.hour * 60 + ts.dt.minute
    return np.where(hours < 10 * 60, "OPEN", np.where(hours >= 14 * 60 + 30, "CLOSE", "MID"))


def label_from_net(net: Optional[float]) -> Optional[int]:
    if net is None or pd.isna(net):
        return None
    return 1 if float(net) > 0 else 0


def ternary_from_net(net: Optional[float], cost_floor: float) -> Optional[str]:
    if net is None or pd.isna(net):
        return None
    if float(net) > max(cost_floor, 0.0):
        return "GOOD"
    if float(net) < -max(cost_floor, 0.0):
        return "BAD"
    return "NEUTRAL"


def load_forward_observer_dataset(cost_model: CostModel) -> pd.DataFrame:
    signals = load_sqlite_frame(OBSERVER_DB, "SELECT * FROM forward_signals ORDER BY timestamp ASC")
    outcomes = load_sqlite_frame(OBSERVER_DB, "SELECT * FROM forward_signal_outcomes ORDER BY observer_signal_id, horizon_minutes ASC")
    if signals.empty:
        return pd.DataFrame()

    signals["timestamp"] = pd.to_datetime(signals["timestamp"], errors="coerce")
    signals["trading_date"] = pd.to_datetime(signals["trading_date"], errors="coerce").dt.date
    signals["session_bucket"] = compute_session_bucket(signals["timestamp"])

    feature_records = []
    regime_records = []
    for row in signals.itertuples(index=False):
        features = parse_json_column(getattr(row, "feature_snapshot_json", None))
        regime = parse_json_column(getattr(row, "regime_features_json", None))
        drift = parse_json_column(getattr(row, "drift_sensitive_features_json", None))
        merged = {}
        merged.update(features)
        merged.update({f"regime_{k}": v for k, v in regime.items()})
        merged.update({f"drift_{k}": v for k, v in drift.items()})
        feature_records.append(merged)
        regime_records.append(regime)
    feature_df = pd.DataFrame(feature_records)
    signals = pd.concat([signals.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    wide = outcomes.pivot_table(
        index="observer_signal_id",
        columns="horizon_minutes",
        values=["option_ltp_at_horizon", "gross_pnl", "net_pnl_after_costs", "max_favorable_excursion_until_horizon", "max_adverse_excursion_until_horizon", "result_label"],
        aggfunc="first",
    )
    if not wide.empty:
        wide.columns = [f"{name}_{int(h)}m" for name, h in wide.columns]
        wide = wide.reset_index()
        signals = signals.merge(wide, on="observer_signal_id", how="left")

    for horizon in (5, 10, 15):
        net_col = f"net_pnl_after_costs_{horizon}m"
        gross_col = f"gross_pnl_{horizon}m"
        signals[f"simulated_net_pnl_{horizon}m"] = pd.to_numeric(signals.get(net_col), errors="coerce")
        signals[f"cost_adjusted_success_{horizon}m"] = signals[f"simulated_net_pnl_{horizon}m"].map(label_from_net)
        gross_series = pd.to_numeric(signals[gross_col], errors="coerce") if gross_col in signals.columns else pd.Series([np.nan] * len(signals), index=signals.index)
        signals[f"option_trade_success_{horizon}m"] = gross_series.map(label_from_net)
        floor = float(getattr(cost_model.assumptions, "minimum_net_edge_pct", 0.0) or 0.0)
        signals[f"ternary_trade_quality_{horizon}m"] = signals[f"simulated_net_pnl_{horizon}m"].map(lambda x: ternary_from_net(x, floor))

    rename_map = {
        "selected_option_symbol": "selected_option_symbol",
        "selected_option_token": "selected_option_token",
        "CE_or_PE": "CE_or_PE",
        "option_ltp_at_signal": "option_ltp_at_signal",
        "option_volume": "option_volume",
        "option_oi": "option_oi",
        "option_change_oi": "option_change_oi",
        "option_iv": "option_iv",
        "spot_price": "spot_price",
        "spot_open": "spot_open",
        "spot_high": "spot_high",
        "spot_low": "spot_low",
        "spot_close": "spot_close",
        "spot_volume": "spot_volume",
        "signal_reason": "entry_rule_that_fired",
        "predicted_probability": "previous_model_probability",
    }
    signals = signals.rename(columns=rename_map)
    return signals


def load_predictions_dataset() -> pd.DataFrame:
    preds = load_sqlite_frame(
        TRADES_DB,
        """
        SELECT prediction_id, timestamp, symbol, option_symbol, underlying_price, probability, predicted_class,
               confidence_bucket, label_policy_version, feature_snapshot_json, features_json, reason, regime, strategy_signal
        FROM predictions
        ORDER BY timestamp ASC
        """,
    )
    if preds.empty:
        return preds
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], errors="coerce")
    preds["trading_date"] = preds["timestamp"].dt.date
    preds["session_bucket"] = compute_session_bucket(preds["timestamp"])
    feature_records = []
    for row in preds.itertuples(index=False):
        features = parse_json_column(getattr(row, "features_json", None))
        if not features:
            features = parse_json_column(getattr(row, "feature_snapshot_json", None))
        feature_records.append(features)
    preds = pd.concat([preds.reset_index(drop=True), pd.DataFrame(feature_records).reset_index(drop=True)], axis=1)
    preds["entry_rule_that_fired"] = preds["reason"]
    preds["previous_model_probability"] = pd.to_numeric(preds["probability"], errors="coerce")
    return preds


def merge_sources(observer_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    if observer_df.empty and preds_df.empty:
        return pd.DataFrame()
    if observer_df.empty:
        return preds_df.copy()
    if preds_df.empty:
        return observer_df.copy()

    base = observer_df.copy()
    missing_pred_mask = base["prediction_id"].isna() | base["prediction_id"].astype(str).eq("")
    base.loc[missing_pred_mask, "prediction_id"] = None
    pred = preds_df.copy()
    rename_cols = {}
    for col in pred.columns:
        if col == "prediction_id":
            continue
        if col in base.columns:
            rename_cols[col] = f"pred_{col}"
    pred = pred.rename(columns=rename_cols)
    merged = base.merge(
        pred.drop_duplicates(subset=["prediction_id"]),
        on="prediction_id",
        how="left",
        suffixes=("", "_pred"),
    )
    for col in ["timestamp", "trading_date", "session_bucket", "entry_rule_that_fired", "previous_model_probability"]:
        pred_col = f"pred_{col}"
        if pred_col in merged.columns and col in merged.columns:
            merged[col] = merged[col].fillna(merged[pred_col])
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    return merged


def build_dataset_quality_report(df: pd.DataFrame, validity: Dict[str, Any]) -> str:
    lines = [
        "# Edge Dataset Quality Report",
        "",
        f"- Rows: `{len(df)}`",
        f"- Option symbol coverage: `{validity['coverage'].get('selected_option_symbol', 0.0):.2%}`",
        f"- Option LTP coverage: `{validity['coverage'].get('option_ltp_at_signal', 0.0):.2%}`",
        f"- Cost-adjusted 15m label coverage: `{validity['coverage'].get('cost_adjusted_success_15m', 0.0):.2%}`",
        f"- Dataset validity: `{validity['status']}`",
        "",
    ]
    if validity["failures"]:
        lines.append("## Failures")
        lines.append("")
        lines.extend([f"- {failure}" for failure in validity["failures"]])
    return "\n".join(lines) + "\n"


def build_missing_data_report(validity: Dict[str, Any]) -> str:
    lines = [
        "# Missing Data Report",
        "",
        "The dataset builder does not fake missing option-chain or outcome fields.",
        "",
        "## Coverage",
        "",
    ]
    for key, value in validity["coverage"].items():
        lines.append(f"- {key}: `{value:.2%}`")
    if validity["missing_core_fields"]:
        lines.append("")
        lines.append("## Core Missing Fields")
        lines.append("")
        lines.extend([f"- {field}" for field in validity["missing_core_fields"]])
    return "\n".join(lines) + "\n"


def dataset_validity(df: pd.DataFrame) -> Dict[str, Any]:
    coverage: Dict[str, float] = {}
    for field in ["selected_option_symbol", "option_ltp_at_signal", "cost_adjusted_success_5m", "cost_adjusted_success_10m", "cost_adjusted_success_15m"]:
        if field not in df.columns or df.empty:
            coverage[field] = 0.0
        else:
            series = df[field]
            if series.dtype == object:
                coverage[field] = float(series.fillna("").astype(str).ne("").mean())
            else:
                coverage[field] = float(series.notna().mean())

    # FIX 6: accept cost-aware label set with fallback when
    # cost_adjusted_success_15m is absent (which is the case for the
    # canonical nifty_option_chain_cost_aware_edge_dataset_*.csv).
    # We pick the first label that is present in the dataframe AND has
    # >= 80% coverage, scanning the candidates in priority order.
    label_candidates = [
        "cost_adjusted_success_15m",
        "profitable_trade_label",
        "strong_profitable_trade_label",
        "cost_survivor_label",
        "strong_profitable_trade_label_v2",
        "cost_survivor_label_v2",
    ]
    chosen_label: Optional[str] = None
    label_coverage: float = 0.0
    for candidate in label_candidates:
        if candidate not in df.columns:
            continue
        if df.empty:
            continue
        series = df[candidate]
        if series.dtype == object:
            cov = float(series.fillna("").astype(str).ne("").mean())
        else:
            cov = float(series.notna().mean())
        if cov >= MIN_COVERAGE:
            chosen_label = candidate
            label_coverage = cov
            break
    positives = int(pd.to_numeric(df.get(chosen_label), errors="coerce").fillna(0).eq(1).sum()) if chosen_label else 0
    usable_rows = int(len(df))
    positive_share = float(positives / usable_rows) if usable_rows else 0.0
    # Missing-field check: the strict option-chain columns are no longer
    # required because the cost-aware datasets are derived from a
    # different source. We only require timestamp + a price proxy + a label.
    has_timestamp = "timestamp" in df.columns and not df["timestamp"].isna().all()
    has_price_proxy = any(field in df.columns and not df[field].isna().all() for field in ("option_ltp_at_signal", "ltp", "mid_price"))
    has_label = chosen_label is not None
    missing_core_fields: List[str] = []
    if not has_timestamp:
        missing_core_fields.append("timestamp")
    if not has_price_proxy:
        missing_core_fields.append("option_ltp_at_signal|ltp|mid_price")
    if not has_label:
        missing_core_fields.append("binary_label(any of: " + ", ".join(label_candidates) + ")")
    failures: List[str] = []
    if usable_rows < MIN_ROWS:
        failures.append(f"usable_rows < {MIN_ROWS}")
    if positives < MIN_POSITIVE:
        failures.append(f"positive_labels_for_{chosen_label} < {MIN_POSITIVE}")
    if positive_share < MIN_POSITIVE_SHARE:
        failures.append(f"positive_share_for_{chosen_label} < {MIN_POSITIVE_SHARE:.0%}")
    if chosen_label is None:
        failures.append(
            "no_usable_label_column: none of "
            + ", ".join(label_candidates)
            + f" reached {MIN_COVERAGE:.0%} coverage"
        )
    if coverage.get("selected_option_symbol", 0.0) < MIN_COVERAGE:
        failures.append("option_symbol coverage below 80%")

    return {
        "chosen_binary_label": chosen_label,
        "chosen_label": chosen_label,
        "label_coverage": label_coverage,
        "usable_rows": usable_rows,
        "positive_labels": positives,
        "positive_share": positive_share,
        "coverage": coverage,
        "missing_core_fields": missing_core_fields,
        "leakage_check": "No future outcome columns are intended for model features; enforced again in retraining script.",
        "status": "DATASET_NOT_READY_FOR_EDGE_RETRAINING" if failures else "DATASET_READY_FOR_RESEARCH_RETRAINING",
        "failures": failures,
    }


def save_dataset(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    paths: Dict[str, Any] = {}
    parquet_path = out_dir / "edge_dataset.parquet"
    csv_path = out_dir / "edge_dataset.csv"
    try:
        df.to_parquet(parquet_path, index=False)
        paths["parquet"] = str(parquet_path)
    except Exception:
        paths["parquet"] = None
    df.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)
    schema = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [{"name": str(col), "dtype": str(df[col].dtype)} for col in df.columns],
    }
    write_json(out_dir / "edge_dataset_schema.json", ensure_serializable(schema))
    return paths


def label_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for col in [c for c in df.columns if c.startswith(("option_trade_success_", "cost_adjusted_success_", "ternary_trade_quality_"))]:
        series = df[col]
        report[col] = series.fillna("MISSING").astype(str).value_counts(dropna=False).to_dict()
    return report


def main() -> None:
    out_dir = make_output_dir()
    cost_model = CostModel()
    observer_df = load_forward_observer_dataset(cost_model)
    preds_df = load_predictions_dataset()
    dataset = merge_sources(observer_df, preds_df)
    validity = dataset_validity(dataset)
    save_paths = save_dataset(dataset, out_dir)
    write_json(out_dir / "label_distribution_report.json", ensure_serializable(label_distribution(dataset)))
    (out_dir / "edge_dataset_quality_report.md").write_text(build_dataset_quality_report(dataset, validity), encoding="utf-8")
    (out_dir / "missing_data_report.md").write_text(build_missing_data_report(validity), encoding="utf-8")
    write_json(out_dir / "dataset_validity_report.json", ensure_serializable(validity))

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "usable_rows": int(len(dataset)),
                "option_chain_features_found": bool(validity["coverage"].get("selected_option_symbol", 0.0) >= MIN_COVERAGE),
                "actual_option_outcome_labels_found": bool(validity["coverage"].get("cost_adjusted_success_15m", 0.0) >= MIN_COVERAGE),
                "dataset_validity_result": validity["status"],
                "csv_path": save_paths["csv"],
                "parquet_path": save_paths["parquet"],
                "reason_if_stopped": "; ".join(validity["failures"]) if validity["failures"] else "",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
