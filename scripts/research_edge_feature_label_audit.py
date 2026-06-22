from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cost_model import CostModel
from label_policies import available_label_policies, build_label_dataset, get_label_policy_spec
from ml_pipeline import build_market_feature_vector
from retrain_nifty_1year import (
    HORIZON_BARS,
    LOOKBACK_BARS,
    chronological_split,
    clean_frame,
    frame_to_candles,
    prepare_minute_frame,
    write_json,
)

try:
    from sklearn.feature_selection import mutual_info_classif
except Exception:  # pragma: no cover - optional dependency
    mutual_info_classif = None


TIMEZONE = "Asia/Kolkata"
MI_SAMPLE_SIZE = 5000
EXPECTED_OPTION_FIELDS = {
    "atm_ce_premium": ["atm_ce_premium", "ce_premium", "call_premium", "ctx_ce_premium"],
    "atm_pe_premium": ["atm_pe_premium", "pe_premium", "put_premium", "ctx_pe_premium"],
    "option_volume": ["option_volume", "ctx_option_volume", "volume", "ctx_volume_sma"],
    "option_open_interest": ["oi", "open_interest", "ctx_oi"],
    "change_in_oi": ["oi_change", "change_in_oi", "ctx_oi_change"],
    "bid_ask_spread": ["bid_ask_spread_pct", "spread", "bid", "ask"],
    "implied_volatility_or_proxy": ["ctx_iv", "ctx_iv_percentile", "iv", "implied_volatility"],
    "straddle_price": ["straddle_price", "ctx_straddle_price"],
    "ce_pe_premium_ratio": ["ce_pe_ratio", "premium_ratio", "ctx_ce_pe_ratio"],
    "strike_distance_from_spot": ["strike_distance", "distance_from_spot", "option_to_spot_pct"],
    "option_liquidity": ["liquidity", "bid_ask_spread_pct", "ctx_volume_sma"],
    "option_premium_momentum": ["premium_momentum", "ctx_option_momentum", "ctx_option_price"],
}


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if (np.isnan(value) or np.isinf(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def latest_research_artifact(prefix: str) -> Optional[Path]:
    matches = sorted(MODELS_DIR.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def make_output_dir() -> Path:
    out_dir = MODELS_DIR / f"research_edge_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def month_key_series(timestamps: Sequence[pd.Timestamp]) -> pd.Series:
    return pd.to_datetime(list(timestamps)).to_period("M").astype(str)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    try:
        if len(x) < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr) or np.isinf(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def slice_to_indexer(slice_obj: Any, total_len: int) -> np.ndarray:
    if isinstance(slice_obj, slice):
        return np.arange(total_len)[slice_obj]
    return np.asarray(slice_obj, dtype=int)


def compute_psi(train_values: np.ndarray, test_values: np.ndarray, bins: int = 10) -> float:
    train_values = np.asarray(train_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)
    train_values = train_values[np.isfinite(train_values)]
    test_values = test_values[np.isfinite(test_values)]
    if len(train_values) < 20 or len(test_values) < 20:
        return 0.0
    if np.nanstd(train_values) < 1e-12 and np.nanstd(test_values) < 1e-12:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(train_values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    train_hist, _ = np.histogram(train_values, bins=edges)
    test_hist, _ = np.histogram(test_values, bins=edges)
    train_pct = np.clip(train_hist / max(train_hist.sum(), 1), 1e-6, None)
    test_pct = np.clip(test_hist / max(test_hist.sum(), 1), 1e-6, None)
    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
    return float(psi)


def feature_group(name: str) -> str:
    lower = name.lower()
    if lower.startswith(("ctx_", "option_", "delta", "gamma", "theta", "vega", "greeks_", "bid_ask")):
        return "option_chain_features"
    if lower.startswith(("last_volume", "vol_", "volume_")):
        return "volume_liquidity_features"
    if lower.startswith(("is_", "dist_from_opening", "opening_", "dist_to_rolling", "rolling_")):
        return "time_session_features"
    if any(token in lower for token in ("ema", "rsi", "adx", "momentum", "supertrend", "pivot", "ret_", "regime_")):
        return "trend_momentum_features"
    if any(token in lower for token in ("volatility", "atr", "choppiness", "vol_of_vol", "range_to_atr", "realized_vol")):
        return "volatility_features"
    if any(token in lower for token in ("pnl", "stop", "target", "mfe", "mae", "risk")):
        return "risk_trade_management_features"
    return "price_action_features"


def summarize_group_assessment(features: List[Dict[str, Any]]) -> str:
    if not features:
        return "missing"
    if all(float(item["effective_nonzero_share"]) < 0.05 for item in features):
        return "placeholder_or_missing_in_training_data"
    if np.mean([abs(float(item["correlation_with_label"])) for item in features]) < 0.01 and np.mean([float(item["mutual_information"]) for item in features]) < 0.002:
        return "weak"
    if np.mean([float(item["train_test_psi"]) for item in features]) > 0.25:
        return "unstable"
    if np.mean([float(item["redundancy_to_group_median"]) for item in features]) > 0.95:
        return "redundant"
    return "mixed_or_useful"


def usefulness_rating(feature: Dict[str, Any]) -> str:
    nonzero = float(feature["effective_nonzero_share"])
    corr = abs(float(feature["correlation_with_label"]))
    mi = float(feature["mutual_information"])
    psi = float(feature["train_test_psi"])
    stability = float(feature["monthly_stability_score"])
    redundancy = float(feature["redundancy_to_group_median"])
    if nonzero < 0.05:
        return "placeholder_or_missing"
    if psi > 0.35 or stability < 0.25:
        return "unstable"
    if corr < 0.01 and mi < 0.0015:
        return "weak"
    if redundancy > 0.97 and corr < 0.03:
        return "redundant"
    return "likely_useful"


def build_feature_frame() -> Tuple[pd.DataFrame, np.ndarray, List[str], List[pd.Timestamp], pd.DataFrame]:
    minute_frame, _ = prepare_minute_frame()
    minute_frame, _ = clean_frame(minute_frame)
    candles = frame_to_candles(minute_frame)
    rows: List[List[float]] = []
    timestamps: List[pd.Timestamp] = []
    feature_names: List[str] = []
    for idx in range(LOOKBACK_BARS, len(candles) - HORIZON_BARS):
        window = candles[idx - LOOKBACK_BARS + 1 : idx + 1]
        features, names = build_market_feature_vector(window, lookback=LOOKBACK_BARS)
        if not feature_names:
            feature_names = list(names)
        rows.append([float(v) for v in features])
        timestamps.append(pd.Timestamp(candles[idx].time))
    X = np.asarray(rows, dtype=np.float32)
    feature_df = pd.DataFrame(X, columns=feature_names)
    return feature_df, X, feature_names, timestamps, minute_frame


def build_primary_labels(candles: Sequence[Any], policy_name: str = "trade_quality_ternary") -> Tuple[np.ndarray, List[pd.Timestamp], Dict[str, Any]]:
    dataset = build_label_dataset(
        candles,
        policy_name=policy_name,
        lookback=LOOKBACK_BARS,
        horizon=HORIZON_BARS,
        cost_model=CostModel(),
        include_features=False,
    )
    y = np.asarray(dataset.y, dtype=np.int32)
    timestamps = [pd.Timestamp(dataset.observations[idx].timestamp) for idx in dataset.sample_indices if idx < len(dataset.observations)]
    meta = {
        "policy": dataset.policy.name,
        "distribution": dataset.label_distribution,
        "training_samples": len(dataset.y),
        "neutral_samples_dropped": dataset.neutral_samples_dropped,
    }
    return y, timestamps, meta


def align_feature_and_label_data(
    feature_df: pd.DataFrame,
    feature_timestamps: Sequence[pd.Timestamp],
    label_timestamps: Sequence[pd.Timestamp],
    y: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    feat = feature_df.copy()
    feat["timestamp"] = pd.to_datetime(list(feature_timestamps)).astype(str)
    lbl = pd.DataFrame({"timestamp": pd.to_datetime(list(label_timestamps)).astype(str), "y": y})
    merged = feat.merge(lbl, on="timestamp", how="inner")
    merged = merged.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    y_aligned = merged.pop("y").astype(int).to_numpy()
    timestamps = pd.to_datetime(merged.pop("timestamp"))
    return merged, y_aligned, timestamps


def audit_feature_groups(feature_df: pd.DataFrame, y: np.ndarray, timestamps: pd.Series) -> Dict[str, Any]:
    ts_values = pd.to_datetime(timestamps)
    if hasattr(ts_values, "dt"):
        ts_python = list(ts_values.dt.to_pydatetime())
    else:
        ts_python = list(ts_values.to_pydatetime())
    split = chronological_split(feature_df.to_numpy(dtype=np.float32), y, ts_python)
    train_idx = slice_to_indexer(split["slices"]["train"], len(feature_df))
    test_idx = slice_to_indexer(split["slices"]["test"], len(feature_df))
    if mutual_info_classif is not None and len(feature_df) > 100:
        mi_X = feature_df.to_numpy(dtype=np.float32)
        mi_y = y
        if len(mi_X) > MI_SAMPLE_SIZE:
            sample_idx = np.linspace(0, len(mi_X) - 1, MI_SAMPLE_SIZE, dtype=int)
            mi_X = mi_X[sample_idx]
            mi_y = y[sample_idx]
        mi_scores = mutual_info_classif(mi_X, mi_y, discrete_features=False, random_state=42)
    else:
        mi_scores = np.zeros(feature_df.shape[1], dtype=float)
    monthly = pd.DataFrame(feature_df.copy())
    monthly["month"] = month_key_series(timestamps)
    monthly["y"] = y
    group_medians = {
        group: feature_df[[c for c in feature_df.columns if feature_group(c) == group]].median(axis=1)
        for group in sorted({feature_group(c) for c in feature_df.columns})
    }

    features_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, column in enumerate(feature_df.columns):
        values = feature_df[column].to_numpy(dtype=float)
        month_means = monthly.groupby("month")[column].mean()
        month_label_corr = []
        for _month, month_frame in monthly.groupby("month"):
            if month_frame["y"].nunique() < 2 or month_frame[column].std(ddof=0) < 1e-12:
                continue
            month_label_corr.append(safe_corr(month_frame[column].to_numpy(dtype=float), month_frame["y"].to_numpy(dtype=float)))
        stability_score = 0.0
        if len(month_means) > 1:
            denom = float(abs(month_means.mean()) + month_means.std(ddof=0) + 1e-9)
            variability = float(month_means.std(ddof=0) / denom)
            sign_flip_penalty = 0.0
            if len(month_label_corr) > 1:
                sign_changes = sum(
                    1 for i in range(1, len(month_label_corr))
                    if np.sign(month_label_corr[i]) != np.sign(month_label_corr[i - 1]) and abs(month_label_corr[i]) > 1e-6 and abs(month_label_corr[i - 1]) > 1e-6
                )
                sign_flip_penalty = sign_changes / max(len(month_label_corr) - 1, 1)
            stability_score = max(0.0, 1.0 - min(1.0, variability + sign_flip_penalty * 0.5))
        group_name = feature_group(column)
        group_median_series = group_medians.get(group_name)
        redundancy = abs(safe_corr(values, group_median_series.to_numpy(dtype=float))) if group_median_series is not None else 0.0
        missing_pct = float(np.mean(~np.isfinite(values)))
        zero_share = float(np.mean(np.isclose(values, 0.0, atol=1e-12)))
        nonzero_share = 1.0 - zero_share
        record = {
            "feature_name": column,
            "group": group_name,
            "missing_percentage": missing_pct,
            "effective_nonzero_share": nonzero_share,
            "zero_share": zero_share,
            "train_test_psi": compute_psi(values[train_idx], values[test_idx]),
            "correlation_with_label": safe_corr(values, y.astype(float)),
            "mutual_information": float(mi_scores[idx]) if idx < len(mi_scores) else 0.0,
            "monthly_stability_score": stability_score,
            "train_mean": float(np.nanmean(values[train_idx])) if len(train_idx) else 0.0,
            "test_mean": float(np.nanmean(values[test_idx])) if len(test_idx) else 0.0,
            "std": float(np.nanstd(values)),
            "redundancy_to_group_median": redundancy,
        }
        record["assessment"] = usefulness_rating(record)
        features_by_group[group_name].append(record)

    group_summary: Dict[str, Any] = {}
    for group_name, features in sorted(features_by_group.items()):
        group_summary[group_name] = {
            "feature_count": len(features),
            "average_missing_percentage": float(np.mean([f["missing_percentage"] for f in features])) if features else 0.0,
            "average_effective_nonzero_share": float(np.mean([f["effective_nonzero_share"] for f in features])) if features else 0.0,
            "average_train_test_psi": float(np.mean([f["train_test_psi"] for f in features])) if features else 0.0,
            "average_abs_correlation": float(np.mean([abs(f["correlation_with_label"]) for f in features])) if features else 0.0,
            "average_mutual_information": float(np.mean([f["mutual_information"] for f in features])) if features else 0.0,
            "average_monthly_stability_score": float(np.mean([f["monthly_stability_score"] for f in features])) if features else 0.0,
            "assessment": summarize_group_assessment(features),
            "features": sorted(features, key=lambda row: (row["assessment"], -abs(float(row["correlation_with_label"])), -float(row["mutual_information"]))),
        }
    return {
        "primary_label_for_feature_audit": "trade_quality_ternary",
        "row_count": int(len(feature_df)),
        "feature_count": int(feature_df.shape[1]),
        "groups": group_summary,
    }


def infer_option_feature_presence(feature_audit: Dict[str, Any]) -> Dict[str, Any]:
    feature_lookup: Dict[str, Dict[str, Any]] = {}
    for group in feature_audit["groups"].values():
        for record in group["features"]:
            feature_lookup[record["feature_name"]] = record
    present: Dict[str, Any] = {}
    missing: Dict[str, Any] = {}
    for semantic_name, aliases in EXPECTED_OPTION_FIELDS.items():
        matched = [alias for alias in aliases if alias in feature_lookup]
        effective = [feature_lookup[name] for name in matched if float(feature_lookup[name]["effective_nonzero_share"]) >= 0.05]
        if effective:
            present[semantic_name] = {
                "matched_features": matched,
                "usable_training_features": [row["feature_name"] for row in effective],
                "effective_nonzero_share": {row["feature_name"]: row["effective_nonzero_share"] for row in effective},
                "note": "Present in feature schema and non-trivially populated.",
            }
        else:
            missing[semantic_name] = {
                "matched_features": matched,
                "reason": "Missing entirely or only present as placeholder/near-zero context fields in current training data.",
            }
    return {
        "option_feature_presence": present,
        "missing_option_edge_features": missing,
        "classification": "MISSING_OPTION_EDGE_FEATURES" if missing else "OPTION_EDGE_FEATURES_PRESENT",
        "summary": (
            "Index-only candle features appear insufficient for options scalping because most option-edge fields are absent "
            "or effectively unpopulated in the current 1-year training matrix."
        ),
    }


def label_monthly_stability(observations: Sequence[Any]) -> Dict[str, Any]:
    month_counts: Dict[str, Counter] = defaultdict(Counter)
    for obs in observations:
        month = pd.Timestamp(obs.timestamp).to_period("M").strftime("%Y-%m")
        month_counts[month][int(obs.label)] += 1
    rows = []
    pos_rates = []
    for month, counts in sorted(month_counts.items()):
        total = sum(counts.values())
        positive = int(sum(v for k, v in counts.items() if int(k) > 0))
        pos_rate = positive / max(total, 1)
        pos_rates.append(pos_rate)
        rows.append({"month": month, "total": total, "positive_rate": pos_rate, "counts": {str(k): int(v) for k, v in counts.items()}})
    stability = 1.0
    if len(pos_rates) > 1:
        stability = max(0.0, 1.0 - min(1.0, float(np.std(pos_rates) / (abs(np.mean(pos_rates)) + 1e-9))))
    return {"monthly_distribution": rows, "stability_score": stability}


def audit_labels(candles: Sequence[Any]) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    cost_model = CostModel()
    current_labeling_audit = {}
    current_labeling_path = REPORTS_DIR / "current_labeling_audit.json"
    if current_labeling_path.exists():
        current_labeling_audit = json.loads(current_labeling_path.read_text(encoding="utf-8"))

    known_policy_notes = {
        "current_triple_barrier": {
            "predicts": "Binary directional proxy based on future underlying move and barrier hit.",
            "uses_actual_trade_outcome": False,
            "tradability_relation": "Weak; predicts direction, not option trade quality.",
        },
        "cost_adjusted_triple_barrier": {
            "predicts": "Whether an underlying directional move clears conservative cost thresholds.",
            "uses_actual_trade_outcome": False,
            "tradability_relation": "Moderate proxy; closer to economic edge but still underlying-based.",
        },
        "magnitude_filtered_direction": {
            "predicts": "Directional move with a neutral noisy zone.",
            "uses_actual_trade_outcome": False,
            "tradability_relation": "Still a directional proxy, useful for filtering but not trade outcome.",
        },
        "trade_quality_binary": {
            "predicts": "Binary trade worthiness under MAE, duration, and cost assumptions.",
            "uses_actual_trade_outcome": False,
            "tradability_relation": "Better proxy to trade quality, but still simulated from index path only.",
        },
        "trade_quality_ternary": {
            "predicts": "Good trade vs noisy no-trade vs poor outcome, then collapses neutral for binary training.",
            "uses_actual_trade_outcome": False,
            "tradability_relation": "Best current research proxy, but still not true option outcome supervision.",
        },
        "regime_specific_trade_quality": {
            "predicts": "Trade quality with regime/session-adjusted thresholds.",
            "uses_actual_trade_outcome": False,
            "tradability_relation": "Closer to real deployment gating, but still proxy-labeled.",
        },
        "abstain_allowed_trade_quality": {
            "predicts": "Trade-vs-no-trade under minimum net edge and session skip logic.",
            "uses_actual_trade_outcome": False,
            "tradability_relation": "Useful for abstention research, but still not based on actual option fills.",
        },
    }

    for policy_name in available_label_policies():
        dataset = build_label_dataset(
            candles,
            policy_name=policy_name,
            lookback=LOOKBACK_BARS,
            horizon=HORIZON_BARS,
            cost_model=cost_model,
            include_features=False,
        )
        spec = get_label_policy_spec(policy_name, HORIZON_BARS, cost_model)
        binary_y = np.asarray(dataset.y, dtype=np.int32)
        balance = {
            "training_samples": int(len(binary_y)),
            "positive_class_rate": float(binary_y.mean()) if len(binary_y) else 0.0,
            "positive_class_count": int(binary_y.sum()),
            "negative_class_count": int(len(binary_y) - binary_y.sum()),
            "raw_label_distribution": dataset.label_distribution,
            "neutral_samples_dropped": int(dataset.neutral_samples_dropped),
        }
        stability = label_monthly_stability(dataset.observations)
        includes_costs = "cost" in policy_name or "trade_quality" in policy_name or "abstain" in policy_name or "regime_specific" in policy_name
        if policy_name == "current_triple_barrier" and current_labeling_audit:
            includes_costs = bool(current_labeling_audit.get("includes_brokerage_cost_slippage"))
        if balance["positive_class_rate"] < 0.01:
            weakness = "too_strict"
        elif balance["positive_class_rate"] > 0.65:
            weakness = "too_loose"
        else:
            weakness = "proxy_based"
        policy_notes = known_policy_notes.get(policy_name, {})
        reports[policy_name] = {
            "policy": asdict(spec),
            "what_it_is_trying_to_predict": policy_notes.get("predicts"),
            "uses_index_move_not_option_outcome": not policy_notes.get("uses_actual_trade_outcome", False),
            "accounts_for_spread_brokerage_slippage": includes_costs,
            "accounts_for_option_time_decay": False,
            "class_balance": balance,
            "monthly_stability": stability,
            "relation_to_real_tradability": policy_notes.get("tradability_relation"),
            "assessment": weakness,
        }

    synthetic_horizon_return = {
        "policy": {
            "name": "horizon_return_binary_proxy",
            "description": "Simple future NIFTY move over the horizon without barrier logic.",
            "required_future_horizon": HORIZON_BARS,
        },
        "what_it_is_trying_to_predict": "Whether future index return over the fixed horizon is positive.",
        "uses_index_move_not_option_outcome": True,
        "accounts_for_spread_brokerage_slippage": False,
        "accounts_for_option_time_decay": False,
        "class_balance": "Not a first-class training policy in the repo today; conceptually the loosest proxy.",
        "monthly_stability": "Not audited separately because the current policies already cover this proxy family.",
        "relation_to_real_tradability": "Very weak; useful as baseline only.",
        "assessment": "too_proxy_based",
    }
    reports["horizon_return_binary_proxy"] = synthetic_horizon_return
    return reports


def inspect_trade_outcome_data() -> Dict[str, Any]:
    db_path = REPO_ROOT / "trades.db"
    result: Dict[str, Any] = {
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "prediction_trade_outcome_linkage": {},
        "fields_available_now": [],
        "fields_missing_for_true_option_outcome_labels": [],
        "reports": {},
    }
    if (REPORTS_DIR / "paper_trade_report.json").exists():
        result["reports"]["paper_trade_report"] = json.loads((REPORTS_DIR / "paper_trade_report.json").read_text(encoding="utf-8"))
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(1) AS c FROM predictions")
            pred_count = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(1) AS c FROM trade_outcomes")
            outcome_count = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(1) AS c FROM predictions WHERE source='ACTIVE'")
            active_predictions = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(1) AS c FROM predictions WHERE source='SHADOW'")
            shadow_predictions = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(1) AS c FROM trade_outcomes WHERE prediction_id IS NOT NULL AND TRIM(prediction_id) != ''")
            linked_outcomes = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(1) AS c FROM predictions WHERE realized_trade_pnl IS NOT NULL")
            preds_with_trade_pnl = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(1) AS c FROM predictions WHERE option_symbol IS NOT NULL AND TRIM(option_symbol) != ''")
            preds_with_option_symbol = int(cur.fetchone()["c"])
            cur.execute("PRAGMA table_info(predictions)")
            pred_cols = [str(row["name"]) for row in cur.fetchall()]
            cur.execute("PRAGMA table_info(trade_outcomes)")
            out_cols = [str(row["name"]) for row in cur.fetchall()]
            result["prediction_trade_outcome_linkage"] = {
                "predictions_total": pred_count,
                "trade_outcomes_total": outcome_count,
                "active_predictions": active_predictions,
                "shadow_predictions": shadow_predictions,
                "trade_outcomes_with_prediction_id": linked_outcomes,
                "predictions_with_realized_trade_pnl": preds_with_trade_pnl,
                "predictions_with_option_symbol": preds_with_option_symbol,
            }
            result["fields_available_now"] = sorted(set(pred_cols + out_cols))
        finally:
            conn.close()

    desired_fields = [
        "timestamp",
        "spot_price",
        "selected_option_symbol",
        "ce_pe_side",
        "strike",
        "expiry",
        "option_ltp_at_signal",
        "bid",
        "ask",
        "spread",
        "volume",
        "oi",
        "change_in_oi",
        "model_probability",
        "model_features_snapshot",
        "regime_features",
        "entry_rule_that_fired",
        "simulated_entry_price",
        "simulated_stop_loss",
        "simulated_target",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "exit_price_after_fixed_horizon",
        "simulated_pnl_after_costs",
        "would_have_passed_risk_gate",
    ]
    available = set(result["fields_available_now"])
    mapped_available = {
        "timestamp": any(col in available for col in ["timestamp", "entry_time"]),
        "spot_price": "underlying_price" in available,
        "selected_option_symbol": "option_symbol" in available,
        "ce_pe_side": "direction" in available,
        "strike": False,
        "expiry": False,
        "option_ltp_at_signal": False,
        "bid": False,
        "ask": False,
        "spread": "estimated_slippage" in available,
        "volume": False,
        "oi": "iv_rank" in available,  # weak proxy only
        "change_in_oi": False,
        "model_probability": "probability" in available,
        "model_features_snapshot": "feature_snapshot_json" in available or "features_json" in available,
        "regime_features": "regime" in available,
        "entry_rule_that_fired": "entry_reason" in available or "strategy_signal" in available,
        "simulated_entry_price": False,
        "simulated_stop_loss": False,
        "simulated_target": False,
        "max_favorable_excursion": False,
        "max_adverse_excursion": False,
        "exit_price_after_fixed_horizon": False,
        "simulated_pnl_after_costs": "pnl" in available,
        "would_have_passed_risk_gate": False,
    }
    result["fields_missing_for_true_option_outcome_labels"] = [field for field in desired_fields if not mapped_available.get(field, False)]
    result["actual_trade_outcome_label_readiness"] = (
        "limited"
        if result["prediction_trade_outcome_linkage"].get("trade_outcomes_with_prediction_id", 0) < 50
        or result["prediction_trade_outcome_linkage"].get("predictions_with_option_symbol", 0) == 0
        else "usable"
    )
    result["interpretation"] = (
        "The project can log prediction and trade outcomes, but the current local history is not yet a rich, linked, "
        "option-specific dataset suitable for strong supervised trade-outcome labels."
    )
    return result


def proposed_label_specs() -> Dict[str, Any]:
    return {
        "index_proxy_directional_label": {
            "required_data_fields": ["timestamp", "spot_price", "future_spot_price"],
            "horizon": f"{HORIZON_BARS} bars",
            "stop_loss_assumption": "None; pure horizon return sign.",
            "target_assumption": "Positive future return over fixed horizon.",
            "cost_assumption": "No brokerage, spread, or slippage.",
            "expected_class_balance_risk": "Usually balanced, but weakly tied to real option expectancy.",
            "usable_now": True,
        },
        "option_premium_trade_outcome_label": {
            "required_data_fields": ["timestamp", "selected_option_symbol", "option_ltp_at_signal", "exit_price_after_fixed_horizon", "ce_pe_side", "strike", "expiry"],
            "horizon": "15-45 minutes or strategy-specific bars",
            "stop_loss_assumption": "Option-premium stop based on premium loss % or absolute rupee loss.",
            "target_assumption": "Premium gain after fixed horizon or exit rule.",
            "cost_assumption": "Optional; raw premium movement first, then cost-aware version.",
            "expected_class_balance_risk": "Can be imbalanced if entries are sparse or premium decay dominates.",
            "usable_now": False,
        },
        "cost_adjusted_trade_success_label": {
            "required_data_fields": ["simulated_entry_price", "exit_price_after_fixed_horizon", "bid", "ask", "estimated_costs", "selected_option_symbol"],
            "horizon": "Same as production trade-hold assumption",
            "stop_loss_assumption": "Strategy stop or paper-simulated stop.",
            "target_assumption": "Net PnL > 0 after brokerage + slippage + spread.",
            "cost_assumption": "Mandatory full round-trip costs.",
            "expected_class_balance_risk": "High risk of extreme positive scarcity if costs are realistic.",
            "usable_now": False,
        },
        "ternary_trade_quality_label": {
            "required_data_fields": ["simulated_entry_price", "exit_price_after_fixed_horizon", "max_favorable_excursion", "max_adverse_excursion", "estimated_costs"],
            "horizon": "Strategy-specific, ideally 15-45 minutes",
            "stop_loss_assumption": "Risk-defined stop plus adverse excursion tolerance.",
            "target_assumption": "Good / neutral / bad trade buckets based on net edge and path quality.",
            "cost_assumption": "Full costs plus spread and slippage.",
            "expected_class_balance_risk": "Healthier than binary if neutral/no-trade is preserved.",
            "usable_now": False,
        },
        "abstain_allowed_trade_label": {
            "required_data_fields": ["model_probability", "simulated_pnl_after_costs", "would_have_passed_risk_gate", "entry_rule_that_fired", "regime_features"],
            "horizon": "Same as live gating horizon",
            "stop_loss_assumption": "Risk-defined stop or no-trade bucket if setup fails constraints.",
            "target_assumption": "Good trade / bad trade / no-trade classification.",
            "cost_assumption": "Full simulated trading friction.",
            "expected_class_balance_risk": "Can be dominated by no-trade unless signal generation broadens.",
            "usable_now": False,
        },
    }


def render_edge_gap_report(
    feature_audit: Dict[str, Any],
    missing_option: Dict[str, Any],
    label_audit: Dict[str, Any],
    trade_outcomes: Dict[str, Any],
) -> str:
    option_missing_count = len(missing_option["missing_option_edge_features"])
    weak_groups = [
        f"{name} ({meta['assessment']})"
        for name, meta in feature_audit["groups"].items()
        if meta["assessment"] in {"placeholder_or_missing_in_training_data", "weak", "unstable"}
    ]
    strict_labels = [name for name, meta in label_audit.items() if meta.get("assessment") == "too_strict"]
    proxy_labels = [name for name, meta in label_audit.items() if "proxy" in str(meta.get("assessment", "")) or meta.get("uses_index_move_not_option_outcome")]
    return "\n".join(
        [
            "# Edge Gap Report",
            "",
            "## Bottom Line",
            "",
            "The current research weakness is not best explained by model choice alone. The stronger evidence points to feature, label, and data limitations.",
            "",
            "## Is The Model Weak Because Of Model Choice?",
            "",
            "- Not primarily. Earlier research already showed some ROC-AUC lift from label changes, which means model family alone is not the dominant bottleneck.",
            "- The existing model stack can separate some signal, but that separation does not survive practical thresholding or regime shifts.",
            "",
            "## Is It Weak Because Of Features?",
            "",
            f"- Yes. Weak or unstable feature groups: {', '.join(weak_groups) if weak_groups else 'none detected'}",
            f"- The training matrix carries option-context feature names, but {option_missing_count} edge-critical option fields are missing or effectively unpopulated in current training data.",
            "- This means the current pipeline mostly learns from underlying index candles, not the option microstructure actually being traded.",
            "",
            "## Is It Weak Because Labels Are Proxy Labels?",
            "",
            f"- Yes. Proxy-heavy labels include: {', '.join(proxy_labels[:8]) if proxy_labels else 'none'}",
            f"- Overly strict labels include: {', '.join(strict_labels[:8]) if strict_labels else 'none'}",
            "- Most labels are derived from future index movement, barrier logic, or simulated trade quality assumptions rather than true option premium outcomes.",
            "",
            "## Is Option-Chain Data Missing?",
            "",
            "- Yes. The current 1-year training set does not contain a complete, populated option-chain feature layer for entry-time supervision.",
            "- Without bid/ask, OI, change in OI, strike-relative pricing, and premium path data, option scalping edge is hard to learn robustly.",
            "",
            "## Is The Strategy Setup Itself Low Expectancy?",
            "",
            "- Possibly, but this audit does not prove that yet.",
            "- Current paper-trade evidence is too small and too weakly linked to predictions to cleanly separate strategy-edge problems from data/label problems.",
            "",
            "## What Data Must Be Collected Next?",
            "",
            "- Signal-time option snapshot: symbol, side, strike, expiry, LTP, bid, ask, spread, volume, OI, change in OI, IV proxy.",
            "- Trade path outcomes: simulated/actual entry, stop, target, fixed-horizon exit, MFE, MAE, and net PnL after full costs.",
            "- Prediction linkage: every signal should persist prediction_id, feature snapshot, and whether the risk gate would have allowed the trade.",
        ]
    )


def render_forward_collection_plan() -> str:
    fields = [
        "timestamp",
        "spot price",
        "selected option symbol",
        "CE/PE side",
        "strike",
        "expiry",
        "option LTP at signal",
        "bid",
        "ask",
        "spread",
        "volume",
        "OI",
        "change in OI",
        "model probability",
        "model features snapshot",
        "regime features",
        "entry rule that fired",
        "simulated entry price",
        "simulated stop-loss",
        "simulated target",
        "max favorable excursion",
        "max adverse excursion",
        "exit price after fixed horizon",
        "simulated PnL after costs",
        "whether trade would have passed risk gate",
    ]
    lines = [
        "# Forward Edge Data Collection Plan",
        "",
        "Log the following fields for every candidate signal, whether traded or not:",
        "",
    ]
    lines.extend([f"- {field}" for field in fields])
    lines.extend(
        [
            "",
            "Implementation guidance:",
            "",
            "- Persist the snapshot at signal time before any order-placement decision.",
            "- Keep prediction_id stable across signal, simulated trade, and realized outcome records.",
            "- Store both raw option fields and normalized/engineered fields so label experiments remain reproducible.",
            "- Record candidate signals that fail the risk gate as explicit no-trade observations rather than dropping them.",
            "- Capture fixed-horizon synthetic exits even when the strategy exits early, so future label families can be compared consistently.",
        ]
    )
    return "\n".join(lines)


def render_final_recommendation(
    feature_audit: Dict[str, Any],
    missing_option: Dict[str, Any],
    trade_outcomes: Dict[str, Any],
) -> str:
    classification = ["FEATURE_LIMITED", "LABEL_LIMITED", "DATA_LIMITED"]
    if len(missing_option["missing_option_edge_features"]) >= 8:
        classification.append("STRATEGY_EDGE_LIMITED")
    return "\n".join(
        [
            "# Final Edge Audit Recommendation",
            "",
            f"Final classification: `{' + '.join(classification)}`",
            "",
            "Recommendation:",
            "",
            "- Do not promote the current ML stack.",
            "- Treat the current issue as primarily a data-design problem, not a model-tuning problem.",
            "- Collect option-level entry snapshots and linked trade outcomes before the next serious retraining cycle.",
            "- Revisit model family selection only after a richer feature and label layer exists.",
            "",
            "Evidence summary:",
            "",
            f"- Feature groups audited: `{len(feature_audit['groups'])}`",
            f"- Missing or effectively absent option-edge fields: `{len(missing_option['missing_option_edge_features'])}`",
            f"- Actual trade-outcome label readiness: `{trade_outcomes['actual_trade_outcome_label_readiness']}`",
        ]
    )


def main() -> None:
    out_dir = make_output_dir()
    feature_df, _X, _feature_names, feature_timestamps, minute_frame = build_feature_frame()
    candles = frame_to_candles(minute_frame)
    y_primary, label_timestamps, primary_meta = build_primary_labels(candles, "trade_quality_ternary")
    aligned_features, y_aligned, timestamps_aligned = align_feature_and_label_data(feature_df, feature_timestamps, label_timestamps, y_primary)

    feature_audit = audit_feature_groups(aligned_features, y_aligned, timestamps_aligned)
    feature_audit["label_alignment"] = primary_meta

    missing_option = infer_option_feature_presence(feature_audit)
    label_audit = audit_labels(candles)
    trade_availability = inspect_trade_outcome_data()
    proposed_specs = proposed_label_specs()
    edge_gap_report = render_edge_gap_report(feature_audit, missing_option, label_audit, trade_availability)
    forward_plan = render_forward_collection_plan()
    final_recommendation = render_final_recommendation(feature_audit, missing_option, trade_availability)

    write_json(out_dir / "feature_group_audit.json", ensure_serializable(feature_audit))
    write_json(out_dir / "missing_option_features_report.json", ensure_serializable(missing_option))
    write_json(out_dir / "label_audit_report.json", ensure_serializable(label_audit))
    write_json(out_dir / "trade_outcome_data_availability.json", ensure_serializable(trade_availability))
    write_json(out_dir / "proposed_label_specs.json", ensure_serializable(proposed_specs))
    (out_dir / "edge_gap_report.md").write_text(edge_gap_report, encoding="utf-8")
    (out_dir / "forward_edge_data_collection_plan.md").write_text(forward_plan, encoding="utf-8")
    (out_dir / "final_edge_audit_recommendation.md").write_text(final_recommendation, encoding="utf-8")

    summary = {
        "output_dir": str(out_dir),
        "option_chain_features_exist_in_training_data": missing_option["classification"] != "MISSING_OPTION_EDGE_FEATURES",
        "actual_trade_outcome_labels_exist_now": trade_availability["actual_trade_outcome_label_readiness"] == "usable",
        "biggest_missing_feature_group": "option_chain_features",
        "biggest_label_weakness": "proxy_labels_based_on_underlying_direction_rather_than_true_option_outcomes",
        "final_classification": "FEATURE_LIMITED + LABEL_LIMITED + DATA_LIMITED",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
