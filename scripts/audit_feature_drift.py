#!/usr/bin/env python3
"""
Feature Drift Audit Script for NiftyScalper Model Stack.
Research-only. Computes Population Stability Index (PSI) for each
production model feature against the latest processed dataset.
"""

import os
import sys
import re
import json
import glob
import math
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

FEATURE_LIST_PATH = os.path.join(
    MODELS_DIR, "core_retrain_20260606_093842", "feature_list_used.json"
)

# Fallback: look for any feature_list_used.json
FALLBACK_FEATURE_LIST_GLOB = os.path.join(MODELS_DIR, "*/feature_list_used.json")

# Fallback dataset glob
LATEST_DATASET_GLOB = os.path.join(PROCESSED_DIR, "nifty_option_chain_cost_aware_edge_dataset_*.csv")

BINS = 10
EXPECTED_PCT = 0.70  # earlier rows = expected (training)
PSI_SEVERE = 0.30
PSI_SIGNIFICANT = 0.20
PSI_STABLE_MAX = 0.15
PSI_WATCH_MAX = 0.25

REPLACEMENT_MAP = {
    "ltp": "option_to_spot_pct, premium_zscore, premium_rank_pct",
    "ctx_option_price": "normalized premium features (premium / spot or premium / atr)",
    "dist_to_rolling_low_20": "percentile rank version or distance_pct_rank_20",
    "dist_to_rolling_high_20": "similar rank version",
    "high_spot": "normalize by spot_close; use returns or pct changes",
    "low_spot": "normalize by spot_close; use returns or pct changes",
    "open_spot": "normalize by spot_close; use returns or pct changes",
    "close": "normalize by spot_close; use returns or pct changes",
    "spot_close": "normalize by spot_close; use returns or pct changes",
    "last_high": "normalize by spot_close; use returns or pct changes",
    "last_low": "normalize by spot_close; use returns or pct changes",
    "last_open": "normalize by spot_close; use returns or pct changes",
    "last_close": "normalize by spot_close; use returns or pct changes",
    "spot_vwap": "normalize by spot_close; use pct deviation from VWAP",
    "ctx_spot": "normalize by spot_close; use returns or pct changes",
    "month": "seasonality indicator or cyclical encoding; verify sample bias",
    "ctx_time_sin": "verify timestamp distribution; consider session-relative time",
    "ctx_time_cos": "verify timestamp distribution; consider session-relative time",
    "is_opening_session": "verify timestamp distribution; consider fixed window",
    "volume_spot": "normalize by rolling average volume (volume_ratio)",
    "last_volume": "normalize by rolling average volume (volume_ratio)",
    "spot_atr": "expected drift in volatility regime; monitor regime classifier",
    "atr_14": "expected drift in volatility regime; monitor regime classifier",
    "atr_pct": "expected drift in volatility regime; monitor regime classifier",
    "distance_from_spot": "use strike_distance_pct instead of absolute distance",
    "strike_price": "remove or replace with moneyness / strike_distance_pct",
}


def _find_latest_dataset() -> str:
    """Pick the most recent processed CSV by timestamp embedded in filename."""
    candidates = glob.glob(LATEST_DATASET_GLOB)
    if not candidates:
        # ultimate fallback: any csv in processed
        candidates = glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No processed datasets found in {PROCESSED_DIR}")

    def _timestamp_key(path: str) -> str:
        base = os.path.basename(path)
        m = re.search(r"(\d{8})_(\d{6})", base)
        if m:
            return m.group(1) + m.group(2)
        m = re.search(r"(\d{14})", base)
        if m:
            return m.group(1)
        # last resort: mtime
        return str(os.path.getmtime(path))

    candidates.sort(key=_timestamp_key, reverse=True)
    return candidates[0]


def _find_feature_list() -> List[str]:
    """Return the ordered list of model features."""
    path = FEATURE_LIST_PATH
    if not os.path.exists(path):
        fallbacks = glob.glob(FALLBACK_FEATURE_LIST_GLOB)
        if not fallbacks:
            raise FileNotFoundError("No feature_list_used.json found under models/")
        fallbacks.sort(reverse=True)
        path = fallbacks[0]

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Accept either top-level list or dict with 'features' key
    if isinstance(data, list):
        return list(data)
    return list(data.get("features", data.get("baseline_features", [])))


def _extract_features_from_pkl(path: str) -> List[str]:
    """Last-resort: read feature names from a pickled sklearn model."""
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        return []
    # Pipeline or direct estimator
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return []


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = BINS) -> float:
    """
    Population Stability Index.
    Uses fixed bins derived from the expected distribution.
    Automatically handles boolean / low-cardinality categorical features.
    """
    # Drop NaNs and infinities
    e = expected.replace([np.inf, -np.inf], np.nan).dropna()
    a = actual.replace([np.inf, -np.inf], np.nan).dropna()

    if len(e) == 0 or len(a) == 0:
        return float("nan")

    # Detect boolean / categorical with few unique values
    is_bool = e.dtype == bool or a.dtype == bool
    unique_vals = pd.concat([e, a]).unique()
    n_unique = len(unique_vals)

    eps = 1e-6

    if is_bool or n_unique <= 2:
        # Binary / boolean case: two categories (0/1 or True/False)
        e_counts = e.value_counts().sort_index()
        a_counts = a.value_counts().sort_index()
        # Align indexes
        idx = e_counts.index.union(a_counts.index)
        e_pct = (e_counts.reindex(idx, fill_value=0).values / len(e)) + eps
        a_pct = (a_counts.reindex(idx, fill_value=0).values / len(a)) + eps
        psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
        return float(psi)

    if n_unique <= bins:
        # Low-cardinality integer/categorical: use each unique value as a bin
        e_counts = e.value_counts().sort_index()
        a_counts = a.value_counts().sort_index()
        idx = e_counts.index.union(a_counts.index)
        e_pct = (e_counts.reindex(idx, fill_value=0).values / len(e)) + eps
        a_pct = (a_counts.reindex(idx, fill_value=0).values / len(a)) + eps
        psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
        return float(psi)

    # Continuous / high-cardinality case: quantile-based bins
    e_min, e_max = e.min(), e.max()
    if e_min == e_max:
        a_min, a_max = a.min(), a.max()
        return 0.05 if (a_min == a_max and a_min == e_min) else 0.35

    breaks = np.quantile(e, np.linspace(0, 1, bins + 1))
    breaks = np.unique(breaks)
    if len(breaks) < 3:
        breaks = np.linspace(e_min, e_max, bins + 1)

    e_counts, _ = np.histogram(e, bins=breaks)
    a_counts, _ = np.histogram(a, bins=breaks)

    e_pct = (e_counts / e_counts.sum()) + eps
    a_pct = (a_counts / a_counts.sum()) + eps
    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(psi)


def drift_category(psi: float) -> str:
    if math.isnan(psi):
        return "unknown"
    if psi < PSI_STABLE_MAX:
        return "stable"
    if psi < PSI_WATCH_MAX:
        return "moderate"
    if psi >= PSI_SIGNIFICANT:
        return "severe"
    return "moderate"


def _build_recommendation(feature: str, psi: float) -> str:
    for key, rec in REPLACEMENT_MAP.items():
        if key in feature.lower():
            return rec
    return ""


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset_path = _find_latest_dataset()
    print(f"[audit] Loading dataset: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"[audit] Dataset shape: {df.shape}")

    # Sort by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    elif "trading_day" in df.columns:
        df = df.sort_values("trading_day", ascending=True).reset_index(drop=True)

    split_idx = int(len(df) * EXPECTED_PCT)
    expected_df = df.iloc[:split_idx]
    actual_df = df.iloc[split_idx:]
    print(f"[audit] Expected rows: {len(expected_df)}, Actual rows: {len(actual_df)}")

    # ------------------------------------------------------------------
    # 2. Load feature list
    # ------------------------------------------------------------------
    features = _find_feature_list()
    if not features:
        print("[audit] ERROR: Could not load feature list.", file=sys.stderr)
        sys.exit(1)
    print(f"[audit] Features loaded: {len(features)}")

    # Verify features exist in dataset; warn / skip missing
    missing = [f for f in features if f not in df.columns]
    available_features = [f for f in features if f in df.columns]
    if missing:
        print(f"[audit] WARNING: {len(missing)} features missing from dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # ------------------------------------------------------------------
    # 3. Compute PSI per feature
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    for feat in available_features:
        psi_val = compute_psi(expected_df[feat], actual_df[feat], bins=BINS)
        cat = drift_category(psi_val)
        rec = _build_recommendation(feat, psi_val)
        results.append({
            "feature": feat,
            "psi": round(psi_val, 6) if not math.isnan(psi_val) else None,
            "drift_category": cat,
            "recommendation": rec,
        })

    results.sort(key=lambda x: x["psi"] if x["psi"] is not None else -1, reverse=True)

    # ------------------------------------------------------------------
    # 4. Build stable/moderate/severe lists
    # ------------------------------------------------------------------
    stable = [r for r in results if r["drift_category"] == "stable"]
    moderate = [r for r in results if r["drift_category"] == "moderate"]
    severe = [r for r in results if r["drift_category"] == "severe"]
    unknown = [r for r in results if r["drift_category"] == "unknown"]

    stable_feature_names = [r["feature"] for r in stable]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # 5. Save stable feature list
    # ------------------------------------------------------------------
    stable_list_path = os.path.join(REPORTS_DIR, f"stable_feature_list_{timestamp}.json")
    stable_payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": os.path.basename(dataset_path),
        "expected_rows": len(expected_df),
        "actual_rows": len(actual_df),
        "psi_thresholds": {
            "stable_max": PSI_STABLE_MAX,
            "watch_max": PSI_WATCH_MAX,
            "significant": PSI_SIGNIFICANT,
            "severe": PSI_SEVERE,
        },
        "stable_features": stable_feature_names,
        "moderate_features": [r["feature"] for r in moderate],
        "severe_features": [r["feature"] for r in severe],
        "unknown_features": [r["feature"] for r in unknown],
        "stable_count": len(stable),
        "moderate_count": len(moderate),
        "severe_count": len(severe),
        "unknown_count": len(unknown),
    }
    with open(stable_list_path, "w", encoding="utf-8") as fh:
        json.dump(stable_payload, fh, indent=2)
    print(f"[audit] Saved stable feature list -> {stable_list_path}")

    # ------------------------------------------------------------------
    # 6. Save JSON report
    # ------------------------------------------------------------------
    json_report_path = os.path.join(REPORTS_DIR, f"feature_drift_audit_{timestamp}.json")
    json_payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "dataset_basename": os.path.basename(dataset_path),
        "expected_rows": len(expected_df),
        "actual_rows": len(actual_df),
        "bins": BINS,
        "psi_thresholds": {
            "stable_max": PSI_STABLE_MAX,
            "watch_max": PSI_WATCH_MAX,
            "significant": PSI_SIGNIFICANT,
            "severe": PSI_SEVERE,
        },
        "summary": {
            "total_features_evaluated": len(results),
            "stable": len(stable),
            "moderate": len(moderate),
            "severe": len(severe),
            "unknown": len(unknown),
        },
        "features": results,
    }
    with open(json_report_path, "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"[audit] Saved JSON report -> {json_report_path}")

    # ------------------------------------------------------------------
    # 7. Save Markdown report
    # ------------------------------------------------------------------
    md_path = os.path.join(REPORTS_DIR, f"feature_drift_audit_{timestamp}.md")
    lines = [
        "# Feature Drift Audit Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Dataset:** `{os.path.basename(dataset_path)}`",
        f"**Expected Rows:** {len(expected_df):,}",
        f"**Actual Rows:** {len(actual_df):,}",
        f"**Bins:** {BINS}",
        "",
        "## PSI Thresholds",
        "",
        f"- PSI < {PSI_STABLE_MAX}: **Stable**",
        f"- {PSI_STABLE_MAX} ≤ PSI < {PSI_WATCH_MAX}: **Moderate (watch)**",
        f"- {PSI_WATCH_MAX} ≤ PSI < {PSI_SEVERE}: **Significant**",
        f"- PSI ≥ {PSI_SEVERE}: **Severe**",
        "",
        "## Summary",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Stable | {len(stable)} |",
        f"| Moderate | {len(moderate)} |",
        f"| Severe | {len(severe)} |",
        f"| Unknown | {len(unknown)} |",
        "",
        "## Severe Drift Features",
        "",
        "| Feature | PSI | Recommendation |",
        "|---------|-----|----------------|",
    ]
    for r in severe:
        psi_str = f"{r['psi']:.6f}" if r["psi"] is not None else "N/A"
        lines.append(f"| {r['feature']} | {psi_str} | {r['recommendation']} |")
    if not severe:
        lines.append("_No features with severe drift._")

    lines += [
        "",
        "## Moderate Drift Features",
        "",
        "| Feature | PSI | Recommendation |",
        "|---------|-----|----------------|",
    ]
    for r in moderate:
        psi_str = f"{r['psi']:.6f}" if r["psi"] is not None else "N/A"
        lines.append(f"| {r['feature']} | {psi_str} | {r['recommendation']} |")
    if not moderate:
        lines.append("_No features with moderate drift._")

    lines += [
        "",
        "## Top Stable Features",
        "",
        "| Feature | PSI |",
        "|---------|-----|",
    ]
    for r in stable[:20]:
        psi_str = f"{r['psi']:.6f}" if r["psi"] is not None else "N/A"
        lines.append(f"| {r['feature']} | {psi_str} |")
    if len(stable) > 20:
        lines.append(f"| ... ({len(stable) - 20} more) | |")

    lines += [
        "",
        "## Full Feature List",
        "",
        "| Feature | PSI | Category | Recommendation |",
        "|---------|-----|----------|----------------|",
    ]
    for r in results:
        psi_str = f"{r['psi']:.6f}" if r["psi"] is not None else "N/A"
        lines.append(
            f"| {r['feature']} | {psi_str} | {r['drift_category']} | {r['recommendation']} |"
        )

    lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[audit] Saved Markdown report -> {md_path}")

    print("[audit] Feature drift audit complete.")


if __name__ == "__main__":
    main()
