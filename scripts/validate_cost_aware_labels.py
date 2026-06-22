#!/usr/bin/env python3
"""Validate cost-aware labels for NiftyScalper research.

This script inspects the most recent cost-aware edge dataset, validates that
cost-aware labels exist and are properly constructed, checks their positive
rates, cost embedding, and stress multiplier logic, and writes a research-only
report pair (JSON + Markdown) to reports/.

No live trading. No model promotion. No dataset regeneration.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Expected labels (primary targets first)
# ---------------------------------------------------------------------------
REQUIRED_LABELS = (
    "cost_survivor_label_v2",
    "strong_profitable_trade_label_v2",
)
OPTIONAL_LABELS = (
    "high_conviction_trade_label_v2",
    "weak_trade_label",
    "cost_survivor_label",
    "strong_profitable_trade_label",
    "high_conviction_trade_label",
    "paper_candidate_label",
    "paper_candidate_label_v2",
    "no_trade_label",
)
ALL_LABELS = REQUIRED_LABELS + OPTIONAL_LABELS

# Cost components we expect in the logic
EXPECTED_COST_COMPONENTS = (
    "brokerage",
    "slippage",
    "exchange_charges",
    "stt",
    "gst",
    "sebi",
    "spread",
)

# Stress multipliers we expect to be supported
EXPECTED_STRESS_MULTIPLIERS = (1.25, 1.50, 2.00)

# Positive-rate target band
POS_RATE_LOWER = 0.20
POS_RATE_UPPER = 0.50


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _find_latest_cost_aware_dataset() -> Path | None:
    candidates = sorted(DATA_DIR.glob("nifty_option_chain_cost_aware_edge_dataset_*.csv"))
    return candidates[-1] if candidates else None


def _read_dataset(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if max_rows:
        return pd.read_csv(path, low_memory=False, nrows=max_rows)
    return pd.read_csv(path, low_memory=False)


def _compute_positive_rate(series: pd.Series) -> Dict[str, Any]:
    """Compute positive rate and metadata for a binary label series."""
    valid = series.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"exists": False, "positive_count": 0, "positive_rate": 0.0, "n_valid": 0, "in_target_band": False}
    pos_mask = series.fillna(0.0).eq(1.0)
    pos_count = int(pos_mask.sum())
    pos_rate = float(pos_count / n_valid)
    return {
        "exists": True,
        "positive_count": pos_count,
        "positive_rate": pos_rate,
        "n_valid": n_valid,
        "in_target_band": POS_RATE_LOWER <= pos_rate <= POS_RATE_UPPER,
        "too_rare": pos_rate < 0.005,
        "too_broad": pos_rate > POS_RATE_UPPER,
    }


def _compute_label_statistics(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Compute comprehensive stats for a label column."""
    if label not in df.columns:
        return {"exists": False, "reason": "column_missing"}

    base = _compute_positive_rate(df[label])
    if not base["exists"]:
        return base

    # Per-label return stats on positive samples
    pos_mask = df[label].fillna(0.0).eq(1.0)
    if "net_forward_return" in df.columns:
        net = pd.to_numeric(df.loc[pos_mask, "net_forward_return"], errors="coerce").dropna()
        base["avg_net_return_positive"] = float(net.mean()) if net.size else None
        base["median_net_return_positive"] = float(net.median()) if net.size else None
        base["std_net_return_positive"] = float(net.std(ddof=1)) if net.size > 1 else None
        base["min_net_return_positive"] = float(net.min()) if net.size else None
        base["max_net_return_positive"] = float(net.max()) if net.size else None
    else:
        base["avg_net_return_positive"] = None
        base["median_net_return_positive"] = None

    return base


def _check_cost_embedding(df: pd.DataFrame) -> Dict[str, Any]:
    """Check that cost components are embedded correctly."""
    # We look for evidence of cost in the dataset and in derived columns
    report: Dict[str, Any] = {
        "cost_columns_in_dataset": [],
        "cost_return_units_estimated_exists": False,
        "bid_ask_spread_pct_exists": False,
        "bid_ask_spread_pct_notna_count": 0,
        "bid_ask_spread_pct_is_all_nan": True,
        "return_to_cost_ratio_exists": False,
        "expected_return_after_cost_exists": False,
        "embedded_cost_inferred": None,
        "estimated_cost_mean": None,
        "estimated_cost_median": None,
    }

    for c in ("cost_return_units_estimated", "return_to_cost_ratio",
              "expected_return_after_cost", "bid_ask_spread_pct"):
        if c in df.columns:
            report["cost_columns_in_dataset"].append(c)

    if "cost_return_units_estimated" in df.columns:
        s = pd.to_numeric(df["cost_return_units_estimated"], errors="coerce").dropna()
        report["cost_return_units_estimated_exists"] = True
        report["estimated_cost_mean"] = float(s.mean()) if s.size else None
        report["estimated_cost_median"] = float(s.median()) if s.size else None

    if "bid_ask_spread_pct" in df.columns:
        s = df["bid_ask_spread_pct"]
        report["bid_ask_spread_pct_exists"] = True
        report["bid_ask_spread_pct_notna_count"] = int(s.notna().sum())
        report["bid_ask_spread_pct_is_all_nan"] = s.isna().all()

    if "return_to_cost_ratio" in df.columns:
        report["return_to_cost_ratio_exists"] = True
    if "expected_return_after_cost" in df.columns:
        report["expected_return_after_cost_exists"] = True

    # Infer embedded cost from gross vs net returns
    if "gross_forward_return" in df.columns and "net_forward_return" in df.columns:
        gross = pd.to_numeric(df["gross_forward_return"], errors="coerce")
        net = pd.to_numeric(df["net_forward_return"], errors="coerce")
        diff = (gross - net).replace([np.inf, -np.inf], np.nan).dropna()
        pos_diff = diff[diff > 0.0]
        report["embedded_cost_inferred"] = float(pos_diff.median()) if not pos_diff.empty else None

    return report


def _check_stress_multipliers(df: pd.DataFrame) -> Dict[str, Any]:
    """Check that stress multiplier logic is available."""
    report: Dict[str, Any] = {
        "stress_multipliers_expected": list(EXPECTED_STRESS_MULTIPLIERS),
        "stress_multiplier_columns_in_dataset": [],
        "computed_survival_rates": {},
    }

    # The dataset itself may not have pre-computed stress columns; we compute them
    if "cost_return_units_estimated" in df.columns and "net_forward_return" in df.columns:
        net = pd.to_numeric(df["net_forward_return"], errors="coerce")
        cost = pd.to_numeric(df["cost_return_units_estimated"], errors="coerce")
        valid = net.notna() & cost.notna()
        net_v = net[valid].to_numpy(dtype=float)
        cost_v = cost[valid].to_numpy(dtype=float)

        for mult in EXPECTED_STRESS_MULTIPLIERS:
            survived = (net_v - mult * cost_v) > 0.0
            total = survived.size
            report["computed_survival_rates"][f"cost_{mult:.2f}x_survival"] = float(survived.mean()) if total else None

    return report


def _build_summary(label_stats: Dict[str, Any], cost_report: Dict[str, Any], stress_report: Dict[str, Any], dataset_path: Path, df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_exists": True,
        "dataset_row_count": int(df.shape[0]),
        "dataset_column_count": int(df.shape[1]),
        "dataset_date_range": _safe_date_range(df),
        "dataset_age_days": _dataset_age_days(dataset_path),
        "labels": label_stats,
        "cost_embedding": cost_report,
        "stress_multipliers": stress_report,
        "gaps": [],
        "recommendations": [],
    }

    # Gap: required labels missing or out of band
    for label in REQUIRED_LABELS:
        stats = label_stats.get(label, {})
        if not stats.get("exists"):
            summary["gaps"].append(f"Required label '{label}' is missing")
        elif not stats.get("in_target_band", False):
            summary["gaps"].append(
                f"Required label '{label}' positive rate {stats.get('positive_rate', 0):.4f} outside target band [{POS_RATE_LOWER}, {POS_RATE_UPPER}]"
            )

    # Gap: optional labels out of target band (for research awareness)
    for label in OPTIONAL_LABELS:
        stats = label_stats.get(label, {})
        if not stats.get("exists"):
            continue
        if stats.get("too_rare"):
            summary["gaps"].append(f"Label '{label}' is too rare (positive rate {stats.get('positive_rate', 0):.4f} < 0.005)")
        elif not stats.get("in_target_band", False):
            # no_trade_label being too broad is expected (it's a catch-all avoid label)
            if label != "no_trade_label":
                summary["gaps"].append(
                    f"Label '{label}' positive rate {stats.get('positive_rate', 0):.4f} outside target band [{POS_RATE_LOWER}, {POS_RATE_UPPER}]"
                )

    # Gap: cost columns missing
    if not cost_report["cost_return_units_estimated_exists"]:
        summary["gaps"].append("cost_return_units_estimated column missing")
    if not cost_report["return_to_cost_ratio_exists"]:
        summary["gaps"].append("return_to_cost_ratio column missing")
    if not cost_report["expected_return_after_cost_exists"]:
        summary["gaps"].append("expected_return_after_cost column missing")

    # Gap: spread data missing
    if cost_report["bid_ask_spread_pct_exists"] and cost_report["bid_ask_spread_pct_is_all_nan"]:
        summary["gaps"].append("bid_ask_spread_pct is present but all NaN — spread cost component is effectively zero")

    summary["overall_valid"] = len(summary["gaps"]) == 0
    return summary


def _safe_date_range(df: pd.DataFrame) -> str:
    for col in ("timestamp", "trading_day"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.notna().any():
                return f"{ts.min()} -> {ts.max()}"
    return "unknown"


def _dataset_age_days(path: Path) -> float | None:
    try:
        # Extract timestamp from filename: nifty_option_chain_cost_aware_edge_dataset_YYYYMMDD_HHMMSS.csv
        stem = path.stem
        parts = stem.rsplit("_", 2)
        if len(parts) >= 3:
            ts_str = parts[-2] + parts[-1]
            built = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - built).total_seconds() / 86400.0
    except Exception:
        pass
    return None


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cost-Aware Label Validation Report")
    lines.append("")
    lines.append(f"- **Generated**: `{summary['generated_at']}`")
    lines.append(f"- **Dataset**: `{summary['dataset_path']}`")
    lines.append(f"- **Rows**: `{summary['dataset_row_count']:,}`")
    lines.append(f"- **Columns**: `{summary['dataset_column_count']}`")
    lines.append(f"- **Date Range**: `{summary['dataset_date_range']}`")
    lines.append(f"- **Dataset Age (days)**: `{summary['dataset_age_days']:.2f}`" if summary.get("dataset_age_days") is not None else "- **Dataset Age (days)**: `unknown`")
    lines.append("")
    lines.append("## Label Availability & Quality")
    lines.append("")
    lines.append("| label | exists | positive_count | positive_rate | in_target_band | too_rare | too_broad | avg_net_ret | median_net_ret |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label, stats in summary["labels"].items():
        if not stats.get("exists"):
            lines.append(f"| {label} | no | — | — | — | — | — | — | — |")
            continue
        avg_ret = f"{stats.get('avg_net_return_positive', 0):.4f}" if stats.get('avg_net_return_positive') is not None else "n/a"
        med_ret = f"{stats.get('median_net_return_positive', 0):.4f}" if stats.get('median_net_return_positive') is not None else "n/a"
        lines.append(
            f"| {label} | yes | {stats['positive_count']:,} | {stats['positive_rate']:.4f} | "
            f"{stats.get('in_target_band', False)} | {stats.get('too_rare', False)} | {stats.get('too_broad', False)} | "
            f"{avg_ret} | {med_ret} |"
        )
    lines.append("")
    lines.append("## Cost Embedding")
    lines.append("")
    ce = summary["cost_embedding"]
    lines.append(f"- `cost_return_units_estimated` exists: `{ce['cost_return_units_estimated_exists']}`")
    lines.append(f"- `return_to_cost_ratio` exists: `{ce['return_to_cost_ratio_exists']}`")
    lines.append(f"- `expected_return_after_cost` exists: `{ce['expected_return_after_cost_exists']}`")
    lines.append(f"- `bid_ask_spread_pct` exists: `{ce['bid_ask_spread_pct_exists']}`")
    lines.append(f"- `bid_ask_spread_pct` all NaN: `{ce['bid_ask_spread_pct_is_all_nan']}`")
    lines.append(f"- Inferred embedded cost (median gross-net): `{ce['embedded_cost_inferred']}`")
    lines.append(f"- Estimated cost mean: `{ce['estimated_cost_mean']}`")
    lines.append(f"- Estimated cost median: `{ce['estimated_cost_median']}`")
    lines.append("")
    lines.append("### Expected Cost Components")
    for comp in EXPECTED_COST_COMPONENTS:
        lines.append(f"- {comp}: expected in logic")
    lines.append("")
    lines.append("## Stress Multipliers")
    lines.append("")
    sm = summary["stress_multipliers"]
    for mult in EXPECTED_STRESS_MULTIPLIERS:
        key = f"cost_{mult:.2f}x_survival"
        val = sm["computed_survival_rates"].get(key)
        lines.append(f"- {key}: `{val}`" if val is not None else f"- {key}: `unable_to_compute`")
    lines.append("")
    lines.append("## Gaps Found")
    if summary["gaps"]:
        for gap in summary["gaps"]:
            lines.append(f"- ⚠️ {gap}")
    else:
        lines.append("- ✅ None")
    lines.append("")
    lines.append(f"## Overall Valid")
    lines.append(f"- `{summary['overall_valid']}`")
    lines.append("")
    lines.append("---")
    lines.append("*Research-only output. No live trading. No model promotion.*")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    dataset_path = _find_latest_cost_aware_dataset()
    if dataset_path is None:
        print("[validate] No cost-aware dataset found in data/processed/", file=sys.stderr)
        # Write a failure report
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_exists": False,
            "dataset_path": None,
            "gaps": ["No cost-aware dataset found in data/processed/"],
            "overall_valid": False,
        }
        _write_json(REPORTS_DIR / f"cost_aware_label_validation_{TIMESTAMP}.json", summary)
        _write_text(REPORTS_DIR / f"cost_aware_label_validation_{TIMESTAMP}.md",
                    "# Cost-Aware Label Validation Report\n\n**Dataset not found.**\n")
        return 1

    print(f"[validate] Loading dataset: {dataset_path}")
    df = _read_dataset(dataset_path)
    print(f"[validate] Rows: {df.shape[0]} Cols: {df.shape[1]}")

    label_stats: Dict[str, Any] = {}
    for label in ALL_LABELS:
        label_stats[label] = _compute_label_statistics(df, label)

    cost_report = _check_cost_embedding(df)
    stress_report = _check_stress_multipliers(df)
    summary = _build_summary(label_stats, cost_report, stress_report, dataset_path, df)

    json_path = REPORTS_DIR / f"cost_aware_label_validation_{TIMESTAMP}.json"
    md_path = REPORTS_DIR / f"cost_aware_label_validation_{TIMESTAMP}.md"
    _write_json(json_path, summary)
    _write_text(md_path, _render_markdown(summary))

    print(f"[validate] Wrote: {json_path}")
    print(f"[validate] Wrote: {md_path}")
    print(f"[validate] Overall valid: {summary['overall_valid']}")
    if summary["gaps"]:
        print(f"[validate] Gaps ({len(summary['gaps'])}):")
        for g in summary["gaps"]:
            print(f"  - {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
