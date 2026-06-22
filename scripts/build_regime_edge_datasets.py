#!/usr/bin/env python3
"""
Build regime edge datasets for NiftyScalper research.

Reads the latest processed option chain cost-aware edge dataset,
subsets by regime conditions, generates statistics reports,
and saves filtered datasets.

Research only. No live trading.
"""

import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
MIN_ROWS = 1000

# Regime definitions: name -> filter expression (as string for reporting)
REGIME_DEFINITIONS = {
    "high_volatility": "(regime_volatile == 1) | (realized_vol_percentile_60 > 0.7)",
    "low_volatility": "(regime_quiet == 1) | (realized_vol_percentile_60 < 0.3)",
    "trending": "regime_trending == 1",
    "mean_reverting": "regime_mean_reverting == 1",
    "quiet": "regime_quiet == 1",
    "volatile_pe": "(regime_volatile == 1) & (option_type == 'PE')",
    "volatile_ce": "(regime_volatile == 1) & (option_type == 'CE')",
}

# Columns of interest for regime inspection
REQUIRED_REGIME_COLS = [
    "regime_trending",
    "regime_volatile",
    "regime_mean_reverting",
    "regime_quiet",
    "volatility_regime_classifier",
    "atr_pct_regime_10",
    "realized_vol_percentile_60",
]


def find_latest_dataset() -> Path:
    """Find the most recent cost-aware edge dataset in data/processed/."""
    pattern = str(PROCESSED_DIR / "nifty_option_chain_cost_aware_edge_dataset_*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"ERROR: No matching dataset found in {PROCESSED_DIR}")
        sys.exit(1)
    # Sort by modification time, newest first
    latest = max(files, key=os.path.getmtime)
    return Path(latest)


def resolve_target_label(df: pd.DataFrame) -> str | None:
    """Pick the best available target label column."""
    candidates = ["cost_survivor_label_v2", "strong_profitable_trade_label_v2"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_spread_col(df: pd.DataFrame) -> str | None:
    """Pick the best available spread column.

    wide_spread_flag is only accepted if it has actual True/1 values,
    since a flag that is uniformly False is not a useful spread proxy.
    """
    for col in ["bid_ask_spread_pct", "bid_ask_spread"]:
        if col in df.columns and df[col].notna().sum() > 0:
            return col
    # wide_spread_flag as a last resort, but only if it shows variation
    if "wide_spread_flag" in df.columns:
        s = df["wide_spread_flag"].dropna()
        if len(s) > 0 and s.astype(bool).sum() > 0:
            return "wide_spread_flag"
    return None


def resolve_premium_col(df: pd.DataFrame) -> str | None:
    """Pick the best available premium column."""
    candidates = ["premium", "option_to_spot_pct"]
    for col in candidates:
        if col in df.columns and df[col].notna().sum() > 0:
            return col
    return None


def compute_feature_coverage(df: pd.DataFrame) -> float:
    """Compute overall feature coverage (non-null proportion) across likely feature columns.

    Excludes metadata, labels, flags, and regime columns.
    """
    exclude_prefixes = (
        "timestamp", "trading_day", "source_file", "instrument_key", "trading_symbol",
        "expiry", "weekly", "option_type", "strike_price", "label", "flag", "regime_",
        "profitable_trade_label", "avoid_trade_label", "strong_profitable_trade_label",
        "high_conviction_trade_label", "weak_trade_label", "no_trade_label",
        "cost_survivor_label", "paper_candidate_label", "bs_iv_source",
        "greeks_source", "row_enrichment_error", "has_valid_bid_ask",
    )
    feature_cols = [
        c for c in df.columns
        if not any(c.lower().startswith(p.lower()) for p in exclude_prefixes)
        and not any(p.lower() in c.lower() for p in ["_label", "_flag", "source", "error"])
    ]
    if not feature_cols:
        return float("nan")
    total_cells = len(df) * len(feature_cols)
    non_null_cells = df[feature_cols].notna().sum().sum()
    return non_null_cells / total_cells


def build_regime_subset(df: pd.DataFrame, name: str, expr: str) -> dict:
    """Build a single regime subset and return its report metadata."""
    print(f"  Building regime: {name} ...")
    subset = df.query(expr).copy()
    rows = len(subset)
    passed = rows >= MIN_ROWS

    target_col = resolve_target_label(df)
    spread_col = resolve_spread_col(df)
    premium_col = resolve_premium_col(df)

    # Date range
    date_min = date_max = None
    if "timestamp" in subset.columns and subset["timestamp"].notna().any():
        ts = pd.to_datetime(subset["timestamp"], errors="coerce")
        date_min = str(ts.min()) if ts.notna().any() else None
        date_max = str(ts.max()) if ts.notna().any() else None
    elif "trading_day" in subset.columns and subset["trading_day"].notna().any():
        td = pd.to_datetime(subset["trading_day"], errors="coerce")
        date_min = str(td.min()) if td.notna().any() else None
        date_max = str(td.max()) if td.notna().any() else None

    # CE/PE counts
    ce_pe_counts = {}
    if "option_type" in subset.columns:
        ce_pe_counts = subset["option_type"].value_counts().to_dict()

    # Target positive rate
    positive_rate = None
    if target_col and target_col in subset.columns:
        pos = subset[target_col].dropna()
        if len(pos) > 0:
            positive_rate = float(pos.mean())

    # Average spread
    avg_spread = None
    if spread_col and spread_col in subset.columns:
        s = subset[spread_col].dropna()
        if len(s) > 0:
            avg_spread = float(s.mean())

    # Average premium
    avg_premium = None
    if premium_col and premium_col in subset.columns:
        p = subset[premium_col].dropna()
        if len(p) > 0:
            avg_premium = float(p.mean())

    # Missing values per column (top 20)
    missing_counts = subset.isna().sum().sort_values(ascending=False).head(20)
    missing_report = missing_counts[missing_counts > 0].to_dict()

    # Feature coverage
    feat_cov = compute_feature_coverage(subset)

    meta = {
        "regime_name": name,
        "filter_expression": expr,
        "rows": rows,
        "passed_threshold": passed,
        "date_min": date_min,
        "date_max": date_max,
        "ce_pe_counts": {str(k): int(v) for k, v in ce_pe_counts.items()},
        "target_label_column": target_col,
        "positive_rate": positive_rate,
        "spread_column": spread_col,
        "avg_spread": avg_spread,
        "premium_column": premium_col,
        "avg_premium": avg_premium,
        "missing_values_top20": {k: int(v) for k, v in missing_report.items()},
        "feature_coverage": round(feat_cov, 6) if pd.notna(feat_cov) else None,
    }

    if passed:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = PROCESSED_DIR / f"regime_edge_{name}_{ts_str}.csv"
        subset.to_csv(out_path, index=False)
        meta["output_file"] = str(out_path)
        print(f"    -> Saved {rows:,} rows to {out_path.name}")
    else:
        meta["output_file"] = None
        print(f"    -> REJECTED (only {rows:,} rows, need {MIN_ROWS:,})")

    return meta


def generate_markdown_report(report: dict, out_path: Path) -> None:
    """Write a human-readable Markdown report."""
    lines = []
    lines.append("# Regime Edge Dataset Build Report\n")
    lines.append(f"- **Generated:** {report['build_timestamp']}\n")
    lines.append(f"- **Source Dataset:** `{report['source_dataset']}`\n")
    lines.append(f"- **Source Rows:** {report['source_rows']:,}\n")
    lines.append(f"- **Minimum Threshold:** {report['min_rows']:,} rows\n")
    lines.append("---\n")

    lines.append("## Regime Subsets\n")
    for meta in report["regimes"]:
        name = meta["regime_name"]
        status = "✅ PASSED" if meta["passed_threshold"] else "❌ REJECTED"
        lines.append(f"### {name} — {status}\n")
        lines.append(f"- **Filter:** `{meta['filter_expression']}`\n")
        lines.append(f"- **Rows:** {meta['rows']:,}\n")
        lines.append(f"- **Date Range:** {meta['date_min']} → {meta['date_max']}\n")

        ce_pe = meta["ce_pe_counts"]
        if ce_pe:
            parts = [f"{k}: {v:,}" for k, v in ce_pe.items()]
            lines.append(f"- **CE/PE Breakdown:** {', '.join(parts)}\n")

        if meta["positive_rate"] is not None:
            lines.append(f"- **Positive Rate ({meta['target_label_column']}):** {meta['positive_rate']:.4f}\n")

        if meta["avg_spread"] is not None:
            lines.append(f"- **Avg Spread ({meta['spread_column']}):** {meta['avg_spread']:.6f}\n")
        else:
            lines.append(f"- **Avg Spread:** N/A (no valid spread column)\n")

        if meta["avg_premium"] is not None:
            lines.append(f"- **Avg Premium ({meta['premium_column']}):** {meta['avg_premium']:.6f}\n")
        else:
            lines.append(f"- **Avg Premium:** N/A (no valid premium column)\n")

        lines.append(f"- **Feature Coverage:** {meta['feature_coverage']}\n")

        if meta["missing_values_top20"]:
            lines.append(f"- **Top Missing Values:**\n")
            for col, cnt in list(meta["missing_values_top20"].items())[:10]:
                lines.append(f"  - `{col}`: {cnt:,}\n")

        if meta["output_file"]:
            lines.append(f"- **Output:** `{meta['output_file']}`\n")
        lines.append("\n")

    lines.append("---\n")
    lines.append("## Available Regime Columns in Source\n")
    for col in report.get("available_regime_columns", []):
        if col in report.get("regime_column_counts", {}):
            cnt = report["regime_column_counts"][col]
            lines.append(f"- `{col}`: {cnt}\n")
        else:
            lines.append(f"- `{col}`\n")

    lines.append("\n")
    lines.append("*Research artifact. Not for live trading.*\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("Build Regime Edge Datasets")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Find latest dataset
    source_path = find_latest_dataset()
    print(f"Source dataset: {source_path}")

    # 2. Read dataset
    print("Reading dataset (this may take a moment) ...")
    df = pd.read_csv(source_path, low_memory=False)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows x {len(df.columns)} columns")

    # 3. Inspect available regime columns
    available_regime_cols = [c for c in REQUIRED_REGIME_COLS if c in df.columns]
    missing_regime_cols = [c for c in REQUIRED_REGIME_COLS if c not in df.columns]
    print(f"\nAvailable regime columns: {available_regime_cols}")
    if missing_regime_cols:
        print(f"MISSING regime columns: {missing_regime_cols}")

    regime_counts = {}
    for col in available_regime_cols:
        if col in df.columns:
            regime_counts[col] = int(df[col].notna().sum())

    # 4. Build regime subsets
    report = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_path),
        "source_rows": total_rows,
        "min_rows": MIN_ROWS,
        "available_regime_columns": available_regime_cols,
        "missing_regime_columns": missing_regime_cols,
        "regime_column_counts": regime_counts,
        "regimes": [],
    }

    print(f"\nBuilding {len(REGIME_DEFINITIONS)} regime subsets ...")
    for name, expr in REGIME_DEFINITIONS.items():
        meta = build_regime_subset(df, name, expr)
        report["regimes"].append(meta)

    # 5. Save reports
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = REPORTS_DIR / f"regime_dataset_build_{ts_str}.md"
    json_path = REPORTS_DIR / f"regime_dataset_build_{ts_str}.json"

    generate_markdown_report(report, md_path)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nReports saved:")
    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")

    # Summary
    passed = [r for r in report["regimes"] if r["passed_threshold"]]
    rejected = [r for r in report["regimes"] if not r["passed_threshold"]]
    print(f"\nSummary: {len(passed)} passed, {len(rejected)} rejected")
    for r in report["regimes"]:
        status = "PASS" if r["passed_threshold"] else "FAIL"
        print(f"  [{status}] {r['regime_name']}: {r['rows']:,} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
