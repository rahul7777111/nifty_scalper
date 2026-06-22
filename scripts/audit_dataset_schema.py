#!/usr/bin/env python3
"""
Dataset and Schema Auditor — Subagent 1
========================================
Loads the cost-aware edge dataset, audits columns, checks for leakage,
and produces structured markdown + JSON reports.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv"
)
REPORTS_DIR = REPO_ROOT / "reports"
TIMESTAMP = "20260608_110000"

# ------------------------------------------------------------------
# Expected / reference sets
# ------------------------------------------------------------------
COST_AWARE_LABELS_REQUIRED = [
    "cost_survivor_label_v2",
    "strong_profitable_trade_label_v2",
    "high_conviction_trade_label_v2",
    "weak_trade_label",
]

REGIME_COLUMNS_REQUIRED = [
    "regime_trending",
    "regime_volatile",
    "regime_mean_reverting",
    "regime_quiet",
    "volatility_regime_classifier",
    "atr_pct_regime_10",
    "realized_vol_percentile_60",
]

STRICT_FORBIDDEN_TOKENS = (
    "future",
    "realized",
    "next",
    "target",
    "label",
    "pnl",
    "profit",
    "loss",
    "return",
    "outcome",
    "exit",
    "entry_result",
    "trade_result",
    "hit_target",
    "hit_sl",
    "mae",
    "mfe",
    "forward",
    "future_price",
    "future_return",
    "post_trade",
)

FORBIDDEN_COLUMN_TOKENS = (
    "pnl",
    "profit",
    "target",
    "label",
    "future",
    "forward",
    "next",
    "exit",
    "loss",
    "outcome",
    "next_return",
    "net_forward_return",
    "gross_forward_return",
    "return",
    "realized",
    "trade_result",
    "post_entry_return",
)

# Non-leaky exceptions for "realized"
NON_LEAKY_REALIZED_FEATURE_TOKENS = (
    "realized_vol",
    "realized_volatility",
    "realized_vol_percentile",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_forbidden_feature_name(name: str) -> bool:
    lowered = name.lower()
    if any(lowered.startswith(tok) for tok in NON_LEAKY_REALIZED_FEATURE_TOKENS):
        return False
    return any(token in lowered for token in STRICT_FORBIDDEN_TOKENS) or any(
        token in lowered for token in FORBIDDEN_COLUMN_TOKENS
    )


def _classify_label(col: str) -> str:
    lc = col.lower()
    if "survivor" in lc:
        return "survivor"
    if "strong" in lc or "profitable" in lc:
        return "strong_profit"
    if "high" in lc or "conviction" in lc:
        return "high_conviction"
    if "weak" in lc:
        return "weak"
    if "no_trade" in lc or "avoid" in lc:
        return "avoid"
    if "paper" in lc:
        return "paper"
    return "other"


# ------------------------------------------------------------------
# Main audit
# ------------------------------------------------------------------

def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset: {DATASET_PATH.name}")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ---------- 1. Basic stats ----------
    n_rows, n_cols = len(df), len(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ---------- 2. Timestamp audit ----------
    timestamp_cols = [c for c in df.columns if "time" in c.lower() or c.lower() == "timestamp"]
    time_sorted = None
    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            time_sorted = bool(ts.is_monotonic_increasing)
        except Exception:
            time_sorted = None

    # ---------- 3. Identify category columns ----------
    symbol_cols = [c for c in df.columns if any(tok in c.lower() for tok in ("symbol", "instrument", "token"))]
    strike_cols = [c for c in df.columns if "strike" in c.lower()]
    expiry_cols = [c for c in df.columns if "expir" in c.lower() or c.lower() == "dte_days"]
    option_type_cols = [c for c in df.columns if "option_type" in c.lower()]

    ohlcv_cols = [c for c in df.columns if c.lower() in {"open", "high", "low", "close", "ltp", "volume", "oi"}]
    bid_ask_cols = [c for c in df.columns if any(tok in c.lower() for tok in ("bid", "ask", "spread", "mid_price", "premium"))]

    label_cols = [c for c in df.columns if "label" in c.lower()]
    label_cols_v2 = [c for c in label_cols if "v2" in c.lower()]

    return_cols = [c for c in df.columns if any(tok in c.lower() for tok in ("return", "forward_return", "gross_return", "net_return"))]
    pnl_cols = [c for c in df.columns if "pnl" in c.lower()]
    score_cols = [c for c in df.columns if any(tok in c.lower() for tok in ("score", "expected_return"))]

    # ---------- 4. Forbidden columns ----------
    forbidden_cols = sorted([c for c in df.columns if _is_forbidden_feature_name(c)])
    forbidden_detail = {c: f"matches token(s) in forbidden set" for c in forbidden_cols}

    # ---------- 5. Duplicates / missing ----------
    dup_rows = int(df.duplicated().sum())
    missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    missing_pct = {c: round(100 * int(v) / n_rows, 4) for c, v in missing.items()}

    # ---------- 6. Class imbalance for labels ----------
    label_imbalance: Dict[str, Dict[str, Any]] = {}
    for col in label_cols:
        vc = df[col].value_counts(dropna=False).to_dict()
        total = sum(vc.values())
        pos_rate = None
        if set(vc.keys()).issubset({0, 1, 0.0, 1.0, True, False, np.nan}):
            pos = sum(v for k, v in vc.items() if k in (1, 1.0, True))
            pos_rate = round(pos / total, 4) if total else None
        label_imbalance[col] = {
            "total": total,
            "value_counts": {str(k): int(v) for k, v in vc.items()},
            "positive_rate": pos_rate,
        }

    # ---------- 7. Regime columns existence ----------
    regime_exists = {c: c in df.columns for c in REGIME_COLUMNS_REQUIRED}

    # ---------- 8. Cost-aware label existence ----------
    cost_label_exists = {c: c in df.columns for c in COST_AWARE_LABELS_REQUIRED}

    # ---------- 9. Time-series sanity ----------
    time_gaps = None
    if "timestamp" in df.columns:
        try:
            ts_sorted = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).sort_values()
            diffs = ts_sorted.diff().dropna()
            if len(diffs) > 0:
                median_gap = diffs.median()
                max_gap = diffs.max()
                time_gaps = {
                    "median_gap_seconds": median_gap.total_seconds() if hasattr(median_gap, "total_seconds") else None,
                    "max_gap_seconds": max_gap.total_seconds() if hasattr(max_gap, "total_seconds") else None,
                }
        except Exception:
            time_gaps = None

    # ---------- 10. Cross-section uniqueness (timestamp + symbol + strike + option_type) ----------
    id_cols = [c for c in ["timestamp", "trading_symbol", "strike_price", "option_type"] if c in df.columns]
    if id_cols:
        dup_keys = int(df[id_cols].duplicated().sum())
    else:
        dup_keys = None

    # ---------- 11. Build structured report ----------
    report: Dict[str, Any] = {
        "audit_timestamp": TIMESTAMP,
        "dataset_path": str(DATASET_PATH),
        "dataset_name": DATASET_PATH.name,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "basic": {
            "numeric_columns": numeric_cols,
            "object_columns": object_cols,
            "dtype_summary": dtypes,
        },
        "timestamp_audit": {
            "timestamp_columns": timestamp_cols,
            "time_sorted": time_sorted,
            "time_gaps": time_gaps,
        },
        "instrument_identity": {
            "symbol_columns": symbol_cols,
            "strike_columns": strike_cols,
            "expiry_columns": expiry_cols,
            "option_type_columns": option_type_cols,
        },
        "price_volume": {
            "ohlcv_columns": ohlcv_cols,
            "bid_ask_columns": bid_ask_cols,
        },
        "labels": {
            "all_label_columns": label_cols,
            "v2_label_columns": label_cols_v2,
            "cost_aware_labels_exist": cost_label_exists,
            "class_imbalance": label_imbalance,
        },
        "returns_and_pnl": {
            "return_columns": return_cols,
            "pnl_columns": pnl_cols,
            "score_columns": score_cols,
        },
        "forbidden_columns": {
            "strict_forbidden_tokens": list(STRICT_FORBIDDEN_TOKENS),
            "forbidden_column_tokens": list(FORBIDDEN_COLUMN_TOKENS),
            "forbidden_columns_found": forbidden_cols,
            "forbidden_count": len(forbidden_cols),
            "detail": forbidden_detail,
        },
        "regime_columns": {
            "required_regime_columns": REGIME_COLUMNS_REQUIRED,
            "regime_columns_exist": regime_exists,
        },
        "data_quality": {
            "duplicate_rows": dup_rows,
            "duplicate_key_rows": dup_keys,
            "missing_values": missing,
            "missing_percent": missing_pct,
            "columns_with_missing": list(missing.keys()),
        },
    }

    # ---------- 12. Problems & required changes ----------
    problems: List[str] = []
    required_changes: List[str] = []

    if time_sorted is False:
        problems.append("Dataset is NOT sorted by timestamp (monotonic).")
        required_changes.append("Sort dataset by timestamp before any time-based split.")

    if dup_rows > 0:
        problems.append(f"Found {dup_rows:,} fully duplicate rows.")
        required_changes.append("Drop duplicate rows before training.")

    if dup_keys and dup_keys > 0:
        problems.append(f"Found {dup_keys:,} duplicate (timestamp, symbol, strike, option_type) keys.")
        required_changes.append("Investigate duplicate keys; may indicate multiple records per snapshot.")

    missing_critical = [c for c in missing if missing[c] > n_rows * 0.01]
    if missing_critical:
        problems.append(f"Columns with >1% missing: {missing_critical}")
        required_changes.append("Impute or drop high-missing columns before training.")

    for col, exists in cost_label_exists.items():
        if not exists:
            problems.append(f"Required cost-aware label missing: {col}")
            required_changes.append(f"Add/reconstruct {col} in dataset build pipeline.")

    for col in COST_AWARE_LABELS_REQUIRED:
        if col in df.columns:
            pos_rate = label_imbalance.get(col, {}).get("positive_rate")
            if pos_rate is not None and not (0.20 <= pos_rate <= 0.50):
                problems.append(f"Label '{col}' positive rate is {pos_rate} (outside 20%-50% target).")

    # cost_survivor_label_v2 positive rate expectation ~33%
    if "cost_survivor_label_v2" in df.columns:
        csr = label_imbalance.get("cost_survivor_label_v2", {}).get("positive_rate")
        if csr is not None:
            if abs(csr - 0.33) > 0.10:
                problems.append(f"cost_survivor_label_v2 positive rate={csr}, expected ~0.33 ±0.10.")
            else:
                report["labels"]["cost_survivor_label_v2_positive_rate_ok"] = True

    for col, exists in regime_exists.items():
        if not exists:
            problems.append(f"Required regime column missing: {col}")
            required_changes.append(f"Add {col} to feature engineering pipeline.")

    if forbidden_cols:
        problems.append(f"Found {len(forbidden_cols)} forbidden columns that MUST NOT be used as features.")
        required_changes.append("Strip all forbidden columns from feature matrix before model fitting.")

    report["problems"] = problems
    report["required_changes"] = required_changes
    report["verdict"] = "PASS" if not problems else "FAIL"

    # ---------- 13. Write reports ----------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / f"dataset_schema_audit_{TIMESTAMP}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote JSON report: {json_path}")

    md_path = REPORTS_DIR / f"dataset_schema_audit_{TIMESTAMP}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))
    print(f"Wrote Markdown report: {md_path}")

    print(f"\nAudit verdict: {report['verdict']}")
    if problems:
        print(f"Problems found ({len(problems)}):")
        for p in problems:
            print(f"  • {p}")
    if required_changes:
        print(f"Required changes ({len(required_changes)}):")
        for r in required_changes:
            print(f"  → {r}")

    return 0


# ------------------------------------------------------------------
# Markdown renderer
# ------------------------------------------------------------------

def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Dataset & Schema Audit Report")
    lines.append(f"**Audit ID:** `{report['audit_timestamp']}`")
    lines.append(f"**Dataset:** `{report['dataset_name']}`")
    lines.append(f"**Rows:** {report['n_rows']:,} | **Columns:** {report['n_columns']}")
    lines.append("")

    # Verdict
    v = report["verdict"]
    badge = "PASS" if v == "PASS" else "FAIL"
    lines.append(f"## Verdict: {badge}")
    lines.append("")

    # Timestamp
    t = report["timestamp_audit"]
    lines.append("## Timestamp Audit")
    lines.append(f"- Timestamp columns: `{t['timestamp_columns']}`")
    lines.append(f"- Sorted by time: `{t['time_sorted']}`")
    if t.get("time_gaps"):
        lines.append(f"- Median gap (sec): {t['time_gaps']['median_gap_seconds']}")
        lines.append(f"- Max gap (sec): {t['time_gaps']['max_gap_seconds']}")
    lines.append("")

    # Instrument identity
    i = report["instrument_identity"]
    lines.append("## Instrument Identity Columns")
    lines.append(f"- Symbol: `{i['symbol_columns']}`")
    lines.append(f"- Strike: `{i['strike_columns']}`")
    lines.append(f"- Expiry/DTE: `{i['expiry_columns']}`")
    lines.append(f"- Option type: `{i['option_type_columns']}`")
    lines.append("")

    # Price / volume
    p = report["price_volume"]
    lines.append("## Price / Volume Columns")
    lines.append(f"- OHLCV: `{p['ohlcv_columns']}`")
    lines.append(f"- Bid/Ask/Spread: `{p['bid_ask_columns']}`")
    lines.append("")

    # Labels
    l = report["labels"]
    lines.append("## Label Columns")
    lines.append(f"- All labels: `{l['all_label_columns']}`")
    lines.append(f"- V2 labels: `{l['v2_label_columns']}`")
    lines.append("")
    lines.append("### Cost-Aware Label Existence")
    for col, exists in l["cost_aware_labels_exist"].items():
        mark = "✅" if exists else "❌"
        lines.append(f"- {mark} `{col}`")
    lines.append("")
    lines.append("### Class Imbalance")
    for col, info in l["class_imbalance"].items():
        pos = info.get("positive_rate")
        if pos is not None:
            lines.append(f"- `{col}`: positive_rate={pos}, counts={info['value_counts']}")
        else:
            lines.append(f"- `{col}`: counts={info['value_counts']}")
    lines.append("")

    # Returns / PnL
    r = report["returns_and_pnl"]
    lines.append("## Return / PnL / Scoring Columns")
    lines.append(f"- Return cols: `{r['return_columns']}`")
    lines.append(f"- PnL cols: `{r['pnl_columns']}`")
    lines.append(f"- Score cols: `{r['score_columns']}`")
    lines.append("")

    # Regime
    lines.append("## Regime Columns")
    for col, exists in report["regime_columns"]["regime_columns_exist"].items():
        mark = "✅" if exists else "❌"
        lines.append(f"- {mark} `{col}`")
    lines.append("")

    # Forbidden
    f = report["forbidden_columns"]
    lines.append("## Forbidden Columns (MUST NOT be used as model input)")
    lines.append(f"**Count:** {f['forbidden_count']}")
    for c in f["forbidden_columns_found"]:
        lines.append(f"- ❌ `{c}`")
    lines.append("")

    # Data quality
    d = report["data_quality"]
    lines.append("## Data Quality")
    lines.append(f"- Duplicate rows: {d['duplicate_rows']:,}")
    lines.append(f"- Duplicate key rows: {d['duplicate_key_rows']}")
    lines.append(f"- Columns with any missing: `{d['columns_with_missing']}`")
    lines.append("")
    if d["missing_values"]:
        lines.append("| Column | Missing | Missing % |")
        lines.append("|--------|---------|-----------|")
        for c, v in d["missing_values"].items():
            pct = d["missing_percent"][c]
            lines.append(f"| {c} | {v:,} | {pct}% |")
    lines.append("")

    # Problems & required changes
    lines.append("## Problems Found")
    if report["problems"]:
        for p in report["problems"]:
            lines.append(f"- ⚠️ {p}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Required Code Changes")
    if report["required_changes"]:
        for r in report["required_changes"]:
            lines.append(f"- {r}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by Subagent 1 (Dataset and Schema Auditor)*")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
