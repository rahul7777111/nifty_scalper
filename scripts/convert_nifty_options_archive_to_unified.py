#!/usr/bin/env python
r"""
Convert the raw Nifty options historical archive (daily CSVs, unsorted across tree)
into unified, chronologically sorted (by timestamp asc, then instrument_key) datasets
with the *exact* same columns (197) as the project's cost-aware edge datasets.

- Recursively finds all nifty_options_*.csv under the source dir.
- Parses the compact symbol (NIFTYDDMMMYYSTRIKECP) into expiry/strike/option_type.
- Builds timestamp (Asia/Kolkata), maps close->ltp, constructs instrument_key/trading_symbol/weekly.
- Sorts globally (per-year for memory).
- Adds the full live-computable feature set via the project's option_chain_pipeline_lib
  (_add_research_features + reconstruct_missing_live_features_for_option_chain).
- Pads all remaining columns from the canonical 197-col schema with NaN (spot, BS greeks,
  bid/ask, some flags, enrichment metadata are not computable without additional source data).
- Computes the full set of labels (original + V1 cost-aware + V2 + cost-stress) using
  the same logic as build_cost_aware_edge_dataset (forward 3-bar, cost estimates via
  ml_execution_costs with fallback when no spread data).
- Writes one CSV per year (sorted) under data/processed/historical_unified_nifty_options/
  plus a manifest + small head sample.

Output is directly usable for the same retrain / audit pipelines (with the understanding
that spot-dependent and quote-quality features will be NaN or defaulted).

Usage:
  python scripts/convert_nifty_options_archive_to_unified.py \
      --source "C:/Users/rahul/Downloads/archive/nifty_data/nifty_options" \
      --years 2024          # optional, comma sep or 'all'
      --output-dir data/processed/historical_unified_nifty_options
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from option_chain_pipeline_lib import (
    _add_research_features,
    _add_labels as _lib_add_labels,
    reconstruct_missing_live_features_for_option_chain,
    TIMEZONE,
)
from ml_execution_costs import estimate_option_execution_costs_frame

# Canonical columns from the latest cost-aware edge dataset (197 total).
# We ensure the output always has exactly these, in this order (or close).
CANONICAL_COLUMNS: List[str] = [
    "timestamp", "open", "high", "low", "ltp", "volume", "oi", "expiry", "instrument_key",
    "trading_symbol", "option_type", "strike_price", "weekly", "source_file", "trading_day",
    "dte_days", "is_weekly", "option_type_ce", "option_type_pe", "range_pct", "oc_change_pct",
    "hl_change_pct", "oi_change_pct", "volume_change_pct", "ret_1", "ret_3", "ret_5", "oi_z_5",
    "volume_z_5", "weekday", "month", "future_close", "gross_forward_return", "net_forward_return",
    "profitable_trade_label", "avoid_trade_label", "trading_day_spot", "open_spot", "high_spot",
    "low_spot", "close", "volume_spot", "spot_close", "spot_return_1", "spot_return_3", "spot_return_5",
    "spot_range_pct", "spot_atr", "spot_rsi", "spot_vwap", "ctx_time_sin", "ctx_time_cos",
    "is_opening_session", "is_closing_session", "is_midday_lull", "weekday_spot", "spot_source",
    "ctx_spot", "distance_from_spot", "moneyness", "atm_distance", "strike_distance_pct", "ctx_dte_norm",
    "last_open", "last_high", "last_low", "last_close", "last_volume", "body_pct", "gap_pct",
    "upper_wick_pct", "lower_wick_pct", "close_location_pct", "ret_10", "ema_fast", "ema_slow",
    "ema_diff_pct", "rsi_14", "atr_14", "atr_pct", "ctx_option_price", "option_to_spot_pct",
    "volume_ratio", "bullish_engulfing", "bearish_engulfing", "doji", "hammer", "shooting_star",
    "ret_mean", "ret_std", "ret_min", "ret_max", "vol_mean", "vol_std", "vol_min", "vol_max",
    "adx_14", "choppiness_14", "supertrend_dir", "pivot_pp_dist_pct", "pivot_r1_dist_pct",
    "pivot_s1_dist_pct", "close_vs_open_pct", "range_to_atr", "momentum_lookback_pct",
    "regime_trending", "regime_volatile", "regime_mean_reverting", "regime_quiet", "ctx_adx",
    "ctx_trend_strength", "ctx_choppiness", "ctx_volume_sma", "vol_of_vol_14",
    "dist_from_opening_high_pct", "dist_from_opening_low_pct", "opening_range_width_pct",
    "opening_range_breakout_strength", "dist_to_rolling_high_20", "dist_to_rolling_low_20",
    "rolling_range_width_20", "rolling_range_position_20", "realized_vol_30", "atr_pct_regime_10",
    "atr_percentile_60", "realized_vol_percentile_60", "volatility_percentile_60",
    "volatility_regime_classifier", "oi_CE", "oi_PE", "volume_CE", "volume_PE", "ce_pe_oi_ratio",
    "ce_pe_volume_ratio", "bs_iv", "bs_iv_solve_ok", "bs_iv_source", "final_iv", "bs_delta",
    "bs_gamma", "bs_theta", "bs_vega", "bs_rho", "log_moneyness", "distance_from_atm",
    "distance_from_atm_pct", "intrinsic_value", "extrinsic_value", "time_to_expiry_days",
    "time_to_expiry_years", "is_expiry_day", "is_near_expiry", "option_side_normalized",
    "bid_ask_spread", "bid_ask_spread_pct", "mid_price", "ltp_vs_mid_diff", "ltp_vs_mid_diff_pct",
    "has_valid_bid_ask", "low_price_flag", "wide_spread_flag", "bad_iv_flag", "bad_greek_flag",
    "stale_or_invalid_quote_flag", "deep_itm_flag", "deep_otm_flag", "greeks_source",
    "greeks_quality_score", "row_enrichment_ok", "row_enrichment_error",
    # Cost-aware labels + diags (V1)
    "strong_profitable_trade_label", "high_conviction_trade_label", "weak_trade_label",
    "no_trade_label", "cost_survivor_label", "paper_candidate_label",
    "expected_return_after_cost", "return_to_cost_ratio", "cost_return_units_estimated",
    # V2
    "strong_profitable_trade_label_v2", "cost_survivor_label_v2", "paper_candidate_label_v2",
    "high_conviction_trade_label_v2",
    # More returns / cost stress
    "gross_return", "net_return_after_cost", "estimated_cost_bps", "spread_cost_component",
    "slippage_cost_component", "brokerage_or_fee_component",
    "cost_stress_1_0x_return", "cost_stress_1_5x_return", "cost_stress_2_0x_return",
    "cost_stress_1_0x_label", "cost_stress_1_5x_label", "cost_stress_2_0x_label",
    "cost_return_units", "cost_pct_of_premium",
]

# The columns that are strictly outcomes / leakage (will be computed but must not be used as features)
FORBIDDEN_FOR_FEATURES = {
    "future_close", "gross_forward_return", "net_forward_return",
    "profitable_trade_label", "avoid_trade_label",
    "strong_profitable_trade_label", "high_conviction_trade_label", "weak_trade_label",
    "no_trade_label", "cost_survivor_label", "paper_candidate_label",
    "strong_profitable_trade_label_v2", "cost_survivor_label_v2",
    "paper_candidate_label_v2", "high_conviction_trade_label_v2",
    "cost_stress_1_0x_return", "cost_stress_1_5x_return", "cost_stress_2_0x_return",
    "cost_stress_1_0x_label", "cost_stress_1_5x_label", "cost_stress_2_0x_label",
    "net_return_after_cost",
}

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
SYMBOL_RE = re.compile(r"NIFTY(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$", re.I)


def parse_nifty_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    m = SYMBOL_RE.match(str(symbol).upper().strip())
    if not m:
        return None
    dd, mmm, yy, strike_str, typ = m.groups()
    year = 2000 + int(yy)
    month = MONTH_MAP.get(mmm, 1)
    try:
        exp = date(year, month, int(dd))
    except ValueError:
        return None
    strike = float(strike_str)
    return {
        "expiry": exp,
        "strike_price": strike,
        "option_type": typ,
        "trading_symbol": str(symbol),
    }


def build_instrument_key(parsed: Dict[str, Any], raw_symbol: str) -> str:
    exp = parsed["expiry"]
    sk = int(parsed["strike_price"])
    return f"ARCHIVE_NIFTY_{exp:%Y%m%d}_{sk}_{parsed['option_type']}"


def discover_files(source_dir: Path) -> List[Path]:
    files = sorted(source_dir.rglob("nifty_options_*.csv"))
    # Sort by (year, month, day) extracted from path/name for chronological processing
    def key(p: Path):
        m = re.search(r"(\d{4})[/\\](\d{1,2})[/\\]nifty_options_(\d{2})_(\d{2})_(\d{4})", str(p))
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        # fallback lexical
        return (9999, 99, 99, 99, 9999)
    files.sort(key=key)
    return files


def load_and_parse_year(files: Iterable[Path], year: int) -> pd.DataFrame:
    """Load all daily files for a year (vectorized per file for speed), parse to base schema."""
    frames: List[pd.DataFrame] = []
    file_list = [f for f in files if str(year) in str(f) or f.name.endswith(f"_{year}.csv")]
    total_f = len(file_list)
    print(f"  [year {year}] scanning {total_f} candidate daily files (vectorized parse)...")
    seen = 0
    for f in file_list:
        seen += 1
        if seen % 30 == 0 or seen <= 3 or seen == total_f:
            print(f"    processed {seen}/{total_f} files, accumulated_frames={len(frames)}")
        try:
            df = pd.read_csv(f, usecols=["date", "time", "symbol", "open", "high", "low", "close", "oi", "volume"])
        except Exception:
            df = pd.read_csv(f)
        if df.empty:
            continue
        # Vectorized symbol parse
        syms = df["symbol"].astype(str).str.upper().str.strip()
        parsed = syms.apply(parse_nifty_symbol)
        valid_mask = parsed.notna()
        if not valid_mask.any():
            continue
        sub = df.loc[valid_mask].copy()
        pser = parsed.loc[valid_mask]

        sub["expiry"] = pser.apply(lambda x: x["expiry"])
        sub["strike_price"] = pser.apply(lambda x: x["strike_price"])
        sub["option_type"] = pser.apply(lambda x: x["option_type"])
        sub["trading_symbol"] = pser.apply(lambda x: x["trading_symbol"])

        sub["timestamp"] = pd.to_datetime(sub["date"].astype(str) + " " + sub["time"].astype(str), errors="coerce")
        sub = sub[sub["timestamp"].notna()]
        if sub.empty:
            continue

        sub["ltp"] = pd.to_numeric(sub["close"], errors="coerce")
        sub["instrument_key"] = (
            "ARCHIVE_NIFTY_" +
            sub["expiry"].astype(str).str.replace("-", "", regex=False) + "_" +
            sub["strike_price"].astype(int).astype(str) + "_" +
            sub["option_type"].astype(str)
        )
        sub["weekly"] = True
        sub["source_file"] = f.name
        sub["open"] = pd.to_numeric(sub["open"], errors="coerce")
        sub["high"] = pd.to_numeric(sub["high"], errors="coerce")
        sub["low"] = pd.to_numeric(sub["low"], errors="coerce")
        sub["volume"] = pd.to_numeric(sub["volume"], errors="coerce").fillna(0).astype(int)
        sub["oi"] = pd.to_numeric(sub["oi"], errors="coerce")

        keep = ["timestamp", "open", "high", "low", "ltp", "volume", "oi",
                "expiry", "strike_price", "option_type", "trading_symbol",
                "instrument_key", "weekly", "source_file"]
        frames.append(sub[keep])

    print(f"  [year {year}] vectorized parse complete, num_frames={len(frames)}")
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    base = pd.concat(frames, ignore_index=True)
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    base["expiry"] = pd.to_datetime(base["expiry"])
    base["trading_day"] = base["timestamp"].dt.normalize()
    base["option_type"] = base["option_type"].astype(str).str.upper()
    print(f"  [year {year}] final base rows for year: {len(base):,}")
    return base


def pad_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has exactly the CANONICAL_COLUMNS (in order), adding missing as NaN."""
    work = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in work.columns:
            work[col] = np.nan
    # Keep only canonical + any extra we don't want to drop yet, but reorder to canonical first
    extras = [c for c in work.columns if c not in CANONICAL_COLUMNS]
    ordered = CANONICAL_COLUMNS + extras
    work = work[ordered]
    # Cast a few known types
    for c in ["strike_price", "ltp", "open", "high", "low", "volume", "oi"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    return work


def add_cost_aware_labels_and_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Layer the full cost-aware V1/V2 + stress labels on top of a DF that already has net_forward_return (or compute)."""
    work = df.copy()
    if "net_forward_return" not in work.columns or work["net_forward_return"].isna().all():
        # Compute basic forward labels first (3 bar)
        labeled, _ = _lib_add_labels(work, horizon_bars=3)
        work = labeled
    else:
        labeled = work

    # Now the rich cost-aware logic (simplified but faithful to build_cost_aware_edge_dataset)
    ret = pd.to_numeric(work.get("net_forward_return", work.get("gross_forward_return", pd.Series(np.nan, index=work.index))), errors="coerce")
    # cost_return_units: use embedded 0.0025 + spread component if available, else estimate frame (falls back)
    embedded = 0.0025
    cost_ru = pd.Series(np.full(len(work), embedded, dtype=float), index=work.index)
    try:
        # If we had bid_ask_spread_pct we would pass real; here it will use embedded inside the estimator
        cf = estimate_option_execution_costs_frame(work)
        if not cf.empty and "cost_pct_of_premium" in cf.columns:
            cost_ru = cf["cost_pct_of_premium"].astype(float).fillna(embedded)
    except Exception:
        pass

    safe_cost = cost_ru.replace(0.0, np.nan)
    return_to_cost = ret / safe_cost
    expected_after = ret - cost_ru

    # V1
    work["strong_profitable_trade_label"] = (
        (ret > cost_ru * 1.5) & (return_to_cost >= 1.5) & (expected_after > 0.0)
    ).astype(float)
    work.loc[ret.isna(), "strong_profitable_trade_label"] = np.nan

    work["high_conviction_trade_label"] = (
        (ret > cost_ru * 2.0) & (return_to_cost >= 2.0)
    ).astype(float)
    work.loc[ret.isna(), "high_conviction_trade_label"] = np.nan

    work["weak_trade_label"] = ((ret > 0.0) & (return_to_cost < 1.25)).astype(float)
    work.loc[ret.isna(), "weak_trade_label"] = np.nan

    work["no_trade_label"] = (ret.notna() & (expected_after <= 0.0)).astype(float)
    work.loc[ret.isna(), "no_trade_label"] = np.nan

    work["cost_survivor_label"] = ((ret - 1.25 * cost_ru) > 0.0).astype(float)
    work.loc[ret.isna(), "cost_survivor_label"] = np.nan

    work["paper_candidate_label"] = (
        (return_to_cost >= 1.5) & ((ret - 1.25 * cost_ru) > 0.0)
    ).astype(float)
    work.loc[ret.isna(), "paper_candidate_label"] = np.nan

    # Diags
    work["expected_return_after_cost"] = expected_after
    work["return_to_cost_ratio"] = return_to_cost
    work["cost_return_units_estimated"] = cost_ru

    # V2 (relaxed)
    work["strong_profitable_trade_label_v2"] = (
        (ret > cost_ru * 1.25) & (return_to_cost >= 1.25) & (expected_after > 0.0)
    ).astype(float)
    work.loc[ret.isna(), "strong_profitable_trade_label_v2"] = np.nan

    work["cost_survivor_label_v2"] = ((ret - 1.25 * cost_ru) > 0.0).astype(float)
    work.loc[ret.isna(), "cost_survivor_label_v2"] = np.nan

    work["paper_candidate_label_v2"] = (
        (return_to_cost >= 1.25) & ((ret - 1.25 * cost_ru) > 0.0)
    ).astype(float)
    work.loc[ret.isna(), "paper_candidate_label_v2"] = np.nan

    work["high_conviction_trade_label_v2"] = (
        (ret > cost_ru * 1.5) & (return_to_cost >= 1.5)
    ).astype(float)
    work.loc[ret.isna(), "high_conviction_trade_label_v2"] = np.nan

    # Stress + raw cost
    work["gross_return"] = work.get("gross_forward_return", ret)
    work["net_return_after_cost"] = ret - cost_ru
    work["estimated_cost_bps"] = (cost_ru * 10000.0)  # rough
    work["spread_cost_component"] = np.nan
    work["slippage_cost_component"] = np.nan
    work["brokerage_or_fee_component"] = np.nan

    work["cost_stress_1_0x_return"] = ret - 1.0 * cost_ru
    work["cost_stress_1_5x_return"] = ret - 1.5 * cost_ru
    work["cost_stress_2_0x_return"] = ret - 2.0 * cost_ru
    work["cost_stress_1_0x_label"] = (work["cost_stress_1_0x_return"] > 0.0).astype(float)
    work["cost_stress_1_5x_label"] = (work["cost_stress_1_5x_return"] > 0.0).astype(float)
    work["cost_stress_2_0x_label"] = (work["cost_stress_2_0x_return"] > 0.0).astype(float)

    work["cost_return_units"] = cost_ru
    work["cost_pct_of_premium"] = cost_ru  # already in return units for parity with net_forward_return

    # Fill NaNs on label rows where no future
    for c in [c for c in work.columns if "_label" in c]:
        if c in work.columns:
            work.loc[work.get("future_close", pd.Series(True, index=work.index)).isna() & work[c].notna(), c] = np.nan

    return work


def process_year(files_for_year: List[Path], year: int, out_dir: Path) -> Dict[str, Any]:
    print(f"[convert] Processing year {year} ({len(files_for_year)} files)...")
    base = load_and_parse_year(files_for_year, year)
    if base.empty:
        print(f"[convert]   Year {year}: no usable rows after parse.")
        return {"year": year, "rows": 0}

    print(f"[convert]   Sorting base for year {year}...")
    base = base.sort_values(["timestamp", "instrument_key"], kind="stable").reset_index(drop=True)

    print(f"[convert]   Adding research features (basic live set) for {len(base):,} rows...")
    enriched, _, _ = _add_research_features(base)

    # Full reconstruct of missing live features (the bulk of the 160)
    print(f"[convert]   Running reconstruct_missing_live_features (heavy step, ~45 extra features)...")
    tmp_in = out_dir / f"_tmp_year_{year}_in.csv"
    enriched.to_csv(tmp_in, index=False)
    rec_out = out_dir / f"_tmp_year_{year}_reconstructed.csv"
    rec_report = reconstruct_missing_live_features_for_option_chain(
        tmp_in, output_dataset=rec_out
    )
    tmp_in.unlink(missing_ok=True)

    if not rec_out.exists():
        print(f"[convert]   WARN: reconstruct did not produce output for {year}, using enriched only.")
        df_feat = enriched
    else:
        print(f"[convert]   Loading reconstructed output for year {year}...")
        df_feat = pd.read_csv(rec_out, low_memory=False)
        rec_out.unlink(missing_ok=True)

    print(f"[convert]   Padding to full 197-col canonical schema + adding cost-aware labels/stress...")
    df_full = pad_to_canonical(df_feat)
    df_full = add_cost_aware_labels_and_stress(df_full)

    print(f"[convert]   Final sort + column alignment for {year}...")
    df_full = df_full.sort_values(["timestamp", "instrument_key"], kind="stable").reset_index(drop=True)
    df_full = df_full[[c for c in CANONICAL_COLUMNS if c in df_full.columns]]
    df_full = pad_to_canonical(df_full)
    df_full = df_full[CANONICAL_COLUMNS]

    out_path = out_dir / f"nifty_option_chain_historical_unified_{year}.csv"
    print(f"[convert]   Writing {len(df_full):,} rows x {len(df_full.columns)} cols -> {out_path.name}")
    df_full.to_csv(out_path, index=False)

    print(f"[convert]   Year {year} COMPLETE: {len(df_full):,} rows, {len(df_full.columns)} columns.")
    return {
        "year": year,
        "rows": int(len(df_full)),
        "columns": len(df_full.columns),
        "path": str(out_path),
        "date_range": {
            "start": str(df_full["timestamp"].min()) if len(df_full) else None,
            "end": str(df_full["timestamp"].max()) if len(df_full) else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Root of the nifty_options archive (contains 2020/, 2021/, ...)")
    ap.add_argument("--output-dir", default="data/processed/historical_unified_nifty_options")
    ap.add_argument("--years", default="all", help="Comma-separated years or 'all' (default)")
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = discover_files(source)
    print(f"[convert] Discovered {len(all_files)} daily files under {source}")

    if args.years.lower() == "all":
        years = sorted({int(re.search(r"(\d{4})", str(p)).group(1)) for p in all_files if re.search(r"(\d{4})", str(p))})
    else:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    print(f"[convert] Target years: {years}")

    manifest_entries = []
    for y in years:
        files_y = [p for p in all_files if str(y) in str(p)]
        entry = process_year(files_y, y, out_dir)
        manifest_entries.append(entry)

    # Write manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output_dir": str(out_dir),
        "years": years,
        "total_rows": sum(e.get("rows", 0) for e in manifest_entries),
        "columns": len(CANONICAL_COLUMNS),
        "canonical_columns": CANONICAL_COLUMNS,
        "forbidden_feature_columns": sorted(FORBIDDEN_FOR_FEATURES),
        "notes": [
            "Data is sorted by timestamp (ascending) then instrument_key within each yearly file.",
            "Features that require spot underlying (spot_*, many regime/distances), Black-Scholes greeks/IV (no spot + premium to solve), and bid/ask quote data are NaN or defaulted.",
            "Labels (including cost-aware V1/V2 and stress) are computed with 3-bar forward horizon using the same formulas as the project's cost-aware builder.",
            "instrument_key uses ARCHIVE_ prefix (not exchange token). Group-by on it is still valid for walk-forward and per-contract features.",
            "Use these files with the same retrain/audit scripts as the live cost-aware datasets (filter or impute NaNs as appropriate for your experiment).",
        ],
        "files": manifest_entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Small head sample across years (first 200 rows of first available year file)
    sample_path = out_dir / "sample_head_200.csv"
    for e in manifest_entries:
        p = Path(e.get("path", ""))
        if p.exists():
            try:
                pd.read_csv(p, nrows=200).to_csv(sample_path, index=False)
                break
            except Exception:
                pass

    print("\n[convert] DONE.")
    print(f"Output directory : {out_dir}")
    print(f"Manifest         : {manifest_path}")
    print(f"Sample (head)    : {sample_path if sample_path.exists() else 'N/A'}")
    print(f"Total rows across years: {manifest['total_rows']:,}")
    print(f"Columns (exact match to cost-aware): {manifest['columns']}")
    print("\nYearly files:")
    for e in manifest_entries:
        print(f"  {e.get('path')}  ({e.get('rows',0):,} rows)")


if __name__ == "__main__":
    main()
