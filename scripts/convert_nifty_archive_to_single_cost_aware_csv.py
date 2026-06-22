#!/usr/bin/env python
"""Build one ML-ready 197-column CSV from the raw Nifty options archive.

This is a memory-safer path for the large archive in
``C:/Users/rahul/Downloads/archive/nifty_data``.  It processes one year at a
time, derives option-only features and labels, pads the exact cost-aware edge
dataset schema, and appends year outputs into one globally timestamp-sorted CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_nifty_options_archive_to_unified import (  # noqa: E402
    CANONICAL_COLUMNS,
    discover_files,
    load_and_parse_year,
    pad_to_canonical,
)


BOOL_FALSE_COLUMNS = [
    "bs_iv_solve_ok",
    "is_expiry_day",
    "is_near_expiry",
    "has_valid_bid_ask",
    "low_price_flag",
    "wide_spread_flag",
    "bad_iv_flag",
    "bad_greek_flag",
    "stale_or_invalid_quote_flag",
    "deep_itm_flag",
    "deep_otm_flag",
]

START_TIME = time.perf_counter()


def fmt_elapsed(seconds: float) -> str:
    hours, rem = divmod(int(max(seconds, 0.0)), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log_progress(message: str, *, year: int | None = None, stage_start: float | None = None) -> None:
    total = fmt_elapsed(time.perf_counter() - START_TIME)
    prefix = f"[single-convert {total}]"
    if year is not None:
        prefix += f" year={year}"
    if stage_start is not None:
        message = f"{message} | stage {fmt_elapsed(time.perf_counter() - stage_start)}"
    print(f"{prefix} {message}", flush=True)


def _safe_div(num: Any, den: Any) -> Any:
    den = den.replace(0, np.nan) if isinstance(den, pd.Series) else den
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def _localize_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize("Asia/Kolkata", nonexistent="NaT", ambiguous="NaT")
    return ts.dt.tz_convert("Asia/Kolkata")


def add_option_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = _localize_timestamp(work["timestamp"])
    work["expiry"] = pd.to_datetime(work["expiry"], errors="coerce")
    work["trading_day"] = work["timestamp"].dt.date.astype(str)
    work["dte_days"] = (work["expiry"].dt.normalize() - pd.to_datetime(work["trading_day"])).dt.days.astype(float)
    work["is_weekly"] = work["weekly"].astype(int)
    work["option_type"] = work["option_type"].astype(str).str.upper()
    work["option_type_ce"] = (work["option_type"] == "CE").astype(int)
    work["option_type_pe"] = (work["option_type"] == "PE").astype(int)
    work["option_side_normalized"] = work["option_type"]
    work["weekday"] = pd.to_datetime(work["trading_day"]).dt.weekday.astype(float)
    work["month"] = pd.to_datetime(work["trading_day"]).dt.month.astype(float)

    for col in ["open", "high", "low", "ltp", "volume", "oi", "strike_price"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("instrument_key", sort=False)
    work["range_pct"] = _safe_div(work["high"] - work["low"], work["ltp"])
    work["oc_change_pct"] = _safe_div(work["ltp"] - work["open"], work["open"])
    work["hl_change_pct"] = _safe_div(work["high"] - work["low"], work["low"])
    work["oi_change_pct"] = grouped["oi"].pct_change().replace([np.inf, -np.inf], np.nan)
    work["volume_change_pct"] = grouped["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    work["ret_1"] = grouped["ltp"].pct_change().replace([np.inf, -np.inf], np.nan)
    work["ret_3"] = grouped["ltp"].pct_change(3).replace([np.inf, -np.inf], np.nan)
    work["ret_5"] = grouped["ltp"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    work["ret_10"] = grouped["ltp"].pct_change(10).replace([np.inf, -np.inf], np.nan)

    work["oi_z_5"] = grouped["oi"].transform(
        lambda s: (s - s.rolling(5, min_periods=2).mean()) / s.rolling(5, min_periods=2).std(ddof=0)
    ).replace([np.inf, -np.inf], np.nan)
    work["volume_z_5"] = grouped["volume"].transform(
        lambda s: (s - s.rolling(5, min_periods=2).mean()) / s.rolling(5, min_periods=2).std(ddof=0)
    ).replace([np.inf, -np.inf], np.nan)

    work["last_open"] = work["open"]
    work["last_high"] = work["high"]
    work["last_low"] = work["low"]
    work["last_close"] = work["ltp"]
    work["last_volume"] = work["volume"]
    candle_range = (work["high"] - work["low"]).replace(0, np.nan)
    work["body_pct"] = _safe_div((work["ltp"] - work["open"]).abs(), work["ltp"])
    work["gap_pct"] = grouped["open"].pct_change().replace([np.inf, -np.inf], np.nan)
    work["upper_wick_pct"] = _safe_div(work["high"] - np.maximum(work["open"], work["ltp"]), candle_range)
    work["lower_wick_pct"] = _safe_div(np.minimum(work["open"], work["ltp"]) - work["low"], candle_range)
    work["close_location_pct"] = _safe_div(work["ltp"] - work["low"], candle_range)
    work["close_vs_open_pct"] = work["oc_change_pct"]
    work["momentum_lookback_pct"] = work["ret_10"]

    work["ema_fast"] = grouped["ltp"].transform(lambda s: s.ewm(span=9, adjust=False, min_periods=3).mean())
    work["ema_slow"] = grouped["ltp"].transform(lambda s: s.ewm(span=21, adjust=False, min_periods=5).mean())
    work["ema_diff_pct"] = _safe_div(work["ema_fast"] - work["ema_slow"], work["ltp"])
    work["atr_14"] = grouped.apply(
        lambda g: (g["high"] - g["low"]).rolling(14, min_periods=3).mean()
    ).reset_index(level=0, drop=True)
    work["atr_pct"] = _safe_div(work["atr_14"], work["ltp"])
    work["range_to_atr"] = _safe_div(work["high"] - work["low"], work["atr_14"])
    work["ctx_option_price"] = work["ltp"]

    work["ret_mean"] = grouped["ret_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    work["ret_std"] = grouped["ret_1"].transform(lambda s: s.rolling(5, min_periods=2).std(ddof=0))
    work["ret_min"] = grouped["ret_1"].transform(lambda s: s.rolling(5, min_periods=2).min())
    work["ret_max"] = grouped["ret_1"].transform(lambda s: s.rolling(5, min_periods=2).max())
    work["vol_mean"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    work["vol_std"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=2).std(ddof=0))
    work["vol_min"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=1).min())
    work["vol_max"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=1).max())
    work["volume_ratio"] = _safe_div(work["volume"], work["vol_mean"])
    work["ctx_volume_sma"] = work["vol_mean"]

    work["bullish_engulfing"] = 0.0
    work["bearish_engulfing"] = 0.0
    work["doji"] = (work["body_pct"].fillna(0.0) <= 0.001).astype(float)
    work["hammer"] = ((work["lower_wick_pct"].fillna(0.0) > 0.6) & (work["upper_wick_pct"].fillna(0.0) < 0.25)).astype(float)
    work["shooting_star"] = ((work["upper_wick_pct"].fillna(0.0) > 0.6) & (work["lower_wick_pct"].fillna(0.0) < 0.25)).astype(float)

    minutes = work["timestamp"].dt.hour * 60 + work["timestamp"].dt.minute
    market_minutes = (15 * 60 + 30) - (9 * 60 + 15)
    elapsed = (minutes - (9 * 60 + 15)).clip(lower=0, upper=market_minutes)
    angle = 2.0 * np.pi * elapsed / market_minutes
    work["ctx_time_sin"] = np.sin(angle)
    work["ctx_time_cos"] = np.cos(angle)
    work["is_opening_session"] = (minutes < 10 * 60).astype(float)
    work["is_closing_session"] = (minutes >= 14 * 60 + 30).astype(float)
    work["is_midday_lull"] = ((minutes >= 11 * 60 + 30) & (minutes < 13 * 60 + 30)).astype(float)

    day_group = work.groupby(["trading_day", "instrument_key"], sort=False)
    opening_high = day_group["high"].transform("first")
    opening_low = day_group["low"].transform("first")
    work["dist_from_opening_high_pct"] = _safe_div(work["ltp"] - opening_high, opening_high)
    work["dist_from_opening_low_pct"] = _safe_div(work["ltp"] - opening_low, opening_low)
    work["opening_range_width_pct"] = _safe_div(opening_high - opening_low, work["ltp"])
    work["opening_range_breakout_strength"] = _safe_div(work["ltp"] - opening_high, opening_high)
    rolling_high = grouped["high"].transform(lambda s: s.rolling(20, min_periods=2).max())
    rolling_low = grouped["low"].transform(lambda s: s.rolling(20, min_periods=2).min())
    work["dist_to_rolling_high_20"] = _safe_div(work["ltp"] - rolling_high, rolling_high)
    work["dist_to_rolling_low_20"] = _safe_div(work["ltp"] - rolling_low, rolling_low)
    work["rolling_range_width_20"] = _safe_div(rolling_high - rolling_low, work["ltp"])
    work["rolling_range_position_20"] = _safe_div(work["ltp"] - rolling_low, rolling_high - rolling_low)
    work["realized_vol_30"] = grouped["ret_1"].transform(lambda s: s.rolling(30, min_periods=5).std(ddof=0))
    work["atr_pct_regime_10"] = grouped["atr_pct"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    work["vol_of_vol_14"] = grouped["ret_std"].transform(lambda s: s.rolling(14, min_periods=4).std(ddof=0))
    work["regime_trending"] = (work["ema_diff_pct"].abs() > 0.01).astype(float)
    work["regime_volatile"] = (work["atr_pct"] > work["atr_pct_regime_10"]).astype(float)
    work["regime_mean_reverting"] = ((work["regime_trending"] == 0.0) & (work["regime_volatile"] == 1.0)).astype(float)
    work["regime_quiet"] = ((work["regime_trending"] == 0.0) & (work["regime_volatile"] == 0.0)).astype(float)

    work["is_expiry_day"] = work["dte_days"].eq(0)
    work["is_near_expiry"] = work["dte_days"].le(2)
    work["time_to_expiry_days"] = work["dte_days"]
    work["time_to_expiry_years"] = work["dte_days"] / 365.0
    work["ctx_dte_norm"] = work["dte_days"] / 30.0
    work["greeks_source"] = "not_available_in_archive"
    work["greeks_quality_score"] = 0.0
    work["row_enrichment_ok"] = True
    work["row_enrichment_error"] = np.nan
    for col in BOOL_FALSE_COLUMNS:
        if col in work.columns:
            work[col] = work[col].fillna(False).astype(bool)
    return work


def add_labels(df: pd.DataFrame, horizon_bars: int = 3) -> pd.DataFrame:
    work = df
    grouped = work.groupby("instrument_key", sort=False)
    future_close = grouped["ltp"].shift(-horizon_bars)
    gross = _safe_div(future_close - work["ltp"], work["ltp"])
    embedded_cost = 0.0035
    cost_ru = pd.Series(np.full(len(work), embedded_cost, dtype=float), index=work.index)
    net = gross - embedded_cost
    return_to_cost = net / cost_ru.replace(0.0, np.nan)
    expected_after = net - cost_ru

    work["future_close"] = future_close
    work["gross_forward_return"] = gross
    work["net_forward_return"] = net
    work["profitable_trade_label"] = (net > 0.0).astype(float)
    work["avoid_trade_label"] = (net <= 0.0).astype(float)
    work["strong_profitable_trade_label"] = ((net > cost_ru * 1.5) & (return_to_cost >= 1.5) & (expected_after > 0.0)).astype(float)
    work["high_conviction_trade_label"] = ((net > cost_ru * 2.0) & (return_to_cost >= 2.0)).astype(float)
    work["weak_trade_label"] = ((net > 0.0) & (return_to_cost < 1.25)).astype(float)
    work["no_trade_label"] = (expected_after <= 0.0).astype(float)
    work["cost_survivor_label"] = ((net - 1.25 * cost_ru) > 0.0).astype(float)
    work["paper_candidate_label"] = ((return_to_cost >= 1.5) & ((net - 1.25 * cost_ru) > 0.0)).astype(float)
    work["expected_return_after_cost"] = expected_after
    work["return_to_cost_ratio"] = return_to_cost
    work["cost_return_units_estimated"] = cost_ru
    work["strong_profitable_trade_label_v2"] = ((net > cost_ru * 1.25) & (return_to_cost >= 1.25) & (expected_after > 0.0)).astype(float)
    work["cost_survivor_label_v2"] = ((net - 1.25 * cost_ru) > 0.0).astype(float)
    work["paper_candidate_label_v2"] = ((return_to_cost >= 1.25) & ((net - 1.25 * cost_ru) > 0.0)).astype(float)
    work["high_conviction_trade_label_v2"] = ((net > cost_ru * 1.5) & (return_to_cost >= 1.5)).astype(float)
    work["gross_return"] = gross
    work["net_return_after_cost"] = net - cost_ru
    work["estimated_cost_bps"] = cost_ru * 10000.0
    work["spread_cost_component"] = 0.0010
    work["slippage_cost_component"] = 0.0025
    work["brokerage_or_fee_component"] = 0.0
    work["cost_stress_1_0x_return"] = net - 1.0 * cost_ru
    work["cost_stress_1_5x_return"] = net - 1.5 * cost_ru
    work["cost_stress_2_0x_return"] = net - 2.0 * cost_ru
    work["cost_stress_1_0x_label"] = (work["cost_stress_1_0x_return"] > 0.0).astype(float)
    work["cost_stress_1_5x_label"] = (work["cost_stress_1_5x_return"] > 0.0).astype(float)
    work["cost_stress_2_0x_label"] = (work["cost_stress_2_0x_return"] > 0.0).astype(float)
    work["cost_return_units"] = cost_ru
    work["cost_pct_of_premium"] = cost_ru

    no_future = future_close.isna()
    for col in [c for c in work.columns if "label" in c]:
        work.loc[no_future, col] = np.nan
    return work


def process_year(files: list[Path], year: int, output_dir: Path, horizon_bars: int) -> dict[str, Any]:
    year_start = time.perf_counter()
    out_path = output_dir / f"nifty_option_chain_historical_cost_aware_{year}.csv"
    if out_path.exists() and out_path.stat().st_size > 0:
        log_progress(f"reusing existing {out_path.name}", year=year, stage_start=year_start)
        return {
            "year": year,
            "rows": int(max(count_csv_lines(out_path) - 1, 0)),
            "columns": int(len(CANONICAL_COLUMNS)),
            "path": str(out_path),
            "reused_existing": True,
        }
    log_progress("loading raw archive", year=year, stage_start=year_start)
    base = load_and_parse_year(files, year)
    if base.empty:
        return {"year": year, "rows": 0, "path": None}
    base = base.sort_values(["timestamp", "instrument_key"], kind="stable").reset_index(drop=True)
    log_progress(f"deriving features for {len(base):,} rows", year=year, stage_start=year_start)
    featured = add_option_features(base)
    log_progress("computing labels", year=year, stage_start=year_start)
    labeled = add_labels(featured, horizon_bars=horizon_bars)
    log_progress("sorting labeled rows", year=year, stage_start=year_start)
    labeled = labeled.sort_values(["timestamp", "instrument_key"], kind="stable").reset_index(drop=True)
    log_progress(f"writing {out_path.name} in chunks", year=year, stage_start=year_start)
    write_canonical_csv(labeled, out_path, year=year, stage_start=year_start)
    log_progress(f"completed {len(labeled):,} rows", year=year, stage_start=year_start)
    return {
        "year": year,
        "rows": int(len(labeled)),
        "columns": int(len(CANONICAL_COLUMNS)),
        "path": str(out_path),
        "timestamp_min": str(labeled["timestamp"].min()),
        "timestamp_max": str(labeled["timestamp"].max()),
        "label_rows": int(labeled["profitable_trade_label"].notna().sum()),
        "positive_label_rows": int(labeled["profitable_trade_label"].fillna(0).eq(1).sum()),
    }


def count_csv_lines(path: Path, chunk_size: int = 16 * 1024 * 1024) -> int:
    count = 0
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            count += data.count(b"\n")
    return count


def write_canonical_csv(
    df: pd.DataFrame,
    out_path: Path,
    chunk_size: int = 200_000,
    *,
    year: int | None = None,
    stage_start: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = True
    last_log = time.perf_counter()
    for start in range(0, len(df), chunk_size):
        stop = min(start + chunk_size, len(df))
        chunk = df.iloc[start:stop].copy()
        for col in CANONICAL_COLUMNS:
            if col not in chunk.columns:
                chunk[col] = np.nan
        chunk = chunk[CANONICAL_COLUMNS]
        chunk.to_csv(out_path, index=False, mode="w" if header else "a", header=header)
        header = False
        now = time.perf_counter()
        if now - last_log >= 30 or stop >= len(df):
            pct = 100.0 * stop / max(len(df), 1)
            log_progress(f"write progress {stop:,}/{len(df):,} rows ({pct:.1f}%)", year=year, stage_start=stage_start)
            last_log = now


def concatenate_year_files(year_entries: list[dict[str, Any]], final_path: Path) -> None:
    concat_start = time.perf_counter()
    log_progress(f"writing single file {final_path}", stage_start=concat_start)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(Path(e["path"]).stat().st_size for e in year_entries if e.get("path") and Path(e["path"]).exists())
    copied_bytes = 0
    last_log = time.perf_counter()
    with final_path.open("wb") as out:
        wrote_header = False
        for entry in year_entries:
            path_str = entry.get("path")
            if not path_str:
                continue
            path = Path(path_str)
            with path.open("rb") as src:
                header = src.readline()
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                while True:
                    chunk = src.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    copied_bytes += len(chunk)
                    now = time.perf_counter()
                    if now - last_log >= 30:
                        pct = 100.0 * copied_bytes / max(total_bytes, 1)
                        log_progress(
                            f"concat progress {copied_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GB ({pct:.1f}%)",
                            stage_start=concat_start,
                        )
                        last_log = now
    log_progress("single file complete", stage_start=concat_start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="data/processed/historical_unified_nifty_options_single")
    parser.add_argument("--final-name", default="nifty_option_chain_historical_cost_aware_all_years.csv")
    parser.add_argument("--years", default="all")
    parser.add_argument("--horizon-bars", type=int, default=3)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files = discover_files(source)
    if args.years.lower() == "all":
        years = sorted({int(part) for p in all_files for part in p.parts if part.isdigit() and len(part) == 4})
    else:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    entries = []
    for year in years:
        files_y = [p for p in all_files if f"\\{year}\\" in str(p) or f"/{year}/" in str(p)]
        entries.append(process_year(files_y, year, output_dir, args.horizon_bars))

    final_path = output_dir / args.final_name
    concatenate_year_files(entries, final_path)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output_dir": str(output_dir),
        "final_csv": str(final_path),
        "schema_reference_columns": len(CANONICAL_COLUMNS),
        "columns": CANONICAL_COLUMNS,
        "years": years,
        "total_rows": int(sum(e.get("rows", 0) for e in entries)),
        "files": entries,
        "sort_order": "timestamp ascending, then instrument_key ascending; final file is concatenated in ascending year order",
        "notes": [
            "Raw archive contains option candles only; spot, bid/ask, IV, and Black-Scholes Greek fields are present in schema but blank/defaulted.",
            "Labels are computed per instrument_key with a 3-bar forward horizon and an embedded 0.35% cost penalty.",
            "This output is intended to load cleanly in the existing ML/candidate scripts that expect the cost-aware edge dataset column set.",
        ],
    }
    manifest_path = output_dir / "manifest_single.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log_progress(f"complete: {final_path}")
    log_progress(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
