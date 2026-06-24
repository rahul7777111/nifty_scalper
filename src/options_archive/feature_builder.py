from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

from .models import ARCHIVE_SCHEMA_VERSION


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _series_percentile_rank(values: pd.Series, current: float) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float((clean <= float(current)).sum() / len(clean))


def _liquidity_bucket(total_volume: float, median_spread_bps: float) -> str:
    if total_volume >= 100000 and median_spread_bps <= 8:
        return "high"
    if total_volume >= 25000 and median_spread_bps <= 20:
        return "medium"
    return "low"


def _volatility_bucket(iv_percentile: float) -> str:
    if pd.isna(iv_percentile):
        return "unknown"
    if iv_percentile < 0.33:
        return "low"
    if iv_percentile < 0.67:
        return "medium"
    return "high"


def _expiry_cycle_type(expiry: Optional[date], session_date: Optional[date]) -> str:
    if expiry is None or session_date is None:
        return "unknown"
    if expiry.weekday() != 3:
        return "custom"
    next_week = expiry + pd.Timedelta(days=7)
    if next_week.month != expiry.month:
        return "monthly"
    return "weekly"


def build_snapshot_summary(clean_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate contract rows into snapshot-level point-in-time summaries."""

    if clean_frame.empty:
        return clean_frame.copy()

    frame = clean_frame.copy()
    frame["as_of_ts"] = _ensure_datetime(frame["as_of_ts"])
    frame["expiry"] = pd.to_datetime(frame["expiry"], errors="coerce").dt.date
    frame["spot"] = pd.to_numeric(frame.get("spot"), errors="coerce")
    frame["oi"] = pd.to_numeric(frame.get("oi"), errors="coerce").fillna(0.0)
    frame["iv"] = pd.to_numeric(frame.get("iv"), errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    frame["bid"] = pd.to_numeric(frame.get("bid"), errors="coerce")
    frame["ask"] = pd.to_numeric(frame.get("ask"), errors="coerce")
    frame["option_type"] = frame["option_type"].astype(str).str.upper()

    def _nearest_atm_iv(group: pd.DataFrame) -> float:
        spot_series = group["spot"].dropna()
        if spot_series.empty:
            return float("nan")
        spot = float(spot_series.iloc[0])
        working = group.copy()
        working["strike_distance"] = (working["strike"] - float(spot)).abs()
        atm_row = working.sort_values(["strike_distance", "expiry", "option_type"]).head(1)
        if atm_row.empty:
            return float("nan")
        return float(pd.to_numeric(atm_row["iv"], errors="coerce").iloc[0])

    rows: List[Dict[str, Any]] = []
    for snapshot_id, group in frame.groupby("snapshot_id", sort=True):
        spot_series = pd.to_numeric(group["spot"], errors="coerce").dropna()
        spot = float(spot_series.iloc[0]) if not spot_series.empty else float("nan")
        calls = group[group["option_type"] == "CE"]
        puts = group[group["option_type"] == "PE"]
        total_call_oi = float(pd.to_numeric(calls["oi"], errors="coerce").fillna(0.0).sum())
        total_put_oi = float(pd.to_numeric(puts["oi"], errors="coerce").fillna(0.0).sum())
        total_volume = float(pd.to_numeric(group["volume"], errors="coerce").fillna(0.0).sum())
        spread_bps = []
        for _, row in group.iterrows():
            bid = row.get("bid")
            ask = row.get("ask")
            spot = row.get("spot")
            if pd.notna(bid) and pd.notna(ask) and pd.notna(spot) and float(spot) > 0:
                spread_bps.append(((float(ask) - float(bid)) / float(spot)) * 10000.0)
        median_spread_bps = float(pd.Series(spread_bps).median()) if spread_bps else float("nan")
        atm_iv = _nearest_atm_iv(group)
        expiry_min = group["expiry"].dropna().min() if not group["expiry"].dropna().empty else None
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "as_of_ts": group["as_of_ts"].iloc[0],
                "session_date": group["session_date"].iloc[0],
                "symbol_key": group["symbol_key"].iloc[0] if "symbol_key" in group.columns else group["underlying"].iloc[0],
                "symbol_id": int(pd.to_numeric(group["symbol_id"], errors="coerce").iloc[0]) if "symbol_id" in group.columns and not pd.to_numeric(group["symbol_id"], errors="coerce").dropna().empty else 0,
                "index_family": group["index_family"].iloc[0] if "index_family" in group.columns else group["underlying"].iloc[0],
                "underlying": group["underlying"].iloc[0],
                "spot": spot,
                "contract_count": int(len(group)),
                "call_count": int(len(calls)),
                "put_count": int(len(puts)),
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "total_volume": total_volume,
                "median_spread_bps": median_spread_bps,
                "pcr_oi": float("nan") if total_call_oi <= 0 else float(total_put_oi / total_call_oi),
                "atm_iv": atm_iv,
                "min_expiry": expiry_min,
                "schema_version": ARCHIVE_SCHEMA_VERSION,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(["as_of_ts", "snapshot_id"]).reset_index(drop=True)
    summary["iv_percentile"] = float("nan")
    return summary


def build_point_in_time_features(clean_frame: pd.DataFrame, *, iv_percentile_window: int = 252) -> pd.DataFrame:
    """Build the feature layer by joining snapshot summaries back to contract rows.

    Rolling IV percentile is calculated strictly from prior snapshots so the
    current snapshot never sees its own future or same-bar history.
    """

    if clean_frame.empty:
        return clean_frame.copy()

    frame = clean_frame.copy()
    frame["as_of_ts"] = _ensure_datetime(frame["as_of_ts"])
    frame["expiry"] = pd.to_datetime(frame["expiry"], errors="coerce").dt.date
    frame["spot"] = pd.to_numeric(frame.get("spot"), errors="coerce")
    frame["strike"] = pd.to_numeric(frame.get("strike"), errors="coerce")
    frame["bid"] = pd.to_numeric(frame.get("bid"), errors="coerce")
    frame["ask"] = pd.to_numeric(frame.get("ask"), errors="coerce")
    frame["ltp"] = pd.to_numeric(frame.get("ltp"), errors="coerce")
    frame["oi"] = pd.to_numeric(frame.get("oi"), errors="coerce").fillna(0.0)
    frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    frame["iv"] = pd.to_numeric(frame.get("iv"), errors="coerce")
    frame["delta"] = pd.to_numeric(frame.get("delta"), errors="coerce")
    frame["gamma"] = pd.to_numeric(frame.get("gamma"), errors="coerce")
    frame["vega"] = pd.to_numeric(frame.get("vega"), errors="coerce")
    frame["theta"] = pd.to_numeric(frame.get("theta"), errors="coerce")
    frame["option_type"] = frame["option_type"].astype(str).str.upper()
    frame["mid"] = ((frame["bid"].fillna(frame["ltp"]) + frame["ask"].fillna(frame["ltp"])) / 2.0).fillna(frame["ltp"])
    frame["spread"] = (frame["ask"] - frame["bid"]).where(frame["ask"].notna() & frame["bid"].notna())
    frame["dte"] = [max(0, (exp - ts.date()).days) if isinstance(exp, date) and pd.notna(ts) else None for ts, exp in zip(frame["as_of_ts"], frame["expiry"])]
    frame["is_atm"] = False
    frame["moneyness"] = frame["strike"] / frame["spot"]

    summary = build_snapshot_summary(frame)
    if summary.empty:
        return frame

    summary = summary.sort_values(["as_of_ts", "snapshot_id"]).reset_index(drop=True)
    rolling_history: List[float] = []
    percentiles: List[float] = []
    for current_iv in summary["atm_iv"].tolist():
        if pd.isna(current_iv):
            percentiles.append(float("nan"))
        else:
            window = pd.Series(rolling_history[-int(iv_percentile_window) :]) if rolling_history else pd.Series(dtype=float)
            percentiles.append(_series_percentile_rank(window, float(current_iv)))
            rolling_history.append(float(current_iv))
    summary["iv_percentile"] = percentiles
    summary["volatility_bucket"] = summary["iv_percentile"].apply(_volatility_bucket)
    summary["liquidity_bucket"] = [
        _liquidity_bucket(float(v or 0.0), float(s or 0.0))
        for v, s in zip(summary.get("total_volume", pd.Series(dtype=float)), summary.get("median_spread_bps", pd.Series(dtype=float)))
    ]
    summary["expiry_cycle_type"] = [
        _expiry_cycle_type(pd.to_datetime(exp).date() if pd.notna(exp) else None, pd.to_datetime(ts).date() if pd.notna(ts) else None)
        for exp, ts in zip(summary.get("min_expiry", pd.Series(dtype=object)), summary.get("session_date", pd.Series(dtype=object)))
    ]

    feature_frame = frame.merge(
        summary[
            [
                "snapshot_id",
                "symbol_key",
                "symbol_id",
                "index_family",
                "contract_count",
                "call_count",
                "put_count",
                "total_call_oi",
                "total_put_oi",
                "total_volume",
                "median_spread_bps",
                "pcr_oi",
                "atm_iv",
                "iv_percentile",
                "volatility_bucket",
                "liquidity_bucket",
                "expiry_cycle_type",
                "min_expiry",
            ]
        ],
        on="snapshot_id",
        how="left",
    )
    for base_name in ("symbol_key", "symbol_id", "index_family"):
        left_name = f"{base_name}_x"
        right_name = f"{base_name}_y"
        if left_name in feature_frame.columns:
            feature_frame[base_name] = feature_frame[left_name]
            feature_frame = feature_frame.drop(columns=[col for col in (left_name, right_name) if col in feature_frame.columns])
        elif right_name in feature_frame.columns:
            feature_frame[base_name] = feature_frame[right_name]
            feature_frame = feature_frame.drop(columns=[col for col in (right_name,) if col in feature_frame.columns])
    feature_frame["vol_risk_premium"] = feature_frame["iv"] - feature_frame["atm_iv"]
    feature_frame["normalized_spot_return"] = feature_frame.groupby("symbol_key")["spot"].pct_change().fillna(0.0) if "symbol_key" in feature_frame.columns else 0.0
    feature_frame["normalized_moneyness"] = feature_frame["strike"] / feature_frame["spot"]
    feature_frame["normalized_volume"] = feature_frame["volume"].apply(lambda x: float(np.log1p(float(x))) if pd.notna(x) else 0.0)
    feature_frame["normalized_oi"] = feature_frame["oi"].apply(lambda x: float(np.log1p(float(x))) if pd.notna(x) else 0.0)
    feature_frame["normalized_spread_bps"] = feature_frame["spread"].fillna(0.0).astype(float) / feature_frame["spot"].replace(0, np.nan).astype(float)
    feature_frame["normalized_iv"] = feature_frame["iv"].fillna(0.0)
    feature_frame["normalized_pcr_oi"] = feature_frame["pcr_oi"].fillna(0.0)
    feature_frame["feature_schema_version"] = ARCHIVE_SCHEMA_VERSION
    return feature_frame


def read_clean_partitions(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not Path(path).exists():
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_feature_partition(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")
    return output_path
