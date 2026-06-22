from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .feature_builder import read_clean_partitions


@dataclass(slots=True)
class QAReport:
    session_date: date
    snapshot_count: int
    contract_count: int
    raw_completeness_pct: float
    clean_completeness_pct: float
    missing_iv_rows: int
    missing_greeks_rows: int
    duplicate_contract_rows: int
    status: str
    details: Dict[str, Any]


class DailyQAReporter:
    """Validate archive completeness and generate a daily QA report."""

    def build_report(self, session_date: date, clean_paths: List[Path]) -> QAReport:
        frame = read_clean_partitions(clean_paths)
        if frame.empty:
            return QAReport(
                session_date=session_date,
                snapshot_count=0,
                contract_count=0,
                raw_completeness_pct=0.0,
                clean_completeness_pct=0.0,
                missing_iv_rows=0,
                missing_greeks_rows=0,
                duplicate_contract_rows=0,
                status="empty",
                details={},
            )

        frame = frame.copy()
        frame["as_of_ts"] = pd.to_datetime(frame["as_of_ts"], errors="coerce")
        snapshot_count = int(frame["snapshot_id"].nunique())
        contract_count = int(len(frame))
        required_cols = ["symbol_key", "symbol_id", "index_family", "spot", "strike", "expiry", "option_type", "ltp", "oi", "volume", "iv", "delta", "gamma", "vega", "theta"]
        completeness = float(frame[required_cols].notna().mean().mean() * 100.0) if required_cols else 0.0
        missing_iv_rows = int(frame["iv"].isna().sum()) if "iv" in frame else 0
        greek_cols = [col for col in ["delta", "gamma", "vega", "theta"] if col in frame.columns]
        missing_greeks_rows = int(frame[greek_cols].isna().any(axis=1).sum()) if greek_cols else 0
        duplicate_contract_rows = int(frame.duplicated(subset=["snapshot_id", "symbol_key", "strike", "expiry", "option_type"]).sum())
        by_symbol = {}
        if "symbol_key" in frame.columns:
            for symbol_key, group in frame.groupby("symbol_key"):
                by_symbol[str(symbol_key)] = {
                    "snapshot_count": int(group["snapshot_id"].nunique()),
                    "contract_count": int(len(group)),
                    "completeness_pct": float(group[required_cols].notna().mean().mean() * 100.0),
                    "duplicate_rows": int(group.duplicated(subset=["snapshot_id", "strike", "expiry", "option_type"]).sum()),
                }
        status = "pass" if completeness >= 90.0 and duplicate_contract_rows == 0 else "warn"
        details = {
            "as_of_min": frame["as_of_ts"].min().isoformat() if frame["as_of_ts"].notna().any() else None,
            "as_of_max": frame["as_of_ts"].max().isoformat() if frame["as_of_ts"].notna().any() else None,
            "unique_expiries": int(frame["expiry"].nunique(dropna=True)) if "expiry" in frame else 0,
            "by_symbol": by_symbol,
        }
        return QAReport(
            session_date=session_date,
            snapshot_count=snapshot_count,
            contract_count=contract_count,
            raw_completeness_pct=completeness,
            clean_completeness_pct=completeness,
            missing_iv_rows=missing_iv_rows,
            missing_greeks_rows=missing_greeks_rows,
            duplicate_contract_rows=duplicate_contract_rows,
            status=status,
            details=details,
        )
