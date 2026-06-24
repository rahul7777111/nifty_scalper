from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from logger_setup import get_logger

from .feature_builder import build_point_in_time_features, read_clean_partitions, write_feature_partition
from .models import ARCHIVE_SCHEMA_VERSION, ArchiveConfig, ArchiveResult, ArchiveSnapshot, ArchiveState


IST = ZoneInfo("Asia/Kolkata")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OptionsArchiveStorage:
    """Persist raw, clean, and feature layers for the options archive."""

    def __init__(self, config: ArchiveConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or get_logger("options_archive")
        self._ensure_directories()
        self._init_catalog()

    def _ensure_directories(self) -> None:
        for path in (
            self.config.raw_root(),
            self.config.clean_root(),
            self.config.feature_root(),
            self.config.report_root(),
            self.config.state_root(),
            self.config.archive_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.catalog_path())
        conn.row_factory = sqlite3.Row
        return conn

    def _init_catalog(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    symbol_key TEXT NOT NULL,
                    symbol_id INTEGER NOT NULL,
                    index_family TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    spot_symbol TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    as_of_ts TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    symbol_key TEXT NOT NULL,
                    symbol_id INTEGER NOT NULL,
                    index_family TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    spot_symbol TEXT NOT NULL,
                    spot REAL NOT NULL,
                    source TEXT NOT NULL,
                    capture_reason TEXT NOT NULL,
                    spot_source TEXT NOT NULL,
                    is_expiry_day INTEGER NOT NULL,
                    spot_move_points REAL,
                    spot_move_pct REAL,
                    volatility_bucket TEXT,
                    liquidity_bucket TEXT,
                    expiry_cycle_type TEXT,
                    contract_count INTEGER NOT NULL,
                    raw_path TEXT NOT NULL,
                    raw_sha256 TEXT NOT NULL,
                    clean_path TEXT,
                    clean_sha256 TEXT,
                    feature_path TEXT,
                    feature_sha256 TEXT,
                    schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_reports (
                    session_date TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    json_path TEXT NOT NULL,
                    markdown_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS recovery_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_date ON snapshots(session_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_date ON snapshots(symbol_key, session_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_reason ON snapshots(capture_reason)")
            conn.commit()

    def write_run_start(self, run_id: str, snapshot: ArchiveSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO collection_runs (
                    run_id, started_at, status, symbol_key, symbol_id, index_family, underlying, spot_symbol, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    snapshot.as_of_ts.isoformat(),
                    "running",
                    snapshot.symbol_key,
                    snapshot.symbol_id,
                    snapshot.index_family,
                    snapshot.underlying,
                    snapshot.spot_symbol,
                    json.dumps(self.config.to_dict(), ensure_ascii=False, default=_json_default),
                ),
            )
            conn.commit()

    def write_run_finish(self, run_id: str, status: str, error_message: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE collection_runs
                SET finished_at = ?, status = ?, error_message = ?
                WHERE run_id = ?
                """,
                (datetime.now(tz=IST).isoformat(), status, error_message, run_id),
            )
            conn.commit()

    def load_state(self) -> ArchiveState:
        state = ArchiveState()
        path = self.config.state_root() / "collector_state.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("last_snapshot_ts"):
                    state.last_snapshot_ts = datetime.fromisoformat(payload["last_snapshot_ts"])
                if payload.get("last_snapshot_spot") is not None:
                    state.last_snapshot_spot = float(payload["last_snapshot_spot"])
                if payload.get("last_base_snapshot_ts"):
                    state.last_base_snapshot_ts = datetime.fromisoformat(payload["last_base_snapshot_ts"])
                if payload.get("last_base_snapshot_spot") is not None:
                    state.last_base_snapshot_spot = float(payload["last_base_snapshot_spot"])
                if payload.get("last_daily_qa_date"):
                    state.last_daily_qa_date = date.fromisoformat(payload["last_daily_qa_date"])
                if payload.get("last_snapshot_id"):
                    state.last_snapshot_id = str(payload["last_snapshot_id"])
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to load archive state: %s", exc)
        return state

    def save_state(self, state: ArchiveState) -> None:
        payload = {
            "last_snapshot_ts": state.last_snapshot_ts.isoformat() if state.last_snapshot_ts else None,
            "last_snapshot_spot": state.last_snapshot_spot,
            "last_base_snapshot_ts": state.last_base_snapshot_ts.isoformat() if state.last_base_snapshot_ts else None,
            "last_base_snapshot_spot": state.last_base_snapshot_spot,
            "last_daily_qa_date": state.last_daily_qa_date.isoformat() if state.last_daily_qa_date else None,
            "last_snapshot_id": state.last_snapshot_id,
        }
        path = self.config.state_root() / "collector_state.json"
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)

    def _snapshot_paths(self, snapshot: ArchiveSnapshot) -> Tuple[Path, Path, Path]:
        date_part = snapshot.session_date.isoformat()
        ts_part = snapshot.as_of_ts.strftime("%Y%m%dT%H%M%S%z")
        base_dir = self.config.raw_root() / f"index_family={snapshot.index_family}" / f"symbol_key={snapshot.symbol_key}" / f"session_date={date_part}" / "frequency=5m"
        raw_path = base_dir / f"snapshot_{ts_part}_{snapshot.snapshot_id}.jsonl.gz"
        clean_path = self.config.clean_root() / f"index_family={snapshot.index_family}" / f"symbol_key={snapshot.symbol_key}" / f"session_date={date_part}" / "frequency=5m" / f"snapshot_{ts_part}_{snapshot.snapshot_id}.parquet"
        feature_path = self.config.feature_root() / f"index_family={snapshot.index_family}" / f"symbol_key={snapshot.symbol_key}" / f"session_date={date_part}" / f"snapshot_{ts_part}_{snapshot.snapshot_id}.parquet"
        return raw_path, clean_path, feature_path

    def _write_jsonl_gz(self, path: Path, lines: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
                for record in lines:
                    handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
                    handle.write("\n")
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _write_parquet(self, frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp.parquet")
        frame.to_parquet(tmp_path, index=False, compression="zstd")
        tmp_path.replace(path)
        return path

    def persist_snapshot(self, snapshot: ArchiveSnapshot) -> ArchiveResult:
        raw_path, clean_path, feature_path = self._snapshot_paths(snapshot)

        raw_lines: List[Dict[str, Any]] = [
            {
                "record_type": "snapshot",
                "snapshot_id": snapshot.snapshot_id,
                "schema_version": snapshot.schema_version,
                "as_of_ts": snapshot.as_of_ts.isoformat(),
                "session_date": snapshot.session_date.isoformat(),
                "symbol_key": snapshot.symbol_key,
                "symbol_id": snapshot.symbol_id,
                "index_family": snapshot.index_family,
                "underlying": snapshot.underlying,
                "spot_symbol": snapshot.spot_symbol,
                "spot": snapshot.spot,
                "source": snapshot.source,
                "capture_reason": snapshot.capture_reason,
                "spot_source": snapshot.spot_source,
                "is_expiry_day": snapshot.is_expiry_day,
                "spot_move_points": snapshot.spot_move_points,
                "spot_move_pct": snapshot.spot_move_pct,
                "contract_count": len(snapshot.chain_rows),
                "raw_payload": snapshot.raw_payload,
            }
        ]
        for row in snapshot.chain_rows:
            raw_row = {"record_type": "contract", "snapshot_id": snapshot.snapshot_id, **row}
            raw_lines.append(raw_row)

        self._write_jsonl_gz(raw_path, raw_lines)
        raw_sha256 = _sha256_file(raw_path)

        frame = pd.DataFrame(snapshot.chain_rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=[
                    "snapshot_id",
                    "as_of_ts",
                    "session_date",
                    "symbol_key",
                    "symbol_id",
                    "index_family",
                    "underlying",
                    "spot_symbol",
                    "spot",
                    "source",
                    "capture_reason",
                    "spot_source",
                    "is_expiry_day",
                    "spot_move_points",
                    "spot_move_pct",
                    "volatility_bucket",
                    "liquidity_bucket",
                    "expiry_cycle_type",
                    "strike",
                    "expiry",
                    "option_type",
                    "symbol",
                    "ltp",
                    "bid",
                    "ask",
                    "volume",
                    "oi",
                    "delta",
                    "gamma",
                    "vega",
                    "theta",
                    "iv",
                    "spot",
                    "schema_version",
                ]
            )
        else:
            frame = frame.copy()
        for column, value in (
            ("snapshot_id", snapshot.snapshot_id),
            ("as_of_ts", snapshot.as_of_ts.isoformat()),
            ("session_date", snapshot.session_date.isoformat()),
            ("symbol_key", snapshot.symbol_key),
            ("symbol_id", snapshot.symbol_id),
            ("index_family", snapshot.index_family),
            ("underlying", snapshot.underlying),
            ("spot_symbol", snapshot.spot_symbol),
            ("spot", snapshot.spot),
            ("source", snapshot.source),
            ("capture_reason", snapshot.capture_reason),
            ("spot_source", snapshot.spot_source),
            ("is_expiry_day", int(snapshot.is_expiry_day)),
            ("spot_move_points", snapshot.spot_move_points),
            ("spot_move_pct", snapshot.spot_move_pct),
            ("volatility_bucket", snapshot.volatility_bucket),
            ("liquidity_bucket", snapshot.liquidity_bucket),
            ("expiry_cycle_type", snapshot.expiry_cycle_type),
            ("schema_version", snapshot.schema_version),
            ("raw_path", str(raw_path)),
            ("raw_sha256", raw_sha256),
        ):
            frame[column] = value

        clean_path = self._write_parquet(frame, clean_path)
        clean_sha256 = _sha256_file(clean_path)

        feature_written_path: Optional[Path] = None
        feature_sha256: Optional[str] = None
        if self.config.write_feature_layer:
            try:
                feature_frame = build_point_in_time_features(frame)
                feature_written_path = self._write_parquet(feature_frame, feature_path)
                feature_sha256 = _sha256_file(feature_written_path)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Feature layer build failed for %s: %s", snapshot.snapshot_id, exc)

        self._register_snapshot(snapshot, raw_path, raw_sha256, clean_path, clean_sha256, feature_written_path, feature_sha256)
        return ArchiveResult(
            snapshot_id=snapshot.snapshot_id,
            status="ok",
            capture_reason=snapshot.capture_reason,
            raw_path=raw_path,
            clean_path=clean_path,
            feature_path=feature_written_path,
            raw_sha256=raw_sha256,
            clean_sha256=clean_sha256,
            contract_count=len(snapshot.chain_rows),
        )

    def _register_snapshot(
        self,
        snapshot: ArchiveSnapshot,
        raw_path: Path,
        raw_sha256: str,
        clean_path: Path,
        clean_sha256: str,
        feature_path: Optional[Path],
        feature_sha256: Optional[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO snapshots (
                    snapshot_id, as_of_ts, session_date, symbol_key, symbol_id, index_family, underlying, spot_symbol, spot, source,
                    capture_reason, spot_source, is_expiry_day, spot_move_points, spot_move_pct,
                    volatility_bucket, liquidity_bucket, expiry_cycle_type,
                    contract_count, raw_path, raw_sha256, clean_path, clean_sha256, feature_path,
                    feature_sha256, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.as_of_ts.isoformat(),
                    snapshot.session_date.isoformat(),
                    snapshot.symbol_key,
                    snapshot.symbol_id,
                    snapshot.index_family,
                    snapshot.underlying,
                    snapshot.spot_symbol,
                    snapshot.spot,
                    snapshot.source,
                    snapshot.capture_reason,
                    snapshot.spot_source,
                    int(snapshot.is_expiry_day),
                    snapshot.spot_move_points,
                    snapshot.spot_move_pct,
                    snapshot.volatility_bucket,
                    snapshot.liquidity_bucket,
                    snapshot.expiry_cycle_type,
                    len(snapshot.chain_rows),
                    str(raw_path),
                    raw_sha256,
                    str(clean_path),
                    clean_sha256,
                    str(feature_path) if feature_path else None,
                    feature_sha256,
                    snapshot.schema_version,
                ),
            )
            conn.commit()

    def list_clean_paths(self, session_date: Optional[date] = None, symbol_key: Optional[str] = None) -> List[Path]:
        with self._connect() as conn:
            if session_date is None and symbol_key is None:
                rows = conn.execute("SELECT clean_path FROM snapshots WHERE clean_path IS NOT NULL ORDER BY as_of_ts").fetchall()
            else:
                clauses = ["clean_path IS NOT NULL"]
                params: List[Any] = []
                if session_date is not None:
                    clauses.append("session_date = ?")
                    params.append(session_date.isoformat())
                if symbol_key is not None:
                    clauses.append("symbol_key = ?")
                    params.append(str(symbol_key).upper())
                query = "SELECT clean_path FROM snapshots WHERE " + " AND ".join(clauses) + " ORDER BY as_of_ts"
                rows = conn.execute(query, params).fetchall()
        return [Path(row[0]) for row in rows if row[0]]

    def rebuild_feature_layer(self, session_date: Optional[date] = None, symbol_key: Optional[str] = None) -> List[Path]:
        clean_paths = self.list_clean_paths(session_date=session_date, symbol_key=symbol_key)
        if not clean_paths:
            return []
        frame = read_clean_partitions(clean_paths)
        if frame.empty:
            return []
        feature_frame = build_point_in_time_features(frame)
        written: List[Path] = []
        for snapshot_id, group in feature_frame.groupby("snapshot_id", sort=True):
            snapshot_date = pd.to_datetime(group["session_date"].iloc[0]).date()
            as_of = pd.to_datetime(group["as_of_ts"].iloc[0])
            group_symbol = str(group["symbol_key"].iloc[0]) if "symbol_key" in group.columns else str(group["underlying"].iloc[0])
            group_family = str(group["index_family"].iloc[0]) if "index_family" in group.columns else group_symbol
            output = self.config.feature_root() / f"index_family={group_family}" / f"symbol_key={group_symbol}" / f"session_date={snapshot_date.isoformat()}" / f"snapshot_{as_of.strftime('%Y%m%dT%H%M%S%z')}_{snapshot_id}.parquet"
            written.append(write_feature_partition(group, output))
            feature_sha256 = _sha256_file(output)
            with self._connect() as conn:
                conn.execute(
                    "UPDATE snapshots SET feature_path = ?, feature_sha256 = ? WHERE snapshot_id = ?",
                    (str(output), feature_sha256, snapshot_id),
                )
                conn.commit()
        return written

    def write_daily_qa_report(self, session_date: date, summary: Dict[str, Any]) -> Tuple[Path, Path]:
        report_dir = self.config.report_root() / session_date.isoformat()
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / "qa_report.json"
        md_path = report_dir / "qa_report.md"
        json_payload = json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
        json_path.write_text(json_payload, encoding="utf-8")
        md_lines = [
            f"# Daily QA Report - {session_date.isoformat()}",
            "",
            f"- Snapshots: {summary.get('snapshot_count', 0)}",
            f"- Contracts: {summary.get('contract_count', 0)}",
            f"- Raw completeness: {summary.get('raw_completeness_pct', 0.0):.2f}%",
            f"- Clean completeness: {summary.get('clean_completeness_pct', 0.0):.2f}%",
            f"- Missing IV rows: {summary.get('missing_iv_rows', 0)}",
            f"- Missing greeks rows: {summary.get('missing_greeks_rows', 0)}",
            f"- Duplicate contract rows: {summary.get('duplicate_contract_rows', 0)}",
            f"- QA status: {summary.get('status', 'unknown')}",
        ]
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO qa_reports (session_date, run_id, json_path, markdown_path, summary_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_date.isoformat(), str(summary.get("run_id") or ""), str(json_path), str(md_path), json_payload),
            )
            conn.commit()
        return json_path, md_path

    def recover_pending_snapshots(self) -> List[ArchiveResult]:
        recovered: List[ArchiveResult] = []
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM snapshots WHERE clean_path IS NULL OR feature_path IS NULL ORDER BY as_of_ts").fetchall()
        for row in rows:
            raw_path = Path(row["raw_path"])
            if not raw_path.exists():
                continue
            try:
                recovered.append(self._recover_from_raw(raw_path))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to recover %s: %s", raw_path, exc)
        return recovered

    def _recover_from_raw(self, raw_path: Path) -> ArchiveResult:
        lines = []
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                lines.append(json.loads(line))
        header = next((line for line in lines if line.get("record_type") == "snapshot"), None)
        if header is None:
            raise ValueError(f"Raw snapshot missing header: {raw_path}")
        contract_rows = [line for line in lines if line.get("record_type") == "contract"]
        snapshot = ArchiveSnapshot(
            snapshot_id=str(header["snapshot_id"]),
            as_of_ts=datetime.fromisoformat(header["as_of_ts"]),
            session_date=date.fromisoformat(header["session_date"]),
            underlying=str(header["underlying"]),
            spot_symbol=str(header["spot_symbol"]),
            spot=float(header["spot"]),
            source=str(header.get("source") or "mstock"),
            capture_reason=str(header.get("capture_reason") or "recovery"),
            spot_source=str(header.get("spot_source") or "mstock_ltp"),
            is_expiry_day=bool(header.get("is_expiry_day")),
            spot_move_points=header.get("spot_move_points"),
            spot_move_pct=header.get("spot_move_pct"),
            raw_payload={"recovered": True},
            chain_rows=[{k: v for k, v in row.items() if k != "record_type"} for row in contract_rows],
        )
        return self.persist_snapshot(snapshot)
