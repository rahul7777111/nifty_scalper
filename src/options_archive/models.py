from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ARCHIVE_SCHEMA_VERSION = "2026-05-31"
IST_TZ = "Asia/Kolkata"


@dataclass(slots=True, frozen=True)
class SymbolSpec:
    """Canonical index symbol metadata used by collection and training."""

    symbol_key: str
    symbol_id: int
    index_family: str
    underlying: str
    spot_symbol: str
    chain_underlying: str
    lot_size: int
    expiry_cycle_type: str = "weekly"


@dataclass(slots=True)
class ArchiveConfig:
    """Collector and archive configuration."""

    symbol_key: str = "NIFTY"
    symbol_id: int = 1
    index_family: str = "NIFTY"
    archive_root: Path = Path("archive") / "options"
    underlying: str = "NIFTY"
    spot_symbol: str = "NSE:NIFTY"
    chain_underlying: str = "NIFTY"
    lot_size: int = 65
    expiry_cycle_type: str = "weekly"
    snapshot_interval_minutes: int = 5
    scheduler_poll_seconds: int = 60
    open_snapshot_window_minutes: int = 2
    expiry_boost_start_hhmm: str = "14:00"
    expiry_boost_frequency_minutes: int = 1
    large_move_pct: float = 0.35
    large_move_points: float = 50.0
    risk_free_rate: float = 0.06
    max_chain_rows: int = 0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    write_feature_layer: bool = True

    def raw_root(self) -> Path:
        return self.archive_root / "raw"

    def clean_root(self) -> Path:
        return self.archive_root / "clean"

    def feature_root(self) -> Path:
        return self.archive_root / "feature"

    def report_root(self) -> Path:
        return self.archive_root / "reports"

    def state_root(self) -> Path:
        return self.archive_root / "state"

    def catalog_path(self) -> Path:
        return self.archive_root / "catalog.sqlite3"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["archive_root"] = str(self.archive_root)
        return payload


@dataclass(slots=True)
class ArchiveSnapshot:
    """Single point-in-time snapshot bundle."""

    snapshot_id: str
    as_of_ts: datetime
    session_date: date
    symbol_key: str
    symbol_id: int
    index_family: str
    underlying: str
    spot_symbol: str
    spot: float
    source: str
    capture_reason: str
    spot_source: str
    is_expiry_day: bool
    spot_move_points: Optional[float]
    spot_move_pct: Optional[float]
    volatility_bucket: Optional[str] = None
    liquidity_bucket: Optional[str] = None
    expiry_cycle_type: str = "weekly"
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    chain_rows: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: str = ARCHIVE_SCHEMA_VERSION


@dataclass(slots=True)
class ArchiveState:
    """Minimal durable state for recovery and trigger evaluation."""

    last_snapshot_ts: Optional[datetime] = None
    last_snapshot_spot: Optional[float] = None
    last_base_snapshot_ts: Optional[datetime] = None
    last_base_snapshot_spot: Optional[float] = None
    last_daily_qa_date: Optional[date] = None
    last_snapshot_id: Optional[str] = None


@dataclass(slots=True)
class ArchiveResult:
    """Outcome returned by a collection attempt."""

    snapshot_id: str
    status: str
    capture_reason: str
    raw_path: Optional[Path] = None
    clean_path: Optional[Path] = None
    feature_path: Optional[Path] = None
    raw_sha256: Optional[str] = None
    clean_sha256: Optional[str] = None
    contract_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
