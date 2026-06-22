"""
SQLite Trade Database Layer

Provides:
- Production-grade SQLite schema for trade state persistence
- CRUD operations for positions, orders, fills, strategy state, risk state
- Migration support
- Thread-safe access via connection pooling (single writer, multiple readers)
- Audit logging

Replaces:
- File-based persistence for trade/position state

Tables:
- positions: Open and historical positions
- orders: Order lifecycle tracking
- fills: Individual fill records
- strategy_state: Strategy parameters and runtime state
- risk_state: Risk metrics and limits
- audit_log: Immutable audit trail
- shadow_predictions: Shadow mode prediction records
"""

import os
import json
import time
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    position_id       TEXT PRIMARY KEY,
    strategy_name     TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    underlying        TEXT NOT NULL,
    option_type       TEXT NOT NULL CHECK(option_type IN ('CE', 'PE')),
    strike            REAL NOT NULL,
    expiry            TEXT NOT NULL,
    direction         TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
    quantity          INTEGER NOT NULL,
    entry_price       REAL NOT NULL,
    entry_time        TEXT NOT NULL,
    exit_price        REAL,
    exit_time         TEXT,
    mtm               REAL NOT NULL DEFAULT 0.0,
    status            TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'CLOSED', 'CANCELLED', 'PENDING')),
    parent_trade_id   TEXT,
    meta_json         TEXT DEFAULT '{}',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_underlying ON positions(underlying);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_name);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_time);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id          TEXT PRIMARY KEY,
    position_id       TEXT NOT NULL REFERENCES positions(position_id),
    broker_order_id   TEXT,
    symbol            TEXT NOT NULL,
    order_type        TEXT NOT NULL CHECK(order_type IN ('MARKET', 'LIMIT', 'SL', 'SL-M')),
    side              TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    quantity          INTEGER NOT NULL,
    price             REAL,
    trigger_price     REAL,
    status            TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'OPEN', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED')),
    filled_qty        INTEGER NOT NULL DEFAULT 0,
    avg_fill_price    REAL,
    reject_reason     TEXT,
    meta_json         TEXT DEFAULT '{}',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_position_id ON orders(position_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_broker_id ON orders(broker_order_id);

-- Fills table
CREATE TABLE IF NOT EXISTS fills (
    fill_id           TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL REFERENCES orders(order_id),
    position_id       TEXT NOT NULL REFERENCES positions(position_id),
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    quantity          INTEGER NOT NULL,
    price             REAL NOT NULL,
    fill_time         TEXT NOT NULL,
    exchange_order_id TEXT,
    meta_json         TEXT DEFAULT '{}',
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_position_id ON fills(position_id);

-- Strategy state table (key-value store for runtime state)
CREATE TABLE IF NOT EXISTS strategy_state (
    key               TEXT PRIMARY KEY,
    value_json        TEXT NOT NULL,
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Risk state table
CREATE TABLE IF NOT EXISTS risk_state (
    key               TEXT PRIMARY KEY,
    value_json        TEXT NOT NULL,
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Shadow predictions table
CREATE TABLE IF NOT EXISTS shadow_predictions (
    prediction_id     TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    underlying        TEXT NOT NULL,
    signal            TEXT NOT NULL CHECK(signal IN ('BUY', 'SELL', 'HOLD', 'NO_TRADE')),
    confidence        REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    predicted_label   INTEGER,
    actual_outcome    INTEGER,
    model_version     TEXT NOT NULL,
    regime            TEXT,
    feature_values    TEXT DEFAULT '{}',
    meta_json         TEXT DEFAULT '{}',
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_shadow_pred_timestamp ON shadow_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_shadow_pred_signal ON shadow_predictions(signal);
CREATE INDEX IF NOT EXISTS idx_shadow_pred_underlying ON shadow_predictions(underlying);

-- Audit log (immutable)
CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL DEFAULT (datetime('now')),
    event_type        TEXT NOT NULL,
    severity          TEXT NOT NULL CHECK(severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    component         TEXT NOT NULL,
    message           TEXT NOT NULL,
    data_json         TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_log(severity);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);

-- Performance metrics (for backtest/paper-trade results)
CREATE TABLE IF NOT EXISTS performance_metrics (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    metric_name       TEXT NOT NULL,
    metric_value      REAL NOT NULL,
    params_json       TEXT DEFAULT '{}',
    timestamp         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_perf_run_id ON performance_metrics(run_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version           INTEGER PRIMARY KEY,
    applied_at        TEXT NOT NULL DEFAULT (datetime('now')),
    description       TEXT
);

-- Insert initial schema version if not present
INSERT OR IGNORE INTO schema_version (version, description) VALUES (1, 'Initial schema: positions, orders, fills, strategy_state, risk_state, shadow_predictions, audit_log');
"""


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------

MIGRATIONS: Dict[int, str] = {
    2: """
        ALTER TABLE positions ADD COLUMN IF NOT EXISTS unrealized_pnl REAL DEFAULT 0.0;
        ALTER TABLE positions ADD COLUMN IF NOT EXISTS realized_pnl REAL DEFAULT 0.0;
    """,
    3: """
        CREATE TABLE IF NOT EXISTS daily_summary (
            date              TEXT PRIMARY KEY,
            total_trades      INTEGER NOT NULL DEFAULT 0,
            winning_trades    INTEGER NOT NULL DEFAULT 0,
            losing_trades     INTEGER NOT NULL DEFAULT 0,
            gross_pnl         REAL NOT NULL DEFAULT 0.0,
            net_pnl           REAL NOT NULL DEFAULT 0.0,
            max_drawdown_pct  REAL NOT NULL DEFAULT 0.0,
            turnover          REAL NOT NULL DEFAULT 0.0,
            meta_json         TEXT DEFAULT '{}',
            created_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """,
}


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class TradeDatabase:
    """
    SQLite-backed trade state persistence.

    Thread-safe: uses a single writer connection with a lock,
    and supports multiple readers via separate connections.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = str(db_path or self._default_db_path())
        self._write_lock = threading.Lock()

        # Connection caches per thread
        self._write_conn: Optional[sqlite3.Connection] = None
        self._read_conns: Dict[int, sqlite3.Connection] = {}

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()
        logger.info("TradeDatabase initialized at %s", self.db_path)

    @staticmethod
    def _default_db_path() -> Path:
        """Default database path: repo_root/data/trading_state.db"""
        repo_root = Path(__file__).resolve().parent.parent.parent
        return repo_root / "data" / "trading_state.db"

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    def _get_write_conn(self) -> sqlite3.Connection:
        """Get or create the write connection."""
        if self._write_conn is None:
            self._write_conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False,
            )
            self._write_conn.execute("PRAGMA journal_mode=WAL")
            self._write_conn.execute("PRAGMA synchronous=NORMAL")
            self._write_conn.execute("PRAGMA busy_timeout=5000")
            self._write_conn.execute("PRAGMA foreign_keys=ON")
        return self._write_conn

    def _get_read_conn(self) -> sqlite3.Connection:
        """Get or create a read connection for the current thread."""
        tid = threading.get_ident()
        if tid not in self._read_conns:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._read_conns[tid] = conn
        return self._read_conns[tid]

    @contextmanager
    def _write_cursor(self):
        """Context manager for write operations."""
        conn = self._get_write_conn()
        cursor = conn.cursor()
        try:
            with self._write_lock:
                yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    @contextmanager
    def _read_cursor(self):
        """Context manager for read operations."""
        conn = self._get_read_conn()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    # -----------------------------------------------------------------------
    # Schema initialization & migration
    # -----------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Initialize database schema and run any pending migrations."""
        with self._write_cursor() as cur:
            # Create initial tables
            cur.executescript(SCHEMA_SQL)

            # Check current version
            cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
            current_version = cur.fetchone()[0]

            # Apply pending migrations
            for version, sql in sorted(MIGRATIONS.items()):
                if version > current_version:
                    logger.info("Applying schema migration v%d", version)
                    try:
                        cur.executescript(sql)
                        cur.execute(
                            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                            (version, f"Migration v{version}"),
                        )
                    except Exception as e:
                        logger.error("Migration v%d failed: %s", version, e)
                        raise

    # -----------------------------------------------------------------------
    # Position operations
    # -----------------------------------------------------------------------

    def save_position(self, position: Dict[str, Any]) -> str:
        """Insert or update a position record. Returns position_id."""
        with self._write_cursor() as cur:
            position_id = position["position_id"]
            cur.execute(
                """INSERT INTO positions (
                    position_id, strategy_name, symbol, underlying, option_type,
                    strike, expiry, direction, quantity, entry_price, entry_time,
                    exit_price, exit_time, mtm, status, parent_trade_id, meta_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(position_id) DO UPDATE SET
                    quantity=excluded.quantity,
                    exit_price=COALESCE(excluded.exit_price, positions.exit_price),
                    exit_time=COALESCE(excluded.exit_time, positions.exit_time),
                    mtm=excluded.mtm,
                    status=excluded.status,
                    meta_json=excluded.meta_json,
                    updated_at=datetime('now')""",
                (
                    position_id,
                    position.get("strategy_name", ""),
                    position.get("symbol", ""),
                    position.get("underlying", ""),
                    position.get("option_type", ""),
                    position.get("strike", 0.0),
                    position.get("expiry", ""),
                    position.get("direction", ""),
                    position.get("quantity", 0),
                    position.get("entry_price", 0.0),
                    position.get("entry_time", datetime.now(timezone.utc).isoformat()),
                    position.get("exit_price"),
                    position.get("exit_time"),
                    position.get("mtm", 0.0),
                    position.get("status", "OPEN"),
                    position.get("parent_trade_id"),
                    json.dumps(position.get("meta", {})),
                ),
            )
            return position_id

    def get_open_positions(self, underlying: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open positions, optionally filtered by underlying."""
        with self._read_cursor() as cur:
            if underlying:
                cur.execute(
                    "SELECT * FROM positions WHERE status = 'OPEN' AND underlying = ? ORDER BY entry_time DESC",
                    (underlying,),
                )
            else:
                cur.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_time DESC")
            return [dict(row) for row in cur.fetchall()]

    def get_all_positions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all positions (open and closed)."""
        with self._read_cursor() as cur:
            cur.execute(
                "SELECT * FROM positions ORDER BY entry_time DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_position_by_id(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get a single position by ID."""
        with self._read_cursor() as cur:
            cur.execute("SELECT * FROM positions WHERE position_id = ?", (position_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: Optional[str] = None,
    ) -> bool:
        """Mark a position as CLOSED."""
        with self._write_cursor() as cur:
            cur.execute(
                """UPDATE positions SET
                    status = 'CLOSED',
                    exit_price = ?,
                    exit_time = COALESCE(?, exit_time),
                    updated_at = datetime('now')
                WHERE position_id = ? AND status = 'OPEN'""",
                (exit_price, exit_time or datetime.now(timezone.utc).isoformat(), position_id),
            )
            return cur.rowcount > 0

    def update_position_mtm(self, position_id: str, mtm: float) -> None:
        """Update MTM for an open position."""
        with self._write_cursor() as cur:
            cur.execute(
                "UPDATE positions SET mtm = ?, updated_at = datetime('now') WHERE position_id = ?",
                (mtm, position_id),
            )

    # -----------------------------------------------------------------------
    # Order operations
    # -----------------------------------------------------------------------

    def save_order(self, order: Dict[str, Any]) -> str:
        """Insert or update an order record. Returns order_id."""
        with self._write_cursor() as cur:
            order_id = order["order_id"]
            cur.execute(
                """INSERT INTO orders (
                    order_id, position_id, broker_order_id, symbol, order_type,
                    side, quantity, price, trigger_price, status, filled_qty,
                    avg_fill_price, reject_reason, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(order_id) DO UPDATE SET
                    status=excluded.status,
                    filled_qty=excluded.filled_qty,
                    avg_fill_price=COALESCE(excluded.avg_fill_price, orders.avg_fill_price),
                    reject_reason=excluded.reject_reason,
                    updated_at=datetime('now')""",
                (
                    order_id,
                    order.get("position_id", ""),
                    order.get("broker_order_id"),
                    order.get("symbol", ""),
                    order.get("order_type", "MARKET"),
                    order.get("side", "BUY"),
                    order.get("quantity", 0),
                    order.get("price"),
                    order.get("trigger_price"),
                    order.get("status", "PENDING"),
                    order.get("filled_qty", 0),
                    order.get("avg_fill_price"),
                    order.get("reject_reason"),
                    json.dumps(order.get("meta", {})),
                ),
            )
            return order_id

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending/open orders."""
        with self._read_cursor() as cur:
            cur.execute(
                "SELECT * FROM orders WHERE status IN ('PENDING', 'OPEN', 'PARTIALLY_FILLED') ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]

    def get_orders_for_position(self, position_id: str) -> List[Dict[str, Any]]:
        """Get all orders for a position."""
        with self._read_cursor() as cur:
            cur.execute(
                "SELECT * FROM orders WHERE position_id = ? ORDER BY created_at ASC",
                (position_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    # -----------------------------------------------------------------------
    # Fill operations
    # -----------------------------------------------------------------------

    def save_fill(self, fill: Dict[str, Any]) -> str:
        """Save a fill record. Returns fill_id."""
        with self._write_cursor() as cur:
            fill_id = fill["fill_id"]
            cur.execute(
                """INSERT INTO fills (
                    fill_id, order_id, position_id, symbol, side,
                    quantity, price, fill_time, exchange_order_id, meta_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(fill_id) DO NOTHING""",
                (
                    fill_id,
                    fill.get("order_id", ""),
                    fill.get("position_id", ""),
                    fill.get("symbol", ""),
                    fill.get("side", "BUY"),
                    fill.get("quantity", 0),
                    fill.get("price", 0.0),
                    fill.get("fill_time", datetime.now(timezone.utc).isoformat()),
                    fill.get("exchange_order_id"),
                    json.dumps(fill.get("meta", {})),
                ),
            )
            return fill_id

    # -----------------------------------------------------------------------
    # Strategy state (key-value)
    # -----------------------------------------------------------------------

    def set_strategy_state(self, key: str, value: Any) -> None:
        """Store a strategy state value (JSON-serialized)."""
        with self._write_cursor() as cur:
            cur.execute(
                """INSERT INTO strategy_state (key, value_json, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=datetime('now')""",
                (key, json.dumps(value)),
            )

    def get_strategy_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a strategy state value."""
        with self._read_cursor() as cur:
            cur.execute("SELECT value_json FROM strategy_state WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row["value_json"])
            return default

    def get_all_strategy_state(self) -> Dict[str, Any]:
        """Get all strategy state as a dict."""
        with self._read_cursor() as cur:
            cur.execute("SELECT key, value_json FROM strategy_state")
            return {row["key"]: json.loads(row["value_json"]) for row in cur.fetchall()}

    # -----------------------------------------------------------------------
    # Risk state (key-value)
    # -----------------------------------------------------------------------

    def set_risk_state(self, key: str, value: Any) -> None:
        """Store a risk state value."""
        with self._write_cursor() as cur:
            cur.execute(
                """INSERT INTO risk_state (key, value_json, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=datetime('now')""",
                (key, json.dumps(value)),
            )

    def get_risk_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a risk state value."""
        with self._read_cursor() as cur:
            cur.execute("SELECT value_json FROM risk_state WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row["value_json"])
            return default

    def get_all_risk_state(self) -> Dict[str, Any]:
        """Get all risk state as a dict."""
        with self._read_cursor() as cur:
            cur.execute("SELECT key, value_json FROM risk_state")
            return {row["key"]: json.loads(row["value_json"]) for row in cur.fetchall()}

    # -----------------------------------------------------------------------
    # Shadow predictions
    # -----------------------------------------------------------------------

    def save_prediction(self, prediction: Dict[str, Any]) -> str:
        """Save a shadow mode prediction. Returns prediction_id."""
        with self._write_cursor() as cur:
            pred_id = prediction["prediction_id"]
            cur.execute(
                """INSERT INTO shadow_predictions (
                    prediction_id, timestamp, symbol, underlying, signal,
                    confidence, predicted_label, actual_outcome, model_version,
                    regime, feature_values, meta_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(prediction_id) DO NOTHING""",
                (
                    pred_id,
                    prediction.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    prediction.get("symbol", ""),
                    prediction.get("underlying", ""),
                    prediction.get("signal", "HOLD"),
                    prediction.get("confidence", 0.5),
                    prediction.get("predicted_label"),
                    prediction.get("actual_outcome"),
                    prediction.get("model_version", "unknown"),
                    prediction.get("regime"),
                    json.dumps(prediction.get("feature_values", {})),
                    json.dumps(prediction.get("meta", {})),
                ),
            )
            return pred_id

    def update_prediction_outcome(self, prediction_id: str, actual_outcome: int) -> None:
        """Update the actual outcome for a prediction."""
        with self._write_cursor() as cur:
            cur.execute(
                "UPDATE shadow_predictions SET actual_outcome = ? WHERE prediction_id = ?",
                (actual_outcome, prediction_id),
            )

    def get_predictions(
        self,
        limit: int = 1000,
        offset: int = 0,
        signal_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get prediction records."""
        with self._read_cursor() as cur:
            if signal_filter:
                cur.execute(
                    """SELECT * FROM shadow_predictions WHERE signal = ?
                    ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
                    (signal_filter, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT * FROM shadow_predictions ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            return [dict(row) for row in cur.fetchall()]

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary statistics for shadow predictions."""
        with self._read_cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM shadow_predictions")
            total = cur.fetchone()["total"]

            cur.execute(
                "SELECT COUNT(*) as count FROM shadow_predictions WHERE actual_outcome IS NOT NULL"
            )
            resolved = cur.fetchone()["count"]

            if resolved > 0:
                cur.execute(
                    """SELECT
                        AVG(CASE WHEN actual_outcome = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                        SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN actual_outcome = 0 THEN 1 ELSE 0 END) as losses
                    FROM shadow_predictions WHERE actual_outcome IS NOT NULL"""
                )
                row = cur.fetchone()
                accuracy = row["accuracy"]
                wins = row["wins"]
                losses = row["losses"]
            else:
                accuracy = 0.0
                wins = 0
                losses = 0

            return {
                "total_predictions": total,
                "resolved_predictions": resolved,
                "accuracy": accuracy,
                "wins": wins,
                "losses": losses,
            }

    # -----------------------------------------------------------------------
    # Audit log
    # -----------------------------------------------------------------------

    def log_audit(
        self,
        event_type: str,
        severity: str,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append an audit log entry. Returns row ID."""
        with self._write_cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (event_type, severity, component, message, data_json) VALUES (?, ?, ?, ?, ?)",
                (event_type, severity, component, message, json.dumps(data or {})),
            )
            return cur.lastrowid

    def get_audit_log(
        self,
        limit: int = 100,
        severity_filter: Optional[str] = None,
        component_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries with optional filters."""
        with self._read_cursor() as cur:
            query = "SELECT * FROM audit_log WHERE 1=1"
            params: List[Any] = []
            if severity_filter:
                query += " AND severity = ?"
                params.append(severity_filter)
            if component_filter:
                query += " AND component = ?"
                params.append(component_filter)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    # -----------------------------------------------------------------------
    # Performance metrics
    # -----------------------------------------------------------------------

    def save_metrics(self, run_id: str, metrics: Dict[str, float], params: Optional[Dict[str, Any]] = None) -> None:
        """Save performance metrics for a run."""
        with self._write_cursor() as cur:
            for metric_name, metric_value in metrics.items():
                cur.execute(
                    "INSERT INTO performance_metrics (run_id, metric_name, metric_value, params_json) VALUES (?, ?, ?, ?)",
                    (run_id, metric_name, metric_value, json.dumps(params or {})),
                )

    def get_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics for a specific run."""
        with self._read_cursor() as cur:
            cur.execute(
                "SELECT metric_name, metric_value FROM performance_metrics WHERE run_id = ?",
                (run_id,),
            )
            return {row["metric_name"]: row["metric_value"] for row in cur.fetchall()}

    # -----------------------------------------------------------------------
    # Daily summary
    # -----------------------------------------------------------------------

    def get_daily_summary(self, date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get daily summary for a given date (default: today)."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._read_cursor() as cur:
            cur.execute("SELECT * FROM daily_summary WHERE date = ?", (date_str,))
            row = cur.fetchone()
            return dict(row) if row else None

    def upsert_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Insert or update a daily summary."""
        with self._write_cursor() as cur:
            cur.execute(
                """INSERT INTO daily_summary (
                    date, total_trades, winning_trades, losing_trades,
                    gross_pnl, net_pnl, max_drawdown_pct, turnover, meta_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(date) DO UPDATE SET
                    total_trades=excluded.total_trades,
                    winning_trades=excluded.winning_trades,
                    losing_trades=excluded.losing_trades,
                    gross_pnl=excluded.gross_pnl,
                    net_pnl=excluded.net_pnl,
                    max_drawdown_pct=excluded.max_drawdown_pct,
                    turnover=excluded.turnover,
                    meta_json=excluded.meta_json""",
                (
                    summary["date"],
                    summary.get("total_trades", 0),
                    summary.get("winning_trades", 0),
                    summary.get("losing_trades", 0),
                    summary.get("gross_pnl", 0.0),
                    summary.get("net_pnl", 0.0),
                    summary.get("max_drawdown_pct", 0.0),
                    summary.get("turnover", 0.0),
                    json.dumps(summary.get("meta", {})),
                ),
            )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close all database connections."""
        if self._write_conn:
            self._write_conn.close()
            self._write_conn = None
        for tid, conn in self._read_conns.items():
            conn.close()
        self._read_conns.clear()
        logger.info("TradeDatabase closed")

    def __enter__(self) -> 'TradeDatabase':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()