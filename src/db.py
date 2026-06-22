import json
import sqlite3
import time
import logging
import hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

try:
    import pytz
except Exception:
    pytz = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

if ZoneInfo is not None:
    try:
        IST = ZoneInfo("Asia/Kolkata")
    except Exception:
        IST = None
elif pytz is not None:
    try:
        IST = pytz.timezone("Asia/Kolkata")
    except Exception:
        IST = None
else:
    IST = None

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite connections and trade history persistence."""
    
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = Path(__file__).parent.parent / db_path
        self._init_db()

    def close(self) -> None:
        """Compatibility close hook for callers that manage lifecycles explicitly.

        This manager currently uses short-lived context-managed SQLite
        connections, so there is no persistent connection to tear down here.
        The method exists so shutdown code can safely call `close()` on any
        DB-like dependency without branching on implementation details.
        """
        return

    def _init_db(self):
        """Creates the necessary tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price REAL,
                        exit_price REAL,
                        realized_pnl REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT NOT NULL,
                        event TEXT NOT NULL,
                        position_type TEXT,
                        name TEXT,
                        ts REAL,
                        mtm REAL,
                        realized REAL,
                        reason TEXT,
                        margin_required REAL,
                        legs_json TEXT,
                        meta_json TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Daily summary view/table could also go here
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_summary (
                        date DATE PRIMARY KEY,
                        total_pnl REAL DEFAULT 0,
                        trades_count INTEGER DEFAULT 0
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prediction_id TEXT UNIQUE,
                        ts REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        exchange TEXT,
                        token TEXT,
                        option_symbol TEXT,
                        underlying_price REAL,
                        direction TEXT,
                        regime TEXT,
                        model_version TEXT,
                        model_artifact_path TEXT,
                        model_checksum TEXT,
                        label_policy_version TEXT,
                        feature_set_version TEXT,
                        threshold REAL,
                        probability REAL,
                        prediction REAL,
                        predicted_class INTEGER,
                        confidence REAL,
                        confidence_bucket TEXT,
                        strategy_context TEXT,
                        strategy_signal TEXT,
                        source TEXT,
                        reason TEXT,
                        trade_candidate INTEGER DEFAULT 0,
                        trade_taken INTEGER DEFAULT 0,
                        no_trade_reason TEXT,
                        feature_snapshot_json TEXT,
                        features_json TEXT,
                        feature_vector_checksum TEXT,
                        future_label_status TEXT DEFAULT 'pending',
                        resolved_label INTEGER,
                        realized_forward_return REAL,
                        realized_trade_pnl REAL,
                        horizon_bars INTEGER,
                        trade_id TEXT
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT NOT NULL,
                        prediction_id TEXT,
                        entry_time TEXT,
                        exit_time TEXT,
                        duration_minutes REAL,
                        strategy TEXT,
                        regime TEXT,
                        strategy_signal TEXT,
                        entry_reason TEXT,
                        exit_reason TEXT,
                        model_version TEXT,
                        label_policy_version TEXT,
                        feature_set_version TEXT,
                        probability REAL,
                        threshold REAL,
                        confidence REAL,
                        confidence_bucket TEXT,
                        gross_pnl REAL,
                        estimated_costs REAL,
                        estimated_slippage REAL,
                        pnl REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                self._ensure_column(cursor, "predictions", "prediction_id", "TEXT")
                self._ensure_column(cursor, "predictions", "ts", "REAL")
                self._ensure_column(cursor, "predictions", "timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP")
                self._ensure_column(cursor, "predictions", "symbol", "TEXT")
                self._ensure_column(cursor, "predictions", "exchange", "TEXT")
                self._ensure_column(cursor, "predictions", "token", "TEXT")
                self._ensure_column(cursor, "predictions", "option_symbol", "TEXT")
                self._ensure_column(cursor, "predictions", "underlying_price", "REAL")
                self._ensure_column(cursor, "predictions", "direction", "TEXT")
                self._ensure_column(cursor, "predictions", "regime", "TEXT")
                self._ensure_column(cursor, "predictions", "model_version", "TEXT")
                self._ensure_column(cursor, "predictions", "model_artifact_path", "TEXT")
                self._ensure_column(cursor, "predictions", "model_checksum", "TEXT")
                self._ensure_column(cursor, "predictions", "label_policy_version", "TEXT")
                self._ensure_column(cursor, "predictions", "feature_set_version", "TEXT")
                self._ensure_column(cursor, "predictions", "threshold", "REAL")
                self._ensure_column(cursor, "predictions", "probability", "REAL")
                self._ensure_column(cursor, "predictions", "prediction", "REAL")
                self._ensure_column(cursor, "predictions", "predicted_class", "INTEGER")
                self._ensure_column(cursor, "predictions", "confidence", "REAL")
                self._ensure_column(cursor, "predictions", "confidence_bucket", "TEXT")
                self._ensure_column(cursor, "predictions", "strategy_context", "TEXT")
                self._ensure_column(cursor, "predictions", "strategy_signal", "TEXT")
                self._ensure_column(cursor, "predictions", "source", "TEXT")
                self._ensure_column(cursor, "predictions", "reason", "TEXT")
                self._ensure_column(cursor, "predictions", "trade_candidate", "INTEGER DEFAULT 0")
                self._ensure_column(cursor, "predictions", "trade_taken", "INTEGER DEFAULT 0")
                self._ensure_column(cursor, "predictions", "no_trade_reason", "TEXT")
                self._ensure_column(cursor, "predictions", "feature_snapshot_json", "TEXT")
                self._ensure_column(cursor, "predictions", "features_json", "TEXT")
                self._ensure_column(cursor, "predictions", "feature_vector_checksum", "TEXT")
                self._ensure_column(cursor, "predictions", "future_label_status", "TEXT DEFAULT 'pending'")
                self._ensure_column(cursor, "predictions", "resolved_label", "INTEGER")
                self._ensure_column(cursor, "predictions", "realized_forward_return", "REAL")
                self._ensure_column(cursor, "predictions", "realized_trade_pnl", "REAL")
                self._ensure_column(cursor, "predictions", "horizon_bars", "INTEGER")
                self._ensure_column(cursor, "predictions", "trade_id", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "prediction_id", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "duration_minutes", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "strategy_signal", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "entry_reason", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "exit_reason", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "model_version", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "label_policy_version", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "feature_set_version", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "probability", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "threshold", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "confidence_bucket", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "gross_pnl", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "estimated_costs", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "estimated_slippage", "REAL")
                # Paper journal validation fields
                self._ensure_column(cursor, "trade_events", "symbol", "TEXT")
                self._ensure_column(cursor, "trade_events", "strike", "REAL")
                self._ensure_column(cursor, "trade_events", "option_type", "TEXT")
                self._ensure_column(cursor, "trade_events", "side", "TEXT")
                self._ensure_column(cursor, "trade_events", "quantity", "INTEGER")
                self._ensure_column(cursor, "trade_events", "bid", "REAL")
                self._ensure_column(cursor, "trade_events", "ask", "REAL")
                self._ensure_column(cursor, "trade_events", "ltp", "REAL")
                self._ensure_column(cursor, "trade_events", "execution_price", "REAL")
                self._ensure_column(cursor, "trade_events", "execution_price_source", "TEXT")
                self._ensure_column(cursor, "trade_events", "entry_price", "REAL")
                self._ensure_column(cursor, "trade_events", "exit_price", "REAL")
                self._ensure_column(cursor, "trade_events", "gross_pnl", "REAL")
                self._ensure_column(cursor, "trade_events", "net_pnl", "REAL")
                self._ensure_column(cursor, "trade_events", "spread_cost", "REAL")
                self._ensure_column(cursor, "trade_events", "slippage_cost", "REAL")
                self._ensure_column(cursor, "trade_events", "brokerage_cost", "REAL")
                self._ensure_column(cursor, "trade_events", "exit_reason", "TEXT")
                self._ensure_column(cursor, "trade_events", "risk_filter_decisions_json", "TEXT")
                self._ensure_column(cursor, "trade_events", "realized_slippage_pct", "REAL")
                self._ensure_column(cursor, "trade_events", "paper_mode", "INTEGER DEFAULT 1")
                # Additional required paper journal fields
                self._ensure_column(cursor, "trade_events", "timestamp", "TEXT")
                self._ensure_column(cursor, "trade_events", "skip_reason", "TEXT")
                self._ensure_column(cursor, "trade_events", "spread_pct_at_entry", "REAL")
                self._ensure_column(cursor, "trade_events", "spread_pct_at_exit", "REAL")
                self._ensure_column(cursor, "trade_events", "filter_premium_ok", "INTEGER")
                self._ensure_column(cursor, "trade_events", "filter_spread_ok", "INTEGER")
                self._ensure_column(cursor, "trade_events", "filter_bid_ask_ok", "INTEGER")
                self._ensure_column(cursor, "trade_outcomes", "net_pnl", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "spread_cost", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "slippage_cost", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "brokerage_cost", "REAL")
                self._ensure_column(cursor, "trade_outcomes", "exit_reason", "TEXT")
                self._ensure_column(cursor, "trade_outcomes", "spread_pct", "REAL")
                self._ensure_prediction_id_unique_constraint(cursor)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_prediction_id ON predictions(prediction_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_trade_id ON predictions(trade_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_outcomes_trade_id ON trade_outcomes(trade_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_outcomes_prediction_id ON trade_outcomes(prediction_id)")
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _ensure_column(self, cursor: sqlite3.Cursor, table: str, column_name: str, column_sql: str) -> None:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            existing = {str(row[1]) for row in cursor.fetchall()}
            if column_name not in existing:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_sql}")
        except Exception as exc:
            logger.warning("Failed ensuring column %s.%s: %s", table, column_name, exc)

    def _ensure_prediction_id_unique_constraint(self, cursor: sqlite3.Cursor) -> None:
        """Upgrade legacy prediction tables so ON CONFLICT(prediction_id) is valid."""
        try:
            cursor.execute(
                """
                UPDATE predictions
                SET prediction_id = NULL
                WHERE prediction_id IS NOT NULL AND TRIM(prediction_id) = ''
                """
            )
            cursor.execute(
                """
                SELECT prediction_id, GROUP_CONCAT(id), COUNT(1)
                FROM predictions
                WHERE prediction_id IS NOT NULL AND TRIM(prediction_id) != ''
                GROUP BY prediction_id
                HAVING COUNT(1) > 1
                """
            )
            for prediction_id, ids_csv, _count in cursor.fetchall():
                ids = [int(x) for x in str(ids_csv or "").split(",") if str(x).strip().isdigit()]
                for ordinal, row_id in enumerate(ids[1:], start=2):
                    cursor.execute(
                        "UPDATE predictions SET prediction_id = ? WHERE id = ?",
                        (f"{prediction_id}__DUP{ordinal}", row_id),
                    )
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_prediction_id ON predictions(prediction_id)")
        except Exception as exc:
            logger.warning("Could not ensure unique prediction_id constraint: %s", exc)

    def insert_trade(self, trade_id: str, symbol: str, side: str = "", quantity: int = 0, entry_price: float = None, exit_price: float = None, realized_pnl: float = 0.0):
        """Inserts a completed trade into the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (trade_id, symbol, side, quantity, entry_price, exit_price, realized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (trade_id, symbol, side, quantity, entry_price, exit_price, realized_pnl))
                
                # Update daily summary
                today = datetime.now().date().isoformat()
                cursor.execute('''
                    INSERT INTO daily_summary (date, total_pnl, trades_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(date) DO UPDATE SET 
                        total_pnl = total_pnl + excluded.total_pnl,
                        trades_count = trades_count + 1
                ''', (today, realized_pnl))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert trade {trade_id} into database: {e}")

    def _calculate_duration_minutes(self, entry_time: Optional[str], exit_time: Optional[str]) -> Optional[float]:
        def _parse(value: Optional[str]) -> Optional[datetime]:
            raw = str(value or "").strip()
            if not raw:
                return None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(raw, fmt)
                except Exception:
                    continue
            return None

        start = _parse(entry_time)
        end = _parse(exit_time)
        if start is None or end is None or end < start:
            return None
        return (end - start).total_seconds() / 60.0

    def _prediction_payload_for_trade(self, cursor: sqlite3.Cursor, trade_id: str) -> Dict[str, Any]:
        try:
            cursor.execute(
                """
                SELECT prediction_id, model_version, label_policy_version, feature_set_version,
                       probability, threshold, confidence, confidence_bucket, regime, strategy_signal, reason
                FROM predictions
                WHERE trade_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (str(trade_id),),
            )
            row = cursor.fetchone()
            if row is None:
                return {}
            keys = [
                "prediction_id",
                "model_version",
                "label_policy_version",
                "feature_set_version",
                "probability",
                "threshold",
                "confidence",
                "confidence_bucket",
                "regime",
                "strategy_signal",
                "entry_reason",
            ]
            return {key: row[idx] for idx, key in enumerate(keys)}
        except Exception:
            return {}

    def insert_trade_outcome(self, outcome: Dict[str, Any]) -> None:
        try:
            trade_id = str(outcome.get("trade_id") or "").strip()
            if not trade_id:
                return
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                payload = self._prediction_payload_for_trade(cursor, trade_id)
                prediction_id = str(outcome.get("prediction_id") or payload.get("prediction_id") or "").strip() or None
                entry_time = str(outcome.get("entry_time") or "") or None
                exit_time = str(outcome.get("exit_time") or "") or None
                duration_minutes = outcome.get("duration_minutes")
                if duration_minutes is None:
                    duration_minutes = self._calculate_duration_minutes(entry_time, exit_time)
                gross_pnl = outcome.get("gross_pnl")
                pnl = outcome.get("pnl")
                if gross_pnl is None and pnl is not None:
                    gross_pnl = pnl
                cursor.execute(
                    """
                    INSERT INTO trade_outcomes (
                        trade_id, prediction_id, entry_time, exit_time, duration_minutes, strategy, regime,
                        strategy_signal, entry_reason, exit_reason, model_version, label_policy_version,
                        feature_set_version, probability, threshold, confidence, confidence_bucket, gross_pnl,
                        estimated_costs, estimated_slippage, pnl,
                        net_pnl, spread_cost, slippage_cost, brokerage_cost, spread_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id,
                        prediction_id,
                        entry_time,
                        exit_time,
                        duration_minutes,
                        str(outcome.get("strategy") or ""),
                        str(outcome.get("regime") or payload.get("regime") or ""),
                        str(outcome.get("strategy_signal") or payload.get("strategy_signal") or ""),
                        str(outcome.get("entry_reason") or payload.get("entry_reason") or ""),
                        str(outcome.get("exit_reason") or ""),
                        str(outcome.get("model_version") or payload.get("model_version") or ""),
                        str(outcome.get("label_policy_version") or payload.get("label_policy_version") or ""),
                        str(outcome.get("feature_set_version") or payload.get("feature_set_version") or ""),
                        outcome.get("probability", payload.get("probability")),
                        outcome.get("threshold", payload.get("threshold")),
                        outcome.get("confidence", payload.get("confidence")),
                        str(outcome.get("confidence_bucket") or payload.get("confidence_bucket") or ""),
                        gross_pnl,
                        outcome.get("estimated_costs"),
                        outcome.get("estimated_slippage"),
                        pnl,
                        outcome.get("net_pnl"),
                        outcome.get("spread_cost"),
                        outcome.get("slippage_cost"),
                        outcome.get("brokerage_cost"),
                        outcome.get("spread_pct"),
                    ),
                )
                if prediction_id:
                    cursor.execute(
                        """
                        UPDATE predictions
                        SET trade_taken = 1,
                            trade_id = COALESCE(trade_id, ?),
                            realized_trade_pnl = ?
                        WHERE prediction_id = ?
                        """,
                        (trade_id, pnl, prediction_id),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert trade outcome for {outcome.get('trade_id')}: {e}")

    def insert_prediction(self, record: Dict[str, Any]) -> str:
        prediction_id = str(record.get("prediction_id") or "").strip()
        if not prediction_id:
            raw = f"{record.get('ts')}|{record.get('symbol')}|{record.get('probability')}|{record.get('threshold')}"
            prediction_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        prediction_id, ts, symbol, exchange, token, option_symbol, underlying_price, direction, regime,
                        model_version, model_artifact_path, model_checksum, label_policy_version, feature_set_version,
                        threshold, probability, prediction, predicted_class, confidence, confidence_bucket,
                        strategy_context, strategy_signal, source, reason, trade_candidate, trade_taken,
                        no_trade_reason, feature_snapshot_json, features_json, feature_vector_checksum,
                        future_label_status, resolved_label, realized_forward_return, realized_trade_pnl, horizon_bars, trade_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(prediction_id) DO UPDATE SET
                        trade_candidate = excluded.trade_candidate,
                        trade_taken = excluded.trade_taken,
                        no_trade_reason = excluded.no_trade_reason,
                        source = excluded.source,
                        reason = excluded.reason,
                        probability = excluded.probability,
                        prediction = excluded.prediction,
                        predicted_class = excluded.predicted_class,
                        confidence = excluded.confidence,
                        confidence_bucket = excluded.confidence_bucket,
                        regime = excluded.regime,
                        strategy_context = excluded.strategy_context,
                        strategy_signal = excluded.strategy_signal,
                        feature_snapshot_json = excluded.feature_snapshot_json,
                        features_json = excluded.features_json,
                        feature_vector_checksum = excluded.feature_vector_checksum,
                        threshold = excluded.threshold,
                        horizon_bars = excluded.horizon_bars
                    """,
                    (
                        prediction_id,
                        float(record.get("ts") or time.time()),
                        str(record.get("symbol") or ""),
                        str(record.get("exchange") or ""),
                        str(record.get("token") or ""),
                        str(record.get("option_symbol") or ""),
                        record.get("underlying_price"),
                        str(record.get("direction") or ""),
                        str(record.get("regime") or ""),
                        str(record.get("model_version") or ""),
                        str(record.get("model_artifact_path") or ""),
                        str(record.get("model_checksum") or ""),
                        str(record.get("label_policy_version") or ""),
                        str(record.get("feature_set_version") or ""),
                        record.get("threshold"),
                        record.get("probability"),
                        record.get("prediction"),
                        record.get("predicted_class"),
                        record.get("confidence"),
                        str(record.get("confidence_bucket") or ""),
                        str(record.get("strategy_context") or ""),
                        str(record.get("strategy_signal") or ""),
                        str(record.get("source") or "ACTIVE"),
                        str(record.get("reason") or ""),
                        1 if bool(record.get("trade_candidate")) else 0,
                        1 if bool(record.get("trade_taken")) else 0,
                        str(record.get("no_trade_reason") or ""),
                        json.dumps(record.get("feature_snapshot") or {}, default=str),
                        json.dumps(record.get("features") or {}, default=str),
                        str(record.get("feature_vector_checksum") or ""),
                        str(record.get("future_label_status") or "pending"),
                        record.get("resolved_label"),
                        record.get("realized_forward_return"),
                        record.get("realized_trade_pnl"),
                        record.get("horizon_bars"),
                        str(record.get("trade_id") or "") or None,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert prediction {prediction_id}: {e}")
        return prediction_id

    def link_prediction_to_trade(self, prediction_id: str, trade_id: str, *, trade_taken: bool = True, no_trade_reason: Optional[str] = None) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE predictions
                    SET trade_id = ?, trade_taken = ?, no_trade_reason = COALESCE(?, no_trade_reason)
                    WHERE prediction_id = ?
                    """,
                    (str(trade_id), 1 if trade_taken else 0, no_trade_reason, str(prediction_id)),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to link prediction {prediction_id} to trade {trade_id}: {e}")

    def resolve_prediction(self, prediction_id: str, *, resolved_label: Optional[int], realized_forward_return: Optional[float], future_label_status: str, realized_trade_pnl: Optional[float] = None) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE predictions
                    SET resolved_label = ?, realized_forward_return = ?, future_label_status = ?, realized_trade_pnl = COALESCE(?, realized_trade_pnl)
                    WHERE prediction_id = ?
                    """,
                    (resolved_label, realized_forward_return, str(future_label_status), realized_trade_pnl, str(prediction_id)),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to resolve prediction {prediction_id}: {e}")

    def list_recent_prediction_markers(self, limit: int = 200) -> List[Dict[str, Any]]:
        try:
            limit = max(1, int(limit))
        except Exception:
            limit = 200
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT prediction_id, ts, symbol, probability, predicted_class, confidence_bucket, regime, trade_candidate, trade_taken
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list recent prediction markers: {e}")
            return []

    def insert_trade_event(self, event: dict) -> None:
        """Persist a full trade lifecycle event for journaling."""
        try:
            trade_id = str(event.get("trade_id") or "").strip()
            if not trade_id:
                return
            legs_json = json.dumps(event.get("legs") or [], default=str, ensure_ascii=False)
            meta_json = json.dumps(event.get("meta") or {}, default=str, ensure_ascii=False)
            risk_filter_json = json.dumps(event.get("risk_filter_decisions") or {}, default=str, ensure_ascii=False)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                def _iso_ts(ts_val: Any) -> Optional[str]:
                    try:
                        return dt_datetime.fromtimestamp(float(ts_val), tz=IST if IST else None).isoformat()
                    except Exception:
                        try:
                            return str(ts_val) if ts_val else None
                        except Exception:
                            return None

                # Derive ISO timestamp from numeric ts if not provided
                event_ts = float(event.get("ts") or time.time())
                iso_ts = str(event.get("timestamp") or "") or _iso_ts(event_ts)

                # Resolve filter bools
                filter_premium = event.get("filter_premium_ok")
                filter_spread = event.get("filter_spread_ok")
                filter_bid_ask = event.get("filter_bid_ask_ok")

                cursor.execute(
                    '''
                    INSERT INTO trade_events (
                        trade_id, event, position_type, name, ts, mtm, realized, reason, margin_required,
                        legs_json, meta_json, symbol, strike, option_type, side, quantity,
                        bid, ask, ltp, execution_price, execution_price_source,
                        entry_price, exit_price, gross_pnl, net_pnl,
                        spread_cost, slippage_cost, brokerage_cost,
                        exit_reason, risk_filter_decisions_json, realized_slippage_pct, paper_mode,
                        timestamp, skip_reason, spread_pct_at_entry, spread_pct_at_exit,
                        filter_premium_ok, filter_spread_ok, filter_bid_ask_ok
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        trade_id,
                        str(event.get("event") or ""),
                        str(event.get("position_type") or ""),
                        str(event.get("name") or ""),
                        event_ts,
                        event.get("mtm"),
                        event.get("realized"),
                        str(event.get("reason") or "") or None,
                        event.get("margin_required"),
                        legs_json,
                        meta_json,
                        str(event.get("symbol") or "") or None,
                        event.get("strike"),
                        str(event.get("option_type") or "") or None,
                        str(event.get("side") or "") or None,
                        event.get("quantity"),
                        event.get("bid"),
                        event.get("ask"),
                        event.get("ltp"),
                        event.get("execution_price"),
                        str(event.get("execution_price_source") or "") or None,
                        event.get("entry_price"),
                        event.get("exit_price"),
                        event.get("gross_pnl"),
                        event.get("net_pnl"),
                        event.get("spread_cost"),
                        event.get("slippage_cost"),
                        event.get("brokerage_cost"),
                        str(event.get("exit_reason") or "") or None,
                        risk_filter_json or None,
                        event.get("realized_slippage_pct"),
                        1 if bool(event.get("paper_mode", True)) else 0,
                        iso_ts,
                        str(event.get("skip_reason") or "") or None,
                        event.get("spread_pct_at_entry"),
                        event.get("spread_pct_at_exit"),
                        1 if filter_premium is True else (0 if filter_premium is False else None),
                        1 if filter_spread is True else (0 if filter_spread is False else None),
                        1 if filter_bid_ask is True else (0 if filter_bid_ask is False else None),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert trade event {event.get('trade_id')} into database: {e}")

    def list_recent_trade_events(self, limit: int = 200):
        """Return a compact list of recent trade events for journaling/UI."""
        try:
            limit = max(1, int(limit))
        except Exception:
            limit = 200
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT trade_id, event, position_type, name, ts, mtm, realized, reason, margin_required, '
                    'legs_json, meta_json, symbol, strike, option_type, side, quantity, '
                    'bid, ask, ltp, execution_price, execution_price_source, '
                    'entry_price, exit_price, gross_pnl, net_pnl, '
                    'spread_cost, slippage_cost, brokerage_cost, '
                    'exit_reason, risk_filter_decisions_json, realized_slippage_pct, paper_mode, '
                    'timestamp, skip_reason, spread_pct_at_entry, spread_pct_at_exit, '
                    'filter_premium_ok, filter_spread_ok, filter_bid_ask_ok '
                    'FROM trade_events ORDER BY id DESC LIMIT ?',
                    (limit,),
                )
                rows = []
                for row in cursor.fetchall():
                    rows.append({
                        "trade_id": row["trade_id"],
                        "event": row["event"],
                        "position_type": row["position_type"],
                        "name": row["name"],
                        "ts": row["ts"],
                        "mtm": row["mtm"],
                        "realized": row["realized"],
                        "reason": row["reason"],
                        "margin_required": row["margin_required"],
                        "legs": json.loads(row["legs_json"] or "[]"),
                        "meta": json.loads(row["meta_json"] or "{}"),
                        "symbol": row["symbol"],
                        "strike": row["strike"],
                        "option_type": row["option_type"],
                        "side": row["side"],
                        "quantity": row["quantity"],
                        "bid": row["bid"],
                        "ask": row["ask"],
                        "ltp": row["ltp"],
                        "execution_price": row["execution_price"],
                        "execution_price_source": row["execution_price_source"],
                        "entry_price": row["entry_price"],
                        "exit_price": row["exit_price"],
                        "gross_pnl": row["gross_pnl"],
                        "net_pnl": row["net_pnl"],
                        "spread_cost": row["spread_cost"],
                        "slippage_cost": row["slippage_cost"],
                        "brokerage_cost": row["brokerage_cost"],
                        "exit_reason": row["exit_reason"],
                        "risk_filter_decisions": json.loads(row["risk_filter_decisions_json"] or "{}"),
                        "realized_slippage_pct": row["realized_slippage_pct"],
                        "paper_mode": bool(row["paper_mode"]),
                        "timestamp": row["timestamp"],
                        "skip_reason": row["skip_reason"],
                        "spread_pct_at_entry": row["spread_pct_at_entry"],
                        "spread_pct_at_exit": row["spread_pct_at_exit"],
                        "filter_premium_ok": bool(row["filter_premium_ok"]) if row["filter_premium_ok"] is not None else None,
                        "filter_spread_ok": bool(row["filter_spread_ok"]) if row["filter_spread_ok"] is not None else None,
                        "filter_bid_ask_ok": bool(row["filter_bid_ask_ok"]) if row["filter_bid_ask_ok"] is not None else None,
                    })
                return rows
        except Exception as e:
            logger.error(f"Failed to list recent trade events: {e}")
            return []

    def get_todays_realized_pnl(self) -> float:
        """Helper to quickly fetch today's accumulated PnL for the Kill Switch."""
        try:
            today = datetime.now().date().isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT total_pnl FROM daily_summary WHERE date = ?', (today,))
                row = cursor.fetchone()
                if row:
                    return float(row[0])
                return 0.0
        except Exception as e:
            logger.error(f"Failed to aggregate today's PnL: {e}")
            return 0.0
