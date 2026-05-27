import json
import sqlite3
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite connections and trade history persistence."""
    
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = Path(__file__).parent.parent / db_path
        self._init_db()

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
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def insert_trade(self, trade_id: str, symbol: str, side: str, quantity: int, entry_price: float = None, exit_price: float = None, realized_pnl: float = 0.0):
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

    def insert_trade_event(self, event: dict) -> None:
        """Persist a full trade lifecycle event for journaling."""
        try:
            trade_id = str(event.get("trade_id") or "").strip()
            if not trade_id:
                return
            legs_json = json.dumps(event.get("legs") or [], default=str, ensure_ascii=False)
            meta_json = json.dumps(event.get("meta") or {}, default=str, ensure_ascii=False)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO trade_events (
                        trade_id, event, position_type, name, ts, mtm, realized, reason, margin_required, legs_json, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        trade_id,
                        str(event.get("event") or ""),
                        str(event.get("position_type") or ""),
                        str(event.get("name") or ""),
                        float(event.get("ts") or time.time()),
                        event.get("mtm"),
                        event.get("realized"),
                        str(event.get("reason") or "") or None,
                        event.get("margin_required"),
                        legs_json,
                        meta_json,
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
                    'SELECT trade_id, event, position_type, name, ts, mtm, realized, reason, margin_required, legs_json, meta_json FROM trade_events ORDER BY id DESC LIMIT ?',
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
