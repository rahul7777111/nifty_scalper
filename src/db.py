import sqlite3
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

    def insert_trade(self, trade_id: str, symbol: str, side: str, quantity: int, entry_price: float, exit_price: float, realized_pnl: float):
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
