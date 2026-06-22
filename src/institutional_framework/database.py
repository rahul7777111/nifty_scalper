"""High-Throughput SQLite Write-Ahead Logging (WAL) Interface.

Implements an asynchronous, thread-safe, and non-blocking database write pipeline for
latency-sensitive options telemetry logs, optimized for 8GB RAM CPU-only execution.
"""

from __future__ import annotations

import os
import queue
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

class AsyncTelemetryDatabase:
    """Thread-safe, non-blocking asynchronous writer to a SQLite WAL database."""
    
    def __init__(self, db_path: str, batch_size: int = 25, flush_interval_sec: float = 2.0):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.write_queue: queue.Queue[Tuple[str, tuple]] = queue.Queue()
        self.running = False
        self.writer_thread: Optional[threading.Thread] = None
        
    def initialize_schema(self):
        """Initializes tables using highly optimized synchronous settings."""
        conn = self._get_optimized_conn()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        tick_arrival TEXT,
                        latency_ms REAL,
                        spot REAL,
                        opt_bid REAL,
                        opt_ask REAL,
                        fill_price REAL,
                        slippage_ticks REAL,
                        shortfall REAL,
                        spread REAL,
                        quality TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        feature_vector TEXT,
                        ensemble_prob REAL,
                        regime TEXT,
                        actual_label INTEGER
                    )
                """)
        finally:
            conn.close()

    def _get_optimized_conn(self) -> sqlite3.Connection:
        """Configures institutional-grade performance parameters for SQLite."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-40000;")  # 40MB Page Cache
        return conn

    def start(self):
        """Starts the background worker thread."""
        if self.running:
            return
        self.running = True
        self.initialize_schema()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="AsyncDBWriter")
        self.writer_thread.start()

    def stop(self):
        """Signals shutdown and waits for background thread to drain queues."""
        if not self.running:
            return
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)

    def enqueue_write(self, sql: str, params: tuple):
        """Non-blocking insert. Pushes the SQL command to the fast thread-safe queue."""
        self.write_queue.put((sql, params))

    def _writer_loop(self):
        """Background thread executing bulk batched database inserts."""
        conn = self._get_optimized_conn()
        batch: List[Tuple[str, tuple]] = []
        
        while self.running or not self.write_queue.empty():
            try:
                # Blocks with timeout to avoid CPU polling spikes
                item = self.write_queue.get(timeout=self.flush_interval_sec)
                batch.append(item)
                
                if len(batch) >= self.batch_size:
                    self._flush_batch(conn, batch)
                    batch.clear()
            except queue.Empty:
                if batch:
                    self._flush_batch(conn, batch)
                    batch.clear()
            except Exception:
                # Prevent database loop crashes from terminating the runtime
                pass
                
        # Final drain of remaining items
        if batch:
            self._flush_batch(conn, batch)
        conn.close()

    def _flush_batch(self, conn: sqlite3.Connection, batch: List[Tuple[str, tuple]]):
        """Flushes a list of transactions in a single atomic database operation."""
        try:
            with conn:
                for sql, params in batch:
                    conn.execute(sql, params)
        except Exception:
            pass
