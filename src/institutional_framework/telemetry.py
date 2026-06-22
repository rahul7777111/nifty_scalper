"""Operational Telemetry and Latency Logging Engine.

Tracks tick-to-fill latency metrics, slip metrics, implementation shortfall (IS), and
sends structured results to the non-blocking asynchronous SQLite writer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from .database import AsyncTelemetryDatabase

@dataclass
class ExecutionTelemetryRecord:
    timestamp: str
    tick_arrival: str
    latency_ms: float
    spot: float
    opt_bid: float
    opt_ask: float
    fill_price: float
    slippage_ticks: float
    shortfall: float
    spread: float
    quality: str

class TelemetryTracker:
    """Measures precise multi-point latency and execution quality parameters."""
    
    def __init__(self, db_writer: AsyncTelemetryDatabase, tick_size: float = 0.05):
        self.db_writer = db_writer
        self.tick_size = tick_size

    def measure_execution(
        self,
        t_tick_arrival: float,        # Unix fractional timestamp of tick arrival
        t_signal_generated: float,   # Unix fractional timestamp of signal generation
        t_order_sent: float,         # Unix fractional timestamp of order sending
        t_broker_fill: float,         # Unix fractional timestamp of broker fill confirmation
        spot: float,
        opt_bid: float,
        opt_ask: float,
        fill_price: float,
        order_type: str = "BUY"
    ) -> ExecutionTelemetryRecord:
        """Processes time logs and calculates shortfall and slip.
        
        Args:
            t_tick_arrival: System tick receipt epoch.
            t_signal_generated: Signal computation finish epoch.
            t_order_sent: Packet transmission start epoch.
            t_broker_fill: Fill acknowledgment epoch.
            spot: NIFTY spot index level at tick arrival.
            opt_bid: Option bid price at arrival.
            opt_ask: Option ask price at arrival.
            fill_price: Final execution price returned by the broker.
            order_type: Direction ("BUY" or "SELL").
        """
        # Latency calculations in milliseconds
        total_loop_ms = (t_broker_fill - t_tick_arrival) * 1000.0
        
        # Spread calculation
        spread = abs(opt_ask - opt_bid)
        mid_price = (opt_bid + opt_ask) / 2.0
        
        # Slippage in ticks
        if order_type.upper() == "BUY":
            slippage_ticks = (fill_price - opt_ask) / self.tick_size
            shortfall = fill_price - mid_price
        else:
            slippage_ticks = (opt_bid - fill_price) / self.tick_size
            shortfall = mid_price - fill_price
            
        # Determine execution quality category
        if total_loop_ms < 150.0 and slippage_ticks <= 1.0:
            quality = "EXCELLENT"
        elif total_loop_ms < 300.0 and slippage_ticks <= 3.0:
            quality = "FAIR"
        else:
            quality = "POOR"
            
        record = ExecutionTelemetryRecord(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            tick_arrival=datetime.fromtimestamp(t_tick_arrival, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            latency_ms=total_loop_ms,
            spot=spot,
            opt_bid=opt_bid,
            opt_ask=opt_ask,
            fill_price=fill_price,
            slippage_ticks=round(slippage_ticks, 2),
            shortfall=round(shortfall, 4),
            spread=round(spread, 4),
            quality=quality
        )
        
        # Enqueue database log operation without blocking
        sql = """
            INSERT INTO telemetry 
            (timestamp, tick_arrival, latency_ms, spot, opt_bid, opt_ask, fill_price, slippage_ticks, shortfall, spread, quality) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            record.timestamp,
            record.tick_arrival,
            record.latency_ms,
            record.spot,
            record.opt_bid,
            record.opt_ask,
            record.fill_price,
            record.slippage_ticks,
            record.shortfall,
            record.spread,
            record.quality
        )
        self.db_writer.enqueue_write(sql, params)
        
        return record
