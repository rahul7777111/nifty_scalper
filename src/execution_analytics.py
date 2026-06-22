import time
import logging
from collections import deque
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ExecutionAnalyticsEngine:
    """Execution Quality Analytics Engine.
    
    Tracks expected vs actual entry/exit pricing, latency, entry spreads,
    rejections, and calculates rolling statistics with zero database overhead.
    """

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        
        # O(1) rolling statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.total_slippage = 0.0
        self.worst_slippage = 0.0
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0
        
        # deque cache of recent records for visual representation in UI (last 20)
        self.recent_records = deque(maxlen=20)

    def record_execution(self, db_manager, trade_id: str, symbol: str, expected_price: float, actual_price: float, latency_ms: float, spread_entry: float, spread_widening: float, status: str, db_executor: Any = None) -> dict:
        """Processes a new execution tick, updates rolling stats, and logs to SQLite database."""
        try:
            expected = float(expected_price)
            actual = float(actual_price)
            slippage = actual - expected if expected > 0 else 0.0
            
            # For short/sell trades, negative slippage might represent selling at a worse price.
            # Let's standardize slippage as absolute deviance or directional. 
            # We'll use actual - expected as standard.
            
            # Update O(1) stats
            self.total_orders += 1
            if status.upper() == "FILLED" or status.upper() == "SUCCESS":
                self.filled_orders += 1
                self.total_slippage += abs(slippage)
                if abs(slippage) > self.worst_slippage:
                    self.worst_slippage = abs(slippage)
                
                self.total_latency_ms += latency_ms
                if latency_ms > self.max_latency_ms:
                    self.max_latency_ms = latency_ms

            record = {
                "trade_id": trade_id,
                "symbol": symbol,
                "expected_price": expected,
                "actual_price": actual,
                "slippage": slippage,
                "latency_ms": latency_ms,
                "spread_entry": spread_entry,
                "spread_widening": spread_widening,
                "status": status,
                "ts": time.time()
            }
            self.recent_records.append(record)
            
            # Async-safe DB write
            if db_manager is not None:
                if db_executor is not None:
                    db_executor.submit(
                        db_manager.insert_execution_analytic,
                        trade_id=trade_id,
                        symbol=symbol,
                        expected_price=expected,
                        actual_price=actual,
                        slippage=slippage,
                        latency_ms=latency_ms,
                        spread_entry=spread_entry,
                        spread_widening=spread_widening,
                        status=status
                    )
                else:
                    try:
                        db_manager.insert_execution_analytic(
                            trade_id=trade_id,
                            symbol=symbol,
                            expected_price=expected,
                            actual_price=actual,
                            slippage=slippage,
                            latency_ms=latency_ms,
                            spread_entry=spread_entry,
                            spread_widening=spread_widening,
                            status=status
                        )
                    except Exception as e:
                        logger.error(f"[EXECUTION] Failed to persist analytics in database: {e}")
                    
            return record
        except Exception as e:
            logger.error(f"[EXECUTION] Failed to record execution analytics: {e}")
            return {}

    def get_summary_metrics(self) -> dict:
        """Returns aggregated metrics for dashboard consumption."""
        avg_slippage = 0.0
        avg_latency = 0.0
        success_rate = 0.0
        
        if self.filled_orders > 0:
            avg_slippage = self.total_slippage / self.filled_orders
            avg_latency = self.total_latency_ms / self.filled_orders
            
        if self.total_orders > 0:
            success_rate = (self.filled_orders / self.total_orders) * 100.0
            
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "fill_success_rate": success_rate,
            "average_slippage": avg_slippage,
            "worst_slippage": self.worst_slippage,
            "average_latency_ms": avg_latency,
            "worst_latency_ms": self.max_latency_ms
        }
