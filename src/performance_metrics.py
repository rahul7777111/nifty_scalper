import os
import json
import time
import math
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Quantitative performance tracking subsystem.
    
    Tracks closed trade histories, execution prices (slippage), profits, losses, 
    and computes institutional strategy metrics (Win Rate, Profit Factor, Expectancy,
    Sharpe Ratio, Max Drawdown). Persists results to a workspace-local JSON file.
    """
    
    def __init__(self, filepath: str = "performance_metrics.json"):
        self.filepath = filepath
        self.trades: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "slippage_points": 0.0,
            "last_updated": 0.0
        }
        self.load_metrics()

    def load_metrics(self) -> None:
        """Load persisted metrics from JSON if available."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.trades = data.get("trades", [])
                    self.metrics = data.get("metrics", self.metrics)
                logger.info(f"[PERFORMANCE] Loaded {len(self.trades)} trades from {self.filepath}")
            except Exception as e:
                logger.error(f"[PERFORMANCE] Failed to load metrics: {e}")

    def save_metrics(self) -> None:
        """Persist trades and metrics to JSON."""
        try:
            # Absolute path inside NiftyScalper workspace
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "metrics": self.metrics,
                    "trades": self.trades
                }, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[PERFORMANCE] Failed to save metrics: {e}")

    def record_trade(self, *, symbol: str, side: str, qty: int, entry_price: float, exit_price: float, 
                     realized_pnl: float, theoretical_entry: Optional[float] = None, 
                     theoretical_exit: Optional[float] = None) -> None:
        """Records a closed trade, updates metrics, and triggers persistence."""
        try:
            # Slippage calculation
            slip_entry = abs(entry_price - theoretical_entry) if theoretical_entry is not None else 0.0
            slip_exit = abs(exit_price - theoretical_exit) if theoretical_exit is not None else 0.0
            total_slippage = slip_entry + slip_exit

            trade_record = {
                "ts": time.time(),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl": float(realized_pnl),
                "slippage": float(total_slippage)
            }
            self.trades.append(trade_record)
            self._update_all_metrics()
            self.save_metrics()
            logger.info(f"[PERFORMANCE] Trade recorded for {symbol}. PnL={realized_pnl:.2f}, Slippage={total_slippage:.2f}")
        except Exception as e:
            logger.error(f"[PERFORMANCE] Error recording trade: {e}")

    def _update_all_metrics(self) -> None:
        """Recalculates Win Rate, Profit Factor, Sharpe Ratio, and Drawdown based on trade history."""
        if not self.trades:
            return

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
        
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        total_pnl = sum(t["pnl"] for t in self.trades)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0.0
        total_slippage = sum(t["slippage"] for t in self.trades)

        # Sharpe Ratio Calculation (hourly/period approximation)
        pnls = [t["pnl"] for t in self.trades]
        if len(pnls) > 1:
            mean_pnl = float(np_mean(pnls))
            std_pnl = float(np_std(pnls))
            # Annualization factor assuming ~1000 trades/year; handle std close to zero
            sharpe = (mean_pnl / std_pnl) * math.sqrt(252.0) if std_pnl > 1e-6 else 0.0
        else:
            sharpe = 0.0

        # Maximum Drawdown Calculation (peak-to-trough on equity curve)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            equity += t["pnl"]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        self.metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "slippage_points": float(total_slippage),
            "last_updated": time.time()
        }

def np_mean(arr: List[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0

def np_std(arr: List[float]) -> float:
    if not arr or len(arr) <= 1:
        return 0.0
    m = np_mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance)
