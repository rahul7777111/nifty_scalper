import logging
import math
from collections import deque, defaultdict
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class StrategyDecayDetector:
    """Strategy Decay Detection Engine.
    
    Tracks rolling performance metrics per strategy family.
    Implements degradation alerts and auto-cooldown limits to protect capital.
    """

    def __init__(self, window_size: int = 15, min_rolling_sharpe: float = 0.0, max_drawdown_pct: float = 0.15):
        self.window_size = window_size
        self.min_rolling_sharpe = min_rolling_sharpe
        self.max_drawdown_pct = max_drawdown_pct
        
        # rolling deque of PnL entries: strategy_name -> deque([float])
        self.pnl_history = defaultdict(lambda: deque(maxlen=window_size))
        # degraded strategies list: strategy_name -> cooldown_expiry_ts (0 means disabled)
        self.cooldown_strategies = {}

    def record_trade_outcome(self, strategy_name: str, pnl: float) -> dict:
        """Records a completed trade outcome and evaluates strategy decay."""
        try:
            name = str(strategy_name).strip().lower()
            self.pnl_history[name].append(pnl)
            
            # Re-evaluate rolling metrics
            metrics = self.calculate_metrics(name)
            
            # Check degradation threshold
            is_degraded = False
            reason = ""
            
            if len(self.pnl_history[name]) >= 5: # Warmup before policing
                # 1. Sharpe check
                if metrics["sharpe"] < self.min_rolling_sharpe:
                    is_degraded = True
                    reason = f"Rolling Sharpe ({metrics['sharpe']:.2f}) < threshold ({self.min_rolling_sharpe})"
                # 2. Drawdown check
                elif metrics["drawdown_pct"] > self.max_drawdown_pct:
                    is_degraded = True
                    reason = f"Drawdown ({metrics['drawdown_pct']*100:.1f}%) > threshold ({self.max_drawdown_pct*100:.1f}%)"
                    
            if is_degraded:
                logger.warning(f"[STRATEGY DECAY] Strategy '{strategy_name}' degraded: {reason}. Placing in cooldown.")
                # Cooldown for 300 seconds
                import time
                self.cooldown_strategies[name] = time.time() + 300.0
            
            return metrics
        except Exception as e:
            logger.error(f"[STRATEGY DECAY] Error recording trade outcome: {e}")
            return {}

    def calculate_metrics(self, strategy_name: str) -> dict:
        """Computes Sharpe, expectancy, drawdown, winrate, and profit factor over rolling window."""
        name = str(strategy_name).strip().lower()
        history = list(self.pnl_history[name])
        
        if not history:
            return {"sharpe": 0.0, "expectancy": 0.0, "drawdown_pct": 0.0, "winrate": 0.0, "profit_factor": 1.0}
            
        wins = [x for x in history if x > 0]
        losses = [x for x in history if x < 0]
        
        # 1. Win Rate
        winrate = len(wins) / len(history) if history else 0.0
        
        # 2. Profit Factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 1.0)
        
        # 3. Expectancy
        expectancy = sum(history) / len(history)
        
        # 4. Sharpe Ratio (rolling return / rolling std dev)
        # Daily scaling factor approximated to rolling trades scale
        mean_pnl = sum(history) / len(history)
        var_pnl = sum((x - mean_pnl)**2 for x in history) / len(history) if len(history) > 1 else 0.0
        std_pnl = math.sqrt(var_pnl)
        
        sharpe = (mean_pnl / std_pnl) * math.sqrt(252) if std_pnl > 1e-9 else 0.0
        
        # 5. Drawdown
        cum_pnl = []
        curr = 0.0
        for x in history:
            curr += x
            cum_pnl.append(curr)
        
        peak = -1e9
        max_dd = 0.0
        for val in cum_pnl:
            if val > peak:
                peak = val
            if peak > 0:
                dd = (peak - val) / peak if peak != 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                    
        return {
            "sharpe": sharpe,
            "expectancy": expectancy,
            "drawdown_pct": max_dd,
            "winrate": winrate,
            "profit_factor": profit_factor,
            "trades_count": len(history)
        }

    def is_strategy_blocked(self, strategy_name: str) -> Tuple[bool, float]:
        """Checks if a strategy is blocked by cooldown limits.
        
        Returns:
            Tuple of [is_blocked, remaining_cooldown_seconds]
        """
        import time
        name = str(strategy_name).strip().lower()
        expiry = self.cooldown_strategies.get(name, 0.0)
        if expiry > time.time():
            return True, expiry - time.time()
        # Clean expiry if outdated
        if name in self.cooldown_strategies and expiry <= time.time():
            logger.info(f"[STRATEGY DECAY] Strategy '{strategy_name}' cooldown expired. Re-enabling.")
            del self.cooldown_strategies[name]
        return False, 0.0
