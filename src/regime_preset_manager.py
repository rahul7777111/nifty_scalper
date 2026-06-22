import time
import math
import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)

class AdaptiveRegimeSwitch:
    """Adaptive Preset Regime-Aware Switch Manager.
    
    Dynamically computes sub-factor scores for Momentum, Volatility, Liquidity, and Health
    and automatically overrides StrategyConfig values to protect portfolio capital.
    """
    def __init__(self, bot):
        self.bot = bot
        self.current_regime = "chop"
        self.last_switch_ts = 0.0
        self.switch_cooldown = 180.0  # Limit preset oscillation to 3 minutes
        self.s_h_breach_count = 0  # Persist consecutive breaches of health safety limits
        
        self.scores = {
            "S_M": 0.0,
            "S_V": 0.0,
            "S_L": 0.0,
            "S_H": 0.0
        }

    def calculate_regime_scores(self, spot_ltp: float, atr: float, adx: float, 
                                 realized_vol: float, spreads: list, 
                                 ws_latency: float, event_lag: float) -> Dict[str, float]:
        """Computes standardized quant factor scores between 0.0 and 1.0."""
        try:
            # 1. Momentum Score S_M (normalized ADX and EMA slope tanh)
            s_m = min(1.0, adx / 50.0) if adx > 0 else 0.0
            
            # 2. Volatility Score S_V (relative ATR range expansion)
            atr_pct = (atr / spot_ltp) * 100.0 if spot_ltp > 0 else 0.0
            s_v = min(1.0, atr_pct / 2.0)
            
            # 3. Liquidity Score S_L (Rolling Z-score relative liquidity model)
            if spreads and len(spreads) >= 5:
                current_spread = float(spreads[-1])
                mean_spread = float(np.mean(spreads))
                std_spread = float(np.std(spreads))
                if std_spread > 0:
                    z_score = (current_spread - mean_spread) / std_spread
                    # Map Z-score to 0.0 - 1.0. A Z-score of +3.0 indicates extreme spread expansion.
                    s_l = min(1.0, max(0.0, z_score / 3.0))
                else:
                    s_l = 0.0
            else:
                avg_spread = np.mean(spreads) if (spreads and len(spreads) > 0) else 0.5
                # For small spread history, map conservatively to prevent false low-liquidity triggers.
                # Threshold for low liquidity is 0.75. Cap small history S_L at 0.35.
                s_l = min(0.35, avg_spread / 3.0)
            
            # 4. System Health Score S_H (Strict Logical OR Failsafe with Temporal Smoothing)
            # If EITHER network latency OR CPU GIL event lag breaches safety limits, we note an instant breach.
            ws_fail = ws_latency > 500.0
            cpu_fail = event_lag > 150.0
            s_h_instant = 1.0 if (ws_fail or cpu_fail) else 0.0
            
            if s_h_instant >= 1.0:
                self.s_h_breach_count += 1
            else:
                self.s_h_breach_count = 0
                
            # Event score triggers Low Liquidity Mode only if it persists for 3 consecutive ticks
            s_h = 1.0 if self.s_h_breach_count >= 3 else 0.0
            
            self.scores = {
                "S_M": float(s_m),
                "S_V": float(s_v),
                "S_L": float(s_l),
                "S_H": float(s_h)
            }
        except Exception as e:
            logger.error(f"[REGIME SWITCH] Error calculating scores: {e}")
            
        return self.scores

    def evaluate_regime_switch(self, spot_ltp: float, atr: float, adx: float, 
                                realized_vol: float, spreads: list, 
                                ws_latency: float, event_lag: float, 
                                is_expiry_day: bool) -> str:
        """Evaluates whether to switch system presets based on current factor scores."""
        now = time.time()
        
        # Calculate latest scores
        scores = self.calculate_regime_scores(spot_ltp, atr, adx, realized_vol, spreads, ws_latency, event_lag)
        
        # Priority 1: Expiry-Day Safety Mode
        if is_expiry_day:
            if self.current_regime != "expiry":
                logger.warning("[REGIME SWITCH] Expiry Day detected. Engaging Expiry Safety Mode.")
                self.current_regime = "expiry"
                self.last_switch_ts = now
                self._apply_preset_settings("expiry")
            return "expiry"
            
        # Priority 2: System Health Degradation / Low Liquidity Failsafe
        if scores["S_H"] >= 0.70 or scores["S_L"] >= 0.75:
            if self.current_regime != "low_liquidity":
                logger.warning(f"[REGIME SWITCH] Safety threshold breached! S_H={scores['S_H']:.2f}, S_L={scores['S_L']:.2f}. Forced Low Liquidity Mode.")
                self.current_regime = "low_liquidity"
                self.last_switch_ts = now
                self._apply_preset_settings("low_liquidity")
            return "low_liquidity"
            
        # Check transition cooldown limit
        if now - self.last_switch_ts < self.switch_cooldown:
            return self.current_regime

        # Priority 3: Volatility Panic Regime
        if scores["S_V"] >= 0.65:
            new_regime = "high_volatility"
        # Priority 4: Trending Momentum
        elif scores["S_M"] >= 0.55 and scores["S_V"] <= 0.50:
            new_regime = "trend"
        # Priority 5: Standard Chop Reversion
        else:
            new_regime = "chop"

        if new_regime != self.current_regime:
            logger.info(f"[REGIME SWITCH] Transitioning from '{self.current_regime}' to '{new_regime}' preset. Scores: S_M={scores['S_M']:.2f}, S_V={scores['S_V']:.2f}")
            self.current_regime = new_regime
            self.last_switch_ts = now
            self._apply_preset_settings(new_regime)
            
        return self.current_regime

    def _apply_preset_settings(self, regime: str) -> None:
        """Overrides StrategyConfig parameters in real-time."""
        try:
            cfg = self.bot.cfg
            if regime == "trend":
                cfg.timeframe = "2m"
                cfg.cooldown_sec = 30.0
                cfg.cooldown_after_stopout_sec = 90.0
                cfg.risk_scale_max_qty = 450
                cfg.enable_dynamic_pyramiding = True
                cfg.dir_premium_trail_pct = 0.06
                cfg.dir_partial_target_mult = 1.5
                cfg.risk_scale_recovery_wins = 1
                cfg.strategy_router_mode = "aggressive"
                cfg.mtf_hard_filter = True
                cfg.theta_stop_widen_max_mult = 1.5
                cfg.delta_hedge_vol_low_tol_factor = 0.7
                cfg.delta_hedge_vol_high_tol_factor = 1.4
                
            elif regime == "chop":
                cfg.timeframe = "3m"
                cfg.cooldown_sec = 60.0
                cfg.cooldown_after_stopout_sec = 180.0
                cfg.risk_scale_max_qty = 200
                cfg.enable_dynamic_pyramiding = False
                cfg.dir_premium_trail_pct = 0.12
                cfg.dir_partial_target_mult = 0.8
                cfg.risk_scale_recovery_wins = 2
                cfg.strategy_router_mode = "balanced"
                cfg.mtf_hard_filter = False
                cfg.theta_stop_widen_max_mult = 2.0
                cfg.delta_hedge_vol_low_tol_factor = 0.9
                cfg.delta_hedge_vol_high_tol_factor = 1.2
                
            elif regime == "expiry":
                cfg.timeframe = "1m"
                cfg.cooldown_sec = 90.0
                cfg.cooldown_after_stopout_sec = 300.0
                cfg.risk_scale_max_qty = 100
                cfg.enable_dynamic_pyramiding = False
                cfg.dir_premium_trail_pct = 0.15
                cfg.dir_partial_target_mult = 1.0
                cfg.risk_scale_recovery_wins = 3
                cfg.strategy_router_mode = "conservative"
                cfg.mtf_hard_filter = True
                cfg.theta_stop_widen_max_mult = 1.0
                cfg.delta_hedge_vol_low_tol_factor = 0.95
                cfg.delta_hedge_vol_high_tol_factor = 1.05
                
            elif regime == "high_volatility":
                cfg.timeframe = "3m"
                cfg.cooldown_sec = 75.0
                cfg.cooldown_after_stopout_sec = 240.0
                cfg.risk_scale_max_qty = 150
                cfg.enable_dynamic_pyramiding = False
                cfg.dir_premium_trail_pct = 0.18
                cfg.dir_partial_target_mult = 1.8
                cfg.risk_scale_recovery_wins = 3
                cfg.strategy_router_mode = "conservative"
                cfg.mtf_hard_filter = True
                cfg.theta_stop_widen_max_mult = 1.2
                cfg.delta_hedge_vol_low_tol_factor = 0.8
                cfg.delta_hedge_vol_high_tol_factor = 1.3
                
            elif regime == "low_liquidity":
                cfg.timeframe = "5m"
                cfg.cooldown_sec = 120.0
                cfg.cooldown_after_stopout_sec = 360.0
                cfg.risk_scale_max_qty = 50
                cfg.enable_dynamic_pyramiding = False
                cfg.dir_premium_trail_pct = 0.10
                cfg.dir_partial_target_mult = 1.0
                cfg.risk_scale_recovery_wins = 3
                cfg.strategy_router_mode = "conservative"
                cfg.mtf_hard_filter = False
                cfg.theta_stop_widen_max_mult = 1.0
                cfg.delta_hedge_vol_low_tol_factor = 0.85
                cfg.delta_hedge_vol_high_tol_factor = 1.2
                
        except Exception as e:
            logger.error(f"[REGIME SWITCH] Error applying presets settings: {e}")
