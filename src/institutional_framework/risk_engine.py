"""Institutional-Grade Options Capital Protection and Risk Engine.

Implements Kelly position sizing, volatility targeting, soft/hard drawdown controls,
exposure limits, portfolio heat parameters, and emergency flattening.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class RiskLimits:
    max_open_positions: int = 5
    max_portfolio_heat: float = 0.20        # Maximum 20% capital exposed across all positions
    soft_drawdown_limit: float = -0.015     # -1.5% daily drawdown (restrict entries)
    hard_drawdown_limit: float = -0.030     # -3.0% daily drawdown (trigger flattening)
    max_correlated_exposure: float = 0.12   # Maximum 12% in highly correlated option legs

@dataclass
class PositionSizeReport:
    recommended_qty: int
    leverage_mult: float
    kelly_fraction: float
    is_allowed: bool
    rejection_reason: str

class OptionsRiskEngine:
    """Manages risk limits, capital allocation, and triggers safety kill-switches."""
    
    def __init__(
        self,
        starting_capital: float = 100000.0,
        limits: Optional[RiskLimits] = None,
        kelly_fraction_limit: float = 0.15      # Scaled down to 15% (fractional Kelly)
    ):
        self.starting_capital = starting_capital
        self.limits = limits or RiskLimits()
        self.kelly_fraction_limit = kelly_fraction_limit
        
        self.daily_pnl = 0.0
        self.active_positions: Dict[str, Dict] = {}  # leg_id -> position dict
        
    def update_daily_pnl(self, realized_pnl: float, unrealized_pnl: float):
        """Updates the running daily profit and loss."""
        self.daily_pnl = realized_pnl + unrealized_pnl

    def check_drawdown_state(self) -> Tuple[bool, bool]:
        """Evaluates drawdown conditions.
        
        Returns:
            Tuple: (soft_breached, hard_breached)
        """
        pct_drawdown = self.daily_pnl / self.starting_capital
        
        soft_breached = pct_drawdown <= self.limits.soft_drawdown_limit
        hard_breached = pct_drawdown <= self.limits.hard_drawdown_limit
        
        return soft_breached, hard_breached

    def calculate_kelly_size(
        self,
        prob: float,                  # Platt calibrated win probability
        risk_reward: float,           # Reward-to-Risk ratio (Profit Target / Stop Loss)
        spot_volatility: float,       # Current realized volatility
        target_volatility: float,     # Strategy target volatility (e.g. 15% annualized)
        option_premium: float,
        lot_size: int = 75            # NIFTY lot size is 75 (or current NSE standard)
    ) -> PositionSizeReport:
        """Calculates Fractional Kelly position sizing with ex-ante volatility adjustments."""
        # Check drawdown limits
        soft_breached, hard_breached = self.check_drawdown_state()
        if hard_breached:
            return PositionSizeReport(0, 0.0, 0.0, False, "HARD DRAWDOWN LIMIT BREACHED - KILL SWITCH ACTIVE")
            
        # Limit positions count
        if len(self.active_positions) >= self.limits.max_open_positions:
            return PositionSizeReport(0, 0.0, 0.0, False, "MAXIMUM OPEN POSITIONS BREACHED")
            
        # 1. Base Kelly Sizing: f* = (p * R - (1 - p)) / R
        p = prob
        q = 1.0 - p
        b = risk_reward
        
        kelly_f = (p * b - q) / b if b > 0 else 0.0
        
        if kelly_f <= 0.0:
            return PositionSizeReport(0, 0.0, 0.0, False, "NEGATIVE STRATEGY EXPECTANCY")
            
        # Apply fractional Kelly scaling factor
        scaled_kelly = kelly_f * self.kelly_fraction_limit
        
        # 2. Ex-Ante Volatility Targeting: Scale position based on current spot vs target volatility
        vol_factor = target_volatility / spot_volatility if spot_volatility > 0 else 1.0
        leverage_mult = min(1.0, vol_factor)
        
        # Soft drawdown restriction: Slashes leverage by 50%
        if soft_breached:
            leverage_mult *= 0.50
            
        final_capital_fraction = scaled_kelly * leverage_mult
        
        # Ensure portfolio heat limit is not exceeded
        current_heat = sum(pos["premium"] * pos["qty"] for pos in self.active_positions.values()) / self.starting_capital
        if current_heat + final_capital_fraction > self.limits.max_portfolio_heat:
            remaining_heat = max(0.0, self.limits.max_portfolio_heat - current_heat)
            final_capital_fraction = min(final_capital_fraction, remaining_heat)
            
        # 3. Translate capital fraction into physical options quantity
        allocated_capital = self.starting_capital * final_capital_fraction
        cost_per_lot = option_premium * lot_size
        
        if cost_per_lot <= 0:
            return PositionSizeReport(0, 0.0, 0.0, False, "INVALID OPTION PREMIUM COST")
            
        recommended_lots = int(allocated_capital / cost_per_lot)
        recommended_qty = recommended_lots * lot_size
        
        if recommended_qty == 0:
            return PositionSizeReport(0, leverage_mult, scaled_kelly, False, "INSUFFICIENT CAPITAL FOR MINIMUM LOT SIZE")
            
        return PositionSizeReport(
            recommended_qty=recommended_qty,
            leverage_mult=round(leverage_mult, 4),
            kelly_fraction=round(scaled_kelly, 4),
            is_allowed=True,
            rejection_reason=""
        )

    def trigger_emergency_flattening(self) -> List[str]:
        """Triggers emergency flattening. Returns a list of active position IDs to close."""
        to_close = list(self.active_positions.keys())
        self.active_positions.clear()
        return to_close
