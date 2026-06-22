"""Regime Detection and Dynamic Ensemble Routing Engine.

Detects current NIFTY spot index regimes (Trending, Choppy, Volatility Shock, Macro News Event)
and dynamically routes weight distributions across Logistic Regression, RandomForest, and XGBoost.
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    CHOPPY = "CHOPPY"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    NEWS_EVENT = "NEWS_EVENT"

@dataclass
class RegimeWeights:
    w_lr: float
    w_rf: float
    w_xgb: float
    trading_enabled: bool

class RegimeRouter:
    """Classifies NIFTY market states and dynamically re-allocates ensemble weights."""
    
    def __init__(self, ema_period: int = 20):
        self.ema_period = ema_period
        self.iv_history: List[float] = []
        self.vix_history: List[float] = []
        self.atr_history: List[float] = []
        
    def classify_regime(
        self,
        adx: float,
        atr: float,
        bb_bandwidth: float,
        current_iv: float,
        current_vix: float,
        is_news_window: bool = False
    ) -> MarketRegime:
        """Determines the market state based on technical indicators and volatility metrics.
        
        Args:
            adx: Average Directional Index (trend strength).
            atr: Average True Range (normalized price volatility).
            bb_bandwidth: Bollinger Band Width (volatility contraction/expansion).
            current_iv: At-the-money implied volatility of front-month NIFTY options.
            current_vix: India VIX index value.
            is_news_window: Flag indicating if a macro event (RBI Policy, Budget) is active.
        """
        # 1. News Event Block (Highest priority safety veto)
        if is_news_window or current_vix > 35.0:
            return MarketRegime.NEWS_EVENT
            
        # Update rolling histories
        self.iv_history.append(current_iv)
        self.vix_history.append(current_vix)
        self.atr_history.append(atr)
        if len(self.iv_history) > self.ema_period:
            self.iv_history.pop(0)
            self.vix_history.pop(0)
            self.atr_history.pop(0)

        # Calculate baselines
        mean_iv = np.mean(self.iv_history) if self.iv_history else current_iv
        
        # 2. Volatility Shock Regime
        if current_iv > 1.35 * mean_iv or current_vix > 24.0:
            return MarketRegime.VOLATILITY_SHOCK
            
        # 3. Trending Regime
        if adx > 25.0 or (bb_bandwidth > 0.15 and adx > 20.0):
            return MarketRegime.TRENDING
            
        # 4. Choppy/Range-Bound Regime
        return MarketRegime.CHOPPY

    def get_routing_weights(self, regime: MarketRegime) -> RegimeWeights:
        """Translates market regimes into dynamic ensemble weight configurations.
        
        Optimized for 8GB RAM CPU execution:
        - Trending: Heavily weights XGBoost to capture fast non-linear spot momentum.
        - Choppy: Elevates Logistic Regression to prevent chasing noise and mean-revert.
        - Volatility Shock: Dominates with RandomForest to leverage low-variance bagging.
        - News-Event: Disables trading entirely to prevent slippage/extreme spread losses.
        """
        if regime == MarketRegime.NEWS_EVENT:
            return RegimeWeights(w_lr=0.0, w_rf=0.0, w_xgb=0.0, trading_enabled=False)
            
        elif regime == MarketRegime.VOLATILITY_SHOCK:
            # Defensive bagging dominates
            return RegimeWeights(w_lr=0.20, w_rf=0.55, w_xgb=0.25, trading_enabled=True)
            
        elif regime == MarketRegime.TRENDING:
            # Gradient boosting captures the trend
            return RegimeWeights(w_lr=0.10, w_rf=0.25, w_xgb=0.65, trading_enabled=True)
            
        elif regime == MarketRegime.CHOPPY:
            # Linear regularized baseline limits false breakouts
            return RegimeWeights(w_lr=0.60, w_rf=0.20, w_xgb=0.20, trading_enabled=True)
            
        return RegimeWeights(w_lr=0.40, w_rf=0.25, w_xgb=0.35, trading_enabled=True)
