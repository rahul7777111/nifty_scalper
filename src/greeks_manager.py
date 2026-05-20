"""Advanced Greeks management and gamma scalping helpers.

This scaffold exposes a `GreeksManager` class that tracks portfolio-level
greek exposures and provides simple rebalancing suggestions. The goal is
to centralize greeks-related logic so the strategy can call optimized
routines without mixing calculation details into `strategy.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class GreeksExposure:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


class GreeksManager:
    def __init__(self) -> None:
        self._exposures: Dict[str, GreeksExposure] = {}

    def update_leg(self, key: str, delta: float, gamma: float, vega: float, theta: float) -> None:
        self._exposures[key] = GreeksExposure(delta=delta, gamma=gamma, vega=vega, theta=theta)

    def portfolio_exposure(self) -> GreeksExposure:
        p = GreeksExposure()
        for g in self._exposures.values():
            p.delta += g.delta
            p.gamma += g.gamma
            p.vega += g.vega
            p.theta += g.theta
        return p

    def should_gamma_scalp(self, gamma_threshold: float = 0.5) -> bool:
        """Return True when gamma exposure magnitude is larger than threshold."""
        p = self.portfolio_exposure()
        return abs(p.gamma) >= abs(gamma_threshold)

    def suggest_delta_rebalance(self, target_delta: float = 0.0) -> float:
        """Return the delta amount to trade to reach target delta (positive -> buy underlying)."""
        p = self.portfolio_exposure()
        return float(target_delta - p.delta)
