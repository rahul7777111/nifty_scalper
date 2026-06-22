"""Realistic Options Market and Greeks Model.

Provides Black-Scholes calculators for Greeks (Delta, Gamma, Theta, Vega, Rho)
and simulates non-linear time decay, Gamma acceleration near expiration,
Implied Volatility (IV) crush events, and bid-ask spread widening under stress.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import numpy as np
import scipy.stats as stats


class OptionsGreeksModel:
    """Analytical options Greeks and premium stress simulation engine."""

    @staticmethod
    def black_scholes(
        spot: float,
        strike: float,
        time_to_expiry: float,  # in years (e.g. 7/365 for weekly)
        volatility: float,      # decimal (e.g. 0.16 for 16% IV)
        risk_free_rate: float = 0.06,  # 6% typical for India
        option_type: str = "call"
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the Black-Scholes option price and Greeks.

        Returns:
            price: Option premium
            greeks: Dictionary containing Delta, Gamma, Theta, Vega, Rho
        """
        # Clean inputs to prevent numerical division by zero
        S = max(1.0, float(spot))
        K = max(1.0, float(strike))
        T = max(1e-5, float(time_to_expiry))
        v = max(1e-4, float(volatility))
        r = float(risk_free_rate)
        opt_type = option_type.lower().strip()

        d1 = (math.log(S / K) + (r + 0.5 * v ** 2) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)

        n_d1 = stats.norm.cdf(d1)
        n_d2 = stats.norm.cdf(d2)
        n_minus_d1 = stats.norm.cdf(-d1)
        n_minus_d2 = stats.norm.cdf(-d2)
        pdf_d1 = stats.norm.pdf(d1)

        # Price
        if opt_type == "call":
            price = S * n_d1 - K * math.exp(-r * T) * n_d2
            delta = n_d1
            rho = K * T * math.exp(-r * T) * n_d2
        else:  # put
            price = K * math.exp(-r * T) * n_minus_d2 - S * n_minus_d1
            delta = n_d1 - 1.0
            rho = -K * T * math.exp(-r * T) * n_minus_d2

        # Shared Greeks
        gamma = pdf_d1 / (S * v * math.sqrt(T))
        vega = S * math.sqrt(T) * pdf_d1

        # Theta (annualized)
        term1 = -(S * pdf_d1 * v) / (2 * math.sqrt(T))
        if opt_type == "call":
            term2 = r * K * math.exp(-r * T) * n_d2
            theta_ann = term1 - term2
        else:
            term2 = r * K * math.exp(-r * T) * n_minus_d2
            theta_ann = term1 + term2

        # Daily Greeks conversion
        theta_daily = theta_ann / 365.0
        # Vega for a 1% absolute move in IV
        vega_1pct = vega * 0.01

        greeks = {
            "delta": delta,
            "gamma": gamma,
            "theta_annual": theta_ann,
            "theta_daily": theta_daily,
            "vega": vega,
            "vega_1pct": vega_1pct,
            "rho": rho
        }

        return max(0.01, price), greeks

    def simulate_path_greeks_decay(
        self,
        start_spot: float = 22000.0,
        strike: float = 22000.0,
        days_to_expiry: float = 7.0,
        start_iv: float = 0.16,
        daily_drift: float = 0.0,
        daily_vol: float = 0.01,
        option_type: str = "call",
        iv_crush_day: Optional[int] = None,
        iv_crush_pct: float = 0.30,
        spread_widening_factor: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Simulate day-by-day options Greeks decay and P&L path.

        Simulates realistic option behaviors:
            1. Non-linear Theta decay (faster decay as expiry approaches).
            2. Gamma acceleration near expiration (spiking sensitivity).
            3. IV Crush (sudden volatility contraction).
            4. Bid-Ask Spread Widening (simulates execution friction spikes).
        """
        N = int(math.ceil(days_to_expiry))
        
        spots = np.zeros(N + 1)
        times = np.zeros(N + 1)
        ivs = np.zeros(N + 1)
        prices = np.zeros(N + 1)
        bid_ask_spreads = np.zeros(N + 1)
        
        deltas = np.zeros(N + 1)
        gammas = np.zeros(N + 1)
        thetas = np.zeros(N + 1)
        vegas = np.zeros(N + 1)

        # Initial states
        curr_spot = start_spot
        curr_iv = start_iv
        
        for day in range(N + 1):
            t_rem = (days_to_expiry - day) / 365.0
            if t_rem < 0:
                t_rem = 0.0
                
            # Simulate Spot drift and random volatility walk
            if day > 0:
                # Geometric Brownian Motion step
                rand = np.random.normal(0, 1)
                curr_spot = curr_spot * math.exp(daily_drift + daily_vol * rand)
                
                # Check for IV Crush event
                if iv_crush_day is not None and day == iv_crush_day:
                    curr_iv = curr_iv * (1.0 - iv_crush_pct)
                else:
                    # Random small fluctuation in IV
                    curr_iv = max(0.05, curr_iv + np.random.normal(0, 0.005))

            # Compute Black-Scholes price & Greeks
            price, greeks = self.black_scholes(
                spot=curr_spot,
                strike=strike,
                time_to_expiry=t_rem,
                volatility=curr_iv,
                option_type=option_type
            )

            # Bid-Ask spread widening under volatility stress or near expiration
            # Base spread = 0.1% of price or min 1 rupee
            base_spread = max(1.0, price * 0.002)
            # Widening under high IV and low liquidity (very close to expiration)
            vol_multiplier = max(1.0, curr_iv / 0.15)
            time_stress = 2.0 if t_rem < (1.0 / 365.0) else 1.0
            spread = base_spread * vol_multiplier * time_stress * spread_widening_factor

            spots[day] = curr_spot
            times[day] = t_rem * 365.0
            ivs[day] = curr_iv
            prices[day] = price
            bid_ask_spreads[day] = spread
            
            deltas[day] = greeks["delta"]
            gammas[day] = greeks["gamma"]
            thetas[day] = greeks["theta_daily"]
            vegas[day] = greeks["vega_1pct"]

        # Calculate daily trade-level P&L including spread penalty
        option_pnl = np.diff(prices)
        spread_penalty = (bid_ask_spreads[:-1] + bid_ask_spreads[1:]) * 0.5
        adjusted_pnl = option_pnl - (spread_penalty * 0.1)  # assuming 10% spread slippage cost

        return {
            "spot": spots,
            "days_to_expiry": times,
            "iv": ivs,
            "premium": prices,
            "spread": bid_ask_spreads,
            "delta": deltas,
            "gamma": gammas,
            "theta": thetas,
            "vega": vegas,
            "daily_pnl": option_pnl,
            "adjusted_pnl": adjusted_pnl
        }
