"""Low-Latency Option Execution Simulator for Indian Derivatives Markets.

Models realistic execution friction on the National Stock Exchange of India (NSE),
including broker commissions, GST, SEBI turnover fees, STT, and latency queue delays.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TransactionCosts:
    brokerage: float
    stt: float
    exchange_txn_charge: float
    sebi_turnover_fee: float
    gst: float
    stamp_duty: float
    total_friction: float

@dataclass
class FillReport:
    fill_price: float
    filled_qty: int
    slippage_ticks: float
    latency_injected_sec: float
    costs: TransactionCosts

class IndianOptionsExecutionEngine:
    """Simulates realistic options execution with NSE transaction fees and execution delays."""
    
    def __init__(
        self,
        base_brokerage: float = 20.0,         # Flat ₹20 per executed order (e.g., Zerodha)
        tick_size: float = 0.05,
        latency_base_sec: float = 0.08,        # Base API round-trip network time (80ms)
        clearing_charge: float = 0.0
    ):
        self.base_brokerage = base_brokerage
        self.tick_size = tick_size
        self.latency_base_sec = latency_base_sec
        self.clearing_charge = clearing_charge

    def calculate_nse_friction(
        self,
        premium: float,
        qty: int,
        strike: float,
        is_sell: bool,
        is_exercised: bool = False
    ) -> TransactionCosts:
        """Calculates precise transaction friction charges based on NSE fee schedules."""
        turnover = premium * qty
        
        # 1. Brokerage: flat fee
        brokerage = self.base_brokerage
        
        # 2. Securities Transaction Tax (STT)
        # STT is only applicable on SELL transactions for option premiums (0.0625%)
        # If exercised, STT is 0.125% of the entire contract value (strike * qty)
        stt = 0.0
        if is_sell:
            if is_exercised:
                stt = 0.00125 * strike * qty
            else:
                stt = 0.000625 * turnover
                
        # 3. Exchange Transaction Charges: NSE Option Premium charge is 0.0505% (as of latest NSE rates)
        exchange_txn_charge = 0.000505 * turnover
        
        # 4. SEBI Turnover Fee: ₹10 per crore (0.0001%)
        sebi_turnover_fee = 0.0000001 * turnover
        
        # 5. GST (18% on Brokerage + Exchange Transaction + Clearing Charges)
        gst = 0.18 * (brokerage + exchange_txn_charge + self.clearing_charge)
        
        # 6. Stamp Duty: 0.003% on BUY transactions only (options)
        stamp_duty = 0.0
        if not is_sell:
            stamp_duty = 0.00003 * turnover
            
        total_friction = brokerage + stt + exchange_txn_charge + sebi_turnover_fee + gst + stamp_duty
        
        return TransactionCosts(
            brokerage=round(brokerage, 2),
            stt=round(stt, 2),
            exchange_txn_charge=round(exchange_txn_charge, 2),
            sebi_turnover_fee=round(sebi_turnover_fee, 4),
            gst=round(gst, 2),
            stamp_duty=round(stamp_duty, 2),
            total_friction=round(total_friction, 2)
        )

    def simulate_fill(
        self,
        target_price: float,         # Decision price (mid or bid/ask at arrival)
        qty: int,
        bid: float,
        ask: float,
        spot_velocity: float,        # Current spot price change rate (momentum)
        current_iv: float,
        is_buy: bool
    ) -> FillReport:
        """Simulates low-latency price fills with queue delay and slippage model."""
        # 1. Inject random network and processing latency
        rng = np.random.default_rng()
        latency_injected = self.latency_base_sec + rng.uniform(0.01, 0.12)  # Adds jitter (10ms to 120ms)
        
        # 2. Simulate price movement during latency window
        price_drift = spot_velocity * latency_injected
        
        # 3. Model spread widening in volatile/illiquid states
        spread = abs(ask - bid)
        volatility_multiplier = 1.0 + (current_iv - 0.15) * 4.0 if current_iv > 0.15 else 1.0
        widened_spread = spread * max(1.0, volatility_multiplier)
        
        # 4. Fill Price simulation (impacted by queue delay and slippage)
        slippage_ticks = 0.0
        if is_buy:
            execution_limit = ask + price_drift
            # Add tick slippage based on volatility shock
            slippage_ticks = round(rng.lognormal(0.4, 0.5) * volatility_multiplier)
            fill_price = execution_limit + (slippage_ticks * self.tick_size)
        else:
            execution_limit = bid + price_drift
            slippage_ticks = round(rng.lognormal(0.4, 0.5) * volatility_multiplier)
            fill_price = execution_limit - (slippage_ticks * self.tick_size)
            
        # Bound fill price to valid options space (> 0.05)
        fill_price = max(0.05, round(fill_price / self.tick_size) * self.tick_size)
        
        # Calculate NSE transaction friction
        costs = self.calculate_nse_friction(
            premium=fill_price,
            qty=qty,
            strike=target_price,
            is_sell=not is_buy
        )
        
        # Simulate partial fill in fast market states
        filled_qty = qty
        if current_iv > 0.35 and rng.random() > 0.92:
            # 8% probability of partial fill during high IV shocks
            filled_qty = max(1, int(qty * rng.uniform(0.3, 0.8)))
            
        return FillReport(
            fill_price=fill_price,
            filled_qty=filled_qty,
            slippage_ticks=float(slippage_ticks),
            latency_injected_sec=latency_injected,
            costs=costs
        )
