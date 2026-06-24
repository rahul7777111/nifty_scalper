"""Enhanced backtest harness with multi-leg strategies, realistic execution costs, and performance metrics.

This module provides comprehensive backtesting capabilities including:
- Multi-leg options strategies (spreads, straddles, strangles, etc.)
- Realistic transaction cost modeling (STT, exchange charges, GST, brokerage)
- Performance metrics (Sharpe, Sortino, max drawdown, Calmar ratio)
- Parameter optimization support
- Walk-forward analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
import math
from datetime import datetime

# Handle both package and non-package imports
try:
    from .exit_optimizer import ExitOptimizer, ExitDecision  # Integrated exit optimization
except ImportError:
    from exit_optimizer import ExitOptimizer, ExitDecision  # Integrated exit optimization


@dataclass
class TransactionCosts:
    """Transaction cost structure for Indian markets."""
    brokerage_per_order: float = 20.0  # Flat fee per order
    stt_percentage: float = 0.0125  # Securities Transaction Tax (0.0125% for options)
    exchange_charges_percentage: float = 0.0025  # NSE/BSE charges
    gst_percentage: float = 18.0  # GST on brokerage + exchange charges
    stamp_duty_percentage: float = 0.0003  # Stamp duty
    slippage_bps: float = 5.0  # Slippage in basis points
    
    def calculate_costs(self, buy_value: float, sell_value: float, 
                        num_orders: int = 2) -> Dict[str, float]:
        """Calculate all transaction costs for a round-trip trade."""
        # Brokerage (capped at 20 per order for discount brokers)
        brokerage = min(self.brokerage_per_order * num_orders, 
                       (buy_value + sell_value) * 0.0003)  # 0.03% max
        
        # STT on sell side only for options
        stt = sell_value * self.stt_percentage / 100.0
        
        # Exchange charges on both buy and sell
        exchange_charges = (buy_value + sell_value) * self.exchange_charges_percentage / 100.0
        
        # GST on brokerage + exchange charges
        gst = (brokerage + exchange_charges) * self.gst_percentage / 100.0
        
        # Stamp duty on buy side
        stamp_duty = buy_value * self.stamp_duty_percentage / 100.0
        
        total_costs = brokerage + stt + exchange_charges + gst + stamp_duty
        
        return {
            'brokerage': brokerage,
            'stt': stt,
            'exchange_charges': exchange_charges,
            'gst': gst,
            'stamp_duty': stamp_duty,
            'total': total_costs,
        }


@dataclass
class Leg:
    """Represents one leg of a multi-leg strategy."""
    type: str  # 'call' or 'put'
    side: str  # 'buy' or 'sell'
    strike: float
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 1
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    @property
    def is_long(self) -> bool:
        return self.side == 'buy'
    
    def calculate_pnl(self) -> float:
        """Calculate P&L for this leg."""
        multiplier = 1 if self.is_long else -1
        return (self.exit_price - self.entry_price) * self.quantity * multiplier * 75  # Lot size = 75


@dataclass
class MultiLegTrade:
    """Represents a multi-leg options trade."""
    trade_id: str
    strategy_type: str  # 'bull_call_spread', 'iron_condor', etc.
    legs: List[Leg]
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    underlying_entry: float = 0.0
    underlying_exit: float = 0.0
    underlying_entry_iv: float = 0.0
    net_pnl: float = 0.0
    transaction_costs: float = 0.0
    reason_for_exit: str = ""
    max_profit_seen: float = 0.0
    max_loss_seen: float = 0.0
    
    def calculate_pnl(self, costs: Optional[TransactionCosts] = None) -> float:
        """Calculate net P&L including transaction costs."""
        gross_pnl = sum(leg.calculate_pnl() for leg in self.legs)
        
        if costs:
            # Calculate total buy and sell values
            buy_value = sum(leg.entry_price * leg.quantity * 75 
                          for leg in self.legs if leg.is_long)
            sell_value = sum(leg.exit_price * leg.quantity * 75 
                           for leg in self.legs if not leg.is_long)
            
            cost_breakdown = costs.calculate_costs(buy_value, sell_value, 
                                                   num_orders=len(self.legs) * 2)
            self.transaction_costs = cost_breakdown['total']
        else:
            self.transaction_costs = 0.0
        
        self.net_pnl = gross_pnl - self.transaction_costs
        return self.net_pnl


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    total_pnl: float
    gross_pnl: float
    total_costs: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    trade_list: List[MultiLegTrade]
    equity_curve: List[float]
    returns: List[float]
    
    @property
    def expectancy(self) -> float:
        """Average expected return per trade."""
        if self.num_trades == 0:
            return 0.0
        return self.total_pnl / self.num_trades


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.05) -> float:
    """Calculate Sharpe ratio from returns series."""
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    returns_arr = np.array(returns)
    excess_returns = returns_arr - risk_free_rate / 252  # Daily risk-free rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return float(sharpe)


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.05) -> float:
    """Calculate Sortino ratio (only penalizes downside deviation)."""
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    returns_arr = np.array(returns)
    excess_returns = returns_arr - risk_free_rate / 252
    
    # Only consider negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    return float(sortino)


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
    """Calculate maximum drawdown in absolute and percentage terms."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    
    import numpy as np
    equity = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity)
    
    # Calculate drawdown
    drawdown = equity - running_max
    drawdown_pct = drawdown / running_max * 100.0
    
    max_dd = float(np.min(drawdown))
    max_dd_pct = float(np.min(drawdown_pct))
    
    return max_dd, max_dd_pct


def simulate_multi_leg(
    strategy_type: str,
    entry_signals: List[int],
    price_data: List[Dict[str, Any]],
    position_size: int = 1,
    *,
    costs: Optional[TransactionCosts] = None,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    """Simulate multi-leg options strategy backtest."""
    if not entry_signals or not price_data or len(entry_signals) != len(price_data):
        return _empty_result()
    
    if costs is None:
        costs = TransactionCosts()
    
    trades: List[MultiLegTrade] = []
    equity_curve = [0.0]
    returns = []
    in_trade = False
    current_trade: Optional[MultiLegTrade] = None
    
    slippage_multiplier = 1 + slippage_bps / 10000.0
    
    for idx, (signal, data) in enumerate(zip(entry_signals, price_data)):
        timestamp = data.get('timestamp', datetime.utcnow())
        
        # Entry logic
        if signal == 1 and not in_trade:
            current_trade = _create_multi_leg_trade(
                strategy_type, data, position_size, timestamp
            )
            
            if current_trade:
                for leg in current_trade.legs:
                    if leg.is_long:
                        leg.entry_price *= slippage_multiplier
                    else:
                        leg.entry_price /= slippage_multiplier
                
                in_trade = True
        
        # Exit logic (simplified - exit on next bar or stop loss)
        elif in_trade and (signal == -1 or idx == len(entry_signals) - 1):
            if current_trade:
                for leg in current_trade.legs:
                    exit_price = _get_exit_price(leg, data)
                    if exit_price:
                        if leg.is_long:
                            leg.exit_price = exit_price / slippage_multiplier
                        else:
                            leg.exit_price = exit_price * slippage_multiplier
                    else:
                        leg.exit_price = leg.entry_price
                
                current_trade.exit_time = timestamp
                current_trade.calculate_pnl(costs)
                trades.append(current_trade)
                
                equity_curve.append(equity_curve[-1] + current_trade.net_pnl)
                if len(equity_curve) > 1:
                    ret = (equity_curve[-1] - equity_curve[-2]) / max(abs(equity_curve[-2]), 1.0)
                    returns.append(ret)
                
                in_trade = False
                current_trade = None
    
    return _compile_results(trades, equity_curve, returns)


def _create_multi_leg_trade(strategy_type: str, data: Dict[str, Any], 
                            position_size: int, timestamp: datetime) -> Optional[MultiLegTrade]:
    """Create a multi-leg trade based on strategy type."""
    underlying = data.get('underlying_price', 0.0)
    underlying_entry_iv = data.get('iv', 0.0)
    
    if strategy_type == 'bull_call_spread':
        lower_strike = _find_strike(underlying, data.get('calls', []), 'nearest')
        higher_strike = _find_strike(underlying * 1.02, data.get('calls', []), 'nearest')
        
        if not lower_strike or not higher_strike:
            return None
        
        long_call = Leg(type='call', side='buy', strike=lower_strike['strike'],
                       entry_price=lower_strike['price'], quantity=position_size)
        short_call = Leg(type='call', side='sell', strike=higher_strike['strike'],
                        entry_price=higher_strike['price'], quantity=position_size)
        
        return MultiLegTrade(
            trade_id=f"trade_{timestamp.isoformat()}",
            strategy_type=strategy_type,
            legs=[long_call, short_call],
            entry_time=timestamp,
            underlying_entry=underlying,
            underlying_entry_iv=underlying_entry_iv,
        )
    
    elif strategy_type == 'iron_condor':
        otm_call = _find_strike(underlying * 1.02, data.get('calls', []), 'nearest')
        higher_call = _find_strike(underlying * 1.04, data.get('calls', []), 'nearest')
        otm_put = _find_strike(underlying * 0.98, data.get('puts', []), 'nearest')
        lower_put = _find_strike(underlying * 0.96, data.get('puts', []), 'nearest')
        
        if not all([otm_call, higher_call, otm_put, lower_put]):
            return None
        
        legs = [
            Leg(type='call', side='sell', strike=otm_call['strike'],
                entry_price=otm_call['price'], quantity=position_size),
            Leg(type='call', side='buy', strike=higher_call['strike'],
                entry_price=higher_call['price'], quantity=position_size),
            Leg(type='put', side='sell', strike=otm_put['strike'],
                entry_price=otm_put['price'], quantity=position_size),
            Leg(type='put', side='buy', strike=lower_put['strike'],
                entry_price=lower_put['price'], quantity=position_size),
        ]
        
        return MultiLegTrade(
            trade_id=f"trade_{timestamp.isoformat()}",
            strategy_type=strategy_type,
            legs=legs,
            entry_time=timestamp,
            underlying_entry=underlying,
            underlying_entry_iv=underlying_entry_iv,
        )
    
    return None


def _find_strike(target: float, options: List[Dict], method: str = 'nearest') -> Optional[Dict]:
    """Find option with strike closest to target."""
    if not options:
        return None
    
    if method == 'nearest':
        return min(options, key=lambda x: abs(x['strike'] - target))
    
    return None


def _get_exit_price(leg: Leg, data: Dict[str, Any]) -> Optional[float]:
    """Get exit price for a leg from current data."""
    options = data.get(f"{leg.type}s", [])
    for opt in options:
        if opt.get('strike') == leg.strike:
            return opt.get('price')
    return None


def _empty_result() -> BacktestResult:
    """Return empty backtest result."""
    return BacktestResult(
        total_pnl=0.0, gross_pnl=0.0, total_costs=0.0,
        num_trades=0, winning_trades=0, losing_trades=0,
        win_rate=0.0, avg_win=0.0, avg_loss=0.0,
        profit_factor=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        max_drawdown=0.0, max_drawdown_pct=0.0, calmar_ratio=0.0,
        trade_list=[], equity_curve=[0.0], returns=[],
    )


def _compile_results(trades: List[MultiLegTrade], equity_curve: List[float],
                     returns: List[float]) -> BacktestResult:
    """Compile backtest results into BacktestResult."""
    if not trades:
        return _empty_result()
    
    import numpy as np
    
    total_pnl = sum(t.net_pnl for t in trades)
    gross_pnl = sum(sum(leg.calculate_pnl() for leg in t.legs) for t in trades)
    total_costs = sum(t.transaction_costs for t in trades)
    
    winning = [t for t in trades if t.net_pnl > 0]
    losing = [t for t in trades if t.net_pnl <= 0]
    
    win_rate = len(winning) / len(trades) if trades else 0.0
    avg_win = sum(t.net_pnl for t in winning) / len(winning) if winning else 0.0
    avg_loss = sum(t.net_pnl for t in losing) / len(losing) if losing else 0.0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)
    
    if len(equity_curve) > 1 and max_dd_pct != 0:
        total_return = (equity_curve[-1] - equity_curve[0]) / max(abs(equity_curve[0]), 1.0)
        annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        calmar = annualized_return / abs(max_dd_pct / 100.0)
    else:
        calmar = 0.0
    
    return BacktestResult(
        total_pnl=total_pnl,
        gross_pnl=gross_pnl,
        total_costs=total_costs,
        num_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        calmar_ratio=calmar,
        trade_list=trades,
        equity_curve=equity_curve,
        returns=returns,
    )


def _construct_trade_state(trade: MultiLegTrade, data: Dict[str, Any], 
                           current_time: datetime, stop_loss: float, 
                           profit_target: float) -> Dict[str, Any]:
    """Construct trade state dictionary for exit optimizer input."""
    unrealized_pnl = 0.0
    for leg in trade.legs:
        current_price = _get_exit_price(leg, data)
        if current_price is None:
            current_price = leg.entry_price
        multiplier = 1 if leg.is_long else -1
        leg_pnl = (current_price - leg.entry_price) * leg.quantity * multiplier * 75
        unrealized_pnl += leg_pnl
    
    entry_value = sum(leg.entry_price * leg.quantity * 75 for leg in trade.legs)
    pnl_pct = (unrealized_pnl / entry_value * 100.0) if entry_value != 0 else 0.0
    
    time_in_trade = (current_time - trade.entry_time).total_seconds() / 60.0
    
    if unrealized_pnl > trade.max_profit_seen:
        trade.max_profit_seen = unrealized_pnl
    if unrealized_pnl < trade.max_loss_seen:
        trade.max_loss_seen = unrealized_pnl
    
    current_drawdown = trade.max_profit_seen - unrealized_pnl
    
    delta = 0.0
    gamma = 0.0
    theta = 0.0
    vega = 0.0
    for leg in trade.legs:
        option_key = f"{leg.type}_{leg.strike}"
        greeks = data.get('greeks', {}).get(option_key, {})
        leg_mult = 1 if leg.is_long else -1
        delta += greeks.get('delta', 0.0) * leg.quantity * leg_mult
        gamma += greeks.get('gamma', 0.0) * leg.quantity * leg_mult
        theta += greeks.get('theta', 0.0) * leg.quantity * leg_mult
        vega += greeks.get('vega', 0.0) * leg.quantity * leg_mult
    
    current_iv = data.get('iv', trade.underlying_entry_iv)
    iv_change_pct = ((current_iv - trade.underlying_entry_iv) / trade.underlying_entry_iv * 100.0) if trade.underlying_entry_iv != 0 else 0.0
    
    current_underlying = data.get('underlying_price', trade.underlying_entry)
    underlying_change_pct = ((current_underlying - trade.underlying_entry) / trade.underlying_entry * 100.0) if trade.underlying_entry != 0 else 0.0
    
    return {
        "unrealized_pnl": unrealized_pnl,
        "pnl_pct": pnl_pct,
        "time_in_trade_minutes": time_in_trade,
        "entry_price": entry_value / (75 * sum(leg.quantity for leg in trade.legs)) if sum(leg.quantity for leg in trade.legs) != 0 else 0.0,
        "current_price": current_underlying,
        "stop_loss": stop_loss,
        "profit_target": profit_target,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "iv": current_iv,
        "iv_change_pct": iv_change_pct,
        "underlying_change_pct": underlying_change_pct,
        "max_profit_seen": trade.max_profit_seen,
        "max_loss_seen": trade.max_loss_seen,
        "current_drawdown": current_drawdown,
    }


def simulate_with_exit_optimizer(
    strategy_type: str,
    entry_signals: List[int],
    price_data: List[Dict[str, Any]],
    exit_optimizer: ExitOptimizer,
    position_size: int = 1,
    *,
    costs: Optional[TransactionCosts] = None,
    slippage_bps: float = 5.0,
    stop_loss: float = -1e9,
    profit_target: float = 1e9,
) -> BacktestResult:
    """Simulate multi-leg options strategy with integrated exit optimizer.
    
    Uses ExitOptimizer to make exit decisions for open positions, providing
    more realistic backtesting that matches live trading exit logic.
    """
    if not entry_signals or not price_data or len(entry_signals) != len(price_data):
        return _empty_result()
    
    if costs is None:
        costs = TransactionCosts()
    
    trades: List[MultiLegTrade] = []
    equity_curve = [0.0]
    returns = []
    in_trade = False
    current_trade: Optional[MultiLegTrade] = None
    slippage_multiplier = 1 + slippage_bps / 10000.0
    
    for idx, (signal, data) in enumerate(zip(entry_signals, price_data)):
        timestamp = data.get('timestamp', datetime.utcnow())
        
        # Entry logic
        if signal == 1 and not in_trade:
            current_trade = _create_multi_leg_trade(strategy_type, data, position_size, timestamp)
            
            if current_trade:
                for leg in current_trade.legs:
                    if leg.is_long:
                        leg.entry_price *= slippage_multiplier
                    else:
                        leg.entry_price /= slippage_multiplier
                in_trade = True
        
        # Exit logic using exit optimizer
        if in_trade and current_trade:
            trade_state = _construct_trade_state(current_trade, data, timestamp, stop_loss, profit_target)
            exit_decision: ExitDecision = exit_optimizer.suggest_exit(trade_state)
            
            if exit_decision.action != "hold":
                exit_percentage = exit_decision.exit_percentage if exit_decision.action == "partial_exit" else 1.0
                
                for leg in current_trade.legs:
                    current_price = _get_exit_price(leg, data)
                    if current_price is None:
                        current_price = leg.entry_price
                    
                    if leg.is_long:
                        exit_price = current_price / slippage_multiplier
                    else:
                        exit_price = current_price * slippage_multiplier
                    
                    if exit_percentage < 1.0:
                        leg.exit_price = (leg.entry_price * (1 - exit_percentage) + exit_price * exit_percentage)
                    else:
                        leg.exit_price = exit_price
                
                current_trade.exit_time = timestamp
                current_trade.reason_for_exit = exit_decision.reason
                current_trade.calculate_pnl(costs)
                trades.append(current_trade)
                
                equity_curve.append(equity_curve[-1] + current_trade.net_pnl)
                if len(equity_curve) > 1:
                    ret = (equity_curve[-1] - equity_curve[-2]) / max(abs(equity_curve[-2]), 1.0)
                    returns.append(ret)
                
                if exit_decision.action == "partial_exit" and exit_percentage < 1.0:
                    for leg in current_trade.legs:
                        leg.quantity = int(leg.quantity * (1 - exit_percentage))
                        leg.entry_price = leg.exit_price
                    current_trade.max_profit_seen = 0.0
                    current_trade.max_loss_seen = 0.0
                else:
                    in_trade = False
                    current_trade = None
        
        # Force exit on last data point
        if idx == len(entry_signals) - 1 and in_trade and current_trade:
            for leg in current_trade.legs:
                current_price = _get_exit_price(leg, data)
                if current_price:
                    if leg.is_long:
                        leg.exit_price = current_price / slippage_multiplier
                    else:
                        leg.exit_price = current_price * slippage_multiplier
                else:
                    leg.exit_price = leg.entry_price
            
            current_trade.exit_time = timestamp
            current_trade.reason_for_exit = "end_of_data"
            current_trade.calculate_pnl(costs)
            trades.append(current_trade)
            
            equity_curve.append(equity_curve[-1] + current_trade.net_pnl)
            if len(equity_curve) > 1:
                ret = (equity_curve[-1] - equity_curve[-2]) / max(abs(equity_curve[-2]), 1.0)
                returns.append(ret)
            
            in_trade = False
            current_trade = None
    
    return _compile_results(trades, equity_curve, returns)


# Keep original simple simulation for backward compatibility
@dataclass
class SimulatedTrade:
    direction: int
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    quantity: float
    fill_fraction: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float


def _signal_direction(signal: int) -> int:
    try:
        v = int(signal)
    except Exception:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def simulate_simple(
    price_series: Iterable[float],
    signals: Iterable[int],
    position_size: int = 1,
    *,
    slippage_bps: float = 0.0,
    fee_per_order: float = 0.0,
    partial_fill_rate: float = 1.0,
    return_trades: bool = False,
) -> Tuple[float, List[float]]:
    """Simulate simple long-only trades where `signals` is 1 for long, 0 for flat.

    Entry/exit occurs on signal transitions. Signals may also be -1 for short.
    Returns total PnL and list of trade PnLs or detailed trade dictionaries.
    """
    prices = list(price_series)
    sigs = list(signals)
    if not prices or not sigs or len(prices) != len(sigs):
        return 0.0, []
    in_trade = False
    direction = 0
    entry_price = 0.0
    entry_index = 0
    trade_pnls: List[float] = []
    trade_details: List[SimulatedTrade] = []

    def slippage_price(price: float, side: int, is_entry: bool) -> float:
        slip = abs(float(slippage_bps)) / 10000.0
        if slip <= 0:
            return float(price)
        if side > 0:
            return float(price) * (1.0 + slip if is_entry else 1.0 - slip)
        return float(price) * (1.0 - slip if is_entry else 1.0 + slip)

    fill_fraction = max(0.0, min(1.0, float(partial_fill_rate)))
    eff_qty = float(max(1, int(position_size))) * fill_fraction

    for idx, (p, s_raw) in enumerate(zip(prices, sigs)):
        s = _signal_direction(s_raw)
        if not in_trade and s != 0:
            in_trade = True
            direction = s
            entry_index = idx
            entry_price = slippage_price(float(p), direction, True)
            continue

        if in_trade and s != direction:
            exit_price = slippage_price(float(p), direction, False)
            gross = (exit_price - entry_price) * eff_qty * float(direction)
            slippage_cost = abs(float(p) - exit_price) * eff_qty + abs(float(p) - entry_price) * eff_qty
            fees = float(fee_per_order) * 2.0
            net = gross - fees
            trade_pnls.append(net)
            trade_details.append(
                SimulatedTrade(
                    direction=direction,
                    entry_index=entry_index,
                    exit_index=idx,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    quantity=eff_qty,
                    fill_fraction=fill_fraction,
                    gross_pnl=float(gross),
                    fees=fees,
                    slippage_cost=float(slippage_cost),
                    net_pnl=float(net),
                )
            )
            in_trade = False
            direction = 0
            if s != 0:
                in_trade = True
                direction = s
                entry_index = idx
                entry_price = slippage_price(float(p), direction, True)

    if in_trade:
        exit_price = slippage_price(float(prices[-1]), direction, False)
        gross = (exit_price - entry_price) * eff_qty * float(direction)
        slippage_cost = abs(float(prices[-1]) - exit_price) * eff_qty + abs(float(prices[-1]) - entry_price) * eff_qty
        fees = float(fee_per_order) * 2.0
        net = gross - fees
        trade_pnls.append(net)
        trade_details.append(
            SimulatedTrade(
                direction=direction,
                entry_index=entry_index,
                exit_index=len(prices) - 1,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                quantity=eff_qty,
                fill_fraction=fill_fraction,
                gross_pnl=float(gross),
                fees=fees,
                slippage_cost=float(slippage_cost),
                net_pnl=float(net),
            )
        )

    total = float(sum(trade_pnls))
    if return_trades:
        return total, [t.__dict__ for t in trade_details]
    return total, trade_pnls