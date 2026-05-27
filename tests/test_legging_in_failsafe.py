import sys
import os
import types
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.abspath("src"))

from config import StrategyConfig
from market_data import Candle
from strategy import NiftyScalper, TradeState

class MockUnwindClient:
    def __init__(self) -> None:
        self.orders_placed = []
        
    def place_order(self, **kwargs):
        # Record the placed order
        symbol = kwargs.get("symbol")
        side = kwargs.get("side")
        qty = kwargs.get("quantity")
        self.orders_placed.append((symbol, side, qty))
        
        # Simulate a broker rejection/failure on the second leg (NIFTY_PE)!
        if symbol == "NIFTY_PE":
            raise RuntimeError("Broker rejection: Insufficient Margin for SELL leg")
            
        # Return a dummy order object
        order = types.SimpleNamespace()
        order.order_id = f"ord_{len(self.orders_placed)}"
        return order

    def get_candles(self, *args, **kwargs):
        return []

    def get_option_chain(self, *args, **kwargs):
        return []

def test_legging_in_protection_failsafe_rollback():
    print("=== Testing Legging-In Fail-Safe Rollback ===")
    
    # 1. Setup strategy config and fake client
    cfg = StrategyConfig(
        strategy_name="directional",
        enable_live_trading=True, # force live order placement path
        lot_size=50,
        order_retry_attempts=1
    )
    client = MockUnwindClient()
    
    # 2. Mock strategy setup
    bot = NiftyScalper.__new__(NiftyScalper)
    bot.cfg = cfg
    bot.client = client
    bot.state = TradeState(open_orders=[], open_multi=[], open_directional=[])
    
    # Mock necessary recording stubs
    bot._record_multi_trade_with_meta = lambda *args: None
    
    # Build synthetic options contracts
    call_leg = {"symbol": "NIFTY_CE", "token": "1", "exchange": "NFO", "strike": 22000.0}
    put_leg = {"symbol": "NIFTY_PE", "token": "2", "exchange": "NFO", "strike": 22000.0}
    
    # 3. Simulate sequential straddle entry.
    # Order 1 (call CE): side="SELL", qty=50 -> Should succeed.
    # Order 2 (put PE): side="SELL", qty=50 -> Should fail and trigger rollback of Order 1!
    legs_to_place = [
        (call_leg, "SELL", 50),
        (put_leg, "SELL", 50)
    ]
    
    raw_legs = [dict(call_leg), dict(put_leg)]
    meta = {"entry_spot": 22000.0}
    
    # Verify that a RuntimeError is raised out of the transactional helper
    try:
        bot._place_multi_leg_transactional(legs_to_place, "short_straddle", raw_legs, meta)
        assert False, "Should have raised RuntimeError on order failure"
    except RuntimeError as exc:
        print(f"Caught expected transaction failure exception: {exc}")
        assert "Multi-leg transaction failed" in str(exc)
        
    # 4. Verify Rollback Execution
    # Placed orders should have been:
    # 1. ("NIFTY_CE", "SELL", 50) -> The original short call entry leg.
    # 2. ("NIFTY_PE", "SELL", 50) -> The short put entry leg (raised exception).
    # 3. ("NIFTY_CE", "BUY", 50) -> The EMERGENCY offsetting market order to buy back the call!
    print("Placed orders during execution loop:")
    for symbol, side, qty in client.orders_placed:
        print(f"  Order: symbol={symbol} side={side} qty={qty}")
        
    assert len(client.orders_placed) == 3, f"Expected 3 orders, got {len(client.orders_placed)}"
    assert client.orders_placed[0] == ("NIFTY_CE", "SELL", 50), "First order must be the short call entry"
    assert client.orders_placed[1] == ("NIFTY_PE", "SELL", 50), "Second order must be the short put entry"
    assert client.orders_placed[2] == ("NIFTY_CE", "BUY", 50), "Third order must be the BUY rollback offsetting order"
    
    print("✔ Legging-In Protection Fail-Safe Rollback verification PASSED!")

if __name__ == "__main__":
    test_legging_in_protection_failsafe_rollback()
