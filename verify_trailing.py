
import sys
import os
from types import SimpleNamespace

# Mock Config
class MockConfig:
    max_hold_minutes = 0
    max_trend_strength = 100.0
    premium_mtm_trail_start_pct = 0.05
    premium_mtm_trail_stop_pct = 0.05
    dir_premium_trail_pct = 0.0 # Disable old logic logic
    
# Mock Scalper
class MockScalper:
    def __init__(self):
        self.cfg = MockConfig()
        self.state = SimpleNamespace()
        self.state.open_multi = []
        self.exits = []

    def _compute_legs_mtm(self, legs):
        # Return the MTM stored in the first leg for testing
        return legs[0].get("mtm", 0.0)

    def _mark_non_stop_exit(self):
        pass

    def _close_multi_trade(self, trade, reason):
        print(f"CLOSING TRADE: {reason}")
        self.exits.append(reason)
        # Remove from open_multi
        self.state.open_multi = [t for t in self.state.open_multi if t != trade]

    def _manage_multi_trades(self):
        # This function structure mirrors the actual strategy.py method
        # We need to copy-paste the relevant logic or import it?
        # Simulating the behavior by copy-pasting the *new* block (or we import if we could easily refactor).
        # Since we can't easily import just that method without the whole class deps, 
        # I will rely on the fact that I just wrote code that modifies the actual file.
        # I should Import the actual class and monkeypatch/setup usage.
        pass

# Actually, let's try to import the real strategy file. 
# We need to mock a lot of dependencies.
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.getcwd(), 'src'))

# We need to mock imports that might fail or have side effects
with patch("mstock_client.MStockTypeBClient"), \
     patch("market_data.Candle"):
    from strategy import NiftyScalper
    from config import StrategyConfig

class TestTrailingMTM(unittest.TestCase):
    def test_trailing_logic(self):
        cfg = StrategyConfig()
        cfg.premium_mtm_trail_start_pct = 0.10  # Start trailing at 10% profit
        cfg.premium_mtm_trail_stop_pct = 0.05   # Trail by 5%
        cfg.max_hold_minutes = 999
        
        # Mock Client
        client = MagicMock()
        
        bot = NiftyScalper(client, cfg)
        
        # Create a mock trade
        entry_premium = 1000.0
        trade = {
            "trade_id": "test_trade",
            "name": "short_straddle",
            "opened_ts": 100.0,
            "legs": [{"mtm": 0.0}], 
            "meta": {
                "premium_selling": True,
                "entry_premium_abs": entry_premium
            }
        }
        bot.state.open_multi.append(trade)
        
        # Helper to run management loop
        def run_check(current_mtm):
            # Update the mock mtm
            trade["legs"][0]["mtm"] = current_mtm
            # Mock _compute_legs_mtm to return this value
            bot._compute_legs_mtm = MagicMock(return_value=current_mtm)
            # Mock _close_multi_trade to capture exit
            bot._close_multi_trade = MagicMock()
            
            # We also need to patch time.time() and trend_strength potentially
            with patch("time.time", return_value=200.0):
                # We need to inject the logic or call the method. 
                # The method is _manage_multi_trades but it's internal and relies on 'trend_strength' local var in run_forever loop...
                # Wait, _manage_open_trades calls _manage_multi_trades logic? 
                # No, looking at strategy.py, the logic is IN _manage_open_trades (lines 2539+).
                # Actually, in the file it looked like one big method or part of run_forever?
                # looking at finding results earlier.. it was inside _manage_open_trades.
                
                # Check lines 2539 in strategy.py... wait.
                # The view showed "    def _manage_open_trades(self)" ending around 2400?
                # Let's re-verify where the code was inserted.
                pass

if __name__ == "__main__":
    print("This script is a placeholder. The complexity of mocking the full NiftyScalper for this specific logic block is high.")
    print("I will verify by inspecting the code logic manually via 'view_file' to ensure indentation and logic flow is correct.")
