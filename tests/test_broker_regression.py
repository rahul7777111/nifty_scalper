import sys
import os
import time
import types
from datetime import date

# Setup paths
sys.path.insert(0, os.path.abspath("src"))

from config import StrategyConfig
from market_data import Candle
from strategy import NiftyScalper, TradeState

class MockBrokerClient:
    def __init__(self) -> None:
        self.orders_placed = []
        self.ltp_requests = []
        
    def _get_scripmaster(self):
        return None
        
    def resolve_exchange_token_symbol(self, symbol, exchange_hint=None):
        if "750PE" in symbol or "23750PE" in symbol:
            return "NFO", "12345", "NIFTY2660223750PE"
        if "950CE" in symbol or "23950CE" in symbol:
            return "NFO", "12346", "NIFTY2660223950CE"
        return "NFO", "99999", symbol

    def get_ltp(self, symbol):
        self.ltp_requests.append(symbol)
        return 150.0

    def get_bid_ask(self, symbol, exchange_hint=None):
        return 149.0, 151.0, 150.0

    def get_candles(self, *args, **kwargs):
        return []

    def place_order(self, **kwargs):
        # Record the placed order
        symbol = kwargs.get("symbol")
        side = kwargs.get("side")
        qty = kwargs.get("quantity")
        exchange = kwargs.get("exchange")
        token = kwargs.get("symbol_token")
        self.orders_placed.append({
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "exchange": exchange,
            "symbol_token": token
        })
        order = types.SimpleNamespace()
        order.order_id = f"ord_{len(self.orders_placed)}"
        order.status = "SUCCESS"
        return order

def test_open_close_ce_uses_identical_broker_symbol():
    cfg = StrategyConfig(
        strategy_name="directional",
        enable_live_trading=True,
        lot_size=50,
        underlying="NIFTY",
        max_hold_minutes=10,
        target_expiry="02JUN23",
    )
    client = MockBrokerClient()
    
    # Initialize the NiftyScalper bot
    bot = NiftyScalper(client, cfg)
    
    opt = {
        "symbol": "NIFTY2660223950CE", # Correct broker symbol from option chain
        "token": "12346",
        "exchange": "NFO",
        "strike": 23950.0,
        "option_type": "CE",
        "expiry": date(2023, 6, 2)
    }
    
    # Trigger option entry
    tr = bot._open_directional_from_option(opt, name="long_call", spot=23950.0, atr_val=10.0)
    
    # Verify the real broker symbol was stored
    assert tr is not None
    assert tr["symbol"] == "NIFTY2660223950CE"
    assert tr["token"] == "12346"
    assert tr["exchange"] == "NFO"
    assert tr["legs"][0]["symbol"] == "NIFTY2660223950CE"
    assert tr["legs"][0]["token"] == "12346"
    
    # Now simulate closing the trade
    # Modify tr opened_ts to force time-based exit
    tr["opened_ts"] = time.time() - 1000.0 # 1000 seconds ago, greater than 10 mins (600s)
    
    bot.state.open_directional = [tr]
    bot._manage_open_trades()
    
    # Verify the identical broker symbol and token were passed to place_order at exit
    assert len(client.orders_placed) == 2 # 1 entry + 1 exit
    placed = client.orders_placed[1]
    assert placed["symbol"] == "NIFTY2660223950CE"
    assert placed["symbol_token"] == "12346"
    assert placed["exchange"] == "NFO"
    assert placed["side"] == "SELL" # Close BUY with SELL
    
    # Verify that the trade was removed from open_directional since the close succeeded
    assert len(bot.state.open_directional) == 0

def test_open_close_pe_uses_identical_broker_symbol():
    cfg = StrategyConfig(
        strategy_name="directional",
        enable_live_trading=True,
        lot_size=50,
        underlying="NIFTY",
        max_hold_minutes=10,
        target_expiry="02JUN23",
    )
    client = MockBrokerClient()
    
    bot = NiftyScalper(client, cfg)
    
    opt = {
        "symbol": "NIFTY2660223750PE", # Correct broker symbol from option chain
        "token": "12345",
        "exchange": "NFO",
        "strike": 23750.0,
        "option_type": "PE",
        "expiry": date(2023, 6, 2)
    }
    
    # Trigger option entry
    tr = bot._open_directional_from_option(opt, name="long_put", spot=23750.0, atr_val=10.0)
    
    # Verify real broker symbol was stored
    assert tr is not None
    assert tr["symbol"] == "NIFTY2660223750PE"
    assert tr["token"] == "12345"
    assert tr["exchange"] == "NFO"
    
    # Simulate time exit
    tr["opened_ts"] = time.time() - 1000.0
    bot.state.open_directional = [tr]
    
    bot._manage_open_trades()
    
    # Verify exact broker symbol passed to exit
    assert len(client.orders_placed) == 2 # 1 entry + 1 exit
    placed = client.orders_placed[1]
    assert placed["symbol"] == "NIFTY2660223750PE"
    assert placed["symbol_token"] == "12345"
    
    # Verify removed from open_directional
    assert len(bot.state.open_directional) == 0

def test_basic_directional_resolves_option_symbol_in_paper():
    cfg = StrategyConfig(
        strategy_name="directional",
        enable_live_trading=False, # Paper mode
        lot_size=50,
        underlying="NIFTY",
        max_hold_minutes=10,
        target_expiry="02JUN23",
    )
    client = MockBrokerClient()
    bot = NiftyScalper(client, cfg)
    
    candles = [Candle(time=date.today(), open=23950.1, high=23950.1, low=23950.1, close=23950.1, volume=1000.0)]
    
    # Trigger directional entry signal in paper mode
    # option_symbol is NIFTY02JUN23950CE, which resolves to NIFTY2660223950CE
    tr = bot._simulate_or_place_basic_directional(take=True, size=50, reason="bullish", candles=candles)
    
    assert tr is not None
    assert tr["symbol"] == "NIFTY2660223950CE"
    assert tr["token"] == "12346"
    assert tr["exchange"] == "NFO"

def test_infinite_loop_prevention_on_live_close_failure():
    cfg = StrategyConfig(
        strategy_name="directional",
        enable_live_trading=True,
        lot_size=50,
        underlying="NIFTY",
        max_hold_minutes=10,
        target_expiry="02JUN23",
    )
    
    class FailingClient(MockBrokerClient):
        def place_order(self, **kwargs):
            # Fail live orders to test error handling / retries
            raise RuntimeError("m.Stock exchange reject: Scrip Name not found in exchange file.")
            
    client = FailingClient()
    bot = NiftyScalper(client, cfg)
    
    # Set up an existing open trade already in the state
    tr = {
        "trade_id": "D1",
        "name": "auto_ml_directional",
        "symbol": "NIFTY2660223750PE",
        "token": "12345",
        "exchange": "NFO",
        "side": "BUY",
        "quantity": 50,
        "entry_spot": 23750.0,
        "entry_price": 150.0,
        "atr": 10.0,
        "legs": [
            {
                "symbol": "NIFTY2660223750PE",
                "token": "12345",
                "exchange": "NFO",
                "quantity": 50,
                "side": "BUY",
            }
        ],
        "opened_ts": time.time() - 1000.0, # force time exit
    }
    
    bot.state.open_directional = [tr]
    
    # First close attempt
    bot._manage_open_trades()
    
    # Verify it failed but remained in the open list
    assert len(bot.state.open_directional) == 1
    stored_tr = bot.state.open_directional[0]
    assert stored_tr["close_attempts"] == 1
    assert not stored_tr.get("close_failed")
    
    # Second close attempt within 30s (should be skipped due to cooldown!)
    bot._manage_open_trades()
    assert stored_tr["close_attempts"] == 1 # still 1 because skipped during cooldown!
    
    # Force bypass cooldown to simulate second retry
    stored_tr["last_close_attempt_ts"] = time.time() - 100.0
    bot._manage_open_trades()
    assert stored_tr["close_attempts"] == 2
    assert not stored_tr.get("close_failed")
    
    # Force bypass cooldown to simulate third retry
    stored_tr["last_close_attempt_ts"] = time.time() - 100.0
    bot._manage_open_trades()
    assert stored_tr["close_attempts"] == 3
    assert stored_tr["close_failed"] is True # marked as failed close!
    
    # Subsequent cycles should skip the close attempt entirely
    stored_tr["last_close_attempt_ts"] = time.time() - 100.0
    bot._manage_open_trades()
    assert stored_tr["close_attempts"] == 3 # no new attempts made!


def test_get_historical_candles_success():
    from unittest.mock import MagicMock, patch
    from mstock_client import MStockTypeBClient
    from config import APIConfig
    
    cfg = APIConfig(
        base_url="https://api.mstock.trade",
        api_key="mock_api_key",
        api_secret="mock_api_secret",
        client_id="mock_client_id"
    )
    client = MStockTypeBClient(cfg)
    
    # Mock _ensure_valid_token to prevent doing real auth checks
    client._ensure_valid_token = MagicMock()
    
    # Mock data matching m.Stock response structure
    mock_response = {
        "status": "true",
        "message": "success",
        "errorcode": "",
        "data": {
            "candles": [
                [
                    "2024-08-02T09:15:00+05",
                    3790.0,
                    3832.0,
                    3773.35,
                    3803.0,
                    825219
                ],
                [
                    "2024-08-03T09:15:00+05",
                    3811.1,
                    3811.1,
                    3767.25,
                    3781.15,
                    1343722
                ]
            ]
        }
    }
    
    with patch.object(client, '_fetch_historical_chart_direct', return_value=mock_response) as mock_fetch:
        candles = client.get_historical_candles(
            symbol_token="11536",
            exchange="NSE",
            interval="1m",
            from_date="2024-08-02 09:15",
            to_date="2024-08-03 09:15"
        )
        
        # Verify direct fetcher was called with normalized interval
        mock_fetch.assert_called_once_with(
            exchange="NSE",
            symboltoken="11536",
            interval="ONE_MINUTE",
            from_date="2024-08-02 09:15",
            to_date="2024-08-03 09:15"
        )
        
        # Verify candles were parsed correctly
        assert len(candles) == 2
        assert candles[0].open == 3790.0
        assert candles[0].high == 3832.0
        assert candles[0].low == 3773.35
        assert candles[0].close == 3803.0
        assert candles[0].volume == 825219.0
        
        assert candles[1].open == 3811.1
        assert candles[1].high == 3811.1
        assert candles[1].low == 3767.25
        assert candles[1].close == 3781.15
        assert candles[1].volume == 1343722.0


def test_get_historical_candles_propagates_error():
    from unittest.mock import MagicMock, patch
    from mstock_client import MStockTypeBClient
    from config import APIConfig
    
    cfg = APIConfig(
        base_url="https://api.mstock.trade",
        api_key="mock_api_key",
        api_secret="mock_api_secret",
        client_id="mock_client_id"
    )
    client = MStockTypeBClient(cfg)
    client._ensure_valid_token = MagicMock()
    
    with patch.object(client, '_fetch_historical_chart_direct', side_effect=RuntimeError("HTTP 401 Unauthorized")) as mock_fetch:
        try:
            client.get_historical_candles(
                symbol_token="11536",
                exchange="NSE",
                interval="ONE_MINUTE",
                from_date="2024-08-02 09:15",
                to_date="2024-08-03 09:15"
            )
            assert False, "Expected RuntimeError to propagate"
        except RuntimeError as e:
            assert "HTTP 401 Unauthorized" in str(e)


def test_parse_option_symbol_strike_reconstruction():
    from mstock_client import MStockTypeBClient
    from config import APIConfig
    
    cfg = APIConfig(
        base_url="https://api.mstock.trade",
        api_key="mock_api_key",
        api_secret="mock_api_secret",
        client_id="mock_client_id"
    )
    client = MStockTypeBClient(cfg)
    
    cases = {
        "NIFTY02JUN23750PE": 23750.0,
        "NIFTY02JUN23950CE": 23950.0,
        "NIFTY02JUN24000CE": 24000.0,
        "NIFTY02JUN23850PE": 23850.0,
    }
    
    for symbol, expected_strike in cases.items():
        parsed = client._parse_option_symbol(symbol)
        assert parsed is not None, f"Failed to parse {symbol}"
        assert parsed["strike"] == expected_strike, f"For {symbol}, expected strike {expected_strike} but got {parsed['strike']}"
        # Expressly reject the wrong truncated strikes
        assert parsed["strike"] not in {750.0, 950.0, 0.0, 850.0}, f"Parser produced invalid truncated strike {parsed['strike']} for {symbol}"



