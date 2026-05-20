import sys
import os
import time

# Ensure src is in path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Mock Env
os.environ["MSTOCK_USE_INTRADAY_CHART"] = "true"
os.environ["MSTOCK_UNDERLYING"] = "NIFTY"
os.environ["MSTOCK_UNDERLYING_TOKEN"] = "26000"
os.environ["MSTOCK_API_KEY"] = "dummy" 

try:
    from mstock_client import MStockTypeBClient
    from config import APIConfig, StrategyConfig
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Mock _fetch_intraday_chart to fail or return few candles
def mock_fetch_intraday(*args, **kwargs):
    print("DEBUG: mock_fetch_intraday called")
    return []

# Mock get_historical_chart to fail
class MockRaw:
    def get_historical_chart(*args):
        raise Exception("Mock Historical Data Unavailable")

def test_integration():
    print("--- Testing MStock Client Yahoo Fallback ---")
    
    cfg = APIConfig(api_key="test", base_url="https://example.com", api_secret="test", client_id="test")
    client = MStockTypeBClient(cfg)
    
    # Patch methods
    client._fetch_intraday_chart = mock_fetch_intraday
    client._resolve_symbol_token = lambda s: "26000"
    client._raw = MockRaw()
    client.login = lambda: None # Mock login
    
    # Test get_candles
    print("Calling get_candles('26000', '5m')...")
    # We expect this to fail m.Stock calls and fall into Yahoo
    try:
        candles = client.get_candles("26000", "5m")
        print(f"Result: Got {len(candles)} candles")
        if candles:
            print(f"First: {candles[0]}")
            print(f"Last: {candles[-1]}")
    except Exception as e:
        print(f"get_candles failed: {e}")

if __name__ == "__main__":
    test_integration()
