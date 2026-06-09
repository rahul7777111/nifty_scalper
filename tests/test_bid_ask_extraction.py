"""Tests for extract_bid_ask helper and bid/ask blocking logic."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mstock_client import MStockTypeBClient

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ea(payload: dict):
    """Short alias for extract_bid_ask."""
    return MStockTypeBClient.extract_bid_ask(payload)


# ---------------------------------------------------------------------
# Test 1: Direct bid_price / ask_price (flat keys)
# ---------------------------------------------------------------------

def test_direct_bid_price_ask_price():
    payload = {
        "symbol": "NIFTY26600CE",
        "bid_price": 145.50,
        "ask_price": 147.25,
        "bid_qty": 150,
        "ask_qty": 150,
    }
    bid, ask, bq, aq = ea(payload)
    assert bid == 145.50, f"bid expected 145.50, got {bid}"
    assert ask == 147.25, f"ask expected 147.25, got {ask}"
    assert bq == 150
    assert aq == 150


def test_best_bid_price_best_ask_price():
    payload = {
        "symbol": "NIFTY26600CE",
        "best_bid_price": 88.0,
        "best_ask_price": 89.5,
        "best_bid_qty": 300,
        "best_ask_qty": 300,
    }
    bid, ask, bq, aq = ea(payload)
    assert bid == 88.0, f"bid expected 88.0, got {bid}"
    assert ask == 89.5, f"ask expected 89.5, got {ask}"
    assert bq == 300
    assert aq == 300


def test_short_bid_ask_keys():
    """bid / ask short-form keys."""
    payload = {"symbol": "NIFTY26600PE", "bid": 72.0, "ask": 73.1}
    bid, ask, bq, aq = ea(payload)
    assert bid == 72.0
    assert ask == 73.1
    assert bq is None
    assert aq is None


# ---------------------------------------------------------------------
# Test 2: depth.buy[0].price / depth.sell[0].price
# ---------------------------------------------------------------------

def test_depth_buy_sell_price():
    payload = {
        "symbol": "NIFTY26600CE",
        "depth": {
            "buy": [{"price": 145.0, "quantity": 600, "side": "BUY"}],
            "sell": [{"price": 147.0, "quantity": 600, "side": "SELL"}],
        }
    }
    bid, ask, bq, aq = ea(payload)
    assert bid == 145.0, f"bid expected 145.0, got {bid}"
    assert ask == 147.0, f"ask expected 147.0, got {ask}"
    assert bq == 600
    assert aq == 600


def test_depth_nested_other_keys():
    """depth.buy[0] with non-standard inner key names."""
    payload = {
        "symbol": "NIFTY26600CE",
        "depth": {
            "buy": [{"rate": 144.5, "qty": 450}],
            "sell": [{"rate": 146.8, "qty": 450}],
        }
    }
    bid, ask, bq, aq = ea(payload)
    assert bid == 144.5, f"bid expected 144.5, got {bid}"
    assert ask == 146.8, f"ask expected 146.8, got {ask}"


def test_market_depth_buy_sell():
    """market_depth.buy[0].price / market_depth.sell[0].price variant."""
    payload = {
        "symbol": "NIFTY26600CE",
        "marketDepth": {
            "buy": [{"price": 145.25, "quantity": 750}],
            "sell": [{"price": 147.05, "quantity": 750}],
        }
    }
    bid, ask, bq, aq = ea(payload)
    assert bid == 145.25
    assert ask == 147.05
    assert bq == 750
    assert aq == 750


# ---------------------------------------------------------------------
# Test 3: flat keys take priority over depth
# ---------------------------------------------------------------------

def test_flat_keys_override_depth():
    """When both flat bid_price AND depth.buy[0].price exist, flat wins."""
    payload = {
        "bid_price": 145.99,   # should win
        "ask_price": 148.01,
        "depth": {
            "buy": [{"price": 1.0}],
            "sell": [{"price": 999.0}],
        }
    }
    bid, ask, _, _ = ea(payload)
    assert bid == 145.99, f"flat bid_price should take priority, got {bid}"
    assert ask == 148.01


# ---------------------------------------------------------------------
# Test 4: Missing bid/ask → None, None
# ---------------------------------------------------------------------

def test_missing_bid_ask_returns_none():
    """No bid/ask fields at all → returns (None, None, None, None)."""
    payload = {
        "symbol": "NIFTY26600CE",
        "ltp": 146.50,   # LTP alone must NOT be used as a bid/ask substitute
        "depth": {},      # empty depth
    }
    bid, ask, bq, aq = ea(payload)
    assert bid is None, f"bid should be None when no bid field, got {bid}"
    assert ask is None, f"ask should be None when no ask field, got {ask}"
    assert bq is None
    assert aq is None


def test_ltp_alone_does_not_become_bid_or_ask():
    """LTP must never be promoted to bid or ask."""
    for ltp_key in ("ltp", "lastPrice", "lastTradedPrice"):
        payload = {ltp_key: 146.50, "depth": {}}
        bid, ask, _, _ = ea(payload)
        assert bid is None, f"ltp={ltp_key}: bid must be None, got {bid}"
        assert ask is None, f"ltp={ltp_key}: ask must be None, got {ask}"


# ---------------------------------------------------------------------
# Test 5: Quantity extraction from depth
# ---------------------------------------------------------------------

def test_depth_buy_sell_quantity():
    payload = {
        "depth": {
            "buy": [{"price": 100.0, "quantity": 1200}],
            "sell": [{"price": 101.0, "quantity": 900}],
        }
    }
    _, _, bq, aq = ea(payload)
    assert bq == 1200, f"bid_qty expected 1200, got {bq}"
    assert aq == 900, f"ask_qty expected 900, got {aq}"


def test_depth_qty_with_non_standard_keys():
    """quantity, qty, bq, sq variants."""
    payload = {
        "depth": {
            "buy": [{"price": 100.0, "qty": 500}],
            "sell": [{"price": 101.0, "qty": 700}],
        }
    }
    _, _, bq, aq = ea(payload)
    assert bq == 500
    assert aq == 700


# ---------------------------------------------------------------------
# Test 6: List-of-dict depth (side field variant)
# ---------------------------------------------------------------------

def test_depth_list_with_side_field():
    payload = {
        "depth": [
            {"side": "BID", "price": 145.0, "quantity": 300},
            {"side": "ASK", "price": 147.0, "quantity": 300},
        ]
    }
    bid, ask, bq, aq = ea(payload)
    assert bid == 145.0
    assert ask == 147.0
    assert bq == 300
    assert aq == 300


def test_depth_list_fallback_first_two():
    """Without side field, first two rows are bid/ask."""
    payload = {
        "depth": [
            {"price": 144.8},
            {"price": 146.9},
        ]
    }
    bid, ask, _, _ = ea(payload)
    assert bid == 144.8
    assert ask == 146.9


# ---------------------------------------------------------------------
# Test 7: Invalid / None inputs don't crash
# ---------------------------------------------------------------------

def test_none_input():
    bid, ask, bq, aq = ea(None)
    assert bid is None and ask is None


def test_empty_dict():
    bid, ask, bq, aq = ea({})
    assert bid is None and ask is None


def test_malformed_depth():
    """Depth is a list-of-strings or non-dict — must not crash."""
    payload = {"depth": ["not a dict", 42, None], "bid_price": 10.0, "ask_price": 11.0}
    bid, ask, _, _ = ea(payload)
    assert bid == 10.0
    assert ask == 11.0