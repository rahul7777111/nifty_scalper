"""Yahoo Finance integration for fallback market data.

This module provides a backup source for index candles when the primary broker API
fails to provide sufficient historical data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

_yf = None
_yf_import_error: str | None = None

try:
    from .market_data import Candle
except ImportError:
    from market_data import Candle


@dataclass(frozen=True)
class YahooQuote:
    symbol: str
    ltp: float
    ts: datetime


def _check_enabled() -> None:
    if _get_yf() is None:
        detail = f" ({_yf_import_error})" if _yf_import_error else ""
        raise RuntimeError("yfinance not installed/usable. Run `pip install yfinance` to enable fallback." + detail)


def _get_yf():
    """Import yfinance on demand so UI startup does not depend on Yahoo stack."""
    global _yf, _yf_import_error
    if _yf is not None:
        return _yf
    try:
        import yfinance as yf_mod

        _yf = yf_mod
        _yf_import_error = None
        return _yf
    except Exception as exc:
        _yf_import_error = str(exc)
        return None


def yahoo_symbol_for_underlying(underlying: str, *, default: str = "") -> str:
    """Map broker underlying names to Yahoo Finance tickers."""
    u = str(underlying or "").strip().upper()
    mapping = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",  # Valid as of recent yfinance
        "MIDCPNIFTY": "^NSEMDCP50",
    }
    # Direct NSE symbol support (e.g. "RELIANCE")
    if u not in mapping and u.isalpha():
        # Heuristic: if it looks like a stock symbol, try appending .NS
        return f"{u}.NS"
    
    return mapping.get(u, default or u)


def fetch_yahoo_candles(
    yahoo_symbol: str,
    *,
    interval: str = "1m",
    range_: str = "5d",
    limit: int = 1000,
    timeout: float = 15.0,
) -> List[Candle]:
    """Fetch candles from Yahoo Finance with automatic fallback for known indices."""
    _check_enabled()
    yf = _get_yf()
    if yf is None:
        return []
    
    primary_sym = str(yahoo_symbol or "").strip()
    if not primary_sym:
        return []

    # Define fallbacks for common flaky symbols
    fallbacks = []
    if primary_sym == "^NSEI":
        fallbacks = ["NIFTY_50.NS"]
    elif primary_sym == "^NSEBANK":
        fallbacks = ["NIFTY_BANK.NS"]
    
    symbols_to_try = [primary_sym] + fallbacks
    
    def _aggregate_candles(candles: List[Candle], minutes: int) -> List[Candle]:
        if minutes <= 1 or not candles:
            return candles
        out: List[Candle] = []
        bucket: List[Candle] = []
        bucket_start: Optional[datetime] = None

        for candle in sorted(candles, key=lambda x: x.time):
            ts = candle.time
            start_minute = (ts.minute // minutes) * minutes
            key = ts.replace(minute=start_minute, second=0, microsecond=0)
            if bucket_start is None or key != bucket_start:
                if bucket:
                    out.append(
                        Candle(
                            time=bucket_start,
                            open=float(bucket[0].open),
                            high=float(max(c.high for c in bucket)),
                            low=float(min(c.low for c in bucket)),
                            close=float(bucket[-1].close),
                            volume=float(sum(float(c.volume or 0.0) for c in bucket)),
                        )
                    )
                bucket_start = key
                bucket = [candle]
            else:
                bucket.append(candle)

        if bucket and bucket_start is not None:
            out.append(
                Candle(
                    time=bucket_start,
                    open=float(bucket[0].open),
                    high=float(max(c.high for c in bucket)),
                    low=float(min(c.low for c in bucket)),
                    close=float(bucket[-1].close),
                    volume=float(sum(float(c.volume or 0.0) for c in bucket)),
                )
            )
        return out

    requested_interval = str(interval or "1m").strip().lower()
    resample_minutes = 0
    # Map intervals
    interval_map = {
        "ONE_MINUTE": "1m", "1m": "1m",
        "THREE_MINUTE": "1m", "3m": "1m",
        "FIVE_MINUTE": "5m", "5m": "5m",
        "TEN_MINUTE": "15m", "10m": "15m",
        "FIFTEEN_MINUTE": "15m", "15m": "15m",
        "THIRTY_MINUTE": "30m", "30m": "30m",
        "ONE_HOUR": "1h", "1h": "1h",
        "ONE_DAY": "1d", "1d": "1d",
    }
    yf_interval = interval_map.get(interval, interval)
    if requested_interval in {"3m", "three_minute"}:
        resample_minutes = 3
    if yf_interval == "1m":
        range_ = "5d"

    for sym in symbols_to_try:
        try:
            print(f"[YAHOO] Fetching {sym} ({interval}->{yf_interval})...")
            ticker = yf.Ticker(sym)
            df = ticker.history(period=range_, interval=yf_interval, timeout=timeout)
            
            if df.empty:
                print(f"[YAHOO] {sym} returned empty data.")
                continue
                
            candles: List[Candle] = []
            for ts, row in df.iterrows():
                try:
                    # Strip timezone to ensure naive datetime for compatibility
                    dt_obj = ts.to_pydatetime().replace(tzinfo=None)
                    
                    c = Candle(
                        time=dt_obj,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                    )
                    candles.append(c)
                except Exception:
                    continue
            
            if not candles:
                print(f"[YAHOO] {sym} had data but failed to parse candles.")
                continue

            # Sort oldest first
            candles.sort(key=lambda x: x.time)
            if resample_minutes > 1:
                candles = _aggregate_candles(candles, resample_minutes)
            
            print(f"[YAHOO] Success: Got {len(candles)} candles from {sym}")
            if limit:
                return candles[-limit:]
            return candles

        except Exception as exc:
            print(f"[YAHOO] Failed to fetch {sym}: {exc}")
            continue

    print(f"[YAHOO] All candidates failed for {primary_sym}")
    return []


def fetch_yahoo_ltp(
    yahoo_symbol: str,
    *,
    timeout: float = 10.0,
) -> Optional[YahooQuote]:
    """Fetch latest price from Yahoo."""
    _check_enabled()
    candles = fetch_yahoo_candles(
        yahoo_symbol, 
        interval="1m", 
        range_="1d", 
        limit=5, 
        timeout=timeout
    )
    if not candles:
        return None
    
    last = candles[-1]
    return YahooQuote(
        symbol=yahoo_symbol,
        ltp=last.close,
        ts=last.time,
    )
