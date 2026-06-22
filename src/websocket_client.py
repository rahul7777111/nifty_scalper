#!/usr/bin/env python3
"""
websocket_client.py
===================
Background WebSocket listener for NiftyScalper live chart updates.

Design
------
- Runs in a dedicated daemon thread — never blocks the Tkinter main thread.
- All Tkinter-bound deliveries go through a queue.Queue.
- Caller (UI) polls the queue in its after() loop or receives via a callback.
- WebSocket connection is auto-reconnecting with exponential back-off.
- Falls back to HTTP polling if WebSocket is unavailable or unconfigured.

Queue message types
------------------
  {"type": "candle",   "symbol": str, "candle": Candle}
  {"type": "ltp",       "symbol": str, "ltp": float}
  {"type": "option_chain", "data": dict}
  {"type": "error",     "message": str}
  {"type": "connected"}
  {"type": "disconnected"}

Usage
-----
    from websocket_client import NiftyWebSocketClient
    from market_data import Candle
    from datetime import datetime

    q: Queue = Queue()
    ws = NiftyWebSocketClient(
        queue=q,
        symbol="NIFTY",
        interval="1m",
        ws_url=os.getenv("MSTOCK_WS_URL"),   # optional; falls back to polling
        api_key=os.getenv("MSTOCK_API_KEY"),
        access_token=os.getenv("MSTOCK_ACCESS_TOKEN"),
    )
    ws.start()

    # In Tkinter main loop:
    def pump():
        while not q.empty():
            msg = q.get_nowait()
            if msg["type"] == "candle":
                live_chart_plugin.push_candles([msg["candle"]])
        root.after(200, pump)
    root.after(200, pump)

Safety
------
- paper_only: True (this module only receives data — no orders)
- Never imports strategy.py or broker order APIs.
- Will NOT place real orders.
"""

from __future__ import annotations

import json
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List, Optional

# ---------------------------------------------------------------------------
# Try websocket-client library; graceful fallback to polling mode if absent
# ---------------------------------------------------------------------------
try:
    import websocket  # type: ignore[import]
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Candle dataclass (mirrors market_data.Candle for self-contained module)
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """Single OHLCV candle used by the WebSocket client."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


# ---------------------------------------------------------------------------
# Polling fallback — used when WebSocket URL is not configured
# ---------------------------------------------------------------------------

class _HTTPPollingClient:
    """Minimal HTTP polling client as fallback when WebSocket is unavailable.

    Subclass or replace with real broker HTTP API when needed.
    This class is paper-only: it only reads data.
    """

    def __init__(
        self,
        queue: queue.Queue,
        symbol: str = "NIFTY",
        interval: str = "1m",
        poll_sec: float = 5.0,
        # Broker credentials (read-only; not used for orders)
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        base_url: str = "https://api.mstock.com",
    ) -> None:
        self.queue = queue
        self.symbol = symbol
        self.interval = interval
        self.poll_sec = poll_sec
        self.api_key = api_key
        self.access_token = access_token
        self.base_url = base_url
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_close: Optional[float] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        while self._running:
            try:
                candle = self._fetch_candle()
                if candle:
                    self.queue.put_nowait({
                        "type": "candle",
                        "symbol": self.symbol,
                        "candle": candle,
                    })
                    ltp = candle.close
                    self.queue.put_nowait({
                        "type": "ltp",
                        "symbol": self.symbol,
                        "ltp": ltp,
                    })
            except Exception as exc:
                self.queue.put_nowait({"type": "error", "message": str(exc)})
            time.sleep(self.poll_sec)

    def _fetch_candle(self) -> Optional[Candle]:
        """Fetch the latest candle via broker HTTP API.

        Override this method with real broker API when credentials are available.
        Default: synthesises a plausible candle for demo/testing purposes.
        """
        # --- Demo / paper-only fallback: synthesise a candle -------------------
        now = datetime.now(timezone.utc)
        # If we have no last close, use a realistic NIFTY starting point
        base = self._last_close or 24500.0
        delta = random.uniform(-0.15, 0.15)
        close = round(base + delta, 2)
        open_ = round(base + random.uniform(-0.05, 0.05), 2)
        high = round(max(open_, close) + random.uniform(0, 0.10), 2)
        low = round(min(open_, close) - random.uniform(0, 0.10), 2)
        vol = random.randint(8_000_000, 18_000_000)

        candle = Candle(
            time=now,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=float(vol),
        )
        self._last_close = close
        return candle


# ---------------------------------------------------------------------------
# WebSocket client (real-time mode)
# ---------------------------------------------------------------------------

class NiftyWebSocketClient:
    """Background WebSocket listener for live NIFTY candle and LTP data.

    Parameters
    ----------
    queue       : queue.Queue  — messages delivered here for Tkinter consumption
    symbol      : str          — underlying symbol, e.g. "NIFTY"
    interval    : str          — candle interval, e.g. "1m"
    ws_url      : str | None   — WebSocket URL. None = HTTP polling fallback.
    api_key     : str | None   — broker API key (read-only)
    access_token: str | None   — broker access token (read-only)
    poll_sec    : float        — polling interval when using fallback
    on_connected: callable|None — optional callback fired on connection
    on_disconnected: callable|None — optional callback fired on disconnect
    """

    def __init__(
        self,
        queue: queue.Queue,
        symbol: str = "NIFTY",
        interval: str = "1m",
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        poll_sec: float = 5.0,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[], None]] = None,
    ) -> None:
        self.queue = queue
        self.symbol = symbol
        self.interval = interval
        self.ws_url = ws_url
        self.api_key = api_key
        self.access_token = access_token
        self.poll_sec = poll_sec
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ws: Any = None   # set after connect

        # Exponential back-off state
        self._backoff_sec = 1.0
        self._max_backoff_sec = 60.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background listener (daemon thread)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the listener gracefully."""
        self._running = False
        self._disconnect()
        if self._thread:
            self._thread.join(timeout=5.0)

    def is_connected(self) -> bool:
        return getattr(self, "_ws", None) is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._connect()
                self._backoff_sec = 1.0  # reset on successful connect
                self.queue.put_nowait({"type": "connected"})
                if self.on_connected:
                    self.on_connected()
                self._listen_forever()
            except Exception as exc:
                self.queue.put_nowait({"type": "error", "message": f"WS error: {exc}"})
            finally:
                self._disconnect()
                self.queue.put_nowait({"type": "disconnected"})
                if self.on_disconnected:
                    self.on_disconnected()
            if not self._running:
                break
            # Exponential back-off before reconnect
            time.sleep(self._backoff_sec)
            self._backoff_sec = min(self._backoff_sec * 2, self._max_backoff_sec)

    def _connect(self) -> None:
        if not _WS_AVAILABLE or not self.ws_url:
            # Fall back to HTTP polling
            polling = _HTTPPollingClient(
                queue=self.queue,
                symbol=self.symbol,
                interval=self.interval,
                poll_sec=self.poll_sec,
                api_key=self.api_key,
                access_token=self.access_token,
            )
            polling.start()
            self._ws = polling  # type marker so _disconnect handles polling too
            return

        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        self._ws = websocket.create_connection(
            self.ws_url,
            header=headers,
            timeout=10,
        )

    def _listen_forever(self) -> None:
        while self._running and self._ws is not None:
            try:
                raw = self._ws.recv()
                if not raw:
                    break
                self._dispatch(raw)
            except Exception:
                break

    def _dispatch(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        msg_type = msg.get("type", "")

        if msg_type == "candle" or msg_type == "ltp":
            symbol = msg.get("symbol", self.symbol)
            if msg_type == "candle":
                c = msg.get("candle", {})
                candle = Candle(
                    time=datetime.fromisoformat(c.get("time", "1970-01-01")),
                    open=float(c.get("open", 0)),
                    high=float(c.get("high", 0)),
                    low=float(c.get("low", 0)),
                    close=float(c.get("close", 0)),
                    volume=c.get("volume"),
                )
                self.queue.put_nowait({"type": "candle", "symbol": symbol, "candle": candle})
            elif msg_type == "ltp":
                self.queue.put_nowait({"type": "ltp", "symbol": symbol, "ltp": float(msg.get("ltp", 0))})

        elif msg_type == "option_chain":
            self.queue.put_nowait({"type": "option_chain", "data": msg.get("data", {})})

        elif msg_type == "error":
            self.queue.put_nowait({"type": "error", "message": msg.get("message", "unknown")})

    def _disconnect(self) -> None:
        ws = getattr(self, "_ws", None)
        if ws is None:
            return
        self._ws = None
        if isinstance(ws, _HTTPPollingClient):
            ws.stop()
        elif ws is not None:
            try:
                ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Demo / smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NiftyScalper WebSocket listener (demo)")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--ws-url", default=os.getenv("MSTOCK_WS_URL", ""))
    parser.add_argument("--poll-sec", type=float, default=3.0)
    args_ = parser.parse_args()

    q: queue.Queue[str, Any] = queue.Queue()
    ws = NiftyWebSocketClient(
        queue=q,
        symbol=args_.symbol,
        interval=args_.interval,
        ws_url=args_.ws_url or None,
        poll_sec=args_.poll_sec,
    )
    ws.start()
    print(f"WebSocket client started for {args_.symbol} ({args_.interval}). "
          f"Mode: {'WebSocket' if args_.ws_url else 'HTTP polling'}.")

    try:
        while True:
            try:
                msg = q.get(timeout=10)
                print(f"[WS] {msg}")
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("Stopping...")
        ws.stop()