"""
Production Watchdog & Self-Healing Module

Provides:
- Heartbeat monitoring with configurable timeouts
- Automatic detection of hangs, deadlocks, stalled API calls
- Stale market data detection
- Health endpoint (exposed via HTTP for external monitoring)
- Automatic restart coordination
- Process supervision

Design:
- Runs as a separate thread monitoring the main loop
- Uses a shared heartbeat timestamp that the main loop periodically updates
- If heartbeat is not updated within the timeout, triggers recovery
- Exposes a lightweight HTTP health endpoint for load balancer / K8s probes
"""

import os
import sys
import json
import time
import signal
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    """Serializable health status object."""
    status: str = "unknown"  # healthy | degraded | down
    uptime_seconds: float = 0.0
    last_heartbeat_ts: Optional[str] = None
    last_candle_ts: Optional[str] = None
    last_order_ts: Optional[str] = None
    active_positions: int = 0
    pending_orders: int = 0
    api_latency_ms: float = 0.0
    missed_candles_1m: int = 0
    consecutive_stalls: int = 0
    error_count_1h: int = 0
    memory_usage_pct: float = 0.0
    cpu_usage_pct: float = 0.0
    is_running: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatchdogConfig:
    """Configuration for the watchdog service."""
    heartbeat_timeout_sec: float = 30.0
    candle_stale_threshold_sec: float = 120.0
    api_call_timeout_sec: float = 15.0
    health_port: int = 9090
    health_host: str = "127.0.0.1"
    auto_restart: bool = True
    max_restarts_per_hour: int = 3
    restart_cooldown_sec: float = 30.0
    log_health_interval_sec: float = 60.0
    process_name: str = "NiftyScalper"


# ---------------------------------------------------------------------------
# Lightweight health HTTP server
# ---------------------------------------------------------------------------

class HealthEndpointHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that returns JSON health status."""

    watchdog_ref: Optional['Watchdog'] = None

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond_json(self.watchdog_ref.get_health_status() if self.watchdog_ref else {"status": "unknown"})
        elif self.path == "/health/live":
            self._respond_json({"status": "alive"})
        elif self.path == "/health/ready":
            status = self.watchdog_ref.get_health_status() if self.watchdog_ref else {"status": "unknown"}
            ready = status.get("status") in ("healthy", "degraded")
            self._respond_json({"status": "ready" if ready else "not_ready"}, 200 if ready else 503)
        elif self.path == "/metrics":
            status = self.watchdog_ref.get_health_status() if self.watchdog_ref else {}
            # Prometheus-style text output
            lines = [
                "# HELP nifty_scalper_status Current health status (1=healthy, 0=degraded, -1=down)",
                f"nifty_scalper_status {1 if status.get('status') == 'healthy' else (0 if status.get('status') == 'degraded' else -1)}",
                f"# HELP nifty_scalper_uptime_seconds Uptime in seconds",
                f"nifty_scalper_uptime_seconds {status.get('uptime_seconds', 0)}",
                f"# HELP nifty_scalper_active_positions Number of open positions",
                f"nifty_scalper_active_positions {status.get('active_positions', 0)}",
                f"# HELP nifty_scalper_pending_orders Number of pending orders",
                f"nifty_scalper_pending_orders {status.get('pending_orders', 0)}",
                f"# HELP nifty_scalper_missed_candles_1m Missed 1-min candles in current session",
                f"nifty_scalper_missed_candles_1m {status.get('missed_candles_1m', 0)}",
                f"# HELP nifty_scalper_errors_1h Errors in last hour",
                f"nifty_scalper_errors_1h {status.get('error_count_1h', 0)}",
                f"# HELP nifty_scalper_api_latency_ms API call latency in ms",
                f"nifty_scalper_api_latency_ms {status.get('api_latency_ms', 0)}",
            ]
            self._respond_text("\n".join(lines) + "\n")
        else:
            self._respond_json({"error": "not_found"}, 404)

    def _respond_json(self, data: dict, status_code: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_text(self, text: str, status_code: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress HTTP log spam; log at debug level."""
        logger.debug(format, *args)


# ---------------------------------------------------------------------------
# Watchdog service
# ---------------------------------------------------------------------------

class Watchdog:
    """
    Production watchdog for the scalper main loop.

    Monitors:
    - Heartbeat freshness (main loop must tick every N seconds)
    - Market data staleness (last candle timestamp)
    - API call responsiveness
    - Process health

    Actions:
    - Logs warnings at first sign of trouble
    - Attempts automatic recovery (restart) if configured
    - Exposes health endpoint for external monitoring
    """

    def __init__(
        self,
        config: Optional[WatchdogConfig] = None,
    ) -> None:
        self.config = config or WatchdogConfig()

        # State
        self._start_time: float = time.time()
        self._last_heartbeat: float = time.time()
        self._last_candle_ts: Optional[float] = None
        self._last_order_ts: Optional[float] = None
        self._stall_counter: int = 0
        self._restart_count: int = 0
        self._last_restart_time: float = 0.0
        self._error_count_1h: int = 0
        self._error_timestamps: list[float] = []
        self._missed_candles: int = 0
        self._api_latency_sum: float = 0.0
        self._api_call_count: int = 0
        self._is_running: bool = False
        self._is_shutting_down: bool = False
        self._lock = threading.Lock()

        # Callbacks
        self._on_stall_callbacks: list[Callable] = []
        self._on_critical_callbacks: list[Callable] = []
        self._state_provider: Optional[Callable[[], Dict[str, Any]]] = None

        # Health HTTP server
        self._http_server: Optional[HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None

        # Register signal handlers
        self._original_sigint: Any = None
        self._original_sigterm: Any = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def register_state_provider(self, fn: Callable[[], Dict[str, Any]]) -> None:
        """Register a callable that returns current bot state dict."""
        self._state_provider = fn

    def on_stall(self, callback: Callable) -> None:
        """Register callback invoked when a stall is detected."""
        self._on_stall_callbacks.append(callback)

    def on_critical(self, callback: Callable) -> None:
        """Register callback invoked on critical failure."""
        self._on_critical_callbacks.append(callback)

    def heartbeat(self) -> None:
        """Must be called by the main loop periodically (every iteration)."""
        with self._lock:
            self._last_heartbeat = time.time()

    def record_candle(self, ts: float) -> None:
        """Record the timestamp of the latest received candle."""
        with self._lock:
            self._last_candle_ts = ts

    def record_order(self) -> None:
        """Record that an order was placed."""
        with self._lock:
            self._last_order_ts = time.time()

    def record_api_latency(self, latency_ms: float) -> None:
        """Record API call latency for metrics."""
        with self._lock:
            self._api_latency_sum += latency_ms
            self._api_call_count += 1

    def record_error(self) -> None:
        """Record an error occurrence."""
        with self._lock:
            now = time.time()
            self._error_count_1h += 1
            self._error_timestamps.append(now)
            # Prune errors older than 1 hour
            cutoff = now - 3600
            self._error_timestamps = [t for t in self._error_timestamps if t > cutoff]
            self._error_count_1h = len(self._error_timestamps)

    def record_missed_candle(self) -> None:
        """Increment missed candle counter."""
        with self._lock:
            self._missed_candles += 1

    def get_health_status(self) -> Dict[str, Any]:
        """Return current health status as a serializable dict."""
        with self._lock:
            now = time.time()
            uptime = now - self._start_time
            hb_age = now - self._last_heartbeat

            # Determine status
            status = "healthy"
            details = {}

            if hb_age > self.config.heartbeat_timeout_sec * 2:
                status = "down"
                details["heartbeat_age"] = round(hb_age, 1)
            elif hb_age > self.config.heartbeat_timeout_sec:
                status = "degraded"
                details["heartbeat_age"] = round(hb_age, 1)

            if self._last_candle_ts is not None:
                candle_age = now - self._last_candle_ts
                if candle_age > self.config.candle_stale_threshold_sec * 2:
                    status = "degraded"
                    details["candle_age"] = round(candle_age, 1)
                elif candle_age > self.config.candle_stale_threshold_sec:
                    if status == "healthy":
                        status = "degraded"
                    details["candle_age_warning"] = round(candle_age, 1)

            if self._error_count_1h > 50:
                status = "degraded"
                details["error_rate_high"] = self._error_count_1h

            if self._stall_counter >= 3:
                status = "down"
                details["stall_count"] = self._stall_counter

            avg_latency = 0.0
            if self._api_call_count > 0:
                avg_latency = self._api_latency_sum / self._api_call_count

            return {
                "status": status,
                "uptime_seconds": round(uptime, 1),
                "last_heartbeat_ts": datetime.fromtimestamp(self._last_heartbeat, tz=timezone.utc).isoformat(),
                "last_heartbeat_age_sec": round(hb_age, 1),
                "last_candle_ts": datetime.fromtimestamp(self._last_candle_ts, tz=timezone.utc).isoformat() if self._last_candle_ts else None,
                "last_order_ts": datetime.fromtimestamp(self._last_order_ts, tz=timezone.utc).isoformat() if self._last_order_ts else None,
                "active_positions": 0,
                "pending_orders": 0,
                "api_latency_ms": round(avg_latency, 1),
                "missed_candles_1m": self._missed_candles,
                "consecutive_stalls": self._stall_counter,
                "error_count_1h": self._error_count_1h,
                "memory_usage_pct": self._get_memory_usage(),
                "cpu_usage_pct": self._get_cpu_usage(),
                "is_running": self._is_running,
                "restart_count": self._restart_count,
                "details": details,
            }

    def start(self) -> None:
        """Start the watchdog monitoring thread and health HTTP server."""
        if self._is_running:
            return
        self._is_running = True
        self._start_time = time.time()
        self._last_heartbeat = time.time()

        # Start monitor thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="watchdog-monitor")
        self._monitor_thread.start()

        # Start health HTTP server
        self._start_http_server()

        # Register signal handlers
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Not in main thread; signals can't be set
            pass

        logger.info(
            "Watchdog started | heartbeat_timeout=%.1fs | candle_stale=%.1fs | health_port=%d",
            self.config.heartbeat_timeout_sec,
            self.config.candle_stale_threshold_sec,
            self.config.health_port,
        )

    def stop(self) -> None:
        """Stop the watchdog gracefully."""
        self._is_running = False
        self._stop_http_server()
        logger.info("Watchdog stopped")

    def request_shutdown(self) -> None:
        """Request graceful shutdown from the main loop."""
        self._is_shutting_down = True
        logger.warning("Watchdog requested shutdown")

    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down

    @property
    def is_running(self) -> bool:
        return self._is_running

    # -----------------------------------------------------------------------
    # Internal: monitoring loop
    # -----------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background thread that periodically checks health."""
        last_log_time = 0.0

        while self._is_running:
            time.sleep(5.0)  # Check every 5 seconds
            if not self._is_running:
                break

            now = time.time()
            with self._lock:
                hb_age = now - self._last_heartbeat
                candle_age = now - self._last_candle_ts if self._last_candle_ts else None

            # Check for stall
            if hb_age > self.config.heartbeat_timeout_sec:
                self._stall_counter += 1
                logger.warning(
                    "Heartbeat stall detected | age=%.1fs | counter=%d",
                    hb_age, self._stall_counter,
                )
                for cb in self._on_stall_callbacks:
                    try:
                        cb()
                    except Exception as e:
                        logger.error("Stall callback failed: %s", e)

                # Critical: multiple consecutive stalls
                if self._stall_counter >= 3:
                    logger.critical(
                        "CRITICAL: %d consecutive stalls. %s",
                        self._stall_counter,
                        "Attempting auto-restart..." if self.config.auto_restart else "Manual intervention required.",
                    )
                    for cb in self._on_critical_callbacks:
                        try:
                            cb()
                        except Exception as e:
                            logger.error("Critical callback failed: %s", e)

                    if self.config.auto_restart:
                        self._attempt_restart()
            else:
                # Reset stall counter on healthy heartbeat
                if self._stall_counter > 0:
                    self._stall_counter = 0

            # Check candle staleness
            if candle_age is not None and candle_age > self.config.candle_stale_threshold_sec:
                logger.warning(
                    "Stale market data | last_candle_age=%.1fs | threshold=%.1fs",
                    candle_age, self.config.candle_stale_threshold_sec,
                )

            # Periodic health log
            if now - last_log_time > self.config.log_health_interval_sec:
                last_log_time = now
                status = self.get_health_status()
                logger.info(
                    "Health | status=%s | uptime=%.1fs | hb_age=%.1fs | stalls=%d | errors_1h=%d | missed_candles=%d",
                    status["status"],
                    status["uptime_seconds"],
                    status["last_heartbeat_age_sec"],
                    status["consecutive_stalls"],
                    status["error_count_1h"],
                    status["missed_candles_1m"],
                )

    def _attempt_restart(self) -> None:
        """Attempt to restart the bot process."""
        now = time.time()
        # Cooldown check
        if now - self._last_restart_time < self.config.restart_cooldown_sec:
            logger.warning("Restart cooldown active, skipping restart")
            return

        # Rate limit check
        one_hour_ago = now - 3600
        # Track restarts in a simple way: just count and time
        if self._restart_count >= self.config.max_restarts_per_hour:
            logger.critical(
                "Max restarts per hour reached (%d). Manual intervention required.",
                self.config.max_restarts_per_hour,
            )
            return

        self._restart_count += 1
        self._last_restart_time = now
        logger.warning("Attempting restart #%d...", self._restart_count)

        # In production, this would:
        # 1. Save state to database
        # 2. Signal the main loop to stop
        # 3. Wait for graceful shutdown
        # 4. Restart the process
        #
        # For now, log the intent and signal shutdown
        self._is_shutting_down = True

        # Force restart via subprocess (platform-specific)
        try:
            python = sys.executable
            script = sys.argv[0]
            args = sys.argv[1:]
            logger.info("Restarting: %s %s %s", python, script, " ".join(args))
            subprocess.Popen([python, script] + args)
            sys.exit(0)
        except Exception as e:
            logger.error("Restart failed: %s", e)

    # -----------------------------------------------------------------------
    # Internal: health HTTP server
    # -----------------------------------------------------------------------

    def _start_http_server(self) -> None:
        """Start the lightweight health HTTP server in a daemon thread."""
        try:
            HealthEndpointHandler.watchdog_ref = self
            self._http_server = HTTPServer(
                (self.config.health_host, self.config.health_port),
                HealthEndpointHandler,
            )
            self._http_thread = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True,
                name="watchdog-http",
            )
            self._http_thread.start()
            logger.info("Health endpoint listening on http://%s:%d", self.config.health_host, self.config.health_port)
        except OSError as e:
            logger.warning("Could not start health HTTP server on port %d: %s", self.config.health_port, e)

    def _stop_http_server(self) -> None:
        """Stop the health HTTP server."""
        if self._http_server:
            try:
                self._http_server.shutdown()
            except Exception:
                pass
            self._http_server = None

    # -----------------------------------------------------------------------
    # Internal: signal handling
    # -----------------------------------------------------------------------

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        signame = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.warning("Received %s, initiating graceful shutdown...", signame)
        self._is_shutting_down = True

        # Restore original handlers to ensure subsequent signals kill the process
        try:
            signal.signal(signal.SIGINT, self._original_sigint or signal.default_int_handler)
            signal.signal(signal.SIGTERM, self._original_sigterm or signal.SIG_DFL)
        except ValueError:
            pass

    # -----------------------------------------------------------------------
    # Internal: system metrics
    # -----------------------------------------------------------------------

    def _get_memory_usage(self) -> float:
        """Get current process memory usage as percentage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_percent()
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current process CPU usage as percentage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0


# ---------------------------------------------------------------------------
# Convenience: create and start watchdog, integrate with main loop
# ---------------------------------------------------------------------------

def create_default_watchdog() -> Watchdog:
    """Create a Watchdog with production defaults from environment variables."""
    import os as _os
    config = WatchdogConfig(
        heartbeat_timeout_sec=float(_os.getenv("WATCHDOG_HEARTBEAT_TIMEOUT", "30.0")),
        candle_stale_threshold_sec=float(_os.getenv("WATCHDOG_CANDLE_STALE_THRESHOLD", "120.0")),
        health_port=int(_os.getenv("WATCHDOG_HEALTH_PORT", "9090")),
        health_host=_os.getenv("WATCHDOG_HEALTH_HOST", "127.0.0.1"),
        auto_restart=_os.getenv("WATCHDOG_AUTO_RESTART", "true").lower() == "true",
        max_restarts_per_hour=int(_os.getenv("WATCHDOG_MAX_RESTARTS", "3")),
    )
    return Watchdog(config)