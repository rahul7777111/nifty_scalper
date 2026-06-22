"""
performance_monitor.py
======================
Tracks RAM, CPU, queue depths, and latencies (tick, chart, GPT).
Logs every 60 seconds.
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
import psutil
from typing import Any, Dict

logger = logging.getLogger("performance_monitor")
logger.setLevel(logging.INFO)

# Global metrics
latest_tick_latency: float = 0.0
latest_chart_latency: float = 0.0
latest_gpt_latency: float = 0.0
latest_candle_delay: float = 0.0
latest_model_prediction_latency: float = 0.0
registered_queues: Dict[str, Any] = {}
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "runtime_performance.jsonl"

def register_queue(name: str, q: Any) -> None:
    """Register a queue to monitor its depth."""
    registered_queues[name] = q

def update_tick_latency(val: float) -> None:
    """Update latest tick processing latency (in seconds)."""
    global latest_tick_latency
    latest_tick_latency = val

def update_chart_latency(val: float) -> None:
    """Update latest chart rendering latency (in seconds)."""
    global latest_chart_latency
    latest_chart_latency = val

def update_gpt_latency(val: float) -> None:
    """Update latest GPT advice request latency (in seconds)."""
    global latest_gpt_latency
    latest_gpt_latency = val

def update_candle_delay(val: float) -> None:
    global latest_candle_delay
    latest_candle_delay = val

def update_model_prediction_latency(val: float) -> None:
    global latest_model_prediction_latency
    latest_model_prediction_latency = val

def get_token_cache_metrics() -> Dict[str, int]:
    """Dynamically get token cache metrics from mstock_client."""
    try:
        import mstock_client
        if hasattr(mstock_client, "get_failed_lookup_cache_metrics"):
            return dict(mstock_client.get_failed_lookup_cache_metrics())
        return {
            "lookup_success": getattr(mstock_client, "lookup_success", 0),
            "lookup_failure": getattr(mstock_client, "lookup_failure", 0),
            "cache_hits": getattr(mstock_client, "cache_hits", 0),
            "cache_evictions": getattr(mstock_client, "cache_evictions", 0),
            "failed_lookups_size": len(getattr(mstock_client, "FAILED_LOOKUPS", {}))
        }
    except Exception:
        return {}

def monitor_loop() -> None:
    """Telemetry logging loop running every 60 seconds."""
    process = psutil.Process(os.getpid())
    while True:
        try:
            # RAM in MB
            ram_mb = process.memory_info().rss / (1024 * 1024)
            # CPU percentage (non-blocking call)
            cpu_pct = process.cpu_percent(interval=None)
            
            # Queue depths
            q_depths = {}
            for name, q in list(registered_queues.items()):
                try:
                    q_depths[name] = q.qsize()
                except Exception:
                    pass
            
            # Token Cache metrics
            cache_metrics = get_token_cache_metrics()
            
            # Log metrics
            payload = {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "rss_mb": round(float(ram_mb), 2),
                "cpu_percent": round(float(cpu_pct), 2),
                "queue_sizes": q_depths,
                "tick_latency_sec": float(latest_tick_latency),
                "chart_latency_sec": float(latest_chart_latency),
                "gpt_latency_sec": float(latest_gpt_latency),
                "candle_delay_sec": float(latest_candle_delay),
                "model_prediction_latency_sec": float(latest_model_prediction_latency),
                "token_cache": cache_metrics,
            }
            logger.info(
                f"[TELEMETRY] RAM: {ram_mb:.2f} MB | CPU: {cpu_pct:.1f}% | "
                f"Queues: {q_depths} | Tick: {latest_tick_latency:.4f}s | "
                f"Chart: {latest_chart_latency:.4f}s | GPT: {latest_gpt_latency:.4f}s | "
                f"Candle Delay: {latest_candle_delay:.4f}s | Predict: {latest_model_prediction_latency:.4f}s | "
                f"Token Cache: {cache_metrics}"
            )
            if any(size > 100 for size in q_depths.values()):
                logger.warning("[TELEMETRY] Queue backlog warning: %s", q_depths)
            try:
                REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with REPORT_PATH.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[TELEMETRY] Error in monitor loop: {e}")
        time.sleep(60.0)

def start_performance_monitor() -> None:
    """Start the performance monitor in a background daemon thread."""
    t = threading.Thread(target=monitor_loop, daemon=True)
    t.start()
    logger.info("Performance monitor telemetry thread started.")
