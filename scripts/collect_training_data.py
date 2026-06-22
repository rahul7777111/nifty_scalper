"""
collect_training_data.py
========================
Fetches historical 1-minute NIFTY candles from m.Stock and saves them to the
``data/`` directory as JSON files ready for ML training.

Run manually (or via Task Scheduler after market close):

    .venv\\Scripts\\python.exe scripts\\collect_training_data.py

Environment variables (from .env):
    MSTOCK_API_KEY        - API key
    MSTOCK_ACCESS_TOKEN   - Valid JWT session token
    COLLECT_DAYS_BACK     - Number of trading days to fetch (default 65)
    COLLECT_SYMBOL_TOKEN  - Symbol token (default 26000 = NIFTY 50)
    COLLECT_EXCHANGE      - Exchange (default NSE)

Output:
    data/candles_YYYYMMDD.json   - One file per trading date
    data/collection_log.json     - Metadata about last collection run
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add src/ to path so project modules are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "src"
DATA_DIR  = REPO_ROOT / "data"
sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".scalper.env")
except ImportError:
    pass  # dotenv optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("collect_training_data")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DAYS_BACK     = int(os.getenv("COLLECT_DAYS_BACK",     "65"))   # ~3 months
SYMBOL_TOKEN  = str(os.getenv("COLLECT_SYMBOL_TOKEN",  "26000"))
EXCHANGE      = str(os.getenv("COLLECT_EXCHANGE",       "NSE"))
INTERVAL      = "ONE_MINUTE"
SESSION_START = "09:15"
SESSION_END   = "15:30"

MIN_CANDLES_PER_DAY = 200   # reject partial sessions below this
MAX_CANDLES_PER_DAY = 380   # sanity cap


def is_trading_day(d: date) -> bool:
    """Returns True if d is a weekday (basic check — NSE holidays not enumerated)."""
    return d.weekday() < 5


def last_completed_trading_day() -> date:
    d = date.today() - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_between(start: date, end: date) -> list[date]:
    """Returns all weekdays in [start, end] inclusive."""
    days = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def build_client():
    from mstock_client import APIConfig, MStockTypeBClient
    cfg = APIConfig(
        base_url=os.getenv("MSTOCK_BASE_URL", "https://api.mstock.trade"),
        api_key=os.getenv("MSTOCK_API_KEY", "").strip(),
        api_secret=os.getenv("MSTOCK_API_SECRET", "").strip(),
        client_id=os.getenv("MSTOCK_CLIENT_ID", "").strip(),
    )
    return MStockTypeBClient(cfg)


def fetch_day_candles(client, trading_date: date) -> list[dict]:
    """Fetch 1-minute candles for a single completed trading day."""
    from_str = trading_date.strftime(f"%Y-%m-%d {SESSION_START}")
    to_str   = trading_date.strftime(f"%Y-%m-%d {SESSION_END}")
    try:
        candles = client.get_historical_candles(
            symbol_token=SYMBOL_TOKEN,
            exchange=EXCHANGE,
            interval=INTERVAL,
            from_date=from_str,
            to_date=to_str,
        )
        if not candles:
            return []
        result = []
        for c in candles:
            result.append({
                "time":   str(getattr(c, "time",   "")),
                "open":   float(getattr(c, "open",  0.0) or 0.0),
                "high":   float(getattr(c, "high",  0.0) or 0.0),
                "low":    float(getattr(c, "low",   0.0) or 0.0),
                "close":  float(getattr(c, "close", 0.0) or 0.0),
                "volume": float(getattr(c, "volume",0.0) or 0.0),
            })
        return result
    except Exception as exc:
        log.warning("fetch_day_candles(%s) failed: %s", trading_date, exc)
        return []


def save_day_candles(candles: list[dict], trading_date: date) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"candles_{trading_date.strftime('%Y%m%d')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"date": str(trading_date), "candles": candles}, f, indent=2)
    return out_path


def load_all_candles() -> list[dict]:
    """Load all candle files from data/ and return a flat sorted list."""
    all_candles = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            all_candles.extend(data.get("candles") or [])
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
    return all_candles


def summarize_collection() -> dict:
    """Return a summary of all collected candle files."""
    files = sorted(DATA_DIR.glob("candles_*.json"))
    total_candles = 0
    dates = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            n = len(data.get("candles") or [])
            total_candles += n
            dates.append(data.get("date", ""))
        except Exception:
            pass
    return {
        "files":         len(files),
        "trading_days":  len(dates),
        "total_candles": total_candles,
        "date_range":    f"{min(dates)} to {max(dates)}" if dates else "none",
        "ready_to_train": total_candles >= 6250,  # minimum 25 sessions × 250 bars
    }


def run():
    log.info("=" * 60)
    log.info("NiftyScalper — Historical Training Data Collector")
    log.info("=" * 60)

    api_key = os.getenv("MSTOCK_API_KEY", "").strip()
    access_token = os.getenv("MSTOCK_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        log.error("MSTOCK_API_KEY and MSTOCK_ACCESS_TOKEN must be set in .env")
        sys.exit(1)

    t_end   = last_completed_trading_day()
    t_start = t_end - timedelta(days=int(DAYS_BACK * 7 / 5) + 10)  # pad for weekends
    all_days = trading_days_between(t_start, t_end)[-DAYS_BACK:]

    log.info("Target: %d trading days (%s to %s)", len(all_days), all_days[0], all_days[-1])

    # Skip days we already have
    existing = {
        p.stem.replace("candles_", "")
        for p in DATA_DIR.glob("candles_*.json")
    }
    days_to_fetch = [d for d in all_days if d.strftime("%Y%m%d") not in existing]
    log.info("Already cached: %d days. Fetching: %d new days.",
             len(all_days) - len(days_to_fetch), len(days_to_fetch))

    if not days_to_fetch:
        log.info("All days already cached. Nothing to fetch.")
    else:
        try:
            client = build_client()
        except Exception as exc:
            log.error("Failed to build API client: %s", exc)
            sys.exit(1)

        fetched_ok = 0
        for i, trading_date in enumerate(days_to_fetch, 1):
            log.info("[%d/%d] Fetching %s ...", i, len(days_to_fetch), trading_date)
            candles = fetch_day_candles(client, trading_date)

            if len(candles) < MIN_CANDLES_PER_DAY:
                log.warning("  Skipped %s: only %d candles (holiday or API gap)",
                            trading_date, len(candles))
                continue

            if len(candles) > MAX_CANDLES_PER_DAY:
                candles = candles[:MAX_CANDLES_PER_DAY]

            out_path = save_day_candles(candles, trading_date)
            log.info("  Saved %d candles -> %s", len(candles), out_path.name)
            fetched_ok += 1

            # Polite rate-limiting (avoid hitting API too fast)
            if i < len(days_to_fetch):
                time.sleep(0.6)

        log.info("Fetched %d new days.", fetched_ok)

    summary = summarize_collection()
    log.info("")
    log.info("Collection Summary:")
    log.info("  Files:         %d", summary["files"])
    log.info("  Trading days:  %d", summary["trading_days"])
    log.info("  Total candles: %d", summary["total_candles"])
    log.info("  Date range:    %s", summary["date_range"])
    log.info("  Ready to train: %s", summary["ready_to_train"])

    if not summary["ready_to_train"]:
        log.warning(
            "Insufficient data: need >= 6,250 candles (25 sessions). "
            "Currently have %d. Run again tomorrow to accumulate more.",
            summary["total_candles"],
        )
    else:
        log.info("Data collection COMPLETE. Ready to run post_market_retrain.py")

    # Write collection log
    log_path = DATA_DIR / "collection_log.json"
    summary["last_run"] = datetime.utcnow().isoformat() + "Z"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run()
