import os
import sys
from pathlib import Path

# Ensure repo `src/` is importable as a top-level module path.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_api_config, load_strategy_config  # noqa: E402
from indicators import atr, rsi  # noqa: E402
from mstock_client import MStockTypeBClient  # noqa: E402


def main() -> None:
    api_cfg = load_api_config()
    strat_cfg = load_strategy_config()

    print("underlying:", strat_cfg.underlying, "timeframe:", strat_cfg.timeframe)
    print("MSTOCK_USE_INTRADAY_CHART=", os.getenv("MSTOCK_USE_INTRADAY_CHART"))
    print("MSTOCK_INTRADAY_ONLY=", os.getenv("MSTOCK_INTRADAY_ONLY"))
    print("MSTOCK_UNDERLYING_TOKEN=", os.getenv("MSTOCK_UNDERLYING_TOKEN"))

    client = MStockTypeBClient(api_cfg)
    candles = client.get_candles(symbol=strat_cfg.underlying, timeframe=strat_cfg.timeframe, limit=150)

    print("candles:", len(candles))
    if not candles:
        return

    candles = sorted(candles, key=lambda c: c.time)

    last = candles[-1]
    print("last:", last.time, float(last.open), float(last.high), float(last.low), float(last.close), "range=", float(last.high) - float(last.low))

    tail = candles[-25:]
    tail_ranges = [float(c.high) - float(c.low) for c in tail]
    print("tail range min/max:", min(tail_ranges), max(tail_ranges))

    closes = [float(c.close) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]

    atr_p = int(getattr(strat_cfg, "atr_period", 14) or 14)
    atr_val = atr(highs, lows, closes, period=atr_p)
    rsi_val = rsi(closes, period=14)
    print("ATR(p=%d):" % atr_p, atr_val)
    print("RSI(p=14):", rsi_val)


if __name__ == "__main__":
    main()
