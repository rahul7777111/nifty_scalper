from __future__ import annotations

import math
from typing import Optional, Sequence


def _validate_period(values: Sequence[float], period: int) -> bool:
    return period > 0 and len(values) >= period


def sma(values: Sequence[float], period: int) -> Optional[float]:
    """Simple moving average of the last `period` values.

    Returns None if there is not enough data.
    """
    if not _validate_period(values, period):
        return None
    window = values[-period:]
    return sum(window) / float(period)


def ema(values: Sequence[float], period: int) -> Optional[float]:
    """Exponential moving average of the last `period` values.

    Uses the standard smoothing factor 2 / (period + 1).
    Returns None if there is not enough data.
    """
    if not _validate_period(values, period):
        return None

    k = 2.0 / (period + 1.0)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1.0 - k)
    return ema_val


def rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index (RSI).

    Returns None if there is not enough data.
    """
    if not _validate_period(values, period + 1):  # need period+1 closes
        return None

    gains: list[float] = []
    losses: list[float] = []
    for prev, curr in zip(values[-(period + 1) : -1], values[-period:]):
        change = curr - prev
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Optional[float]:
    """Average True Range (ATR).

    Returns None if there is not enough data.
    """
    if period <= 0:
        return None
    if not (len(highs) == len(lows) == len(closes)):
        return None
    if len(closes) < period + 1:
        return None

    trs: list[float] = []
    for i in range(len(closes) - period, len(closes)):
        high = float(highs[i])
        low = float(lows[i])
        prev_close = float(closes[i - 1])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    return sum(trs) / float(period)


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Optional[float]:
    """Average Directional Index (ADX) using Wilder's Smoothing.

    Returns None if there is not enough data.
    Requires at least 2 * period + 1 candles to stabilize.
    """
    if period <= 0:
        return None
    if not (len(highs) == len(lows) == len(closes)):
        return None
    
    # We need enough data for the initial SMA + subsequent RMAs.
    if len(closes) < 2 * period:
        return None

    # 1. Calculate TR, +DM, -DM
    trs = []
    plus_dms = []
    minus_dms = []

    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        prev_c = closes[i-1]
        prev_h = highs[i-1]
        prev_l = lows[i-1]

        # True Range
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        
        # Directional Movement
        up_move = h - prev_h
        down_move = prev_l - l

        if up_move > down_move and up_move > 0:
            plus_dm = up_move
        else:
            plus_dm = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm = down_move
        else:
            minus_dm = 0.0

        trs.append(tr)
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)

    # Wilder's Smoothing Helper (RMA)
    def wilders_smooth(values: list[float], n: int) -> list[float]:
        if len(values) < n:
            return []
        
        smoothed = []
        # Initial SMA
        curr_smooth = sum(values[:n]) / n
        smoothed.append(curr_smooth)

        # Subsequent RMA
        for i in range(n, len(values)):
            curr_smooth = ((curr_smooth * (n - 1)) + values[i]) / n
            smoothed.append(curr_smooth)
        
        return smoothed

    # 2. Smooth TR, +DM, -DM
    tr_smooth = wilders_smooth(trs, period)
    plus_dm_smooth = wilders_smooth(plus_dms, period)
    minus_dm_smooth = wilders_smooth(minus_dms, period)

    if not tr_smooth:
        return None

    # 3. Calculate +DI, -DI, and DX
    dx_values = []
    for i in range(len(tr_smooth)):
        tr_val = tr_smooth[i]
        if tr_val == 0:
            dx_values.append(0.0)
            continue
            
        plus_di = (plus_dm_smooth[i] / tr_val) * 100.0
        minus_di = (minus_dm_smooth[i] / tr_val) * 100.0
        
        sum_di = plus_di + minus_di
        if sum_di == 0:
            dx = 0.0
        else:
            dx = (abs(plus_di - minus_di) / sum_di) * 100.0
        dx_values.append(dx)

    # 4. Smooth DX to get ADX
    adx_smooth = wilders_smooth(dx_values, period)

    if not adx_smooth:
        return None

    return adx_smooth[-1]


def supertrend(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> Optional[float]:
    """Calculate Supertrend indicator.

    Returns the Supertrend value for the last candle.
    Returns None if there is not enough data.
    """
    if period <= 0 or multiplier <= 0:
        return None
    if not (len(highs) == len(lows) == len(closes)):
        return None
    if len(closes) < period + 1:
        return None

    # 1. Compute True Range series
    tr_vals = [0.0] * len(closes)
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr_vals[i] = max(h - l, abs(h - pc), abs(l - pc))
    
    # 2. Compute ATR (Wilder's smoothing)
    atr_vals = [0.0] * len(closes)
    if len(closes) > period:
        atr_vals[period] = sum(tr_vals[1:period+1]) / period
        for i in range(period + 1, len(closes)):
            atr_vals[i] = (atr_vals[i-1] * (period - 1) + tr_vals[i]) / period

    if len(closes) <= period:
        return None

    # 3. Compute Supertrend
    upper_band = [0.0] * len(closes)
    lower_band = [0.0] * len(closes)
    supertrend_vals = [0.0] * len(closes)
    trend = [1] * len(closes)  # 1 = Uptrend, -1 = Downtrend

    first_valid = period
    hl2 = (highs[first_valid] + lows[first_valid]) / 2
    curr_atr = atr_vals[first_valid]
    upper_band[first_valid] = hl2 + (multiplier * curr_atr)
    lower_band[first_valid] = hl2 - (multiplier * curr_atr)
    supertrend_vals[first_valid] = upper_band[first_valid]

    for i in range(first_valid + 1, len(closes)):
        hl2 = (highs[i] + lows[i]) / 2
        curr_atr = atr_vals[i]
        
        # Basic Bands
        ub = hl2 + (multiplier * curr_atr)
        lb = hl2 - (multiplier * curr_atr)
        
        # Final Bands
        prev_ub = upper_band[i-1]
        prev_lb = lower_band[i-1]
        prev_close = closes[i-1]
        
        if ub < prev_ub or prev_close > prev_ub:
            upper_band[i] = ub
        else:
            upper_band[i] = prev_ub
            
        if lb > prev_lb or prev_close < prev_lb:
            lower_band[i] = lb
        else:
            lower_band[i] = prev_lb
            
        # Trend Direction
        prev_trend = trend[i-1]
        curr_close = closes[i]
        
        if prev_trend == 1:
            if curr_close < lower_band[i]:
                trend[i] = -1
                supertrend_vals[i] = upper_band[i]
            else:
                trend[i] = 1
                supertrend_vals[i] = lower_band[i]
        else:
            if curr_close > upper_band[i]:
                trend[i] = 1
                supertrend_vals[i] = lower_band[i]
            else:
                trend[i] = -1
                supertrend_vals[i] = upper_band[i]

    return supertrend_vals[-1]


def roc(values: Sequence[float], period: int = 14) -> Optional[float]:
    """Rate of Change (ROC).

    (Current Price - Price n periods ago) / Price n periods ago * 100
    """
    if not _validate_period(values, period + 1):
        return None
    
    current = values[-1]
    prev = values[-(period + 1)]
    if prev == 0:
        return 0.0
    return ((current - prev) / prev) * 100.0


def choppiness_index(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Optional[float]:
    """Choppiness Index.

    Values range from 0 to 100.
    Lower values = Strong Trend, Higher values = Sideways/Choppy.
    Standard thresholds: > 61.8 (Choppy), < 38.2 (Trending).
    """
    if not (len(highs) == len(lows) == len(closes)):
        return None
    if len(closes) < period + 1:
        return None

    # 1. Sum of True Range over the period
    tr_sum = 0.0
    for i in range(len(closes) - period, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_sum += tr

    if tr_sum <= 0:
        return 0.0

    # 2. Maximum High and Minimum Low over the period
    window_highs = highs[-period:]
    window_lows = lows[-period:]
    max_h = max(window_highs)
    min_l = min(window_lows)
    
    range_dist = max_h - min_l
    if range_dist <= 0:
        return 100.0

    # 3. Choppiness Formula
    # 100 * LOG10( SUM(TR, n) / (MAX(HIGH, n) - MIN(LOW, n)) ) / LOG10(n)
    return 100.0 * math.log10(tr_sum / range_dist) / math.log10(period)


def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculate Classic Pivot Points and Support/Resistance levels.
    """
    pp = (high + low + close) / 3.0
    r1 = (2.0 * pp) - low
    s1 = (2.0 * pp) - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2.0 * (pp - low)
    s3 = low - 2.0 * (high - pp)
    
    return {
        "pp": pp,
        "r1": r1, "s1": s1,
        "r2": r2, "s2": s2,
        "r3": r3, "s3": s3
    }
