from __future__ import annotations

from typing import Sequence

from market_data import Candle


def _body_size(c: Candle) -> float:
    return abs(c.close - c.open)


def _upper_shadow(c: Candle) -> float:
    return c.high - max(c.close, c.open)


def _lower_shadow(c: Candle) -> float:
    return min(c.close, c.open) - c.low


def _is_bullish(c: Candle) -> bool:
    return c.close > c.open


def _is_bearish(c: Candle) -> bool:
    return c.close < c.open


def _is_small_body(c: Candle, threshold: float = 0.35) -> bool:
    full_range = c.high - c.low
    if full_range <= 0:
        return False
    return (_body_size(c) / full_range) <= threshold


def _is_uptrend_before_last(candles: Sequence[Candle], lookback: int = 3) -> bool:
    if len(candles) < lookback + 1:
        return False
    seg = candles[-(lookback + 1) : -1]
    if len(seg) < 2:
        return False
    bullish_count = sum(1 for c in seg if _is_bullish(c))
    return seg[-1].close > seg[0].close and bullish_count >= max(1, lookback - 1)


def _is_downtrend_before_last(candles: Sequence[Candle], lookback: int = 3) -> bool:
    if len(candles) < lookback + 1:
        return False
    seg = candles[-(lookback + 1) : -1]
    if len(seg) < 2:
        return False
    bearish_count = sum(1 for c in seg if _is_bearish(c))
    return seg[-1].close < seg[0].close and bearish_count >= max(1, lookback - 1)


def _is_hammer_shape(c: Candle) -> bool:
    body = _body_size(c)
    lower = _lower_shadow(c)
    upper = _upper_shadow(c)
    if body == 0:
        return False
    return lower >= 2 * body and upper <= body


def _is_inverted_hammer_shape(c: Candle) -> bool:
    body = _body_size(c)
    lower = _lower_shadow(c)
    upper = _upper_shadow(c)
    if body == 0:
        return False
    return upper >= 2 * body and lower <= body


def is_bullish_engulfing(candles: Sequence[Candle]) -> bool:
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    prev_body = prev.close - prev.open
    curr_body = curr.close - curr.open
    if prev_body >= 0 or curr_body <= 0:
        return False
    return curr.open <= prev.close and curr.close >= prev.open


def is_bearish_engulfing(candles: Sequence[Candle]) -> bool:
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    prev_body = prev.close - prev.open
    curr_body = curr.close - curr.open
    if prev_body <= 0 or curr_body >= 0:
        return False
    return curr.open >= prev.close and curr.close <= prev.open


def is_hammer(candles: Sequence[Candle]) -> bool:
    if not candles:
        return False
    return _is_hammer_shape(candles[-1]) and _is_downtrend_before_last(candles)


def is_hanging_man(candles: Sequence[Candle]) -> bool:
    if not candles:
        return False
    return _is_hammer_shape(candles[-1]) and _is_uptrend_before_last(candles)


def is_inverted_hammer(candles: Sequence[Candle]) -> bool:
    if not candles:
        return False
    return _is_inverted_hammer_shape(candles[-1]) and _is_downtrend_before_last(candles)


def is_shooting_star(candles: Sequence[Candle]) -> bool:
    if not candles:
        return False
    return _is_inverted_hammer_shape(candles[-1]) and _is_uptrend_before_last(candles)


def is_piercing_line(candles: Sequence[Candle]) -> bool:
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not (_is_bearish(prev) and _is_bullish(curr)):
        return False
    midpoint_prev = (prev.open + prev.close) / 2.0
    return (
        curr.open < prev.close
        and curr.close > midpoint_prev
        and curr.close < prev.open
    )


def is_dark_cloud_cover(candles: Sequence[Candle]) -> bool:
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not (_is_bullish(prev) and _is_bearish(curr)):
        return False
    midpoint_prev = (prev.open + prev.close) / 2.0
    return (
        curr.open > prev.close
        and curr.close < midpoint_prev
        and curr.close > prev.open
    )


def is_morning_star(candles: Sequence[Candle]) -> bool:
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not (_is_bearish(c1) and _is_small_body(c2) and _is_bullish(c3)):
        return False
    midpoint_c1 = (c1.open + c1.close) / 2.0
    return c3.close > midpoint_c1 and c2.low <= min(c1.close, c3.open)


def is_evening_star(candles: Sequence[Candle]) -> bool:
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not (_is_bullish(c1) and _is_small_body(c2) and _is_bearish(c3)):
        return False
    midpoint_c1 = (c1.open + c1.close) / 2.0
    return c3.close < midpoint_c1 and c2.high >= max(c1.close, c3.open)


def is_three_white_soldiers(candles: Sequence[Candle]) -> bool:
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not (_is_bullish(c1) and _is_bullish(c2) and _is_bullish(c3)):
        return False
    return (
        c2.close > c1.close
        and c3.close > c2.close
        and min(c1.open, c1.close) <= c2.open <= max(c1.open, c1.close)
        and min(c2.open, c2.close) <= c3.open <= max(c2.open, c2.close)
    )


def is_three_black_crows(candles: Sequence[Candle]) -> bool:
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not (_is_bearish(c1) and _is_bearish(c2) and _is_bearish(c3)):
        return False
    return (
        c2.close < c1.close
        and c3.close < c2.close
        and min(c1.open, c1.close) <= c2.open <= max(c1.open, c1.close)
        and min(c2.open, c2.close) <= c3.open <= max(c2.open, c2.close)
    )


def is_rising_three_methods(candles: Sequence[Candle]) -> bool:
    if len(candles) < 5:
        return False
    c1, m1, m2, m3, c5 = candles[-5], candles[-4], candles[-3], candles[-2], candles[-1]
    if not (_is_bullish(c1) and _is_bullish(c5)):
        return False
    middle = (m1, m2, m3)
    if not all(_is_bearish(c) for c in middle):
        return False
    if not all(c.high <= c1.high and c.low >= c1.low for c in middle):
        return False
    return c5.close > c1.close and c5.open >= c1.open


def is_falling_three_methods(candles: Sequence[Candle]) -> bool:
    if len(candles) < 5:
        return False
    c1, m1, m2, m3, c5 = candles[-5], candles[-4], candles[-3], candles[-2], candles[-1]
    if not (_is_bearish(c1) and _is_bearish(c5)):
        return False
    middle = (m1, m2, m3)
    if not all(_is_bullish(c) for c in middle):
        return False
    if not all(c.high <= c1.high and c.low >= c1.low for c in middle):
        return False
    return c5.close < c1.close and c5.open <= c1.open


def is_doji(candles: Sequence[Candle], threshold: float = 0.1) -> bool:
    if not candles:
        return False
    c = candles[-1]
    full_range = c.high - c.low
    if full_range == 0:
        return False
    body = _body_size(c)
    return (body / full_range) <= threshold
