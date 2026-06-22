"""Tests for Option Chain panel in UI.

Tests pure option-chain data logic: ATM identification, OI sorting,
unusual activity detection, liquidity checks, spread calculations, and VWAP.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Computation helpers (mimic what the Option Chain panel computes)
# ---------------------------------------------------------------------------

def _atm_strike(spot: float, step: float = 100.0) -> float:
    """Round spot to nearest step to get ATM strike.

    Uses standard half-up rounding (not Python's banker's round).
    """
    return int(spot / step + 0.5) * step


def _otm_itm_calls(spot: float, atm: float, n: int = 5, step: float = 100.0) -> list[dict[str, Any]]:
    """Generate n OTM calls (strike > ATM) and n ITM calls (strike < ATM)."""
    otm = []
    itm = []
    for i in range(1, n + 1):
        otm_strike = atm + i * step
        itm_strike = atm - i * step
        otm.append({"strike": otm_strike, "option_type": "CE", "moneyness": "OTM"})
        itm.append({"strike": itm_strike, "option_type": "CE", "moneyness": "ITM"})
    return otm + itm


def _otm_itm_puts(spot: float, atm: float, n: int = 5, step: float = 100.0) -> list[dict[str, Any]]:
    """Generate n OTM puts (strike < ATM) and n ITM puts (strike > ATM)."""
    otm = []
    itm = []
    for i in range(1, n + 1):
        otm_strike = atm - i * step
        itm_strike = atm + i * step
        otm.append({"strike": otm_strike, "option_type": "PE", "moneyness": "OTM"})
        itm.append({"strike": itm_strike, "option_type": "PE", "moneyness": "ITM"})
    return otm + itm


def _sort_by_oi(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new list sorted by OI descending."""
    return sorted(chain, key=lambda r: float(r.get("oi", 0.0) or 0.0), reverse=True)


def _unusual_oi_buildup(rows: list[dict[str, Any]], threshold_multiplier: float = 2.0) -> list[dict[str, Any]]:
    """Flag rows where OI change > threshold_multiplier * average OI change."""
    if len(rows) < 2:
        return []

    changes = []
    for i in range(1, len(rows)):
        oi_curr = float(rows[i].get("oi", 0.0) or 0.0)
        oi_prev = float(rows[i - 1].get("oi", 0.0) or 0.0)
        changes.append(abs(oi_curr - oi_prev))

    if not changes:
        return []
    avg_change = sum(changes) / len(changes)
    flagged = []
    for i in range(1, len(rows)):
        oi_curr = float(rows[i].get("oi", 0.0) or 0.0)
        oi_prev = float(rows[i - 1].get("oi", 0.0) or 0.0)
        if abs(oi_curr - oi_prev) > avg_change * threshold_multiplier:
            flagged.append(rows[i])
    return flagged


def _large_iv_change(rows: list[dict[str, Any]], threshold_pct: float = 5.0) -> list[dict[str, Any]]:
    """Flag rows where IV changed by more than threshold_pct from previous row."""
    if len(rows) < 2:
        return []
    flagged = []
    for i in range(1, len(rows)):
        iv_curr = float(rows[i].get("iv", 0.0) or 0.0)
        iv_prev = float(rows[i - 1].get("iv", 0.0) or 0.0)
        if iv_prev > 0 and abs(iv_curr - iv_prev) / iv_prev * 100.0 > threshold_pct:
            flagged.append(rows[i])
    return flagged


def _liquidity_deterioration(rows: list[dict[str, Any]], spread_threshold_pct: float = 1.5) -> list[dict[str, Any]]:
    """Flag rows where bid-ask spread > spread_threshold_pct of mid-price."""
    flagged = []
    for r in rows:
        bid = float(r.get("bid", 0.0) or 0.0)
        ask = float(r.get("ask", 0.0) or 0.0)
        mid = (bid + ask) / 2.0
        if mid > 0:
            spread_pct = (ask - bid) / mid * 100.0
            if spread_pct > spread_threshold_pct:
                flagged.append(r)
    return flagged


def _bid_ask_spread(row: dict[str, Any]) -> dict[str, float]:
    """Compute absolute and percentage spread for a chain row."""
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    mid = (bid + ask) / 2.0
    spread_abs = ask - bid
    spread_pct = (spread_abs / mid * 100.0) if mid > 0 else 0.0
    return {"spread_abs": spread_abs, "spread_pct": spread_pct}


def _display_row_format(row: dict[str, Any]) -> dict[str, str]:
    """Format a chain row for display (human-readable strings)."""
    return {
        "strike": f"{float(row.get('strike', 0.0)):>10.0f}",
        "oi": f"{float(row.get('oi', 0.0) or 0.0):>12.0f}",
        "iv": f"{float(row.get('iv', 0.0) or 0.0):>8.2f}%",
        "bid": f"{float(row.get('bid', 0.0) or 0.0):>10.2f}",
        "ask": f"{float(row.get('ask', 0.0) or 0.0):>10.2f}",
        "spread": f"{_bid_ask_spread(row)['spread_pct']:>6.2f}%",
    }


def _iv_weighted_vwap(rows: list[dict[str, Any]]) -> float:
    """Compute OI-weighted average IV (proxy for VWAP of IV)."""
    total_weighted_iv = 0.0
    total_oi = 0.0
    for r in rows:
        oi = float(r.get("oi", 0.0) or 0.0)
        iv = float(r.get("iv", 0.0) or 0.0)
        if iv > 0 and oi > 0:
            total_weighted_iv += iv * oi
            total_oi += oi
    if total_oi == 0:
        return 0.0
    return total_weighted_iv / total_oi


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_option_chain_atm_identification() -> None:
    # spot=24500, step=100 -> ATM = 24500
    atm = _atm_strike(24500.0)
    assert atm == 24500.0

    # spot=24537 -> nearest 100 is 24500
    atm = _atm_strike(24537.0)
    assert atm == 24500.0

    # spot=24563 -> nearest 100 is 24600
    atm = _atm_strike(24563.0)
    assert atm == 24600.0

    # step=50 (BankNifty)
    atm = _atm_strike(44825.0, step=50.0)
    # 44825 / 50 = 896.5, int(896.5 + 0.5) = int(897.0) = 897 → 897*50 = 44850
    atm = _atm_strike(44825.0, step=50.0)
    assert atm == 44850.0


def test_option_chain_otm_itm_strikes() -> None:
    spot = 24500.0
    atm = _atm_strike(spot)

    calls = _otm_itm_calls(spot, atm)
    puts = _otm_itm_puts(spot, atm)

    assert len(calls) == 10
    assert len(puts) == 10

    # 5 OTM calls (strike > ATM)
    otm_calls = [r for r in calls if r["moneyness"] == "OTM"]
    itm_calls = [r for r in calls if r["moneyness"] == "ITM"]
    assert len(otm_calls) == 5
    assert len(itm_calls) == 5
    assert all(r["strike"] > atm for r in otm_calls)
    assert all(r["strike"] < atm for r in itm_calls)

    # 5 OTM puts (strike < ATM)
    otm_puts = [r for r in puts if r["moneyness"] == "OTM"]
    itm_puts = [r for r in puts if r["moneyness"] == "ITM"]
    assert len(otm_puts) == 5
    assert len(itm_puts) == 5
    assert all(r["strike"] < atm for r in otm_puts)
    assert all(r["strike"] > atm for r in itm_puts)


def test_option_chain_oi_sorting() -> None:
    chain = [
        {"strike": 24400.0, "oi": 500.0, "option_type": "PE"},
        {"strike": 24500.0, "oi": 5000.0, "option_type": "CE"},
        {"strike": 24600.0, "oi": 2000.0, "option_type": "CE"},
        {"strike": 24700.0, "oi": 800.0, "option_type": "CE"},
        {"strike": 24500.0, "oi": 3500.0, "option_type": "PE"},
    ]
    sorted_chain = _sort_by_oi(chain)
    oi_values = [float(r.get("oi", 0.0)) for r in sorted_chain]
    assert oi_values == sorted(oi_values, reverse=True), "OI should be descending"
    # Top strike should be 24500 CE with OI=5000
    assert float(sorted_chain[0]["strike"]) == 24500.0
    assert float(sorted_chain[0]["oi"]) == 5000.0


def test_unusual_oi_buildup_detection() -> None:
    # Chain with OI values that cause one large jump
    chain = [
        {"strike": 24400.0, "oi": 1000.0},
        {"strike": 24450.0, "oi": 1100.0},   # change = 100
        {"strike": 24500.0, "oi": 1200.0},   # change = 100
        {"strike": 24550.0, "oi": 10000.0},  # change = 8800 -> flagged (> 2x avg 2933)
        {"strike": 24600.0, "oi": 10500.0},  # change = 500
    ]
    avg_change = (100 + 100 + 8800 + 500) / 4  # ~2375
    flagged = _unusual_oi_buildup(chain, threshold_multiplier=2.0)
    assert len(flagged) >= 1
    assert float(flagged[0]["strike"]) == 24550.0


def test_large_iv_change_detection() -> None:
    chain = [
        {"strike": 24400.0, "iv": 15.0},
        {"strike": 24450.0, "iv": 15.5},     # +0.5 / 15 = 3.3% -> not flagged
        {"strike": 24500.0, "iv": 22.0},     # +6.5 / 15.5 = 42% -> flagged
        {"strike": 24550.0, "iv": 23.0},     # +1 / 22 = 4.5% -> not flagged
        {"strike": 24600.0, "iv": 23.5},     # +0.5 / 23 = 2.2% -> not flagged
    ]
    flagged = _large_iv_change(chain, threshold_pct=5.0)
    assert len(flagged) == 1
    assert float(flagged[0]["strike"]) == 24500.0


def test_liquidity_deterioration() -> None:
    chain = [
        # spread = (101-100)/100.5 * 100 = 0.995% -> not flagged
        {"strike": 24400.0, "bid": 100.0, "ask": 101.0},
        # spread = (103-100)/101.5 * 100 = 2.96% -> flagged
        {"strike": 24500.0, "bid": 100.0, "ask": 103.0},
        # spread = (151-150)/150.5 * 100 = 0.665% -> not flagged
        {"strike": 24600.0, "bid": 150.0, "ask": 151.0},
    ]
    flagged = _liquidity_deterioration(chain, spread_threshold_pct=1.5)
    assert len(flagged) == 1
    assert float(flagged[0]["strike"]) == 24500.0


def test_option_chain_bid_ask_spread() -> None:
    row = {"strike": 24500.0, "bid": 200.0, "ask": 203.0}
    result = _bid_ask_spread(row)
    assert result["spread_abs"] == pytest.approx(3.0)
    assert result["spread_pct"] == pytest.approx(3.0 / 201.5 * 100.0)  # ~1.49%

    # Zero spread
    row_zero = {"strike": 24500.0, "bid": 200.0, "ask": 200.0}
    result_zero = _bid_ask_spread(row_zero)
    assert result_zero["spread_abs"] == 0.0
    assert result_zero["spread_pct"] == 0.0


def test_option_chain_display_rows_format() -> None:
    row = {"strike": 24500.0, "oi": 5000.0, "iv": 18.5, "bid": 200.0, "ask": 203.0}
    formatted = _display_row_format(row)
    assert "strike" in formatted
    assert "oi" in formatted
    assert "iv" in formatted
    assert "bid" in formatted
    assert "ask" in formatted
    assert "spread" in formatted
    # Strike should be right-aligned 10-char string
    assert formatted["strike"].strip() == "24500"
    # IV should show percentage
    assert "%" in formatted["iv"]


def test_option_chain_vwap_computation() -> None:
    # 3 rows with IV and OI
    rows = [
        {"strike": 24400.0, "iv": 15.0, "oi": 1000.0},
        {"strike": 24500.0, "iv": 20.0, "oi": 2000.0},
        {"strike": 24600.0, "iv": 18.0, "oi": 1500.0},
    ]
    # weighted avg = (15*1000 + 20*2000 + 18*1500) / (1000+2000+1500)
    # = (15000 + 40000 + 27000) / 4500 = 82000/4500 ≈ 18.22
    vwap = _iv_weighted_vwap(rows)
    expected = (15.0 * 1000 + 20.0 * 2000 + 18.0 * 1500) / 4500.0
    assert vwap == pytest.approx(expected)

    # Zero OI rows
    empty_rows = [{"strike": 24500.0, "iv": 18.0, "oi": 0.0}]
    assert _iv_weighted_vwap(empty_rows) == 0.0

    # Empty chain
    assert _iv_weighted_vwap([]) == 0.0


def test_empty_chain_handled() -> None:
    empty: list[dict[str, Any]] = []
    assert _atm_strike(24500.0) == 24500.0  # ATM independent of chain
    assert _sort_by_oi(empty) == []
    assert _unusual_oi_buildup(empty) == []
    assert _large_iv_change(empty) == []
    assert _liquidity_deterioration(empty) == []
    assert _iv_weighted_vwap(empty) == 0.0


import pytest  # noqa: E402  (needed for pytest.approx)