from __future__ import annotations

import gpt_advisor as ga


def reset_gpt_state():
    ga.circuit_breaker = ga.GPTCircuitBreaker(max_failures=3, cooldown_seconds=600.0)
    ga._cached_market_analysis = None
    ga._cached_trade_advice = None


def test_gpt_circuit_breaker_opens_after_three_failures(monkeypatch):
    reset_gpt_state()
    ga._cached_market_analysis = ga.GPTMarketAnalysis(ce_pe_bias="CE", reason="cached", confidence=0.8)
    call_counter = {"count": 0}

    def fail_market(**kwargs):
        call_counter["count"] += 1
        raise RuntimeError("HTTP 504 gateway timeout")

    monkeypatch.setattr(ga, "_analyze_market_raw", fail_market)

    for _ in range(3):
        result = ga.analyze_market(snapshot={}, model="gpt-4o-mini", api_key="test-key")
        assert result.ce_pe_bias == "CE"

    assert ga.circuit_breaker.is_available() is False
    calls_before_open_guard = call_counter["count"]
    result = ga.analyze_market(snapshot={}, model="gpt-4o-mini", api_key="test-key")
    assert result.ce_pe_bias == "CE"
    assert call_counter["count"] == calls_before_open_guard


def test_gpt_circuit_breaker_success_resets_failure_count(monkeypatch):
    reset_gpt_state()

    def success_market(**kwargs):
        return ga.GPTMarketAnalysis(ce_pe_bias="PE", reason="ok", confidence=0.7)

    monkeypatch.setattr(ga, "_analyze_market_raw", success_market)
    result = ga.analyze_market(snapshot={}, model="gpt-4o-mini", api_key="test-key")
    assert result.ce_pe_bias == "PE"
    assert ga.circuit_breaker.failure_count == 0
    assert ga.circuit_breaker.is_available() is True
