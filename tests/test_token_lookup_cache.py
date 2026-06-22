from __future__ import annotations

import mstock_client as mc


def test_failed_lookup_cache_honors_ttl():
    mc.clear_failed_lookup_cache()
    key = ("NIFTY", "NSE")
    mc.remember_failed_lookup(key, now_ts=0.0)
    assert mc.should_skip_failed_lookup(key, now_ts=30.0) is True
    assert mc.should_skip_failed_lookup(key, now_ts=61.0) is False


def test_failed_lookup_cache_evicts_oldest_entries():
    mc.clear_failed_lookup_cache()
    for idx in range(mc.FAILED_LOOKUP_MAX_KEYS + 5):
        mc.remember_failed_lookup((f"SYM{idx}", "NSE"), now_ts=float(idx))
    metrics = mc.get_failed_lookup_cache_metrics()
    assert metrics["failed_lookups_size"] == mc.FAILED_LOOKUP_MAX_KEYS
    assert metrics["cache_evictions"] == 5
