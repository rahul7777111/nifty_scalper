# Bid/Ask Execution Fix Report

**Generated:** 2026-06-07 16:38:59  
**Branch:** research_v2  
**Status:** Complete

---

## Summary

Implemented realistic paper execution pricing using real bid/ask prices. Paper trades now execute at proper market-side prices (ask for BUY entries, bid for SELL/short entries) and block gracefully when bid/ask data is unavailable.

---

## Changes Made

### 1. `src/config.py` — New config options (lines 641-649)

Added three new `StrategyConfig` fields:

```python
# ---- Paper Execution Realism ----
paper_use_bid_ask_execution: bool = True    # Default True (realistic)
paper_allow_ltp_fallback: bool = False      # Default False (safety)
paper_ltp_fallback_spread_pct: float = 0.05 # 5% penalty if fallback used
```

Added env loading in `load_strategy_config()` for:
- `MSTOCK_PAPER_USE_BID_ASK_EXECUTION`
- `MSTOCK_PAPER_ALLOW_LTP_FALLBACK`
- `MSTOCK_PAPER_LTP_FALLBACK_SPREAD_PCT`

### 2. `src/strategy.py` — `_open_directional_from_option()` (lines 7932-7982)

**Before:** Used `_try_get_ltp_for_leg()` which returns mid/LTP price — unrealistic for paper simulation.

**After:** Fetches real bid/ask via `client.get_bid_ask()` and applies correct side-aware pricing:

| Position Type | Action | Price Used | Rationale |
|---|---|---|---|
| BUY (long) | Entry | **ask** | Buyer pays seller's asking price |
| SELL (short) | Entry | **bid** | Short seller receives buyer's bid |
| BUY (close short) | Exit | **ask** | To close short you buy back → pay ask |
| SELL (close long) | Exit | **bid** | To close long you sell → receive bid |

**Missing bid/ask handling:**
- If `paper_allow_ltp_fallback=False` (default): **BLOCK** trade, log `[PAPER][BLOCKED]`
- If `paper_allow_ltp_fallback=True`: use LTP with ±spread penalty, log `[PAPER][LTP FALLBACK]`

Added `execution_price_source` to trade metadata (values: `"ask"`, `"bid"`, `"ltp"`, `"ltp_fallback"`, `"live"`).

### 3. `tests/test_paper_engine_realism.py` — 11 tests, all passing

| Test | Purpose |
|---|---|
| `test_long_buy_entry_uses_ask` | BUY entry → ask price ✓ |
| `test_short_sell_entry_uses_bid` | SELL entry → bid price ✓ |
| `test_long_sell_exit_uses_bid` | Close BUY (SELL) → bid ✓ |
| `test_short_buy_exit_uses_ask` | Close SELL (BUY) → ask ✓ |
| `test_missing_bid_ask_blocks_trade_by_default` | No fallback → BLOCKED ✓ |
| `test_ltp_fallback_only_when_explicitly_enabled` | Fallback → LTP±penalty ✓ |
| `test_missing_bid_ask_does_not_block_when_fallback_enabled` | Fallback enabled → no block ✓ |
| `test_paper_use_bid_ask_false_uses_ltp` | Flag=False → LTP used ✓ |
| `test_config_defaults_for_paper_execution` | Defaults correct ✓ |
| `test_paper_entry_blocks_on_premium_filter_failure` | Premium filter works ✓ |
| `test_short_entry_ltp_fallback_applies_positive_penalty` | SELL fallback adds penalty ✓ |

---

## Close Path (already correct, no changes needed)

Verified that `_close_directional_leg()` at line ~6285 already implements correct bid/ask logic:
- Closing BUY → `bid_raw` required; uses `bid_raw` as exit price
- Closing SELL → `ask_raw` required; uses `ask_raw` as exit price
- Missing price blocks with `[PAPER][BLOCKED]` log

---

## Backward Compatibility

- `paper_use_bid_ask_execution=True` by default — existing users get more realistic paper execution immediately
- Set `MSTOCK_PAPER_ALLOW_LTP_FALLBACK=1` to restore old LTP-based behavior with spread penalty

---

## Files Changed

1. `src/config.py` — config fields + env loading (~20 lines)
2. `src/strategy.py` — `_open_directional_from_option()` bid/ask block (~50 lines)
3. `tests/test_paper_engine_realism.py` — 11 unit tests (new file)
4. `reports/bid_ask_execution_fix_20260607_163859.md` — this report
5. `reports/bid_ask_execution_fix_20260607_163859.json` — JSON summary