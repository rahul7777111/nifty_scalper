#!/usr/bin/env python3
"""Apply all profit enhancement config defaults and add new feature params."""

import re

with open('src/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    # ---- Phase 1: Toggle features ON ----
    'enable_roc_filter: bool = False': 'enable_roc_filter: bool = True',
    'enable_choppiness_filter: bool = False': 'enable_choppiness_filter: bool = True',
    'enable_delta_strike_selection: bool = False': 'enable_delta_strike_selection: bool = True',
    'enable_theta_decay_filter: bool = False': 'enable_theta_decay_filter: bool = True',
    'choppiness_hard_filter: bool = False': 'choppiness_hard_filter: bool = True',
    'enable_rsi_confluence: bool = False': 'enable_rsi_confluence: bool = True',
    'dir_allow_tie_break_entries: bool = False': 'dir_allow_tie_break_entries: bool = True',
    'dir_flip_long_to_short_on_stop: bool = False': 'dir_flip_long_to_short_on_stop: bool = True',

    # ---- Phase 1: Set non-boolean defaults ----
    'entry_require_bid_ask: bool = False': 'entry_require_bid_ask: bool = True',
    'entry_spread_shock_mult: float = 0.0': 'entry_spread_shock_mult: float = 2.5',
    'max_consecutive_same_trade_type: int = 0': 'max_consecutive_same_trade_type: int = 2',
    'cooldown_atr_mult: float = 0.0': 'cooldown_atr_mult: float = 0.25',
    'max_portfolio_delta_abs: float = 0.0': 'max_portfolio_delta_abs: float = 500.0',
    'max_portfolio_option_notional: float = 0.0': 'max_portfolio_option_notional: float = 500000.0',
    "delta_hedge_scope: str = 'strategy_only'": "delta_hedge_scope: str = 'multi_only'",
    'premium_entry_cutoff_hhmm: str = "': 'premium_entry_cutoff_hhmm: str = "14:00"  #',  # will be replaced more precisely below
}

# Fix premium_entry_cutoff_hhmm precisely since it has empty string default
old_premium_cutoff = '    premium_entry_cutoff_hhmm: str = ""  # do not open new premium trades after this time'
new_premium_cutoff = '    premium_entry_cutoff_hhmm: str = "14:00"  # do not open new premium trades after this time'
content = content.replace(old_premium_cutoff, new_premium_cutoff)

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"  Applied: {old} -> {new}")
    else:
        print(f"  NOT FOUND (skipping): {old}")

# ---- Fix the separate delta_hedge_scope with double-quote variant ----
# Check both single and double quote variants
content = content.replace('delta_hedge_scope: str = "strategy_only"', 'delta_hedge_scope: str = "multi_only"')

# ---- Phase 2: Add new config params for new features ----
# Find the right insertion point - after the existing # ---- Risk Management Enhancement Settings ---- section
insertion_marker = "max_portfolio_delta_abs: float = 0.0"
new_params = """
    # ---- IV Percentile Position Sizing ----
    # When enabled, position sizes are scaled inversely to IV percentile.
    # High IV => smaller size, Low IV => full size.
    iv_sizing_enabled: bool = True
    # Number of past days to use for IV percentile calculation.
    iv_percentile_period: int = 252
    # Below this IV percentile, use full configured size.
    iv_sizing_min_pct: float = 50.0
    # Above this IV percentile, scale down to iv_sizing_factor_at_max.
    iv_sizing_max_pct: float = 80.0
    # At max IV percentile, multiply size by this factor.
    iv_sizing_factor_at_max: float = 0.5
    # IV percentile at which scaling starts (between min_pct and max_pct, linear interpolation).
    iv_sizing_factor_min: float = 1.0

    # ---- Session-Specific Exit Adjustments ----
    # When enabled, exit parameters (SL, trail, target) vary by market session.
    session_exit_enabled: bool = True
    # Opening session (first 45 min): tighter stops to avoid whipsaw
    session_exit_sl_atr_mult_open: float = 1.0
    session_exit_trail_atr_mult_open: float = 0.5
    session_exit_tp_atr_mult_open: float = 2.0
    # Mid session (core hours): standard parameters
    session_exit_sl_atr_mult_mid: float = 1.5
    session_exit_trail_atr_mult_mid: float = 0.8
    session_exit_tp_atr_mult_mid: float = 2.5
    # Closing session (last 90 min): moderate to avoid late-day reversals
    session_exit_sl_atr_mult_close: float = 1.2
    session_exit_trail_atr_mult_close: float = 0.6
    session_exit_tp_atr_mult_close: float = 2.0

    # ---- Strategy Win-Rate Tracker ----
    # When enabled, tracks win/loss per strategy and auto-disables losers.
    winrate_tracker_enabled: bool = True
    # Max consecutive losses before auto-disabling a strategy.
    winrate_tracker_max_losses: int = 3
    # After disabling, re-enable after this many consecutive wins on other strategies.
    winrate_tracker_recovery_wins: int = 1
    # Rolling lookback for win-rate display (not for disable logic).
    winrate_tracker_lookback: int = 20
"""

# Find the end of the config class - insert before the closing blank line before _default_credential_path
# Actually, let's just add it right after the max_portfolio_delta_abs line
if content.count(insertion_marker) == 1:
    content = content.replace(insertion_marker, insertion_marker + new_params)
    print("  Added new feature config params")
else:
    print(f"  ERROR: insertion marker '{insertion_marker}' not found or ambiguous")

with open('src/config.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("\nDone! config.py updated.")
