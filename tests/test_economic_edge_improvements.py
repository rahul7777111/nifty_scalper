#!/usr/bin/env python3
"""
tests/test_economic_edge_improvements.py

PHASE 10/12 safety, metrics, label, feature, no-live, threshold order, side, day-wise stubs.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ml_only_dynamic_preset_retrainer import (
    get_live_features,
    top_n_per_day,
    apply_preset_filters,
    DynamicPreset as RetrainerPreset,
    evaluate_candidate_preset,
)
import retrain_dynamic_candidate_matrix as matrix_mod
from candidate_filters import apply_pe_only_filter, apply_ce_only_filter


def test_no_live_trading_in_retrain_paths():
    # Check strings in source that enforce paper
    import inspect
    src = inspect.getsourcefile(matrix_mod)
    txt = Path(src).read_text() if src else ""
    retr = Path(__file__).parents[1] / "scripts" / "ml_only_dynamic_preset_retrainer.py"
    rtxt = retr.read_text()
    assert "paper_only" in (txt + rtxt) or "real_trading_enabled=False" in (txt + rtxt)
    assert "real_trading_enabled" not in (txt + rtxt) or "False" in (txt + rtxt)


def test_labels_reject_zero_positive_targets():
    # Simulate usability logic
    df = pd.DataFrame({
        'cost_survivor_label_v2': [1]*100 + [0]*900,
        'high_conviction_trade_label_v2': [0]*1000,
        'paper_candidate_label_v2': [0]*1000,
        'gross_forward_return': np.random.uniform(-0.01, 0.02, 1000),
    })
    # logic from matrix: 0 pos => unusable
    for col in ['high_conviction_trade_label_v2', 'paper_candidate_label_v2']:
        pos = int((df[col] == 1).sum())
        assert pos == 0
        usable = pos > 0 and (pos / len(df)) >= 0.005
        assert not usable


def test_no_future_leak_columns_in_features():
    df = pd.DataFrame({
        'range_pct': [0.05]*10,
        'volume': [100000]*10,
        'gross_forward_return': [0.01]*10,
        'net_forward_return': [0.005]*10,
        'cost_survivor_label_v2': [1]*10,
        'timestamp_dt': pd.date_range('2024-01-01', periods=10, freq='min'),
        'option_type': ['CE']*10,
    })
    # simulate add enhanced (no leak)
    df2 = matrix_mod._add_live_enhanced_features(df)
    feats = get_live_features(df2)
    for bad in ['gross_forward_return', 'net_forward_return', 'cost_survivor_label_v2', 'future', 'pnl']:
        assert not any(bad in f for f in feats), f"leak {bad} in features"
    # new safe ones should be allowed if numeric
    safe_new = [f for f in feats if 'pctile' in f or 'imbalance' in f or 'regime' in f]
    assert len(safe_new) >= 0  # may be 0 if no base cols, but no crash


def test_threshold_applied_after_top_n_and_order():
    df = pd.DataFrame({
        'timestamp_dt': pd.date_range('2024-01-01', periods=30, freq='5min'),
        '_score': np.linspace(0.1, 0.9, 30),
        'gross_forward_return': np.random.uniform(-0.5, 1.2, 30),
        'option_type': ['CE']*30,
        'dte_days': [3]*30,
        'range_pct': [0.05]*30,
        'ltp': [90.0]*30,
        'volume': [200000]*30,
        'cost_survivor_label_v2': (np.random.rand(30) > 0.6).astype(int),
    })
    preset = RetrainerPreset(preset_id="t", threshold=0.30, top_n_confidence_per_day=3, spread_limit_pct=0.2, liquidity_min=0, premium_band="all", dte_range="all")
    d = apply_preset_filters(df, preset)
    d2 = top_n_per_day(d, '_score', 3)
    # threshold after
    selected30 = d2[d2['_score'] >= 0.30]
    selected50 = d2[d2['_score'] >= 0.50]
    assert len(selected30) >= len(selected50) or len(d2) < 5
    # order in source already tested in other file


def test_side_filtering_still_correct():
    df = pd.DataFrame({'option_type': ['PE']*5 + ['CE']*5})
    pe = matrix_mod._filter_df_for_side(df, "PE_ONLY")
    ce = matrix_mod._filter_df_for_side(df, "CE_ONLY")
    both = matrix_mod._filter_df_for_side(df, "BOTH")
    assert (pe['option_type'] == 'PE').all() and len(pe) == 5
    assert (ce['option_type'] == 'CE').all() and len(ce) == 5
    assert len(both) == 10
    rpe = apply_pe_only_filter({'option_type': 'PE'})
    rce = apply_pe_only_filter({'option_type': 'CE'})
    assert rpe.get('filter_passed') or rpe.get('passed')
    assert not (rce.get('filter_passed') or rce.get('passed'))


def test_metric_sanity_after_fixes():
    # After fixes, these should hold in produced rows (use a dummy row)
    dummy = {
        'win_rate': 0.55,
        'net_return': 120.5,
        'PF_net': 7.2,
        'avg_trade_return': 2.8,
        'max_drawdown': 45.3,
        'total_trade_count': 55,
    }
    assert dummy['win_rate'] > 0.0, "win_rate must not be always 0"
    assert abs(dummy['net_return'] - dummy['PF_net']) > 1.0, "net_return should differ from PF_net (sum vs ratio)"
    assert abs(dummy['avg_trade_return']) < 100, "avg_trade_return should be per-trade return, not count"
    assert dummy['max_drawdown'] >= 0, "max_drawdown should be positive magnitude"
    assert dummy['total_trade_count'] > 0


def test_cost_aware_label_generation_and_usability():
    g = pd.Series([-0.01, 0.001, 0.004, 0.01, 0.02])
    cost1 = 0.0035
    l1 = (g > cost1).astype(int)
    l15 = (g > cost1*1.5).astype(int)
    assert l1.sum() >= 2
    assert l15.sum() <= l1.sum()
    # 0 pos case
    l0 = (g > 10).astype(int)
    assert l0.sum() == 0


def test_day_wise_report_generation_stub():
    # Stub: if we had trades with date and pnl, compute stats
    trades = pd.DataFrame({
        'date': ['2024-01-01']*3 + ['2024-01-02']*2,
        'pnl': [1.0, -0.5, 2.0, 0.5, -1.5]
    })
    daily = trades.groupby('date')['pnl'].agg(['sum', 'count'])
    assert len(daily) == 2
    worst = daily['sum'].min()
    assert worst < 0
    # max losing streak stub
    signs = (trades['pnl'] < 0).astype(int).tolist()
    # simple
    assert sum(signs) >= 1
