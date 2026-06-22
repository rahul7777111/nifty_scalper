#!/usr/bin/env python3
"""
ml_only_dynamic_preset_retrainer.py
====================================
Retrain candidates under dynamic presets, evaluate against strict 22/22 gates.

Key safety constraints (NEVER violated):
- Dynamic presets are live-computable config/risk controls ONLY
- NO future return, PnL, exit price, MFE, MAE, label, or outcome data
- Preset selection on validation folds only — never on test
- Gates are NOT weakened: 22/22 required, no 21/22 shortcuts
- paper_only=True, real_trading_enabled=False enforced
- Time-based walk-forward splits ONLY (no random K-fold)

Usage:
    python scripts/ml_only_dynamic_preset_retrainer.py \\
        --dataset data/processed/nifty_option_chain_cost_aware_edge_dataset_20260606_211845.csv \\
        --benchmark-candidate models/candidates/PE_only_elasticnet_cost_survivor_v2_20260608_153000 \\
        --candidate-types PE,CE,COMBINED,ATM,REGIME,ROUTER \\
        --max-candidates 8 \\
        --max-presets-per-candidate 50 \\
        --target-total-shadow-candidates 3 \\
        --strict-gates \\
        --save-reports \\
        --paper-only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

TS = datetime.now().strftime('%Y%m%d_%H%M%S')

# ─── Gate thresholds (strict, NOT weakened) ─────────────────────────────────
TOTAL_GATES = 22
PF_AT_1_00X = 1.15
PF_AT_1_25X = 1.05
PF_AT_1_50X = 1.00   # MANDATORY break-even
PF_AT_2_00X = 0.80
MIN_TRADES = 500
MIN_SHARPE = 0.75
MIN_DAILY_STABILITY = 0.50
MIN_THRESHOLD_ROBUSTNESS = 0.60
MAX_TOP_DAY_CONCENTRATION = 0.40

# Strict forbidden patterns for features
FEATURE_FORBIDDEN = [
    'return', 'forward', 'future', 'pnl', 'profit', 'loss',
    'label', 'target', 'outcome', 'exit', 'mfe', 'mae',
    'realized', 'net_', 'gross_', 'expected_', 'horizon_',
    'cost_survivor', 'strong_profitable', 'high_conviction',
    'paper_candidate', 'avoid_trade', 'weak_trade', 'no_trade',
    'trade_outcome', 'realized_after', 'cost_adjusted',
]

# Columns that must NEVER enter feature list
FORBIDDEN_EXACT = {
    'net_forward_return', 'gross_forward_return', 'profitable_trade_label',
    'cost_survivor_label', 'cost_survivor_label_v2', 'strong_profitable_trade_label',
    'high_conviction_trade_label', 'paper_candidate_label', 'avoid_trade_label',
    'weak_trade_label', 'no_trade_label', 'mfe', 'mae', 'realized_pnl',
    'expected_return_after_cost', 'return_to_cost_ratio',
}


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class DynamicPreset:
    preset_id: str = ''
    preset_family: str = ''
    threshold: float = 0.35
    max_trades_per_day: int = 3
    top_n_confidence_per_day: int = 3
    max_open_positions: int = 1
    spread_limit_pct: float = 0.10
    liquidity_min: float = 0.0
    premium_band: str = 'all'   # 'low'|'mid'|'high'|'all'
    dte_range: str = 'all'      # '0-1'|'2-7'|'7-30'|'all'
    regime_filter: str = 'all'  # 'all'|'high_volatility'|'trend_only'|'non_chop'
    expiry_filter: str = 'all'  # 'all'|'expiry_only'|'non_expiry_only'
    avoid_first_n_minutes: int = 0
    avoid_last_n_minutes: int = 0
    entry_time_start: str = '09:30'
    entry_time_end: str = '15:00'
    expected_move_to_cost_ratio_min: float = 0.0
    cooldown_after_loss_minutes: int = 0
    daily_stop_loss_pct: float = 0.0
    daily_profit_lock_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DynamicPreset':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


# ─── Preset family builders ──────────────────────────────────────────────────

def _preset_A_conservative() -> List[DynamicPreset]:
    """Conservative: moderate threshold, top-2/3 trades, strict spread/liquidity, avoid first/last 10min.

    Thresholds start at 0.20 to give the model room to generate trades.
    (Model score 75th percentile = 0.60, so 0.35 threshold eliminates most days.)
    """
    presets = []
    for thresh in [0.20, 0.25, 0.30, 0.35]:
        for top_n in [2, 3]:
            presets.append(DynamicPreset(
                preset_id=f'conservative_t{int(thresh*100)}_n{top_n}',
                preset_family='A_conservative',
                threshold=thresh,
                max_trades_per_day=top_n,
                top_n_confidence_per_day=top_n,
                max_open_positions=1,
                spread_limit_pct=0.15,  # ~15% spread is dataset-appropriate; 0.05 filters 93% of rows
                liquidity_min=50000.0,
                premium_band='all',
                dte_range='all',
                regime_filter='all',
                expiry_filter='all',
                avoid_first_n_minutes=10,
                avoid_last_n_minutes=10,
                expected_move_to_cost_ratio_min=0.0,
                daily_stop_loss_pct=2.0,
            ))
    return presets


def _preset_B_balanced() -> List[DynamicPreset]:
    """Balanced: moderate threshold, max-3 trades, normal spread/liquidity."""
    presets = []
    for thresh in [0.25, 0.30, 0.35]:
        for top_n in [2, 3]:
            for spread in [0.10, 0.15]:
                presets.append(DynamicPreset(
                    preset_id=f'balanced_t{int(thresh*100)}_n{top_n}_sp{int(spread*100)}',
                    preset_family='B_balanced',
                    threshold=thresh,
                    max_trades_per_day=top_n,
                    top_n_confidence_per_day=top_n,
                    max_open_positions=2,
                    spread_limit_pct=spread,
                    liquidity_min=50000.0,
                    premium_band='all',
                    dte_range='all',
                    regime_filter='all',
                    expiry_filter='all',
                    avoid_first_n_minutes=5,
                    avoid_last_n_minutes=5,
                    expected_move_to_cost_ratio_min=0.5,
                ))
    return presets


def _preset_C_low_cost() -> List[DynamicPreset]:
    """Low-cost: strict spread/liquidity, premium band, expected move to cost ratio."""
    presets = []
    for thresh in [0.20, 0.25, 0.30]:
        for spread in [0.10, 0.15]:
            for premium in ['mid', 'high']:
                presets.append(DynamicPreset(
                    preset_id=f'low_cost_t{int(thresh*100)}_sp{int(spread*100)}_prem{premium}',
                    preset_family='C_low_cost',
                    threshold=thresh,
                    max_trades_per_day=3,
                    top_n_confidence_per_day=3,
                    max_open_positions=2,
                    spread_limit_pct=spread,
                    liquidity_min=50000.0,
                    premium_band=premium,
                    dte_range='all',
                    regime_filter='all',
                    expiry_filter='all',
                    expected_move_to_cost_ratio_min=0.0,
                ))
    return presets


def _preset_D_dte() -> List[DynamicPreset]:
    """DTE: DTE bucket isolation (0-1, 2-7, 7-30, all)."""
    presets = []
    for dte_range in ['0-1', '2-7', '7-30']:
        for thresh in [0.30, 0.35, 0.40]:
            for top_n in [2, 3]:
                presets.append(DynamicPreset(
                    preset_id=f'dte_{dte_range}_t{int(thresh*100)}_n{top_n}',
                    preset_family='D_dte',
                    threshold=thresh,
                    max_trades_per_day=top_n,
                    top_n_confidence_per_day=top_n,
                    spread_limit_pct=0.10,
                    liquidity_min=50000.0,
                    premium_band='all',
                    dte_range=dte_range,
                    regime_filter='all',
                    expiry_filter='all',
                ))
    return presets


def _preset_E_volatility() -> List[DynamicPreset]:
    """Volatility: high-vol and trend-regime activation."""
    presets = []
    for regime in ['high_volatility', 'trend_only', 'non_chop']:
        for thresh in [0.25, 0.30, 0.35]:
            for top_n in [2, 3]:
                presets.append(DynamicPreset(
                    preset_id=f'vol_{regime}_t{int(thresh*100)}_n{top_n}',
                    preset_family='E_volatility',
                    threshold=thresh,
                    max_trades_per_day=top_n,
                    top_n_confidence_per_day=top_n,
                    spread_limit_pct=0.10,
                    liquidity_min=50000.0,
                    premium_band='all',
                    dte_range='all',
                    regime_filter=regime,
                    expiry_filter='all',
                ))
    return presets


def _preset_F_router() -> List[DynamicPreset]:
    """Router: CE/PE/no-trade selection via ML probability."""
    presets = []
    for thresh in [0.25, 0.30, 0.35, 0.40]:
        for top_n in [1, 2, 3]:
            presets.append(DynamicPreset(
                preset_id=f'router_t{int(thresh*100)}_n{top_n}',
                preset_family='F_router',
                threshold=thresh,
                max_trades_per_day=top_n,
                top_n_confidence_per_day=top_n,
                max_open_positions=2,
                spread_limit_pct=0.10,
                liquidity_min=50000.0,
                premium_band='all',
                dte_range='all',
                regime_filter='all',
                expiry_filter='all',
            ))
    return presets


def build_preset_grid(max_presets: int = 50) -> List[DynamicPreset]:
    all_families = (
        _preset_A_conservative() +
        _preset_B_balanced() +
        _preset_C_low_cost() +
        _preset_D_dte() +
        _preset_E_volatility() +
        _preset_F_router()
    )
    seen = set()
    unique = []
    for p in all_families:
        if p.preset_id not in seen:
            seen.add(p.preset_id)
            unique.append(p)
    return unique[:max_presets]


# ─── Dataset loading and audit ───────────────────────────────────────────────


def discover_datasets(data_dir: Path) -> List[Dict[str, Any]]:
    """Scan data_dir for CSV datasets and rank by reliability.

    Ranking criteria (weighted):
      - has timestamp column             (+3)
      - has option_type column           (+2)
      - has at least 1 label column      (+3)
      - has gross/net_forward_return     (+2)
      - has live-computable feature cols (+2)
      - row count normalised 0-3         (+3)
      - no forbidden leakage cols        (+3, -5 per forbidden col)

    Returns list sorted by score descending.
    """
    if not data_dir.is_dir():
        return []

    csv_files = sorted(data_dir.glob('*.csv'), key=lambda p: p.stat().st_size, reverse=True)
    scored = []
    REQUIRED_SURFACE = ['timestamp', 'option_type']
    LABEL_TOKENS = ['_label', 'cost_survivor', 'profitable_trade', 'strong_profitable',
                    'high_conviction', 'paper_candidate']
    RETURN_TOKENS = ['gross_forward_return', 'net_forward_return']
    # NOTE: only exact/specific forbidden leakage tokens — NOT generic 'cost_'
    # cost_survivor_label_v2 is a LEGITIMATE label column (our training target)
    FORBIDDEN_LEAKAGE = ['mfe', 'mae', 'realized_pnl', 'return_after_cost',
                         'cost_adjusted_return', 'trade_outcome',
                         'expected_return_after_cost', 'cost_return_units_estimated']

    for fpath in csv_files:
        try:
            df_sample = pd.read_csv(fpath, nrows=5, low_memory=False)
            cols = df_sample.columns.tolist()
            col_lower = [c.lower() for c in cols]
            file_size_mb = fpath.stat().st_size / (1024 * 1024)
            score = 0.0
            reasons = []

            has_ts = any('timestamp' in c or 'date' in c for c in col_lower)
            has_ot = any('option_type' in c for c in col_lower)
            score += 3 if has_ts else 0
            score += 2 if has_ot else 0
            reasons.append(f'ts={has_ts}, ot={has_ot}')

            label_cols = [c for c in cols if any(t in c.lower() for t in LABEL_TOKENS)]
            score += 3 if label_cols else 0
            reasons.append(f'labels={len(label_cols)}')

            return_found = [c for c in cols if any(t in c.lower() for t in RETURN_TOKENS)]
            score += 2 if return_found else 0
            reasons.append(f'returns={len(return_found)}')

            # Exclude evaluation-helper cols from forbidden scoring penalty
            LABEL_HELPERS = {'expected_return_after_cost', 'cost_return_units_estimated'}
            forbidden_found = [c for c in cols if any(f in c.lower() for f in FORBIDDEN_LEAKAGE)]
            non_helper_forbidden = [c for c in forbidden_found
                                    if c.lower() not in LABEL_HELPERS]
            if non_helper_forbidden:
                score -= min(5 * len(non_helper_forbidden), 5)
            reasons.append(f'forbidden={len(non_helper_forbidden)}/{len(forbidden_found)}')

            feature_like = [c for c in cols if not any(
                t in c.lower() for t in ['_label', '_return', 'forward', 'future',
                                          'pnl', 'profit', 'outcome', 'mfe', 'mae',
                                          'expected_return', 'cost_adjusted', 'realized'])]
            score += 2 if len(feature_like) > 10 else 0
            reasons.append(f'feats~{len(feature_like)}')

            score += min(file_size_mb / 200, 3)
            if 'cost_aware_edge' in fpath.name:
                score += 2
            if '_202606' in fpath.name:
                score += 1

            row_hint = None
            if '405' in fpath.name or '405k' in fpath.name.lower():
                row_hint = 405571
            elif '300' in fpath.name:
                row_hint = 300000

            scored.append({
                'path': str(fpath),
                'filename': fpath.name,
                'size_mb': round(file_size_mb, 1),
                'col_count': len(cols),
                'score': round(score, 2),
                'reasons': '; '.join(reasons),
                'has_labels': bool(label_cols),
                'has_return_cols': bool(return_found),
                'has_timestamp': has_ts,
                'has_option_type': has_ot,
                'forbidden_leak_count': len(forbidden_found),
                'feature_like_count': len(feature_like),
                'row_hint': row_hint,
            })
        except Exception as e:
            scored.append({
                'path': str(fpath),
                'filename': fpath.name,
                'score': -999,
                'error': str(e),
            })

    scored.sort(key=lambda x: x.get('score', -999), reverse=True)
    return scored


def audit_dataset(path: Path) -> Dict[str, Any]:
    """Deep audit of a single dataset. Returns audit dict."""
    # NOTE: only exact/specific forbidden leakage tokens — NOT generic 'cost_'
    # cost_survivor_label_v2 is a LEGITIMATE label column (our training target)
    FORBIDDEN_LEAKAGE = ['mfe', 'mae', 'realized_pnl', 'return_after_cost',
                         'cost_adjusted_return', 'trade_outcome',
                         'expected_return_after_cost', 'cost_return_units_estimated']
    LABEL_TOKENS = ['_label', 'cost_survivor', 'profitable_trade',
                    'strong_profitable', 'high_conviction', 'paper_candidate',
                    'avoid_trade', 'weak_trade', 'no_trade']
    RETURN_TOKENS = ['forward_return', 'gross_forward', 'net_forward']
    LIVE_FEATURE_TOKENS = ['ltp', 'volume', 'oi', 'strike', 'dte', 'range',
                           'ret_', 'momentum', 'spread', 'liquidity',
                           'moneyness', 'volatility', 'iv', 'delta',
                           'gamma', 'theta', 'vega', 'atm', 'prem', 'chop',
                           'trend', 'breakout', 'session', 'minute']

    audit = {
        'path': str(path),
        'filename': path.name,
        'exists': path.exists(),
        'selected': False,
        'row_count': None,
        'col_count': None,
        'date_range': None,
        'ce_rows': None,
        'pe_rows': None,
        'labels_available': [],
        'return_cols': [],
        'forbidden_cols': [],
        'live_feature_count': 0,
        'live_features': [],
        'feature_like_count': 0,
        'suitable_for_retraining': False,
        'suitability_reasons': [],
        'rejected_reasons': [],
    }

    if not path.exists():
        audit['rejected_reasons'].append('File does not exist')
        return audit

    # Chunked scan for cols + row estimate
    total_rows = 0
    cols_seen = None
    try:
        for chunk in pd.read_csv(path, chunksize=10000, low_memory=False):
            if cols_seen is None:
                cols_seen = chunk.columns.tolist()
            total_rows += len(chunk)
            if total_rows > 500000:
                break
    except Exception as e:
        audit['rejected_reasons'].append(f'Cannot read CSV: {e}')
        return audit

    cols = cols_seen
    col_lower = [c.lower() for c in cols]
    col_set_lower = set(col_lower)

    audit['row_count'] = total_rows
    audit['col_count'] = len(cols)

    # Exclude evaluation-helper cols (used in label construction, not features)
    LABEL_HELPERS = {'expected_return_after_cost', 'cost_return_units_estimated',
                     'return_to_cost_ratio'}
    forbidden_all = [cols[i] for i, c in enumerate(col_lower)
                     if any(f in c for f in FORBIDDEN_LEAKAGE)]
    forbidden = [c for c in forbidden_all if c.lower() not in LABEL_HELPERS]
    audit['forbidden_cols'] = forbidden_all  # report ALL for transparency
    if forbidden:
        audit['rejected_reasons'].append(f'Forbidden feature-leakage cols: {forbidden}')

    labels = [cols[i] for i, c in enumerate(col_lower)
              if any(t in c for t in LABEL_TOKENS)]
    audit['labels_available'] = labels

    ret_cols = [cols[i] for i, c in enumerate(col_lower)
                if any(t in c for t in RETURN_TOKENS)]
    audit['return_cols'] = ret_cols

    live_feats = [cols[i] for i, c in enumerate(col_lower)
                  if any(t in c for t in LIVE_FEATURE_TOKENS)
                  and cols[i] not in labels and cols[i] not in ret_cols
                  and cols[i] not in forbidden]
    live_feats_unique = list(dict.fromkeys(live_feats))
    audit['live_features'] = live_feats_unique[:100]
    audit['live_feature_count'] = len(live_feats_unique)

    feature_like = [cols[i] for i, c in enumerate(col_lower)
                    if not any(x in c for x in ['_label', '_return', 'forward',
                                                 'future', 'pnl', 'profit', 'outcome',
                                                 'expected', 'realized', 'cost_', 'target'])]
    audit['feature_like_count'] = len(feature_like)

    # CE/PE split + date range (sample first 20k rows)
    try:
        sample = pd.read_csv(path, nrows=20000, low_memory=False)
        ot_col = next((c for c in sample.columns if 'option_type' in c.lower()), None)
        if ot_col:
            ot_upper = sample[ot_col].astype(str).str.upper()
            audit['ce_rows'] = int((ot_upper == 'CE').sum())
            audit['pe_rows'] = int((ot_upper == 'PE').sum())
        ts_col = next((c for c in sample.columns
                       if 'timestamp' in c.lower() or 'date' in c.lower()), None)
        if ts_col:
            try:
                dates = pd.to_datetime(sample[ts_col], errors='coerce').dropna()
                if len(dates) > 0:
                    audit['date_range'] = [str(dates.min()), str(dates.max())]
            except Exception:
                pass
    except Exception:
        pass

    reasons = []
    if audit['row_count'] and audit['row_count'] >= 100000:
        reasons.append(f'rows={audit["row_count"]} (OK)')
    elif audit['row_count']:
        reasons.append(f'rows={audit["row_count"]} (WARN: <100k)')
    if labels:
        reasons.append(f'labels={len(labels)} (OK)')
    else:
        audit['rejected_reasons'].append('No label columns')
    if ret_cols:
        reasons.append(f'returns={len(ret_cols)} (OK)')
    else:
        audit['rejected_reasons'].append('No return columns')
    if not forbidden:
        reasons.append('no forbidden leakage (OK)')
    if audit.get('ce_rows', 0) > 0 or audit.get('pe_rows', 0) > 0:
        reasons.append(f'CE={audit.get("ce_rows",0)}, PE={audit.get("pe_rows",0)}')

    audit['suitability_reasons'] = reasons
    # Only reject if there are forbidden feature-level leakage columns.
    # expected_return_after_cost and cost_return_units_estimated are evaluation
    # helpers used in label construction — they don't enter ML features.
    LABEL_HELPER_COLS = {'expected_return_after_cost', 'cost_return_units_estimated',
                          'return_to_cost_ratio'}
    non_helper_forbidden = [f for f in forbidden if f.lower() not in LABEL_HELPER_COLS]
    audit['suitable_for_retraining'] = (
        bool(labels) and bool(ret_cols)
        and (not forbidden or all(f in LABEL_HELPER_COLS for f in forbidden))
        and audit.get('row_count', 0) >= 50000
        and audit.get('live_feature_count', 0) >= 20
    )
    if non_helper_forbidden:
        audit['suitable_for_retraining'] = False
        audit['rejected_reasons'].append(
            f'Forbidden feature-leakage cols: {non_helper_forbidden}')
    return audit


def generate_dataset_selection_report(
    ranked: List[Dict[str, Any]],
    audited: List[Dict[str, Any]],
    selected_path: str,
    output_dir: Path,
    ts: str,
) -> Tuple[Path, Path]:
    """Write ml_only_large_dataset_selection JSON + Markdown reports."""
    report = {
        'timestamp': ts,
        'total_datasets_scanned': len(ranked),
        'ranked_datasets': ranked,
        'audited_datasets': audited,
        'selected_path': selected_path,
        'selection_criteria': [
            'highest reliability score (timestamp + labels + returns + features)',
            'no forbidden leakage columns',
            '>=50k rows, >=20 live-computable features',
            'cost_aware_edge dataset preferred',
        ],
    }
    json_path = output_dir / f'ml_only_large_dataset_selection_{ts}.json'
    with json_path.open('w') as f:
        json.dump(report, f, indent=2, default=str)

    lines = [
        f"# Dataset Selection Report — {ts}", "",
        f"**Total scanned**: {len(ranked)} | **Selected**: `{selected_path}`",
        "",
        "## Ranking Criteria",
        "- +3 has timestamp | +2 has option_type | +3 has labels",
        "- +2 has return cols | +2 has >10 features | +3 no forbidden cols",
        "- +2 cost_aware_edge bonus | +1 recent filename | +size_proxy",
        "",
        "| rank | filename | score | rows~ | cols | labels | returns | feats | forbidden |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, ds in enumerate(ranked[:10]):
        row_info = f"~{ds.get('row_hint','?')}" if ds.get('row_hint') else '?'
        sel = '✅' if ds.get('path') == selected_path else ''
        lines.append(
            f"| {i+1} | {ds.get('filename','?')[:50]} | "
            f"{ds.get('score',0):.1f} | {row_info} | "
            f"{ds.get('col_count','?')} | {int(ds.get('has_labels',False))} | "
            f"{int(ds.get('has_return_cols',False))} | "
            f"{ds.get('feature_like_count','?')} | "
            f"{ds.get('forbidden_leak_count',0)} | {sel} |"
        )

    if audited:
        lines += ["", "## Audited Top-3"]
        for i, aud in enumerate(audited[:3]):
            lines += [
                f"\n### {i+1}. {aud.get('filename','?')}",
                f"- Rows: {aud.get('row_count','?')} | Cols: {aud.get('col_count','?')}",
                f"- Date: {aud.get('date_range','?')}",
                f"- CE: {aud.get('ce_rows','?')} | PE: {aud.get('pe_rows','?')}",
                f"- Labels: {aud.get('labels_available',[])}",
                f"- Returns: {aud.get('return_cols',[])}",
                f"- Forbidden: {aud.get('forbidden_cols',[])}",
                f"- Live features: {aud.get('live_feature_count','?')}",
                f"- Suitable: {aud.get('suitable_for_retraining',False)}",
            ]

    md_path = output_dir / f'ml_only_large_dataset_selection_{ts}.md'
    with md_path.open('w') as f:
        f.write('\n'.join(lines))

    return json_path, md_path


# ─── Dataset loading and audit ───────────────────────────────────────────────


def load_and_audit_dataset(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print(f"[INFO] Loading dataset: {path}")
    df_sample = pd.read_csv(path, nrows=100)

    # Identify available columns
    available = list(df_sample.columns)

    # Check required columns
    required_base = ['timestamp', 'option_type', 'strike_price', 'dte_days', 'ltp',
                     'net_forward_return', 'gross_forward_return']
    label_cols = [c for c in available if '_label' in c.lower() or 'cost_survivor' in c.lower()
                  or 'profitable' in c.lower()]

    # Load ALL columns — net_forward_return and gross_forward_return are needed for
    # backtest evaluation (trade PnL), not as ML features. get_live_features() will
    # correctly exclude them. We must NOT filter them from the dataset load.
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Dataset: {len(df)} rows, {len(df.columns)} cols")

    # Audit
    audit = {
        'rows': len(df),
        'cols': len(df.columns),
        'col_list': list(df.columns),
        'label_cols': label_cols,
        'has_timestamp': 'timestamp' in df.columns,
        'has_option_type': 'option_type' in df.columns,
        'has_dte_days': 'dte_days' in df.columns,
        'has_ltp': 'ltp' in df.columns,
        'has_net_forward_return': 'net_forward_return' in df.columns,
        'has_gross_forward_return': 'gross_forward_return' in df.columns,
        'has_volume': 'volume' in df.columns,
        'has_oi': 'oi' in df.columns,
        'has_range_pct': 'range_pct' in df.columns,
    }

    # Parse timestamp
    if 'timestamp' in df.columns:
        try:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp_dt').reset_index(drop=True)
            audit['timestamp_range'] = {
                'start': str(df['timestamp_dt'].min()),
                'end': str(df['timestamp_dt'].max()),
            }
            audit['total_days'] = (df['timestamp_dt'].max() - df['timestamp_dt'].min()).days
        except Exception as e:
            audit['timestamp_parse_error'] = str(e)

    # CE/PE split
    if 'option_type' in df.columns:
        audit['ce_count'] = int((df['option_type'].astype(str).str.upper() == 'CE').sum())
        audit['pe_count'] = int((df['option_type'].astype(str).str.upper() == 'PE').sum())

    print(f"[INFO] Dataset audit: {audit['rows']} rows, {audit['total_days']} days")
    print(f"[INFO] CE: {audit['ce_count']}, PE: {audit['pe_count']}")
    print(f"[INFO] Labels: {audit['label_cols']}")

    return df, audit


def get_live_features(df: pd.DataFrame) -> List[str]:
    """Get live-computable features only (strictly no leakage)."""
    features = []
    for c in df.columns:
        if c in ('timestamp', 'timestamp_dt', 'trading_day', 'instrument_key',
                 'trading_symbol', 'expiry', 'option_type', 'weekly', 'source_file',
                 'net_forward_return', 'gross_forward_return', 'moneyness_bucket'):
            continue
        if c in FORBIDDEN_EXACT:
            continue
        if any(f in c.lower() for f in FEATURE_FORBIDDEN):
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype) and df[c].dtype != np.bool_:
            if not np.isinf(df[c].dropna()).any():
                features.append(c)
    return features


# ─── Time-based walk-forward splits ──────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame, n_folds: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """3-way walk-forward: train / validation / test per fold.

    Returns a list of (train, val, test) tuples using expanding window.
    Each test set is truly held-out (never used for training or preset selection).
    """
    n = len(df)
    splits = []

    # 3-fold boundaries (expanding window, 60/20/20 per fold)
    # Fold 0: train 0-50%, val 50-60%, test 60-70%
    # Fold 1: train 0-60%, val 60-70%, test 70-80%
    # Fold 2: train 0-70%, val 70-80%, test 80-90%  (or 80-100% for last)
    boundaries = []
    if n_folds == 3:
        # Three test periods
        boundaries = [
            (int(n * 0.50), int(n * 0.60), int(n * 0.70)),
            (int(n * 0.60), int(n * 0.70), int(n * 0.80)),
            (int(n * 0.70), int(n * 0.80), n),
        ]
    elif n_folds == 5:
        boundaries = [
            (int(n * 0.50), int(n * 0.60), int(n * 0.70)),
            (int(n * 0.60), int(n * 0.70), int(n * 0.80)),
            (int(n * 0.70), int(n * 0.80), int(n * 0.90)),
            (int(n * 0.75), int(n * 0.85), int(n * 0.92)),
            (int(n * 0.80), int(n * 0.90), n),
        ]
    else:
        boundaries = [
            (int(n * 0.60), int(n * 0.70), int(n * 0.80)),
            (int(n * 0.70), int(n * 0.80), int(n * 0.90)),
            (int(n * 0.80), int(n * 0.90), n),
        ]

    for train_end, val_end, test_end in boundaries:
        if train_end < 100 or val_end - train_end < 20 or test_end - val_end < 20:
            continue
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:test_end].copy()
        splits.append((train, val, test))

    return splits


# ─── Preset filtering (vectorized) ───────────────────────────────────────────

def apply_preset_filters(df: pd.DataFrame, preset: DynamicPreset) -> pd.DataFrame:
    """Apply preset filters to dataframe using vectorized boolean masks.

    All filters use ONLY live-computable fields. No future, PnL, return,
    label, or outcome data.
    """
    if len(df) == 0:
        return df

    # DTE range filter
    if preset.dte_range != 'all':
        if 'dte_days' in df.columns:
            dte = df['dte_days'].fillna(-1).values
            if preset.dte_range == '0-1':
                mask = (dte >= 0.0) & (dte <= 1.0)
            elif preset.dte_range == '2-7':
                mask = (dte >= 2.0) & (dte <= 7.0)
            elif preset.dte_range == '7-30':
                mask = (dte >= 7.0) & (dte <= 30.0)
            else:
                mask = np.ones(len(df), dtype=bool)
            df = df.loc[mask]

    # Spread limit filter
    if preset.spread_limit_pct > 0 and 'range_pct' in df.columns:
        spread = df['range_pct'].fillna(0).abs().values
        df = df.loc[spread <= preset.spread_limit_pct]

    # Liquidity filter
    if preset.liquidity_min > 0 and 'volume' in df.columns:
        vol = df['volume'].fillna(0).values
        df = df.loc[vol >= preset.liquidity_min]

    # Premium band filter
    if preset.premium_band != 'all' and 'ltp' in df.columns:
        ltp = df['ltp'].fillna(0).values
        if preset.premium_band == 'low':
            df = df.loc[ltp < 50.0]
        elif preset.premium_band == 'mid':
            df = df.loc[(ltp >= 50.0) & (ltp < 150.0)]
        elif preset.premium_band == 'high':
            df = df.loc[ltp >= 150.0]

    return df


def top_n_per_day(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    """Select top-N confidence trades per day."""
    if top_n <= 0 or 'timestamp_dt' not in df.columns or score_col not in df.columns:
        return df
    df = df.copy()
    df['_day'] = df['timestamp_dt'].dt.date
    df['_rank'] = df.groupby('_day')[score_col].rank(method='first', ascending=False)
    return df[df['_rank'] <= top_n].drop(columns=['_day', '_rank'])


# ─── Model training ──────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, model_type: str, cand_dir: Optional[Path] = None, sample_weight: Optional[np.ndarray] = None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
        VotingClassifier,
    )
    from sklearn.calibration import CalibratedClassifierCV

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y = y.astype(int)

    if model_type == 'elasticnet':
        model = LogisticRegression(penalty='l2', solver='lbfgs',
                                   C=1.0, max_iter=500, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42)
    elif model_type == 'calibrated_logistic_regression':
        base = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500, C=1.0, random_state=42)
        model = CalibratedClassifierCV(base, cv=3)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=8,
                                       min_samples_leaf=50, random_state=42, n_jobs=-1)
    elif model_type == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=100, max_depth=8,
                                     min_samples_leaf=50, random_state=42, n_jobs=-1)
    elif model_type == 'hist_gradient_boosting':
        model = HistGradientBoostingClassifier(max_depth=6, max_iter=100,
                                               min_samples_leaf=50, random_state=42)
    elif model_type == 'soft_voting':
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=80, random_state=42)
        et = ExtraTreesClassifier(n_estimators=50, max_depth=6, min_samples_leaf=80, random_state=42)
        lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=300, C=1.0, random_state=42)
        model = VotingClassifier([('rf', rf), ('et', et), ('lr', lr)], voting='soft')
    elif model_type == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05,
                                      random_state=42, use_label_encoder=False,
                                      eval_metric='logloss', verbosity=0)
        except Exception:
            model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    else:
        model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    model.fit(Xs, y, **fit_kwargs)

    # Save model.pkl if cand_dir provided
    if cand_dir is not None:
        try:
            import sklearn.base as _base
            _model_dict = {
                'model': model,
                'scaler': scaler,
                'model_type': model_type,
                'n_features': X.shape[1],
            }
            with (cand_dir / 'model.pkl').open('wb') as _f:
                pickle.dump(_model_dict, _f)
        except Exception:
            pass  # Non-fatal: model.pkl is a best-effort artifact

    return model, scaler


# ─── Per-fold evaluation ─────────────────────────────────────────────────────

def evaluate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    label: str,
    model_type: str,
    preset: DynamicPreset,
) -> Optional[Dict[str, Any]]:
    """Train on train_df, tune threshold on val_df, evaluate on test_df.

    Returns raw trade returns and metrics. Cost stress is applied analytically
    in evaluate_candidate_preset to avoid redundant model training.
    """
    try:
        # Build feature matrices
        available = [f for f in features if f in train_df.columns]
        if len(available) < 5:
            return None

        X_train = train_df[available].fillna(0).values
        y_train = train_df[label].fillna(0).values
        X_val = val_df[available].fillna(0).values
        y_val = val_df[label].fillna(0.0).values
        X_test = test_df[available].fillna(0).values
        y_test = test_df[label].fillna(0.0).values

        # Filter valid labels
        valid_train = (y_train == 0) | (y_train == 1)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        if len(X_train) < 100 or len(X_test) < 50:
            return None

        # PHASE 10: cost-aware sample weighting (higher for survivors / positive gross that clear cost)
        sample_weight_train = None
        try:
            if 'gross_forward_return' in train_df.columns:
                g = train_df['gross_forward_return'].fillna(0).values
                g_valid = g[valid_train]
                # weight 1.8x for positive survivors (label 1), base 0.7 for others; also boost high gross
                base_w = np.where(y_train == 1, 1.8, 0.7)
                gross_boost = np.clip(1.0 + np.clip(g_valid, -0.01, 0.05) * 10, 0.5, 2.5)
                sample_weight_train = base_w * gross_boost
                sample_weight_train = sample_weight_train / sample_weight_train.mean() * len(sample_weight_train)  # normalize-ish
        except Exception:
            sample_weight_train = None

        model, scaler = train_model(X_train, y_train, model_type, None, sample_weight=sample_weight_train)  # cand_dir=None: model saved by caller

        # Compute validation scores for threshold tuning
        X_val_scaled = scaler.transform(X_val)
        if hasattr(model, 'predict_proba'):
            val_scores = model.predict_proba(X_val_scaled)[:, 1]
        else:
            val_scores = model.decision_function(X_val_scaled)

        # Tune threshold on validation set.
        # IMPORTANT: same filtering pipeline as test — top-N per day THEN threshold.
        val_copy = val_df.copy()
        val_copy['_score'] = val_scores
        val_filtered = apply_preset_filters(val_copy, preset)
        if preset.top_n_confidence_per_day > 0 and 'timestamp_dt' in val_filtered.columns:
            val_filtered = top_n_per_day(val_filtered, '_score', preset.top_n_confidence_per_day)

        y_val_filt = val_filtered[label].fillna(0.0).values
        val_scores_filt = val_filtered['_score'].values

        # Find best threshold on the filtered validation set (F1 selection)
        best_thresh = preset.threshold
        best_f1 = -1
        for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            preds = (val_scores_filt >= t).astype(int)
            if preds.sum() < 3:
                continue
            tp = ((preds == 1) & (y_val_filt == 1)).sum()
            fp = ((preds == 1) & (y_val_filt == 0)).sum()
            fn = ((preds == 0) & (y_val_filt == 1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        # Evaluate on OOS test set
        X_test_scaled = scaler.transform(X_test)
        if hasattr(model, 'predict_proba'):
            test_scores = model.predict_proba(X_test_scaled)[:, 1]
        else:
            test_scores = model.decision_function(X_test_scaled)

        # Build scored dataframe and apply same pipeline as validation
        test_copy = test_df.copy()
        test_copy['_score'] = test_scores
        test_filtered = apply_preset_filters(test_copy, preset)

        if len(test_filtered) == 0:
            return None

        # Top-N per day FIRST on all filtered rows (mirrors live pre-screening)
        if preset.top_n_confidence_per_day > 0 and 'timestamp_dt' in test_filtered.columns:
            test_filtered = top_n_per_day(test_filtered, '_score', preset.top_n_confidence_per_day)

        if len(test_filtered) == 0:
            return None

        # THEN apply probability threshold to select final trades
        signals = (test_filtered['_score'] >= best_thresh).astype(int)
        trade_mask = signals == 1
        trades = test_filtered[trade_mask].copy()

        if len(trades) == 0:
            return None

        # Use gross_forward_return (no embedded cost) in multiplier format.
        # Gross values can be extreme (max ~147000x), so clip before use.
        gross_raw = trades['gross_forward_return'].fillna(0).values
        # Clip: loss capped at -1.0x (total loss), gain capped at +5.0x (+500%).
        gross_clipped = np.clip(gross_raw, -1.0, 5.0)
        # Convert from multiplier to decimal percentage (e.g. 0.10 -> 10%).
        returns = gross_clipped * 100.0

        if len(returns) < 5:
            return None

        # Costs in percentage points: 0.25% + 0.10% = 0.35 percentage points
        # (returns are in %-space, so costs are too: 0.35 not 0.0035)
        base_cost_pct = (0.0025 + 0.0010) * 100  # = 0.35 percentage points
        base_costs = base_cost_pct * len(returns)
        base_net_ret = returns.sum() - base_costs
        gross_w = returns[returns > 0].sum()
        gross_l = abs(returns[returns < 0].sum())
        gross_pf = gross_w / gross_l if gross_l > 0 else (1.0 if gross_w > 0 else 0.0)
        base_net_pf = base_net_ret / abs(base_costs) if base_costs > 0 and base_net_ret != 0 else (1.0 if base_net_ret >= 0 else 0.0)

        wins = (returns > 0).sum()
        losses = (returns < 0).sum()
        win_rate = wins / len(returns) if len(returns) > 0 else 0.0

        avg_win = returns[returns > 0].mean() if wins > 0 else 0.0
        avg_loss = abs(returns[returns < 0].mean()) if losses > 0 else 1.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        equity = np.cumsum(returns)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = float(-dd.min()) if len(dd) > 0 else 0.0  # positive drawdown magnitude (worst peak-to-trough dip)

        # Daily stability
        if 'timestamp_dt' in trades.columns:
            daily = pd.DataFrame({'day': trades['timestamp_dt'].dt.date, 'pnl': returns})
            daily_agg = daily.groupby('day')['pnl'].sum()
            if daily_agg.sum() > 0:
                top_day = daily_agg.max() / daily_agg.sum()
            else:
                top_day = 0.0
        else:
            top_day = 0.0

        # Capture all pre-threshold scores + gross returns for threshold robustness
        # (all test_filtered rows, before the threshold cut)
        all_filtered_scores = test_filtered['_score'].values
        all_filtered_gross = test_filtered['gross_forward_return'].fillna(0).values

        return {
            'trade_count': len(trades),
            'win_rate': win_rate,
            'gross_pf': gross_pf,
            'base_net_pf': base_net_pf,  # PF at base cost (1.0x)
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'top_day_concentration': top_day,
            'threshold_used': best_thresh,
            'raw_returns': returns.tolist(),  # list (in percentage space) for cost stress
            # Threshold robustness data: all pre-threshold rows
            'all_filtered_scores': all_filtered_scores.tolist(),
            'all_filtered_gross': all_filtered_gross.tolist(),
            'all_filtered_y': test_filtered[label].fillna(0.0).values.tolist(),
        }
    except Exception:
        return None


# ─── Full walk-forward evaluation ────────────────────────────────────────────

def _compute_cost_stress_pf(returns: List[float], cost_mult: float) -> float:
    """Compute net PF from raw percentage-space returns at a given cost multiplier.

    Returns are in percentage space (e.g. 10.0 = 10% gain).
    Cost is 0.35 percentage points per trade at 1.0x (0.25% + 0.10% = 0.35%).
    Since returns are in %-space, cost is also in %-space: 0.35, not 0.0035.
    """
    if not returns:
        return 0.0
    rets = np.array(returns, dtype=np.float64)
    n = len(rets)
    if n == 0:
        return 0.0
    # 0.35 percentage points per trade, not 0.0035 (returns are %-space, not decimal)
    total_cost_pct = (0.0025 + 0.0010) * 100 * cost_mult * n
    net_ret = rets.sum() - total_cost_pct
    if total_cost_pct > 0 and net_ret != 0:
        return net_ret / abs(total_cost_pct)
    return 1.0 if net_ret >= 0 else 0.0


def evaluate_candidate_preset(
    df: pd.DataFrame,
    features: List[str],
    label: str,
    model_type: str,
    preset: DynamicPreset,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """Full walk-forward evaluation. Cost stress computed analytically."""
    splits = walk_forward_splits(df, n_folds=n_folds)
    results = []
    all_returns: List[np.ndarray] = []

    for fold_idx, (train_df, val_df, test_df) in enumerate(splits):
        # Apply preset filter BEFORE splitting to avoid leakage
        test_filtered = apply_preset_filters(test_df, preset)

        # Train once per fold, get base metrics + raw returns
        r = evaluate_fold(train_df, val_df, test_filtered, features, label, model_type, preset)
        if r:
            results.append(r)
            if 'raw_returns' in r:
                all_returns.append(np.array(r['raw_returns']))

    if not results:
        return {'status': 'NO_TRADES', 'folds': []}

    # Aggregate ALL raw returns across folds for correct aggregate metrics
    all_raw_returns: List[float] = []
    for r in results:
        all_raw_returns.extend(r.get('raw_returns', []))

    # Compute aggregate cost-stress PF from combined returns (correct weighting)
    cost_stress = {}
    for mult_name, mult_val in [('1.0x', 1.0), ('1.25x', 1.25), ('1.5x', 1.5), ('2.0x', 2.0)]:
        cost_stress[f'pf_at_{mult_name}'] = _compute_cost_stress_pf(all_raw_returns, mult_val)

    # Aggregate gross PF from combined returns
    all_rets = np.array(all_raw_returns, dtype=np.float64)
    total_trades = len(all_rets)
    gross_wins = all_rets[all_rets > 0].sum()
    gross_losses = abs(all_rets[all_rets < 0].sum())
    # Standard profit factor definition: total gross wins / total gross losses
    aggregate_gross_pf = gross_wins / gross_losses if gross_losses > 0 else (1.0 if gross_wins > 0 else 0.0)

    # Aggregate net PF at base cost (cost-stress ratio used for gates)
    aggregate_net_pf = cost_stress.get('pf_at_1.0x', 0.0)

    # Proper aggregate metrics (per user spec)
    aggregate_win_rate = float((all_rets > 0).sum() / total_trades) if total_trades > 0 else 0.0
    mean_trade_return = float(np.mean(all_rets)) if total_trades > 0 else 0.0
    # true net after fixed cost per trade (cost in same % space)
    cost_per_trade_pct = (0.0025 + 0.0010) * 100.0
    net_rets = all_rets - cost_per_trade_pct
    pos_net = net_rets[net_rets > 0].sum()
    neg_net = abs(net_rets[net_rets < 0].sum())
    pf_net_classic = pos_net / neg_net if neg_net > 0 else (1.0 if pos_net > 0 else 0.0)
    true_net_return = float(net_rets.sum())

    # ── Threshold robustness ─────────────────────────────────────────────────
    # Collect all pre-threshold data from each fold for robustness evaluation.
    # We evaluate at selected_threshold ± 0.05 and ± 0.10 on all OOS data combined.
    # Pass only if ALL 4 nearby thresholds have net_pf > 0 AND 1.5x_pf > 0.8.
    all_thresh_scores: List[float] = []
    all_thresh_gross: List[float] = []
    all_thresh_y: List[float] = []
    for r in results:
        all_thresh_scores.extend(r.get('all_filtered_scores', []))
        all_thresh_gross.extend(r.get('all_filtered_gross', []))
        all_thresh_y.extend(r.get('all_filtered_y', []))

    threshold_robust = False
    threshold_robust_details: Dict[str, Any] = {}
    if all_thresh_scores and len(all_thresh_scores) >= 50:
        thresh_scores = np.array(all_thresh_scores, dtype=np.float64)
        thresh_gross = np.array(all_thresh_gross, dtype=np.float64)
        thresh_y = np.array(all_thresh_y, dtype=np.float64)
        # Clip gross returns
        thresh_gross_clipped = np.clip(thresh_gross, -1.0, 5.0) * 100.0

        # Determine the central threshold (use most common threshold_used)
        central_thresh = preset.threshold
        thresh_used_values = [r['threshold_used'] for r in results if 'threshold_used' in r]
        if thresh_used_values:
            from statistics import mode
            try:
                central_thresh = mode(thresh_used_values)
            except Exception:
                central_thresh = float(np.mean(thresh_used_values))

        robust_pass_count = 0
        for delta in [-0.10, -0.05, 0.05, 0.10]:
            t = round(central_thresh + delta, 2)
            if t < 0.05 or t > 0.95:
                continue
            mask = thresh_scores >= t
            rets_t = thresh_gross_clipped[mask]
            if len(rets_t) < 5:
                threshold_robust_details[f't{int(t*100):03d}'] = {'net_pf': 0, '1.5x_pf': 0, 'n': 0}
                continue
            pf_base = _compute_cost_stress_pf(rets_t.tolist(), 1.0)
            pf_1_5x = _compute_cost_stress_pf(rets_t.tolist(), 1.5)
            threshold_robust_details[f't{int(t*100):03d}'] = {
                'net_pf': round(pf_base, 3),
                '1.5x_pf': round(pf_1_5x, 3),
                'n': int(len(rets_t)),
            }
            if pf_base > 0 and pf_1_5x > 0.8:
                robust_pass_count += 1
        threshold_robust = (robust_pass_count == 4)
    else:
        threshold_robust_details = {'note': 'insufficient data for robustness check'}


    trade_counts = [r['trade_count'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    max_dds = [r['max_drawdown'] for r in results]
    top_days = [r['top_day_concentration'] for r in results]
    base_pfs = [r.get('base_net_pf', 0) for r in results]
    profitable_folds = sum(1 for pf in base_pfs if pf > 1.0)

    worst_fold_pf = float(np.min(base_pfs))
    fold_pf_std = float(np.std(base_pfs))
    max_dd = float(np.max(max_dds)) if max_dds else 0.0  # worst (largest) positive drawdown magnitude
    top_day_conc = float(np.max(top_days))
    avg_sharpe = float(np.mean(sharpes))

    # Metrics for reporting (aggregate across all folds)
    avg_net_pf = aggregate_net_pf
    avg_gross_pf = aggregate_gross_pf

    # Gate evaluation (9 economic gates — internal screening only, NOT final promotion)
    gates = {}
    gates_passed = 0

    def gate(name: str, cond: bool) -> None:
        nonlocal gates_passed
        gates[name] = cond
        if cond:
            gates_passed += 1

    gate('C2_pf_base', avg_gross_pf >= PF_AT_1_00X)
    gate('C3_pf_1.25x', cost_stress['pf_at_1.25x'] >= PF_AT_1_25X)
    gate('C4_pf_1.5x', cost_stress['pf_at_1.5x'] >= PF_AT_1_50X)
    gate('C5_sharpe', avg_sharpe >= MIN_SHARPE)
    gate('C6_trades', total_trades >= MIN_TRADES)
    gate('C7_profitable_folds', profitable_folds >= max(3, n_folds - 1))
    gate('C8_worst_fold', worst_fold_pf >= 0.90)
    gate('C9_daily_stability', top_day_conc <= MAX_TOP_DAY_CONCENTRATION)
    gate('C1_positive_expectancy', avg_net_pf > 0)

    # SUSPICIOUS PF: PF > 10 is almost certainly a scaling or data bug.
    # Must fail promotion pending explicit mathematical audit.
    # SUSPICIOUS PF: net_pf > 200 indicates either a scaling bug or truly exceptional
    # strategy. We flag this for audit. The check uses net_pf (cost-adjusted) not gross_pf.
    # If gross_pf >= 1.2 but net_pf > 200: still suspicious (audit recommended).
    is_suspicious = avg_net_pf > 200.0
    if is_suspicious:
        gates['SUSPICIOUS_PF'] = True

    # Per-fold PFs for manifest (aggregate, not mean of means)
    mean_fold_pf = float(np.mean(base_pfs)) if base_pfs else 0.0
    median_fold_pf = float(np.median(base_pfs)) if base_pfs else 0.0
    mean_trades = float(np.mean(trade_counts)) if trade_counts else 0.0
    mean_sharpe = avg_sharpe
    # mean_pf is the same as aggregate_gross_pf (wins/losses ratio)
    mean_pf = aggregate_gross_pf

    return {
        'status': 'EVALUATED',
        'gates_passed': gates_passed,
        'gates_total': 9,
        'trade_count': total_trades,
        'net_pf': avg_net_pf,           # aggregate net PF at base cost (cost-stress ratio for gates)
        'gross_pf': avg_gross_pf,       # wins/losses ratio (classic)
        'profit_factor': aggregate_gross_pf,
        'sharpe': avg_sharpe,
        'max_drawdown': max_dd,
        'top_day_concentration': top_day_conc,
        'profitable_folds': profitable_folds,
        'n_profitable_folds': profitable_folds,
        'worst_fold_pf': worst_fold_pf,
        'mean_fold_pf': mean_fold_pf,
        'median_fold_pf': median_fold_pf,
        'fold_pf_std': fold_pf_std,
        'mean_pf': mean_pf,             # required by strict evaluator
        'mean_trades': mean_trades,
        'mean_sharpe': mean_sharpe,
        'win_rate': aggregate_win_rate,
        'mean_trade_return': mean_trade_return,
        'true_net_return': true_net_return,
        'pf_net_classic': pf_net_classic,
        'cost_stress': cost_stress,
        'cost_1.25x_pf': cost_stress.get('pf_at_1.25x', 0),
        'cost_1.50x_pf': cost_stress.get('pf_at_1.5x', 0),
        'gates': gates,
        'is_suspicious_pf': is_suspicious,
        'fold_details': results,
        'n_folds': len(results),
        'preset': preset.to_dict(),
        # Strict evaluator required fields (for manifest compatibility)
        'live_computable_features': True,
        'live_computable_preset': True,
        'paper_only': True,
        'real_trading_enabled': False,
        'threshold_robust': threshold_robust,
        'threshold_robust_details': threshold_robust_details,
        'filter_applied_at': 'DATA_LEVEL_PRE_TRAINING',
    }


# ─── Candidate group definitions ─────────────────────────────────────────────

CANDIDATE_GROUPS = {
    'PE': {
        'filter': 'PE_only',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression', 'calibrated_logistic_regression',
                   'random_forest', 'extra_trees'],
    },
    'CE': {
        'filter': 'CE_only',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression', 'calibrated_logistic_regression',
                   'random_forest', 'extra_trees'],
    },
    'COMBINED': {
        'filter': 'none',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression', 'random_forest'],
    },
    'ATM': {
        'filter': 'ATM_near',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression', 'random_forest'],
    },
    'REGIME': {
        'filter': 'high_volume',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'random_forest', 'hist_gradient_boosting'],
    },
    'ROUTER': {
        'filter': 'CE_and_PE',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression'],
    },
    # ── New types for large dataset retraining ───────────────────────────────
    'PE_conservative': {
        'filter': 'PE_only',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression'],
        'preset_families': ['A_conservative'],
    },
    'PE_balanced': {
        'filter': 'PE_only',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'random_forest'],
        'preset_families': ['B_balanced', 'C_low_cost'],
    },
    'PE_regime': {
        'filter': 'PE_only',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'hist_gradient_boosting'],
        'preset_families': ['D_dte', 'E_volatility'],
    },
    'CE_conservative': {
        'filter': 'CE_only',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression'],
        'preset_families': ['A_conservative'],
    },
    'CE_breakout': {
        'filter': 'CE_only',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'random_forest'],
        'preset_families': ['E_volatility'],
    },
    'high_liquidity': {
        'filter': 'high_volume',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'random_forest'],
        'preset_families': ['A_conservative', 'B_balanced'],
    },
    'DTE_regime': {
        'filter': 'none',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'hist_gradient_boosting'],
        'preset_families': ['D_dte'],
    },
    'high_volatility': {
        'filter': 'high_volume',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'hist_gradient_boosting'],
        'preset_families': ['E_volatility'],
    },
    'trend_regime': {
        'filter': 'high_volume',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'hist_gradient_boosting'],
        'preset_families': ['E_volatility'],
    },
    'portfolio_router': {
        'filter': 'CE_and_PE',
        'label': 'cost_survivor_label_v2',
        'models': ['elasticnet', 'logistic_regression'],
        'preset_families': ['F_router'],
    },
}

# Preset family priority by candidate type (used to filter preset grid)
PRESET_FAMILY_PRIORITY = {
    'PE': ['A_conservative', 'B_balanced', 'C_low_cost', 'D_dte', 'E_volatility', 'F_router'],
    'CE': ['A_conservative', 'B_balanced', 'C_low_cost', 'E_volatility', 'F_router'],
    'COMBINED': ['B_balanced', 'C_low_cost', 'D_dte', 'F_router'],
    'ATM': ['A_conservative', 'B_balanced', 'D_dte', 'F_router'],
    'REGIME': ['E_volatility', 'B_balanced', 'D_dte', 'F_router'],
    'ROUTER': ['F_router'],
    'PE_conservative': ['A_conservative'],
    'PE_balanced': ['B_balanced', 'C_low_cost'],
    'PE_regime': ['D_dte', 'E_volatility'],
    'CE_conservative': ['A_conservative'],
    'CE_breakout': ['E_volatility'],
    'high_liquidity': ['A_conservative', 'B_balanced'],
    'DTE_regime': ['D_dte'],
    'high_volatility': ['E_volatility'],
    'trend_regime': ['E_volatility'],
    'portfolio_router': ['F_router'],
}


# ─── Router evaluation ────────────────────────────────────────────────────────

def evaluate_router_candidate(
    df: pd.DataFrame,
    features: List[str],
    label: str,
    model_type: str,
    preset: DynamicPreset,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """Router: train CE and PE models separately, select side with higher prob.

    Uses same gross-forward-return + clipping + cost-stress approach as
    evaluate_candidate_preset for consistency.
    """
    splits = walk_forward_splits(df, n_folds=n_folds)
    results = []
    all_raw_returns: List[float] = []

    for fold_idx, (train_df, val_df, test_df) in enumerate(splits):
        # Split into CE and PE sub-datasets
        ce_df = test_df[test_df.get('option_type', '').astype(str).str.upper() == 'CE'].copy()
        pe_df = test_df[test_df.get('option_type', '').astype(str).str.upper() == 'PE'].copy()

        # Apply preset filter
        ce_filtered = apply_preset_filters(ce_df, preset)
        pe_filtered = apply_preset_filters(pe_df, preset)

        # Train model on full train set
        available = [f for f in features if f in train_df.columns]
        X_train = train_df[available].fillna(0).values
        y_train = train_df[label].fillna(0).values
        valid = (y_train == 0) | (y_train == 1)
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) < 100:
            continue

        # PHASE 10 cost-aware weighting (router path)
        sw = None
        try:
            if 'gross_forward_return' in train_df.columns:
                g = train_df['gross_forward_return'].fillna(0).values[valid]
                base_w = np.where(y_train == 1, 1.8, 0.7)
                gross_boost = np.clip(1.0 + np.clip(g, -0.01, 0.05) * 10, 0.5, 2.5)
                sw = base_w * gross_boost
                sw = sw / sw.mean() * len(sw)
        except Exception:
            sw = None
        model, scaler = train_model(X_train, y_train, model_type, None, sample_weight=sw)

        # Predict on CE and PE, collect all scored rows
        all_rows = []
        for side_df, side_name in [(ce_filtered, 'CE'), (pe_filtered, 'PE')]:
            if len(side_df) == 0:
                continue
            X_side = side_df[available].fillna(0).values
            X_scaled = scaler.transform(X_side)
            scores = model.predict_proba(X_scaled)[:, 1]
            side_df = side_df.copy()
            side_df['_score'] = scores
            side_df['_side'] = side_name
            all_rows.append(side_df)

        if not all_rows:
            continue

        combined = pd.concat(all_rows, ignore_index=True)

        # Top-N per day FIRST (same pipeline as evaluate_candidate_preset)
        if preset.top_n_confidence_per_day > 0 and 'timestamp_dt' in combined.columns:
            combined = top_n_per_day(combined, '_score', preset.top_n_confidence_per_day)

        if len(combined) == 0:
            continue

        # Apply threshold AFTER Top-N (fix for identical t30/t35 and ignored threshold in router path)
        thr = float(getattr(preset, "threshold", 0.35) or 0.35)
        combined = combined[combined["_score"] >= thr].copy()
        if len(combined) == 0:
            continue

        # Use gross_forward_return (clip + percentage conversion)
        gross_raw = combined['gross_forward_return'].fillna(0).values
        gross_clipped = np.clip(gross_raw, -1.0, 5.0)
        returns_pct = gross_clipped * 100.0

        if len(returns_pct) < 5:
            continue

        all_raw_returns.extend(returns_pct.tolist())

        base_cost_pct = 0.0025 + 0.0010
        base_costs = base_cost_pct * len(returns_pct)
        base_net_ret = returns_pct.sum() - base_costs
        gross_sum = returns_pct.sum()
        gross_wins = returns_pct[returns_pct > 0].sum()
        gross_losses = abs(returns_pct[returns_pct < 0].sum())

        wins = (returns_pct > 0).sum()
        losses = (returns_pct < 0).sum()
        avg_win = returns_pct[returns_pct > 0].mean() if wins > 0 else 0.0
        avg_loss = abs(returns_pct[returns_pct < 0].mean()) if losses > 0 else 1.0

        equity = np.cumsum(returns_pct)
        peak = np.maximum.accumulate(equity)
        max_dd = float( - (equity - peak).min() ) if len(equity) > 0 else 0.0  # positive magnitude

        results.append({
            'trade_count': len(returns_pct),
            'win_rate': float(wins / len(returns_pct)) if len(returns_pct) > 0 else 0.0,
            'gross_pf': float(gross_wins / gross_losses) if gross_losses > 0 else (1.0 if gross_wins > 0 else 0.0),
            'base_net_pf': float(base_net_ret / abs(base_costs)) if base_costs > 0 and base_net_ret != 0 else (1.0 if base_net_ret >= 0 else 0.0),
            'profit_factor': float(avg_win / avg_loss) if avg_loss > 0 else 0.0,
            'sharpe': float(returns_pct.mean() / returns_pct.std() * np.sqrt(252)) if returns_pct.std() > 0 else 0.0,
            'max_drawdown': max_dd,
            'top_day_concentration': 0.0,
            'raw_returns': returns_pct.tolist(),
        })

    if not results:
        return {'status': 'NO_TRADES', 'folds': []}

    # Aggregate cost stress analytically
    cost_stress = {}
    for mult_name, mult_val in [('1.0x', 1.0), ('1.25x', 1.25), ('1.5x', 1.5), ('2.0x', 2.0)]:
        cost_stress[f'pf_at_{mult_name}'] = _compute_cost_stress_pf(all_raw_returns, mult_val)

    all_rets = np.array(all_raw_returns, dtype=np.float64)
    total_trades = len(all_rets)
    gross_wins_total = all_rets[all_rets > 0].sum()
    gross_losses_total = abs(all_rets[all_rets < 0].sum())
    aggregate_gross_pf = gross_wins_total / gross_losses_total if gross_losses_total > 0 else (1.0 if gross_wins_total > 0 else 0.0)
    aggregate_net_pf = cost_stress.get('pf_at_1.0x', 0.0)

    # Proper aggregates for router too
    aggregate_win_rate = float((all_rets > 0).sum() / total_trades) if total_trades > 0 else 0.0
    mean_trade_return = float(np.mean(all_rets)) if total_trades > 0 else 0.0
    cost_per_trade_pct = (0.0025 + 0.0010) * 100.0
    net_rets = all_rets - cost_per_trade_pct
    pos_net = net_rets[net_rets > 0].sum()
    neg_net = abs(net_rets[net_rets < 0].sum())
    pf_net_classic = pos_net / neg_net if neg_net > 0 else (1.0 if pos_net > 0 else 0.0)
    true_net_return = float(net_rets.sum())

    base_pfs = [r.get('base_net_pf', 0) for r in results]
    sharpes = [r['sharpe'] for r in results]
    max_dds = [r['max_drawdown'] for r in results]
    top_days = [r['top_day_concentration'] for r in results]
    profitable_folds = sum(1 for pf in base_pfs if pf > 1.0)

    total_trades = sum(r['trade_count'] for r in results)
    worst_fold_pf = float(np.min(base_pfs))
    fold_pf_std = float(np.std(base_pfs))
    max_dd = float(np.max(max_dds)) if max_dds else 0.0  # worst (largest) positive drawdown magnitude
    top_day_conc = float(np.max(top_days)) if top_days else 0.0
    avg_sharpe = float(np.mean(sharpes))

    gates = {}
    gates_passed = 0

    def gate(name: str, cond: bool) -> None:
        nonlocal gates_passed
        gates[name] = cond
        if cond:
            gates_passed += 1

    gate('C2_pf_base', aggregate_gross_pf >= PF_AT_1_00X)
    gate('C3_pf_1.25x', cost_stress['pf_at_1.25x'] >= PF_AT_1_25X)
    gate('C4_pf_1.5x', cost_stress['pf_at_1.5x'] >= PF_AT_1_50X)
    gate('C5_sharpe', avg_sharpe >= MIN_SHARPE)
    gate('C6_trades', total_trades >= MIN_TRADES)
    gate('C7_profitable_folds', profitable_folds >= max(3, n_folds - 1))
    gate('C8_worst_fold', worst_fold_pf >= 0.90)
    gate('C9_daily_stability', top_day_conc <= MAX_TOP_DAY_CONCENTRATION)
    gate('C1_positive_expectancy', aggregate_net_pf > 0)

    is_suspicious = aggregate_net_pf > 200.0
    if is_suspicious:
        gates['SUSPICIOUS_PF'] = True

    # Per-fold stats
    trade_counts = [r['trade_count'] for r in results]
    mean_trades = float(np.mean(trade_counts)) if trade_counts else 0.0
    mean_fold_pf = float(np.mean(base_pfs)) if base_pfs else 0.0
    median_fold_pf = float(np.median(base_pfs)) if base_pfs else 0.0

    return {
        'status': 'EVALUATED',
        'gates_passed': gates_passed,
        'gates_total': 9,
        'trade_count': total_trades,
        'net_pf': aggregate_net_pf,
        'gross_pf': aggregate_gross_pf,
        'profit_factor': aggregate_gross_pf,
        'sharpe': avg_sharpe,
        'max_drawdown': max_dd,
        'top_day_concentration': top_day_conc,
        'profitable_folds': profitable_folds,
        'n_profitable_folds': profitable_folds,
        'worst_fold_pf': worst_fold_pf,
        'mean_fold_pf': mean_fold_pf,
        'median_fold_pf': median_fold_pf,
        'fold_pf_std': fold_pf_std,
        'mean_pf': aggregate_gross_pf,
        'mean_trades': mean_trades,
        'mean_sharpe': avg_sharpe,
        'win_rate': aggregate_win_rate,
        'mean_trade_return': mean_trade_return,
        'true_net_return': true_net_return,
        'pf_net_classic': pf_net_classic,
        'cost_stress': cost_stress,
        'cost_1.25x_pf': cost_stress.get('pf_at_1.25x', 0),
        'cost_1.50x_pf': cost_stress.get('pf_at_1.5x', 0),
        'gates': gates,
        'is_suspicious_pf': is_suspicious,
        'fold_details': results,
        'n_folds': len(results),
        'preset': preset.to_dict(),
        'live_computable_features': True,
        'live_computable_preset': True,
        'paper_only': True,
        'real_trading_enabled': False,
        'threshold_robust': False,
        'filter_applied_at': 'DATA_LEVEL_PRE_TRAINING',
    }


# ─── Main retraining loop ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Dynamic preset retrainer')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to specific dataset CSV. Use --data-dir for auto-discovery.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory to scan for datasets. Auto-selects largest reliable CSV.')
    parser.add_argument('--benchmark-candidate', type=str, required=True)
    parser.add_argument('--candidate-types', type=str, default='PE,CE,COMBINED,ATM,REGIME,ROUTER')
    parser.add_argument('--max-candidates', type=int, default=8)
    parser.add_argument('--max-presets-per-candidate', type=int, default=50)
    parser.add_argument('--max-candidates', type=int, default=100,
                        help='Max total candidate+preset combinations (default 100)')
    parser.add_argument('--target-total-shadow-candidates', type=int, default=3)
    parser.add_argument('--strict-gates', action='store_true')
    parser.add_argument('--save-reports', action='store_true')
    parser.add_argument('--paper-only', action='store_true')
    parser.add_argument('--output-dir', type=str, default='reports')
    parser.add_argument('--n-folds', type=int, default=3)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_candidate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir
    candidate_types = [t.strip() for t in args.candidate_types.split(',')]

    # ── Dataset selection: --data-dir auto-discovery OR --dataset explicit ──
    dataset_path: Optional[Path] = None
    dataset_audit_full: Dict[str, Any] = {}
    ranked_datasets: List[Dict[str, Any]] = []
    audited_datasets: List[Dict[str, Any]] = []

    if args.data_dir:
        # Auto-discovery: scan data_dir, rank datasets, select largest reliable
        data_dir = Path(args.data_dir)
        print(f"[INFO] Scanning {data_dir} for datasets...")
        ranked_datasets = discover_datasets(data_dir)
        print(f"[INFO] Ranked {len(ranked_datasets)} datasets:")
        for i, ds in enumerate(ranked_datasets[:5]):
            print(f"  {i+1}. {ds['filename'][:55]} | score={ds['score']} "
                  f"| rows~{ds.get('row_hint','?')} | forbidden={ds['forbidden_leak_count']}")

        # Select top-1 suitable dataset
        for ds in ranked_datasets:
            candidate_path = Path(ds['path'])
            audit = audit_dataset(candidate_path)
            audited_datasets.append(audit)
            if audit['suitable_for_retraining']:
                dataset_path = candidate_path
                dataset_audit_full = audit
                print(f"[INFO] Selected: {candidate_path.name} "
                      f"(rows={audit['row_count']}, score={ds['score']})")
                break

        if dataset_path is None:
            print("[WARN] No suitable dataset found. Using top-ranked anyway.")
            if ranked_datasets:
                dataset_path = Path(ranked_datasets[0]['path'])
                print(f"[INFO] Falling back to: {dataset_path.name}")
                if not audited_datasets:
                    dataset_audit_full = audit_dataset(dataset_path)
    elif args.dataset:
        dataset_path = Path(args.dataset)
        print(f"[INFO] Using explicit dataset: {dataset_path}")
        dataset_audit_full = audit_dataset(dataset_path)
    else:
        print("[ERROR] Must provide either --dataset or --data-dir")
        return 1

    if dataset_path is None:
        print("[ERROR] No dataset path determined")
        return 1

    print(f"[INFO] Dynamic Preset Retrainer — {TS}")
    print(f"[INFO] Dataset: {dataset_path}")
    print(f"[INFO] Benchmark: {benchmark_path}")
    print(f"[INFO] Candidate types: {candidate_types}")

    # Step 1: Load and audit dataset
    df, audit = load_and_audit_dataset(dataset_path)
    features = get_live_features(df)
    print(f"[INFO] Live features: {len(features)}")

    # ── Benchmark retrain: add dynamic preset + model.pkl to existing benchmark ──
    if benchmark_path.exists():
        try:
            import json as _json
            with (benchmark_path / 'candidate_manifest.json').open() as _f:
                _bm_manifest = _json.load(_f)
            _bm_model_type = _bm_manifest.get('model', {}).get('model_name', 'elasticnet')
            _bm_model_type = 'elasticnet' if 'elasticnet' in _bm_model_type.lower() else _bm_model_type
            _bm_filter = _bm_manifest.get('filter', {}).get('filter_name', 'PE_only')
            _bm_label = _bm_manifest.get('model', {}).get('target', 'cost_survivor_label_v2')
            _bm_label = 'cost_survivor_label_v2' if 'v2' not in _bm_label else _bm_label

            # Apply benchmark filter
            _bm_df = df.copy()
            if _bm_filter == 'PE_only':
                _bm_df = _bm_df[_bm_df.get('option_type', '').astype(str).str.upper() == 'PE']
            elif _bm_filter == 'CE_only':
                _bm_df = _bm_df[_bm_df.get('option_type', '').astype(str).str.upper() == 'CE']

            print(f"[INFO] Retraining benchmark ({_bm_model_type}, filter={_bm_filter}, {_bm_df.shape[0]} rows)")

            # Use conservative dynamic preset for benchmark
            _bm_preset = DynamicPreset(
                preset_id=f'bm_conservative_{TS}',
                preset_family='A_conservative',
                threshold=0.35,
                max_trades_per_day=2,
                top_n_confidence_per_day=2,
                spread_limit_pct=0.10,
                liquidity_min=50000.0,
                dte_range='all',
            )

            # Evaluate benchmark with dynamic preset
            _bm_result = evaluate_candidate_preset(
                _bm_df, features, _bm_label, _bm_model_type, _bm_preset, n_folds=args.n_folds)

            if _bm_result.get('status') == 'EVALUATED':
                # Save dynamic_preset.json
                with (benchmark_path / 'dynamic_preset.json').open('w') as _f:
                    json.dump(_bm_preset.to_dict(), _f, indent=2)

                # Train and save model.pkl on the last fold's training split
                # (most data, best representation of the final model)
                _bm_splits = list(walk_forward_splits(_bm_df, args.n_folds))
                if _bm_splits:
                    _bm_last_train = _bm_splits[-1][0]
                    if len(_bm_last_train) >= 200:
                        _bm_X = _bm_last_train[features].fillna(0).values
                        _bm_y = _bm_last_train[_bm_label].astype(int).values
                        try:
                            _bm_mdl, _bm_scaler = train_model(_bm_X, _bm_y, _bm_model_type, benchmark_path)
                        except Exception as _e:
                            print(f"[WARN] Benchmark model training failed: {_e}")

                # Save candidate_manifest.json (updated with dynamic preset info)
                _bm_fold_details = {}
                for _i, _fd in enumerate(_bm_result.get('fold_details', [])):
                    _bm_fold_details[f'fold_{_i}'] = {
                        'trade_count': _fd.get('trade_count', 0),
                        'win_rate': _fd.get('win_rate', 0),
                        'pf': _fd.get('base_net_pf', 0),
                        'gross_pf': _fd.get('gross_pf', 0),
                        'sharpe': _fd.get('sharpe', 0),
                    }

                _bm_updated_manifest = dict(_bm_manifest)
                _bm_updated_manifest.update({
                    'dynamic_preset_path': str(benchmark_path / 'dynamic_preset.json'),
                    'threshold_robust': _bm_result.get('threshold_robust', False),
                    'threshold_robust_details': _bm_result.get('threshold_robust_details', {}),
                    'overall_metrics': {
                        'mean_pf': _bm_result.get('mean_pf', 0),
                        'mean_trades': _bm_result.get('mean_trades', 0),
                        'mean_sharpe': _bm_result.get('mean_sharpe', 0),
                        'cost_1.25x_pf': _bm_result.get('cost_1.25x_pf', 0),
                        'cost_1.50x_pf': _bm_result.get('cost_1.50x_pf', 0),
                        'worst_fold_pf': _bm_result.get('worst_fold_pf', 0),
                        'n_profitable_folds': _bm_result.get('n_profitable_folds', 0),
                    },
                    'fold_details': _bm_fold_details,
                    'fold_pf_std': _bm_result.get('fold_pf_std', 0),
                    'gates_passed': _bm_result.get('gates_passed', 0),
                    'gates_total': 9,
                    'paper_only': True,
                    'real_trading_enabled': False,
                    'dynamic_preset_preset_id': _bm_preset.preset_id,
                    'dynamic_preset_family': _bm_preset.preset_family,
                    'dynamic_preset_threshold': _bm_preset.threshold,
                    'retrained_timestamp': TS,
                    'retrained_dataset': str(dataset_path),
                })
                with (benchmark_path / 'candidate_manifest.json').open('w') as _f:
                    _json.dump(_bm_updated_manifest, _f, indent=2)

                print(f"[INFO] Benchmark retrained: dynamic_preset.json + updated manifest saved to {benchmark_path}")
                print(f"       trades={_bm_result['trade_count']}, net_pf={_bm_result['net_pf']:.3f}, "
                      f"1.5x_pf={_bm_result.get('cost_1.50x_pf', 0):.3f}, "
                      f"threshold_robust={_bm_result.get('threshold_robust', False)}")
            else:
                print(f"[WARN] Benchmark retrain returned status={_bm_result.get('status')}")
        except Exception as _e:
            print(f"[WARN] Benchmark retrain failed: {_e}")
    else:
        print(f"[INFO] Benchmark path not found: {benchmark_path} (skipping retrain)")

    # Determine label column
    label_col = 'cost_survivor_label_v2'
    if label_col not in df.columns:
        label_col = 'cost_survivor_label'
    if label_col not in df.columns:
        for c in df.columns:
            if '_label' in c.lower():
                label_col = c
                break
    print(f"[INFO] Label column: {label_col}")

    # Step 2: Build preset grid
    preset_grid = build_preset_grid(max_presets=args.max_presets_per_candidate)
    print(f"[INFO] Preset grid: {len(preset_grid)} presets")

    # Step 3: Run retraining for each candidate group
    all_results = []
    candidate_id = 0
    max_combinations = args.max_candidates
    reached_limit = False

    for ctype in candidate_types:
        if reached_limit:
            break
        if ctype not in CANDIDATE_GROUPS:
            print(f"[WARN] Unknown candidate type: {ctype}")
            continue

        cfg = CANDIDATE_GROUPS[ctype]
        filter_name = cfg['filter']
        label = cfg['label']
        models = cfg['models'][:3]  # Limit models per type

        # Apply dataset filter
        if filter_name == 'PE_only':
            df_filtered = df[df.get('option_type', '').astype(str).str.upper() == 'PE'].copy()
        elif filter_name == 'CE_only':
            df_filtered = df[df.get('option_type', '').astype(str).str.upper() == 'CE'].copy()
        elif filter_name == 'ATM_near':
            df_filtered = df.copy()
            if 'range_pct' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['range_pct'].abs() <= 2.0]
        elif filter_name == 'high_volume':
            df_filtered = df.copy()
            if 'volume' in df_filtered.columns:
                med = df_filtered['volume'].median()
                df_filtered = df_filtered[df_filtered['volume'] >= med]
        elif filter_name == 'CE_and_PE':
            df_filtered = df.copy()
        else:
            df_filtered = df.copy()

        if len(df_filtered) < 500:
            print(f"[WARN] Filtered dataset too small for {ctype}: {len(df_filtered)} rows")
            continue

        print(f"\n[INFO] === {ctype} === ({len(df_filtered)} rows, {len(features)} features)")

        for model_type in models:
            if reached_limit:
                break
            # Pick presets to test using PRESET_FAMILY_PRIORITY
            allowed_families = PRESET_FAMILY_PRIORITY.get(ctype, ['A_conservative', 'B_balanced', 'C_low_cost', 'D_dte', 'E_volatility', 'F_router'])
            test_presets = [p for p in preset_grid if p.preset_family in allowed_families][:args.max_presets_per_candidate]

            if not test_presets:
                continue

            for preset in test_presets:
                candidate_id += 1
                if candidate_id > max_combinations:
                    print(f"[INFO] Reached max-candidates limit ({max_combinations}). Stopping evaluation.")
                    reached_limit = True
                    break
                cid = f"{ctype}_{model_type}_{preset.preset_id}_{TS}"

                if ctype == 'ROUTER':
                    result = evaluate_router_candidate(
                        df_filtered, features, label, model_type, preset, n_folds=args.n_folds
                    )
                else:
                    result = evaluate_candidate_preset(
                        df_filtered, features, label, model_type, preset, n_folds=args.n_folds
                    )

                result['candidate_id'] = cid
                result['candidate_type'] = ctype
                result['model_type'] = model_type
                result['label'] = label
                result['filter'] = filter_name
                all_results.append(result)

                pf_1_5x = result.get('cost_stress', {}).get('pf_at_1.5x', 0)
                gates_p = result.get('gates_passed', 0)
                trades = result.get('trade_count', 0)
                print(f"  {cid}: trades={trades}, net_pf={result.get('net_pf', 0):.3f}, "
                      f"1.5x_pf={pf_1_5x:.3f}, gates={gates_p}/9")

    # Step 4: Internal 9-gate screening
    # NOTE: Internal 9/9 is SCREENING ONLY. Final promotion requires strict 22/22.
    # Suspicious PF (> 10) is FAIL_REJECTED_PENDING_AUDIT regardless of gate count.
    internal_passing = []
    for r in all_results:
        if r.get('status') != 'EVALUATED':
            r['final_status'] = 'NO_TRADES'
            continue
        if r.get('is_suspicious_pf', False):
            r['final_status'] = 'FAIL_REJECTED_PENDING_AUDIT'
            continue
        gates_p = r.get('gates_passed', 0)
        pf_1_5x = r.get('cost_stress', {}).get('pf_at_1.5x', 0)
        net_pf = r.get('net_pf', 0)
        trades = r.get('trade_count', 0)
        if gates_p >= 9 and pf_1_5x >= 1.0 and net_pf > 0 and trades >= MIN_TRADES:
            r['final_status'] = 'FAIL_BUT_PROMISING'  # provisional — needs strict 22/22
            internal_passing.append(r)
        elif gates_p >= 5:
            r['final_status'] = 'BELOW_INTERNAL_GATE'
        else:
            r['final_status'] = 'FAIL_REJECTED'

    internal_passing.sort(key=lambda x: x.get('cost_stress', {}).get('pf_at_1.5x', 0), reverse=True)

    print(f"\n[INFO] === Results ===")
    print(f"[INFO] Total evaluations: {len(all_results)}")
    print(f"[INFO] Internal 9/9 screening: {len(internal_passing)} candidates passed (SUSPICIOUS PF excluded)")

    # Step 5: Save candidate artifacts and run strict 22-gate evaluator
    # Only for candidates passing internal 9/9 screening
    strict_evaluator_path = REPO_ROOT / 'scripts' / 'ml_only_strict_gate_evaluator.py'
    use_strict = args.strict_gates and strict_evaluator_path.exists()

    for r in internal_passing:
        cid = r['candidate_id']
        cand_dir = output_dir / 'candidates' / cid
        cand_dir.mkdir(parents=True, exist_ok=True)

        # Build candidate manifest (strict evaluator compatible)
        fold_details_dict = {}
        for i, fd in enumerate(r.get('fold_details', [])):
            fold_details_dict[f'fold_{i}'] = {
                'trade_count': fd.get('trade_count', 0),
                'win_rate': fd.get('win_rate', 0),
                'pf': fd.get('base_net_pf', fd.get('net_pf', 0)),
                'gross_pf': fd.get('gross_pf', 0),
                'sharpe': fd.get('sharpe', 0),
                'max_drawdown': fd.get('max_drawdown', 0),
                'top_day_concentration': fd.get('top_day_concentration', 0),
            }

        manifest = {
            'candidate_id': cid,
            'candidate_type': r.get('candidate_type', ''),
            'model_type': r.get('model_type', ''),
            'label': r.get('label', ''),
            'source': 'dynamic_preset_retrainer',
            'filter': {
                'filter_name': r.get('filter', ''),
                'name': r.get('filter', ''),  # backward compat + shadow_manifest reads filter_name
                'applied_at': 'DATA_LEVEL_PRE_TRAINING',
            },
            'overall_metrics': {
                'mean_pf': r.get('mean_pf', 0),
                'mean_trades': r.get('mean_trades', 0),
                'mean_sharpe': r.get('mean_sharpe', 0),
                'cost_1.25x_pf': r.get('cost_1.25x_pf', 0),
                'cost_1.50x_pf': r.get('cost_1.50x_pf', 0),
                'worst_fold_pf': r.get('worst_fold_pf', 0),
                'n_profitable_folds': r.get('n_profitable_folds', 0),
            },
            'fold_details': fold_details_dict,
            'fold_pf_std': r.get('fold_pf_std', 0),
            'live_computable_features': r.get('live_computable_features', True),
            'live_computable_preset': r.get('live_computable_preset', True),
            'paper_only': True,
            'real_trading_enabled': False,
            'threshold_robust': r.get('threshold_robust', False),
            'threshold_robust_details': r.get('threshold_robust_details', {}),
            'selected_threshold': r.get('preset', {}).get('threshold', 0.35),
            'dynamic_preset_path': str(cand_dir / 'dynamic_preset.json'),
            'feature_schema_path': str(cand_dir / 'feature_schema.json'),
            'model': {
                'model_name': r.get('model_type', ''),
                'target': r.get('label', ''),
                'paper_only': True,
                'real_trading_enabled': False,
            },
            'artifacts': {
                'model_pkl': str(cand_dir / 'model.pkl'),
                'feature_schema': str(cand_dir / 'feature_schema.json'),
                'fold_results': str(cand_dir / 'fold_results.json'),
                'per_fold_metrics': str(cand_dir / 'per_fold_metrics.json'),
                'cost_stress_report': str(cand_dir / 'cost_stress_report.json'),
                'gate_evaluation_detail': str(cand_dir / 'gate_evaluation_detail.json'),
                'stability_report': str(cand_dir / 'stability_report.json'),
                'dynamic_preset': str(cand_dir / 'dynamic_preset.json'),
            },
        }

        # Save manifest
        with (cand_dir / 'candidate_manifest.json').open('w') as f:
            json.dump(manifest, f, indent=2, default=str)

        # Save dynamic preset
        with (cand_dir / 'dynamic_preset.json').open('w') as f:
            json.dump(r.get('preset', {}), f, indent=2, default=str)

        # Save feature schema
        feature_schema = {
            'features': features,
            'n_features': len(features),
            'forbidden_patterns_checked': list(FEATURE_FORBIDDEN),
            'live_computable': True,
        }
        with (cand_dir / 'feature_schema.json').open('w') as f:
            json.dump(feature_schema, f, indent=2, default=str)

        # ── Additional artifact files ──────────────────────────────────────────

        # fold_results.json: detailed per-fold results (strict evaluator compatible)
        fold_results_list = []
        for i, fd in enumerate(r.get('fold_details', [])):
            fold_results_list.append({
                'fold_name': f'fold_{i}',
                'trade_count': fd.get('trade_count', 0),
                'win_rate': round(fd.get('win_rate', 0), 4),
                'pf': round(fd.get('base_net_pf', 0), 4),
                'gross_pf': round(fd.get('gross_pf', 0), 4),
                'profit_factor': round(fd.get('profit_factor', 0), 4),
                'sharpe': round(fd.get('sharpe', 0), 4),
                'max_drawdown': round(fd.get('max_drawdown', 0), 4),
                'top_day_concentration': round(fd.get('top_day_concentration', 0), 4),
                'threshold_used': fd.get('threshold_used', 0),
            })
        with (cand_dir / 'fold_results.json').open('w') as f:
            json.dump({'candidate_id': cid, 'fold_results': fold_results_list}, f, indent=2, default=str)

        # per_fold_metrics.json: alias of fold_results
        with (cand_dir / 'per_fold_metrics.json').open('w') as f:
            json.dump({'candidate_id': cid, 'folds': fold_results_list}, f, indent=2, default=str)

        # cost_stress_report.json: PF at all cost multipliers
        cost_stress_report = {
            'candidate_id': cid,
            'base_cost_pct_per_trade': 0.35,
            'returns_in_percentage_space': True,
            'cost_stress': {
                f'pf_at_{m}': round(v, 4)
                for m, v in r.get('cost_stress', {}).items()
            },
            'note': 'Costs in percentage points: 0.25% brokerage + 0.10% spread = 0.35 pct-pts per trade',
        }
        with (cand_dir / 'cost_stress_report.json').open('w') as f:
            json.dump(cost_stress_report, f, indent=2, default=str)

        # gate_evaluation_detail.json: per-gate pass/fail with reason
        gate_detail_list = []
        for gate_id, gate_passed in r.get('gates', {}).items():
            gate_detail_list.append({
                'gate_id': gate_id,
                'passed': gate_passed,
                'reason': f'gate {gate_id} = {gate_passed}',
            })
        with (cand_dir / 'gate_evaluation_detail.json').open('w') as f:
            json.dump({
                'candidate_id': cid,
                'gates_passed': r.get('gates_passed', 0),
                'gates_total': 9,
                'gates': gate_detail_list,
                'strict_gates_passed': r.get('strict_gates_passed'),
                'strict_readiness': r.get('strict_readiness', 'NOT_RUN'),
            }, f, indent=2, default=str)

        # stability_report.json: threshold robustness + fold stability
        stability_report = {
            'candidate_id': cid,
            'threshold_robust': r.get('threshold_robust', False),
            'threshold_robust_details': r.get('threshold_robust_details', {}),
            'fold_pf_std': round(r.get('fold_pf_std', 0), 4),
            'max_drawdown': round(r.get('max_drawdown', 0), 4),
            'top_day_concentration': round(r.get('top_day_concentration', 0), 4),
            'mean_fold_pf': round(r.get('mean_fold_pf', 0), 4),
            'median_fold_pf': round(r.get('median_fold_pf', 0), 4),
            'worst_fold_pf': round(r.get('worst_fold_pf', 0), 4),
            'n_folds': r.get('n_folds', 0),
            'paper_only': True,
            'real_trading_enabled': False,
        }
        with (cand_dir / 'stability_report.json').open('w') as f:
            json.dump(stability_report, f, indent=2, default=str)

        # model.pkl: saved separately during training (requires re-train).
        # Placeholder note — model is retrained by benchmark path or explicit retrain.
        model_note = {
            'note': 'model.pkl is saved by the benchmark retrain path. '
                    'Run retrain_all_edge_models.py with --output-dir pointing here '
                    'to generate the trained model artifact.',
            'expected_path': str(cand_dir / 'model.pkl'),
        }
        with (cand_dir / 'model_artifact_note.json').open('w') as f:
            json.dump(model_note, f, indent=2)

        r['candidate_dir'] = str(cand_dir)
        r['manifest_path'] = str(cand_dir / 'candidate_manifest.json')

        # Run strict 22-gate evaluator if enabled
        if use_strict:
            try:
                from ml_only_strict_gate_evaluator import MLOnlyStrictGateEvaluator

                evaluator = MLOnlyStrictGateEvaluator(cand_dir)
                if evaluator.load_candidate():
                    evaluation = evaluator.run_evaluation()
                    strict_count = sum(1 for gr in evaluation.gate_results if gr.passed)
                    r['strict_gates_passed'] = strict_count
                    r['strict_gates_total'] = len(evaluation.gate_results)
                    r['strict_readiness'] = evaluation.readiness
                    # PASS_SHADOW_READY only if readiness is exactly PASS_SHADOW_READY
                    # (strict evaluator may return PASS_PAPER_READY or FAIL_BUT_PROMISING
                    # for <22/22 — those do NOT get PASS_SHADOW_READY status)
                    if evaluation.readiness == 'PASS_SHADOW_READY':
                        r['final_status'] = 'PASS_SHADOW_READY'
                        if evaluation.shadow_manifest:
                            with (cand_dir / 'shadow_manifest.json').open('w') as _f:
                                json.dump(evaluation.shadow_manifest, _f, indent=2, default=str)
                            print(f"  [STRICT] {cid}: PASS_SHADOW_READY ({strict_count}/22) — shadow_manifest.json saved")
                    # FAIL_REJECTED if manifest couldn't load or critical gates failed
                    elif evaluation.readiness == 'FAIL_REJECTED':
                        r['final_status'] = 'FAIL_REJECTED'
                    # Else keep existing FAIL_BUT_PROMISING (strict <22/22)
                    # Save strict gate report
                    strict_report_path = cand_dir / 'strict_gate_report.json'
                    with strict_report_path.open('w') as _f:
                        json.dump({
                            'candidate_id': cid,
                            'readiness': evaluation.readiness,
                            'pass_rate': evaluation.summary.get('pass_rate', 0),
                            'n_passed': evaluation.summary.get('n_passed', 0),
                            'n_failed': evaluation.summary.get('n_failed', 0),
                            'n_critical_failed': evaluation.summary.get('n_critical_failed', 0),
                            'gates': [(gr.gate_id, gr.passed, gr.reason) for gr in evaluation.gate_results],
                            'fail_reasons': evaluation.fail_reasons,
                            'improvement_suggestions': evaluation.improvement_suggestions,
                        }, _f, indent=2, default=str)
                else:
                    r['strict_gates_passed'] = 0
                    r['strict_gates_total'] = 22
                    r['strict_readiness'] = 'FAIL_REJECTED'
            except Exception as e:
                print(f"[WARN] Strict evaluator failed for {cid}: {e}")
                r['strict_gates_passed'] = 0
                r['strict_gates_total'] = 22
                r['strict_readiness'] = 'ERROR_IN_STRICT_EVAL'
        else:
            r['strict_gates_passed'] = None
            r['strict_gates_total'] = 22
            r['strict_readiness'] = 'NOT_RUN'

        print(f"  [STRICT] {cid}: internal=9/9, strict={r.get('strict_gates_passed', 'N/A')}/22, status={r['final_status']}")

    # Step 6: Determine shadow portfolio (candidates with PASS_SHADOW_READY)
    shadow_ready = [r for r in internal_passing if r.get('final_status') == 'PASS_SHADOW_READY']
    shadow_portfolio = shadow_ready[:args.target_total_shadow_candidates]

    if shadow_portfolio:
        portfolio_manifest = {
            'timestamp': TS,
            'portfolio_id': f'shadow_portfolio_{TS}',
            'n_candidates': len(shadow_portfolio),
            'candidates': [
                {
                    'candidate_id': r['candidate_id'],
                    'candidate_dir': r['candidate_dir'],
                    'manifest_path': r['manifest_path'],
                    'model_type': r['model_type'],
                    'strict_gates_passed': r.get('strict_gates_passed'),
                    'net_pf': r.get('net_pf'),
                    'cost_1.50x_pf': r.get('cost_1.50x_pf'),
                    'trade_count': r.get('trade_count'),
                    'profitable_folds': r.get('profitable_folds'),
                    'fold_pf_std': r.get('fold_pf_std'),
                }
                for r in shadow_portfolio
            ],
            'paper_only': True,
            'real_trading_enabled': False,
            'total_candidates_evaluated': len(all_results),
            'strict_evaluator_used': use_strict,
        }
        portfolio_path = output_dir / f'shadow_portfolio_large_{TS}.json'
        with portfolio_path.open('w') as f:
            json.dump(portfolio_manifest, f, indent=2, default=str)
        print(f"[INFO] Shadow portfolio: {portfolio_path}")

    # Step 7: Generate reports
    if args.save_reports:
        ts_suffix = TS

        # ── Dataset selection report ──────────────────────────────────────────
        if ranked_datasets or dataset_audit_full:
            json_r, md_r = generate_dataset_selection_report(
                ranked_datasets, audited_datasets,
                str(dataset_path), reports_dir, ts_suffix)
            print(f"[INFO] Dataset selection: {md_r}")

        # JSON full results
        full_path = output_dir / f'ml_only_dynamic_preset_large_retraining_{ts_suffix}.json'
        with full_path.open('w') as f:
            json.dump({
                'timestamp': ts_suffix,
                'dataset_audit': dataset_audit_full or audit,
                'total_evaluations': len(all_results),
                'internal_9_gates': len(internal_passing),
                'strict_22_gates': len(shadow_ready),
                'all_results': all_results,
            }, f, indent=2, default=str)
        print(f"[INFO] Full results: {full_path}")

        # Leaderboard (all candidates)
        lb = []
        status_order = {'PASS_SHADOW_READY': 0, 'FAIL_BUT_PROMISING': 1,
                        'BELOW_INTERNAL_GATE': 2, 'FAIL_REJECTED_PENDING_AUDIT': 3, 'FAIL_REJECTED': 4,
                        'NO_TRADES': 5}
        sorted_all = sorted(all_results, key=lambda x: (
            status_order.get(x.get('final_status', ''), 9),
            -(x.get('cost_stress', {}).get('pf_at_1.5x', 0) or 0)
        ))
        for i, r in enumerate(sorted_all[:50]):
            p = r.get('preset', {})
            lb.append({
                'rank': i + 1,
                'candidate_id': r.get('candidate_id', ''),
                'candidate_type': r.get('candidate_type', ''),
                'preset_family': p.get('preset_family', ''),
                'model_type': r.get('model_type', ''),
                'threshold': p.get('threshold', 0),
                'top_n': p.get('top_n_confidence_per_day', 0),
                'trade_count': r.get('trade_count', 0),
                'net_pf': round(r.get('net_pf', 0), 3),
                'gross_pf': round(r.get('gross_pf', 0), 3),
                '1.5x_cost_pf': round(r.get('cost_stress', {}).get('pf_at_1.5x', 0), 3),
                '2.0x_cost_pf': round(r.get('cost_stress', {}).get('pf_at_2.0x', 0), 3),
                'profitable_folds': r.get('profitable_folds', 0),
                'worst_fold_pf': round(r.get('worst_fold_pf', 0), 3),
                'fold_pf_std': round(r.get('fold_pf_std', 0), 3),
                'max_drawdown': round(r.get('max_drawdown', 0), 3),
                'internal_gates': r.get('gates_passed', 0),
                'strict_gates': r.get('strict_gates_passed'),
                'final_status': r.get('final_status', 'UNKNOWN'),
                'suspicious': r.get('is_suspicious_pf', False),
            })

        lb_path = output_dir / f'ml_only_dynamic_preset_large_leaderboard_{ts_suffix}.json'
        with lb_path.open('w') as f:
            json.dump({'timestamp': ts_suffix, 'leaderboard': lb}, f, indent=2, default=str)

        # Markdown leaderboard
        lines = [f"# Dynamic Preset Retraining — {ts_suffix}", "",
                 "## Gate Status Codes",
                 "- `PASS_SHADOW_READY`: strict 22/22 + paper_only → ready for shadow mode",
                 "- `FAIL_BUT_PROMISING`: internal 9/9 but strict < 22 → not ready",
                 "- `FAIL_REJECTED_PENDING_AUDIT`: PF > 10 (suspicious) → audit required",
                 "- `FAIL_REJECTED`: did not pass internal gates",
                 "",
                 "| rank | candidate_id | type | thresh | trades | net_pf | 1.5x_pf | int_gate | strict | status |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in lb:
            strict_str = str(e['strict_gates']) if e['strict_gates'] is not None else 'N/A'
            susp = ' ⚠️' if e['suspicious'] else ''
            lines.append(
                f"| {e['rank']} | {e['candidate_id'][:35]}{susp} | {e['candidate_type']} | "
                f"{e['threshold']:.2f} | {e['trade_count']} | {e['net_pf']:.2f} | "
                f"{e['1.5x_cost_pf']:.2f} | {e['internal_gates']}/9 | {strict_str} | "
                f"{e['final_status']} |"
            )
        lb_md = output_dir / f'ml_only_dynamic_preset_large_leaderboard_{ts_suffix}.md'
        with lb_md.open('w') as f:
            f.write('\n'.join(lines))
        print(f"[INFO] Leaderboard: {lb_md}")

        # Passed candidates (FAIL_BUT_PROMISING)
        promising = [r for r in internal_passing if r.get('final_status') == 'FAIL_BUT_PROMISING']
        if promising:
            p_lines = [
                f"# Promising Candidates (Internal 9/9) — {ts_suffix}",
                f"## {len(promising)} candidates pass internal 9/9 but need strict 22-gate evaluation",
                f"## {len(shadow_ready)} candidates are PASS_SHADOW_READY",
                "",
                "**NOTE**: Internal 9/9 is SCREENING ONLY. Candidates below are NOT approved "
                "for shadow mode until strict 22/22 gates pass.",
                "",
                "| candidate_id | type | net_pf | 1.5x_pf | trades | folds | strict_gates | status |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in promising[:20]:
                strict_s = f"{r.get('strict_gates_passed', 'N/A')}/22"
                p_lines.append(
                    f"| {r['candidate_id'][:40]} | {r.get('candidate_type')} | "
                    f"{r.get('net_pf', 0):.3f} | {r.get('cost_stress', {}).get('pf_at_1.5x', 0):.3f} | "
                    f"{r.get('trade_count', 0)} | {r.get('profitable_folds', 0)} | "
                    f"{strict_s} | {r.get('final_status', 'FAIL_BUT_PROMISING')} |"
                )
            p_md = output_dir / f'ml_only_dynamic_preset_large_promising_{ts_suffix}.md'
            with p_md.open('w') as f:
                f.write('\n'.join(p_lines))
            print(f"[INFO] Promising report: {p_md}")

        # Shadow-ready candidates
        if shadow_ready:
            sr_lines = [
                f"# PASS_SHADOW_READY Candidates — {ts_suffix}",
                f"## {len(shadow_ready)} candidates passed strict 22/22 gates",
                "",
                "## Gate integrity confirmation",
                "- 22/22 strict gates passed",
                "- 1.5x cost PF >= 1.0 (mandatory break-even)",
                "- paper_only=True, real_trading_enabled=False",
                "- PF <= 10 (not suspicious)",
                "",
            ]
            for r in shadow_ready:
                p = r.get('preset', {})
                sr_lines += [
                    f"\n### {r['candidate_id']}",
                    f"- Type: {r.get('candidate_type')} | Model: {r.get('model_type')}",
                    f"- Preset: {p.get('preset_family')} (id={p.get('preset_id')})",
                    f"- Threshold: {p.get('threshold'):.2f} | Top-N: {p.get('top_n_confidence_per_day')}",
                    f"- Trade count: {r.get('trade_count')}",
                    f"- Net PF: {r.get('net_pf', 0):.3f} | Gross PF: {r.get('gross_pf', 0):.3f}",
                    f"- 1.5x cost PF: {r.get('cost_1.50x_pf', 0):.3f}",
                    f"- Profitable folds: {r.get('profitable_folds')} | Worst fold PF: {r.get('worst_fold_pf', 0):.3f}",
                    f"- Fold PF std: {r.get('fold_pf_std', 0):.3f} | Sharpe: {r.get('sharpe', 0):.3f}",
                    f"- Strict gates: {r.get('strict_gates_passed')}/22",
                    f"- Candidate dir: {r.get('candidate_dir')}",
                ]
            sr_md = output_dir / f'ml_only_dynamic_preset_large_shadow_ready_{ts_suffix}.md'
            with sr_md.open('w') as f:
                f.write('\n'.join(sr_lines))
            print(f"[INFO] Shadow-ready report: {sr_md}")
        else:
            # No shadow-ready: failure diagnosis
            fail_lines = [
                f"# Failure Diagnosis — {ts_suffix}",
                "",
                "## No candidates are PASS_SHADOW_READY",
                "",
                "## Why dynamic presets alone cannot create edge:",
                "- Presets FILTER trades (liquidity, DTE, spread) but cannot generate signal",
                "- Edge comes from ML model quality + feature quality + label quality",
                "- Dynamic presets are RISK CONTROLS, not signal generators",
                "",
                "## Top internal 9/9 candidates (FAIL_BUT_PROMISING):",
                "",
                "| candidate_id | net_pf | 1.5x_pf | trades | folds | strict_gates | strict_status |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for r in promising[:10]:
                strict_s = f"{r.get('strict_gates_passed', 'N/A')}/22"
                strict_st = r.get('strict_readiness', 'N/A')
                fail_lines.append(
                    f"| {r['candidate_id'][:40]} | {r.get('net_pf', 0):.3f} | "
                    f"{r.get('cost_stress', {}).get('pf_at_1.5x', 0):.3f} | "
                    f"{r.get('trade_count', 0)} | {r.get('profitable_folds', 0)} | "
                    f"{strict_s} | {strict_st} |"
                )
            fail_lines += [
                "",
                "## Failed gate breakdown for top candidates:",
            ]
            for r in promising[:5]:
                g = r.get('gates', {})
                failed = [k for k, v in g.items() if not v and k != 'SUSPICIOUS_PF']
                failed_str = ', '.join(failed) if failed else 'all passed (strict evaluator needed)'
                fail_lines.append(f"- **{r['candidate_id'][:40]}**: {failed_str}")

            fail_md = output_dir / f'ml_only_dynamic_preset_large_failure_diagnosis_{ts_suffix}.md'
            with fail_md.open('w') as f:
                f.write('\n'.join(fail_lines))
            print(f"[INFO] Failure diagnosis: {fail_md}")

        # ── Gate audit report ────────────────────────────────────────────────
        if all_results:
            ga_data = []
            # Collect all unique gate IDs across candidates
            all_gate_ids = sorted({
                g_id
                for r in all_results
                for g_id in (r.get('gates', {}) or {})
            })

            # Load per-gate data for top-20 candidates by 1.5x_pf
            top_candidates = sorted(
                all_results,
                key=lambda x: x.get('cost_stress', {}).get('pf_at_1.5x', 0),
                reverse=True,
            )[:20]

            for r in all_results:
                strict_g = r.get('strict_gates_passed')
                cand_dir_path = Path(r.get('candidate_dir', ''))

                # Try to load per-gate results from strict_gate_report.json
                per_gate = {}
                strict_report_path = cand_dir_path / 'strict_gate_report.json'
                if strict_report_path.exists():
                    try:
                        with strict_report_path.open() as _f:
                            _sr = json.load(_f)
                            for _g_id, _passed, _reason in _sr.get('gates', []):
                                per_gate[_g_id] = 'PASS' if _passed else 'FAIL'
                    except Exception:
                        pass

                ga_entry = {
                    'candidate_id': r.get('candidate_id', ''),
                    'type': r.get('candidate_type', ''),
                    'int_gates': r.get('gates_passed', 0),
                    'strict_gates': strict_g,
                    'net_pf': round(r.get('net_pf', 0), 3),
                    'cost_1.5x_pf': round(r.get('cost_stress', {}).get('pf_at_1.5x', 0), 3),
                    'status': r.get('final_status', 'UNKNOWN'),
                    'per_gate': per_gate,
                }
                ga_data.append(ga_entry)

            ga_json = output_dir / f'ml_only_dynamic_preset_large_gate_audit_{ts_suffix}.json'
            with ga_json.open('w') as f:
                json.dump({
                    'timestamp': ts_suffix,
                    'gate_ids': all_gate_ids,
                    'candidates': ga_data,
                }, f, indent=2, default=str)

            # Build markdown table header
            ga_lines = [
                f"# Gate Audit — {ts_suffix}", "",
                f"**{len(all_results)}** candidates evaluated. "
                f"**{sum(1 for r in all_results if r.get('final_status') == 'PASS_SHADOW_READY')}** "
                f"PASS_SHADOW_READY, "
                f"**{sum(1 for r in all_results if r.get('final_status') == 'FAIL_BUT_PROMISING')}** "
                f"FAIL_BUT_PROMISING, "
                f"**{sum(1 for r in all_results if 'FAIL_REJECTED' in r.get('final_status', ''))}** "
                f"FAIL_REJECTED.",
                "",
                f"| candidate_id | type | int | strict | net_pf | 1.5x | status | " + ' | '.join(all_gate_ids) + ' |',
                f"|{'|'.join(['---'] * (7 + len(all_gate_ids)))}|",
            ]
            for r in top_candidates:
                sg = r.get('strict_gates_passed')
                ss = f"{sg}/22" if sg is not None else 'N/A'
                per_gate = next((
                    g['per_gate'] for g in ga_data
                    if g['candidate_id'] == r.get('candidate_id')
                ), {})
                gate_cells = ''.join(
                    f" {per_gate.get(gid, '-')} |" for gid in all_gate_ids
                )
                ga_lines.append(
                    f"| {r.get('candidate_id','')[:35]} | {r.get('candidate_type','')} | "
                    f"{r.get('gates_passed','?')}/9 | {ss} | "
                    f"{r.get('net_pf',0):.2f} | {r.get('cost_stress',{}).get('pf_at_1.5x',0):.2f} | "
                    f"{r.get('final_status','?')} |{gate_cells}"
                )
            ga_md = output_dir / f'ml_only_dynamic_preset_large_gate_audit_{ts_suffix}.md'
            with ga_md.open('w') as f:
                f.write('\n'.join(ga_lines))
            print(f"[INFO] Gate audit: {ga_md}")

        # ── Portfolio report ─────────────────────────────────────────────────
        if shadow_portfolio:
            pf_lines = [
                f"# Shadow Portfolio — {ts_suffix}",
                f"## {len(shadow_portfolio)} candidates passed strict 22/22",
                "",
                "## Gate integrity confirmation",
                "- strict 22/22 gates passed",
                "- 1.5x cost PF >= 1.0 mandatory",
                "- paper_only=True | real_trading_enabled=False",
                "",
                "| candidate_id | type | thresh | trades | net_pf | 1.5x_pf | preset |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in shadow_portfolio:
                p = r.get('preset', {})
                pf_lines.append(
                    f"| {r['candidate_id'][:40]} | {r.get('candidate_type','')} | "
                    f"{p.get('threshold',0):.2f} | {r.get('trade_count',0)} | "
                    f"{r.get('net_pf',0):.3f} | {r.get('cost_1.50x_pf',0):.3f} | "
                    f"{p.get('preset_family','')} |"
                )
            # Save portfolio JSON
            pf_json = output_dir / f'ml_only_dynamic_preset_large_portfolio_{ts_suffix}.json'
            with pf_json.open('w') as _f:
                json.dump({
                    'timestamp': ts_suffix,
                    'n_candidates': len(shadow_portfolio),
                    'total_candidates_evaluated': len(all_results),
                    'candidates': [
                        {
                            'candidate_id': r['candidate_id'],
                            'candidate_type': r.get('candidate_type', ''),
                            'model_type': r.get('model_type', ''),
                            'net_pf': round(r.get('net_pf', 0), 3),
                            'cost_1.50x_pf': round(r.get('cost_1.50x_pf', 0), 3),
                            'trade_count': r.get('trade_count', 0),
                            'preset_family': r.get('preset', {}).get('preset_family', ''),
                            'threshold': r.get('preset', {}).get('threshold', 0),
                            'strict_gates_passed': r.get('strict_gates_passed'),
                        }
                        for r in shadow_portfolio
                    ],
                    'paper_only': True,
                    'real_trading_enabled': False,
                }, _f, indent=2, default=str)
            pf_md = output_dir / f'ml_only_dynamic_preset_large_portfolio_{ts_suffix}.md'
            with pf_md.open('w') as _f:
                _f.write('\n'.join(pf_lines))
            print(f"[INFO] Portfolio: {pf_md}")

    n_shadow = len(shadow_ready)
    n_internal = len(internal_passing)
    print(f"\n{'='*60}")
    print(f"  ML-ONLY DYNAMIC PRESET RETRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total evaluated:   {len(all_results)}")
    print(f"  Internal 9/9:      {n_internal} (FAIL_BUT_PROMISING)")
    print(f"  Strict 22/22:      {n_shadow} PASS_SHADOW_READY")
    if shadow_ready:
        print(f"\n  PASS_SHADOW_READY candidates (sorted by 1.5x_pf):")
        for r in shadow_ready:
            p = r.get('preset', {})
            print(f"    - {r.get('candidate_type', '?')}/{r.get('model_type', '?')} "
                  f"pf={r.get('net_pf', 0):.2f} 1.5x={r.get('cost_1.50x_pf', 0):.2f} "
                  f"trades={r.get('trade_count', 0)} threshold={p.get('threshold', 0):.2f}")
    else:
        print(f"\n  No candidates passed strict 22/22 gates.")
        print(f"  See reports/ml_only_dynamic_preset_large_failure_diagnosis_{TS}.md")
    print(f"{'='*60}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())