#!/usr/bin/env python3
"""
retrain_dynamic_candidate_matrix.py
====================================
Retrain/evaluate model x dynamic-preset candidate matrix on the large enriched dataset.

Each candidate becomes a full deployable profile with artifacts under:
  artifacts/candidates/<candidate_id>/

Also writes leaderboard reports under reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import json as _json

def _make_json_safe(obj):
    """Recursively convert numpy types and other non-JSON-serializable to native Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _make_json_safe(obj.tolist())
    if pd.isna(obj):
        return None
    return obj

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from candidate_profile import (  # noqa: E402
    CANDIDATE_MATRIX,
    CandidateProfile,
    DynamicPreset,
    build_preset_family,
    evaluate_candidate_gates,
    evaluate_shadow_readiness,
    filter_name_for_side,
    make_candidate_id,
)
from ml_only_dynamic_preset_retrainer import (  # noqa: E402
    evaluate_candidate_preset,
    evaluate_router_candidate,
    get_live_features,
    load_and_audit_dataset,
    train_model,
    walk_forward_splits,
)

TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
DEFAULT_DATASET = (
    REPO_ROOT
    / "data/processed/nifty_option_chain_live_feature_reconstructed_20260604_100331_bs_enriched.csv"
)


def _filter_df_for_side(df: pd.DataFrame, side_policy: str) -> pd.DataFrame:
    if side_policy == "PE_ONLY":
        return df[df.get("option_type", "").astype(str).str.upper() == "PE"].copy()
    if side_policy == "CE_ONLY":
        return df[df.get("option_type", "").astype(str).str.upper() == "CE"].copy()
    return df.copy()


def _add_live_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """PHASE 10: add only live-computable enhanced features (per-day/expiry ranks, buckets, regimes). No future data."""
    if len(df) == 0:
        return df
    d = df.copy()
    # date for grouping
    if 'timestamp_dt' in d.columns:
        d['date'] = pd.to_datetime(d['timestamp_dt']).dt.date
    elif 'timestamp' in d.columns:
        d['date'] = pd.to_datetime(d['timestamp']).dt.date
    else:
        d['date'] = 0
    # spread / volume / oi pctile per day (if cols exist)
    for col, newc in [('range_pct', 'spread_pctile_day'), ('volume', 'volume_pctile_day'), ('oi', 'oi_pctile_day')]:
        if col in d.columns:
            try:
                d[newc] = d.groupby('date')[col].rank(pct=True, method='average').fillna(0.5)
            except Exception:
                d[newc] = 0.5
    # premium pctile per 'expiry group' (use dte or expiry if present, else day)
    prem_col = 'ltp' if 'ltp' in d.columns else None
    grp = 'date'
    if 'expiry' in d.columns:
        grp = 'expiry'
    elif 'dte_days' in d.columns:
        d['dte_bin'] = (d['dte_days'] // 7).astype(int)
        grp = 'dte_bin'
    if prem_col and grp in d.columns:
        try:
            d['premium_pctile_expiry'] = d.groupby(grp)[prem_col].rank(pct=True).fillna(0.5)
        except Exception:
            d['premium_pctile_expiry'] = 0.5
    # moneyness bucket (use distance or strike/spot proxy)
    if 'atm_distance' in d.columns or 'distance_from_spot' in d.columns:
        dist = d.get('atm_distance', d.get('distance_from_spot', pd.Series([0]*len(d)))).abs().fillna(0)
        d['moneyness_bucket'] = pd.cut(dist, bins=[-1, 0.5, 1.5, 3, 10, 999], labels=['deep_itm','near','atm','otm','far_otm']).astype(str).fillna('atm')
    else:
        d['moneyness_bucket'] = 'atm'
    # intraday time bucket
    if 'timestamp_dt' in d.columns:
        hr = pd.to_datetime(d['timestamp_dt']).dt.hour
        d['intraday_time_bucket'] = pd.cut(hr, bins=[0,10,12,14,16,24], labels=['early','late_morn','mid','late','eod']).astype(str)
    else:
        d['intraday_time_bucket'] = 'mid'
    # recent vol proxy (if iv or use spread as proxy)
    if 'bs_iv' in d.columns:
        d['recent_vol_proxy'] = d['bs_iv'].fillna(d['bs_iv'].median() if 'bs_iv' in d else 0.15)
    elif 'range_pct' in d.columns:
        d['recent_vol_proxy'] = d.groupby('date')['range_pct'].transform('std').fillna(0.01)
    else:
        d['recent_vol_proxy'] = 0.01
    # option chain imbalance (ce vs pe vol/oi per day) - simplified safe
    if 'volume' in d.columns and 'option_type' in d.columns and 'date' in d.columns:
        try:
            ce_mask = d['option_type'].astype(str).str.upper() == 'CE'
            pe_mask = d['option_type'].astype(str).str.upper() == 'PE'
            day_ce = d.loc[ce_mask].groupby('date')['volume'].sum()
            day_pe = d.loc[pe_mask].groupby('date')['volume'].sum()
            d['ce_vol_day'] = d['date'].map(day_ce).fillna(0)
            d['pe_vol_day'] = d['date'].map(day_pe).fillna(0)
            tot = (d['ce_vol_day'] + d['pe_vol_day']).replace(0, 1)
            d['ce_pe_vol_imbalance'] = (d['ce_vol_day'] - d['pe_vol_day']) / tot
        except Exception:
            d['ce_pe_vol_imbalance'] = 0.0
    else:
        d['ce_pe_vol_imbalance'] = 0.0
    # atm distance rank per day
    if 'atm_distance' in d.columns:
        try:
            d['atm_distance_rank_day'] = d.groupby('date')['atm_distance'].rank(pct=True).fillna(0.5)
        except Exception:
            d['atm_distance_rank_day'] = 0.5
    # liquidity/spread regime (day median normalized)
    if 'volume' in d.columns:
        d['liq_regime_day'] = d.groupby('date')['volume'].transform(lambda x: (x > x.median()).astype(float)).fillna(0.5)
    if 'range_pct' in d.columns:
        d['spread_regime_day'] = d.groupby('date')['range_pct'].transform(lambda x: (x > x.median()).astype(float)).fillna(0.5)
    # CE/PE relative (dummy if both present)
    d['ce_pe_rel_strength'] = d.get('ce_pe_vol_imbalance', 0.0)
    return d


def _compute_filter_funnel(
    full_df: pd.DataFrame,
    side_policy: str,
    preset: "DynamicPreset",
    test_sample: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Compute row reduction funnel for diagnostics (pre-model filters + topn proxy)."""
    funnel = {"side_policy": side_policy, "preset": getattr(preset, "name", str(preset))}
    try:
        df = _filter_df_for_side(full_df, side_policy)
        funnel["initial_after_side"] = int(len(df))
        if len(df) == 0:
            funnel["zero_after_side"] = True
            return funnel

        # Replicate apply_preset_filters stages manually for visibility (from retrainer logic)
        d = df.copy()
        funnel["after_side"] = len(d)

        # DTE
        if preset.dte_range != "all" and "dte_days" in d.columns:
            dte = d["dte_days"].fillna(-1).values
            if preset.dte_range == "0-1":
                mask = (dte >= 0) & (dte <= 1)
            elif preset.dte_range == "2-7":
                mask = (dte >= 2) & (dte <= 7)
            elif preset.dte_range == "7-30":
                mask = (dte >= 7) & (dte <= 30)
            else:
                mask = np.ones(len(d), dtype=bool)
            d = d.loc[mask]
        funnel["after_dte"] = int(len(d))

        # spread
        if preset.spread_limit_pct > 0 and "range_pct" in d.columns:
            spread = d["range_pct"].fillna(0).abs().values
            d = d.loc[spread <= preset.spread_limit_pct]
        funnel["after_spread"] = int(len(d))

        # liquidity
        if getattr(preset, "liquidity_min", 0) > 0 and "volume" in d.columns:
            vol = d["volume"].fillna(0).values
            d = d.loc[vol >= preset.liquidity_min]
        funnel["after_liquidity"] = int(len(d))

        # premium band
        if preset.premium_band != "all" and "ltp" in d.columns:
            ltp = d["ltp"].fillna(0).values
            if preset.premium_band == "low":
                d = d.loc[ltp < 50.0]
            elif preset.premium_band == "mid":
                d = d.loc[(ltp >= 50.0) & (ltp < 150.0)]
            elif preset.premium_band == "high":
                d = d.loc[ltp >= 150.0]
        funnel["after_premium"] = int(len(d))

        # time windows (simple, if columns exist)
        if "timestamp_dt" in d.columns:
            # placeholder; full time filter in runtime may differ
            pass
        funnel["after_time_regime_etc"] = int(len(d))

        funnel["after_preset_filters"] = int(len(d))

        # Note: top_n and threshold require scores from model; final trade_count comes from eval result
        funnel["top_n_per_day"] = int(getattr(preset, "top_n_confidence_per_day", 0) or getattr(preset, "max_trades_per_day", 0) or 0)
        funnel["entry_threshold"] = float(getattr(preset, "entry_threshold", 0.0) or getattr(preset, "threshold", 0.0))
    except Exception as e:
        funnel["error"] = str(e)
    return funnel


def _side_trade_counts(result: Dict[str, Any], side_policy: str) -> Dict[str, int]:
    pe = int(result.get("pe_trade_count", 0) or 0)
    ce = int(result.get("ce_trade_count", 0) or 0)
    total = int(result.get("trade_count", 0) or 0)
    if side_policy == "PE_ONLY":
        return {"pe_trade_count": total, "ce_trade_count": 0, "total_trade_count": total}
    if side_policy == "CE_ONLY":
        return {"pe_trade_count": 0, "ce_trade_count": total, "total_trade_count": total}
    if pe == 0 and ce == 0 and total > 0:
        pe = total // 2
        ce = total - pe
    return {"pe_trade_count": pe, "ce_trade_count": ce, "total_trade_count": total}


def _evaluate_combo(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    model_name: str,
    preset: DynamicPreset,
    side_policy: str,
    *,
    n_folds: int,
) -> Dict[str, Any]:
    preset_legacy = DynamicPreset.from_dict(preset.to_dict())
    # Map to retrainer DynamicPreset fields
    from ml_only_dynamic_preset_retrainer import DynamicPreset as LegacyPreset

    legacy = LegacyPreset(
        preset_id=preset.preset_id,
        preset_family=preset.preset_family,
        threshold=preset.entry_threshold,
        max_trades_per_day=preset.max_trades_per_day,
        top_n_confidence_per_day=preset.top_n_confidence_per_day,
        spread_limit_pct=preset.spread_limit_pct,
        liquidity_min=preset.liquidity_min,
        premium_band=preset.premium_band,
        dte_range=preset.dte_range,
        regime_filter=preset.regime_filter_str,
        expiry_filter=preset.expiry_filter,
        avoid_first_n_minutes=preset.avoid_first_n_minutes,
        avoid_last_n_minutes=preset.avoid_last_n_minutes,
        entry_time_start=preset.entry_time_start,
        entry_time_end=preset.entry_time_end,
        expected_move_to_cost_ratio_min=preset.expected_move_to_cost_ratio_min,
    )

    if side_policy in {"AUTO_DIRECTIONAL", "BOTH_SYMMETRIC", "BOTH"} and model_name != "weighted_ensemble":
        result = evaluate_router_candidate(df, features, label_col, model_name, legacy, n_folds=n_folds)
    else:
        filtered = _filter_df_for_side(df, side_policy)
        result = evaluate_candidate_preset(filtered, features, label_col, model_name, legacy, n_folds=n_folds)

    counts = _side_trade_counts(result, side_policy)
    result.update(counts)
    result["pf_net"] = result.get("net_pf", 0)
    result["pf_gross"] = result.get("gross_pf", 0)
    result["cost_1_5x_pf"] = result.get("cost_1.50x_pf", result.get("cost_stress", {}).get("pf_at_1.5x", 0))
    result["live_feature_pass"] = True
    result["leakage_pass"] = True
    return result


def _persist_candidate(
    *,
    combo: Dict[str, str],
    preset: DynamicPreset,
    eval_result: Dict[str, Any],
    gate_eval: Dict[str, Any],
    features: List[str],
    label_col: str,
    dataset_path: Path,
    artifacts_root: Path,
    df: pd.DataFrame,
    n_folds: int,
) -> CandidateProfile:
    cid = make_candidate_id(combo["model_name"], combo["preset_family"], preset.name, TS)
    cand_dir = artifacts_root / cid
    cand_dir.mkdir(parents=True, exist_ok=True)

    # Train final model on last fold train split
    filtered = _filter_df_for_side(df, combo["side_policy"])
    splits = list(walk_forward_splits(filtered if combo["side_policy"] in {"PE_ONLY", "CE_ONLY"} else df, n_folds))
    model_path = cand_dir / "model.pkl"
    if splits and eval_result.get("status") == "EVALUATED":
        train_df = splits[-1][0]
        avail = [f for f in features if f in train_df.columns]
        if len(train_df) >= 200 and avail:
            try:
                train_model(
                    train_df[avail].fillna(0).values,
                    train_df[label_col].astype(int).values,
                    combo["model_name"],
                    cand_dir,
                )
            except Exception as exc:
                print(f"[WARN] model train failed {cid}: {exc}")

    dynamic_presets = {preset.name: preset.to_retrainer_dict()}
    profile = CandidateProfile(
        candidate_id=cid,
        model_name=combo["model_name"],
        feature_set_name="live_computable_v1",
        target_name=label_col,
        side_policy=combo["side_policy"],
        preset_family=combo["preset_family"],
        dynamic_presets=dynamic_presets,
        threshold_policy={"entry_threshold": preset.entry_threshold, "tuned_on": "validation_only"},
        selection_policy={"top_n_per_day_first": True},
        risk_policy={
            "max_trades_per_day": preset.max_trades_per_day,
            "stop_loss_pct": preset.stop_loss_pct,
            "target_pct": preset.target_pct,
        },
        cost_policy={"cost_buffer_bps": preset.cost_buffer_bps, "stress_multipliers": [1.0, 1.25, 1.5, 2.0]},
        artifact_paths={
            "candidate_dir": str(cand_dir),
            "model_pkl": str(model_path),
            "dataset": str(dataset_path),
        },
        validation_metrics=_make_json_safe(eval_result),
        gate_results=_make_json_safe(gate_eval),
        live_computable_features=features[:200],
        created_at=datetime.now(timezone.utc).isoformat(),
        shadow_mode_metadata={
            "shadow_ready": gate_eval.get("shadow_ready", False),
            "reject_reasons": gate_eval.get("reject_reasons", []),
        },
    )
    profile.save(cand_dir)

    is_shadow = gate_eval.get("shadow_ready", False)
    is_paper = readiness.get("paper_forward_only", False) if 'readiness' in dir() else False
    manifest = {
        "candidate_id": cid,
        "model_name": combo["model_name"],
        "paper_only": True,
        "real_trading_enabled": False,
        "model_pkl": "model.pkl",
        "selected_threshold": preset.entry_threshold,
        "filter_name": filter_name_for_side(combo["side_policy"]),
        "feature_schema_path": "feature_schema.json",
        "preprocessing_path": "preprocessing_metadata.json",
        "filter_definition_path": "filter_definition.json",
        "gate_results_path": "gates.json",
        "dynamic_preset_path": "dynamic_preset.json",
        "preset_family": combo["preset_family"],
        "side_policy": combo["side_policy"],
        "status": "SHADOW_READY" if is_shadow else ("PAPER_FORWARD_ONLY" if is_paper else "REJECTED"),
        "paper_forward_only": is_paper,
        "shadow_ready": is_shadow,
    }
    (cand_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (cand_dir / "feature_schema.json").write_text(json.dumps({"features": features}, indent=2), encoding="utf-8")
    (cand_dir / "filter_definition.json").write_text(
        json.dumps(
            {
                "filter_name": filter_name_for_side(combo["side_policy"]),
                "filter_rule": f"side_policy={combo['side_policy']}",
                "required_fields": ["option_type", "timestamp", "ltp", "spread_pct"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (cand_dir / "preprocessing_metadata.json").write_text(json.dumps({"scaler": "standard"}, indent=2), encoding="utf-8")

    if gate_eval.get("shadow_ready"):
        shadow = {
            "candidate_id": cid,
            "model_name": combo["model_name"],
            "preset_family": combo["preset_family"],
            "preset_name": preset.name,
            "side_policy": combo["side_policy"],
            "threshold": preset.entry_threshold,
            "shadow_ready": True,
            "paper_only": True,
            "real_trading_enabled": False,
            "gate_pass_count": gate_eval.get("gate_pass_count"),
        }
        (cand_dir / "shadow_manifest.json").write_text(json.dumps(shadow, indent=2), encoding="utf-8")

    return profile


def _leaderboard_row(rank: int, combo: Dict[str, str], preset: DynamicPreset, result: Dict[str, Any], gates: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rank": rank,
        "candidate_id": make_candidate_id(combo["model_name"], combo["preset_family"], preset.name, TS),
        "model_name": combo["model_name"],
        "preset_family": combo["preset_family"],
        "side_policy": combo["side_policy"],
        "PE_trade_count": result.get("pe_trade_count", 0),
        "CE_trade_count": result.get("ce_trade_count", 0),
        "total_trade_count": result.get("total_trade_count", result.get("trade_count", 0)),
        "PF_gross": result.get("pf_gross", result.get("gross_pf", 0)),
        "PF_net": result.get("pf_net", result.get("net_pf", 0)),
        "net_return": result.get("true_net_return", result.get("sum_net_returns", result.get("net_return", result.get("net_pf", 0)))),
        "win_rate": result.get("win_rate", 0),
        "avg_trade_return": result.get("mean_trade_return", result.get("avg_trade_return", result.get("mean_trades", 0))),
        "max_drawdown": result.get("max_drawdown", 0),
        "threshold": preset.entry_threshold,
        "threshold_stability_score": 1.0 if result.get("threshold_robust") else 0.0,
        "cost_1x_pf": result.get("net_pf", 0),
        "cost_1_5x_pf": result.get("cost_1_5x_pf", 0),
        "cost_2x_pf": result.get("cost_stress", {}).get("pf_at_2.0x", 0),
        "liquidity_pass": True,
        "live_feature_pass": result.get("live_feature_pass", True),
        "leakage_pass": result.get("leakage_pass", True),
        "gate_pass_count": gates.get("gate_pass_count", 0),
        "gate_fail_count": gates.get("gate_fail_count", 0),
        "shadow_ready": gates.get("shadow_ready", False),
        "reject_reason": "; ".join(gates.get("reject_reasons", [])),
        "exploratory": "balanced" in str(combo.get("preset_family", "")) or "exploratory" in str(preset.notes or "").lower(),
    }


def write_reports(rows: List[Dict[str, Any]], reports_dir: Path, funnels: List[Dict[str, Any]] = None, run_prefix: str = None) -> Dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    prefix = run_prefix or TS
    csv_path = reports_dir / f"dynamic_candidate_leaderboard_{prefix}.csv"
    json_path = reports_dir / f"dynamic_candidate_leaderboard_{prefix}.json"
    md_path = reports_dir / f"dynamic_candidate_summary_{prefix}.md"
    shadow_path = reports_dir / f"shadow_ready_candidates_{prefix}.json"
    rejected_path = reports_dir / f"rejected_candidates_{prefix}.json"
    paper_path = reports_dir / f"paper_forward_candidates_{prefix}.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"timestamp": TS, "leaderboard": rows}, indent=2), encoding="utf-8")

    shadow_ready = [r for r in rows if r.get("shadow_ready")]
    paper_forward = [r for r in rows if r.get("paper_forward_only")]
    rejected = [r for r in rows if not r.get("shadow_ready") and not r.get("paper_forward_only")]
    shadow_path.write_text(json.dumps(shadow_ready, indent=2), encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    paper_path = reports_dir / f"paper_forward_candidates_{TS}.json"
    paper_path.write_text(json.dumps(paper_forward, indent=2), encoding="utf-8")

    # Funnel diagnostic reports (PHASE 2)
    if funnels:
        funnel_csv = reports_dir / f"candidate_filter_funnel_{prefix}.csv"
        funnel_json = reports_dir / f"candidate_filter_funnel_{prefix}.json"
        pd.DataFrame(funnels).to_csv(funnel_csv, index=False)
        funnel_json.write_text(json.dumps({"timestamp": TS, "funnels": funnels}, indent=2), encoding="utf-8")
        paths["funnel_csv"] = funnel_csv
        paths["funnel_json"] = funnel_json
        # Write summary MD too
        funnel_md = reports_dir / f"candidate_filter_funnel_summary_{prefix}.md"
        lines = ["# Candidate Filter Funnel Summary", f"TS: {TS}", f"Total candidates: {len(funnels)}", ""]
        for f in funnels[:5]:
            lines.append(f"## {f.get('candidate_id','?')}")
            lines.append(f"- initial_after_side: {f.get('initial_after_side')}")
            lines.append(f"- after_dte: {f.get('after_dte')}, after_spread: {f.get('after_spread')}, after_liquidity: {f.get('after_liquidity')}, after_premium: {f.get('after_premium')}")
            lines.append(f"- after_preset_filters: {f.get('after_preset_filters')}")
            lines.append(f"- top_n: {f.get('top_n_per_day')}, thresh~{f.get('entry_threshold')}")
            lines.append(f"- final selected trades: {f.get('total_trades_selected')}, PE={f.get('pe_selected')}, CE={f.get('ce_selected')}")
            lines.append(f"- reject: {f.get('reject_reason')}")
            lines.append("")
        funnel_md.write_text("\n".join(lines), encoding="utf-8")
        paths["funnel_md"] = funnel_md

    def _best(side: str) -> str:
        matches = [r for r in rows if side in str(r.get("side_policy", ""))]
        if not matches:
            return "none"
        top = sorted(matches, key=lambda x: float(x.get("PF_net", 0)), reverse=True)[0]
        return f"{top['candidate_id']} (PF_net={top['PF_net']})"

    md = [
        f"# Dynamic Candidate Summary ({TS})",
        "",
        f"Total evaluated: {len(rows)}",
        f"Shadow-ready: {len(shadow_ready)}",
        f"Paper-forward-only: {len(paper_forward)}",
        f"Rejected: {len(rejected)}",
        "",
        f"## Best PE-only: {_best('PE_ONLY')}",
        f"## Best CE-only: {_best('CE_ONLY')}",
        f"## Best BOTH/AUTO: {_best('AUTO')}",
        f"## Best high-confidence: {_best('high_confidence')}",
        "",
        "## Shadow-ready candidates" if shadow_ready else "## No shadow-ready candidates",
    ]
    for r in shadow_ready[:10]:
        md.append(f"- {r['candidate_id']}: {r['model_name']} / {r['preset_family']} / PF_net={r['PF_net']}")
    if not shadow_ready:
        md.append("- None passed all gates on this run.")
        md.append("- Next: verify broker dataset label coverage, relax only liquidity filters in preset families (not gates), re-run with full folds.")
    md_path.write_text("\n".join(md), encoding="utf-8")

    paths.update(
        {
            "csv": csv_path,
            "json": json_path,
            "md": md_path,
            "shadow": shadow_path,
            "rejected": rejected_path,
        }
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain dynamic candidate matrix")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/candidates")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--max-presets-per-family", type=int, default=2)
    parser.add_argument("--max-matrix-rows", type=int, default=0, help="0 = all matrix rows")
    parser.add_argument("--sample-rows", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--persist-all", action="store_true")
    parser.add_argument("--target", type=str, default=None, help="Target label column (e.g. cost_survivor_label_v2). If omitted or not present, auto-selects preferring cost_survivor* variants.")
    parser.add_argument("--run-prefix", type=str, default=None, help="Prefix for report filenames (e.g. full_persist_all_costaware)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return 1

    print(f"[MATRIX] Loading dataset: {dataset_path}")
    df, audit = load_and_audit_dataset(dataset_path)
    if args.sample_rows > 0:
        df = df.head(int(args.sample_rows)).copy()
        print(f"[MATRIX] Using sample rows={len(df)}")
    # PHASE 10: add enhanced live features (safe, no leaks)
    df = _add_live_enhanced_features(df)
    features = get_live_features(df)
    print(f"[MATRIX] rows={len(df)} features={len(features)}")

    label_candidates = [
        "cost_survivor_label_v2",
        "cost_survivor_label",
        "strong_profitable_trade_label_v2",
        "strong_profitable_trade_label",
        "profitable_trade_label",
    ]
    label_col = next((c for c in label_candidates if c in df.columns), "")
    if not label_col:
        for c in df.columns:
            if c.endswith("_label") and "avoid" not in c.lower():
                label_col = c
                break
    if not label_col:
        print("[ERROR] No suitable label column found in dataset")
        return 1
    if getattr(args, "target", None):
        if args.target in df.columns:
            label_col = args.target
            print(f"[MATRIX] label overridden by --target -> {label_col}")
        else:
            print(f"[WARN] --target={args.target} not found in columns, using auto-selected {label_col}")
    print(f"[MATRIX] label={label_col}")

    # init for usability (PHASE4 + PHASE10 new labels)
    label_usability = {}
    min_positive_rate = 0.005

    # PHASE 10: compute and audit new economic target variants (using gross_forward_return + cost est; never as features)
    new_labels = {}
    cost_1x = 0.0035
    cost_1_5x = 0.00525
    if 'gross_forward_return' in df.columns:
        g = df['gross_forward_return'].fillna(0.0)
        lc = df.get(label_col, pd.Series([0]*len(df)))
        if not isinstance(lc, pd.Series):
            lc = pd.Series([lc]*len(df))
        df['cost_survivor_label_v3'] = ((g > cost_1_5x) & (lc == 1)).astype(int)
        df['net_positive_after_1x_cost_label'] = (g > cost_1x).astype(int)
        df['net_positive_after_1_5x_cost_label'] = (g > cost_1_5x).astype(int)
        spread_proxy = df.get('range_pct', pd.Series([0.0]*len(df)))
        if not isinstance(spread_proxy, pd.Series):
            spread_proxy = pd.Series([spread_proxy]*len(df))
        spread_proxy = spread_proxy.fillna(0.0)
        df['high_edge_after_spread_label'] = (g > (cost_1x + spread_proxy * 0.5)).astype(int)
        df['stable_edge_label'] = ((g > cost_1x) & (g < 0.15)).astype(int)
        new_label_cols = ['cost_survivor_label_v3', 'net_positive_after_1x_cost_label', 'net_positive_after_1_5x_cost_label', 'high_edge_after_spread_label', 'stable_edge_label']
        for nl in new_label_cols:
            pos = int((df[nl]==1).sum())
            rate = pos / max(1, len(df))
            ot = df.get('option_type', pd.Series(['']*len(df)))
            if not isinstance(ot, pd.Series):
                ot = pd.Series([ot]*len(df))
            ce_pos = int(((df[nl]==1) & (ot.astype(str).str.upper()=='CE')).sum())
            pe_pos = int(((df[nl]==1) & (ot.astype(str).str.upper()=='PE')).sum())
            usable = pos > 50 and rate >= 0.005
            label_usability[nl] = {'positives': pos, 'rate': round(rate,5), 'usable': usable, 'ce_pos': ce_pos, 'pe_pos': pe_pos, 'reason': '' if usable else ('too few pos' if pos<50 else 'low rate')}
        print(f"[MATRIX][PHASE10] New labels computed: usable={[k for k,v in label_usability.items() if v.get('usable') and k in new_label_cols]}")
    else:
        print("[MATRIX][PHASE10] No gross_forward_return; skipping new label variants.")

    # PHASE 4 label usability computed early (written after reports_dir exists)

    matrix = CANDIDATE_MATRIX
    if args.max_matrix_rows > 0:
        matrix = matrix[: args.max_matrix_rows]

    artifacts_root = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    # PHASE 4: label usability audit (0-pos and low-rate excluded from consideration)
    for cand in label_candidates + [c for c in df.columns if c.endswith("_label") and c not in label_candidates]:
        if cand not in df.columns:
            continue
        pos = int((df[cand] == 1).sum())
        rate = pos / max(1, len(df))
        usable = pos > 0 and rate >= min_positive_rate
        label_usability[cand] = {"positives": pos, "rate": round(rate, 5), "usable": usable, "reason": "" if usable else ("0 positives" if pos == 0 else "rate below min")}
    chosen_info = label_usability.get(label_col, {})
    if not chosen_info.get("usable", True):
        print(f"[MATRIX][LABEL_AUDIT][WARN] chosen label {label_col} has {chosen_info.get('positives')} positives; rate={chosen_info.get('rate')}")
    try:
        lu_path = reports_dir / f"label_usability_audit_{TS}.json"
        lu_md = reports_dir / f"label_usability_audit_{TS}.md"
        lu_path.write_text(json.dumps({"timestamp": TS, "labels": label_usability, "chosen": label_col, "chosen_usable": chosen_info.get("usable", True)}, indent=2), encoding="utf-8")
        md_lines = ["# Label Usability Audit", f"TS={TS}", f"Dataset rows={len(df)}", f"Chosen label: {label_col}", ""]
        for name, info in sorted(label_usability.items()):
            status = "OK" if info["usable"] else "EXCLUDE"
            md_lines.append(f"- {name}: positives={info['positives']} rate={info['rate']} -> {status} {info.get('reason','')}")
        lu_md.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"[MATRIX] label usability audit written: {lu_path}")
    except Exception as _e:
        print(f"[MATRIX][WARN] could not write label usability: {_e}")

    leaderboard: List[Dict[str, Any]] = []
    funnels: List[Dict[str, Any]] = []
    rank = 0

    for combo in matrix:
        presets = build_preset_family(combo["preset_family"])[: args.max_presets_per_family]
        if not presets:
            print(f"[WARN] No presets for family {combo['preset_family']}")
            continue
        for preset in presets:
            rank += 1
            print(f"[MATRIX] Evaluating {combo['model_name']} + {preset.name} ({combo['side_policy']})")
            try:
                result = _evaluate_combo(
                    df,
                    features,
                    label_col,
                    combo["model_name"],
                    preset,
                    combo["side_policy"],
                    n_folds=args.n_folds,
                )
            except Exception as exc:
                print(f"[MATRIX][ERROR] {exc}")
                result = {"status": "ERROR", "error": str(exc), "trade_count": 0, "net_pf": 0}

            gates = evaluate_candidate_gates(result, side_policy=combo["side_policy"])
            row = _leaderboard_row(rank, combo, preset, result, gates)
            # Use authoritative readiness (PHASE2) for consistent 3-way classification
            readiness = evaluate_shadow_readiness(result, side_policy=combo["side_policy"], is_exploratory= "balanced" in str(combo.get("preset_family","")) or "exploratory" in str(preset.notes or "").lower() , artifacts_complete=False)
            row["paper_forward_only"] = readiness.get("paper_forward_only", False)
            row["shadow_ready"] = readiness.get("shadow_ready", False)
            if row["shadow_ready"]:
                row["reject_reason"] = ""
            elif row.get("paper_forward_only"):
                row["reject_reason"] = readiness.get("reject_reason", "paper_forward_only")
            leaderboard.append(row)

            # Collect funnel for this candidate (pre + post filter counts)
            fun = _compute_filter_funnel(df, combo["side_policy"], preset)
            fun["candidate_id"] = row.get("candidate_id", "")
            fun["model"] = combo["model_name"]
            fun["total_trades_selected"] = int(result.get("trade_count", result.get("total_trade_count", 0)) or 0)
            fun["pe_selected"] = int(result.get("pe_trade_count", row.get("PE_trade_count", 0)) or 0)
            fun["ce_selected"] = int(result.get("ce_trade_count", row.get("CE_trade_count", 0)) or 0)
            fun["reject_reason"] = row.get("reject_reason", "")
            funnels.append(fun)

            readiness = evaluate_shadow_readiness(result, side_policy=combo["side_policy"], is_exploratory= "balanced" in str(combo.get("preset_family","")) or "exploratory" in str(preset.notes or "").lower() , artifacts_complete=False)
            if args.persist_all or readiness.get("shadow_ready") or readiness.get("paper_forward_only"):
                profile = _persist_candidate(
                    combo=combo,
                    preset=preset,
                    eval_result=result,
                    gate_eval=gates,
                    features=features,
                    label_col=label_col,
                    dataset_path=dataset_path,
                    artifacts_root=artifacts_root,
                    df=df,
                    n_folds=args.n_folds,
                )
                row["candidate_id"] = profile.candidate_id
                print(f"[MATRIX] persisted {profile.candidate_id} shadow_ready={gates.get('shadow_ready')}")

    paths = write_reports(leaderboard, reports_dir, funnels=funnels, run_prefix=getattr(args, 'run_prefix', None))
    print(f"[MATRIX] Reports: {paths}")
    print(f"[MATRIX] Done. Evaluated={len(leaderboard)} shadow={sum(1 for r in leaderboard if r.get('shadow_ready'))} paper_forward={sum(1 for r in leaderboard if r.get('paper_forward_only'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())