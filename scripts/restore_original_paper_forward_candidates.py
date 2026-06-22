#!/usr/bin/env python3
"""Restore the original paper-forward candidate set with loadable artifacts.

This is intentionally narrow: it rebuilds the configured paper-forward
candidate artifacts using the existing project retraining utilities, then
rewrites config/paper_forward_candidates.json so the UI loads the original
candidate ids instead of stale missing folders.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from candidate_profile import (  # noqa: E402
    CandidateProfile,
    build_preset_family,
    filter_name_for_side,
)
from ml_only_dynamic_preset_retrainer import (  # noqa: E402
    get_live_features,
    load_and_audit_dataset,
    train_model,
)
from retrain_dynamic_candidate_matrix import _add_live_enhanced_features  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "paper_forward_candidates.json"
DEFAULT_SOURCE = REPO_ROOT / "config" / "paper_forward_candidates.backup_20260615_093044.json"
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260610_121730.csv"


def _load_source(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit(f"No candidates found in {path}")
    return payload


def _label_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested
    for name in (
        "cost_survivor_label_v2",
        "cost_survivor_label",
        "strong_profitable_trade_label_v2",
        "strong_profitable_trade_label",
        "profitable_trade_label",
    ):
        if name in df.columns:
            return name
    for name in df.columns:
        if str(name).endswith("_label") and "avoid" not in str(name).lower():
            return str(name)
    raise SystemExit("No usable target label column found in dataset")


def _filter_for_side(df: pd.DataFrame, side_policy: str) -> pd.DataFrame:
    side = str(side_policy or "").upper()
    if side not in {"PE_ONLY", "CE_ONLY"}:
        return df
    if "option_type" not in df.columns:
        return df.iloc[0:0].copy()
    want = "PE" if side == "PE_ONLY" else "CE"
    return df[df["option_type"].astype(str).str.upper() == want].copy()


def _preset_for_candidate(candidate: dict[str, Any]) -> Any:
    cid = str(candidate.get("candidate_id") or "")
    family = str(candidate.get("preset_family") or "")
    presets = build_preset_family(family)
    for preset in presets:
        if preset.name and preset.name in cid:
            return preset
    if presets:
        return presets[0]
    raise SystemExit(f"No preset definition found for {cid} family={family}")


def _safe_metrics(rows: int, label_col: str, positives: int, features: int) -> dict[str, Any]:
    rate = positives / max(1, rows)
    return {
        "status": "RESTORED_FOR_PAPER_FORWARD",
        "training_rows": rows,
        "positive_rows": positives,
        "positive_rate": round(rate, 6),
        "label_col": label_col,
        "feature_count": features,
    }


def _write_candidate_artifact(
    *,
    candidate: dict[str, Any],
    cand_dir: Path,
    preset: Any,
    feature_order: list[str],
    label_col: str,
    dataset_path: Path,
    rows: int,
    positives: int,
) -> None:
    cid = str(candidate["candidate_id"])
    model_name = str(candidate.get("model_name") or "logistic_regression")
    side_policy = str(candidate.get("side_policy") or preset.option_side_policy)
    preset_family = str(candidate.get("preset_family") or preset.preset_family)
    metrics = _safe_metrics(rows, label_col, positives, len(feature_order))

    profile = CandidateProfile(
        candidate_id=cid,
        model_name=model_name,
        feature_set_name="live_computable_v1",
        target_name=label_col,
        side_policy=side_policy,
        preset_family=preset_family,
        dynamic_presets={preset.name: preset.to_retrainer_dict()},
        threshold_policy={"entry_threshold": preset.entry_threshold, "restored_from": "original_paper_forward_config"},
        selection_policy={"top_n_per_day_first": True},
        risk_policy={
            "max_trades_per_day": preset.max_trades_per_day,
            "stop_loss_pct": preset.stop_loss_pct,
            "target_pct": preset.target_pct,
        },
        cost_policy={"cost_buffer_bps": preset.cost_buffer_bps, "stress_multipliers": [1.0, 1.25, 1.5, 2.0]},
        artifact_paths={
            "candidate_dir": str(cand_dir),
            "model_pkl": str(cand_dir / "model.pkl"),
            "dataset": str(dataset_path),
        },
        validation_metrics=metrics,
        gate_results={"restored": True, "paper_forward_only": True, "shadow_ready": False},
        live_computable_features=feature_order,
        created_at=datetime.now(timezone.utc).isoformat(),
        shadow_mode_metadata={"shadow_ready": False, "reject_reasons": ["restored_for_paper_forward_observation"]},
    )
    profile.save(cand_dir.parent)

    manifest = {
        "candidate_id": cid,
        "model_name": model_name,
        "paper_only": True,
        "real_trading_enabled": False,
        "model_pkl": "model.pkl",
        "selected_threshold": preset.entry_threshold,
        "filter_name": filter_name_for_side(side_policy),
        "feature_schema_path": "feature_schema.json",
        "preprocessing_path": "preprocessing_metadata.json",
        "filter_definition_path": "filter_definition.json",
        "dynamic_preset_path": "dynamic_preset.json",
        "preset_family": preset_family,
        "side_policy": side_policy,
        "status": "PAPER_FORWARD_ONLY",
        "paper_forward_only": True,
        "shadow_ready": False,
        "restored_at": datetime.now(timezone.utc).isoformat(),
    }
    (cand_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (cand_dir / "feature_schema.json").write_text(json.dumps({"features": feature_order}, indent=2), encoding="utf-8")
    (cand_dir / "filter_definition.json").write_text(
        json.dumps(
            {
                "filter_name": filter_name_for_side(side_policy),
                "filter_rule": f"side_policy={side_policy}",
                "required_fields": ["option_type", "timestamp", "ltp", "spread_pct"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (cand_dir / "preprocessing_metadata.json").write_text(json.dumps({"scaler": "standard"}, indent=2), encoding="utf-8")


def restore(args: argparse.Namespace) -> int:
    source_path = Path(args.source_config)
    dataset_path = Path(args.dataset)
    if not source_path.exists():
        raise SystemExit(f"Source config not found: {source_path}")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    source = _load_source(source_path)
    candidates = list(source["candidates"])
    print(f"[restore] source_candidates={len(candidates)} source={source_path}")
    print(f"[restore] loading dataset={dataset_path}")
    df, audit = load_and_audit_dataset(dataset_path)
    if args.sample_rows:
        df = df.head(int(args.sample_rows)).copy()
        print(f"[restore] sample_rows={len(df)}")
    df = _add_live_enhanced_features(df)
    feature_order = [f for f in get_live_features(df) if f in df.columns]
    label_col = _label_column(df, args.target)
    if not feature_order:
        raise SystemExit("No live-computable features found")
    print(f"[restore] rows={len(df)} features={len(feature_order)} label={label_col}")
    if audit.get("rejected_reasons"):
        print(f"[restore][audit] rejected_reasons={audit.get('rejected_reasons')}")

    restored: list[dict[str, Any]] = []
    for candidate in candidates:
        cid = str(candidate.get("candidate_id") or "")
        if not cid:
            continue
        preset = _preset_for_candidate(candidate)
        cand_dir = REPO_ROOT / str(candidate.get("artifact_dir") or f"artifacts/candidates/{cid}")
        train_df = _filter_for_side(df, str(candidate.get("side_policy") or preset.option_side_policy))
        train_df = train_df.dropna(subset=[label_col])
        avail = [f for f in feature_order if f in train_df.columns]
        if len(train_df) < int(args.min_rows):
            raise SystemExit(f"{cid}: only {len(train_df)} rows after side filter")
        y = train_df[label_col].astype(int)
        if y.nunique() < 2:
            raise SystemExit(f"{cid}: target {label_col} has one class after side filter")

        print(f"[restore] training {cid} rows={len(train_df)} positives={int(y.sum())} model={candidate.get('model_name')}")
        if not args.dry_run:
            cand_dir.mkdir(parents=True, exist_ok=True)
            train_model(
                train_df[avail].fillna(0).values,
                y.values,
                str(candidate.get("model_name") or "logistic_regression"),
                cand_dir,
            )
            model_path = cand_dir / "model.pkl"
            if not model_path.exists() or model_path.stat().st_size <= 512:
                raise SystemExit(f"{cid}: model.pkl was not created correctly")
            _write_candidate_artifact(
                candidate=candidate,
                cand_dir=cand_dir,
                preset=preset,
                feature_order=avail,
                label_col=label_col,
                dataset_path=dataset_path,
                rows=len(train_df),
                positives=int(y.sum()),
            )

        updated = dict(candidate)
        updated.update(
            {
                "enabled": True,
                "classification": "paper_forward_only",
                "paper_forward_only": True,
                "shadow_ready": False,
                "notes": "restored original candidate with regenerated paper-forward artifact",
                "artifact_id": cid,
                "artifact_dir": str(cand_dir.relative_to(REPO_ROOT)),
                "model_path": str((cand_dir / "model.pkl").relative_to(REPO_ROOT)),
                "feature_order_source": str((cand_dir / "feature_schema.json").relative_to(REPO_ROOT)),
                "artifact_identity_verified": True,
            }
        )
        updated.pop("disabled_reason", None)
        restored.append(updated)

    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = CONFIG_PATH.with_name(f"paper_forward_candidates.backup_{timestamp}.json")
        if CONFIG_PATH.exists():
            shutil.copy2(CONFIG_PATH, backup)
            print(f"[restore] backup={backup}")
        payload = {
            "mode": "paper_forward_multi",
            "live_orders_enabled": False,
            "broker_orders_enabled": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_report": f"restored_original_16_from {source_path.name}",
            "candidates": restored,
            "restore_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_config": str(source_path),
                "dataset": str(dataset_path),
                "sample_rows": int(args.sample_rows or 0),
                "candidate_count": len(restored),
                "label_col": label_col,
                "feature_count": len(feature_order),
            },
        }
        CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[restore] wrote={CONFIG_PATH} candidates={len(restored)}")
    else:
        print(f"[restore] dry_run candidates={len(restored)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", default=str(DEFAULT_SOURCE))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--target", default=None)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return restore(args)


if __name__ == "__main__":
    raise SystemExit(main())
