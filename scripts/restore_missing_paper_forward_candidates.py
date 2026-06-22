#!/usr/bin/env python3
"""Materialize artifacts for disabled paper-forward candidates and enable them in config."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from restore_original_paper_forward_candidates import (  # noqa: E402
    CONFIG_PATH,
    _filter_for_side,
    _label_column,
    _preset_for_candidate,
    _write_candidate_artifact,
)
from ml_only_dynamic_preset_retrainer import get_live_features, load_and_audit_dataset, train_model  # noqa: E402
from retrain_dynamic_candidate_matrix import _add_live_enhanced_features  # noqa: E402


def restore_missing(
    *,
    dataset: Path,
    sample_rows: int = 30000,
    min_rows: int = 200,
    dry_run: bool = False,
) -> int:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config not found: {CONFIG_PATH}")
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidates = list(payload.get("candidates") or [])
    missing = [c for c in candidates if not c.get("enabled")]
    if not missing:
        print("[restore-missing] no disabled candidates")
        return 0

    print(f"[restore-missing] disabled={len(missing)} dataset={dataset}")
    df, audit = load_and_audit_dataset(dataset)
    if sample_rows:
        df = df.head(int(sample_rows)).copy()
        print(f"[restore-missing] sample_rows={len(df)}")
    df = _add_live_enhanced_features(df)
    feature_order = [f for f in get_live_features(df) if f in df.columns]
    label_col = _label_column(df, None)
    if not feature_order:
        raise SystemExit("No live-computable features found")
    print(f"[restore-missing] features={len(feature_order)} label={label_col}")
    if audit.get("rejected_reasons"):
        print(f"[restore-missing][audit] {audit.get('rejected_reasons')}")

    by_id = {str(c.get("candidate_id") or ""): c for c in candidates}
    restored_ids: list[str] = []
    for candidate in missing:
        cid = str(candidate.get("candidate_id") or "")
        if not cid:
            continue
        preset = _preset_for_candidate(candidate)
        cand_dir = REPO_ROOT / str(candidate.get("artifact_dir") or f"artifacts/candidates/{cid}")
        train_df = _filter_for_side(df, str(candidate.get("side_policy") or preset.option_side_policy))
        train_df = train_df.dropna(subset=[label_col])
        avail = [f for f in feature_order if f in train_df.columns]
        if len(train_df) < min_rows:
            raise SystemExit(f"{cid}: only {len(train_df)} rows after side filter (need {min_rows})")
        y = train_df[label_col].astype(int)
        if y.nunique() < 2:
            raise SystemExit(f"{cid}: target {label_col} has one class after side filter")

        print(
            f"[restore-missing] training {cid} rows={len(train_df)} "
            f"positives={int(y.sum())} model={candidate.get('model_name')}"
        )
        if not dry_run:
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
                dataset_path=dataset,
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
                "notes": "restored missing paper-forward artifact for monitor enablement",
                "artifact_id": cid,
                "artifact_dir": str(cand_dir.relative_to(REPO_ROOT)),
                "model_path": str((cand_dir / "model.pkl").relative_to(REPO_ROOT)),
                "feature_order_source": str((cand_dir / "feature_schema.json").relative_to(REPO_ROOT)),
                "artifact_identity_verified": True,
            }
        )
        updated.pop("disabled_reason", None)
        by_id[cid] = updated
        restored_ids.append(cid)

    if dry_run:
        print(f"[restore-missing] dry_run would restore {len(restored_ids)} candidates")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CONFIG_PATH.with_name(f"paper_forward_candidates.backup_{timestamp}.json")
    shutil.copy2(CONFIG_PATH, backup)
    payload["candidates"] = list(by_id.values())
    payload["restore_missing_metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "restored_candidate_ids": restored_ids,
        "dataset": str(dataset),
        "sample_rows": sample_rows,
        "enabled_count": sum(1 for c in payload["candidates"] if c.get("enabled")),
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[restore-missing] backup={backup}")
    print(f"[restore-missing] wrote={CONFIG_PATH} restored={len(restored_ids)} enabled_total={payload['restore_missing_metadata']['enabled_count']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "data" / "processed" / "nifty_option_chain_cost_aware_edge_dataset_20260610_121730.csv"),
    )
    parser.add_argument("--sample-rows", type=int, default=30000)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return restore_missing(
        dataset=Path(args.dataset),
        sample_rows=args.sample_rows,
        min_rows=args.min_rows,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())