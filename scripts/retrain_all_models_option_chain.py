from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import (
    adopt_option_chain_models_if_passed,
    audit_option_chain_dataset,
    build_processed_option_chain_dataset,
    compare_old_vs_new,
    evaluate_option_chain_artifacts,
    train_option_chain_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the safe option-chain retraining pipeline.")
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--processed-dataset", default="")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--old-model-dir", default="")
    parser.add_argument("--adopt-if-passed", action="store_true")
    parser.add_argument("--no-adopt", action="store_true")
    parser.add_argument("--min-roc-auc", type=float, default=0.55)
    parser.add_argument("--min-profit-factor", type=float, default=1.15)
    parser.add_argument("--min-sharpe", type=float, default=1.0)
    parser.add_argument("--min-holdout-trades", type=int, default=200)
    parser.add_argument("--min-active-days", type=int, default=20)
    parser.add_argument("--min-recall", type=float, default=0.05)
    parser.add_argument("--min-f1", type=float, default=0.05)
    parser.add_argument("--max-single-day-profit-contribution", type=float, default=0.25)
    parser.add_argument("--min-coverage-rate", type=float, default=0.01)
    parser.add_argument("--horizon-bars", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_folder = Path(args.dataset_folder)
    audit_payload, _, _ = audit_option_chain_dataset(dataset_folder)
    if args.processed_dataset:
        processed_dataset = Path(args.processed_dataset)
        build_payload = {"processed_dataset": str(processed_dataset)}
    else:
        build_payload = build_processed_option_chain_dataset(dataset_folder, horizon_bars=int(args.horizon_bars))
        processed_dataset = Path(build_payload["processed_dataset"])
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    train_payload = train_option_chain_models(
        processed_dataset,
        artifact_dir=artifact_dir,
        old_model_dir=Path(args.old_model_dir) if args.old_model_dir else None,
        min_roc_auc=float(args.min_roc_auc),
        min_profit_factor=float(args.min_profit_factor),
        min_sharpe=float(args.min_sharpe),
        min_holdout_trades=int(args.min_holdout_trades),
        min_active_days=int(args.min_active_days),
        min_recall=float(args.min_recall),
        min_f1=float(args.min_f1),
        max_single_day_profit_contribution=float(args.max_single_day_profit_contribution),
        min_coverage_rate=float(args.min_coverage_rate),
    )
    eval_payload = evaluate_option_chain_artifacts(Path(train_payload["artifact_dir"]))
    compare_payload = compare_old_vs_new(Path(train_payload["artifact_dir"]), Path(args.old_model_dir) if args.old_model_dir else None)
    adopt_payload = {"adopted": False, "reason": "adoption not requested"}
    should_adopt = bool(args.adopt_if_passed) and not bool(args.no_adopt)
    if should_adopt:
        adopt_payload = adopt_option_chain_models_if_passed(Path(train_payload["artifact_dir"]))
    summary = {
        "dataset_audit": audit_payload,
        "build": build_payload,
        "training": {
            "artifact_dir": train_payload["artifact_dir"],
            "champion_model": train_payload["champion"]["model_name"],
            "models_retrained": sorted(train_payload["model_reports"].keys()),
        },
        "evaluation": eval_payload,
        "comparison": compare_payload,
        "adoption": adopt_payload,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
