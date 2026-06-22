from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_chain_pipeline_lib import (
    check_live_schema_compatibility,
    merge_option_context_into_reconstructed_dataset,
    train_option_chain_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge validated option-context data into the reconstructed research dataset.")
    parser.add_argument("--reconstructed-dataset", required=True)
    parser.add_argument("--option-context-dataset", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--allow-asof", action="store_true")
    parser.add_argument("--asof-tolerance-seconds", type=int, default=300)
    parser.add_argument("--validation-json", default="")
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--run-retrain-if-compatible", action="store_true")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--min-profit-factor", type=float, default=1.15)
    parser.add_argument("--min-sharpe", type=float, default=1.0)
    parser.add_argument("--min-holdout-trades", type=int, default=200)
    parser.add_argument("--min-active-days", type=int, default=20)
    parser.add_argument("--min-recall", type=float, default=0.05)
    parser.add_argument("--min-f1", type=float, default=0.05)
    parser.add_argument("--max-single-day-profit-contribution", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_payload = merge_option_context_into_reconstructed_dataset(
        Path(args.reconstructed_dataset),
        Path(args.option_context_dataset),
        output_dataset=Path(args.output) if args.output else None,
        report_dir=Path(args.report_dir) if args.report_dir else None,
        validation_json=Path(args.validation_json) if args.validation_json else None,
        allow_backward_asof=bool(args.allow_asof),
        asof_tolerance=f"{int(args.asof_tolerance_seconds)}s",
    )
    retrain_payload = None
    strict_gate = merge_payload.get("strict_live_schema_gate", {})
    schema_check = check_live_schema_compatibility(
        Path(merge_payload["output_dataset"]),
        report_dir=Path(args.report_dir) if args.report_dir else None,
    )
    should_run_retrain = bool(args.run_retrain_if_compatible) and not bool(args.no_retrain)
    if strict_gate.get("compatible") and schema_check.get("final_verdict") == "LIVE_SCHEMA_COMPATIBLE_RESEARCH_ONLY" and should_run_retrain:
        artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
        retrain_payload = train_option_chain_models(
            Path(merge_payload["output_dataset"]),
            artifact_dir=artifact_dir,
            min_profit_factor=float(args.min_profit_factor),
            min_sharpe=float(args.min_sharpe),
            min_holdout_trades=int(args.min_holdout_trades),
            min_active_days=int(args.min_active_days),
            min_recall=float(args.min_recall),
            min_f1=float(args.min_f1),
            max_single_day_profit_contribution=float(args.max_single_day_profit_contribution),
        )
        gate_report = retrain_payload.get("gate_report", {})
        verdict = "LIVE_SCHEMA_COMPATIBLE_RESEARCH_GATES_PASSED_NEED_PAPER_TRADING"
        if gate_report.get("adoption_decision") != "PASSED":
            verdict = "LIVE_SCHEMA_COMPATIBLE_METRICS_FAILED"
    else:
        reason = "retraining disabled by default"
        if not strict_gate.get("compatible") or schema_check.get("final_verdict") != "LIVE_SCHEMA_COMPATIBLE_RESEARCH_ONLY":
            reason = "full 84/84 schema compatibility not reached"
        retrain_payload = {"skipped": True, "reason": reason}
        verdict = "WAITING_FOR_OPTION_CONTEXT_DATA"
        if merge_payload.get("validation", {}).get("status") != "valid" or not merge_payload.get("validation", {}).get("ok", True):
            verdict = "OPTION_CONTEXT_DATA_INVALID"
        elif merge_payload.get("final_verdict") == "MERGE_INCOMPLETE":
            verdict = "MERGE_INCOMPLETE"
        elif schema_check.get("final_verdict") == "LIVE_SCHEMA_NOT_COMPATIBLE":
            verdict = "LIVE_SCHEMA_NOT_COMPATIBLE"
    print(json.dumps({
        "merge": merge_payload,
        "schema_check": schema_check,
        "retraining": retrain_payload,
        "final_verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
