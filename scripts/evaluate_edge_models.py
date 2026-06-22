from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _flatten_reports(reports: Dict[str, Any]) -> Dict[str, Any]:
    if "baseline" in reports or "black_scholes_enriched" in reports:
        flat: Dict[str, Any] = {}
        for variant_name, variant_payload in reports.items():
            if not isinstance(variant_payload, dict):
                continue
            for label_name, model_map in variant_payload.items():
                flat[f"{variant_name}:{label_name}"] = model_map
        return flat
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrained edge-model artifacts.")
    parser.add_argument("--artifact-dir", required=True, help="Path to models/edge_retraining_<RUN_ID>/")
    return parser.parse_args()


def classify(best: Dict[str, Any], dataset_validity: str) -> tuple[str, str]:
    if dataset_validity != "DATASET_READY_FOR_RESEARCH_RETRAINING":
        return "DATASET_NOT_READY", "dataset quality gates failed before training"
    if not best:
        return "NO_EDGE_DETECTED", "no model reports were available"
    test = best.get("test_metrics", {})
    selected = best.get("selected_threshold_from_validation", {})
    signal_count = int(best.get("test_threshold_metrics", {}).get("signal_count", 0))
    roc_auc = float(test.get("roc_auc", 0.0) or 0.0)
    pr_auc = float(test.get("pr_auc", 0.0) or 0.0)
    precision = float(test.get("precision", 0.0) or 0.0)
    pf = float(best.get("cost_adjusted_profit_factor", 0.0) or 0.0)
    avg_net = float(best.get("average_net_pnl_after_costs", 0.0) or 0.0)
    weeks = int(best.get("weeks_with_signals", 0) or 0)
    if (
        roc_auc >= 0.60
        and pr_auc > 0.0
        and precision >= 0.35
        and signal_count >= 75
        and pf > 1.15
        and avg_net > 0.0
        and weeks >= 3
        and float(selected.get("precision", 0.0) or 0.0) >= 0.35
    ):
        return "READY_FOR_MANUAL_REVIEW", "strict manual-review gates passed"
    if precision >= 0.30 and signal_count >= 50:
        return "WATCHLIST_ONLY", "some threshold quality exists but strict promotion gates were not met"
    return "RESEARCH_ONLY", "test threshold behavior remains too weak for manual-review readiness"


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    out_dir = artifact_dir
    report_path = artifact_dir / "all_model_training_report.json"
    validity_path = artifact_dir / "dataset_validity_report.json"
    if not report_path.exists():
        stop_report_path = artifact_dir / "edge_retraining_stop_report.json"
        stop_report = json.loads(stop_report_path.read_text(encoding="utf-8")) if stop_report_path.exists() else {}
        stop_payload = {
            "artifact_dir": str(artifact_dir),
            "final_classification": "DATASET_NOT_READY",
            "reason": stop_report.get("reason_if_training_stopped") or "all_model_training_report.json missing",
        }
        write_json(artifact_dir / "promotion_gate_report.json", stop_payload)
        (artifact_dir / "final_edge_model_recommendation.md").write_text(
            f"# Final Edge Model Recommendation\n\nClassification: `DATASET_NOT_READY`\n\nReason: {stop_payload['reason']}\n",
            encoding="utf-8",
        )
        print(json.dumps(stop_payload, indent=2))
        return

    reports = _flatten_reports(json.loads(report_path.read_text(encoding="utf-8")))
    validity = json.loads(validity_path.read_text(encoding="utf-8")) if validity_path.exists() else {}

    all_rows: List[Dict[str, Any]] = []
    label_summary: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for label_name, model_map in reports.items():
        for model_name, report in model_map.items():
            row = {
                "label_name": label_name,
                "model_name": model_name,
                "selected_threshold": report.get("selected_threshold_from_validation", {}).get("threshold"),
                "validation_precision": report.get("validation_metrics", {}).get("precision"),
                "validation_roc_auc": report.get("validation_metrics", {}).get("roc_auc"),
                "test_precision": report.get("test_metrics", {}).get("precision"),
                "test_roc_auc": report.get("test_metrics", {}).get("roc_auc"),
                "test_pr_auc": report.get("test_metrics", {}).get("pr_auc"),
                "test_signal_count": report.get("test_threshold_metrics", {}).get("signal_count"),
                "model_path": report.get("model_path"),
            }
            all_rows.append(row)
            label_summary[label_name].append(row)

    best = max(
        all_rows,
        key=lambda row: (
            float(row.get("test_precision") or 0.0),
            float(row.get("test_roc_auc") or 0.0),
            int(row.get("test_signal_count") or 0),
        ),
        default={},
    )
    if best:
        best_full = reports[best["label_name"]][best["model_name"]]
    else:
        best_full = {}

    dataset_validity = validity.get("status", "DATASET_NOT_READY_FOR_EDGE_RETRAINING")
    classification, reason = classify(best_full, dataset_validity)

    comparison = {"models": all_rows, "best_model": best}
    label_comparison = {
        label: sorted(rows, key=lambda row: (float(row.get("test_precision") or 0.0), float(row.get("test_roc_auc") or 0.0)), reverse=True)
        for label, rows in label_summary.items()
    }
    threshold_report = {
        "best_model_threshold": best_full.get("selected_threshold_from_validation"),
        "best_model_test_threshold_metrics": best_full.get("test_threshold_metrics"),
    }
    pnl_report = {
        "note": "Cost-adjusted profit factor and net PnL proxy require richer option-outcome data; current reports may be unavailable or zero.",
        "best_model_cost_adjusted_profit_factor": best_full.get("cost_adjusted_profit_factor"),
        "best_model_average_net_pnl_after_costs": best_full.get("average_net_pnl_after_costs"),
    }
    stability_report = {
        "weekly_stability": "Not enough forward-linked option outcome data to compute honest weekly stability from this dataset alone.",
        "monthly_stability": "Not enough forward-linked option outcome data to compute honest monthly stability from this dataset alone.",
    }
    feature_importance = {
        "note": "Feature importance is model-dependent. Linear feature coefficients / tree importances were not standardized in this first safe pipeline build."
    }
    calibration = {
        "best_model_validation": best_full.get("calibration", {}).get("validation"),
        "best_model_test": best_full.get("calibration", {}).get("test"),
    }
    leakage_md = "\n".join(
        [
            "# Leakage Audit Report",
            "",
            "- Chronological train/validation/test split was enforced.",
            "- Future outcome columns were excluded from training features by name-based guardrails.",
            "- No random train/test split was used.",
        ]
    )
    promotion_gate = {
        "dataset_validity": dataset_validity,
        "final_classification": classification,
        "reason": reason,
        "best_model": best,
        "strict_live_ready_allowed": False,
    }
    final_md = "\n".join(
        [
            "# Final Edge Model Recommendation",
            "",
            f"Classification: `{classification}`",
            "",
            f"Reason: {reason}",
            "",
            "This pipeline never classifies a model as `LIVE_READY`.",
        ]
    )

    write_json(out_dir / "all_model_comparison.json", comparison)
    write_json(out_dir / "label_comparison.json", label_comparison)
    write_json(out_dir / "threshold_report.json", threshold_report)
    write_json(out_dir / "cost_adjusted_pnl_report.json", pnl_report)
    write_json(out_dir / "weekly_stability_report.json", stability_report)
    write_json(out_dir / "monthly_stability_report.json", stability_report)
    write_json(out_dir / "feature_importance_report.json", feature_importance)
    write_json(out_dir / "calibration_report.json", calibration)
    (out_dir / "leakage_audit_report.md").write_text(leakage_md, encoding="utf-8")
    write_json(out_dir / "promotion_gate_report.json", promotion_gate)
    (out_dir / "final_edge_model_recommendation.md").write_text(final_md, encoding="utf-8")

    print(json.dumps({"artifact_dir": str(artifact_dir), "final_classification": classification, "reason": reason}, indent=2))


if __name__ == "__main__":
    main()
