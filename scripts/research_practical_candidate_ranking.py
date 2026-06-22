from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "models"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_signal_quality_improvement import (
    BEST_MODEL,
    BEST_POLICY,
    BEST_TIMEFRAME,
    THRESHOLD_GRID,
    add_context_columns,
    apply_threshold_report,
    build_base_dataset,
    calibration_report,
    fit_base_logistic,
)
from research_1y_diagnostics import latest_artifact_dir
from retraining_validation import classification_metrics


FLOORS = [25, 50, 75, 100, 150]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only practical candidate ranking.")
    parser.add_argument("--signal-quality-dir", default="models/research_signal_quality_20260603_211526")
    parser.add_argument("--diagnostics-dir", default="models/research_1y_diagnostics_20260603_201856")
    parser.add_argument("--artifact-dir", default="models/retrained_1y_nifty_20260603_192015")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / raw


def candidate_from_file(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    payload["_source_file"] = str(path)
    return payload


def load_candidate_inputs(signal_quality_dir: Path, diagnostics_dir: Path) -> Dict[str, Any]:
    files = {p.name: p for p in signal_quality_dir.glob("*") if p.is_file()}
    diag_files = {p.name: p for p in diagnostics_dir.glob("*") if p.is_file()}
    return {
        "regime_filter_report": read_json(files["regime_filter_report.json"]),
        "abstention_policy_report": read_json(files["abstention_policy_report.json"]),
        "validation_threshold_selection": read_json(files["validation_threshold_selection.json"]),
        "calibration_report": read_json(files["calibration_report.json"]),
        "candidate_files": [candidate_from_file(p) for p in sorted(signal_quality_dir.glob("candidate_signal_filter_*.json"))],
        "final_signal_quality_recommendation": files.get("final_signal_quality_recommendation.md").read_text(encoding="utf-8") if files.get("final_signal_quality_recommendation.md") else "",
        "threshold_diagnostics": read_json(diag_files["threshold_diagnostics.json"]),
        "regime_diagnostics": read_json(diag_files["regime_diagnostics.json"]),
        "model_comparison_report": read_json(diag_files["model_comparison_report.json"]),
    }


def build_probability_views(base: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    calibration_rows, _ = calibration_report(base)
    views = {"none": {"val_prob": base["val_prob"], "test_prob": base["test_prob"]}}
    for row in calibration_rows:
        name = str(row["calibration_method"])
        if name == "none":
            continue
        # Current calibration report only stores aggregates, so practical ranking falls back to `none`.
        # Keep the key for completeness but do not synthesize unsupported sample-level probabilities.
    return views


def mask_from_filter_rule(frame: pd.DataFrame, rule: str) -> np.ndarray:
    if rule == "none":
        return np.ones(len(frame), dtype=bool)
    if "=" in rule:
        key, value = rule.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in frame.columns:
            return frame[key].astype(str).to_numpy() == value
    return np.ones(len(frame), dtype=bool)


def mask_from_abstention_rule(frame: pd.DataFrame, rule: str) -> np.ndarray:
    if rule in {"none", "", None}:
        return np.ones(len(frame), dtype=bool)
    if rule == "skip_opening_session":
        return frame["session_bucket"].astype(str).to_numpy() != "opening"
    if rule == "skip_closing_session":
        return frame["session_bucket"].astype(str).to_numpy() != "closing"
    if rule == "skip_top_19pct_realized_vol_30":
        cutoff = frame["realized_vol_30"].quantile(0.81)
        return frame["realized_vol_30"].to_numpy() <= float(cutoff)
    if rule == "skip_top_9pct_realized_vol_30":
        cutoff = frame["realized_vol_30"].quantile(0.91)
        return frame["realized_vol_30"].to_numpy() <= float(cutoff)
    if rule == "skip_top_19pct_atr_pct":
        cutoff = frame["atr_pct"].quantile(0.81)
        return frame["atr_pct"].to_numpy() <= float(cutoff)
    if rule == "skip_top_9pct_atr_pct":
        cutoff = frame["atr_pct"].quantile(0.91)
        return frame["atr_pct"].to_numpy() <= float(cutoff)
    if rule == "skip_multi_feature_drift_score_ge_2":
        score = (
            (frame["realized_vol_30"] > frame["realized_vol_30"].quantile(0.8)).astype(int)
            + (frame["atr_pct"] > frame["atr_pct"].quantile(0.8)).astype(int)
            + (frame["ret_std"] > frame["ret_std"].quantile(0.8)).astype(int)
        )
        return score.to_numpy() < 2
    return np.ones(len(frame), dtype=bool)


def monthly_candidate_stats(
    frame: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    labels: np.ndarray,
) -> Dict[str, Any]:
    work = frame.copy()
    work["prob"] = probs
    work["y"] = labels
    work["signal"] = (work["prob"] >= float(threshold)).astype(int)
    month_rows: List[Dict[str, Any]] = []
    signal_counts = []
    positive_signal_counts = []
    month_precisions = []
    for month, group in work.groupby("month", sort=True):
        signals = group[group["signal"] == 1]
        signal_count = int(len(signals))
        tp = int(np.sum((signals["y"] == 1)))
        precision = float(tp / signal_count) if signal_count else 0.0
        month_rows.append(
            {
                "month": month,
                "signal_count": signal_count,
                "precision": precision,
                "positive_outcomes": tp,
            }
        )
        signal_counts.append(signal_count)
        positive_signal_counts.append(tp)
        if signal_count > 0:
            month_precisions.append(precision)
    total_signals = max(1, sum(signal_counts))
    total_positive = max(1, sum(positive_signal_counts))
    return {
        "months": month_rows,
        "worst_month_precision": float(min(month_precisions)) if month_precisions else 0.0,
        "best_month_precision": float(max(month_precisions)) if month_precisions else 0.0,
        "months_with_lt_10_signals": int(sum(1 for count in signal_counts if count < 10)),
        "one_month_gt_50pct_signals": bool(any(count / total_signals > 0.5 for count in signal_counts if total_signals)),
        "one_month_gt_50pct_positive_outcomes": bool(any(count / total_positive > 0.5 for count in positive_signal_counts if total_positive)),
        "monthly_consistency_score": float(np.mean(month_precisions)) if month_precisions else 0.0,
    }


def candidate_metrics(
    name: str,
    filter_rule: str,
    abstention_rule: str,
    threshold: float,
    calibration_method: str,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_prob: np.ndarray,
    test_prob: np.ndarray,
    fwd_val: np.ndarray,
    fwd_test: np.ndarray,
) -> Dict[str, Any]:
    val_mask = mask_from_filter_rule(val_frame, filter_rule) & mask_from_abstention_rule(val_frame, abstention_rule)
    test_mask = mask_from_filter_rule(test_frame, filter_rule) & mask_from_abstention_rule(test_frame, abstention_rule)
    val_report = apply_threshold_report(y_val[val_mask], val_prob[val_mask], fwd_val[val_mask], threshold)
    test_report = apply_threshold_report(y_test[test_mask], test_prob[test_mask], fwd_test[test_mask], threshold)
    val_precision = float(val_report["precision"])
    test_precision = float(test_report["precision"])
    monthly = monthly_candidate_stats(test_frame.loc[test_mask].reset_index(drop=True), test_prob[test_mask], threshold, y_test[test_mask])
    test_pr_auc = float(pr_auc_or_zero(y_test[test_mask], test_prob[test_mask]))
    robustness = candidate_robustness_score(
        test_precision=test_precision,
        test_signal_count=int(test_report["signal_count"]),
        monthly_consistency=monthly["monthly_consistency_score"],
        validation_test_gap=abs(val_precision - test_precision),
        pr_auc=test_pr_auc,
    )
    return {
        "name": name,
        "model_name": BEST_MODEL,
        "label_policy": BEST_POLICY,
        "timeframe": BEST_TIMEFRAME,
        "calibration_method": calibration_method,
        "threshold": float(threshold),
        "filter_rule": filter_rule,
        "abstention_rule": abstention_rule,
        "validation_precision": val_precision,
        "validation_signal_count": int(val_report["signal_count"]),
        "validation_recall": float(val_report["recall"]),
        "test_precision": test_precision,
        "test_signal_count": int(test_report["signal_count"]),
        "test_recall": float(test_report["recall"]),
        "precision_gap": float(val_precision - test_precision),
        "pr_auc": test_pr_auc,
        "monthly_diagnostics": monthly,
        "robustness_score": robustness,
        "validation_report": val_report,
        "test_report": test_report,
    }


def pr_auc_or_zero(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    try:
        from research_1y_diagnostics import pr_auc_score_safe

        return float(pr_auc_score_safe(y_true, y_prob) or 0.0)
    except Exception:
        return 0.0


def candidate_robustness_score(
    *,
    test_precision: float,
    test_signal_count: int,
    monthly_consistency: float,
    validation_test_gap: float,
    pr_auc: float,
) -> float:
    return (
        test_precision * 0.40
        + min(test_signal_count / 100.0, 1.0) * 0.20
        + monthly_consistency * 0.20
        + max(0.0, 1.0 - abs(validation_test_gap)) * 0.10
        + min(max(pr_auc, 0.0), 1.0) * 0.10
    )


def rejected_for_practicality(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    if candidate["test_signal_count"] < 25:
        return True, "fewer_than_25_test_signals"
    if candidate["monthly_diagnostics"]["one_month_gt_50pct_signals"]:
        return True, "one_month_contributes_more_than_half_of_signals"
    if abs(candidate["precision_gap"]) > 0.25:
        return True, "validation_test_precision_gap_too_large"
    if candidate["test_precision"] > candidate["validation_precision"] and candidate["test_signal_count"] < 30:
        return True, "precision_improvement_is_tiny_sample"
    return False, ""


def floor_winner(candidates: List[Dict[str, Any]], floor: int) -> Optional[Dict[str, Any]]:
    eligible = [c for c in candidates if c["test_signal_count"] >= floor]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda c: (
            c["test_precision"],
            c["validation_precision"],
            -abs(c["precision_gap"]),
            c["test_signal_count"],
            c["monthly_diagnostics"]["monthly_consistency_score"],
            c["pr_auc"],
        ),
        reverse=True,
    )[0]


def main() -> int:
    args = parse_args()
    signal_quality_dir = resolve_dir(args.signal_quality_dir)
    diagnostics_dir = resolve_dir(args.diagnostics_dir)
    artifact_dir = resolve_dir(args.artifact_dir)
    out_dir = MODELS_DIR / f"research_practical_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_candidate_inputs(signal_quality_dir, diagnostics_dir)
    base_dataset = build_base_dataset(artifact_dir)
    feature_df = add_context_columns(base_dataset["dataset"], base_dataset["frame"])
    base = fit_base_logistic(base_dataset["dataset"])
    views = build_probability_views(base)

    split = base["split"]
    val_frame = feature_df.iloc[split["slices"]["validation"]].reset_index(drop=True)
    test_frame = feature_df.iloc[split["slices"]["test"]].reset_index(drop=True)
    val_prob = views["none"]["val_prob"]
    test_prob = views["none"]["test_prob"]

    candidates: List[Dict[str, Any]] = []
    # Explicit key candidates
    explicit = [
        ("ret_std_bucket=high", "none", 0.65, "none"),
        ("realized_vol_30_bucket=high", "none", 0.65, "none"),
        ("atr_pct_bucket=high", "none", 0.65, "none"),
        ("weekday=3", "none", 0.65, "none"),
        ("none", "none", 0.65, "none"),
    ]
    for filter_rule, abstain_rule, threshold, calib in explicit:
        candidates.append(
            candidate_metrics(
                name=f"{filter_rule}|{abstain_rule}|{threshold}",
                filter_rule=filter_rule,
                abstention_rule=abstain_rule,
                threshold=threshold,
                calibration_method=calib,
                val_frame=val_frame,
                test_frame=test_frame,
                y_val=base["y_val"],
                y_test=base["y_test"],
                val_prob=val_prob,
                test_prob=test_prob,
                fwd_val=base["forward_val"],
                fwd_test=base["forward_test"],
            )
        )

    # Candidate files from previous script
    for idx, c in enumerate(inputs["candidate_files"], start=1):
        candidates.append(
            candidate_metrics(
                name=f"candidate_file_{idx}",
                filter_rule=str(c.get("filter_rule") or "none"),
                abstention_rule=str(c.get("abstention_rule") or "none"),
                threshold=float(c.get("threshold") or 0.65),
                calibration_method=str(c.get("calibration_method") or "none"),
                val_frame=val_frame,
                test_frame=test_frame,
                y_val=base["y_val"],
                y_test=base["y_test"],
                val_prob=val_prob,
                test_prob=test_prob,
                fwd_val=base["forward_val"],
                fwd_test=base["forward_test"],
            )
        )

    # Practical candidates from prior reports
    for row in inputs["regime_filter_report"]:
        if row.get("sample_size_too_small"):
            continue
        if float(row.get("precision_test", 0.0)) >= 0.25:
            candidates.append(
                candidate_metrics(
                    name=f"regime::{row['filter_column']}={row['bucket']}",
                    filter_rule=f"{row['filter_column']}={row['bucket']}",
                    abstention_rule="none",
                    threshold=float(row["selected_threshold"]),
                    calibration_method="none",
                    val_frame=val_frame,
                    test_frame=test_frame,
                    y_val=base["y_val"],
                    y_test=base["y_test"],
                    val_prob=val_prob,
                    test_prob=test_prob,
                    fwd_val=base["forward_val"],
                    fwd_test=base["forward_test"],
                )
            )

    for row in inputs["abstention_policy_report"]:
        if float(row.get("test_precision", 0.0)) >= 0.25:
            candidates.append(
                candidate_metrics(
                    name=f"abstention::{row['policy']}",
                    filter_rule="none",
                    abstention_rule=str(row["policy"]),
                    threshold=float(row["selected_threshold"]),
                    calibration_method="none",
                    val_frame=val_frame,
                    test_frame=test_frame,
                    y_val=base["y_val"],
                    y_test=base["y_test"],
                    val_prob=val_prob,
                    test_prob=test_prob,
                    fwd_val=base["forward_val"],
                    fwd_test=base["forward_test"],
                )
            )

    deduped: Dict[Tuple[str, str, float, str], Dict[str, Any]] = {}
    for c in candidates:
        key = (c["filter_rule"], c["abstention_rule"], float(c["threshold"]), c["calibration_method"])
        if key not in deduped or c["robustness_score"] > deduped[key]["robustness_score"]:
            deduped[key] = c
    candidates = list(deduped.values())

    rejected = []
    accepted = []
    for c in candidates:
        reject, reason = rejected_for_practicality(c)
        c["practicality_decision"] = "REJECTED_FOR_PRACTICALITY" if reject else "RETAINED_FOR_RANKING"
        c["practicality_reason"] = reason
        (rejected if reject else accepted).append(c)

    floor_report = []
    for floor in FLOORS:
        winner = floor_winner(accepted, floor)
        floor_report.append(
            {
                "minimum_test_signals": floor,
                "winner": winner,
            }
        )

    practical_ranked = sorted(
        accepted,
        key=lambda c: (c["robustness_score"], c["test_precision"], c["test_signal_count"]),
        reverse=True,
    )

    paper_watchlist = []
    for c in practical_ranked:
        monthly = c["monthly_diagnostics"]
        if (
            c["test_precision"] >= 0.30
            and c["test_signal_count"] >= 50
            and not monthly["one_month_gt_50pct_signals"]
            and abs(c["precision_gap"]) <= 0.20
        ):
            watch = {
                "model_name": c["model_name"],
                "label_policy": c["label_policy"],
                "timeframe": c["timeframe"],
                "threshold": c["threshold"],
                "filter_rule": c["filter_rule"],
                "abstention_rule": c["abstention_rule"],
                "validation_metrics": {
                    "precision": c["validation_precision"],
                    "signal_count": c["validation_signal_count"],
                    "recall": c["validation_recall"],
                },
                "test_metrics": {
                    "precision": c["test_precision"],
                    "signal_count": c["test_signal_count"],
                    "recall": c["test_recall"],
                    "pr_auc": c["pr_auc"],
                },
                "month_wise_diagnostics": c["monthly_diagnostics"],
                "warnings": [],
                "live_promoted": False,
                "paper_trade_enabled": False,
                "decision": "PAPER WATCHLIST ONLY",
            }
            paper_watchlist.append(watch)
    paper_watchlist = paper_watchlist[:3]

    for idx, watch in enumerate(paper_watchlist, start=1):
        write_json(out_dir / f"paper_watchlist_candidate_{idx}.json", watch)

    write_json(out_dir / "practical_candidate_ranking.json", practical_ranked)
    write_json(out_dir / "minimum_signal_floor_report.json", floor_report)
    write_json(out_dir / "stability_report.json", {c["name"]: c["monthly_diagnostics"] for c in candidates})

    compare_targets = {c["filter_rule"]: c for c in candidates}
    key_lines = [
        "# Key Candidate Comparison",
        "",
    ]
    for key in ["ret_std_bucket=high", "realized_vol_30_bucket=high", "atr_pct_bucket=high", "weekday=3", "none"]:
        cand = compare_targets.get(key)
        if not cand:
            continue
        key_lines.extend(
            [
                f"## {key}",
                f"- Threshold: `{cand['threshold']}`",
                f"- Filter rule: `{cand['filter_rule']}`",
                f"- Abstention rule: `{cand['abstention_rule']}`",
                f"- Validation precision: `{cand['validation_precision']:.4f}`",
                f"- Test precision: `{cand['test_precision']:.4f}`",
                f"- Test signal count: `{cand['test_signal_count']}`",
                f"- Recall: `{cand['test_recall']:.4f}`",
                f"- Monthly stability: worst `{cand['monthly_diagnostics']['worst_month_precision']:.4f}`, best `{cand['monthly_diagnostics']['best_month_precision']:.4f}`",
                f"- Practicality: `{cand['practicality_decision']}` ({cand['practicality_reason'] or 'retained'})",
                "",
            ]
        )
    (out_dir / "key_candidate_comparison.md").write_text("\n".join(key_lines), encoding="utf-8")

    watch_lines = [
        "# Paper Watchlist Candidates",
        "",
    ]
    if not paper_watchlist:
        watch_lines.append("No candidates cleared the paper-watchlist practicality gates.")
    else:
        for item in paper_watchlist:
            watch_lines.append(f"- `{item['filter_rule']}` at threshold `{item['threshold']}` with test precision `{item['test_metrics']['precision']:.4f}` and `{item['test_metrics']['signal_count']}` signals.")
    (out_dir / "paper_watchlist_candidates.md").write_text("\n".join(watch_lines) + "\n", encoding="utf-8")

    rejected_lines = [
        "# Rejected Candidates",
        "",
    ]
    for item in rejected:
        rejected_lines.append(f"- `{item['filter_rule']}` / `{item['abstention_rule']}` rejected: `{item['practicality_reason']}`.")
    (out_dir / "rejected_candidates.md").write_text("\n".join(rejected_lines) + "\n", encoding="utf-8")

    best50 = floor_winner(accepted, 50)
    if best50 and best50["test_precision"] >= 0.35 and best50["test_signal_count"] >= 75 and abs(best50["precision_gap"]) <= 0.15 and not best50["monthly_diagnostics"]["one_month_gt_50pct_signals"]:
        final_label = "LIMITED PAPER TRADE CANDIDATE"
        reason = "best practical candidate clears the stronger precision, signal-count, and stability gates"
    elif best50:
        final_label = "PAPER WATCHLIST ONLY"
        reason = "some candidates are more practical than the tiny high-precision slices, but stability and/or signal count remain marginal"
    else:
        final_label = "DO NOT PROMOTE"
        reason = "no candidate met the minimum practical signal floors with acceptable stability"

    final_text = "\n".join(
        [
            f"# {final_label}",
            "",
            f"- Best candidate under 50-signal floor: `{best50['filter_rule'] if best50 else 'none'}`" if best50 else "- No candidate met the 50-signal floor.",
            f"- Reason: {reason}",
            "- This script never promotes a live candidate.",
        ]
    ) + "\n"
    (out_dir / "final_practical_recommendation.md").write_text(final_text, encoding="utf-8")

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
