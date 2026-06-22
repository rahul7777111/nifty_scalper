from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DB_PATH = REPO_ROOT / "trades.db"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db import DatabaseManager


def compute_auc(y_true: List[int], y_prob: List[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    pos = [p for y, p in zip(y_true, y_prob) if y == 1]
    neg = [p for y, p in zip(y_true, y_prob) if y == 0]
    if not pos or not neg:
        return None
    combined = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg], key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(combined, start=1):
        if label == 1:
            rank_sum += rank
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)


def compute_classification_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "resolved_labels": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "false_positives": 0,
            "false_negatives": 0,
        }

    y_true = [int(row["actual_label"]) for row in rows]
    y_prob = [float(row["probability"]) for row in rows]
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    total = len(rows)
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "resolved_labels": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": compute_auc(y_true, y_prob),
        "false_positives": fp,
        "false_negatives": fn,
    }


def bucket_label(probability: float) -> str:
    lower = math.floor(probability * 10.0) / 10.0
    upper = min(1.0, lower + 0.1)
    return f"{lower:.1f}-{upper:.1f}"


def load_report_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    DatabaseManager(str(DB_PATH))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, prediction_id, ts, source, reason, probability, confidence, prediction, symbol, direction, regime,
               trade_id, threshold, future_label_status, resolved_label, realized_forward_return, realized_trade_pnl
        FROM predictions
        WHERE COALESCE(source, '') NOT LIKE 'SHADOW%'
        ORDER BY id ASC
        """
    )
    prediction_rows = [dict(row) for row in cur.fetchall()]
    cur.execute(
        """
        SELECT trade_id, prediction_id, pnl, confidence, timestamp
        FROM trade_outcomes
        WHERE trade_id NOT LIKE 'NS_SHADOW_%'
        ORDER BY id ASC
        """
    )
    outcome_rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    latest_outcome_by_trade: Dict[str, Dict[str, Any]] = {}
    latest_outcome_by_prediction: Dict[str, Dict[str, Any]] = {}
    for row in outcome_rows:
        latest_outcome_by_trade[str(row["trade_id"])] = row
        prediction_id = str(row.get("prediction_id") or "").strip()
        if prediction_id:
            latest_outcome_by_prediction[prediction_id] = row

    resolved: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for pred in prediction_rows:
        prediction_id = str(pred.get("prediction_id") or "").strip()
        trade_id = str(pred.get("trade_id") or pred.get("reason") or "").strip()
        matched = latest_outcome_by_prediction.get(prediction_id) or latest_outcome_by_trade.get(trade_id)
        probability = pred.get("probability")
        if probability is None:
            probability = pred.get("confidence")
        if probability is None:
            probability = pred.get("prediction")
        probability = float(probability or 0.0)
        enriched = {
            **pred,
            "probability": probability,
        }
        if matched is None:
            if pred.get("future_label_status") == "resolved" and pred.get("resolved_label") is not None:
                enriched["actual_label"] = int(pred.get("resolved_label") or 0)
                enriched["pnl"] = float(pred.get("realized_trade_pnl") or 0.0)
                resolved.append(enriched)
                continue
            unresolved.append(enriched)
            continue
        enriched["pnl"] = float(matched.get("pnl") or 0.0)
        enriched["actual_label"] = 1 if float(matched.get("pnl") or 0.0) > 0.0 else 0
        resolved.append(enriched)
    return resolved, unresolved


def build_report() -> Dict[str, Any]:
    resolved, unresolved = load_report_rows()
    metrics = compute_classification_metrics(resolved)

    expectancy_buckets: Dict[str, List[float]] = defaultdict(list)
    calibration_buckets: Dict[str, List[int]] = defaultdict(list)
    for row in resolved:
        label = bucket_label(float(row["probability"]))
        expectancy_buckets[label].append(float(row["pnl"]))
        calibration_buckets[label].append(int(row["actual_label"]))

    expectancy_by_bucket = {
        bucket: {
            "count": len(values),
            "expectancy": (sum(values) / len(values)) if values else 0.0,
        }
        for bucket, values in sorted(expectancy_buckets.items())
    }
    calibration_by_bucket = {
        bucket: {
            "count": len(values),
            "predicted_probability_mid": (float(bucket.split("-")[0]) + float(bucket.split("-")[1])) / 2.0,
            "actual_win_rate": (sum(values) / len(values)) if values else 0.0,
        }
        for bucket, values in sorted(calibration_buckets.items())
    }

    return {
        "total_predictions": len(resolved) + len(unresolved),
        "resolved_labels": len(resolved),
        "unresolved_pending_labels": len(unresolved),
        **metrics,
        "expectancy_by_confidence_bucket": expectancy_by_bucket,
        "calibration_by_probability_bucket": calibration_by_bucket,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Live Prediction Accuracy",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "total_predictions",
        "resolved_labels",
        "unresolved_pending_labels",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positives",
        "false_negatives",
    ):
        lines.append(f"| {key} | {report.get(key)} |")
    lines.extend(["", "## Expectancy By Confidence Bucket", "", "| Bucket | Count | Expectancy |", "|---|---:|---:|"])
    for bucket, values in report.get("expectancy_by_confidence_bucket", {}).items():
        lines.append(f"| {bucket} | {values['count']} | {values['expectancy']:.4f} |")
    lines.extend(["", "## Calibration By Probability Bucket", "", "| Bucket | Count | Predicted Mid | Actual Win Rate |", "|---|---:|---:|---:|"])
    for bucket, values in report.get("calibration_by_probability_bucket", {}).items():
        lines.append(
            f"| {bucket} | {values['count']} | {values['predicted_probability_mid']:.4f} | {values['actual_win_rate']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = REPORTS_DIR / "live_prediction_accuracy.json"
    md_path = REPORTS_DIR / "live_prediction_accuracy.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
