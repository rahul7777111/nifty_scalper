from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"
SOURCE_REPORT_PATH = REPORTS_DIR / "final_label_redesign_and_model_adaptation_report.json"
DEGRADATION_REPORT_PATH = REPORTS_DIR / "promotion_degradation_report.json"
DEGRADATION_REPORT_MD_PATH = REPORTS_DIR / "promotion_degradation_report.md"
PRODUCTION_PROFILE_PATH = CONFIG_DIR / "production_model_profile.json"
REGISTRY_PATH = MODELS_DIR / "model_registry_v2.json"
LIVE_MODEL_PATH = REPO_ROOT / "ml_signal_model.pkl"

MIN_TRADES = 20.0
MIN_PRECISION = 0.40
MIN_F1 = 0.30
MAX_DRAWDOWN = 0.15

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("promote_to_production")


@dataclass
class CandidateResult:
    row: Dict[str, Any]
    score: float
    model_path: Path


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        log.warning("Missing required governance report: %s", path)
        raise RuntimeError(f"Required report is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to parse governance report %s: %s", path, exc)
        raise RuntimeError(f"Governance report is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        log.warning("Governance report has unexpected root type: %s", type(payload).__name__)
        raise RuntimeError(f"Governance report must be a JSON object: {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safety_gate_state(report: Dict[str, Any], policy_name: str) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    audit = report.get("label_policy_safety_audit")
    if not isinstance(audit, dict):
        return False, ["missing label_policy_safety_audit block"]
    if not bool(audit.get("passed")):
        failures.append("global safety audit not passed")
    policies = audit.get("policies")
    if not isinstance(policies, list):
        failures.append("missing safety policy details")
        return False, failures
    matching = None
    for row in policies:
        if isinstance(row, dict) and str(row.get("policy") or "") == str(policy_name or ""):
            matching = row
            break
    if matching is None:
        failures.append(f"missing safety policy row for {policy_name}")
        return False, failures
    if not bool(matching.get("passed")):
        failures.append(f"safety policy row failed for {policy_name}")
    if not bool(matching.get("purged_embargo_splits_available")):
        failures.append(f"purged_embargo_splits_available is false for {policy_name}")
    return len(failures) == 0, failures


def _strategy_alignment_state(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    audit = report.get("current_label_audit")
    if not isinstance(audit, dict):
        return False, ["missing current_label_audit block"]
    if not bool(audit.get("labels_match_actual_strategy_exits")):
        failures.append("labels_match_actual_strategy_exits is false")
    return len(failures) == 0, failures


def _benchmark_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    benchmark = report.get("benchmark_matrix")
    if not isinstance(benchmark, dict):
        raise RuntimeError("Benchmark report is incomplete: missing benchmark_matrix block")
    rows = benchmark.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Benchmark report is incomplete: benchmark_matrix.rows is missing or empty")
    filtered = [row for row in rows if isinstance(row, dict)]
    if not filtered:
        raise RuntimeError("Benchmark report is incomplete: no valid benchmark rows found")
    return filtered


def _resolve_model_artifact_path(report: Dict[str, Any], row: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []

    direct_keys = (
        row.get("calibrated_model_path"),
        row.get("model_path"),
        row.get("artifact_path"),
    )
    for raw in direct_keys:
        if raw:
            candidates.append(Path(str(raw)))

    deployment = report.get("deployment")
    if isinstance(deployment, dict):
        for raw in (deployment.get("model_path"), deployment.get("current_model_path")):
            if raw:
                candidates.append(Path(str(raw)))

    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            current = registry.get("current") if isinstance(registry, dict) else None
            if isinstance(current, dict):
                filename = str(current.get("filename") or "").strip()
                if filename:
                    candidates.append(MODELS_DIR / filename)
        except Exception:
            pass

    if LIVE_MODEL_PATH.exists():
        candidates.append(LIVE_MODEL_PATH)

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        if resolved.exists():
            return resolved.resolve()
    return None


def _row_identifier(row: Dict[str, Any]) -> str:
    return (
        f"{str(row.get('label_policy') or 'unknown_policy')} | "
        f"{str(row.get('variant') or 'unknown_variant')} | "
        f"{str(row.get('model') or 'unknown_model')}"
    )


def _evaluate_row(report: Dict[str, Any], row: Dict[str, Any]) -> Tuple[bool, List[str], Optional[CandidateResult]]:
    failures: List[str] = []
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    stability = row.get("stability")
    stability = stability if isinstance(stability, dict) else {}
    label_policy = str(row.get("label_policy") or "")

    safety_ok, safety_failures = _safety_gate_state(report, label_policy)
    if not safety_ok:
        failures.extend(safety_failures)

    alignment_ok, alignment_failures = _strategy_alignment_state(report)
    if not alignment_ok:
        failures.extend(alignment_failures)

    trades_count = _safe_float(metrics.get("trades_count"))
    precision = _safe_float(metrics.get("precision"))
    f1 = _safe_float(metrics.get("f1"))
    expectancy = _safe_float(metrics.get("expectancy"))
    max_drawdown = _safe_float(metrics.get("max_drawdown", metrics.get("drawdown")))
    f1_std = _safe_float(stability.get("f1_std"))

    if trades_count < MIN_TRADES:
        failures.append(f"trades_count {trades_count:.0f} < {MIN_TRADES:.0f}")
    if precision < MIN_PRECISION:
        failures.append(f"precision {precision:.4f} < {MIN_PRECISION:.2f}")
    if f1 < MIN_F1:
        failures.append(f"f1 {f1:.4f} < {MIN_F1:.2f}")
    if expectancy <= 0.0:
        failures.append(f"expectancy {expectancy:.6f} <= 0.0")
    if max_drawdown > MAX_DRAWDOWN:
        failures.append(f"max_drawdown {max_drawdown:.4f} > {MAX_DRAWDOWN:.2f}")

    artifact_path = _resolve_model_artifact_path(report, row)
    if artifact_path is None:
        failures.append("unable to resolve calibrated model artifact path")

    if failures:
        return False, failures, None

    score = (expectancy * trades_count) / (1.0 + f1_std)
    return True, [], CandidateResult(row=row, score=float(score), model_path=artifact_path)


def _render_degradation_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Promotion Degradation Report",
        "",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- Source report: `{payload.get('source_report')}`",
        f"- Rows evaluated: `{payload.get('rows_evaluated')}`",
        f"- Candidates passed: `{payload.get('rows_passed')}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    for row in payload.get("blocked_rows", []):
        identifier = str(row.get("row_id") or "unknown_row")
        lines.append(f"- `{identifier}`")
        for reason in row.get("failures", []):
            lines.append(f"  - {reason}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _promote_profile(champion: CandidateResult, report: Dict[str, Any]) -> Dict[str, Any]:
    row = champion.row
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    stability = row.get("stability")
    stability = stability if isinstance(stability, dict) else {}

    profile = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_report": str(SOURCE_REPORT_PATH),
        "policy_name": str(row.get("label_policy") or ""),
        "label_policy_version": str(row.get("label_policy_version") or ""),
        "feature_variant": str(row.get("variant") or ""),
        "model_name": str(row.get("model") or ""),
        "optimal_threshold": _safe_float(metrics.get("threshold")),
        "calibrated_model_path": str(champion.model_path),
        "live_model_path": str(LIVE_MODEL_PATH),
        "optimization_score": float(champion.score),
        "metrics": {
            "roc_auc": _safe_float(metrics.get("roc_auc")),
            "accuracy": _safe_float(metrics.get("accuracy")),
            "precision": _safe_float(metrics.get("precision")),
            "recall": _safe_float(metrics.get("recall")),
            "f1": _safe_float(metrics.get("f1")),
            "trades_count": _safe_float(metrics.get("trades_count")),
            "profit_factor": _safe_float(metrics.get("profit_factor")),
            "expectancy": _safe_float(metrics.get("expectancy")),
            "sharpe": _safe_float(metrics.get("sharpe")),
            "max_drawdown": _safe_float(metrics.get("max_drawdown", metrics.get("drawdown"))),
            "roc_auc_std": _safe_float(stability.get("roc_auc_std")),
            "f1_std": _safe_float(stability.get("f1_std")),
        },
        "dataset": report.get("benchmark_matrix", {}).get("dataset", {}),
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(champion.model_path, LIVE_MODEL_PATH)
    _write_json(PRODUCTION_PROFILE_PATH, profile)
    return profile


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    report = _read_json(SOURCE_REPORT_PATH)
    rows = _benchmark_rows(report)

    passed_candidates: List[CandidateResult] = []
    blocked_rows: List[Dict[str, Any]] = []

    for row in rows:
        ok, failures, candidate = _evaluate_row(report, row)
        if ok and candidate is not None:
            passed_candidates.append(candidate)
        else:
            blocked_rows.append(
                {
                    "row_id": _row_identifier(row),
                    "failures": failures,
                    "metrics": dict(row.get("metrics") or {}),
                }
            )

    if not passed_candidates:
        degradation_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_report": str(SOURCE_REPORT_PATH),
            "rows_evaluated": len(rows),
            "rows_passed": 0,
            "blocked_rows": blocked_rows,
        }
        _write_json(DEGRADATION_REPORT_PATH, degradation_payload)
        DEGRADATION_REPORT_MD_PATH.write_text(_render_degradation_markdown(degradation_payload), encoding="utf-8")
        log.error("Promotion blocked. No candidate cleared all institutional gates.")
        for blocked in blocked_rows:
            log.error("%s -> %s", blocked["row_id"], "; ".join(blocked["failures"]))
        sys.exit(1)

    champion = sorted(
        passed_candidates,
        key=lambda item: (
            float(item.score),
            -_safe_float(item.row.get("metrics", {}).get("max_drawdown", item.row.get("metrics", {}).get("drawdown")), default=999.0),
        ),
        reverse=True,
    )[0]

    tied = [
        candidate for candidate in passed_candidates
        if abs(candidate.score - champion.score) <= 1e-12
    ]
    if len(tied) > 1:
        champion = min(
            tied,
            key=lambda item: _safe_float(item.row.get("metrics", {}).get("max_drawdown", item.row.get("metrics", {}).get("drawdown")), default=999.0),
        )

    profile = _promote_profile(champion, report)
    log.info("PROMOTION SUCCESS")
    log.info("Policy: %s", profile["policy_name"])
    log.info("Feature variant: %s", profile["feature_variant"])
    log.info("Model: %s", profile["model_name"])
    log.info("Threshold: %.4f", profile["optimal_threshold"])
    log.info("Calibrated artifact: %s", profile["calibrated_model_path"])
    log.info(
        "Metrics | precision=%.4f f1=%.4f expectancy=%.6f trades=%.0f max_drawdown=%.4f score=%.6f",
        _safe_float(profile["metrics"]["precision"]),
        _safe_float(profile["metrics"]["f1"]),
        _safe_float(profile["metrics"]["expectancy"]),
        _safe_float(profile["metrics"]["trades_count"]),
        _safe_float(profile["metrics"]["max_drawdown"]),
        float(profile["optimization_score"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
