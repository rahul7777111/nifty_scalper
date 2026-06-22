from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

try:
    from scipy.stats import ks_2samp  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    ks_2samp = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "trades.db"
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_BASELINE_PATH = REPORTS_DIR / "training_feature_distribution.json"
DEFAULT_REPORT_PATH = REPORTS_DIR / "prediction_drift_report.json"

logger = logging.getLogger(__name__)

CORE_FEATURES = (
    "rolling_range_position_20",
    "realized_vol_30",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PredictionDriftMonitor:
    """Real-time feature and calibration drift worker for live ML trading."""

    def __init__(
        self,
        *,
        baseline_path: Path | None = None,
        report_path: Path | None = None,
        rolling_feature_window: int = 500,
        evaluation_block_size: int = 100,
        completed_trade_window: int = 50,
        psi_threshold: float = 0.20,
        ks_pvalue_threshold: float = 0.01,
        baseline_brier_threshold: float = 0.25,
        confidence_mean_threshold: float = 0.10,
        class_distribution_threshold: float = 0.15,
        calibration_threshold: float = 0.05,
        notifier: Optional[Any] = None,
        strategy_ref: Optional[Any] = None,
    ) -> None:
        self.baseline_path = Path(baseline_path or DEFAULT_BASELINE_PATH)
        self.report_path = Path(report_path or DEFAULT_REPORT_PATH)
        self.rolling_feature_window = int(max(rolling_feature_window, 50))
        self.evaluation_block_size = int(max(evaluation_block_size, 10))
        self.completed_trade_window = int(max(completed_trade_window, 10))
        self.psi_threshold = float(psi_threshold)
        self.ks_pvalue_threshold = float(ks_pvalue_threshold)
        self.baseline_brier_threshold = float(baseline_brier_threshold)
        self.confidence_mean_threshold = float(confidence_mean_threshold)
        self.class_distribution_threshold = float(class_distribution_threshold)
        self.calibration_threshold = float(calibration_threshold)
        self.notifier = notifier
        self.strategy_ref = strategy_ref

        self._lock = threading.RLock()
        self._feature_baseline = self.load_training_distribution(self.baseline_path)
        self._live_features: Dict[str, Deque[float]] = {
            feature: deque(maxlen=self.rolling_feature_window)
            for feature in CORE_FEATURES
        }
        self._resolved_probs: Deque[float] = deque(maxlen=self.completed_trade_window)
        self._resolved_labels: Deque[int] = deque(maxlen=self.completed_trade_window)
        self._processed_predictions: set[str] = set()
        self._new_observations_since_eval = 0
        self._last_report: Dict[str, Any] = {
            "generated_at": _utc_now(),
            "status": "warmup",
            "alerts": [],
            "feature_metrics": {},
            "rolling_brier_score": None,
            "kill_switch_active": False,
        }

    @staticmethod
    def _brier(probs: Sequence[float], labels: Sequence[int]) -> float | None:
        if not probs or not labels or len(probs) != len(labels):
            return None
        arr_p = np.asarray([_safe_float(v) for v in probs], dtype=float)
        arr_y = np.asarray([int(v) for v in labels], dtype=float)
        if arr_p.size == 0 or arr_y.size == 0:
            return None
        return float(np.mean((arr_p - arr_y) ** 2))

    @staticmethod
    def _ks_test(expected: Sequence[float], actual: Sequence[float]) -> Dict[str, float]:
        exp = np.asarray([_safe_float(v) for v in expected], dtype=float)
        act = np.asarray([_safe_float(v) for v in actual], dtype=float)
        if exp.size < 2 or act.size < 2:
            return {"statistic": 0.0, "pvalue": 1.0}
        if ks_2samp is not None:
            result = ks_2samp(exp, act)
            return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}
        exp_sorted = np.sort(exp)
        act_sorted = np.sort(act)
        merged = np.sort(np.concatenate([exp_sorted, act_sorted]))
        exp_cdf = np.searchsorted(exp_sorted, merged, side="right") / float(exp_sorted.size)
        act_cdf = np.searchsorted(act_sorted, merged, side="right") / float(act_sorted.size)
        statistic = float(np.max(np.abs(exp_cdf - act_cdf)))
        en = math.sqrt((exp_sorted.size * act_sorted.size) / float(exp_sorted.size + act_sorted.size))
        pvalue = float(min(1.0, 2.0 * math.exp(-2.0 * (en * statistic) ** 2))) if statistic > 0 else 1.0
        return {"statistic": statistic, "pvalue": pvalue}

    @staticmethod
    def compute_psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
        exp = np.asarray([_safe_float(v) for v in expected], dtype=float)
        act = np.asarray([_safe_float(v) for v in actual], dtype=float)
        if exp.size < 2 or act.size < 2:
            return 0.0
        quantiles = np.quantile(exp, np.linspace(0.0, 1.0, bins + 1))
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        exp_counts, _ = np.histogram(exp, bins=quantiles)
        act_counts, _ = np.histogram(act, bins=quantiles)
        exp_pct = (exp_counts + 0.5) / (exp.size + 0.5 * bins)
        act_pct = (act_counts + 0.5) / (act.size + 0.5 * bins)
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    @staticmethod
    def load_training_distribution(path: Path | None = None) -> Dict[str, List[float]]:
        baseline_path = Path(path or DEFAULT_BASELINE_PATH)
        if not baseline_path.exists():
            return {}
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        features = payload.get("features") if isinstance(payload, dict) else {}
        out: Dict[str, List[float]] = {}
        if isinstance(features, dict):
            for feature_name in CORE_FEATURES:
                stats = features.get(feature_name)
                if not isinstance(stats, dict):
                    continue
                quantiles = stats.get("quantiles") or []
                if isinstance(quantiles, list) and quantiles:
                    out[feature_name] = [_safe_float(v) for v in quantiles]
        return out

    @staticmethod
    def load_prediction_rows(limit: int = 1000) -> List[Dict[str, Any]]:
        if not DB_PATH.exists():
            return []
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.id, p.prediction_id, p.ts, p.probability, p.confidence, p.prediction,
                   p.features_json, p.resolved_label, p.realized_trade_pnl, o.pnl
            FROM predictions p
            LEFT JOIN trade_outcomes o ON o.prediction_id = p.prediction_id OR o.trade_id = p.trade_id
            WHERE COALESCE(p.source, '') = 'ACTIVE'
            ORDER BY p.id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        rows.reverse()
        return rows

    def ingest_live_observation(
        self,
        *,
        feature_snapshot: Dict[str, Any],
        probability: Optional[float] = None,
        prediction_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        del probability, prediction_id
        with self._lock:
            appended = False
            for feature_name in CORE_FEATURES:
                if feature_name in feature_snapshot:
                    self._live_features.setdefault(feature_name, deque(maxlen=self.rolling_feature_window)).append(
                        _safe_float(feature_snapshot.get(feature_name))
                    )
                    appended = True
            if not appended:
                return None
            self._new_observations_since_eval += 1
            if self._new_observations_since_eval < self.evaluation_block_size:
                return None
            self._new_observations_since_eval = 0
            report = self._evaluate_locked()
            self._write_report_locked(report)
            self._handle_alerts_locked(report)
            return report

    def ingest_resolved_outcome(
        self,
        *,
        probability: float,
        realized_label: int,
        prediction_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if prediction_id:
                key = str(prediction_id)
                if key in self._processed_predictions:
                    return None
                self._processed_predictions.add(key)
            self._resolved_probs.append(_safe_float(probability))
            self._resolved_labels.append(1 if int(realized_label) > 0 else 0)
            report = self._evaluate_locked()
            self._write_report_locked(report)
            self._handle_alerts_locked(report)
            return report

    def refresh_from_prediction_db(self, *, limit: int = 500) -> Dict[str, Any]:
        rows = self.load_prediction_rows(limit=limit)
        with self._lock:
            self._rebuild_from_rows_locked(rows)
            report = self._evaluate_locked()
            self._write_report_locked(report)
            self._handle_alerts_locked(report)
            return report

    def _rebuild_from_rows_locked(self, rows: Sequence[Dict[str, Any]]) -> None:
        self._live_features = {
            feature: deque(maxlen=self.rolling_feature_window)
            for feature in CORE_FEATURES
        }
        self._resolved_probs.clear()
        self._resolved_labels.clear()
        self._processed_predictions.clear()
        for row in rows:
            try:
                features = json.loads(str(row.get("features_json") or "{}"))
            except Exception:
                features = {}
            if isinstance(features, dict):
                for feature_name in CORE_FEATURES:
                    if feature_name in features:
                        self._live_features[feature_name].append(_safe_float(features.get(feature_name)))
            prediction_id = str(row.get("prediction_id") or "").strip()
            resolved = row.get("resolved_label")
            if resolved is None:
                pnl_value = row.get("pnl")
                if pnl_value is None:
                    pnl_value = row.get("realized_trade_pnl")
                if pnl_value is not None:
                    resolved = 1 if _safe_float(pnl_value) > 0.0 else 0
            if resolved is None:
                continue
            prob = row.get("probability")
            if prob is None:
                prob = row.get("confidence")
            self._resolved_probs.append(_safe_float(prob))
            self._resolved_labels.append(1 if int(resolved) > 0 else 0)
            if prediction_id:
                self._processed_predictions.add(prediction_id)

    def _evaluate_locked(self) -> Dict[str, Any]:
        feature_metrics: Dict[str, Dict[str, Any]] = {}
        alerts: List[str] = []
        feature_alerts: List[Dict[str, Any]] = []
        for feature_name in CORE_FEATURES:
            baseline = list(self._feature_baseline.get(feature_name) or [])
            live = list(self._live_features.get(feature_name) or [])
            if len(baseline) < 2 or len(live) < 2:
                feature_metrics[feature_name] = {
                    "sample_count": len(live),
                    "status": "insufficient_live_data",
                    "psi_score": None,
                    "ks_statistic": None,
                    "ks_p_value": None,
                }
                continue
            psi_score = self.compute_psi(baseline, live, bins=10)
            ks_result = self._ks_test(baseline, live)
            status = "stable"
            if psi_score >= self.psi_threshold or ks_result["pvalue"] <= self.ks_pvalue_threshold:
                status = "CRITICAL_FEATURE_DRIFT_WARNING"
                feature_alerts.append(
                    {
                        "feature": feature_name,
                        "psi_score": psi_score,
                        "ks_p_value": ks_result["pvalue"],
                    }
                )
                alerts.append(
                    f"CRITICAL_FEATURE_DRIFT_WARNING:{feature_name}:psi={psi_score:.4f},ks_p={ks_result['pvalue']:.6f}"
                )
            feature_metrics[feature_name] = {
                "sample_count": len(live),
                "status": status,
                "psi_score": float(psi_score),
                "ks_statistic": float(ks_result["statistic"]),
                "ks_p_value": float(ks_result["pvalue"]),
                "baseline_count": len(baseline),
            }

        rolling_brier = self._brier(list(self._resolved_probs), list(self._resolved_labels))
        calibration_alert = None
        if rolling_brier is not None and len(self._resolved_probs) >= self.completed_trade_window and rolling_brier > self.baseline_brier_threshold:
            calibration_alert = f"CALIBRATION_COLLAPSE_ALERT:brier={rolling_brier:.4f}"
            alerts.append(calibration_alert)

        # Preserve older report interface for callers/tests that still use bulk evaluation.
        confidence_mean = float(np.mean(list(self._resolved_probs))) if self._resolved_probs else 0.0
        class_rate = float(np.mean([1 if p >= 0.5 else 0 for p in self._resolved_probs])) if self._resolved_probs else 0.0

        status = "stable"
        if alerts:
            status = "alert"
        elif any(row.get("status") == "insufficient_live_data" for row in feature_metrics.values()):
            status = "warmup"

        report = {
            "generated_at": _utc_now(),
            "status": status,
            "alerts": alerts,
            "feature_metrics": feature_metrics,
            "critical_feature_alerts": feature_alerts,
            "rolling_brier_score": rolling_brier,
            "completed_trade_count": len(self._resolved_probs),
            "confidence_mean": confidence_mean,
            "class_rate": class_rate,
            "baseline_brier_threshold": self.baseline_brier_threshold,
            "kill_switch_active": bool(getattr(self.strategy_ref, "_ml_drift_kill_switch_active", False)) if self.strategy_ref is not None else False,
        }
        self._last_report = report
        return report

    def _write_report_locked(self, report: Dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _send_alert(self, message: str) -> None:
        logger.warning(message)
        if self.notifier is not None and hasattr(self.notifier, "send_message"):
            try:
                self.notifier.send_message(f"<b>NiftyScalper Drift Alert</b>\n{message}")
            except Exception:
                logger.exception("Failed to dispatch notifier alert")

    def _engage_kill_switch_locked(self, report: Dict[str, Any]) -> None:
        strategy = self.strategy_ref
        if strategy is None:
            report["kill_switch_active"] = False
            return
        try:
            setattr(strategy, "_ml_drift_kill_switch_active", True)
            setattr(strategy, "_last_ml_bet_multiplier", 0.0)
            if hasattr(strategy, "cfg"):
                try:
                    setattr(strategy.cfg, "enable_ml_signals", False)
                except Exception:
                    pass
            report["kill_switch_active"] = True
        except Exception:
            logger.exception("Failed to engage ML drift kill switch")

    def _handle_alerts_locked(self, report: Dict[str, Any]) -> None:
        critical = any(str(alert).startswith("CRITICAL_FEATURE_DRIFT_WARNING") for alert in report.get("alerts", []))
        calibration = any(str(alert).startswith("CALIBRATION_COLLAPSE_ALERT") for alert in report.get("alerts", []))
        if not critical and not calibration:
            return
        if critical:
            self._send_alert("Critical live feature drift detected. ML gating kill switch engaged.")
        if calibration:
            self._send_alert("Live calibration collapse detected. ML gating kill switch engaged.")
        self._engage_kill_switch_locked(report)
        self._write_report_locked(report)

    def evaluate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Backward-compatible bulk evaluation used by existing tests/reports."""
        with self._lock:
            self._rebuild_from_rows_locked(rows)
            report = self._evaluate_locked()
            if len(rows) < 20:
                report["status"] = "insufficient_live_predictions"
                report["alerts"] = []
            else:
                midpoint = len(rows) // 2
                baseline = rows[:midpoint]
                current = rows[midpoint:]

                def probs(part: List[Dict[str, Any]]) -> List[float]:
                    out: List[float] = []
                    for row in part:
                        value = row.get("probability")
                        if value is None:
                            value = row.get("confidence")
                        if value is None:
                            value = row.get("prediction")
                        out.append(_safe_float(value))
                    return out

                base_probs = probs(baseline)
                curr_probs = probs(current)
                baseline_conf_mean = float(np.mean(base_probs)) if base_probs else 0.0
                current_conf_mean = float(np.mean(curr_probs)) if curr_probs else 0.0
                confidence_drift = abs(current_conf_mean - baseline_conf_mean)
                baseline_class_rate = float(np.mean([1 if p >= 0.5 else 0 for p in base_probs])) if base_probs else 0.0
                current_class_rate = float(np.mean([1 if p >= 0.5 else 0 for p in curr_probs])) if curr_probs else 0.0
                class_distribution_drift = abs(current_class_rate - baseline_class_rate)
                report["baseline_confidence_mean"] = baseline_conf_mean
                report["current_confidence_mean"] = current_conf_mean
                report["confidence_drift"] = confidence_drift
                report["baseline_class_rate"] = baseline_class_rate
                report["current_class_rate"] = current_class_rate
                report["class_distribution_drift"] = class_distribution_drift
                if confidence_drift > self.confidence_mean_threshold:
                    report["alerts"].append(f"confidence_drift={confidence_drift:.4f}")
                if class_distribution_drift > self.class_distribution_threshold:
                    report["alerts"].append(f"class_distribution_drift={class_distribution_drift:.4f}")
                report["status"] = "alert" if report["alerts"] else report["status"]
            return report

    def write_daily_report(self, rows: List[Dict[str, Any]], *, report_date: str | None = None) -> Dict[str, Any]:
        report = self.evaluate(rows)
        report_date = report_date or datetime.now().strftime("%Y%m%d")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = REPORTS_DIR / f"prediction_drift_report_{report_date}.json"
        md_path = REPORTS_DIR / f"prediction_drift_report_{report_date}.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = [
            "# Prediction Drift Report",
            "",
            f"- Generated at: `{report['generated_at']}`",
            f"- Status: `{report['status']}`",
            f"- Alerts: `{len(report.get('alerts', []))}`",
            f"- Rolling Brier Score: `{report.get('rolling_brier_score')}`",
            "",
            "| Feature | PSI | KS p-value | Status |",
            "|---|---:|---:|---|",
        ]
        feature_metrics = report.get("feature_metrics") if isinstance(report.get("feature_metrics"), dict) else {}
        for feature_name in CORE_FEATURES:
            row = feature_metrics.get(feature_name, {})
            lines.append(
                f"| {feature_name} | {row.get('psi_score')} | {row.get('ks_p_value')} | {row.get('status', 'unknown')} |"
            )
        if report.get("alerts"):
            lines.extend(["", "## Alerts", ""])
            for alert in report["alerts"]:
                lines.append(f"- {alert}")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def run(
        self,
        *,
        stop_event: Optional[threading.Event] = None,
        poll_interval_sec: float = 1.0,
        db_refresh_limit: int = 500,
    ) -> None:
        logger.info(
            "Starting prediction drift worker | poll_interval=%.2fs live_window=%d trade_window=%d",
            float(poll_interval_sec),
            int(self.rolling_feature_window),
            int(self.completed_trade_window),
        )
        stop = stop_event or threading.Event()
        while not stop.is_set():
            try:
                self.refresh_from_prediction_db(limit=db_refresh_limit)
            except Exception:
                logger.exception("Prediction drift worker refresh failed")
            stop.wait(max(0.1, float(poll_interval_sec)))


def main() -> int:
    monitor = PredictionDriftMonitor()
    rows = monitor.load_prediction_rows()
    report = monitor.write_daily_report(rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
