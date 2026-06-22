from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DB_PATH = REPO_ROOT / "trades.db"


@dataclass
class FeatureDriftAlert:
    feature: str
    psi: float
    kl_divergence: float
    mean_shift: float
    variance_shift: float
    status: str


class FeatureDriftMonitor:
    def __init__(
        self,
        *,
        psi_threshold: float = 0.25,
        kl_threshold: float = 0.10,
        mean_shift_threshold: float = 0.75,
        variance_shift_threshold: float = 1.50,
    ) -> None:
        self.psi_threshold = float(psi_threshold)
        self.kl_threshold = float(kl_threshold)
        self.mean_shift_threshold = float(mean_shift_threshold)
        self.variance_shift_threshold = float(variance_shift_threshold)

    @staticmethod
    def _to_array(values: Sequence[float]) -> np.ndarray:
        return np.asarray([float(v) for v in values], dtype=float)

    @staticmethod
    def compute_psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
        exp = FeatureDriftMonitor._to_array(expected)
        act = FeatureDriftMonitor._to_array(actual)
        if len(exp) < 2 or len(act) < 2:
            return 0.0
        bin_edges = np.quantile(exp, np.linspace(0.0, 1.0, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        exp_counts, _ = np.histogram(exp, bins=bin_edges)
        act_counts, _ = np.histogram(act, bins=bin_edges)
        exp_pct = (exp_counts + 0.5) / (len(exp) + 0.5 * bins)
        act_pct = (act_counts + 0.5) / (len(act) + 0.5 * bins)
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    @staticmethod
    def compute_kl_divergence(expected: Sequence[float], actual: Sequence[float], bins: int = 20) -> float:
        exp = FeatureDriftMonitor._to_array(expected)
        act = FeatureDriftMonitor._to_array(actual)
        if len(exp) < 2 or len(act) < 2:
            return 0.0
        low = float(min(np.min(exp), np.min(act)))
        high = float(max(np.max(exp), np.max(act)))
        if math.isclose(low, high):
            return 0.0
        exp_counts, edges = np.histogram(exp, bins=bins, range=(low, high))
        act_counts, _ = np.histogram(act, bins=edges)
        exp_pct = (exp_counts + 0.5) / (len(exp) + 0.5 * bins)
        act_pct = (act_counts + 0.5) / (len(act) + 0.5 * bins)
        return float(np.sum(act_pct * np.log(act_pct / exp_pct)))

    @staticmethod
    def compute_mean_shift(expected: Sequence[float], actual: Sequence[float]) -> float:
        exp = FeatureDriftMonitor._to_array(expected)
        act = FeatureDriftMonitor._to_array(actual)
        exp_std = float(np.std(exp))
        if exp_std <= 1e-9:
            return 0.0
        return float(abs(np.mean(act) - np.mean(exp)) / exp_std)

    @staticmethod
    def compute_variance_shift(expected: Sequence[float], actual: Sequence[float]) -> float:
        exp = FeatureDriftMonitor._to_array(expected)
        act = FeatureDriftMonitor._to_array(actual)
        exp_var = float(np.var(exp))
        if exp_var <= 1e-9:
            return 0.0
        return float(np.var(act) / exp_var)

    def evaluate(self, training_distribution: Dict[str, Sequence[float]], live_distribution: Dict[str, Sequence[float]]) -> Dict[str, Any]:
        feature_rows: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []
        overlapping = sorted(set(training_distribution).intersection(live_distribution))
        for feature_name in overlapping:
            expected = training_distribution[feature_name]
            actual = live_distribution[feature_name]
            if len(expected) < 2 or len(actual) < 2:
                continue
            psi = self.compute_psi(expected, actual)
            kl_div = self.compute_kl_divergence(expected, actual)
            mean_shift = self.compute_mean_shift(expected, actual)
            variance_shift = self.compute_variance_shift(expected, actual)
            status = "stable"
            if (
                psi > self.psi_threshold
                or kl_div > self.kl_threshold
                or mean_shift > self.mean_shift_threshold
                or variance_shift > self.variance_shift_threshold
            ):
                status = "alert"
                alerts.append(
                    {
                        "feature": feature_name,
                        "psi": psi,
                        "kl_divergence": kl_div,
                        "mean_shift": mean_shift,
                        "variance_shift": variance_shift,
                    }
                )
            feature_rows.append(
                {
                    "feature": feature_name,
                    "psi": psi,
                    "kl_divergence": kl_div,
                    "mean_shift": mean_shift,
                    "variance_shift": variance_shift,
                    "status": status,
                    "live_count": len(actual),
                }
            )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "feature_count": len(feature_rows),
            "alerts": alerts,
            "features": feature_rows,
        }

    @staticmethod
    def load_training_distribution(path: Path | None = None) -> Dict[str, List[float]]:
        path = path or (REPORTS_DIR / "training_feature_distribution.json")
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, List[float]] = {}
        for feature_name, stats in (payload.get("features") or {}).items():
            quantiles = stats.get("quantiles") or []
            if quantiles:
                out[feature_name] = [float(v) for v in quantiles]
        return out

    @staticmethod
    def load_live_distribution_from_db(limit: int = 1000) -> Dict[str, List[float]]:
        if not DB_PATH.exists():
            return {}
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT features_json
            FROM predictions
            WHERE COALESCE(source, '') = 'ACTIVE'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        conn.close()
        dist: Dict[str, List[float]] = {}
        for row in rows:
            try:
                features = json.loads(row["features_json"] or "{}")
            except Exception:
                features = {}
            for key, value in features.items():
                try:
                    dist.setdefault(str(key), []).append(float(value))
                except Exception:
                    continue
        return dist

    def write_report(
        self,
        training_distribution: Dict[str, Sequence[float]],
        live_distribution: Dict[str, Sequence[float]],
        *,
        json_path: Path | None = None,
        md_path: Path | None = None,
    ) -> Dict[str, Any]:
        report = self.evaluate(training_distribution, live_distribution)
        json_path = json_path or (REPORTS_DIR / "feature_drift_report.json")
        md_path = md_path or (REPORTS_DIR / "feature_drift_report.md")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = [
            "# Feature Drift Report",
            "",
            f"- Generated at: `{report['generated_at']}`",
            f"- Features evaluated: `{report['feature_count']}`",
            f"- Alerts: `{len(report['alerts'])}`",
            "",
            "| Feature | PSI | KL | Mean Shift | Variance Shift | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in report["features"][:30]:
            lines.append(
                f"| {row['feature']} | {row['psi']:.4f} | {row['kl_divergence']:.4f} | {row['mean_shift']:.4f} | {row['variance_shift']:.4f} | {row['status']} |"
            )
        if not report["features"]:
            lines.append("| none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | insufficient_live_data |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report


def main() -> int:
    monitor = FeatureDriftMonitor()
    training = monitor.load_training_distribution()
    live = monitor.load_live_distribution_from_db()
    report = monitor.write_report(training, live)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
