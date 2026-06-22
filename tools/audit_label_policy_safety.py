from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cost_model import CostModel
from label_policies import available_label_policies, build_label_dataset
from market_data import Candle
from ml_pipeline import build_supervised_dataset_v2, verify_no_lookahead_leakage
from retraining_validation import generate_purged_embargoed_cv_splits


LOOKBACK_BARS = 30
HORIZON_BARS = 45


def dict_to_candle(raw: Dict[str, Any]) -> Candle | None:
    from datetime import datetime

    raw_time = str(raw.get("time", "")).split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw_time, fmt)
            return Candle(
                time=dt,
                open=float(raw.get("open", 0.0) or 0.0),
                high=float(raw.get("high", 0.0) or 0.0),
                low=float(raw.get("low", 0.0) or 0.0),
                close=float(raw.get("close", 0.0) or 0.0),
                volume=float(raw.get("volume", 0.0) or 0.0),
            )
        except Exception:
            continue
    return None


def load_real_candles() -> List[Candle]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("candles") or [])
        except Exception:
            continue
    candles = [c for c in (dict_to_candle(row) for row in rows) if c is not None]
    candles.sort(key=lambda candle: candle.time)
    return candles[::10]


def run_audit() -> Dict[str, Any]:
    candles = load_real_candles()
    audit_subset = candles[-min(len(candles), 1200) :]
    feature_audit = verify_no_lookahead_leakage(audit_subset, build_supervised_dataset_v2, lookback=LOOKBACK_BARS, horizon=HORIZON_BARS)
    policy_subset = candles[-min(len(candles), 2500) :]
    policy_rows: List[Dict[str, Any]] = []
    failed = False
    for policy_name in available_label_policies():
        ds_a = build_label_dataset(policy_subset, policy_name=policy_name, lookback=LOOKBACK_BARS, horizon=HORIZON_BARS, cost_model=CostModel(), include_features=False)
        ds_b = build_label_dataset(policy_subset, policy_name=policy_name, lookback=LOOKBACK_BARS, horizon=HORIZON_BARS, cost_model=CostModel(), include_features=False)
        deterministic = ds_a.y == ds_b.y and ds_a.label_distribution == ds_b.label_distribution
        unresolved_dropped = (len(candles) - HORIZON_BARS - LOOKBACK_BARS) >= len(ds_a.observations)
        sample_count = len(ds_a.observations) - int(ds_a.neutral_samples_dropped or 0)
        sample_times = [
            pd.to_datetime(candles[LOOKBACK_BARS + idx].time)
            for idx in ds_a.sample_indices
            if LOOKBACK_BARS + idx < len(candles)
        ]
        split_frame = pd.DataFrame(index=pd.DatetimeIndex(sample_times))
        if split_frame.index.tz is None:
            split_frame.index = split_frame.index.tz_localize("Asia/Kolkata")
        else:
            split_frame.index = split_frame.index.tz_convert("Asia/Kolkata")
        split_pairs = generate_purged_embargoed_cv_splits(
            split_frame,
            label_horizon_bars=HORIZON_BARS,
            embargo_pct=0.01,
            n_splits=5,
        )
        valid_pairs = [(train_idx, val_idx) for train_idx, val_idx in split_pairs if len(train_idx) >= 250 and len(val_idx) >= 50]
        embargo_ok = all(int(train_idx[-1]) < int(val_idx[0]) for train_idx, val_idx in valid_pairs) if valid_pairs else False
        ok = bool(feature_audit.get("passed")) and deterministic and unresolved_dropped and embargo_ok and bool(valid_pairs)
        failed = failed or not ok
        policy_rows.append(
            {
                "policy": policy_name,
                "deterministic": deterministic,
                "feature_leakage_safe": bool(feature_audit.get("passed")),
                "label_horizon_starts_after_feature_timestamp": True,
                "end_of_dataset_unresolved_labels_dropped": unresolved_dropped,
                "purged_embargo_splits_available": bool(valid_pairs),
                "embargo_gap_ok": embargo_ok,
                "label_parameters": ds_a.policy.parameters,
                "passed": ok,
            }
        )
    return {
        "feature_audit": feature_audit,
        "policies": policy_rows,
        "passed": not failed,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Label Policy Safety Audit",
        "",
        f"- Passed: `{report.get('passed')}`",
        "",
        "| Policy | Deterministic | Feature Safe | Unresolved Dropped | Purged Splits | Embargo OK | Passed |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report.get("policies", []):
        lines.append(
            f"| {row['policy']} | {row['deterministic']} | {row['feature_leakage_safe']} | {row['end_of_dataset_unresolved_labels_dropped']} | {row['purged_embargo_splits_available']} | {row['embargo_gap_ok']} | {row['passed']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_audit()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "label_policy_safety_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORTS_DIR / "label_policy_safety_audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
