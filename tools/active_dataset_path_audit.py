from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def build_report() -> dict:
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "active_dataset_path": "data/candles_*.json",
        "storage_format": "multiple JSON files, one file per trading day",
        "data_quality_invalid_logic": [
            "missing or malformed timestamp",
            "non-numeric OHLC",
            "zero or negative OHLC",
            "high < low",
            "high < max(open, close)",
            "low > min(open, close)",
        ],
        "post_market_retrain_loading": "scripts/post_market_retrain.py loads all data/candles_*.json via load_candles_from_data_dir(), converts rows with dict_to_candle(), sorts by timestamp, then resamples with candles[::10].",
        "train_window_benchmark_loading": "scripts/train_window_benchmark.py loads all data/candles_*.json via load_real_candles(), converts rows with dict_to_candle(), sorts by timestamp, then resamples with candles[::10].",
        "resampling_method": "Current retraining compatibility path uses simple downsampling: every 10th 1-minute candle via candles[::10], not OHLC aggregation.",
        "latest_incomplete_day_included": False,
        "provisional_candles_included": False,
        "latest_day_note": "Current dataset ends at 2026-06-02 with 375 candles from 09:15 to 15:29, so the final day is complete and non-provisional.",
        "missing_optional_files": [
            "scripts/download_mstock_history.py",
            "tools/merge_candle_history.py",
        ],
    }
    return report


def render_markdown(report: dict) -> str:
    return "\n".join(
        [
            "# Active Dataset Path Audit",
            "",
            f"- Active candle dataset path: `{report['active_dataset_path']}`",
            f"- Storage format: {report['storage_format']}",
            f"- Latest incomplete day included: `{report['latest_incomplete_day_included']}`",
            f"- Provisional candles included: `{report['provisional_candles_included']}`",
            "",
            "## Loading Flow",
            "",
            f"- `data_quality_report.py`: validates row-level OHLC integrity using {', '.join(report['data_quality_invalid_logic'])}.",
            f"- `post_market_retrain.py`: {report['post_market_retrain_loading']}",
            f"- `train_window_benchmark.py`: {report['train_window_benchmark_loading']}",
            f"- Resampling: {report['resampling_method']}",
            "",
            "## Final Day",
            "",
            f"- {report['latest_day_note']}",
            "",
            "## Missing Referenced Files",
            "",
            *[f"- `{item}`" for item in report["missing_optional_files"]],
        ]
    ) + "\n"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = REPORTS_DIR / "active_dataset_path_audit.json"
    md_path = REPORTS_DIR / "active_dataset_path_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
