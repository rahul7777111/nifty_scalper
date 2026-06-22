from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from black_scholes_features import enrich_option_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Black-Scholes enrichment for retraining datasets.")
    parser.add_argument("--input", required=True, help="Input CSV or parquet dataset.")
    parser.add_argument("--output", required=True, help="Output CSV or parquet path.")
    parser.add_argument("--risk-free-rate", type=float, default=0.065)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-output", type=int, default=5)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_reports(report_dir: Path, payload: dict) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"black_scholes_enrichment_{stamp}.json"
    md_path = report_dir / f"black_scholes_enrichment_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_lines = [
        "# Black-Scholes Enrichment Report",
        "",
        f"- Input rows: `{payload['total_rows']}`",
        f"- Row errors: `{payload['row_error_count']}`",
        f"- IV solve successes: `{payload['solve_ok_count']}`",
        f"- Existing IV reused: `{payload['final_iv_from_existing_count']}`",
        f"- Valid bid/ask rows: `{payload['rows_with_valid_bid_ask']}`",
        "",
        "## Notes",
    ]
    for note in payload.get("notes", []):
        md_lines.append(f"- {note}")
    md_lines.extend(
        [
            "",
            "## Columns Used",
        ]
    )
    for key, value in payload.get("columns_used", {}).items():
        md_lines.append(f"- `{key}`: `{value}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = load_dataset(input_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    enriched, metadata = enrich_option_frame(
        df,
        risk_free_rate=float(args.risk_free_rate),
        dividend_yield=float(args.dividend_yield),
        timezone_name=str(args.timezone),
        symbol=str(args.symbol),
    )
    save_dataset(enriched, output_path)
    report_paths = write_reports(REPO_ROOT / "reports", metadata)
    sample_columns = [
        column for column in [
            "bs_iv",
            "final_iv",
            "bs_delta",
            "bs_gamma",
            "bs_theta",
            "bs_vega",
            "bs_rho",
            "moneyness",
            "time_to_expiry_days",
            "greeks_quality_score",
            "row_enrichment_ok",
            "row_enrichment_error",
        ] if column in enriched.columns
    ]
    payload = {
        **metadata,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "report_paths": report_paths,
        "sample_output": enriched[sample_columns].head(max(0, int(args.sample_output))).to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
