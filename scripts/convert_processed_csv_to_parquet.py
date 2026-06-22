#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(csv_path: str, output_path: str | None = None, compression: str = "snappy") -> Path:
    src = Path(csv_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() != ".csv":
        raise ValueError(f"Expected .csv input, got {src}")
    dst = Path(output_path).expanduser().resolve() if output_path else src.with_suffix(".parquet")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src, low_memory=False)
    df.to_parquet(dst, index=False, compression=compression)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a processed historical CSV to Parquet for faster backtests.")
    parser.add_argument("csv", help="Processed CSV path")
    parser.add_argument("--output", "-o", default=None, help="Output .parquet path; defaults to input path with .parquet suffix")
    parser.add_argument("--compression", default="snappy", help="Parquet compression codec")
    args = parser.parse_args()
    out = convert_csv_to_parquet(args.csv, args.output, args.compression)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
