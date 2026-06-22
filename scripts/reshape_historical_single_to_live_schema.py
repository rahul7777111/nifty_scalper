#!/usr/bin/env python
"""Rewrite historical unified option CSVs to match a reference CSV schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def iter_source_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.glob("*.csv")
        if path.name.startswith("nifty_option_chain_historical_cost_aware_")
    )


def reshape_file(path: Path, target_columns: list[str], chunk_size: int) -> None:
    source_columns = read_header(path)
    if source_columns == target_columns:
        print(f"Skipping {path.name}; already matches reference schema", flush=True)
        return
    missing = [col for col in target_columns if col not in source_columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    first_chunk = True
    for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=target_columns, low_memory=False):
        chunk = chunk[target_columns]
        chunk.to_csv(tmp_path, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False

    if first_chunk:
        pd.DataFrame(columns=target_columns).to_csv(tmp_path, index=False)

    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/historical_unified_nifty_options_single",
        help="Directory containing historical unified CSV files.",
    )
    parser.add_argument(
        "--reference-file",
        default="data/processed/nifty_option_chain_live_feature_reconstructed_20260604_100331_bs_enriched.csv",
        help="Reference CSV whose column order should be matched.",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of filenames to reshape inside input-dir.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    reference_file = Path(args.reference_file)
    target_columns = read_header(reference_file)
    source_files = iter_source_files(input_dir)
    if args.only:
        wanted = set(args.only)
        source_files = [path for path in source_files if path.name in wanted]

    if not source_files:
        raise FileNotFoundError(f"No historical cost-aware CSV files found in {input_dir}")

    for file_path in source_files:
        print(f"Reshaping {file_path.name} -> {len(target_columns)} columns", flush=True)
        reshape_file(file_path, target_columns, args.chunk_size)


if __name__ == "__main__":
    main()
