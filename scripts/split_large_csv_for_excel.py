from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


EXCEL_MAX_ROWS = 1_048_576
DEFAULT_ROWS_PER_FILE = EXCEL_MAX_ROWS - 1


def split_csv(path: Path, output_dir: Path, rows_per_file: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = path.stem
    parts: list[dict[str, object]] = []
    part_index = 0
    data_rows_in_part = 0
    total_data_rows = 0
    current_out = None
    writer = None
    current_path: Path | None = None

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "source": str(path),
                "source_size_bytes": path.stat().st_size,
                "total_data_rows": 0,
                "parts": [],
            }

        def open_part() -> None:
            nonlocal part_index, data_rows_in_part, current_out, writer, current_path
            if current_out is not None:
                current_out.close()
            part_index += 1
            data_rows_in_part = 0
            current_path = output_dir / f"{base}_excel_part_{part_index:03d}.csv"
            current_out = current_path.open("w", encoding="utf-8-sig", newline="")
            writer = csv.writer(current_out, lineterminator="\n")
            writer.writerow(header)

        open_part()

        for row in reader:
            if data_rows_in_part >= rows_per_file:
                assert current_path is not None
                parts.append(
                    {
                        "file": str(current_path),
                        "data_rows": data_rows_in_part,
                        "excel_rows_including_header": data_rows_in_part + 1,
                    }
                )
                open_part()

            assert writer is not None
            writer.writerow(row)
            data_rows_in_part += 1
            total_data_rows += 1

    if current_out is not None:
        current_out.close()
    if current_path is not None:
        parts.append(
            {
                "file": str(current_path),
                "data_rows": data_rows_in_part,
                "excel_rows_including_header": data_rows_in_part + 1,
            }
        )

    return {
        "source": str(path),
        "source_size_bytes": path.stat().st_size,
        "total_data_rows": total_data_rows,
        "parts": parts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split oversized CSV files into UTF-8 BOM CSV chunks Excel can open."
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rows-per-file", type=int, default=DEFAULT_ROWS_PER_FILE)
    args = parser.parse_args()

    if args.rows_per_file < 1 or args.rows_per_file > DEFAULT_ROWS_PER_FILE:
        raise SystemExit(f"--rows-per-file must be between 1 and {DEFAULT_ROWS_PER_FILE}")

    folder = args.folder.resolve()
    output_dir = (args.output_dir or folder / "excel_openable").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for csv_path in sorted(folder.glob(args.pattern)):
        if not csv_path.is_file() or output_dir in csv_path.parents:
            continue
        print(f"Splitting {csv_path.name} ...", flush=True)
        result = split_csv(csv_path, output_dir / csv_path.stem, args.rows_per_file)
        results.append(result)
        print(
            f"  wrote {len(result['parts'])} part(s), {result['total_data_rows']} data rows",
            flush=True,
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "excel_max_rows": EXCEL_MAX_ROWS,
        "rows_per_file": args.rows_per_file,
        "source_folder": str(folder),
        "output_folder": str(output_dir),
        "files": results,
    }
    manifest_path = output_dir / "excel_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
