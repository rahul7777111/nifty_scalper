from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_api_config  # noqa: E402
from logger_setup import setup_logging  # noqa: E402
from options_archive.collector import OptionsArchiveCollector  # noqa: E402
from options_archive.manager import MultiIndexArchiveManager  # noqa: E402
from options_archive.models import ArchiveConfig  # noqa: E402
from options_archive.symbols import archive_config_for_symbol, iter_supported_symbols, normalize_symbol_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a point-in-time options archive for NiftyScalper.")
    parser.add_argument("--archive-root", default=os.getenv("OPTIONS_ARCHIVE_ROOT", "archive/options"))
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbol keys to collect (default: NIFTY BANKNIFTY).")
    parser.add_argument("--run-once", action="store_true", help="Collect a single snapshot immediately if due.")
    parser.add_argument("--force", action="store_true", help="Force a snapshot even if the scheduler says no.")
    parser.add_argument("--daemon", action="store_true", help="Run the collector loop forever.")
    parser.add_argument("--qa-date", default="", help="Generate a QA report for YYYY-MM-DD.")
    parser.add_argument("--feature-rebuild", action="store_true", help="Rebuild feature partitions from clean snapshots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), log_dir="logs", app_name="options_archive")
    api_cfg = load_api_config()
    symbol_keys = args.symbols or [s.symbol_key for s in iter_supported_symbols(["NIFTY", "BANKNIFTY"])]
    collectors = []
    for symbol_key in symbol_keys:
        spec = normalize_symbol_key(symbol_key)
        archive_cfg = archive_config_for_symbol(spec, archive_root=Path(args.archive_root))
        collectors.append(OptionsArchiveCollector(api_cfg, archive_cfg))
    manager = MultiIndexArchiveManager(collectors)

    if args.feature_rebuild:
        manager.rebuild_feature_layers()
        return 0

    if args.qa_date:
        session_date = date.fromisoformat(args.qa_date)
        manager.run_daily_qa(session_date=session_date)
        return 0

    if args.daemon:
        manager.run_forever()
        return 0

    if args.run_once:
        results = manager.run_once_all(force=args.force)
        if not results:
            print("No snapshots due yet.")
        else:
            for result in results:
                print(f"Snapshot {result.snapshot_id} saved: raw={result.raw_path} clean={result.clean_path}")
        return 0

    parser = argparse.ArgumentParser()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())