import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_iso_timestamp_formats_as_time_only():
    assert ScalperUI._pf_format_last_tick("2026-06-11T13:17:57.431702") == "13:17:57"


def test_datetime_formats_as_time_only():
    assert ScalperUI._pf_format_last_tick(datetime(2026, 6, 11, 13, 17, 57)) == "13:17:57"


def test_invalid_timestamp_formats_na():
    assert ScalperUI._pf_format_last_tick("-11T13:17:57.431702") == "n/a"
    assert ScalperUI._pf_format_last_tick(None) == "n/a"
