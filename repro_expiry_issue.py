
import sys
import os
from datetime import date, datetime
import re

# Mocking the strategy methods to test logic in isolation
class MockStrategy:
    def __init__(self):
        self.cfg = type('Config', (), {'symbol': 'NIFTY'})()

    def _parse_any_date(self, raw: object) -> date | None:
        s = str(raw or "").strip()
        if not s:
            return None

        # Normalize common ISO variants.
        iso = s.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso)
            return parsed.date()
        except Exception:
            pass

        # Common user-input formats.
        for fmt in (
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%d-%B-%Y",
            "%d %b %Y",
            "%d %B %Y",
            "%Y%m%d",
        ):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue

        # Accept DDMMMYY / DDMMMYYYY (e.g. 06FEB26, 06FEB2026).
        up = s.upper().replace("-", "").replace("/", "").replace(" ", "")
        m = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2}|\d{4})", up)
        if not m:
            return None

        dd_s, mon_s, yy_s = m.group(1), m.group(2), m.group(3)
        mon_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        mm = mon_map.get(mon_s)
        if not mm:
            return None

        try:
            dd = int(dd_s)
            if len(yy_s) == 2:
                year = 2000 + int(yy_s)
            else:
                year = int(yy_s)
            return date(year, mm, dd)
        except Exception:
            return None

    def _parse_expiry_from_option_symbol(self, symbol: str) -> date | None:
        s = str(symbol or "").upper().strip()
        if not s:
            return None

        # Support both 2-digit (YY) and 4-digit (20YY) years.
        # Use (20\d{2}|\d{2}) to ensure we don't greedily eat into the strike price
        # if the year is 2 digits but followed by more digits (e.g. JAN2640000 -> 26, not 2640).
        m = re.search(r"(\d{2})([A-Z]{3})(20\d{2}|\d{2})", s)
        if not m:
            return None

        dd_s, mon_s, yy_s = m.group(1), m.group(2), m.group(3)
        try:
            dd = int(dd_s)
            if len(yy_s) == 4:
                year = int(yy_s)
            else:
                year = 2000 + int(yy_s)
        except Exception:
            return None

        mon_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        mm = mon_map.get(mon_s)
        if not mm:
            return None

        try:
            return date(year, mm, dd)
        except Exception:
            return None

    def _row_expiry_date(self, row: dict) -> date | None:
        exp_raw = row.get("expiry")
        if isinstance(exp_raw, datetime):
            return exp_raw.date()
        if isinstance(exp_raw, date):
            return exp_raw
        if exp_raw is not None:
            exp = self._parse_any_date(exp_raw)
            if exp is not None:
                return exp
        sym_raw = str(row.get("symbol") or "").upper()
        sym = sym_raw.split(":", 1)[-1]
        return self._parse_expiry_from_option_symbol(sym)

def test_expiry_parsing():
    strat = MockStrategy()
    
    print("--- Testing _parse_any_date ---")
    inputs = [
        ("2026-01-09", date(2026, 1, 9)),
        ("09-01-2026", date(2026, 1, 9)),
        ("09/01/2026", date(2026, 1, 9)),
        ("09JAN26", date(2026, 1, 9)),
        ("09-Jan-2026", date(2026, 1, 9)),
        ("9-Jan-2026", None), # Current logic might assume 2 digit day? Let's check regex/strptime.
                             # %d matches 01-31, but python strptime often allows single digit if not followed by digit.
                             # But here regex DDMMMYY expects 2 digits.
    ]
    
    for inp, expected in inputs:
        res = strat._parse_any_date(inp)
        print(f"Input: {inp!r} -> Parsed: {res} | Match: {res == expected}")

    print("\n--- Testing _row_expiry_date with mocked rows ---")
    rows = [
        ({"expiry": date(2026, 1, 9)}, date(2026, 1, 9)),
        ({"expiry": "2026-01-09"}, date(2026, 1, 9)),
        ({"symbol": "NIFTY09JAN2618000CE"}, date(2026, 1, 9)),
        ({"symbol": "NIFTY09JAN202618000CE"}, date(2026, 1, 9)), # Test 4-digit year
        ({"symbol": "NIFTY26JAN0918000CE"}, date(2026, 1, 26)), # Wrong order but valid date? No, 26JAN09 -> 2009.
        ({"symbol": "BANKNIFTY15FEB2640000PE"}, date(2026, 2, 15)),
    ]

    for row, expected in rows:
        res = strat._row_expiry_date(row)
        print(f"Row: {row} -> Parsed: {res} | Match: {res == expected}")

if __name__ == "__main__":
    test_expiry_parsing()
