from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import argparse
import sys


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%d %b %y",
    "%d-%m-%y",
    "%d/%m/%y",
)


def _normalize_opt_type(value: str) -> str:
    s = str(value or "").strip().upper()
    if s in {"CE", "CALL"}:
        return "CE"
    if s in {"PE", "PUT"}:
        return "PE"
    if s == "C":
        return "CE"
    if s == "P":
        return "PE"
    return s


def _parse_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    s = str(value).strip()
    if not s:
        return None

    # Try ISO first (handles YYYY-MM-DD and also full timestamps).
    try:
        return datetime.fromisoformat(s.replace("Z", "")).date()
    except Exception:
        pass

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value

    s = str(value).strip()
    if not s:
        return None

    try:
        # Handle "19750.0" and similar.
        return int(float(s))
    except Exception:
        return None


def _get_any(row: Dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _derive_symbol_root(tradingsymbol: str) -> str:
    ts = str(tradingsymbol or "").strip().upper()
    if not ts:
        return ""
    # Common index roots.
    for root in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"):
        if ts.startswith(root):
            return root
    # Fallback: leading alpha prefix.
    out = []
    for ch in ts:
        if "A" <= ch <= "Z":
            out.append(ch)
            continue
        break
    return "".join(out)


def _normalize_root_alias(s: str) -> str:
    """Map common underlying synonyms to their standard option ticker roots."""
    val = str(s or "").strip().upper().replace(" ", "")
    # Standardize NIFTY
    if val in {"NIFTY50", "NIFTY-50", "CNXNIFTY", "NIFTYINDEX"}:
        return "NIFTY"
    # Standardize BANKNIFTY
    if val in {"NIFTYBANK", "BANK-NIFTY", "NSEBANK", "BANKNIFTYINDEX"}:
        return "BANKNIFTY"
    # Standardize FINNIFTY
    if val in {"NIFTYFINSERVICE", "FIN-NIFTY", "FINNIFTYINDEX"}:
        return "FINNIFTY"
    # Standardize MIDCPNIFTY
    if val in {"NIFTYMIDSELECT", "MIDCP-NIFTY", "MIDCPNIFTYINDEX"}:
        return "MIDCPNIFTY"
    return val



@dataclass(frozen=True)
class ScripMasterRow:
    exch: str
    exch_type: str
    token: str
    symbol_root: str
    expiry: Optional[date]
    strike: Optional[int]
    opt_type: str  # "CE" | "PE" | ""
    lot_size: Optional[int]
    tradingsymbol: str
    raw: Dict[str, Any]


class ScripMaster:
    """Lightweight ScripMaster CSV reader (stdlib-only).

    This is meant to be broker-agnostic and fast. It loads the CSV once and
    keeps the parsed rows in memory.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = str(csv_path or "").strip()
        self._rows: Optional[List[ScripMasterRow]] = None
        self._ts_index: Optional[Dict[tuple[str, str], str]] = None

    def _load(self) -> List[ScripMasterRow]:
        if self._rows is not None:
            return self._rows

        p = Path(self.csv_path)
        if not self.csv_path or not p.exists() or not p.is_file():
            self._rows = []
            return self._rows

        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows: List[ScripMasterRow] = []

            for raw_row in reader:
                if not isinstance(raw_row, dict):
                    continue

                # Support both broker "ScripMaster" CSVs and common "instrument master" CSVs.
                exch = _get_any(raw_row, ("Exch", "Exchange", "exchange", "exch"))
                exch_type = _get_any(raw_row, ("ExchType", "Segment", "segment", "ExchSeg", "exch_type"))
                token = _get_any(
                    raw_row,
                    (
                        "ScripCode",
                        "Token",
                        "instrument_token",
                        "instrumentToken",
                        "symbolToken",
                        "symboltoken",
                    ),
                )
                symbol_root = _get_any(raw_row, ("SymbolRoot", "RootSymbol", "Underlying", "symbol_root"))
                expiry = _parse_date(_get_any(raw_row, ("Expiry", "expiry", "Exp", "exp", "ExpDate", "expiry_date")))
                strike = _parse_int(_get_any(raw_row, ("StrikeRate", "strike", "Strike", "strike_price")))
                opt_type = _get_any(
                    raw_row,
                    (
                        "ScripType",
                        "instrument_type",
                        "InstrumentType",
                        "OptionType",
                        "option_type",
                        "opt_type",
                        "CPType",
                    ),
                ).upper()
                lot_size = _parse_int(_get_any(raw_row, ("LotSize", "lot_size", "Lot", "lotsize")))
                tradingsymbol = _get_any(
                    raw_row,
                    (
                        "TradingSymbol",
                        "tradingsymbol",
                        "ScripName",
                        "Symbol",
                        "symbol",
                        "InstrumentName",
                        "name",
                    ),
                )

                if not symbol_root:
                    symbol_root = _derive_symbol_root(tradingsymbol)
                
                symbol_root = _normalize_root_alias(symbol_root)

                token_s = str(token).strip()
                if not token_s.isdigit():
                    continue

                rows.append(
                    ScripMasterRow(
                        exch=str(exch).strip().upper(),
                        exch_type=str(exch_type).strip().upper(),
                        token=token_s,
                        symbol_root=str(symbol_root).strip().upper(),
                        expiry=expiry,
                        strike=strike,
                        opt_type=_normalize_opt_type(str(opt_type or "")),
                        lot_size=lot_size,
                        tradingsymbol=str(tradingsymbol).strip(),
                        raw=dict(raw_row),
                    )
                )

        self._rows = rows
        return rows

    def _build_tradingsymbol_index(self) -> Dict[tuple[str, str], str]:
        if self._ts_index is not None:
            return self._ts_index

        idx: Dict[tuple[str, str], str] = {}
        for r in self._load():
            ts_key = str(r.tradingsymbol or "").strip().upper()
            if not ts_key:
                continue
            exch = str(r.exch or "").strip().upper()
            if exch:
                idx.setdefault((exch, ts_key), r.token)
            idx.setdefault(("", ts_key), r.token)

        self._ts_index = idx
        return idx

    def token_for_tradingsymbol(self, tradingsymbol: str, *, exch: Optional[str] = None) -> Optional[str]:
        ts = str(tradingsymbol or "").strip().upper()
        if not ts:
            return None

        idx = self._build_tradingsymbol_index()
        ex = str(exch or "").strip().upper()

        if ex:
            tok = idx.get((ex, ts))
            if tok:
                return tok
        return idx.get(("", ts))

    def option_rows(
        self,
        *,
        symbol_root: str,
        exch: str = "NFO",
        min_expiry: Optional[date] = None,
    ) -> List[ScripMasterRow]:
        target_root = _normalize_root_alias(symbol_root)
        min_e = min_expiry
        out: List[ScripMasterRow] = []
        for r in self._load():
            if r.exch and str(r.exch).strip().upper() != str(exch).strip().upper():
                continue
            if r.symbol_root != target_root:
                continue
            if r.opt_type not in {"CE", "PE"}:
                continue
            if r.expiry is None:
                continue
            if min_e is not None and r.expiry < min_e:
                continue
            if r.strike is None:
                continue
            out.append(r)

        return out

    def future_rows(
        self,
        *,
        symbol_root: str,
        exch: str = "NFO",
        min_expiry: Optional[date] = None,
    ) -> List[ScripMasterRow]:
        """Return candidate futures rows for an underlying root.

        We rely on loose heuristics because different broker ScripMaster CSVs
        use different instrument type codes (e.g. FUTIDX/FUTSTK/FUT).
        """

        target_root = _normalize_root_alias(symbol_root)
        min_e = min_expiry
        out: List[ScripMasterRow] = []

        for r in self._load():
            if r.exch and str(r.exch).strip().upper() != str(exch).strip().upper():
                continue
            if _normalize_root_alias(r.symbol_root) != target_root:
                continue
            if r.expiry is None:
                continue
            if min_e is not None and r.expiry < min_e:
                continue

            # Exclude options.
            if str(r.opt_type or "").strip().upper() in {"CE", "PE"}:
                continue

            ts = str(r.tradingsymbol or "").strip().upper()
            # Prefer explicit FUT contracts.
            if ts and "FUT" in ts:
                out.append(r)
                continue

            # Fallback: many futures rows still have strike=None and no CE/PE.
            if r.strike is None:
                out.append(r)

        return out

    def nearest_future_tradingsymbol(
        self,
        symbol_root: str,
        *,
        exch: str = "NFO",
        asof: Optional[date] = None,
    ) -> Optional[str]:
        """Return the nearest-expiry future trading symbol for a root."""

        if asof is None:
            asof = date.today()
        rows = self.future_rows(symbol_root=symbol_root, exch=exch, min_expiry=asof)
        if not rows:
            return None

        # Nearest expiry; prefer rows whose symbol contains 'FUT' when tied.
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r.expiry or date.max,
                0 if (str(r.tradingsymbol or "").strip().upper().endswith("FUT") or "FUT" in str(r.tradingsymbol or "").strip().upper()) else 1,
            ),
        )
        ts = str(rows_sorted[0].tradingsymbol or "").strip()
        return ts or None

    def get_available_expiries(self, symbol_root: str, exch: str = "NFO") -> List[date]:
        root = _normalize_root_alias(symbol_root)
        expiries = set()
        for r in self._load():
            if r.exch and str(r.exch).strip().upper() != str(exch).strip().upper():
                continue
            if _normalize_root_alias(r.symbol_root) == root:
                if r.expiry:
                    expiries.add(r.expiry)
        return sorted(list(expiries))

    def has_data(self) -> bool:
        return bool(self._load())


def _cli() -> int:
    p = argparse.ArgumentParser(description="Sanity-check ScripMaster CSV for option token mapping")
    p.add_argument("csv", help="Path to ScripMaster CSV")
    p.add_argument("symbol", nargs="?", default="NIFTY", help="Symbol root (default: NIFTY)")
    p.add_argument(
        "--exch",
        default="NFO",
        help="Exchange filter for options (default: NFO)",
    )
    p.add_argument(
        "--show",
        type=int,
        default=3,
        help="How many nearest expiries to print (default: 3)",
    )
    args = p.parse_args()

    sm = ScripMaster(args.csv)
    rows = sm.option_rows(symbol_root=str(args.symbol), exch=str(args.exch), min_expiry=date.today())
    if not rows:
        print("No option rows found. Check CSV path/columns and filters.")
        return 2

    # Summarize by expiry.
    by_expiry: Dict[date, List[ScripMasterRow]] = {}
    for r in rows:
        if r.expiry is None:
            continue
        by_expiry.setdefault(r.expiry, []).append(r)

    expiries = sorted(by_expiry.keys())
    show = max(1, int(args.show))
    print(f"Loaded {len(rows)} option rows for {str(args.symbol).upper()} ({str(args.exch).upper()})")
    print(f"Upcoming expiries found: {len(expiries)}")

    for exp in expiries[:show]:
        e_rows = by_expiry.get(exp, [])
        ce = [r for r in e_rows if r.opt_type == "CE"]
        pe = [r for r in e_rows if r.opt_type == "PE"]
        strikes = sorted({r.strike for r in e_rows if r.strike is not None})
        strike_min = strikes[0] if strikes else None
        strike_max = strikes[-1] if strikes else None
        print(
            f"Expiry {exp.isoformat()} | total={len(e_rows)} CE={len(ce)} PE={len(pe)} "
            f"strike_range={strike_min}-{strike_max}"
        )

    # Basic pass criteria: at least one expiry with both CE & PE.
    ok = any(
        any(r.opt_type == "CE" for r in by_expiry[exp]) and any(r.opt_type == "PE" for r in by_expiry[exp])
        for exp in expiries
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
