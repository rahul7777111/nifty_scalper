from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import argparse
import sys
import time

try:
    from pf_logging import log_scripmaster_cache
except ImportError:
    def log_scripmaster_cache(**kwargs):  # type: ignore[misc]
        pass

_SCRIPMASTER_INSTANCE_CACHE: Dict[str, "ScripMaster"] = {}


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
    val = str(s or "").strip().upper().replace(" ", "").replace("-", "").replace(":", "")
    # Standardize NIFTY variants including NIFTY-I , NSE:NIFTY etc.
    if val in {"NIFTY50", "NIFTY50", "NIFTY", "NIFTYINDEX", "CNXNIFTY", "NIFTYI", "NIFTYIND", "NSE NIFTY"}:
        return "NIFTY"
    # Standardize BANKNIFTY
    if val in {"NIFTYBANK", "BANKNIFTY", "BANK-NIFTY", "NSEBANK", "BANKNIFTYINDEX", "BANKNIFTYI"}:
        return "BANKNIFTY"
    # Standardize FINNIFTY
    if val in {"NIFTYFINSERVICE", "FINNIFTY", "FIN-NIFTY", "FINNIFTYINDEX"}:
        return "FINNIFTY"
    # Standardize MIDCPNIFTY
    if val in {"NIFTYMIDSELECT", "MIDCPNIFTY", "MIDCP-NIFTY", "MIDCPNIFTYINDEX"}:
        return "MIDCPNIFTY"
    return val

_MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_OPTION_ROOTS = ("SENSEX", "MIDCPNIFTY", "BANKNIFTY", "FINNIFTY", "NIFTY")


def _normalize_symbol_key(sym: str) -> str:
    return str(sym or "").strip().upper().replace(" ", "").replace("-", "")


def parse_option_tradingsymbol(sym: str) -> Optional[Dict[str, Any]]:
    """Parse NIFTY option symbols in DDMMMYY (synthetic) or YYMDD (m.Stock) formats."""
    s = _normalize_symbol_key(sym)
    if not s:
        return None
    underlying = ""
    rest = ""
    for root in _OPTION_ROOTS:
        if s.startswith(root):
            underlying = root
            rest = s[len(root):]
            break
    if not underlying or not rest:
        return None
    if rest.endswith("CE"):
        opt_type = "CE"
        body = rest[:-2]
    elif rest.endswith("PE"):
        opt_type = "PE"
        body = rest[:-2]
    else:
        return None

    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d+)$", body)
    if m:
        dd, mon, yy, strike_s = m.groups()
        mon_i = _MONTH_ABBR.get(mon)
        if mon_i:
            try:
                year = 2000 + int(yy) if len(yy) == 2 else int(yy)
                exp = date(year, mon_i, int(dd))
                strike = int(strike_s)
                return {
                    "underlying": underlying,
                    "expiry": exp,
                    "strike": strike,
                    "option_type": opt_type,
                    "tradingsymbol": s,
                }
            except Exception:
                pass

    m = re.match(r"^(\d{2})(\d)(\d{2})(\d+)$", body)
    if m:
        yy, mon, dd, strike_s = m.groups()
        try:
            year = 2000 + int(yy)
            exp = date(year, int(mon), int(dd))
            strike = int(strike_s)
            return {
                "underlying": underlying,
                "expiry": exp,
                "strike": strike,
                "option_type": opt_type,
                "tradingsymbol": s,
            }
        except Exception:
            pass
    return None


def contract_dict_from_scrip_row(row: "ScripMasterRow") -> Dict[str, Any]:
    sym = str(row.tradingsymbol or "").strip()
    tok = str(row.token or "").strip()
    exp_s = row.expiry.isoformat() if row.expiry else ""
    return {
        "trading_symbol": sym,
        "symbol": sym,
        "tradingsymbol": sym,
        "exchange": _normalize_exch_alias(row.exch) or "NFO",
        "token": tok,
        "symbolToken": tok,
        "security_id": tok,
        "strike_price": float(row.strike) if row.strike is not None else 0.0,
        "strike": float(row.strike) if row.strike is not None else 0.0,
        "option_type": row.opt_type,
        "expiry": exp_s,
        "expiry_date": exp_s,
        "lot_size": row.lot_size,
        "symbol_root": row.symbol_root,
    }


def _normalize_exch_alias(ex: str) -> str:
    """Map many broker exchange/segment aliases to canonical for filtering (NFO etc)."""
    e = str(ex or "").strip().upper().replace(" ", "").replace("_", "").replace("-", "")
    if not e:
        return "NFO"
    # Common NFO / FNO aliases (m.Stock / NSE derivatives)
    nfo_aliases = {"NFO", "NSEFNO", "NSEFO", "NSEFNO", "NSE_FNO", "DERIVATIVES", "OPTIDX", "OPT", "FNO", "FN O", "NSEFO", "5"}
    if e in nfo_aliases or e.endswith("FNO") or e.endswith("FO") or "DERIV" in e or "OPT" in e:
        return "NFO"
    # NSE cash etc, but for options we default NFO
    if e in {"NSE", "NSECASH", "NSECM"}:
        return "NSE"
    return e



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
        self._token_index: Optional[Dict[str, ScripMasterRow]] = None
        self._structured_index: Optional[Dict[str, ScripMasterRow]] = None
        self._load_elapsed_ms: float = 0.0

    @staticmethod
    def _structured_key(
        *,
        underlying: str,
        expiry: Optional[date],
        strike: Optional[int],
        option_type: str,
        exchange: str,
    ) -> str:
        exp_s = expiry.isoformat() if expiry else ""
        return "|".join([
            _normalize_root_alias(underlying),
            exp_s,
            str(strike or ""),
            _normalize_opt_type(option_type),
            _normalize_exch_alias(exchange),
        ])

    def _load(self) -> List[ScripMasterRow]:
        if self._rows is not None:
            return self._rows

        t0 = time.perf_counter()
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
        self._load_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_scripmaster_cache(hit=False, key=self.csv_path, rows=len(rows), elapsed_ms=int(self._load_elapsed_ms))
        return rows

    def _build_token_index(self) -> Dict[str, ScripMasterRow]:
        if self._token_index is not None:
            return self._token_index
        idx: Dict[str, ScripMasterRow] = {}
        for r in self._load():
            tok = str(r.token or "").strip()
            if tok:
                idx.setdefault(tok, r)
        self._token_index = idx
        return idx

    def _build_structured_index(self) -> Dict[str, ScripMasterRow]:
        if self._structured_index is not None:
            return self._structured_index
        idx: Dict[str, ScripMasterRow] = {}
        for r in self._load():
            if r.opt_type not in {"CE", "PE"}:
                continue
            key = self._structured_key(
                underlying=r.symbol_root,
                expiry=r.expiry,
                strike=r.strike,
                option_type=r.opt_type,
                exchange=r.exch,
            )
            idx.setdefault(key, r)
        self._structured_index = idx
        return idx

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

    def row_for_token(self, token: str) -> Optional[ScripMasterRow]:
        tok = str(token or "").strip()
        if not tok:
            return None
        row = self._build_token_index().get(tok)
        log_scripmaster_cache(hit=row is not None, key=f"token:{tok}")
        return row

    def token_for_tradingsymbol(self, tradingsymbol: str, *, exch: Optional[str] = None) -> Optional[str]:
        ts = str(tradingsymbol or "").strip().upper()
        if not ts:
            return None

        idx = self._build_tradingsymbol_index()
        ex = _normalize_exch_alias(str(exch or "").strip().upper()) if exch else ""

        if ex:
            tok = idx.get((ex, ts))
            if tok:
                log_scripmaster_cache(hit=True, key=f"symbol:{ex}:{ts}")
                return tok
        tok = idx.get(("", ts))
        log_scripmaster_cache(hit=tok is not None, key=f"symbol:{ts}")
        return tok

    def lookup_option_contract(
        self,
        *,
        tradingsymbol: str = "",
        underlying: str = "",
        expiry: Optional[date] = None,
        strike: Optional[int] = None,
        option_type: str = "",
        exch: str = "NFO",
    ) -> Optional[ScripMasterRow]:
        target_ex = _normalize_exch_alias(exch)
        opt = _normalize_opt_type(option_type)
        strike_i = _parse_int(strike) if strike is not None else None

        parsed = parse_option_tradingsymbol(tradingsymbol) if tradingsymbol else None
        if parsed:
            underlying = str(parsed.get("underlying") or underlying or "")
            expiry = parsed.get("expiry") or expiry
            strike_i = _parse_int(parsed.get("strike")) or strike_i
            opt = str(parsed.get("option_type") or opt or "")

        ts_key = _normalize_symbol_key(tradingsymbol)
        if ts_key:
            for r in self._load():
                r_ex = _normalize_exch_alias(getattr(r, "exch", ""))
                if r_ex and r_ex != target_ex:
                    continue
                if _normalize_symbol_key(r.tradingsymbol) == ts_key:
                    return r

        target_root = _normalize_root_alias(underlying) if underlying else ""
        if target_root and expiry is not None and strike_i is not None and opt in {"CE", "PE"}:
            skey = self._structured_key(
                underlying=target_root,
                expiry=expiry,
                strike=strike_i,
                option_type=opt,
                exchange=target_ex,
            )
            hit = self._build_structured_index().get(skey)
            log_scripmaster_cache(hit=hit is not None, key=skey)
            if hit is not None:
                return hit
        return None

    def token_for_option_symbol(self, tradingsymbol: str, *, exch: Optional[str] = None) -> Optional[str]:
        ts = str(tradingsymbol or "").strip()
        if not ts:
            return None
        tok = self.token_for_tradingsymbol(ts, exch=exch)
        if tok:
            return tok
        row = self.lookup_option_contract(tradingsymbol=ts, exch=exch or "NFO")
        return row.token if row else None

    def option_rows(
        self,
        *,
        symbol_root: str,
        exch: str = "NFO",
        min_expiry: Optional[date] = None,
        target_expiry: Optional[date] = None,  # exact match if provided (for specific GUI expiry)
    ) -> List[ScripMasterRow]:
        target_root = _normalize_root_alias(symbol_root)
        target_ex = _normalize_exch_alias(exch)
        min_e = min_expiry
        out: List[ScripMasterRow] = []
        for r in self._load():
            r_ex = _normalize_exch_alias(getattr(r, "exch", ""))
            if r_ex and r_ex != target_ex:
                continue
            if r.symbol_root != target_root:
                continue
            if r.opt_type not in {"CE", "PE"}:
                continue
            if r.expiry is None:
                continue
            if target_expiry is not None:
                if r.expiry != target_expiry:
                    continue
            elif min_e is not None and r.expiry < min_e:
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

        target_ex = _normalize_exch_alias(exch)
        for r in self._load():
            r_ex = _normalize_exch_alias(getattr(r, "exch", ""))
            if r_ex and r_ex != target_ex:
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
        target_ex = _normalize_exch_alias(exch)
        expiries = set()
        for r in self._load():
            r_ex = _normalize_exch_alias(getattr(r, "exch", ""))
            if r_ex and r_ex != target_ex:
                continue
            if _normalize_root_alias(r.symbol_root) == root:
                if r.expiry:
                    expiries.add(r.expiry)
        return sorted(list(expiries))

    def has_data(self) -> bool:
        return bool(self._load())


def get_scripmaster_singleton(csv_path: str) -> Optional[ScripMaster]:
    """Return a process-wide cached ScripMaster for the given CSV path."""
    path = str(csv_path or "").strip()
    if not path:
        return None
    cached = _SCRIPMASTER_INSTANCE_CACHE.get(path)
    if cached is not None:
        log_scripmaster_cache(hit=True, key=path, rows=len(cached._rows or []))
        return cached
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    sm = ScripMaster(path)
    _SCRIPMASTER_INSTANCE_CACHE[path] = sm
    return sm


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
