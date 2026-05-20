from __future__ import annotations

import os
import sys
import json
import time
import math
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

try:
    from tradingapi_b.mconnect import MConnectB
except ImportError:  # pragma: no cover
    # Prefer vendored SDK if present (repo includes pytradingapi-typeB-main/).
    _repo_root = Path(__file__).resolve().parent.parent
    _sdk_root = _repo_root / "pytradingapi-typeB-main"
    if _sdk_root.exists():
        sys.path.insert(0, str(_sdk_root))
    from tradingapi_b.mconnect import MConnectB

import yahoo_data
from config import APIConfig
from market_data import Candle
from scripmaster import ScripMaster


# --- m.Stock valid candle intervals ---
MSTOCK_INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}

_MSTOCK_ALLOWED_INTERVALS = set(MSTOCK_INTERVAL_MAP.values())

# m.Stock index candles: ONE_MINUTE can be unreliable; fallback to FIVE_MINUTE.
FALLBACK_INTERVALS = ["ONE_MINUTE", "FIVE_MINUTE"]


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    status: str


class MStockTypeBClient:
    """Thin wrapper around the official m.Stock Type B SDK.

    This class uses the ``tradingapi_b`` package (Mirae's official
    ``MConnectB`` client) and exposes a simplified interface that the
    strategy uses. You are responsible for providing a valid access
    token via configuration or environment variables.
    """

    def __init__(self, cfg: APIConfig) -> None:
        self.cfg = cfg
        api_key = str(cfg.api_key or "").strip()
        access_token = os.getenv("MSTOCK_ACCESS_TOKEN", "")

        # Keep env in sync so interactive auth helpers can reuse the same key
        # even when it was loaded from a saved file or SDK config.
        if api_key and not os.getenv("MSTOCK_API_KEY"):
            os.environ["MSTOCK_API_KEY"] = api_key

        # The official SDK reads default_root_uri and routes from its
        # own config module. You normally only need to supply the
        # API key and an already-generated access token.
        self._raw = MConnectB(api_key=api_key, access_Token=access_token)
        self._instruments_cache: Optional[List[Dict[str, Any]]] = None
        self._instrument_by_token: Optional[Dict[str, Dict[str, Any]]] = None
        self._scripmaster: Optional[ScripMaster] = None

        # Intraday endpoint can be flaky (HTTP 500, empty body). Keep a tiny
        # in-process cooldown to avoid spamming and to keep the bot progressing
        # with historical candles.
        self._intraday_fail_streak: int = 0
        self._intraday_disabled_until_ts: float = 0.0
        self._intraday_last_log_by_key: Dict[str, float] = {}
        self._candle_last_log_by_key: Dict[str, float] = {}
        # Optional startup bootstrap: use historical candles first for spot warmup
        # before switching to intraday flow.
        self._startup_ts: float = float(time.time())
        self._startup_historical_bootstrap_done: bool = False
        self._startup_historical_bootstrap_logged: bool = False

    def _log_intraday_failure(self, key: str, msg: str) -> None:
        """Log intraday failures only when explicitly debugging.

        The intraday endpoint is known to intermittently return HTTP 500.
        This is not fatal because we fall back to historical candles.
        """

        if not self._bool_env("MSTOCK_DEBUG_INTRADAY", False):
            return

        throttle_sec = 30
        try:
            throttle_sec = int(os.getenv("MSTOCK_INTRADAY_LOG_THROTTLE_SEC", str(throttle_sec)))
        except Exception:
            throttle_sec = 30

        now = float(time.time())
        last = float(self._intraday_last_log_by_key.get(str(key), 0.0) or 0.0)
        if now - last < float(throttle_sec):
            return
        self._intraday_last_log_by_key[str(key)] = now
        print(msg)

    def _bool_env(self, name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "yes", "y"}

    def _log_candle_event(self, *, key: str, msg: str) -> None:
        """Throttle repetitive candle-fallback logs.

        Strategy loops can call candle fetches frequently; this keeps logs useful
        without spamming the same fallback note every cycle.
        """

        throttle_sec = 60
        try:
            throttle_sec = int(os.getenv("MSTOCK_CANDLE_LOG_THROTTLE_SEC", str(throttle_sec)))
        except Exception:
            throttle_sec = 60

        now = float(time.time())
        last = float(self._candle_last_log_by_key.get(str(key), 0.0) or 0.0)
        if now - last < float(throttle_sec):
            return
        self._candle_last_log_by_key[str(key)] = now
        print(msg)

    def _should_use_startup_historical_bootstrap(self, symbol: str, *, intraday_enabled: bool) -> bool:
        """Return True when startup should prefer historical candles for spot warmup."""

        if not intraday_enabled:
            return False
        if self._startup_historical_bootstrap_done:
            return False
        if not self._bool_env("MSTOCK_STARTUP_USE_HISTORICAL_CANDLES", False):
            return False

        try:
            window_sec = int(os.getenv("MSTOCK_STARTUP_HISTORICAL_WINDOW_SEC", "180"))
        except Exception:
            window_sec = 180
        if window_sec <= 0:
            return False

        try:
            age = float(time.time() - float(self._startup_ts or 0.0))
        except Exception:
            age = float(window_sec + 1)
        if age > float(window_sec):
            return False

        # Apply only for the underlying/spot symbol.
        sym_raw = str(symbol or "").strip()
        if sym_raw and not sym_raw.isdigit():
            sym_key = self._normalize_symbol_key(sym_raw.split(":", 1)[-1])
        else:
            sym_key = ""

        underlying_raw = os.getenv("MSTOCK_UNDERLYING", os.getenv("MSTOCK_SYMBOL", "")).strip()
        if underlying_raw:
            under_key = self._normalize_symbol_key(str(underlying_raw).split(":", 1)[-1])
            if sym_key and under_key and sym_key != under_key:
                return False
        return True

    def _default_cache_dir(self) -> Path:
        # Prefer per-user location on Windows; fallback to cwd.
        base = os.getenv("APPDATA")
        if base:
            return Path(base) / "scalper"
        return Path(os.getcwd())

    def _exchange_segment_id(self, exchange: str) -> str:
        """Map exchange/segment to the numeric ids expected by openapi intraday.

        Docs: 1-NSE, 2-NFO, 3-CDS, 4-BSE, 5-BFO.
        """

        ex = str(exchange or "").strip().upper()
        if not ex:
            return "1"
        if ex.isdigit():
            return ex
        mapping = {
            "NSE": "1",
            "NFO": "2",
            "CDS": "3",
            "BSE": "4",
            "BFO": "5",
        }
        return mapping.get(ex, "1")

    def _exchange_id_to_str(self, exchange_id: str) -> str:
        """Best-effort mapping for numeric exchange ids -> exchange string.

        The official TypeB SDK historical endpoint examples use string exchanges
        like "NSE"/"NFO".
        """

        ex = str(exchange_id or "").strip()
        mapping = {
            "1": "NSE",
            "2": "NFO",
            "3": "CDS",
            "4": "BSE",
            "5": "BFO",
        }
        return mapping.get(ex, ex)

    def _known_index_token(self, symbol_key: str, *, exchange: str) -> str:
        """Return known spot index token when available.

        Some TypeB instrument masters list only derivatives for indices (NFO:NIFTY)
        which makes auto-resolve ambiguous. In practice, the historical candles
        endpoint accepts these well-known spot index tokens on NSE for many setups.
        """

        ex = str(exchange or "").strip().upper()
        key = str(symbol_key or "").strip().upper()
        if ex in {"1", "NSE", ""}:
            mapping = {
                "NIFTY": "26000",
                "BANKNIFTY": "26009",
                "FINNIFTY": "26037",
                "MIDCPNIFTY": "26074",
            }
            return mapping.get(key, "")
        return ""

    def _fetch_intraday_chart(self, *, exchange_segment: str, symboltoken: str, interval: str, limit: int) -> List[Candle]:
        """Fetch current-day intraday candles via openapi intraday endpoint.

        Enabled by env `MSTOCK_USE_INTRADAY_CHART=1`.
        """

        api_key = str(self.cfg.api_key or "").strip()
        access_token = str(os.getenv("MSTOCK_ACCESS_TOKEN", "")).strip()
        if not (api_key and access_token):
            return []

        seg_raw = str(exchange_segment or "").strip().upper() or "NSE"
        tok = str(symboltoken or "").strip()
        if not (tok.isdigit() and int(tok) > 0):
            return []

        # Accept either a normalized SDK interval constant or a raw tf like "1m".
        interval_norm = str(self._normalize_interval(interval)).strip().upper()
        if interval_norm not in _MSTOCK_ALLOWED_INTERVALS:
            return []

        # The intraday endpoint is for current-day intraday candles only.
        # Avoid calling it for daily intervals (observed to produce HTTP 500s).
        if interval_norm == "ONE_DAY":
            return []

        url = os.getenv(
            "MSTOCK_INTRADAY_URL",
            "https://api.mstock.trade/openapi/typeb/instruments/intraday",
        ).strip()
        if not url:
            return []

        try:
            timeout = float(os.getenv("MSTOCK_INTRADAY_TIMEOUT", "10"))
        except Exception:
            timeout = 10.0

        headers = {
            "X-Mirae-Version": "1",
            "X-PrivateKey": api_key,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        def _do_request_post(exchange_value: str) -> str:
            payload = {
                "exchange": str(exchange_value),
                "symboltoken": tok,
                "interval": interval_norm,
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    return raw.decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = (exc.read() or b"")[:400].decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                raise RuntimeError(
                    f"intraday chart failed with HTTP {exc.code}. Body preview: {body!r}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"intraday chart request failed: {exc}") from exc

        def _do_request_get(exchange_value: str) -> str:
            # Docs are inconsistent about method/body; some builds accept GET with query params.
            q = (
                f"exchange={urllib.parse.quote(str(exchange_value))}"
                f"&symboltoken={urllib.parse.quote(tok)}"
                f"&interval={urllib.parse.quote(interval_norm)}"
            )
            url_get = url
            if "?" in url_get:
                url_get = url_get + "&" + q
            else:
                url_get = url_get + "?" + q

            req = urllib.request.Request(
                url_get,
                headers=headers,
                method="GET",
            )

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    return raw.decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = (exc.read() or b"")[:400].decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                raise RuntimeError(
                    f"intraday chart failed with HTTP {exc.code}. Body preview: {body!r}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"intraday chart request failed: {exc}") from exc

        # Different brokers/builds of TypeB appear to accept either string exchange
        # ("NSE") or numeric segment id ("1"). Try raw first, then numeric when:
        # - the server returns a common invalid-symbol style error, OR
        # - the server succeeds but returns an empty candle set.

        seg_id = self._exchange_segment_id(seg_raw)

        def _parse_text_to_candles(text_in: str, *, exch_sent: str) -> List[Candle]:
            try:
                data = json.loads(text_in)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"intraday chart returned non-JSON: {text_in[:300]!r}") from exc

            candles_out = self._parse_candles_payload(data, limit=limit)

            # When candles are empty, surface status/message/errorcode in debug logs.
            if not candles_out and isinstance(data, dict):
                try:
                    st = data.get("status")
                    msg = data.get("message")
                    ec = data.get("errorcode")
                    if any(v is not None for v in (st, msg, ec)):
                        self._log_intraday_failure(
                            key=f"intraday-empty-meta:{exch_sent}:{tok}:{interval_norm}",
                            msg=(
                                "[INTRADAY] Empty candles meta; "
                                f"exch_sent={exch_sent} token={tok} interval={interval_norm} "
                                f"status={st!r} message={msg!r} errorcode={ec!r}"
                            ),
                        )
                except Exception:
                    pass

            return candles_out

        try:
            text = _do_request_post(seg_raw)
            candles = _parse_text_to_candles(text, exch_sent=str(seg_raw))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            should_retry = (
                (seg_id != seg_raw)
                and (
                    "ia400" in msg
                    or "invalid symbol" in msg
                    or "scrip name not found" in msg
                    or "not found" in msg
                )
            )
            if not should_retry:
                raise
            text = _do_request_post(seg_id)
            candles = _parse_text_to_candles(text, exch_sent=str(seg_id))

        # If POST succeeds but returns empty, retry with GET-style query params.
        if not candles:
            try:
                text_g = _do_request_get(seg_raw)
                candles = _parse_text_to_candles(text_g, exch_sent=str(seg_raw))
            except Exception:
                candles = []
            if (not candles) and (seg_id != seg_raw):
                try:
                    text_g2 = _do_request_get(seg_id)
                    candles = _parse_text_to_candles(text_g2, exch_sent=str(seg_id))
                except Exception:
                    candles = []

        # Retry with numeric segment id when the string exchange yields empty candles.
        if (not candles) and (seg_id != seg_raw):
            try:
                text2 = _do_request_post(seg_id)
                candles2 = _parse_text_to_candles(text2, exch_sent=str(seg_id))
                if candles2:
                    return candles2
            except Exception:
                # Ignore: caller handles failures/empty.
                pass

        return candles

    def _intraday_available(self) -> bool:
        try:
            import time as _time

            return float(_time.time()) >= float(self._intraday_disabled_until_ts or 0.0)
        except Exception:
            return True

    def _note_intraday_failure(self, exc: Exception) -> None:
        # Simple exponential-ish cooldown after repeated failures.
        try:
            import time as _time

            now = float(_time.time())
        except Exception:
            return

        self._intraday_fail_streak = int(self._intraday_fail_streak or 0) + 1

        # Only disable for server-ish problems (HTTP 5xx / empty body).
        msg = str(exc).lower()
        if "http 5" not in msg and "body preview" not in msg:
            return

        # After 3 consecutive failures, cool down for 60 seconds.
        if self._intraday_fail_streak >= 3:
            self._intraday_disabled_until_ts = now + 60.0

    def _note_intraday_success(self) -> None:
        self._intraday_fail_streak = 0
        self._intraday_disabled_until_ts = 0.0

    def _download_openapi_scripmaster(self, *, dest_csv: Path) -> bool:
        """Download OpenAPIScripMaster and write as CSV.

        The endpoint can return CSV text or JSON array depending on broker.
        """

        api_key = str(self.cfg.api_key or "").strip()
        access_token = str(os.getenv("MSTOCK_ACCESS_TOKEN", "")).strip()
        if not (api_key and access_token):
            return False

        url = os.getenv(
            "MSTOCK_SCRIPMASTER_URL",
            "https://api.mstock.trade/openapi/typeb/instruments/OpenAPIScripMaster",
        ).strip()
        if not url:
            return False

        try:
            timeout = float(os.getenv("MSTOCK_SCRIPMASTER_TIMEOUT", "30"))
        except Exception:
            timeout = 30.0

        headers = {
            "X-Mirae-Version": "1",
            "X-PrivateKey": api_key,
            "Authorization": f"Bearer {access_token}",
        }

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                content_type = (resp.headers.get("Content-Type") or "").lower()
        except Exception as exc:  # noqa: BLE001
            print(f"[SCRIPMASTER] Download failed: {exc}")
            return False

        text = ""
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = str(raw)

        dest_csv.parent.mkdir(parents=True, exist_ok=True)

        # If it looks like JSON (or content-type says so), convert to CSV.
        is_json = "json" in content_type or text.lstrip().startswith("[") or text.lstrip().startswith("{")
        if is_json:
            try:
                payload = json.loads(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[SCRIPMASTER] JSON parse failed: {exc}")
                return False

            rows: List[Dict[str, Any]] = []
            if isinstance(payload, list):
                rows = [r for r in payload if isinstance(r, dict)]
            elif isinstance(payload, dict):
                # Some APIs wrap under "data".
                data = payload.get("data")
                if isinstance(data, list):
                    rows = [r for r in data if isinstance(r, dict)]

            if not rows:
                print("[SCRIPMASTER] JSON payload had no rows")
                return False

            # Union of keys (stable-ish ordering).
            keys: List[str] = []
            seen: set[str] = set()
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)

            import csv

            with dest_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k) for k in keys})

            print(f"[SCRIPMASTER] Downloaded and saved JSON->CSV to {dest_csv}")
            return True

        # Otherwise treat as CSV text.
        try:
            dest_csv.write_text(text, encoding="utf-8")
            print(f"[SCRIPMASTER] Downloaded and saved CSV to {dest_csv}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[SCRIPMASTER] Write failed: {exc}")
            return False

    def _get_scripmaster(self) -> Optional[ScripMaster]:
        path = os.getenv("MSTOCK_SCRIPMASTER_PATH", "").strip()

        # If a configured path is present but invalid, don't get stuck.
        # Fall back to auto-download/cached master (zero-config behavior).
        if path:
            try:
                p = Path(path)
                if not (p.exists() and p.is_file()):
                    print(f"[SCRIPMASTER] Configured path not found: {path!r} (will auto-fetch if enabled)")
                    path = ""
            except Exception:
                path = ""
        if not path:
            # Zero-config default: if a file named `scripmaster.csv` exists in the
            # current working directory, use it.
            try:
                default_path = os.path.join(os.getcwd(), "scripmaster.csv")
                if os.path.isfile(default_path):
                    path = default_path
            except Exception:
                path = ""

        if not path:
            # Common fallback: many users already have an instrument master CSV
            # downloaded manually (e.g. "instrument (1).csv"). This file format is
            # compatible with our lightweight ScripMaster reader.
            try:
                cwd = os.getcwd()
                for candidate in (
                    "instrument.csv",
                    "instrument (1).csv",
                    "instruments.csv",
                    "instruments (1).csv",
                ):
                    cand_path = os.path.join(cwd, candidate)
                    if os.path.isfile(cand_path):
                        path = cand_path
                        break
            except Exception:
                path = ""

        # If still missing, optionally auto-download and cache.
        if not path and self._bool_env("MSTOCK_AUTO_FETCH_SCRIPMASTER", True):
            cache_dir = self._default_cache_dir()
            cache_path = cache_dir / "scripmaster.csv"

            # Refresh daily.
            refresh = True
            try:
                if cache_path.exists():
                    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    refresh = mtime.date() != datetime.now().date()
            except Exception:
                refresh = True

            if refresh:
                self._download_openapi_scripmaster(dest_csv=cache_path)

            if cache_path.exists():
                path = str(cache_path)

        if not path:
            return None
        if self._scripmaster is not None and self._scripmaster.csv_path == path:
            return self._scripmaster
        self._scripmaster = ScripMaster(path)
        return self._scripmaster

    def _parse_date_loose(self, value: object) -> Optional[datetime.date]:
        if value is None:
            return None
        if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
            try:
                # date or datetime
                return value.date() if hasattr(value, "date") else value  # type: ignore[return-value]
            except Exception:
                pass

        s = str(value).strip()
        if not s:
            return None

        try:
            return datetime.fromisoformat(s.replace("Z", "")).date()
        except Exception:
            pass

        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d-%b-%y", "%d%b%Y", "%d%b%y"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue
        return None

    def _derive_symbol_root(self, tradingsymbol: str) -> str:
        ts = str(tradingsymbol or "").strip().upper()
        if not ts:
            return ""
        for root in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"):
            if ts.startswith(root):
                return root
        out: List[str] = []
        for ch in ts:
            if "A" <= ch <= "Z":
                out.append(ch)
            else:
                break
        return "".join(out)

    def _resolve_token_for_quote(self, symbol: str, *, exchange_hint: Optional[str] = None) -> tuple[str, str]:
        """Resolve (exchange, token) for quote/LTP calls.

        Accepts:
        - numeric token strings ("26000")
        - "EXCH:TRADINGSYMBOL" ("NFO:NIFTY09JAN26...")
        - plain trading symbol (needs exchange_hint / env / scripmaster)
        """

        raw = str(symbol or "").strip()
        if not raw:
            return "", ""

        exch = str(exchange_hint or "").strip().upper()
        sym = raw
        exchange_explicit = False
        if ":" in raw:
            maybe_exch, maybe_sym = raw.split(":", 1)
            if maybe_exch:
                exchange_explicit = True
                exch = maybe_exch.strip().upper() or exch
            sym = maybe_sym.strip()

        # Heuristic: if the caller didn't specify an exchange and the symbol looks like
        # an option/future contract (e.g. NIFTY...CE/PE), default to NFO so we don't
        # accidentally resolve/quote on NSE.
        if not exch:
            sym_u = str(sym or "").strip().upper()
            if sym_u.endswith("CE") or sym_u.endswith("PE") or sym_u.endswith("FUT"):
                exch = (os.getenv("MSTOCK_SCRIPMASTER_EXCH", "NFO") or "NFO").strip().upper() or "NFO"

        # If caller passed a token directly, use it (accept float-like tokens).
        if sym.isdigit() and int(sym) > 0:
            return (exch or "NSE"), sym
        if sym:
            try:
                sym_float = float(sym)
                if sym_float.is_integer() and int(sym_float) > 0:
                    return (exch or "NSE"), str(int(sym_float))
            except Exception:
                pass

        # Special-case underlying: prefer configured token.
        sym_key = str(sym).strip().upper()
        # Heuristic: if it matches NIFTY or the configured underlying/symbol, use the dedicated token env.
        is_underlying = sym_key in {"NIFTY", "NIFTY50", "NIFTY 50"}
        cfg_underlying = os.getenv("MSTOCK_UNDERLYING", "").strip().upper()
        cfg_symbol = os.getenv("MSTOCK_SYMBOL", "").strip().upper()
        if cfg_underlying and sym_key == cfg_underlying:
            is_underlying = True
        if cfg_symbol and sym_key == cfg_symbol:
            is_underlying = True

        # For spot index underlyings, prefer NSE quotes unless the exchange was explicit.
        if is_underlying and (not exchange_explicit) and exch in {"", "NFO", "BFO"}:
            exch = "NSE"

        # If caller explicitly asked for derivatives exchange (NFO/BFO), do NOT
        # force the spot underlying token; this is commonly used for futures hedges
        # like "NFO:NIFTY".
        if (exch in {"", "NSE", "NFO"}) and is_underlying and not (exchange_explicit and exch in {"NFO", "BFO"}):
            env_tok = os.getenv("MSTOCK_UNDERLYING_TOKEN", "").strip()
            env_exch = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE")).strip().upper() or "NSE"
            # Keep spot indices on NSE for quotes unless explicitly overridden.
            if not exchange_explicit and exch:
                env_exch = exch
            if env_tok.isdigit() and int(env_tok) > 0:
                return env_exch, env_tok

            # Known spot index tokens as a fallback when env token isn't set.
            sym_key_canon = sym_key
            if sym_key_canon in {"NIFTY50", "NIFTY 50"}:
                sym_key_canon = "NIFTY"
            known = self._known_index_token(sym_key_canon, exchange=env_exch)
            if known and known.isdigit():
                return env_exch or "NSE", known

        # Try ScripMaster first (fast exact tradingsymbol match).
        sm = self._get_scripmaster()
        if sm is not None:
            tok = sm.token_for_tradingsymbol(sym, exch=exch or None)
            if tok and tok.isdigit() and int(tok) > 0:
                return (exch or "NFO"), tok

            # Delta-hedge convenience: allow FUT aliases like "NFO:NIFTY" or "NFO:BANKNIFTY"
            # to mean the nearest-expiry index future.
            # We only apply this when the exchange is explicitly/implicitly NFO.
            ex_u = str(exch or "").strip().upper()
            if ex_u in {"NFO", ""}:
                root_guess = str(sym or "").strip().upper()
                # Allow either plain roots or prefixed variants.
                if root_guess in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}:
                    fut_ts = sm.nearest_future_tradingsymbol(root_guess, exch=(os.getenv("MSTOCK_SCRIPMASTER_EXCH", "NFO") or "NFO"))
                    if fut_ts:
                        tok2 = sm.token_for_tradingsymbol(fut_ts, exch=(ex_u or None))
                        if tok2 and tok2.isdigit() and int(tok2) > 0:
                            # Keep exchange on derivatives.
                            return ("NFO" if ex_u == "" else ex_u), tok2

        # Fall back to instruments master resolution.
        resolved = self._resolve_token_from_instruments(sym, exch or "NSE")
        if resolved and resolved.isdigit() and int(resolved) > 0:
            return (exch or "NSE"), resolved

        return "", ""

    def resolve_exchange_token(self, symbol: str, *, exchange_hint: Optional[str] = None) -> tuple[str, str]:
        """Resolve (exchange, token) for a trading symbol.

        This is a small public wrapper around the internal token resolution
        logic used by quotes, so strategy code (e.g. delta hedging) can place
        orders while still reusing ScripMaster/instruments resolution.
        """

        exch, tok = self._resolve_token_for_quote(symbol, exchange_hint=exchange_hint)
        if not exch or not tok:
            raise RuntimeError(
                "Unable to resolve exchange/token for symbol. Provide a numeric token, an EXCH:TRADINGSYMBOL, "
                "or configure ScripMaster/instruments for resolution."
            )
        return exch, tok

    def _get_option_chain_from_csv(self, underlying: str) -> List[Dict[str, Any]]:
        sm = self._get_scripmaster()
        if sm is None:
            return []

        root = str(underlying or "").strip().upper()
        if ":" in root:
            root = root.split(":", 1)[-1].strip().upper()
        if not root:
            return []

        # Filter to future expiries only.
        rows = sm.option_rows(symbol_root=root, exch=os.getenv("MSTOCK_SCRIPMASTER_EXCH", "NFO") or "NFO", min_expiry=datetime.now().date())
        if not rows:
            return []

        chain: List[Dict[str, Any]] = []
        for r in rows:
            # Prefer the CSV trading symbol; fall back to a minimal generated one.
            sym = r.tradingsymbol
            if not sym:
                # Include DDMMMYY so strategy weekly-expiry parser can still work if needed.
                exp = r.expiry.strftime("%d%b%y").upper() if r.expiry else ""
                sym = f"{r.symbol_root}{exp}{int(r.strike or 0)}{r.opt_type}"

            chain.append(
                {
                    "symbol": sym,
                    "token": r.token,
                    "exchange": (r.exch or "NFO").strip().upper(),
                    "strike": float(r.strike) if r.strike is not None else 0.0,
                    "option_type": r.opt_type,
                    "expiry": r.expiry,
                    "symbol_root": r.symbol_root,
                    "lot_size": r.lot_size,
                    "raw": r.raw,
                }
            )

        return chain

    def _get_option_chain_from_api(self) -> List[Dict[str, Any]]:
        """Existing option-chain API integration (kept as-is, env-configured)."""
        exchange_id = os.getenv("MSTOCK_OPTION_EXCHANGE_ID", "").strip()
        expiry = os.getenv("MSTOCK_OPTION_EXPIRY", "").strip()
        token = os.getenv("MSTOCK_OPTION_TOKEN", "").strip()

        if not (exchange_id and expiry and token):
            raise RuntimeError(
                "Option chain parameters are not configured. Set "
                "MSTOCK_OPTION_EXCHANGE_ID, MSTOCK_OPTION_EXPIRY, and "
                "MSTOCK_OPTION_TOKEN according to your m.Stock instruments."
            )

        resp = self._raw.get_option_chain_data(exchange_id, expiry, token)
        payload = self._safe_json(resp, context="get_option_chain_data")

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise RuntimeError(
                "Unexpected option chain response. Inspect "
                "get_option_chain_data().json() and adjust parsing."
            )

        chain: List[Dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue

            raw_type = str(row.get("optionType") or row.get("option_type") or row.get("opt_type") or "").upper().strip()
            if raw_type in {"CALL", "C"}:
                raw_type = "CE"
            elif raw_type in {"PUT", "P"}:
                raw_type = "PE"
            if raw_type not in {"CE", "PE"}:
                continue

            strike_raw = row.get("strikePrice") or row.get("strike_price") or row.get("strike")
            iv_raw = row.get("iv") or row.get("impliedVolatility")

            try:
                strike_val = float(strike_raw) if strike_raw is not None else 0.0
            except Exception:
                strike_val = 0.0

            try:
                iv_val = float(iv_raw) if iv_raw is not None else 0.0
            except Exception:
                iv_val = 0.0

            symbol_val = str(row.get("tradingsymbol") or row.get("tradingSymbol") or row.get("symbol") or row.get("tokenName") or "")
            token_val = str(
                row.get("symboltoken")
                or row.get("symbolToken")
                or row.get("token")
                or row.get("instrumentToken")
                or row.get("symbol_token")
                or ""
            ).strip()
            exch_val = str(
                row.get("exchange")
                or row.get("exch_seg")
                or row.get("exchangeSegment")
                or row.get("exchange_segment")
                or os.getenv("MSTOCK_EXCHANGE", "")
                or ""
            ).strip()

            expiry_val = None
            for k in ("expiry", "expiryDate", "expDate", "expiry_date", "expdate"):
                if k in row and row.get(k):
                    expiry_val = self._parse_date_loose(row.get(k))
                    if expiry_val is not None:
                        break

            symbol_root = str(row.get("symbol_root") or row.get("underlying") or "").strip().upper()
            if not symbol_root and symbol_val:
                symbol_root = self._derive_symbol_root(symbol_val)

            chain.append(
                {
                    "symbol": symbol_val,
                    "token": token_val,
                    "exchange": exch_val,
                    "strike": strike_val,
                    "option_type": raw_type,
                    "iv": iv_val,
                    "expiry": expiry_val,
                    "symbol_root": symbol_root,
                    "raw": row,
                }
            )

        if not chain:
            raise RuntimeError(
                "Parsed option chain is empty. Verify MSTOCK_OPTION_* "
                "settings and response structure, then adjust mapping in "
                "MStockTypeBClient.get_option_chain."
            )

        return chain

    def _normalize_symbol_key(self, s: str) -> str:
        return "".join(ch for ch in (s or "").upper() if ch.isalnum())

    def _load_instruments(self) -> List[Dict[str, Any]]:
        if self._instruments_cache is not None:
            return self._instruments_cache

        resp = self._raw.get_instruments()
        payload = self._safe_json(resp, context="get_instruments")

        data = None
        if isinstance(payload, dict):
            # Common pattern: {"data": [...]} or {"instruments": [...]}
            data = payload.get("data") or payload.get("instruments")
        elif isinstance(payload, list):
            data = payload

        if not isinstance(data, list):
            raise RuntimeError(
                "Unexpected instruments response. Unable to auto-resolve tokens. "
                "Set MSTOCK_UNDERLYING_TOKEN manually."
            )

        instruments: List[Dict[str, Any]] = [row for row in data if isinstance(row, dict)]
        self._instruments_cache = instruments
        self._instrument_by_token = None
        return instruments

    def _get_instrument_by_token(self) -> Dict[str, Dict[str, Any]]:
        if self._instrument_by_token is not None:
            return self._instrument_by_token

        m: Dict[str, Dict[str, Any]] = {}
        for row in self._load_instruments():
            tok = self._extract_token(row)
            if tok and tok.isdigit():
                m.setdefault(tok, row)
        self._instrument_by_token = m
        return m

    def _token_matches_symbol_key(self, token: str, *, symbol_key: str) -> bool:
        tok = str(token or "").strip()
        key = str(symbol_key or "").strip().upper()
        if not (tok and key):
            return True

        row = self._get_instrument_by_token().get(tok)
        if not row:
            return True

        name_blob = str(
            row.get("tradingsymbol")
            or row.get("tradingSymbol")
            or row.get("symbol")
            or row.get("name")
            or row.get("tokenName")
            or ""
        )
        return self._normalize_symbol_key(name_blob) == key

    def _extract_token(self, row: Dict[str, Any]) -> str:
        for k in (
            "symboltoken",
            "symbolToken",
            "instrumentToken",
            "instrument_token",
            "instrumenttoken",
            "token",
            "Token",
            "symbol_token",
        ):
            v = row.get(k)
            if v is None:
                continue
            sv = str(v).strip()
            if sv.isdigit():
                return sv
        return ""

    def _extract_exchange(self, row: Dict[str, Any]) -> str:
        for k in ("exchange", "exch_seg", "exchangeSegment", "exchange_segment"):
            v = row.get(k)
            if v:
                return str(v).strip().upper()
        return ""

    def _candidate_tokens_for_symbol(self, symbol: str, exchange: str, *, limit: int = 12) -> List[Dict[str, str]]:
        """Return a ranked list of candidate (exchange, token, name) for a symbol.

        This is best-effort only: instrument masters can include many NIFTY*
        ETFs and indices. We rank likely index-like instruments higher.
        """

        raw = (symbol or "").strip()
        exch = (exchange or "").strip().upper()
        sym = raw
        if ":" in raw:
            maybe_exch, maybe_sym = raw.split(":", 1)
            if maybe_exch and not exch:
                exch = maybe_exch.strip().upper()
            sym = maybe_sym.strip()

        sym_key = self._normalize_symbol_key(sym)
        if not sym_key:
            return []

        def score(row: Dict[str, Any]) -> int:
            name = str(
                row.get("tradingsymbol")
                or row.get("tradingSymbol")
                or row.get("symbol")
                or row.get("name")
                or row.get("tokenName")
                or ""
            ).upper()
            it = str(
                row.get("instrumenttype")
                or row.get("instrumentType")
                or row.get("segment")
                or row.get("series")
                or ""
            ).upper()

            s = 0
            if sym_key == self._normalize_symbol_key(name):
                s += 25
            if sym_key and sym_key in self._normalize_symbol_key(name):
                s += 10
            if "NIFTY50" in name or "NIFTY 50" in name:
                s += 50
            if "INDEX" in it or "INDEX" in name:
                s += 40
            # If user asked for NIFTY, strongly avoid broader-index ETFs.
            if sym_key in {"NIFTY", "NIFTY50"}:
                for bad in ("NIFTY500", "NIFTY 500", "NIFTY100", "NIFTY 100", "NEXT50", "NEXT 50"):
                    if bad in name:
                        s -= 60
            # Penalize common ETF/factor products.
            for bad in ("ETF", "BEES", "FUND", "ADD", "BETA", "QLITY", "HDFC", "QNIFTY"):
                if bad in name:
                    s -= 12
            if 0 < len(name.strip()) <= 12:
                s += 3
            return s

        instruments = self._load_instruments()
        ranked: List[tuple[int, Dict[str, Any]]] = []
        for row in instruments:
            if not isinstance(row, dict):
                continue
            tok = self._extract_token(row)
            if not tok:
                continue

            row_exch = self._extract_exchange(row)
            # If exchange is known and doesn't match, down-rank but don't drop completely
            # because some masters omit exchange or use variants.
            base = score(row)
            if exch and row_exch and row_exch != exch:
                base -= 15
            # Must at least mention the symbol key somewhere.
            name_blob = str(
                row.get("tradingsymbol")
                or row.get("tradingSymbol")
                or row.get("symbol")
                or row.get("name")
                or row.get("tokenName")
                or ""
            )
            if sym_key not in self._normalize_symbol_key(name_blob):
                continue
            ranked.append((base, row))

        ranked.sort(key=lambda t: t[0], reverse=True)

        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for _, row in ranked:
            tok = self._extract_token(row)
            if not tok or tok in seen:
                continue
            seen.add(tok)
            name = str(
                row.get("tradingsymbol")
                or row.get("tradingSymbol")
                or row.get("symbol")
                or row.get("name")
                or row.get("tokenName")
                or ""
            ).strip()
            row_exch = self._extract_exchange(row) or exch
            out.append({"exchange": row_exch or exch or "NSE", "token": tok, "name": name})
            if len(out) >= int(limit):
                break

        return out

    def _resolve_token_from_instruments(self, symbol: str, exchange: str) -> str:
        # Accept formats like "NSE:NIFTY".
        raw = (symbol or "").strip()
        exch = (exchange or "").strip().upper()
        sym = raw
        if ":" in raw:
            maybe_exch, maybe_sym = raw.split(":", 1)
            if maybe_exch and not exch:
                exch = maybe_exch.strip().upper()
            sym = maybe_sym.strip()

        sym_key = self._normalize_symbol_key(sym)
        if not sym_key:
            return ""

        index_keys = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}
        is_index = sym_key in index_keys

        instruments = self._load_instruments()

        def _collect_candidates(*, allow_mismatch: bool) -> List[Dict[str, Any]]:
            # Prefer exact matches on common fields.
            exact_fields = ("tradingsymbol", "tradingSymbol", "symbol", "name", "tokenName")
            out: List[Dict[str, Any]] = []
            for row in instruments:
                row_exch = self._extract_exchange(row)
                if (not allow_mismatch) and exch and row_exch and row_exch != exch:
                    continue

                for f in exact_fields:
                    v = row.get(f)
                    if not v:
                        continue
                    if self._normalize_symbol_key(str(v)) == sym_key:
                        tok = self._extract_token(row)
                        if tok:
                            out.append(row)
                            break

            # Fallback: allow partial matches against `name`/`symbol`.
            if not out:
                for row in instruments:
                    row_exch = self._extract_exchange(row)
                    if (not allow_mismatch) and exch and row_exch and row_exch != exch:
                        continue
                    for f in ("name", "symbol", "tradingsymbol", "tradingSymbol"):
                        v = row.get(f)
                        if not v:
                            continue
                        key = self._normalize_symbol_key(str(v))
                        if sym_key and sym_key in key:
                            tok = self._extract_token(row)
                            if tok:
                                out.append(row)
                                break
            return out

        # For indices: strongly prefer the requested exchange (spot NSE) first.
        # Only widen to exchange-mismatch if we couldn't find anything.
        candidates = _collect_candidates(allow_mismatch=False)
        if not candidates and is_index:
            # As a last resort, use known spot index tokens for NSE.
            known = self._known_index_token(sym_key, exchange=exch)
            if known:
                return known
            candidates = _collect_candidates(allow_mismatch=True)

        if len(candidates) == 1:
            return self._extract_token(candidates[0])

        if len(candidates) > 1:
            if is_index:
                # If multiple candidates remain for an index, prefer known NSE spot token.
                known = self._known_index_token(sym_key, exchange=exch)
                if known:
                    return known
            # Try to auto-pick an index-like instrument when possible.
            def _score(row: Dict[str, Any]) -> int:
                name = str(
                    row.get("tradingsymbol")
                    or row.get("tradingSymbol")
                    or row.get("symbol")
                    or row.get("name")
                    or ""
                ).upper()
                it = str(
                    row.get("instrumenttype")
                    or row.get("instrumentType")
                    or row.get("segment")
                    or row.get("series")
                    or ""
                ).upper()

                score = 0
                # Heuristic for indices.
                indices = {"NIFTY": "NIFTY", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN", "MIDCPNIFTY": "MIDCP NIFTY"}
                for root, match_key in indices.items():
                    if match_key in name or root in name:
                        score += 50
                
                if "INDEX" in it or "INDEX" in name:
                    score += 40
                
                if sym_key in indices:
                    # Penalize other index products when looking for the main index.
                    for other_root in indices:
                        if other_root != sym_key and other_root in name:
                            score -= 60
                    # Standard NIFTY penalties.
                    if sym_key == "NIFTY":
                        for bad in ("NIFTY500", "NIFTY 500", "NIFTY100", "NIFTY 100", "NEXT50", "NEXT 50"):
                            if bad in name:
                                score -= 60
                # Penalize ETFs/funds and common NIFTY-factor products.
                for bad in ("ETF", "BEES", "FUND", "ADD", "BETA", "QLITY", "HDFC", "QNIFTY"):
                    if bad in name:
                        score -= 10
                if 0 < len(name.strip()) <= 10:
                    score += 5
                return score

            ranked = sorted(candidates, key=_score, reverse=True)
            if ranked:
                top = _score(ranked[0])
                second = _score(ranked[1]) if len(ranked) > 1 else -999
                if top > second:
                    tok = self._extract_token(ranked[0])
                    if tok:
                        return tok

            # Provide a short hint to disambiguate.
            preview: List[str] = []
            for row in candidates[:8]:
                tok = self._extract_token(row)
                ts = str(row.get("tradingsymbol") or row.get("tradingSymbol") or row.get("symbol") or row.get("name") or "")
                ex = self._extract_exchange(row) or exch or ""
                preview.append(f"{ex}:{ts} -> {tok}")
            raise RuntimeError(
                "Multiple instrument tokens match the underlying symbol. "
                "Set MSTOCK_UNDERLYING_TOKEN to the correct one. Candidates: "
                + "; ".join(preview)
            )

        return ""

    def _safe_json(self, resp: Any, *, context: str) -> Any:
        """Parse SDK response JSON with helpful diagnostics.

        The underlying SDK usually returns a `requests.Response`-like object.
        When the API returns an empty body or HTML (auth error, downtime, etc.),
        `response.json()` raises a JSON decode error like:
        "Expecting value: line 1 column 1 (char 0)".

        This helper turns that into a more actionable exception including
        status code and a small body preview.
        """

        if isinstance(resp, (dict, list)):
            return resp

        if not hasattr(resp, "json"):
            return resp

        # If the response has a non-2xx code, surface it early.
        status_code = getattr(resp, "status_code", None)
        req = getattr(resp, "request", None)
        req_method = getattr(req, "method", None) if req is not None else None
        req_url = getattr(req, "url", None) if req is not None else None
        resp_url = getattr(resp, "url", None)

        if isinstance(status_code, int) and status_code >= 400:
            body_preview = ""
            try:
                body_preview = (getattr(resp, "text", "") or "")[:400]
            except Exception:
                body_preview = ""
            header_preview = ""
            try:
                hdrs = getattr(resp, "headers", None)
                if isinstance(hdrs, dict):
                    header_preview = str({k: hdrs.get(k) for k in ("Content-Type", "Date", "Server") if k in hdrs})
            except Exception:
                header_preview = ""

            target = req_url or resp_url
            raise RuntimeError(
                f"{context} failed with HTTP {status_code}. "
                f"request={req_method} {target}. "
                f"headers={header_preview}. "
                f"Body preview: {body_preview!r}"
            )

        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            headers = getattr(resp, "headers", None)
            content_type = None
            if isinstance(headers, dict):
                content_type = headers.get("Content-Type") or headers.get("content-type")

            body_preview = ""
            try:
                body_preview = (getattr(resp, "text", "") or "")[:400]
            except Exception:
                body_preview = ""

            target = req_url or resp_url

            raise RuntimeError(
                f"{context} returned a non-JSON/empty response "
                f"(status={status_code}, content-type={content_type}, request={req_method} {target}). "
                f"Body preview: {body_preview!r}"
            ) from exc

    def _normalize_interval(self, timeframe: str) -> str:
        """Normalize human-friendly timeframes to SDK interval strings."""

        tf = str(timeframe or "").strip().lower()
        mapping = {
            "1m": "ONE_MINUTE",
            "1min": "ONE_MINUTE",
            "one_minute": "ONE_MINUTE",
            "3m": "THREE_MINUTE",
            "3min": "THREE_MINUTE",
            "five_minute": "FIVE_MINUTE",
            "5m": "FIVE_MINUTE",
            "5min": "FIVE_MINUTE",
            "10m": "TEN_MINUTE",
            "10min": "TEN_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "15min": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE",
            "30min": "THIRTY_MINUTE",
            "60m": "ONE_HOUR",
            "1h": "ONE_HOUR",
            "one_hour": "ONE_HOUR",
            "1d": "ONE_DAY",
            "one_day": "ONE_DAY",
        }
        return mapping.get(tf, timeframe)

    def _interval_candidates(self, interval: str) -> List[str]:
        """Return interval candidates limited to broker-accepted constants.

        The Type B chart endpoint rejects alternate spellings. The API error
        explicitly lists valid values, so we only send those.
        """

        allowed = _MSTOCK_ALLOWED_INTERVALS

        base = str(interval or "").strip().upper()
        if not base:
            return []
        if base in allowed:
            return [base]

        # If caller passed something like "1m" without normalize, try once.
        normalized = str(self._normalize_interval(base)).strip().upper()
        if normalized in allowed:
            return [normalized]

        # As a last resort, return empty to force an actionable error upstream.
        return []

    def _parse_candles_payload(self, payload: Any, *, limit: int) -> List[Candle]:
        def _find_rows(obj: Any) -> Optional[List[Any]]:
            if isinstance(obj, list):
                return obj
            if not isinstance(obj, dict):
                return None

            # Most common: {"data": [...]}
            data_val = obj.get("data")
            if isinstance(data_val, list):
                return data_val

            # Common alternates:
            # - {"data": {"candles": [...]}}
            # - {"data": {"data": [...]}}
            # - {"result": [...]}
            # - {"candles": [...]}
            for k in ("candles", "result", "rows"):
                v = obj.get(k)
                if isinstance(v, list):
                    return v

            if isinstance(data_val, dict):
                for k in ("candles", "data", "result", "rows"):
                    v = data_val.get(k)
                    if isinstance(v, list):
                        return v
                    # Some APIs return success with candles=None.
                    if k == "candles" and v is None:
                        return []

            return None

        data = _find_rows(payload)
        if not isinstance(data, list):
            # Include a short preview to help adapt parsing without dumping huge payloads.
            keys = None
            if isinstance(payload, dict):
                keys = list(payload.keys())[:20]
            preview = ""
            try:
                preview = repr(payload)[:600]
            except Exception:
                preview = "(unavailable)"
            raise RuntimeError(
                "Unexpected chart response structure (no candle list found). "
                f"keys={keys} preview={preview}"
            )

        candles: List[Candle] = []
        for row in data[-limit:]:
            try:
                # Dict schema
                if isinstance(row, dict):
                    ts_raw = row.get("time") or row.get("datetime") or row.get("timestamp")
                    if isinstance(ts_raw, str):
                        ts = datetime.fromisoformat(ts_raw.replace("/", "-")[:19])
                    else:
                        ts_val = float(ts_raw)
                        if ts_val > 1e12:
                            ts_val = ts_val / 1000.0
                        ts = datetime.fromtimestamp(ts_val)

                    open_val = row.get("open", row.get("o"))
                    high_val = row.get("high", row.get("h"))
                    low_val = row.get("low", row.get("l"))
                    close_val = row.get("close", row.get("c"))
                    vol_val = row.get("volume")

                    candles.append(
                        Candle(
                            time=ts,
                            open=float(open_val),
                            high=float(high_val),
                            low=float(low_val),
                            close=float(close_val),
                            volume=float(vol_val) if vol_val is not None else None,
                        )
                    )
                    continue

                # List/tuple schema: [ts, open, high, low, close, volume?]
                if isinstance(row, (list, tuple)) and len(row) >= 5:
                    ts_raw = row[0]
                    if isinstance(ts_raw, str):
                        ts = datetime.fromisoformat(str(ts_raw).replace("/", "-")[:19])
                    else:
                        ts_val = float(ts_raw)
                        if ts_val > 1e12:
                            ts_val = ts_val / 1000.0
                        ts = datetime.fromtimestamp(ts_val)
                    vol = None
                    if len(row) >= 6:
                        try:
                            vol = float(row[5])
                        except Exception:
                            vol = None

                    candles.append(
                        Candle(
                            time=ts,
                            open=float(row[1]),
                            high=float(row[2]),
                            low=float(row[3]),
                            close=float(row[4]),
                            volume=vol,
                        )
                    )
                    continue
            except Exception:
                continue

        return candles

    def _candles_look_suspicious(self, candles: List[Candle]) -> bool:
        """Return True when candles appear unusable (flat/zero/NaN prices).

        This protects indicator computations (ATR/ADX) from silently getting 0.0
        due to broker payload quirks (e.g., index historical endpoint returning
        zero-valued OHLC rows).
        """

        if not candles:
            return True

        sample = candles[-min(len(candles), 50) :]

        max_px = -1.0
        max_range = 0.0
        for c in sample:
            try:
                o = float(c.open)
                h = float(c.high)
                l = float(c.low)
                cl = float(c.close)
            except Exception:
                return True

            if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(cl)):
                return True

            max_px = max(max_px, o, h, l, cl)
            try:
                r = float(h) - float(l)
            except Exception:
                r = 0.0
            if r > max_range:
                max_range = r

        # Reject all-zero or non-positive price series.
        if max_px <= 0.0:
            return True

        # Reject completely flat candles (no high-low range) across the recent window.
        if max_range <= 0.0:
            return True

        # Extra guardrail: even if there was some range earlier in the sample,
        # treat the series as suspicious when the *recent* candles are totally flat.
        # This prevents indicator windows (ATR/ADX) from collapsing to 0.0.
        try:
            tail_n = min(len(candles), 25)
            if tail_n >= 6:
                tail = candles[-tail_n:]
                max_tr = 0.0
                # True Range needs a previous close; anchor with the candle just before tail.
                prev_close = None
                if len(candles) > tail_n:
                    try:
                        prev_close = float(candles[-tail_n - 1].close)
                    except Exception:
                        prev_close = None

                for row in tail:
                    try:
                        h = float(row.high)
                        l = float(row.low)
                        cl = float(row.close)
                    except Exception:
                        return True

                    if prev_close is None:
                        tr = max(float(h) - float(l), 0.0)
                    else:
                        tr = max(float(h) - float(l), abs(float(h) - float(prev_close)), abs(float(l) - float(prev_close)))
                    if tr > max_tr:
                        max_tr = tr
                    prev_close = cl

                if max_tr <= 0.0:
                    return True
        except Exception:
            # If this heuristic fails, fall back to the earlier checks.
            pass

        return False

    def _interval_minutes(self, interval: str) -> Optional[int]:
        """Return interval length in minutes for intraday constants.

        Returns None when the interval isn't an intraday minute-based constant.
        """

        v = str(interval or "").strip().upper()
        mapping = {
            "ONE_MINUTE": 1,
            "THREE_MINUTE": 3,
            "FIVE_MINUTE": 5,
            "TEN_MINUTE": 10,
            "FIFTEEN_MINUTE": 15,
            "THIRTY_MINUTE": 30,
            "ONE_HOUR": 60,
        }
        return mapping.get(v)

    def _historical_lookback_days(self, *, interval: str, limit: int, default_days: int) -> int:
        """Return a safe historical lookback window for broker candle limits.

        The broker rejects requests above 1000 candles. Historical requests are
        date-range based, so we derive a conservative day window from the
        requested interval/limit and clamp it below that ceiling.
        """

        try:
            requested_limit = max(1, int(limit))
        except Exception:
            requested_limit = 1

        try:
            configured_days = max(1, int(default_days))
        except Exception:
            configured_days = 1

        interval_key = str(interval or "").strip().upper()
        interval_minutes = self._interval_minutes(interval_key)

        max_broker_candles = 950
        try:
            max_broker_candles = int(os.getenv("MSTOCK_HISTORICAL_MAX_CANDLES", str(max_broker_candles)))
        except Exception:
            max_broker_candles = 950
        max_broker_candles = max(100, min(max_broker_candles, 1000))

        target_candles = min(requested_limit + 50, max_broker_candles)

        if interval_key == "ONE_DAY":
            return min(configured_days, max(1, target_candles))

        if interval_minutes is None or interval_minutes <= 0:
            return configured_days

        candles_per_day = max(1.0, (24.0 * 60.0) / float(interval_minutes))
        safe_days = int(math.ceil(float(target_candles) / candles_per_day))

        if interval_key == "ONE_MINUTE":
            safe_days = min(safe_days, max(1, configured_days))
        else:
            safe_days = min(safe_days, configured_days)

        return max(1, int(safe_days))

    def _resample_candles(self, candles: List[Candle], *, target_minutes: int) -> List[Candle]:
        """Resample smaller-minute candles into larger-minute candles.

        Assumes `candles` are intraday candles for the same symbol/day.
        """

        try:
            m = int(target_minutes)
        except Exception:
            return candles
        if m <= 1:
            return candles
        if not candles:
            return []

        # Sort by time ascending to ensure stable OHLC aggregation.
        src = sorted(candles, key=lambda c: c.time)

        out: List[Candle] = []
        bucket_start: Optional[datetime] = None
        o = h = l = c = None
        vol_sum: Optional[float] = None

        for row in src:
            t = row.time
            # Floor to bucket boundary.
            t0 = t.replace(second=0, microsecond=0)
            delta_min = int(t0.minute % m)
            b = t0 - timedelta(minutes=delta_min)

            if bucket_start is None or b != bucket_start:
                # Flush previous bucket.
                if bucket_start is not None and o is not None and h is not None and l is not None and c is not None:
                    out.append(Candle(time=bucket_start, open=float(o), high=float(h), low=float(l), close=float(c), volume=vol_sum))

                bucket_start = b
                o = row.open
                h = row.high
                l = row.low
                c = row.close
                vol_sum = None
                if row.volume is not None:
                    try:
                        vol_sum = float(row.volume)
                    except Exception:
                        vol_sum = None
            else:
                # Aggregate into current bucket.
                if h is None or row.high > h:
                    h = row.high
                if l is None or row.low < l:
                    l = row.low
                c = row.close
                if row.volume is not None:
                    try:
                        v = float(row.volume)
                    except Exception:
                        v = None
                    if v is not None:
                        if vol_sum is None:
                            vol_sum = v
                        else:
                            vol_sum += v

        # Flush last bucket.
        if bucket_start is not None and o is not None and h is not None and l is not None and c is not None:
            out.append(Candle(time=bucket_start, open=float(o), high=float(h), low=float(l), close=float(c), volume=vol_sum))

        return out

    # ---- Session / auth ----

    def login(self, *, interactive: bool = True) -> None:
        """Ensure the underlying SDK has a valid access token.

        If ``MSTOCK_ACCESS_TOKEN`` was already set in the environment,
        the wrapped ``MConnectB`` instance will pick it up at
        initialisation and this method simply returns.

        Otherwise, it falls back to the interactive helper
        :func:`src.auth.generate_access_token_via_sms_otp`, which
        guides you through username + password + SMS OTP and sets the
        resulting token on this client instance.
        """

        if self._raw.access_token:
            return

        if not interactive:
            raise RuntimeError(
                "MSTOCK_ACCESS_TOKEN is not set. Generate an access token via "
                "`python -m src.auth` (OTP flow) or the UI, then set it in the current shell."
            )

        # Lazy import to avoid circular dependencies at module load time.
        try:
            from auth import generate_access_token_via_sms_otp
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Authentication helper not available. Ensure mStock-"  # noqa: E501
                "TradingApi-B is installed and src/auth.py exists."
            ) from exc

        access_token = generate_access_token_via_sms_otp()
        # Cache on the underlying SDK instance and current process env
        # so subsequent calls and other components can reuse it.
        self._raw.access_token = access_token
        os.environ["MSTOCK_ACCESS_TOKEN"] = access_token

    # ---- Market data ----

    def get_ltp(self, symbol: str) -> float:
        """Return last traded price (LTP) for given symbol.

        ``symbol`` should be in the format expected by the SDK's
        "market LTP" endpoint, for example ``"NSE:ACC"``.
        """
        # Type-B SDK exposes get_market_quote(mode, exchangeTokens).
        # The quote endpoint requires numeric instrument tokens.
        exch, token = self._resolve_token_for_quote(symbol)
        if not (exch and token):
            raise RuntimeError(
                "Unable to resolve token for LTP. Provide a numeric token, an EXCH:TRADINGSYMBOL, "
                "or ensure ScripMaster/instruments are available for resolution. "
                f"symbol={symbol!r}"
            )

        quote_payloads = [{exch: [token]}]
        last_quote_error: Exception | None = None

        def _as_float(v: object) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        def _row_ltp(row: dict) -> Optional[float]:
            if not isinstance(row, dict):
                return None
            # Use only true last-traded fields; avoid close/candle fields for live MTM.
            for k in (
                "ltp",
                "LTP",
                "lastPrice",
                "last_price",
                "lastTradedPrice",
                "LastTradedPrice",
                "last_traded_price",
            ):
                px = _as_float(row.get(k))
                if px is not None:
                    return px
            return None

        def _row_token(row: dict) -> str:
            for k in ("token", "instrumentToken", "instrumenttoken", "symbolToken", "symbol_token"):
                v = row.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
            return ""

        def _row_exch(row: dict) -> str:
            for k in ("exchange", "exchangeSegment", "exch_seg", "segment", "e"):
                v = row.get(k)
                if v is None:
                    continue
                s = str(v).strip().upper()
                if s:
                    return s
            return ""

        candidates: list[tuple[int, float]] = []

        def _add_row(row: object) -> None:
            if not isinstance(row, dict):
                return
            ltp = _row_ltp(row)
            if ltp is None:
                return

            score = 0
            row_tok = _row_token(row)
            if row_tok and row_tok == str(token):
                score += 4
            row_ex = _row_exch(row)
            if row_ex and (row_ex == str(exch).upper() or row_ex == str(seg_id).upper()):
                score += 2
            candidates.append((score, float(ltp)))

        def _walk_rows(obj: object) -> None:
            if isinstance(obj, dict):
                _add_row(obj)
                for child in obj.values():
                    if isinstance(child, (dict, list, tuple)):
                        _walk_rows(child)
            elif isinstance(obj, (list, tuple)):
                for child in obj:
                    if isinstance(child, (dict, list, tuple)):
                        _walk_rows(child)

        # Defensive extraction across common broker/SDK response shapes.
        # Docs show: {"data": {"fetched": [{"ltp": ...}]}}
        for exchange_tokens in quote_payloads:
            try:
                response = self._raw.get_market_quote("LTP", exchange_tokens)
                data = self._safe_json(response, context="get_market_quote")
                _walk_rows(data)
                if candidates:
                    break
            except Exception as exc:
                last_quote_error = exc
                continue

        if candidates:
            positive = [c for c in candidates if float(c[1]) > 0.0]
            if positive:
                positive.sort(key=lambda c: c[0], reverse=True)
                return float(positive[0][1])

            # If parsed values are all non-positive (often stale/invalid),
            # try FULL quote and use bid/ask mid or full-quote LTP.
            try:
                quote_key = f"{exch}:{token}" if exch else token
                bid, ask, ltp_full = self.get_bid_ask(quote_key, exchange_hint=exch)
                if bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0:
                    return (float(bid) + float(ask)) / 2.0
                if ltp_full is not None and float(ltp_full) > 0:
                    return float(ltp_full)
            except Exception:
                pass

            candidates.sort(key=lambda c: c[0], reverse=True)
            return float(candidates[0][1])

        # LTP responses vary by SDK/account/segment. If the lightweight LTP mode
        # returned an unfamiliar shape, try FULL and use either FULL LTP or
        # bid/ask midpoint for live MTM instead of failing silently upstream.
        try:
            quote_key = f"{exch}:{token}" if exch else token
            bid, ask, ltp_full = self.get_bid_ask(quote_key, exchange_hint=exch)
            if ltp_full is not None and float(ltp_full) > 0:
                return float(ltp_full)
            if bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0:
                return (float(bid) + float(ask)) / 2.0
        except Exception:
            pass

        err_suffix = f" Last quote error: {last_quote_error}" if last_quote_error is not None else ""
        raise RuntimeError(
            "Unable to parse LTP from quote response. Inspect `get_market_quote().json()` and update "
            f"MStockTypeBClient.get_ltp accordingly.{err_suffix}"
        )

    def get_bid_ask(
        self,
        symbol: str,
        *,
        exchange_hint: Optional[str] = None,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (best_bid, best_ask, ltp) when available; otherwise (None, None, None)."""
        try:
            exch, token = self._resolve_token_for_quote(symbol, exchange_hint=exchange_hint)
            if not (exch and token):
                return (None, None, None)

            quote_payloads = [{exch: [token]}]

            def _as_float(v: object) -> Optional[float]:
                try:
                    if v is None:
                        return None
                    return float(v)
                except Exception:
                    return None

            def _row_token(row: dict) -> str:
                if not isinstance(row, dict):
                    return ""
                for k in (
                    "token",
                    "instrumentToken",
                    "instrumenttoken",
                    "symbolToken",
                    "symboltoken",
                    "symbol_token",
                    "instrument_token",
                ):
                    v = row.get(k)
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s:
                        return s
                return ""

            def _row_exch(row: dict) -> str:
                if not isinstance(row, dict):
                    return ""
                for k in ("exchange", "exchangeSegment", "exch_seg", "segment", "e"):
                    v = row.get(k)
                    if v is None:
                        continue
                    s = str(v).strip().upper()
                    if s:
                        return s
                return ""

            def _extract_row(row: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
                if not isinstance(row, dict):
                    return (None, None, None)

                # Depth-based extraction when available.
                depth = row.get("depth") or row.get("marketDepth") or row.get("depthData") or row.get("mDepth")
                if isinstance(depth, dict):
                    buy = depth.get("buy") or depth.get("bids") or depth.get("buyDepth")
                    sell = depth.get("sell") or depth.get("asks") or depth.get("sellDepth")
                    if isinstance(buy, list) and buy:
                        b0 = buy[0]
                        if isinstance(b0, dict):
                            bid = _as_float(b0.get("price") or b0.get("bidPrice") or b0.get("rate"))
                            if bid is None:
                                bid = _as_float(b0.get("bp") or b0.get("bestBid") or b0.get("bestBidPrice"))
                        else:
                            bid = _as_float(b0)
                    else:
                        bid = None
                    if isinstance(sell, list) and sell:
                        s0 = sell[0]
                        if isinstance(s0, dict):
                            ask = _as_float(s0.get("price") or s0.get("askPrice") or s0.get("rate"))
                            if ask is None:
                                ask = _as_float(s0.get("sp") or s0.get("bestAsk") or s0.get("bestAskPrice"))
                        else:
                            ask = _as_float(s0)
                    else:
                        ask = None
                elif isinstance(depth, list) and depth:
                    # Some APIs return a list of depth rows.
                    bid = None
                    ask = None
                    for drow in depth:
                        if not isinstance(drow, dict):
                            continue
                        side = str(drow.get("side") or drow.get("type") or "").strip().upper()
                        px = _as_float(drow.get("price") or drow.get("rate") or drow.get("bp") or drow.get("sp"))
                        if px is None:
                            continue
                        if side in {"B", "BUY", "BID"} and bid is None:
                            bid = px
                        if side in {"S", "SELL", "ASK"} and ask is None:
                            ask = px
                    # If side isn't present, just treat first two as bid/ask.
                    if (bid is None or ask is None) and len(depth) >= 2:
                        try:
                            if bid is None:
                                bid = _as_float(depth[0].get("price") or depth[0].get("rate"))  # type: ignore[index]
                            if ask is None:
                                ask = _as_float(depth[1].get("price") or depth[1].get("rate"))  # type: ignore[index]
                        except Exception:
                            pass
                else:
                    bid = None
                    ask = None

                if bid is None:
                    bid = _as_float(
                        row.get("bestBid")
                        or row.get("bestBidPrice")
                        or row.get("bidPrice")
                        or row.get("bid")
                        or row.get("buyPrice")
                        or row.get("bp")
                    )
                if ask is None:
                    ask = _as_float(
                        row.get("bestAsk")
                        or row.get("bestAskPrice")
                        or row.get("askPrice")
                        or row.get("ask")
                        or row.get("sellPrice")
                        or row.get("sp")
                    )

                ltp = _as_float(
                    row.get("ltp")
                    or row.get("lastPrice")
                    or row.get("lastTradedPrice")
                    or row.get("last_traded_price")
                )

                # Additional fallbacks some brokers expose.
                if ltp is None:
                    ltp = _as_float(row.get("LTP") or row.get("LastTradedPrice") or row.get("last_price"))
                return bid, ask, ltp

            # Similar to get_ltp, the response can be nested. Instead of returning
            # the first row, scan all candidates and prefer exact token/exchange matches.
            candidates: list[tuple[int, Optional[float], Optional[float], Optional[float]]] = []

            def _score_row(row: dict, bid: Optional[float], ask: Optional[float], ltp: Optional[float]) -> int:
                score = 0
                try:
                    row_tok = _row_token(row)
                    if row_tok and row_tok == str(token):
                        score += 5
                except Exception:
                    pass
                try:
                    row_ex = _row_exch(row)
                    if row_ex and (row_ex == str(exch).upper() or row_ex == str(seg_id).upper()):
                        score += 2
                except Exception:
                    pass
                if bid is not None:
                    score += 2
                if ask is not None:
                    score += 2
                if ltp is not None:
                    score += 1
                return score

            def _add_candidate(obj: object) -> None:
                if not isinstance(obj, dict):
                    return
                b, a, l = _extract_row(obj)
                if b is None and a is None and l is None:
                    return
                candidates.append((_score_row(obj, b, a, l), b, a, l))

            def _walk_rows(obj: object) -> None:
                if isinstance(obj, dict):
                    _add_candidate(obj)
                    for child in obj.values():
                        if isinstance(child, (dict, list, tuple)):
                            _walk_rows(child)
                elif isinstance(obj, (list, tuple)):
                    for child in obj:
                        if isinstance(child, (dict, list, tuple)):
                            _walk_rows(child)

            for exchange_tokens in quote_payloads:
                try:
                    response = self._raw.get_market_quote("FULL", exchange_tokens)
                    data = self._safe_json(response, context="get_market_quote_full")
                    _walk_rows(data)
                    if candidates:
                        break
                except Exception:
                    continue

            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                _, bid, ask, ltp = candidates[0]
                return bid, ask, ltp
        except Exception:
            return (None, None, None)

        return (None, None, None)

    def get_option_chain(self, underlying: str) -> List[Dict[str, Any]]:
        """Return option chain for the given underlying using SDK endpoints.

        This implementation uses
        ``MConnectB.get_option_chain_data(exchange_id, expiry, token)``.

        To keep the strategy generic, the required parameters are read
        from environment variables so you can configure them without
        changing code:

        - ``MSTOCK_OPTION_EXCHANGE_ID`` – e.g. "5" for the desired segment
        - ``MSTOCK_OPTION_EXPIRY`` – expiry identifier as required by API
        - ``MSTOCK_OPTION_TOKEN`` – underlying token for which chain is fetched

        The raw JSON is normalised into a list of dictionaries with at
        least: ``symbol``, ``strike``, ``option_type`` ("CE"/"PE"),
        and ``iv`` (if present). All original fields remain available
        under the ``raw`` key for advanced use.
        """

        # CSV/ScripMaster primary + API fallback (configurable).
        # - MSTOCK_SCRIPMASTER_PATH: path to ScripMaster CSV
        # - MSTOCK_USE_CSV_ONLY: if true, never call option-chain API
        # - MSTOCK_USE_CHAIN_FALLBACK: if true, use API only if CSV yields no data
        use_csv_only = self._bool_env("MSTOCK_USE_CSV_ONLY", False)
        use_chain_fallback = self._bool_env("MSTOCK_USE_CHAIN_FALLBACK", True)

        csv_chain = self._get_option_chain_from_csv(underlying)
        if csv_chain:
            return csv_chain

        if use_csv_only:
            raise RuntimeError(
                "CSV/ScripMaster mode is enabled but no option rows were found. "
                "Check MSTOCK_SCRIPMASTER_PATH and CSV column names."
            )

        if not use_chain_fallback:
            raise RuntimeError(
                "No CSV option data found and option-chain fallback is disabled. "
                "Set MSTOCK_USE_CHAIN_FALLBACK=1 to enable fallback."
            )

        return self._get_option_chain_from_api()

    def fetch_index_candles(
        self,
        instrument_token: str,
        *,
        exchange: str = "NSE",
        limit: int = 100,
        timeframe: Optional[str] = None,
        force_historical_only: bool = False,
    ) -> tuple[Optional[List[Candle]], Optional[str]]:
        """Fetch candles for an index/underlying token.

        m.Stock can return empty ONE_MINUTE index candles even during market hours.
        We try ONE_MINUTE then FIVE_MINUTE and return the first non-empty set.

        Returns (candles, interval) or (None, None) if no usable data yet.
        When ``force_historical_only`` is True, intraday endpoint calls are skipped.
        """

        token = str(instrument_token or "").strip()
        if not (token.isdigit() and int(token) > 0):
            return None, None

        ex = str(exchange or "").strip().upper() or "NSE"

        # Consolidate candidate intervals. If a specific timeframe was requested,
        # prioritize it at the start of the list.
        intervals_to_try = list(FALLBACK_INTERVALS)
        if timeframe:
            requested_tf = self._normalize_interval(timeframe)
            if requested_tf and requested_tf not in intervals_to_try:
                intervals_to_try.insert(0, requested_tf)
            elif requested_tf and requested_tf in intervals_to_try:
                # Move requested to front
                intervals_to_try.remove(requested_tf)
                intervals_to_try.insert(0, requested_tf)

        use_intraday_chart = self._bool_env("MSTOCK_USE_INTRADAY_CHART", False)
        if force_historical_only:
            use_intraday_chart = False
        # Intraday-only should be an explicit choice. If intraday is enabled but flaky/empty,
        # we allow falling back to historical candles unless MSTOCK_INTRADAY_ONLY=1.
        intraday_only = self._bool_env("MSTOCK_INTRADAY_ONLY", False)
        # Yahoo is a *fallback* data source and should never be used implicitly.
        # Opt-in only via MSTOCK_USE_YAHOO_FALLBACK=1.
        allow_yahoo_fallback = self._bool_env("MSTOCK_USE_YAHOO_FALLBACK", False)
        # When intraday m.Stock candles are enabled, keep candle sourcing strictly
        # within m.Stock endpoints (intraday + historical). This avoids mixing
        # Yahoo candles into indicator calculations.
        if use_intraday_chart and allow_yahoo_fallback:
            allow_yahoo_fallback = False
            print("[CANDLES] Intraday m.Stock candles enabled; disabling Yahoo fallback to avoid mixed candle sources.")

        # If intraday is cooling down, either return nothing (strict mode) or
        # fall back to historical/yahoo (default) so the bot can keep trading.
        if use_intraday_chart and not self._intraday_available():
            if intraday_only:
                self._log_intraday_failure(
                    key=f"cooldown:{token}",
                    msg="[INTRADAY] Cooling down after failures; skipping candles (intraday-only mode).",
                )
                return None, None

        # Optional: intraday chart endpoint (current day only).
        intraday_candles: List[Candle] = []
        intraday_interval_used: Optional[str] = None
        if use_intraday_chart and self._intraday_available():
            seg = os.getenv(
                "MSTOCK_INTRADAY_EXCHANGE",
                ex,
            )
            # intraday_candles initialized in outer scope
            for interval in intervals_to_try:
                try:
                    intraday_candles = self._fetch_intraday_chart(
                        exchange_segment=str(seg),
                        symboltoken=token,
                        interval=str(interval),
                        limit=int(limit),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._note_intraday_failure(exc)
                    self._log_intraday_failure(
                        key=f"{token}:{interval}",
                        msg=f"[INTRADAY] Failed token={token} interval={interval}: {exc}",
                    )
                    intraday_candles = []

                if intraday_candles:
                    self._note_intraday_success()
                    intraday_interval_used = str(interval)
                    print(f"[INTRADAY] Got {len(intraday_candles)} candles using {interval}")
                    # User request: when intraday endpoint is enabled, do not mix in
                    # historical candles. Return best-effort intraday immediately.
                    return intraday_candles, interval

            # If intraday is enabled but returned nothing, either stop here
            # (strict) or fall back to historical/yahoo (default).
            if intraday_only:
                return None, None

        base_lookback_days = 5
        try:
            base_lookback_days = int(os.getenv("MSTOCK_CANDLE_LOOKBACK_DAYS", str(base_lookback_days)))
        except Exception:
            base_lookback_days = 5

        # ONE_MINUTE easily exceeds the broker's 1000-candle cap.
        one_minute_days = 2
        try:
            one_minute_days = int(os.getenv("MSTOCK_CANDLE_LOOKBACK_DAYS_1M", str(one_minute_days)))
        except Exception:
            one_minute_days = 2

        now = datetime.now()

        # Brokers vary on date formats accepted by historical endpoints.
        date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]

        last_exc: Optional[Exception] = None
        saw_security_id_missing = False
        saw_suspicious_candles = False
        printed_token_meta = False
        for interval in intervals_to_try:
            print(f"[CANDLES] Trying interval={interval}")

            lb_days = int(base_lookback_days)
            if interval == "ONE_MINUTE":
                lb_days = int(one_minute_days)
            lb_days = self._historical_lookback_days(
                interval=str(interval),
                limit=int(limit),
                default_days=int(lb_days),
            )
            from_dt = now - timedelta(days=max(1, int(lb_days)))

            # Official TypeB SDK examples use exchange strings like "NSE"/"NFO".
            # Some callers may still supply numeric ids; map those back when possible.
            hist_exchange = self._exchange_id_to_str(ex) if ex.isdigit() else ex

            for fmt in date_formats:
                from_s = from_dt.strftime(fmt)
                to_s = now.strftime(fmt)

                payload = {
                    "exchange": hist_exchange,
                    "instrumentToken": token,
                    "interval": interval,
                    "fromDate": from_s,
                    "toDate": to_s,
                }

                print(
                    f"[CANDLES] Using historical endpoint | "
                    f"exchange={payload['exchange']!r} instrument_token={payload['instrumentToken']!r} interval={payload['interval']!r} "
                    f"from={payload['fromDate']!r} to={payload['toDate']!r}"
                )

                try:
                    resp = self._raw.get_historical_chart(
                        payload["exchange"],
                        payload["instrumentToken"],
                        payload["interval"],
                        payload["fromDate"],
                        payload["toDate"],
                    )
                    data = self._safe_json(resp, context="get_historical_chart")
                    candles = self._parse_candles_payload(data, limit=limit)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    msg = str(exc)
                    if "security id not found" in msg.lower() or "ia400" in msg.lower():
                        saw_security_id_missing = True
                        if not printed_token_meta:
                            printed_token_meta = True
                            try:
                                row = self._get_instrument_by_token().get(token)
                            except Exception:
                                row = None
                            if isinstance(row, dict):
                                name = str(
                                    row.get("tradingsymbol")
                                    or row.get("tradingSymbol")
                                    or row.get("symbol")
                                    or row.get("name")
                                    or row.get("tokenName")
                                    or ""
                                ).strip()
                                row_exch = self._extract_exchange(row)
                                it = str(
                                    row.get("instrumenttype")
                                    or row.get("instrumentType")
                                    or row.get("segment")
                                    or row.get("series")
                                    or ""
                                ).strip()
                                print(
                                    f"[CANDLES] Token meta: token={token} exch_req={hist_exchange!r} exch_row={row_exch!r} type={it!r} name={name!r}"
                                )
                    # Don't spam: strategy loop already logs every cycle.
                    print(f"[CANDLES] Error token={token} interval={interval}: {exc}")
                    candles = []
                    # If this is the broker's 1000-candle cap, date formats won't help.
                    if "maximum limit of 1000 candles" in str(exc).lower():
                        break
                    # If the broker is strict about date formats, try the next one.
                    continue

                # Some broker deployments return a candle list with zero/flat OHLCs
                # (commonly for indices on historical endpoints). Treat these as unusable
                # and try other intervals (or intraday fallback below).
                try:
                    if candles and self._candles_look_suspicious(candles):
                        saw_suspicious_candles = True
                        print(
                            "[CANDLES] Suspicious candles received (flat/zero OHLC). "
                            f"token={token} exch={hist_exchange!r} interval={interval}. Retrying other intervals."
                        )
                        candles = []
                        break
                except Exception:
                    pass

                if candles:
                    print(f"[CANDLES] Got {len(candles)} candles using {interval}")
                    if len(candles) >= min(limit, 20):
                        return candles, interval

                    print(
                        f"[CANDLES] Insufficient historical data ({len(candles)} < {min(limit, 20)}). "
                        f"Yahoo fallback is {'enabled' if allow_yahoo_fallback else 'disabled'}; "
                        "keeping best-effort candles."
                    )
                    # Treat this as our "best effort" fallback if Yahoo also fails.
                    if len(candles) > len(intraday_candles):
                        intraday_candles = candles
                        intraday_interval_used = str(interval)

                    # Stop trying other date formats/intervals; proceed to Yahoo only if allowed.
                    break

                # Call succeeded but returned empty: don't churn through other date formats.
                break

        if last_exc is not None:
            # Helpful hint: empty candles often means wrong token (e.g. ETF instead of index).
            print(
                "[CANDLES] No usable candles yet. "
                "If this persists, verify MSTOCK_UNDERLYING_TOKEN is correct for the selected exchange."
            )

        # If historical fails with IA400/security-id-not-found, try intraday once even
        # when MSTOCK_USE_INTRADAY_CHART is disabled. This helps for indices where
        # the historical endpoint may reject a token that the intraday endpoint accepts.
        if (not intraday_candles) and saw_security_id_missing and (not use_intraday_chart) and self._intraday_available():
            seg = os.getenv("MSTOCK_INTRADAY_EXCHANGE", ex)
            for interval in intervals_to_try:
                try:
                    intraday_candles = self._fetch_intraday_chart(
                        exchange_segment=str(seg),
                        symboltoken=token,
                        interval=str(interval),
                        limit=int(limit),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._note_intraday_failure(exc)
                    print(f"[INTRADAY] One-shot retry failed token={token} interval={interval}: {exc}")
                    intraday_candles = []
                if intraday_candles:
                    self._note_intraday_success()
                    intraday_interval_used = str(interval)
                    print(f"[INTRADAY] One-shot retry got {len(intraday_candles)} candles using {interval}")
                    return intraday_candles, str(interval)

        # If historical returned *suspicious* (flat/zero) candles, try intraday once even
        # when MSTOCK_USE_INTRADAY_CHART is disabled. This keeps the bot from getting
        # stuck with ATR=0 blocks on otherwise-valid sessions.
        if (
            (not intraday_candles)
            and saw_suspicious_candles
            and (not use_intraday_chart)
            and (not force_historical_only)
            and self._intraday_available()
        ):
            seg = os.getenv("MSTOCK_INTRADAY_EXCHANGE", ex)
            for interval in intervals_to_try:
                try:
                    intraday_candles = self._fetch_intraday_chart(
                        exchange_segment=str(seg),
                        symboltoken=token,
                        interval=str(interval),
                        limit=int(limit),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._note_intraday_failure(exc)
                    print(f"[INTRADAY] Suspicious-historical retry failed token={token} interval={interval}: {exc}")
                    intraday_candles = []
                if intraday_candles:
                    self._note_intraday_success()
                    intraday_interval_used = str(interval)
                    print(f"[INTRADAY] Suspicious-historical retry got {len(intraday_candles)} candles using {interval}")
                    return intraday_candles, str(interval)
        
        # ---- YAHOO FINANCE FALLBACK (opt-in) ----
        if allow_yahoo_fallback:
            if intraday_candles:
                self._log_candle_event(
                    key=f"yahoo_fallback_insufficient:{token}",
                    msg=(
                        f"[CANDLES] m.Stock candles insufficient (count={len(intraday_candles)}). "
                        "Attempting Yahoo Finance fallback..."
                    ),
                )
            else:
                self._log_candle_event(
                    key=f"yahoo_fallback_unavailable:{token}",
                    msg="[CANDLES] m.Stock candles unavailable. Attempting Yahoo Finance fallback...",
                )
            try:
                underlying_name = os.getenv("MSTOCK_UNDERLYING", "").strip().upper() or "NIFTY"

                # If the token passed matches a known index token, override.
                if str(token) == "26000":
                    underlying_name = "NIFTY"
                elif str(token) == "26009":
                    underlying_name = "BANKNIFTY"
                elif str(token) == "26037":
                    underlying_name = "FINNIFTY"
                elif str(token) == "26074":
                    underlying_name = "MIDCPNIFTY"

                yf_sym = yahoo_data.yahoo_symbol_for_underlying(underlying_name)
                yf_candles = yahoo_data.fetch_yahoo_candles(
                    yf_sym,
                    interval=str(timeframe or "1m"),
                    limit=limit,
                )
                if yf_candles:
                    print(f"[YAHOO] Success: Fetched {len(yf_candles)} candles for {yf_sym}")
                    return yf_candles, str(timeframe or "1m")
                else:
                    self._log_candle_event(
                        key=f"yahoo_empty:{token}:{yf_sym}",
                        msg=f"[YAHOO] No data found for {yf_sym}",
                    )

            except Exception as exc:
                self._log_candle_event(
                    key=f"yahoo_fail:{token}",
                    msg=f"[YAHOO] Fallback failed: {exc}",
                )
        else:
            if not intraday_candles:
                self._log_candle_event(
                    key=f"yahoo_disabled:{token}",
                    msg="[CANDLES] Yahoo fallback disabled (set MSTOCK_USE_YAHOO_FALLBACK=1 to enable).",
                )
        # ----------------------------------------

        # Fallback: if we had intraday candles but they were insufficient, return them anyway
        # rather than returning nothing.
        if intraday_candles:
            used = intraday_interval_used or (str(intervals_to_try[0]) if intervals_to_try else None)
            self._log_candle_event(
                key=f"best_effort:{token}:{used}",
                msg=(
                    f"[CANDLES] Returning best-effort m.Stock candles (count={len(intraday_candles)}) "
                    f"despite being < {min(limit, 20)}."
                ),
            )
            return intraday_candles, used

        return None, None

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> List[Candle]:
        """Return historical candles for the given symbol.

        The ``timeframe`` should match the interval strings used by
        the SDK (e.g. ``"ONE_MINUTE"``, ``"ONE_HOUR"``). This method
        is hard-locked to the historical endpoint only.
        """
        # --- CACHING LAYER ---
        # Strategy loops every 1s. fetching intraday/history every 1s will cause rate-limits.
        # Cache for 45 seconds (approx new candle time).
        cache_key = f"{symbol}|{timeframe}"
        now = datetime.now()

        if not hasattr(self, "_candle_cache"):
            self._candle_cache: Dict[str, Any] = {}

        cached = self._candle_cache.get(cache_key)
        cached_ts = None
        cached_candles = None
        cached_source = None
        if cached:
            # Back-compat: older cache entries may be (ts, candles)
            try:
                if isinstance(cached, tuple) and len(cached) == 3:
                    cached_ts, cached_candles, cached_source = cached
                elif isinstance(cached, tuple) and len(cached) == 2:
                    cached_ts, cached_candles = cached
                    cached_source = "unknown"
            except Exception:
                cached_ts, cached_candles, cached_source = None, None, None

        # --- FETCH LOGIC ---
        result_candles = []

        def _intraday_cache_fallback() -> Optional[List[Candle]]:
            """Return a best-effort cached intraday series when the intraday endpoint is flaky.

            In intraday-only mode, the Type-B intraday chart API is known to occasionally
            return empty/500. Returning a short-lived cached intraday series is still
            consistent with 'intraday-only' semantics (source remains intraday) and
            avoids blocking strategy warmup.
            """

            try:
                if cached_ts is None or cached_candles is None:
                    return None
                if str(cached_source or "").lower() != "intraday":
                    return None
                max_stale = int(os.getenv("MSTOCK_INTRADAY_CACHE_FALLBACK_SEC", "300"))
                if max_stale <= 0:
                    return None
                age = float((now - cached_ts).total_seconds())
                if age <= float(max_stale):
                    return cached_candles  # type: ignore[return-value]
            except Exception:
                return None
            return None
        
        # Helper to effectively save consistency
        def save_and_return(candles_out, *, source: str):
            # Strategy loops every 1s. Cache to avoid rate limits.
            # For intraday, cache any non-empty response (even small) so transient
            # empty responses don't block intraday-only mode.
            try:
                src = str(source or "").lower()
            except Exception:
                src = ""
            looks_ok = True
            try:
                if isinstance(candles_out, list) and candles_out:
                    looks_ok = not self._candles_look_suspicious(candles_out)
            except Exception:
                looks_ok = True

            should_cache = looks_ok and ((len(candles_out) > 15) or (src == "intraday" and len(candles_out) > 0))
            if should_cache:
                self._candle_cache[cache_key] = (datetime.now(), candles_out, str(source or ""))
            return candles_out

        # Intraday-by-default (safe): if the user hasn't explicitly set
        # MSTOCK_USE_INTRADAY_CHART, enable intraday when credentials exist.
        # This reduces historical endpoint churn during the live loop.
        use_intraday_chart = self._bool_env("MSTOCK_USE_INTRADAY_CHART", False)
        if os.getenv("MSTOCK_USE_INTRADAY_CHART") is None:
            try:
                api_key_present = bool(str(self.cfg.api_key or "").strip())
                access_token_present = bool(str(os.getenv("MSTOCK_ACCESS_TOKEN", "")).strip())
                if api_key_present and access_token_present:
                    use_intraday_chart = True
            except Exception:
                pass
        # Intraday-only should be an explicit choice. If intraday is enabled but flaky/empty,
        # we allow falling back to historical candles unless MSTOCK_INTRADAY_ONLY=1.
        intraday_only = self._bool_env("MSTOCK_INTRADAY_ONLY", False)
        startup_force_historical = self._should_use_startup_historical_bootstrap(
            symbol,
            intraday_enabled=bool(use_intraday_chart),
        )
        if startup_force_historical:
            use_intraday_chart = False
            intraday_only = False
            if not self._startup_historical_bootstrap_logged:
                print("[CANDLES] Startup warmup: using m.Stock historical endpoint for spot candles.")
                self._startup_historical_bootstrap_logged = True

        # If we have a young cache entry, only return it immediately when:
        # - intraday is disabled, OR
        # - intraday is enabled but currently unavailable/cooling down, OR
        # - the cached entry itself came from intraday.
        # This ensures we switch back to intraday as soon as it recovers.
        try:
            cache_is_young = (
                cached_ts is not None
                and cached_candles is not None
                and (now - cached_ts).total_seconds() < 45
            )
        except Exception:
            cache_is_young = False
        if cache_is_young:
            # In intraday-only mode, never return cached historical candles.
            if intraday_only and use_intraday_chart:
                if str(cached_source or "").lower() == "intraday":
                    return cached_candles  # type: ignore[return-value]
            else:
                if (not use_intraday_chart) or (use_intraday_chart and not self._intraday_available()):
                    if startup_force_historical and cached_candles:
                        self._startup_historical_bootstrap_done = True
                    return cached_candles  # type: ignore[return-value]
                if str(cached_source or "").lower() == "intraday":
                    return cached_candles  # type: ignore[return-value]

        # If intraday is cooling down, either return nothing (strict) or
        # fall back to historical (default) so the bot can keep trading.
        if use_intraday_chart and not self._intraday_available() and intraday_only:
            self._log_intraday_failure(
                key=f"cooldown:get_candles:{symbol}:{timeframe}",
                msg="[INTRADAY] Cooling down after failures; returning no candles (intraday-only mode).",
            )
            fb = _intraday_cache_fallback()
            return fb if fb is not None else []

        # --- Bootstrap + merge support ---
        # When running early in the session, intraday candles can be too few for indicators.
        # To avoid repeatedly fetching full historical candles, we bootstrap a historical
        # base once per (symbol,timeframe) and then keep returning a merged series:
        # historical_base + intraday_today.
        if not hasattr(self, "_candle_bootstrap_base"):
            self._candle_bootstrap_base: Dict[str, List[Candle]] = {}

        def _merge_candles(base: Optional[List[Candle]], intraday: Optional[List[Candle]]) -> List[Candle]:
            by_time: Dict[datetime, Candle] = {}
            if isinstance(base, list):
                for c in base:
                    try:
                        by_time[c.time] = c
                    except Exception:
                        continue
            if isinstance(intraday, list):
                for c in intraday:
                    try:
                        # Prefer intraday candles when timestamps overlap.
                        by_time[c.time] = c
                    except Exception:
                        continue
            out = list(by_time.values())
            try:
                out = sorted(out, key=lambda c: c.time)
            except Exception:
                pass
            try:
                if int(limit) > 0 and len(out) > int(limit):
                    out = out[-int(limit) :]
            except Exception:
                pass
            return out

        def _bootstrap_min_needed() -> int:
            # Default aims to cover common EMA/RSI/ATR warmups.
            # Keep configurable for users with larger indicator windows.
            try:
                v = int(os.getenv("MSTOCK_INTRADAY_BOOTSTRAP_MIN_CANDLES", "60"))
            except Exception:
                v = 60
            if v <= 0:
                v = 60
            return int(v)

        bootstrap_base = self._candle_bootstrap_base.get(cache_key)
        intraday_series_for_merge: Optional[List[Candle]] = None
        need_historical_bootstrap = False

        # Optional: broker intraday chart endpoint (current day only).
        # This can be more reliable than historical for some instruments during market hours.
        if use_intraday_chart and self._intraday_available():
            sym_raw = (symbol or "").strip()
            seg = os.getenv(
                "MSTOCK_INTRADAY_EXCHANGE",
                os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE")),
            )
            seg_norm = str(seg or "").strip().upper() or "NSE"
            seg_id = self._exchange_segment_id(seg_norm)
            api_key = str(self.cfg.api_key or "").strip()
            access_token = str(os.getenv("MSTOCK_ACCESS_TOKEN", "")).strip()

            def _present(value: str) -> str:
                return "set" if str(value or "").strip() else "missing"

            can_try_intraday = bool(api_key and access_token)
            if (not can_try_intraday) and intraday_only:
                try:
                    now_ts = float(time.time())
                    k = f"intraday-missing-auth:{symbol}:{timeframe}"
                    last = float(getattr(self, "_intraday_last_log_by_key", {}).get(k, 0.0) or 0.0)
                    if now_ts - last >= 30.0:
                        getattr(self, "_intraday_last_log_by_key", {})[k] = now_ts
                        env_tok_hint = os.getenv("MSTOCK_UNDERLYING_TOKEN", "").strip() or "n/a"
                        print(
                            "[INTRADAY] Intraday chart requires MSTOCK_API_KEY + MSTOCK_ACCESS_TOKEN; "
                            f"api_key={_present(api_key)} access_token={_present(access_token)} "
                            f"token={env_tok_hint} seg={seg_norm}({seg_id})."
                        )
                except Exception:
                    pass
                fb = _intraday_cache_fallback()
                return fb if fb is not None else []

            # Intraday index candles can be unreliable at ONE_MINUTE for some accounts/symbols.
            # To avoid blocking strategy warmup in intraday-only mode, try a small fallback
            # set of intervals (still using intraday endpoint only).
            requested_interval = str(self._normalize_interval(timeframe)).strip().upper()

            # Intraday chart API can be picky and may return empty for some intervals
            # (commonly for index tokens). To keep the requested timeframe semantics,
            # fall back to fetching smaller base intervals (1m/5m) and resample up.
            desired_min = self._interval_minutes(requested_interval)

            intervals_to_try: List[str] = []
            if requested_interval:
                intervals_to_try.append(requested_interval)

            # Always try base intraday intervals as fallbacks.
            # - For 3m, 1m is the only viable resample source.
            # - For 10/15/30, 5m is often available and resamples cleanly.
            if requested_interval != "ONE_MINUTE":
                intervals_to_try.append("ONE_MINUTE")
            if requested_interval != "FIVE_MINUTE":
                intervals_to_try.append("FIVE_MINUTE")

            # When requesting 1m, keep the existing 5m fallback.
            if requested_interval == "ONE_MINUTE":
                intervals_to_try.append("FIVE_MINUTE")

            # De-dup while preserving order.
            seen_int: set[str] = set()
            intervals_to_try = [i for i in intervals_to_try if i and not (i in seen_int or seen_int.add(i))]

            # Intraday endpoint requires a numeric token. Allow callers to pass a symbol
            # like "NIFTY" and resolve it automatically.
            token_for_intraday: str = ""
            token_source = ""
            env_tok = ""
            if sym_raw.isdigit():
                token_for_intraday = sym_raw
                token_source = "symbol"
            else:
                # Prefer configured underlying token.
                env_tok = os.getenv("MSTOCK_UNDERLYING_TOKEN", "").strip()
                if env_tok.isdigit() and int(env_tok) > 0:
                    token_for_intraday = env_tok
                    token_source = "env"
                else:
                    # Best-effort resolve via instruments master.
                    try:
                        exch_hint = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE"))
                        token_for_intraday = self._resolve_token_from_instruments(sym_raw, str(exch_hint or "NSE"))
                        if token_for_intraday:
                            token_source = "resolve"
                    except Exception as exc:  # noqa: BLE001
                        print(f"[INTRADAY] Token resolve failed for {sym_raw!r}: {exc}")
                        token_for_intraday = ""

            if not token_for_intraday and intraday_only:
                try:
                    now_ts = float(time.time())
                    k = f"intraday-missing-token:{sym_raw}:{seg_norm}"
                    last = float(getattr(self, "_intraday_last_log_by_key", {}).get(k, 0.0) or 0.0)
                    if now_ts - last >= 30.0:
                        getattr(self, "_intraday_last_log_by_key", {})[k] = now_ts
                        print(
                            "[INTRADAY] Unable to resolve intraday token in intraday-only mode; "
                            f"symbol={sym_raw!r} env_token={env_tok or 'n/a'} seg={seg_norm}({seg_id}). "
                            "Set MSTOCK_UNDERLYING_TOKEN or pass a numeric token."
                        )
                except Exception:
                    pass

            if token_for_intraday and can_try_intraday:
                # If the configured token/segment yields empty candles (common for indices),
                # try alternate token+segment candidates from instruments master.
                token_pairs: List[tuple[str, str, str]] = []
                # (exchange_segment, token, source_label)
                token_pairs.append((str(seg_norm), str(token_for_intraday), str(token_source or "unknown")))

                if not sym_raw.isdigit():
                    try:
                        for cand in self._candidate_tokens_for_symbol(sym_raw, seg_norm, limit=8):
                            ex_c = str(cand.get("exchange") or "").strip().upper() or str(seg_norm)
                            tok_c = str(cand.get("token") or "").strip()
                            if tok_c.isdigit() and int(tok_c) > 0:
                                token_pairs.append((ex_c, tok_c, "instruments"))
                    except Exception:
                        pass

                # De-dup while preserving order.
                seen_pairs: set[tuple[str, str]] = set()
                unique_pairs: List[tuple[str, str, str]] = []
                for ex_c, tok_c, src_c in token_pairs:
                    key = (str(ex_c).upper(), str(tok_c))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    unique_pairs.append((str(ex_c).upper(), str(tok_c), src_c))

                candles: List[Candle] = []
                for ex_c, tok_c, src_c in unique_pairs:
                    for interval in intervals_to_try or [requested_interval]:
                        try:
                            candles_raw = self._fetch_intraday_chart(
                                exchange_segment=str(ex_c),
                                symboltoken=str(tok_c),
                                interval=str(interval),
                                limit=int(limit),
                            )
                        except Exception as exc:  # noqa: BLE001
                            self._note_intraday_failure(exc)
                            self._log_intraday_failure(
                                key=f"underlying:{ex_c}:{tok_c}:{interval}",
                                msg=f"[INTRADAY] Failed exch={ex_c} token={tok_c} interval={interval}: {exc}",
                            )
                            candles_raw = []

                        candles = candles_raw
                        # If we fetched a smaller interval, resample to the requested timeframe.
                        try:
                            fetch_min = self._interval_minutes(str(interval))
                            if (
                                candles
                                and desired_min is not None
                                and fetch_min is not None
                                and fetch_min < desired_min
                                and desired_min % fetch_min == 0
                            ):
                                candles = self._resample_candles(candles, target_minutes=int(desired_min))
                                if len(candles) > int(limit):
                                    candles = candles[-int(limit) :]
                        except Exception:
                            candles = candles_raw

                        if candles:
                            self._note_intraday_success()
                            # If we had to switch token/segment, persist for this session
                            # so subsequent calls behave consistently.
                            if (tok_c != str(token_for_intraday)) or (ex_c != str(seg_norm).upper()):
                                try:
                                    os.environ["MSTOCK_UNDERLYING_TOKEN"] = str(tok_c)
                                    os.environ["MSTOCK_INTRADAY_EXCHANGE"] = str(ex_c)
                                    print(
                                        "[INTRADAY] Using alternate token/segment for candles; "
                                        f"token={tok_c} exch={ex_c} src={src_c}. "
                                        "Updated MSTOCK_UNDERLYING_TOKEN and MSTOCK_INTRADAY_EXCHANGE for this session."
                                    )
                                except Exception:
                                    pass
                            intraday_series_for_merge = list(candles)
                            # If we already have a historical bootstrap base, merge and return.
                            if isinstance(bootstrap_base, list) and bootstrap_base:
                                merged = _merge_candles(bootstrap_base, intraday_series_for_merge)
                                return save_and_return(merged, source="intraday")

                            # If intraday has enough candles, return it directly.
                            # Otherwise fall through to fetch a one-time historical bootstrap.
                            if len(intraday_series_for_merge) >= min(int(limit), _bootstrap_min_needed()):
                                return save_and_return(intraday_series_for_merge, source="intraday")

                            need_historical_bootstrap = True
                            # Do not return yet; proceed to historical section.
                            break

                    if need_historical_bootstrap:
                        break

                # Intraday enabled but returned empty candles for all attempted intervals.
                # Keep this quiet by default to avoid noisy UI logs.
                if intraday_only:
                    self._log_intraday_failure(
                        key=f"intraday-empty:{token_for_intraday}:{requested_interval}",
                        msg=(
                            "[INTRADAY] Empty candles in intraday-only mode; "
                            f"token={token_for_intraday} src={token_source or 'unknown'} "
                            f"seg={seg_norm}({seg_id}) intervals={intervals_to_try}. "
                            f"auth(api_key={_present(api_key)} access_token={_present(access_token)}). "
                            "Check access token/API key and underlying token/segment."
                        ),
                    )

            # Intraday enabled but empty: stop here only in strict mode.
            if intraday_only:
                fb = _intraday_cache_fallback()
                return fb if fb is not None else []

            # Intraday returned empty (or we deliberately fell through for bootstrap).
            # If we already bootstrapped historical candles, prefer returning that
            # over repeatedly hitting historical endpoints.
            if (not need_historical_bootstrap) and isinstance(bootstrap_base, list) and bootstrap_base:
                merged = _merge_candles(bootstrap_base, cached_candles if str(cached_source or "").lower() == "intraday" else None)
                return save_and_return(merged, source="historical")

        # If intraday is enabled+available but it failed/empty, prefer using a
        # young cached value as a fallback (avoids sudden drops to empty).
        if use_intraday_chart and self._intraday_available() and cache_is_young and cached_candles is not None:
            return cached_candles  # type: ignore[return-value]

        # The SDK chart endpoints typically require:
        # - exchange (e.g. "NSE" or "NFO")
        # - numeric instrument token
        # - interval constant (e.g. "ONE_MINUTE")
        #
        # Strategy code passes `cfg.underlying` here, which is often a symbol
        # string (e.g. "NIFTY" or "NSE:NIFTY"). If `symbol` is not a numeric
        # token, we fall back to environment variables.
        # Underlying candles (signals): default to configured underlying exchange.
        # For index spot this is usually NSE; for futures-based underlying use NFO.
        exchange = os.getenv("MSTOCK_UNDERLYING_EXCHANGE", os.getenv("MSTOCK_EXCHANGE", "NSE"))
        exchange = str(exchange or "").strip().upper() or "NSE"

        token = ""
        raw_symbol = (symbol or "").strip()
        sym = raw_symbol
        exchange_was_explicit = ":" in raw_symbol and not raw_symbol.split(":", 1)[0].strip().isdigit()

        # Allow explicit EXCH:SYMBOL.
        if ":" in sym and not sym.isdigit():
            maybe_exch, maybe_sym = sym.split(":", 1)
            if maybe_exch.strip():
                exchange = maybe_exch.strip().upper()
            sym = maybe_sym.strip()

        # If the user didn't explicitly request a derivative exchange and the underlying
        # is an index name, candles should be fetched from spot (NSE). This prevents
        # ambiguous NFO:NIFTY tokens from instruments master.
        sym_key_for_exchange = self._normalize_symbol_key(sym) if sym and not sym.isdigit() else ""
        if (not exchange_was_explicit) and sym_key_for_exchange in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}:
            if exchange in {"NFO", "BFO"}:
                print(f"[CANDLES] Underlying {sym_key_for_exchange} is an index; forcing exchange NSE for candles (was {exchange}).")
                exchange = "NSE"

        # Prefer configured underlying token for the selected exchange.
        env_tok = os.getenv("MSTOCK_UNDERLYING_TOKEN", "").strip()

        # Build a short retry list. IMPORTANT:
        # If `symbol` is a name like "NIFTY" (non-numeric), prefer the resolved NSE index token
        # over the configured env token (which is often mistakenly set to an ETF token).
        candidate_tokens: List[str] = []
        sym_key = self._normalize_symbol_key(sym.split(":", 1)[-1]) if sym and not sym.isdigit() else ""

        resolved = ""
        if sym and not sym.isdigit():
            try:
                resolved = self._resolve_token_from_instruments(sym, exchange)
            except Exception as exc:  # noqa: BLE001
                # Keep the original error semantics: if we have no candidates at all, we must fail.
                if not (env_tok.isdigit() and int(env_tok) > 0):
                    raise RuntimeError(
                        "Underlying candle data requires MSTOCK_UNDERLYING_TOKEN (numeric token). "
                        f"Auto-resolution failed. Details: {exc}"
                    ) from exc

        if sym.isdigit() and int(sym) > 0:
            candidate_tokens.append(sym)

        if resolved and resolved.isdigit() and int(resolved) > 0:
            candidate_tokens.append(resolved)

        if env_tok.isdigit() and int(env_tok) > 0:
            # Down-rank env token only for NSE-equity/index style symbols.
            # For NFO futures, tokenName/tradingsymbol matching is broker-specific.
            if exchange == "NSE" and sym_key and not self._token_matches_symbol_key(env_tok, symbol_key=sym_key):
                print(
                    f"[CANDLES] Env token {env_tok} does not match requested symbol {sym!r}; "
                    f"will prefer resolved token {resolved or 'N/A'}."
                )
            candidate_tokens.append(env_tok)

        # De-dup while preserving order.
        seen_tok: set[str] = set()
        unique_tokens: List[str] = []
        for t in candidate_tokens:
            if t in seen_tok:
                continue
            seen_tok.add(t)
            unique_tokens.append(t)

        # Try candidates until we get non-empty candles.
        last_tried: Optional[str] = None
        for t in unique_tokens:
            last_tried = t
            candles, _used_interval = self.fetch_index_candles(
                t,
                exchange=exchange,
                limit=limit,
                timeframe=timeframe,
                force_historical_only=bool(startup_force_historical),
            )
            if candles:
                # If we ended up switching away from the env token, persist for this process.
                if env_tok and env_tok != t:
                    os.environ["MSTOCK_UNDERLYING_TOKEN"] = t
                    print(f"[CANDLES] Updated MSTOCK_UNDERLYING_TOKEN -> {t} (previous {env_tok})")
                if startup_force_historical:
                    self._startup_historical_bootstrap_done = True
                # Store as a one-time bootstrap base when we fell through due to
                # intraday series being too short for indicator warmup.
                try:
                    if use_intraday_chart and (need_historical_bootstrap or (intraday_series_for_merge is not None)):
                        self._candle_bootstrap_base[cache_key] = list(candles)
                        bootstrap_base = self._candle_bootstrap_base.get(cache_key)
                except Exception:
                    pass

                # If intraday candles were available, merge them on top of historical.
                if use_intraday_chart and isinstance(intraday_series_for_merge, list) and intraday_series_for_merge:
                    merged = _merge_candles(candles, intraday_series_for_merge)
                    return save_and_return(merged, source="intraday")

                return save_and_return(candles, source="historical")

        # If all candidates returned empty/None, let strategy wait.
        if last_tried:
            # If we had a partial fallback (like cached but stale, or partial results), 
            # we could return it here. But logic inside fetch_index_candles already handles partials.
            # If we have a bootstrap base (from earlier) and intraday was empty/flaky,
            # return that rather than hammering endpoints.
            try:
                if isinstance(bootstrap_base, list) and bootstrap_base:
                    merged = _merge_candles(bootstrap_base, intraday_series_for_merge)
                    return save_and_return(merged, source="historical")
            except Exception:
                pass
            return []
        # No candidates at all.
        raise RuntimeError(
            "Candle data requires a numeric instrument token. "
            "Set MSTOCK_UNDERLYING_TOKEN (recommended) or MSTOCK_SYMBOL_TOKEN, "
            "or pass the numeric token as `symbol` to get_candles()."
        )

        if not token:
            raise RuntimeError(
                "Candle data requires a numeric instrument token. "
                "Set MSTOCK_UNDERLYING_TOKEN (recommended) or MSTOCK_SYMBOL_TOKEN, "
                "or pass the numeric token as `symbol` to get_candles()."
            )

        # Unreachable: returned above.

    # ---- Orders ----

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        *,
        exchange: Optional[str] = None,
        symbol_token: Optional[str] = None,
        product_type: str = "INTRADAY",
        ordertag: str = "nifty-scalper-bot",
    ) -> Order:
        """Place order via the official SDK and return a compact view.

        ``symbol`` here is the trading symbol (e.g. ``"ACC-EQ"``) and
        you also need the corresponding symbol token and exchange. For
        now these are taken from environment variables so you can wire
        them without changing strategy code:

        - ``MSTOCK_SYMBOL_TOKEN`` – numeric token for the instrument
        - ``MSTOCK_EXCHANGE`` – exchange string (e.g. ``"NSE"``)
        """
        exchange_val = (exchange or os.getenv("MSTOCK_EXCHANGE", "NSE")).strip() or "NSE"

        # Normalize "EXCH:TRADINGSYMBOL" into raw trading symbol.
        symbol_val = str(symbol or "").strip()
        if ":" in symbol_val:
            try:
                maybe_exch, maybe_sym = symbol_val.split(":", 1)
                if maybe_exch.strip().upper() in {"NSE", "BSE", "NFO", "NSEFO", "NSECM", "BSECM"} and maybe_sym.strip():
                    symbol_val = maybe_sym.strip()
                    # Prefer explicit prefix exchange when caller didn't provide one.
                    if not (exchange or "").strip():
                        exchange_val = maybe_exch.strip().upper() or exchange_val
            except Exception:
                pass

        symbol_token_val = (symbol_token or os.getenv("MSTOCK_SYMBOL_TOKEN", "")).strip()

        variety = "NORMAL"
        duration = "DAY"
        product_type_val = str(product_type or "").strip().upper() or "INTRADAY"
        ordertag_val = str(ordertag or "").strip() or "nifty-scalper-bot"

        if not symbol_token_val:
            # Best-effort: resolve token/exchange using the instrument master.
            try:
                resolved_exch, resolved_tok = self.resolve_exchange_token(symbol_val, exchange_hint=exchange_val)
                if resolved_tok is not None and str(resolved_tok).strip():
                    symbol_token_val = str(resolved_tok).strip()
                if resolved_exch is not None and str(resolved_exch).strip():
                    exchange_val = str(resolved_exch).strip().upper() or exchange_val
            except Exception:
                pass

        if not symbol_token_val:
            raise RuntimeError(
                f"MSTOCK_SYMBOL_TOKEN is not set and token resolution failed for {exchange_val}:{symbol_val}. "
                "Configure token/exchange mapping before live trading."
            )

        price_str = "0" if price is None else str(price)
        side_upper = side.upper()

        resp = self._raw.place_order(
            variety,
            symbol_val,
            symbol_token_val,
            exchange_val,
            side_upper,
            order_type.upper(),
            str(quantity),
            product_type_val,
            price_str,
            "0",  # triggerprice
            "0",  # squareoff
            "0",  # stoploss
            "",   # trailingStopLoss
            "0",  # disclosedquantity
            duration,
            ordertag_val,
        )

        payload = self._safe_json(resp, context="place_order")
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        order_id = str(data.get("order_id", ""))

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side_upper,
            quantity=quantity,
            price=float(price_str),
            status=str(data.get("status", "")),
        )

    def cancel_order(self, order_id: str) -> None:
        variety = "NORMAL"
        self._raw.cancel_order(variety, order_id)

    def calculate_order_margin_required(
        self,
        *,
        exchange: str,
        symbol: str,
        token: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        product_type: str = "INTRADAY",
    ) -> Optional[float]:
        """Return required margin for a single order, if available.

        This calls the TypeB `calculate_order_margin` endpoint via the official SDK.
        Response schemas can vary; parsing is best-effort and may return None.
        """

        try:
            exchange_val = str(exchange or "").strip().upper() or "NSE"
            symbol_val = str(symbol or "").strip()
            token_val = str(token or "").strip()
            side_upper = str(side or "").strip().upper()
            qty = int(quantity or 0)
        except Exception:
            return None

        # Some callers use "EXCH:TRADINGSYMBOL" format. The margin endpoint
        # expects the raw trading symbol, so strip the prefix when present.
        try:
            if ":" in symbol_val:
                maybe_exch, maybe_sym = symbol_val.split(":", 1)
                if maybe_exch.strip().upper() in {"NSE", "BSE", "NFO", "NSEFO", "NSECM", "BSECM"} and maybe_sym.strip():
                    symbol_val = maybe_sym.strip()
        except Exception:
            pass

        # Broker/API variants sometimes emit NSEFO/NSECM etc.
        # Normalize to the values expected by TypeB calculate_order_margin.
        if exchange_val in {"NSEFO", "NFO"}:
            exchange_val = "NFO"
        elif exchange_val in {"NSECM", "NSE"}:
            exchange_val = "NSE"
        elif exchange_val in {"BSECM", "BSE"}:
            exchange_val = "BSE"

        if not symbol_val or not token_val or qty <= 0 or side_upper not in {"BUY", "SELL"}:
            return None

        price_str = "0" if price is None else str(price)

        resp = self._raw.calculate_order_margin(
            str(product_type or "INTRADAY").strip().upper(),
            side_upper,
            str(qty),
            price_str,
            exchange_val,
            symbol_val,
            token_val,
            "0",
        )

        payload = self._safe_json(resp, context="calculate_order_margin")
        return self._parse_margin_required(payload)

    def _parse_margin_required(self, payload: Any) -> Optional[float]:
        # Response schemas vary across SDK versions and broker deployments.
        # Prefer `payload["data"]` when present, but fall back to the payload
        # itself so we don't return None unnecessarily.
        if isinstance(payload, dict):
            data = payload.get("data", None)
            root: Any = payload if data is None else data
        else:
            root = payload

        def pick_number(obj: Any) -> Optional[float]:
            if isinstance(obj, (int, float)):
                return float(obj)
            if isinstance(obj, str):
                s = obj.strip()
                if not s:
                    return None

                # Common broker formats: "₹1,23,456.78", "1,234.00", "(123.45)".
                neg = False
                if s.startswith("(") and s.endswith(")"):
                    neg = True
                    s = s[1:-1].strip()

                # Remove common currency markers and separators.
                for junk in ("₹", "INR", "inr", "Rs.", "rs.", "Rs", "rs"):
                    s = s.replace(junk, "")
                s = s.replace(",", "").strip()

                # Fast path.
                try:
                    v = float(s)
                    return -float(v) if neg else float(v)
                except Exception:
                    pass

                # Last-resort: keep only numeric/exponent characters.
                try:
                    cleaned = "".join(ch for ch in s if (ch.isdigit() or ch in {".", "-", "+", "e", "E"}))
                    if not cleaned:
                        return None
                    v2 = float(cleaned)
                    return -float(v2) if neg else float(v2)
                except Exception:
                    return None
            return None

        # --- TypeB docs shape ---
        # Docs (Feb 2026): response contains `data.summary.total_charges` and a
        # `data.summary.breakup` list with items like SPANMARGIN/VARELMMARGIN/EXPOMARGIN.
        # Some deployments also return `data.charges: [{ total_charges, breakup: [...] }, ...]`.
        def sum_margin_breakup(items: Any) -> Optional[float]:
            if not isinstance(items, list) or not items:
                return None
            total = 0.0
            any_margin = False
            for it in items:
                if not isinstance(it, dict):
                    continue
                try:
                    name = str(it.get("name") or "").strip().upper()
                except Exception:
                    name = ""
                if name not in {"SPANMARGIN", "VARELMMARGIN", "EXPOMARGIN", "EXPOSUREMARGIN"}:
                    continue
                amt = pick_number(it.get("amount"))
                if amt is None:
                    continue
                total += float(amt)
                any_margin = True
            if any_margin:
                return float(total)
            return None

        def extract_from_typeb_doc_shape(obj: Any) -> Optional[float]:
            if not isinstance(obj, dict):
                return None

            # Prefer explicit margin components when available.
            summary = obj.get("summary")
            if isinstance(summary, dict):
                v = sum_margin_breakup(summary.get("breakup"))
                if v is not None:
                    return v

                # Fallback: if summary contains total_charges, treat it as required margin.
                # (Endpoint name is "calculate order margin"; some responses label this as charges.)
                v2 = pick_number(summary.get("total_charges"))
                if v2 is not None:
                    return float(v2)

            charges = obj.get("charges")
            if isinstance(charges, list) and charges:
                # If multiple orders are returned, sum the margin components across all.
                totals: list[float] = []
                for ch in charges:
                    if not isinstance(ch, dict):
                        continue
                    v = sum_margin_breakup(ch.get("breakup"))
                    if v is not None:
                        totals.append(float(v))
                if totals:
                    return float(sum(totals))

                # Fallback to total_charges per charge item.
                tc_sum = 0.0
                any_tc = False
                for ch in charges:
                    if not isinstance(ch, dict):
                        continue
                    v = pick_number(ch.get("total_charges"))
                    if v is None:
                        continue
                    tc_sum += float(v)
                    any_tc = True
                if any_tc:
                    return float(tc_sum)

            return None

        preferred_keys = [
            "totalMarginRequired",
            "total_margin_required",
            "totalRequiredMargin",
            "requiredMargin",
            "margin_required",
            "marginRequired",
            "total",
            # Some TypeB deployments label required margin as charges.
            "total_charges",
            "totalCharges",
        ]

        preferred_keys_lower = {str(k).lower() for k in preferred_keys}

        def key_priority(k: Any) -> int:
            try:
                k_low = str(k).lower()
            except Exception:
                return 0
            score = 0
            if k_low in preferred_keys_lower:
                score += 20
            if "margin" in k_low:
                score += 8
            if "req" in k_low or "required" in k_low:
                score += 8
            if "total" in k_low:
                score += 2
            if "orders" in k_low or "order" in k_low:
                score += 2
            if "data" in k_low or "result" in k_low or "response" in k_low:
                score += 1
            return score

        def extract_from_dict(d: dict) -> Optional[float]:
            for k in preferred_keys:
                if k in d:
                    v = pick_number(d.get(k))
                    if v is not None:
                        return v

            # Fallback: look for a single numeric field that looks like required margin.
            for k, v_raw in d.items():
                k_low = str(k).lower()
                if "margin" not in k_low:
                    continue
                if "req" not in k_low and "required" not in k_low and "total" not in k_low:
                    continue
                v = pick_number(v_raw)
                if v is not None:
                    return v
            return None

        def extract_any(obj: Any, *, depth: int = 0) -> Optional[float]:
            if depth > 8:
                return None

            if isinstance(obj, dict):
                # TypeB calculate-order-margin docs shape.
                v_doc = extract_from_typeb_doc_shape(obj)
                if v_doc is not None:
                    return v_doc

                # Fast path: direct match.
                direct = extract_from_dict(obj)
                if direct is not None:
                    return direct

                # Common schema: dict with `orders: [...]`.
                orders = obj.get("orders")
                if isinstance(orders, list):
                    total = 0.0
                    any_ok = False
                    for item in orders:
                        v = extract_any(item, depth=depth + 1)
                        if v is None:
                            continue
                        total += float(v)
                        any_ok = True
                    if any_ok:
                        return float(total)

                # Heuristic: walk children, prioritizing keys that look margin-related.
                try:
                    items = list(obj.items())
                except Exception:
                    items = []
                items.sort(key=lambda kv: key_priority(kv[0]), reverse=True)
                for _k, v_raw in items:
                    if isinstance(v_raw, (dict, list)):
                        v = extract_any(v_raw, depth=depth + 1)
                        if v is not None:
                            return v
                return None

            if isinstance(obj, list):
                total = 0.0
                any_ok = False
                for item in obj:
                    v = extract_any(item, depth=depth + 1)
                    if v is None:
                        continue
                    total += float(v)
                    any_ok = True
                if any_ok:
                    return float(total)
                return None

            return None

        return extract_any(root)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        resp = self._raw.get_net_position()
        payload = self._safe_json(resp, context="get_net_position")
        if isinstance(payload, dict) and "data" in payload:
            data = payload["data"]
            if isinstance(data, list):
                return data
        raise RuntimeError(
            "Unable to parse positions. Inspect `get_net_position().json()` "
            "and adjust MStockTypeBClient.get_open_positions."
        )

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Return delivery holdings (equity portfolio).

        The official TypeB SDK exposes `get_holdings()`. This wrapper normalizes
        common response shapes and returns an empty list on missing/None payload.
        """

        resp = self._raw.get_holdings()
        payload = self._safe_json(resp, context="get_holdings")
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
            # Some brokers return "None" or null as a string.
            if data in (None, "None", "none", "NULL", "null"):
                return []
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []
