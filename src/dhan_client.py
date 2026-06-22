from __future__ import annotations

import os
import sys
import time
import logging
import concurrent.futures
from importlib import util as importlib_util
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

LOG = logging.getLogger(__name__)


# ── CF-001: HTTP 429 rate-limit helpers ──────────────────────────────────────

def _parse_retry_after_dhan(value: str) -> float:
    """Parse Retry-After header value (seconds or HTTP-date) into a float."""
    s = str(value or "").strip()
    if not s:
        return 0.0
    try:
        return max(0.0, float(s))
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(s)
            return max(0.0, (dt - datetime.now()).total_seconds())
        except Exception:
            return 0.0


def _apply_dhan_rate_limit_backoff(context: str, retry_after: float = 0.0) -> float:
    """Apply exponential backoff for rate-limited Dhan endpoints.

    Returns the backoff delay applied (seconds). Logs a warning.
    """
    if retry_after > 0:
        delay = float(retry_after)
        print(f"[RATE] dhan {context}: HTTP 429, Retry-After={delay:.1f}s")
    else:
        delay = 1.0
        print(f"[RATE] dhan {context}: HTTP 429, exponential backoff={delay:.1f}s")
    time.sleep(delay)
    return delay


def _http_call_with_retry_dhan(
    method_fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    context: str = "unknown",
    **kwargs,
):
    """Call an HTTP method with exponential backoff on 429 responses.

    Works with both ``urllib`` and ``requests`` response objects.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = method_fn(*args, **kwargs)
            if response is None:
                return None
            status_code = getattr(response, "status_code", 200)
            if status_code == 429:
                hdrs = getattr(response, "headers", {}) or {}
                retry_after_raw = hdrs.get("Retry-After", "") if isinstance(hdrs, dict) else ""
                delay = _parse_retry_after_dhan(str(retry_after_raw))
                if delay <= 0:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                delay = min(delay, max_delay)
                LOG.warning(
                    f"dhan HTTP 429 {context}: retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(delay)
                continue
            return response
        except Exception as exc:
            last_exc = exc
            # Check if it's an HTTP 429 raised as an exception by the SDK.
            err_str = str(exc).lower()
            is_429 = (
                getattr(exc, "status_code", None) == 429
                or "429" in err_str
                or "rate limit" in err_str
                or "too many requests" in err_str
            )
            if is_429 and attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                LOG.warning(
                    f"dhan rate-limit exception {context}: retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries + 1}) — {exc}"
                )
                time.sleep(delay)
                continue
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"dhan HTTP call {context} failed after {max_retries} retries")


from config import StrategyConfig
from dhan_auth import generate_dhan_access_token, is_dhan_token_refresh_due, normalize_dhan_pin, persist_dhan_access_token
from market_data import Candle
from mstock_client import Order

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DHAN_SDK = _REPO_ROOT / "DhanHQ-py-main" / "DhanHQ-py-main" / "src"
if _LOCAL_DHAN_SDK.exists() and str(_LOCAL_DHAN_SDK) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DHAN_SDK))

try:
    from dhanhq import DhanContext, Security, dhanhq as DhanHQClient
except Exception:  # pragma: no cover
    DhanContext = None  # type: ignore[assignment]
    DhanHQClient = None  # type: ignore[assignment]
    Security = None  # type: ignore[assignment]


INDEX_SECURITY_IDS = {
    "NIFTY": "13",
    "BANKNIFTY": "25",
    "FINNIFTY": "27",
}


@dataclass
class DhanCallResult:
    ok: bool
    data: Any = None
    error_code: str = ""
    error_message: str = ""


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = str(os.getenv(key, "") or "").strip()
        if value:
            return value
    return default


def get_dhan_underlying_security_id(underlying: str = "NIFTY") -> str:
    value = _env_first("DHAN_UNDERLYING_SECURITY_ID", "DHAN_UNDER_SECURITY_ID")
    if value:
        os.environ["DHAN_UNDERLYING_SECURITY_ID"] = value
        os.environ["DHAN_UNDER_SECURITY_ID"] = value
        return value
    return str(INDEX_SECURITY_IDS.get(str(underlying or "NIFTY").strip().upper(), "13"))


def dhan_underlying_security_id_present() -> bool:
    return bool(_env_first("DHAN_UNDERLYING_SECURITY_ID", "DHAN_UNDER_SECURITY_ID"))


def _classify_dhan_error(exc: Exception) -> str:
    msg = str(exc or "").lower()
    if "sdk" in msg and ("install" in msg or "import" in msg or "unavailable" in msg):
        return "SDK_IMPORT_FAILED"
    if "credential" in msg or "client_id" in msg or "access_token" in msg or "token_missing" in msg:
        return "CREDENTIALS_MISSING"
    if "unauthor" in msg or "forbidden" in msg or "401" in msg or "403" in msg or "auth" in msg:
        return "AUTH_FAILED"
    if "underlying" in msg and ("invalid" in msg or "security" in msg):
        return "INVALID_UNDERLYING_ID"
    if "timeout" in msg or "timed out" in msg:
        return "TIMEOUT"
    if "schema" in msg or "parse" in msg:
        return "SCHEMA_MISMATCH"
    return "API_ERROR"

EXCHANGE_ALIASES = {
    "NSE": "NSE_EQ",
    "BSE": "BSE_EQ",
    "NFO": "NSE_FNO",
    "NSE_FNO": "NSE_FNO",
    "IDX_I": "IDX_I",
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _now_ist() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return datetime.now()


def _parse_dhan_timestamp(value: Any) -> Any:
    if value in (None, ""):
        return pd.NaT
    try:
        if isinstance(value, (int, float)) or str(value).strip().isdigit():
            raw = float(value)
            unit = "ms" if raw > 10_000_000_000 else "s"
            return pd.to_datetime(raw, unit=unit, errors="coerce")
    except Exception:
        pass
    return pd.to_datetime(value, errors="coerce")


def _safe_payload_sample(value: Any) -> Any:
    try:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, item in list(value.items())[:8]:
                if isinstance(item, list):
                    out[key] = item[:2]
                elif isinstance(item, dict):
                    out[key] = _safe_payload_sample(item)
                else:
                    out[key] = item
            return out
        if isinstance(value, list):
            return value[:2]
    except Exception:
        return "(sample_unavailable)"
    return value


def _normalize_option_type(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"CALL", "C"}:
        return "CE"
    if raw in {"PUT", "P"}:
        return "PE"
    return raw


def _walk_nodes(obj: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(obj, dict):
        out.append(obj)
        for value in obj.values():
            out.extend(_walk_nodes(value))
    elif isinstance(obj, list):
        out.append(obj)
        for item in obj:
            out.extend(_walk_nodes(item))
    return out


def mask_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return "(missing)"
    if len(token) <= 3:
        return token[:2] + "..." + token[-1:]
    if len(token) <= 8:
        return token[:2] + "..." + token[-2:]
    if len(token) >= 24:
        return token[:4] + "..." + token[-2:]
    return token[:4] + "..." + token[-4:]


def get_dhan_sdk_diagnostics() -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "sdk_source": "not_found",
        "sdk_path": "",
        "import_ok": False,
        "error": "",
    }
    try:
        spec = importlib_util.find_spec("dhanhq")
    except Exception as exc:
        diagnostics["error"] = str(exc)
        return diagnostics
    if spec is None:
        diagnostics["error"] = "dhanhq package could not be located"
        return diagnostics
    origin = str(getattr(spec, "origin", "") or "").strip()
    search_locations = list(getattr(spec, "submodule_search_locations", []) or [])
    module_path = origin
    if not module_path and search_locations:
        module_path = str(search_locations[0])
    diagnostics["sdk_path"] = module_path
    try:
        import dhanhq as dhanhq_module  # type: ignore

        module_file = str(getattr(dhanhq_module, "__file__", "") or "").strip()
        if module_file:
            diagnostics["sdk_path"] = module_file
        diagnostics["import_ok"] = True
    except Exception as exc:
        diagnostics["error"] = str(exc)
        if not diagnostics["sdk_path"]:
            diagnostics["sdk_path"] = module_path
        return diagnostics
    try:
        resolved_sdk_path = Path(str(diagnostics["sdk_path"] or "")).resolve()
        resolved_local = _LOCAL_DHAN_SDK.resolve()
        if str(resolved_sdk_path).startswith(str(resolved_local)):
            diagnostics["sdk_source"] = "local_path"
        elif "site-packages" in str(resolved_sdk_path) or "dist-packages" in str(resolved_sdk_path):
            diagnostics["sdk_source"] = "pip_package"
        else:
            diagnostics["sdk_source"] = "unknown"
    except Exception:
        diagnostics["sdk_source"] = "unknown"
    return diagnostics


def validate_dhan_live_readiness(symbol: str = "NIFTY", client: Optional["DhanClient"] = None) -> Dict[str, Any]:
    sdk_diag = get_dhan_sdk_diagnostics()
    readiness: Dict[str, Any] = {
        "sdk_import_ok": bool(sdk_diag.get("import_ok")),
        "sdk_source": str(sdk_diag.get("sdk_source", "unknown") or "unknown"),
        "client_id_present": bool(str(os.getenv("DHAN_CLIENT_ID", "")).strip()),
        "access_token_present": bool(str(os.getenv("DHAN_ACCESS_TOKEN", "")).strip()),
        "underlying_security_id_present": bool(_env_first("DHAN_UNDERLYING_SECURITY_ID", "DHAN_UNDER_SECURITY_ID")),
        "option_chain_ok": False,
        "atm_detected": False,
        "selected_contract_from_live_chain": False,
        "selected_security_id": "",
        "selected_symbol": "",
        "ltp_ok": False,
        "bid_ask_ok": False,
        "positions_ok": False,
        "holdings_ok": False,
        "last_success_timestamp": None,
        "last_error": "",
        "live_order_allowed": False,
    }
    active_client = client
    try:
        if active_client is None:
            active_client = DhanClient()
    except Exception as exc:
        readiness["last_error"] = str(exc)
        return readiness
    if active_client is None:
        readiness["last_error"] = "Dhan client could not be created."
        return readiness
    readiness.update(
        {
            "sdk_import_ok": bool(sdk_diag.get("import_ok")),
            "sdk_source": str(sdk_diag.get("sdk_source", "unknown") or "unknown"),
            "client_id_present": bool(active_client.client_id),
            "access_token_present": bool(active_client.access_token),
            "underlying_security_id_present": bool(_env_first("DHAN_UNDERLYING_SECURITY_ID", "DHAN_UNDER_SECURITY_ID")),
        }
    )
    try:
        chain = active_client.get_option_chain(symbol)
        readiness["option_chain_ok"] = bool(chain)
        ce_rows = [row for row in chain if str(row.get("option_type")) == "CE"]
        pe_rows = [row for row in chain if str(row.get("option_type")) == "PE"]
        atm_row: Optional[Dict[str, Any]] = None
        if ce_rows and pe_rows:
            spot = None
            for row in chain:
                spot = _safe_float(row.get("raw", {}).get("underlying_spot_price"))
                if spot is not None:
                    break
            if spot is not None:
                atm_row = min(chain, key=lambda row: abs(float(row.get("strike") or 0.0) - float(spot)))
                readiness["atm_detected"] = True
        if atm_row is None and chain:
            atm_row = chain[0]
        if atm_row is not None:
            token = str(atm_row.get("token") or "").strip()
            symbol_name = str(atm_row.get("symbol") or "").strip()
            readiness["selected_security_id"] = token
            readiness["selected_symbol"] = symbol_name
            readiness["selected_contract_from_live_chain"] = bool(token and token in active_client._validated_live_security_ids)
            if symbol_name:
                try:
                    active_client.get_bid_ask(symbol_name, exchange_hint=str(atm_row.get("exchange") or ""))
                    readiness["bid_ask_ok"] = True
                except Exception as exc:
                    readiness["last_error"] = str(exc)
            if symbol_name:
                try:
                    active_client.get_ltp(symbol_name)
                    readiness["ltp_ok"] = True
                except Exception:
                    try:
                        active_client.get_ltp(symbol)
                        readiness["ltp_ok"] = True
                    except Exception as exc:
                        readiness["last_error"] = str(exc)
        try:
            active_client.get_open_positions()
            readiness["positions_ok"] = True
        except Exception as exc:
            readiness["last_error"] = readiness["last_error"] or str(exc)
        try:
            active_client.get_holdings()
            readiness["holdings_ok"] = True
        except Exception as exc:
            readiness["last_error"] = readiness["last_error"] or str(exc)
    except Exception as exc:
        readiness["last_error"] = str(exc)
    finalized = active_client.update_live_readiness(readiness)
    if not finalized.get("live_order_allowed") and not finalized.get("last_error"):
        finalized["last_error"] = "One or more Dhan live readiness checks did not pass."
        active_client.update_live_readiness(finalized)
    return active_client.live_readiness()


class DhanClient:
    """Best-effort trading client wrapper that mimics the m.Stock surface used by the app."""

    def __init__(self, cfg: Optional[StrategyConfig] = None) -> None:
        self.cfg = cfg
        self._sdk_diagnostics = get_dhan_sdk_diagnostics()
        self._context = None
        self._raw = None
        self._init_error = ""
        self._init_status = "UNKNOWN"
        self.client_id = str(os.getenv("DHAN_CLIENT_ID", "")).strip()
        self.access_token = str(os.getenv("DHAN_ACCESS_TOKEN", "")).strip()
        self.underlying_security_id = get_dhan_underlying_security_id(getattr(cfg, "underlying", "NIFTY") if cfg else "NIFTY")
        self._security_df: Optional[pd.DataFrame] = None
        self._last_api_error: str = ""
        self._last_success_ts: Optional[float] = None
        self._health_passed_current_session: bool = False
        self._validated_live_security_ids: set[str] = set()
        self._live_readiness: Dict[str, Any] = self._default_live_readiness()
        print(
            "[DHAN-AUTH] "
            f"client_id_present={bool(self.client_id)} "
            f"token_present={bool(self.access_token)} "
            f"underlying_id={self.underlying_security_id or '(missing)'}"
        )
        self._lazy_init()

    def sdk_diagnostics(self) -> Dict[str, Any]:
        return dict(self._sdk_diagnostics)

    def _default_live_readiness(self) -> Dict[str, Any]:
        return {
            "sdk_import_ok": bool(self._sdk_diagnostics.get("import_ok")),
            "sdk_source": str(self._sdk_diagnostics.get("sdk_source", "unknown") or "unknown"),
            "client_id_present": bool(self.client_id),
            "access_token_present": bool(self.access_token),
            "underlying_security_id_present": dhan_underlying_security_id_present(),
            "option_chain_ok": False,
            "atm_detected": False,
            "selected_contract_from_live_chain": False,
            "selected_security_id": "",
            "selected_symbol": "",
            "ltp_ok": False,
            "bid_ask_ok": False,
            "positions_ok": False,
            "holdings_ok": False,
            "last_success_timestamp": self._last_success_ts,
            "last_error": "",
            "live_order_allowed": False,
        }

    def live_readiness(self) -> Dict[str, Any]:
        state = dict(self._live_readiness)
        state["last_success_timestamp"] = self._last_success_ts
        state["last_error"] = state.get("last_error", "") or self._last_api_error
        return state

    def auth_status(self) -> str:
        if DhanContext is None or DhanHQClient is None:
            return "SDK_IMPORT_FAILED"
        if not self.client_id and not self.access_token:
            return "CREDENTIALS_MISSING"
        if self.client_id and not self.access_token:
            return "TOKEN_MISSING"
        if not self.client_id or not self.access_token:
            return "CREDENTIALS_MISSING"
        if self._init_status == "AUTH_FAILED":
            return "AUTH_FAILED"
        if self._raw is not None:
            return "AUTH_OK"
        return "UNKNOWN"

    def _lazy_init(self) -> bool:
        if getattr(self, "_raw", None) is not None:
            return True
        if DhanContext is None or DhanHQClient is None:
            self._init_status = "SDK_IMPORT_FAILED"
            self._init_error = "Dhan SDK is not installed. Install with `pip install dhanhq`."
            self._last_api_error = self._init_error
            print(f"[DHAN-INIT] status={self._init_status} error={self._init_error}")
            return False
        if not self.client_id or not self.access_token:
            self._init_status = "TOKEN_MISSING" if self.client_id and not self.access_token else "CREDENTIALS_MISSING"
            self._init_error = "DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are required."
            self._last_api_error = self._init_error
            print(f"[DHAN-INIT] status={self._init_status} error={self._init_error}")
            return False
        try:
            self._context = DhanContext(self.client_id, self.access_token)
            self._raw = DhanHQClient(self._context)
            self._init_status = "AUTH_OK"
            self._init_error = ""
            print("[DHAN-INIT] status=AUTH_OK")
            return True
        except Exception as exc:
            self._init_status = _classify_dhan_error(exc)
            self._init_error = str(exc)
            self._last_api_error = f"init: {exc}"
            print(f"[DHAN-INIT] status={self._init_status} error={exc}")
            return False

    def _ensure_ready(self) -> None:
        if not self._lazy_init():
            raise RuntimeError(f"{self.auth_status()}: {self._init_error or 'Dhan client is not ready'}")

    def update_live_readiness(self, readiness: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._default_live_readiness()
        merged.update(readiness or {})
        merged["sdk_import_ok"] = bool(merged.get("sdk_import_ok"))
        merged["client_id_present"] = bool(merged.get("client_id_present"))
        merged["access_token_present"] = bool(merged.get("access_token_present"))
        merged["underlying_security_id_present"] = bool(merged.get("underlying_security_id_present"))
        merged["option_chain_ok"] = bool(merged.get("option_chain_ok"))
        merged["atm_detected"] = bool(merged.get("atm_detected"))
        merged["selected_contract_from_live_chain"] = bool(merged.get("selected_contract_from_live_chain"))
        merged["ltp_ok"] = bool(merged.get("ltp_ok"))
        merged["bid_ask_ok"] = bool(merged.get("bid_ask_ok"))
        merged["positions_ok"] = bool(merged.get("positions_ok"))
        merged["holdings_ok"] = bool(merged.get("holdings_ok"))
        merged["sdk_source"] = str(merged.get("sdk_source", "unknown") or "unknown")
        merged["selected_security_id"] = str(merged.get("selected_security_id", "") or "")
        merged["selected_symbol"] = str(merged.get("selected_symbol", "") or "")
        merged["last_error"] = str(merged.get("last_error", "") or "")
        merged["live_order_allowed"] = bool(
            merged["sdk_import_ok"]
            and merged["client_id_present"]
            and merged["access_token_present"]
            and merged["underlying_security_id_present"]
            and merged["option_chain_ok"]
            and merged["atm_detected"]
            and merged["selected_contract_from_live_chain"]
            and merged["ltp_ok"]
            and merged["bid_ask_ok"]
            and merged["positions_ok"]
            and merged["holdings_ok"]
        )
        merged["last_success_timestamp"] = self._last_success_ts
        self._live_readiness = merged
        self._health_passed_current_session = bool(merged["live_order_allowed"])
        return self.live_readiness()

    def _guard_live_order(self, *, symbol: str, token: str, order_mode: str) -> None:
        if str(os.getenv("SCALPER_BROKER", "")).strip().lower() != "dhan":
            raise RuntimeError("Dhan live order placement blocked: SCALPER_BROKER must be set to dhan.")
        if str(os.getenv("SCALPER_ALLOW_LIVE_ORDERS", "")).strip().lower() != "true":
            raise RuntimeError("Dhan live order placement blocked: SCALPER_ALLOW_LIVE_ORDERS=true is required.")
        if str(order_mode or "").strip().lower() != "live":
            raise RuntimeError("Dhan live order placement blocked: order mode must be explicitly set to live.")
        readiness = self.live_readiness()
        if not readiness.get("sdk_import_ok"):
            raise RuntimeError("Dhan live order placement blocked: Dhan SDK import is not healthy in the current session.")
        if not readiness.get("option_chain_ok"):
            raise RuntimeError("Dhan live order placement blocked: live option chain validation has not passed in the current session.")
        if not readiness.get("live_order_allowed") or not self._health_passed_current_session:
            reason = self._live_readiness.get("last_error") or "broker live readiness has not passed in the current session."
            raise RuntimeError(f"Dhan live order placement blocked: {reason}")
        if token not in self._validated_live_security_ids:
            raise RuntimeError("Dhan live order placement blocked: security_id was not validated from live option chain during current session.")
        if str(readiness.get("selected_security_id", "")).strip() and str(readiness.get("selected_security_id", "")).strip() != str(token).strip():
            raise RuntimeError("Dhan live order placement blocked: order security_id does not match the validated live contract.")

    def mark_health_check_passed(self, passed: bool) -> None:
        self._health_passed_current_session = bool(passed)
        self._live_readiness["live_order_allowed"] = bool(passed and self._live_readiness.get("live_order_allowed"))

    def last_success_timestamp(self) -> Optional[float]:
        return self._last_success_ts

    def last_api_error(self) -> str:
        return self._last_api_error

    def _record_success(self) -> None:
        self._last_success_ts = float(pd.Timestamp.utcnow().timestamp())
        self._last_api_error = ""

    def _record_error(self, endpoint: str, exc: Exception) -> None:
        self._last_api_error = f"{endpoint}: {exc}"

    def _call_api(self, category: str, fn, *args, **kwargs):
        try:
            timeout = float(os.getenv("DHAN_API_TIMEOUT_SEC", "15") or 15)
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(fn, *args, **kwargs)
                payload = future.result(timeout=timeout)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            keys: List[str] = []
            if isinstance(payload, dict):
                keys = list(payload.keys())[:12]
            print(
                f"[broker:{'dhan'}] category={category} status=ok "
                f"token={mask_token(self.access_token)} keys={keys}"
            )
            self._record_success()
            return payload
        except Exception as exc:
            self._record_error(category, exc)
            print(
                f"[broker:{'dhan'}] category={category} status=error "
                f"token={mask_token(self.access_token)} error={str(exc)}"
            )
            raise

    def safe_call(self, category: str, fn, *args, **kwargs) -> DhanCallResult:
        try:
            self._ensure_ready()
            return DhanCallResult(ok=True, data=self._call_api_with_retry(category, fn, *args, **kwargs))
        except Exception as exc:
            code = _classify_dhan_error(exc)
            self._record_error(category, exc)
            return DhanCallResult(ok=False, error_code=code, error_message=str(exc))

    def _call_api_with_retry(self, category: str, fn, *args, **kwargs) -> Any:  # noqa: ARG002
        """Call a Dhan SDK method with retry on HTTP 429 rate-limit responses.

        Wraps ``fn`` with exponential backoff. Retries up to 3 times.
        """
        last_exc: Optional[Exception] = None
        max_retries = 3
        base_delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                return self._call_api(category, fn, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                is_429 = (
                    getattr(exc, "status_code", None) == 429
                    or exc.__class__.__name__ in ("HqApiException", "RateLimitException", "429")
                    or "429" in err_str
                    or "rate limit" in err_str
                    or "too many requests" in err_str
                )
                if is_429 and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), 60.0)
                    print(
                        f"[RATE] dhan {category}: HTTP 429, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    time.sleep(delay)
                    continue
                if not is_429 and attempt < max_retries:
                    # Retry transient errors too, but without rate-limit delay.
                    delay = min(base_delay * (2 ** attempt), 10.0)
                    print(
                        f"[RETRY] dhan {category}: {exc}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    time.sleep(delay)
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError(f"dhan _call_api_with_retry {category} failed")

    def login(self, *, interactive: bool = True) -> None:
        self._ensure_ready()
        if not self.access_token:
            raise RuntimeError("DHAN_ACCESS_TOKEN is not set. Paste a valid Dhan access token in Settings.")
        try:
            login_helper = getattr(self._context, "dhan_login", None)
        except Exception:
            login_helper = None
        try:
            if is_dhan_token_refresh_due(self.access_token, within_seconds=900):
                renewed = None
                if login_helper is not None and hasattr(login_helper, "renew_token"):
                    try:
                        payload = login_helper.renew_token(self.access_token)
                        renewed = str(
                            payload.get("accessToken")
                            or payload.get("access_token")
                            or payload.get("token")
                            or ""
                        ).strip()
                    except Exception:
                        renewed = None
                if not renewed:
                    renewed = self._generate_token_from_local_creds()
                if renewed:
                    self._rebuild_context(renewed)
            if login_helper is not None and hasattr(login_helper, "user_profile"):
                login_helper.user_profile(self.access_token)
        except Exception:
            regenerated = self._generate_token_from_local_creds()
            self._rebuild_context(regenerated)
        self._record_success()

    def _rebuild_context(self, access_token: str) -> None:
        self.access_token = str(access_token or "").strip()
        persist_dhan_access_token(self.access_token)
        if DhanContext is None or DhanHQClient is None:
            raise RuntimeError("Dhan SDK is not installed. Install with `pip install dhanhq`.")
        self._context = DhanContext(self.client_id, self.access_token)
        self._raw = DhanHQClient(self._context)
        self._init_status = "AUTH_OK"
        self._init_error = ""

    def _generate_token_from_local_creds(self) -> str:
        pin_raw = str(os.getenv("DHAN_PIN", "")).strip()
        totp_secret = str(os.getenv("DHAN_TOTP_SECRET", os.getenv("MSTOCK_TOTP_SECRET", ""))).strip()
        totp_code = "" if totp_secret else str(os.getenv("DHAN_TOTP_CODE", "")).strip()
        if not pin_raw or not (totp_secret or totp_code):
            raise RuntimeError("Dhan token refresh requires DHAN_PIN and Dhan TOTP secret/code.")
        pin = normalize_dhan_pin(pin_raw)
        return generate_dhan_access_token(
            self.client_id,
            pin,
            totp_secret=totp_secret,
            totp_code=totp_code,
        )

    def _ensure_security_df(self) -> pd.DataFrame:
        if self._security_df is not None:
            return self._security_df
        if Security is None:
            raise RuntimeError("Dhan Security helper is unavailable.")
        cache_path = Path(os.getenv("DHAN_SECURITY_LIST_PATH", "data/dhan_security_list_compact.csv"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            df = pd.read_csv(cache_path, low_memory=False, dtype=str)
        else:
            df = Security.fetch_security_list("compact", filename=str(cache_path))
            if df is None:
                raise RuntimeError("Could not fetch Dhan security list.")
            try:
                df = pd.read_csv(cache_path, low_memory=False, dtype=str)
            except Exception:
                pass
        df.columns = [str(c).strip() for c in df.columns]
        self._security_df = df
        return df

    def _find_security_row(self, symbol: str, exchange_hint: Optional[str] = None) -> Optional[pd.Series]:
        sym = str(symbol or "").strip()
        if not sym:
            return None
        if ":" in sym:
            maybe_exch, maybe_sym = sym.split(":", 1)
            exchange_hint = exchange_hint or maybe_exch
            sym = maybe_sym.strip()
        sym_upper = sym.upper()
        if sym_upper in INDEX_SECURITY_IDS:
            return pd.Series(
                {
                    "SEM_SMST_SECURITY_ID": INDEX_SECURITY_IDS[sym_upper],
                    "SEM_EXM_EXCH_ID": "IDX_I",
                    "SEM_TRADING_SYMBOL": sym_upper,
                    "SEM_CUSTOM_SYMBOL": sym_upper,
                }
            )
        df = self._ensure_security_df()
        candidates = df.copy()
        if exchange_hint:
            ex = EXCHANGE_ALIASES.get(str(exchange_hint).strip().upper(), str(exchange_hint).strip().upper())
            for col in ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "EXCH_ID"]:
                if col in candidates.columns:
                    filtered = candidates[candidates[col].astype(str).str.upper().eq(ex)]
                    if not filtered.empty:
                        candidates = filtered
                        break
        search_cols = [c for c in ["SEM_TRADING_SYMBOL", "SEM_CUSTOM_SYMBOL", "SEM_SYMBOL_NAME"] if c in candidates.columns]
        for col in search_cols:
            exact = candidates[candidates[col].astype(str).str.upper().eq(sym_upper)]
            if not exact.empty:
                return exact.iloc[0]
        for col in search_cols:
            starts = candidates[candidates[col].astype(str).str.upper().str.startswith(sym_upper)]
            if not starts.empty:
                return starts.iloc[0]
        return None

    def resolve_exchange_token(self, symbol: str, *, exchange_hint: Optional[str] = None) -> Tuple[str, str]:
        row = self._find_security_row(symbol, exchange_hint=exchange_hint)
        if row is None:
            return "", ""
        exchange = str(
            row.get("SEM_EXM_EXCH_ID")
            or row.get("SEM_SEGMENT")
            or row.get("EXCH_ID")
            or exchange_hint
            or "NSE_EQ"
        ).strip().upper()
        token = str(
            row.get("SEM_SMST_SECURITY_ID")
            or row.get("securityId")
            or row.get("security_id")
            or row.get("token")
            or ""
        ).strip()
        return exchange, token

    def resolve_exchange_token_symbol(self, symbol: str, *, exchange_hint: Optional[str] = None) -> Tuple[str, str, str]:
        row = self._find_security_row(symbol, exchange_hint=exchange_hint)
        if row is None:
            return "", "", ""
        exchange, token = self.resolve_exchange_token(symbol, exchange_hint=exchange_hint)
        out_symbol = str(row.get("SEM_TRADING_SYMBOL") or row.get("SEM_CUSTOM_SYMBOL") or symbol).strip()
        return exchange, token, out_symbol

    def _extract_payload_rows(self, payload: Any) -> List[Dict[str, Any]]:
        for node in _walk_nodes(payload):
            if isinstance(node, list) and node and all(isinstance(item, dict) for item in node):
                return list(node)
        return []

    def _resolve_expiry(self, underlying: str) -> str:
        self._ensure_ready()
        expiry_env = str(os.getenv("DHAN_TARGET_EXPIRY", os.getenv("MSTOCK_TARGET_EXPIRY", ""))).strip()
        if expiry_env:
            for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                try:
                    return datetime.strptime(expiry_env, fmt).strftime("%Y-%m-%d")
                except Exception:
                    continue
        under_id_raw = get_dhan_underlying_security_id(underlying)
        under_id = int(under_id_raw)
        resp = self._call_api("expiry_list", self._raw.expiry_list, under_security_id=under_id, under_exchange_segment="IDX_I")
        dates: List[str] = []
        for node in _walk_nodes(resp):
            if isinstance(node, list):
                dates.extend(str(item) for item in node if isinstance(item, (str, int, float)))
        clean = sorted({d[:10] for d in dates if len(str(d)) >= 10})
        if not clean:
            raise RuntimeError("Dhan expiry list returned no expiries.")
        return clean[0]

    def _extract_option_chain_rows(self, payload: Any, expiry: str, underlying_name: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def _append_leg(container: Dict[str, Any], strike_key: Any = None, opt_hint: str = "") -> None:
            row = dict(container)
            if strike_key not in (None, ""):
                row.setdefault("strike", strike_key)
                row.setdefault("strikePrice", strike_key)
            if opt_hint:
                row.setdefault("optionType", opt_hint)
                row.setdefault("option_type", opt_hint)
            row.setdefault("expiry", expiry)
            rows.append(row)

        def _walk_option(obj: Any, strike_key: Any = None) -> None:
            if isinstance(obj, dict):
                ce = obj.get("CE") or obj.get("ce") or obj.get("CALL") or obj.get("call")
                pe = obj.get("PE") or obj.get("pe") or obj.get("PUT") or obj.get("put")
                if isinstance(ce, dict) or isinstance(pe, dict):
                    local_strike = obj.get("strike") or obj.get("strikePrice") or obj.get("strike_price") or strike_key
                    if isinstance(ce, dict):
                        _append_leg(ce, local_strike, "CE")
                    if isinstance(pe, dict):
                        _append_leg(pe, local_strike, "PE")
                    return
                if any(k in obj for k in ("optionType", "option_type", "strike", "strikePrice", "securityId", "tradingSymbol")):
                    _append_leg(obj, strike_key)
                    return
                for key in ("oc", "option_chain", "optionChain", "data", "records", "result", "rows"):
                    value = obj.get(key)
                    if isinstance(value, dict):
                        if key == "oc":
                            for sk, nested in value.items():
                                _walk_option(nested, sk)
                        else:
                            _walk_option(value, strike_key)
                    elif isinstance(value, list):
                        for item in value:
                            _walk_option(item, strike_key)
            elif isinstance(obj, list):
                for item in obj:
                    _walk_option(item, strike_key)

        _walk_option(payload)
        if not rows:
            rows = self._extract_payload_rows(payload)
        return rows

    def get_option_chain_result(self, underlying: str) -> DhanCallResult:
        try:
            return DhanCallResult(ok=True, data=self.get_option_chain(underlying))
        except Exception as exc:
            return DhanCallResult(False, None, _classify_dhan_error(exc), str(exc))

    def get_option_chain(self, underlying: str) -> List[Dict[str, Any]]:
        self._ensure_ready()
        underlying_name = str(underlying or getattr(self.cfg, "underlying", "NIFTY") or "NIFTY").strip().upper()
        under_id_raw = get_dhan_underlying_security_id(underlying_name)
        under_id = int(under_id_raw)
        expiry = self._resolve_expiry(underlying_name)
        payload = self._call_api_with_retry(
            "option_chain",
            self._raw.option_chain,
            under_security_id=under_id,
            under_exchange_segment=os.getenv("DHAN_UNDER_EXCHANGE_SEGMENT", "IDX_I"),
            expiry=expiry,
        )
        rows = self._extract_option_chain_rows(payload, expiry, underlying_name)
        chain: List[Dict[str, Any]] = []
        spot = None
        for node in _walk_nodes(payload):
            if isinstance(node, dict):
                spot = _safe_float(
                    node.get("underlyingValue")
                    or node.get("underlying_value")
                    or node.get("underlyingPrice")
                    or node.get("underlying_spot_price")
                    or node.get("spot")
                )
                if spot is not None:
                    break
        for row in rows:
            option_type = _normalize_option_type(
                row.get("optionType") or row.get("option_type") or row.get("instrument_type") or row.get("right")
            )
            strike = _safe_float(row.get("strike") or row.get("strikePrice") or row.get("StrikePrice"))
            if strike is None or option_type not in {"CE", "PE"}:
                continue
            bid = _safe_float(row.get("bid") or row.get("bidPrice") or row.get("bestBidPrice"))
            ask = _safe_float(row.get("ask") or row.get("askPrice") or row.get("bestAskPrice"))
            ltp = row.get("ltp") or row.get("last_price") or row.get("lastPrice") or row.get("LastTradedPrice")
            oi = row.get("open_interest") or row.get("oi") or row.get("openInterest")
            token = str(row.get("securityId") or row.get("security_id") or row.get("securityid") or row.get("token") or "").strip()
            symbol = str(row.get("tradingSymbol") or row.get("tradingsymbol") or row.get("trading_symbol") or row.get("symbol") or "")
            chain.append(
                {
                    "symbol": symbol,
                    "trading_symbol": symbol,
                    "token": token,
                    "exchange": str(row.get("exchangeSegment") or row.get("exchange") or "NSE_FNO"),
                    "strike": float(strike),
                    "strike_price": float(strike),
                    "option_type": option_type,
                    "expiry": str(row.get("expiry") or expiry),
                    "symbol_root": underlying_name,
                    "ltp": _safe_float(ltp),
                    "bid": bid,
                    "ask": ask,
                    "volume": _safe_float(row.get("volume") or row.get("traded_volume") or row.get("totalTradedVolume")),
                    "oi": _safe_float(oi),
                    "iv": _safe_float(row.get("iv") or row.get("impliedVolatility")),
                    "delta": _safe_float(row.get("delta")),
                    "gamma": _safe_float(row.get("gamma")),
                    "theta": _safe_float(row.get("theta")),
                    "vega": _safe_float(row.get("vega")),
                    "spot": spot,
                    "underlying_ltp": spot,
                    "raw": {
                        **row,
                        "ltp": ltp,
                        "open_interest": oi,
                        "change_in_oi": row.get("change_in_oi") or row.get("changeOi") or row.get("oiChange"),
                        "bid": bid,
                        "ask": ask,
                        "spread": (ask - bid) if bid is not None and ask is not None else row.get("spread"),
                        "underlying_spot_price": spot or row.get("underlyingPrice") or row.get("spot"),
                    },
                }
            )
            if token:
                self._validated_live_security_ids.add(token)
        ce_count = len([row for row in chain if row.get("option_type") == "CE"])
        pe_count = len([row for row in chain if row.get("option_type") == "PE"])
        reason = ""
        if not chain:
            reason = "EMPTY_RESPONSE" if rows else "SCHEMA_MISMATCH"
        print(f"[DHAN-CHAIN] rows={len(chain)} expiry={expiry} spot={spot} ce_count={ce_count} pe_count={pe_count} reason={reason or 'OK'}")
        return chain

    def _extract_quote_row(self, payload: Any, token: str) -> Optional[Dict[str, Any]]:
        rows = self._extract_payload_rows(payload)
        for row in rows:
            row_token = str(row.get("securityId") or row.get("security_id") or row.get("securityid") or "").strip()
            if row_token == str(token).strip():
                return row
        for row in rows:
            if isinstance(row, dict):
                return row
        return None

    def get_ltp(self, symbol: str) -> float:
        self._ensure_ready()
        exch, tok = self.resolve_exchange_token(symbol)
        if not tok:
            raise RuntimeError(f"Unable to resolve Dhan token for {symbol!r}")
        payload = self._call_api_with_retry("ltp", self._raw.ticker_data, {exch: [tok]})
        row = self._extract_quote_row(payload, tok)
        if not row:
            raise RuntimeError(f"Unable to parse Dhan LTP for {symbol!r}")
        value = _safe_float(row.get("ltp") or row.get("last_price") or row.get("lastPrice") or row.get("LTP"))
        if value is None:
            raise RuntimeError(f"Unable to parse Dhan LTP for {symbol!r}")
        return float(value)

    def get_bid_ask(self, symbol: str, exchange_hint: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        self._ensure_ready()
        exch, tok = self.resolve_exchange_token(symbol, exchange_hint=exchange_hint)
        if not tok:
            return None, None, None
        payload = self._call_api_with_retry("quote", self._raw.quote_data, {exch: [tok]})
        row = self._extract_quote_row(payload, tok)
        if not row:
            return None, None, None
        bid = _safe_float(row.get("bid") or row.get("bidPrice") or row.get("bestBidPrice"))
        ask = _safe_float(row.get("ask") or row.get("askPrice") or row.get("bestAskPrice"))
        ltp = _safe_float(row.get("ltp") or row.get("last_price") or row.get("lastPrice"))
        return bid, ask, ltp

    def get_candles(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[Candle]:
        self._ensure_ready()
        sym = str(symbol or "").strip()
        if sym.upper() == "NIFTY":
            tok = os.getenv("DHAN_NIFTY_SECURITY_ID", get_dhan_underlying_security_id("NIFTY")).strip() or "13"
            exch = os.getenv("DHAN_NIFTY_EXCHANGE_SEGMENT", os.getenv("DHAN_UNDER_EXCHANGE_SEGMENT", "IDX_I")).strip().upper() or "IDX_I"
        elif sym.isdigit():
            tok = sym
            exch = os.getenv("DHAN_NIFTY_EXCHANGE_SEGMENT", "IDX_I").strip().upper() if tok == os.getenv("DHAN_NIFTY_SECURITY_ID", "13").strip() else "NSE_FNO"
        else:
            exch, tok = self.resolve_exchange_token(symbol)
        if not tok:
            return []
        interval_map = {"1m": 1, "one_minute": 1, "3m": 5, "5m": 5, "10m": 15, "15m": 15, "30m": 25, "1h": 60}
        interval = int(os.getenv("DHAN_INTRADAY_INTERVAL", "") or interval_map.get(str(timeframe).strip().lower(), 1))
        now = _now_ist()
        from_date = os.getenv("DHAN_CANDLE_FROM_DATE", "").strip() or now.strftime("%Y-%m-%d")
        to_date = os.getenv("DHAN_CANDLE_TO_DATE", "").strip() or now.strftime("%Y-%m-%d")
        instrument_type = "INDEX" if exch == "IDX_I" else "EQUITY"
        request = {
            "security_id": tok,
            "exchange_segment": exch,
            "instrument_type": instrument_type,
            "from_date": from_date,
            "to_date": to_date,
            "interval": interval,
            "oi": False,
        }
        payload = self._call_api_with_retry("intraday_minute_data", self._raw.intraday_minute_data, **request)
        self._log_raw_candle_payload(payload)
        rows = self.normalize_candle_payload(payload)
        candles = self._rows_to_candles(rows, limit=limit)
        fallback_attempted = False
        if not candles:
            fallback_attempted = True
            candles = self._fallback_candles(tok, exch, interval, limit)
        first = candles[0].time.isoformat() if candles else "-"
        last = candles[-1].time.isoformat() if candles else "-"
        columns = sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        print(f"[DHAN-CANDLES-NORMALIZED] rows={len(candles)} first={first} last={last} columns={columns}")
        if not candles:
            remarks = payload.get("remarks") if isinstance(payload, dict) else ""
            market_open = self._is_market_open_ist(now)
            print(f"[DHAN-CANDLES][EMPTY_REASON] remarks={remarks} request={request} market_open={market_open} fallback_attempted={fallback_attempted}")
        print(f"[DHAN-CANDLES] interval={timeframe} rows={len(candles)} first={first} last={last}")
        return candles

    def _is_market_open_ist(self, now: Optional[datetime] = None) -> bool:
        now = now or _now_ist()
        try:
            if now.weekday() >= 5:
                return False
            mins = now.hour * 60 + now.minute
            return (9 * 60 + 15) <= mins <= (15 * 60 + 30)
        except Exception:
            return False

    def _log_raw_candle_payload(self, payload: Any) -> None:
        try:
            status = payload.get("status") if isinstance(payload, dict) else ""
            remarks = payload.get("remarks") if isinstance(payload, dict) else ""
            data = payload.get("data") if isinstance(payload, dict) else payload
            data_keys = sorted(list(data.keys()))[:20] if isinstance(data, dict) else []
            print(
                "[DHAN-CANDLES-RAW] "
                f"status={status} remarks={remarks} data_type={type(data).__name__} "
                f"data_keys={data_keys} sample={_safe_payload_sample(data)}"
            )
        except Exception as exc:
            print(f"[DHAN-CANDLES-RAW] log_error={type(exc).__name__}:{exc}")

    def _rows_to_candles(self, rows: List[Dict[str, Any]], *, limit: int = 100) -> List[Candle]:
        candles: List[Candle] = []
        for row in rows[-int(limit):]:
            try:
                ts_raw = row.get("timestamp") or row.get("start_Time") or row.get("start_time") or row.get("time") or row.get("date") or row.get("datetime")
                ts = _parse_dhan_timestamp(ts_raw)
                if pd.isna(ts):
                    continue
                candles.append(
                    Candle(
                        time=ts.to_pydatetime(),
                        open=float(row.get("open")),
                        high=float(row.get("high")),
                        low=float(row.get("low")),
                        close=float(row.get("close")),
                        volume=_safe_float(row.get("volume")),
                    )
                )
            except Exception:
                continue
        return candles

    def _fallback_candles(self, tok: str, exch: str, interval: int, limit: int) -> List[Candle]:
        methods = ("historical_daily_data", "historical_minute_data", "intraday_daily_data")
        for name in methods:
            fn = getattr(self._raw, name, None)
            if fn is None:
                continue
            try:
                payload = self._call_api_with_retry(
                    f"fallback_{name}",
                    fn,
                    security_id=tok,
                    exchange_segment=exch,
                    instrument_type="INDEX" if exch == "IDX_I" else "EQUITY",
                    from_date=(_now_ist() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    to_date=_now_ist().strftime("%Y-%m-%d"),
                    interval=interval,
                    oi=False,
                )
                self._log_raw_candle_payload(payload)
                candles = self._rows_to_candles(self.normalize_candle_payload(payload), limit=limit)
                if candles:
                    print(f"[DHAN-CANDLES] fallback_source={name} rows={len(candles)}")
                    return candles
            except Exception as exc:
                print(f"[DHAN-CANDLES] fallback_source={name} error={type(exc).__name__}:{exc}")
        return []

    def get_candles_result(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> DhanCallResult:
        try:
            data = self.get_candles(symbol, timeframe=timeframe, limit=limit)
            if not data:
                return DhanCallResult(False, [], "EMPTY_RESPONSE", "Dhan returned no normalized candles")
            return DhanCallResult(ok=True, data=data)
        except Exception as exc:
            return DhanCallResult(False, None, _classify_dhan_error(exc), str(exc))

    def normalize_candle_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            array_keys = ("timestamp", "start_Time", "start_time", "time", "datetime", "date", "open", "high", "low", "close", "volume")
            if isinstance(data, dict) and any(isinstance(data.get(k), list) for k in array_keys):
                times = data.get("timestamp") or data.get("start_Time") or data.get("start_time") or data.get("time") or data.get("datetime") or data.get("date") or []
                opens = data.get("open") or []
                highs = data.get("high") or []
                lows = data.get("low") or []
                closes = data.get("close") or []
                vols = data.get("volume") or []
                total = max(len(times), len(opens), len(highs), len(lows), len(closes), len(vols))
                out = []
                for i in range(total):
                    out.append(
                        {
                            "timestamp": times[i] if i < len(times) else None,
                            "start_Time": times[i] if i < len(times) else None,
                            "open": opens[i] if i < len(opens) else None,
                            "high": highs[i] if i < len(highs) else None,
                            "low": lows[i] if i < len(lows) else None,
                            "close": closes[i] if i < len(closes) else None,
                            "volume": vols[i] if i < len(vols) else None,
                        }
                    )
                return out
        return self._extract_payload_rows(payload)

    def fetch_index_candles(self, instrument_token: str, *, exchange: str = "NSE", limit: int = 100, timeframe: Optional[str] = None, force_historical_only: bool = False) -> Tuple[Optional[List[Candle]], Optional[str]]:
        candles = self.get_candles(str(instrument_token), timeframe=timeframe or "1m", limit=limit)
        return (candles or None), (timeframe or "1m")

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
        order_mode: str = "paper",
    ) -> Order:
        exch, tok, resolved_symbol = self.resolve_exchange_token_symbol(symbol, exchange_hint=exchange)
        if symbol_token:
            tok = str(symbol_token).strip() or tok
        if not tok:
            raise RuntimeError(f"Unable to resolve Dhan order token for {symbol!r}")
        self._guard_live_order(symbol=symbol, token=tok, order_mode=order_mode)
        payload = self._call_api(
            "place_order",
            self._raw.place_order,
            security_id=tok,
            exchange_segment=exch or "NSE_FNO",
            transaction_type=str(side).upper(),
            quantity=int(quantity),
            order_type=str(order_type).upper(),
            product_type=str(product_type).upper(),
            price=float(price or 0.0),
            tag=str(ordertag or "").strip() or None,
        )
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        order_id = str(data.get("orderId") or data.get("order_id") or payload.get("orderId") or "")
        status = str(data.get("orderStatus") or data.get("status") or payload.get("status") or "")
        order = Order()
        order.order_id = order_id
        order.symbol = resolved_symbol or symbol
        order.side = str(side).upper()
        order.quantity = int(quantity)
        order.price = float(price or 0.0)
        order.status = status
        return order

    def cancel_order(self, order_id: str) -> None:
        self._call_api("cancel_order", self._raw.cancel_order, order_id)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        payload = self._call_api("positions", self._raw.get_positions)
        rows = self._extract_payload_rows(payload)
        return [row for row in rows if isinstance(row, dict)]

    def get_holdings(self) -> List[Dict[str, Any]]:
        payload = self._call_api("holdings", self._raw.get_holdings)
        rows = self._extract_payload_rows(payload)
        return [row for row in rows if isinstance(row, dict)]
