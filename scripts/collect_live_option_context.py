from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
LOCAL_DHAN_SDK = REPO_ROOT / "DhanHQ-py-main" / "DhanHQ-py-main" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if LOCAL_DHAN_SDK.exists() and str(LOCAL_DHAN_SDK) not in sys.path:
    sys.path.insert(0, str(LOCAL_DHAN_SDK))

from config import APIConfig
from mstock_client import MStockTypeBClient
from option_chain_pipeline_lib import validate_option_context_dataset
from dhan_client import get_dhan_sdk_diagnostics

try:
    from dhanhq import DhanContext, dhanhq as DhanHQClient
except Exception:  # pragma: no cover - optional dependency
    DhanContext = None  # type: ignore[assignment]
    DhanHQClient = None  # type: ignore[assignment]


OUTPUT_COLUMNS = [
    "timestamp",
    "underlying_spot",
    "expiry",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "ltp",
    "volume",
    "open_interest",
    "change_in_oi",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "bid",
    "ask",
    "spread",
    "lot_size",
]

SENSITIVE_KEYS = {
    "authorization",
    "access_token",
    "accesstoken",
    "token",
    "password",
    "passwd",
    "api_secret",
    "client_secret",
    "secret",
}


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect live option-context snapshots without fabricating missing fields.")
    parser.add_argument("--broker", default="mstock", choices=["mstock", "dhan"])
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--expiry", default="")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "option_context_live"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--market-hours-only", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--debug-raw-payload", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--max-strikes-around-atm", type=int, default=20)
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "reports"))
    return parser.parse_args()


def build_mstock_client() -> MStockTypeBClient:
    return MStockTypeBClient(
        APIConfig(
            base_url=os.getenv("MSTOCK_BASE_URL", ""),
            api_key=os.getenv("MSTOCK_API_KEY", ""),
            api_secret=os.getenv("MSTOCK_API_SECRET", ""),
            client_id=os.getenv("MSTOCK_CLIENT_ID", ""),
        )
    )


def build_dhan_client() -> Any:
    if DhanContext is None or DhanHQClient is None:
        raise RuntimeError("Dhan SDK is not installed. Install with `pip install dhanhq` to use --broker dhan.")
    client_id = str(os.getenv("DHAN_CLIENT_ID", "")).strip()
    access_token = str(os.getenv("DHAN_ACCESS_TOKEN", "")).strip()
    if not client_id or not access_token:
        raise RuntimeError("Missing required Dhan env vars: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN")
    return DhanHQClient(DhanContext(client_id, access_token))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_option_type(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"CALL", "C"}:
        return "CE"
    if raw in {"PUT", "P"}:
        return "PE"
    return raw


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text in SENSITIVE_KEYS or "authorization" in key_text or "password" in key_text or "secret" in key_text:
                out[str(key)] = "[REDACTED]"
            else:
                out[str(key)] = redact_sensitive(item)
        return out
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    return value


def debug_payload_dir(base_dir: Optional[Path] = None, *, timestamp: Optional[pd.Timestamp] = None) -> Path:
    ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
    root = base_dir or (REPO_ROOT / "data" / "debug_option_context_payloads")
    day_dir = root / str(ts.date())
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def save_debug_payload(*, endpoint: str, payload: Any, source: str, base_dir: Optional[Path] = None, timestamp: Optional[pd.Timestamp] = None) -> Path:
    ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
    day_dir = debug_payload_dir(base_dir, timestamp=ts)
    path = day_dir / f"{ts.strftime('%Y%m%d_%H%M%S')}_{endpoint}.json"
    wrapped = {
        "captured_at": ts.isoformat(),
        "source_endpoint": endpoint,
        "source": source,
        "payload": redact_sensitive(payload),
    }
    path.write_text(json.dumps(wrapped, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def capture_full_quote_payload(
    client: MStockTypeBClient,
    chain_rows: Sequence[Dict[str, Any]],
    *,
    timestamp: pd.Timestamp,
    max_contracts: int = 6,
) -> Optional[Path]:
    exchange_tokens: Dict[str, List[str]] = {}
    for row in list(chain_rows)[: max(1, int(max_contracts))]:
        exchange = str(row.get("exchange") or "NFO").strip().upper() or "NFO"
        token = str(row.get("token") or "").strip()
        if token:
            exchange_tokens.setdefault(exchange, []).append(token)
    if not exchange_tokens:
        return None
    response = client._raw.get_market_quote("FULL", exchange_tokens)  # type: ignore[attr-defined]
    payload = client._safe_json(response, context="collect_live_option_context_debug_full_quote")
    return save_debug_payload(
        endpoint="market_quote_full",
        payload=payload,
        source="mstock_market_quote_full",
        timestamp=timestamp,
    )


def missing_credentials() -> List[str]:
    required = {
        "MSTOCK_API_KEY": os.getenv("MSTOCK_API_KEY", ""),
        "MSTOCK_ACCESS_TOKEN": os.getenv("MSTOCK_ACCESS_TOKEN", ""),
    }
    return [name for name, value in required.items() if not str(value).strip()]


def missing_dhan_credentials() -> List[str]:
    required = {
        "DHAN_CLIENT_ID": os.getenv("DHAN_CLIENT_ID", ""),
        "DHAN_ACCESS_TOKEN": os.getenv("DHAN_ACCESS_TOKEN", ""),
    }
    return [name for name, value in required.items() if not str(value).strip()]


def default_dhan_underlying_security_id(symbol: str) -> Optional[int]:
    mapping = {
        "NIFTY": 13,
        "BANKNIFTY": 25,
        "FINNIFTY": 27,
    }
    return mapping.get(str(symbol or "").strip().upper())


def extract_underlying_spot(client: MStockTypeBClient, chain_rows: Sequence[Dict[str, Any]], symbol: str) -> Optional[float]:
    for row in chain_rows:
        raw = row.get("raw")
        if isinstance(raw, dict):
            for key in ("underlying_spot_price", "underlyingSpotPrice", "spot_price", "spotPrice", "spot", "underlying_value"):
                px = _safe_float(raw.get(key))
                if px is not None:
                    return px
    try:
        return float(client.get_ltp(symbol))
    except Exception:
        return None


def extract_underlying_spot_from_rows(chain_rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    for row in chain_rows:
        raw = row.get("raw")
        if isinstance(raw, dict):
            for key in ("underlying_spot_price", "underlyingSpotPrice", "spot_price", "spotPrice", "spot", "underlying_value", "underlyingPrice"):
                px = _safe_float(raw.get(key))
                if px is not None:
                    return px
    return None


def normalize_option_chain_snapshot(
    chain_rows: Sequence[Dict[str, Any]],
    *,
    timestamp: Any,
    underlying_spot: Optional[float],
    expiry: str = "",
    max_strikes_around_atm: int = 20,
) -> pd.DataFrame:
    ts = pd.Timestamp(timestamp)
    normalized_rows: List[Dict[str, Any]] = []
    for row in chain_rows:
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
        strike = _safe_float(row.get("strike") or raw.get("strikePrice") or raw.get("strike_price") or raw.get("strike"))
        option_type = _normalize_option_type(row.get("option_type") or raw.get("optionType") or raw.get("option_type"))
        expiry_value = row.get("expiry") or raw.get("expiry") or raw.get("expiryDate") or expiry
        expiry_text = str(pd.Timestamp(expiry_value).date()) if expiry_value not in (None, "", "NaT") else str(expiry or "")
        ltp = _safe_float(raw.get("ltp") or raw.get("close") or row.get("ltp") or row.get("close"))
        bid = _safe_float(raw.get("bid") or raw.get("bidPrice") or row.get("bid"))
        ask = _safe_float(raw.get("ask") or raw.get("askPrice") or row.get("ask"))
        spread = (ask - bid) if bid is not None and ask is not None else None
        normalized_rows.append(
            {
                "timestamp": ts.isoformat(),
                "underlying_spot": underlying_spot,
                "expiry": expiry_text,
                "strike": strike,
                "option_type": option_type,
                "open": _safe_float(raw.get("open") or row.get("open") or ltp),
                "high": _safe_float(raw.get("high") or row.get("high") or ltp),
                "low": _safe_float(raw.get("low") or row.get("low") or ltp),
                "close": _safe_float(raw.get("close") or row.get("close") or ltp),
                "ltp": ltp,
                "volume": _safe_float(raw.get("volume") or raw.get("vol") or row.get("volume")),
                "open_interest": _safe_float(raw.get("open_interest") or raw.get("oi") or row.get("open_interest")),
                "change_in_oi": _safe_float(raw.get("change_in_oi") or raw.get("change_oi") or raw.get("oi_change") or row.get("change_in_oi")),
                "iv": _safe_float(raw.get("iv") or raw.get("implied_volatility") or raw.get("impliedVolatility") or row.get("iv")),
                "delta": _safe_float(raw.get("delta") or row.get("delta")),
                "gamma": _safe_float(raw.get("gamma") or row.get("gamma")),
                "theta": _safe_float(raw.get("theta") or row.get("theta")),
                "vega": _safe_float(raw.get("vega") or row.get("vega")),
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "lot_size": _safe_float(raw.get("lot_size") or raw.get("lotsize") or row.get("lot_size")),
            }
        )
    if not normalized_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.DataFrame(normalized_rows)
    if underlying_spot is not None and pd.notna(underlying_spot):
        frame["_atm_distance"] = (pd.to_numeric(frame["strike"], errors="coerce") - float(underlying_spot)).abs()
        frame = frame.sort_values(["_atm_distance", "expiry", "strike", "option_type"], kind="stable")
        max_rows = max(2, int(max_strikes_around_atm)) * 2
        frame = frame.head(max_rows)
    frame = frame.drop(columns=["_atm_distance"], errors="ignore")
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[OUTPUT_COLUMNS].drop_duplicates(subset=["timestamp", "expiry", "strike", "option_type"], keep="last")
    return frame


def _walk_dicts(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        out.append(obj)
        for value in obj.values():
            out.extend(_walk_dicts(value))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk_dicts(item))
    return out


def _find_first_list_of_dicts(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return list(payload)
    for node in _walk_dicts(payload):
        for value in node.values():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                return list(value)
    return []


def parse_dhan_option_chain_payload(payload: Any, *, symbol: str, expiry: str) -> List[Dict[str, Any]]:
    rows = _find_first_list_of_dicts(payload)
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        option_type = _normalize_option_type(
            row.get("optionType")
            or row.get("option_type")
            or row.get("instrument_type")
            or row.get("right")
        )
        strike = _safe_float(row.get("strike") or row.get("strikePrice") or row.get("StrikePrice"))
        if strike is None or option_type not in {"CE", "PE"}:
            continue
        bid = _safe_float(row.get("bid") or row.get("bidPrice") or row.get("bestBidPrice"))
        ask = _safe_float(row.get("ask") or row.get("askPrice") or row.get("bestAskPrice"))
        normalized.append(
            {
                "symbol": str(row.get("tradingSymbol") or row.get("tradingsymbol") or row.get("symbol") or ""),
                "token": str(row.get("securityId") or row.get("security_id") or row.get("securityid") or row.get("SecurityId") or ""),
                "exchange": str(row.get("exchangeSegment") or row.get("exchange_segment") or row.get("exchange") or "NSE_FNO"),
                "strike": strike,
                "option_type": option_type,
                "expiry": str(row.get("expiry") or row.get("Expiry") or expiry),
                "symbol_root": str(symbol or "").strip().upper(),
                "raw": {
                    **row,
                    "ltp": row.get("ltp") or row.get("last_price") or row.get("lastPrice") or row.get("LastTradedPrice"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "open_interest": row.get("open_interest") or row.get("oi") or row.get("openInterest"),
                    "change_in_oi": row.get("change_in_oi") or row.get("changeOi") or row.get("oiChange"),
                    "iv": row.get("iv") or row.get("impliedVolatility"),
                    "delta": row.get("delta"),
                    "gamma": row.get("gamma"),
                    "theta": row.get("theta"),
                    "vega": row.get("vega"),
                    "bid": bid,
                    "ask": ask,
                    "spread": (ask - bid) if bid is not None and ask is not None else row.get("spread"),
                    "lot_size": row.get("lotSize") or row.get("lot_size"),
                    "underlying_spot_price": row.get("underlyingPrice") or row.get("underlying_spot_price"),
                },
            }
        )
    return normalized


def resolve_dhan_expiry(dhan_client: Any, *, symbol: str, expiry: str) -> str:
    if expiry:
        return expiry
    under_security_id = default_dhan_underlying_security_id(symbol)
    if under_security_id is None:
        raise RuntimeError(f"No default Dhan underlying security id configured for symbol {symbol!r}. Pass --expiry and set DHAN_UNDER_SECURITY_ID if needed.")
    response = dhan_client.expiry_list(
        under_security_id=under_security_id,
        under_exchange_segment="IDX_I",
    )
    candidates: List[str] = []
    for node in _walk_dicts(response):
        for value in node.values():
            if isinstance(value, list):
                candidates.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    clean = sorted({item[:10] for item in candidates if len(str(item)) >= 10})
    if not clean:
        raise RuntimeError("Dhan expiry_list returned no usable expiries.")
    return clean[0]


def append_snapshot(frame: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    if frame.empty:
        raise RuntimeError("No option-context rows to save.")
    ts = pd.Timestamp(frame["timestamp"].iloc[0])
    day_dir = output_dir / str(ts.date())
    day_dir.mkdir(parents=True, exist_ok=True)
    output_path = day_dir / f"option_context_{symbol}_{ts.strftime('%Y%m%d_%H%M%S')}.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def market_hours_open(now: Optional[pd.Timestamp] = None) -> bool:
    now = now or pd.Timestamp.now(tz="Asia/Kolkata")
    return now.weekday() < 5 and ((now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and (now.hour < 15 or (now.hour == 15 and now.minute <= 30)))


def configure_logging(report_dir: Path) -> logging.Logger:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / f"collect_live_option_context_{utc_now_stamp()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("collect_live_option_context")
    logger.info("Logging to %s", log_path)
    return logger


def write_failure_report(*, report_dir: Path, args: argparse.Namespace, reason: str, error: str) -> Dict[str, str]:
    stamp = utc_now_stamp()
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"collect_live_option_context_failure_{stamp}.json"
    md_path = report_dir / f"collect_live_option_context_failure_{stamp}.md"
    payload = {
        "generated_at": stamp,
        "reason": reason,
        "error": error,
        "symbol": str(args.symbol),
        "dry_run": bool(args.dry_run),
        "debug_raw_payload": bool(args.debug_raw_payload),
        "market_hours_only": bool(args.market_hours_only),
        "final_verdict": "BROKER_SESSION_UNAVAILABLE",
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Live Option Context Collector Failure",
                "",
                f"- Reason: `{reason}`",
                f"- Error: `{error}`",
                f"- Dry run: `{bool(args.dry_run)}`",
                f"- Debug raw payload: `{bool(args.debug_raw_payload)}`",
                f"- Final verdict: `BROKER_SESSION_UNAVAILABLE`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path)}


def collect_once(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    snapshot_ts = pd.Timestamp.now(tz="Asia/Kolkata").floor("min")
    if str(args.broker) == "mstock":
        missing = missing_credentials()
        if missing:
            raise RuntimeError(f"Missing required broker credentials/session env vars: {', '.join(missing)}")
        client = build_mstock_client()
        chain = client.get_option_chain(str(args.symbol))
        if args.debug_raw_payload:
            raw_rows = [row.get("raw", row) for row in chain]
            save_debug_payload(endpoint="option_chain", payload=raw_rows, source="mstock_option_chain", timestamp=snapshot_ts)
        if args.expiry:
            chain = [row for row in chain if str(row.get("expiry") or "")[:10] == str(args.expiry)]
        spot = extract_underlying_spot(client, chain, str(args.symbol))
    else:
        sdk_diag = get_dhan_sdk_diagnostics()
        if bool(args.dry_run):
            logger.info("Dhan SDK diagnostics: %s", json.dumps(sdk_diag, ensure_ascii=True))
        missing = missing_dhan_credentials()
        if missing:
            raise RuntimeError(f"Missing required broker credentials/session env vars: {', '.join(missing)}")
        client = build_dhan_client()
        resolved_expiry = resolve_dhan_expiry(client, symbol=str(args.symbol), expiry=str(args.expiry or ""))
        under_security_id = int(os.getenv("DHAN_UNDER_SECURITY_ID", str(default_dhan_underlying_security_id(str(args.symbol)) or 0)))
        payload = client.option_chain(
            under_security_id=under_security_id,
            under_exchange_segment=os.getenv("DHAN_UNDER_EXCHANGE_SEGMENT", "IDX_I"),
            expiry=resolved_expiry,
        )
        if args.debug_raw_payload:
            save_debug_payload(endpoint="option_chain", payload=payload, source="dhan_option_chain", timestamp=snapshot_ts)
        chain = parse_dhan_option_chain_payload(payload, symbol=str(args.symbol), expiry=resolved_expiry)
        spot = extract_underlying_spot_from_rows(chain)
    normalized = normalize_option_chain_snapshot(
        chain,
        timestamp=snapshot_ts,
        underlying_spot=spot,
        expiry=str(args.expiry or ""),
        max_strikes_around_atm=int(args.max_strikes_around_atm),
    )
    if args.debug_raw_payload and str(args.broker) == "mstock":
        try:
            capture_full_quote_payload(client, chain, timestamp=snapshot_ts)
        except Exception as exc:
            logger.warning("FULL quote debug payload capture failed: %s", exc)
    output_path = None
    validation = None
    if not bool(args.dry_run):
        output_path = append_snapshot(normalized, Path(args.output_dir), str(args.symbol))
        validation = validate_option_context_dataset(output_path, report_dir=Path(args.report_dir))
        logger.info("Saved %s rows to %s", len(normalized), output_path)
    else:
        logger.info("Dry-run mode enabled; normalized rows prepared but not written.")
    return {
        "output_path": str(output_path) if output_path else None,
        "row_count": int(len(normalized)),
        "validation": validation,
        "dry_run": bool(args.dry_run),
        "debug_payload_dir": str(debug_payload_dir(timestamp=snapshot_ts)) if bool(args.debug_raw_payload) else None,
        "broker": str(args.broker),
        "sdk_diagnostics": sdk_diag if str(args.broker) == "dhan" else None,
    }


def main() -> None:
    args = parse_args()
    logger = configure_logging(Path(args.report_dir))
    results: List[Dict[str, Any]] = []
    try:
        while True:
            if bool(args.market_hours_only) and not market_hours_open():
                logger.info("Outside market hours, sleeping for %s seconds", args.interval_seconds)
                if args.once:
                    break
                time.sleep(max(1, int(args.interval_seconds)))
                continue
            payload = collect_once(args, logger)
            results.append(payload)
            if args.once:
                break
            time.sleep(max(1, int(args.interval_seconds)))
        print(json.dumps({"runs": results, "final_verdict": "DRY_RUN_OK" if bool(args.dry_run) else "COLLECTION_OK"}, indent=2))
    except Exception as exc:
        report_paths = write_failure_report(
            report_dir=Path(args.report_dir),
            args=args,
            reason="broker_session_or_market_data_unavailable",
            error=str(exc),
        )
        print(
            json.dumps(
                {
                    "runs": results,
                    "error": str(exc),
                    "failure_reports": report_paths,
                    "final_verdict": "BROKER_SESSION_UNAVAILABLE",
                },
                indent=2,
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
