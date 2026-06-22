from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
LOCAL_DHAN_SDK = REPO_ROOT / "DhanHQ-py-main" / "DhanHQ-py-main" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if LOCAL_DHAN_SDK.exists() and str(LOCAL_DHAN_SDK) not in sys.path:
    sys.path.insert(0, str(LOCAL_DHAN_SDK))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from config import load_strategy_config
from dhan_client import DhanClient, get_dhan_sdk_diagnostics, get_dhan_underlying_security_id, mask_token


def _columns(rows: Iterable[Any]) -> List[str]:
    cols: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            cols.update(str(k) for k in row.keys())
    return sorted(cols)


def _safe(name: str, fn) -> Dict[str, Any]:
    try:
        value = fn()
        return {"ok": True, "detail": "ok", "result": value}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "result": None}


def _sample(value: Any, *, max_items: int = 2) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key in list(value.keys())[:8]:
            item = value.get(key)
            if isinstance(item, list):
                out[key] = item[:max_items]
            elif isinstance(item, dict):
                out[key] = _sample(item, max_items=max_items)
            else:
                out[key] = item
        return out
    if isinstance(value, list):
        return value[:max_items]
    return value


def _raw_structure(payload: Any) -> Dict[str, Any]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    return {
        "status": payload.get("status") if isinstance(payload, dict) else "",
        "remarks": payload.get("remarks") if isinstance(payload, dict) else "",
        "data_type": type(data).__name__,
        "data_keys": sorted(list(data.keys()))[:20] if isinstance(data, dict) else [],
        "sample": _sample(data),
    }


def _pf_candidate_status() -> Dict[str, Any]:
    path = REPO_ROOT / "config" / "paper_forward_candidates.json"
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "candidates": 0,
        "enabled": 0,
        "active": 0,
        "active_candidate_id": "",
        "reason": "CONFIG_NOT_FOUND",
    }
    if not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["reason"] = f"CONFIG_PARSE_ERROR:{type(exc).__name__}"
        return out
    raw = payload.get("candidates", payload if isinstance(payload, list) else []) if isinstance(payload, (dict, list)) else []
    candidates = [c for c in raw if isinstance(c, dict)]
    enabled = [c for c in candidates if bool(c.get("enabled", True))]
    active_id = os.getenv("ACTIVE_CANDIDATE_ID", "").strip() or os.getenv("MSTOCK_ACTIVE_CANDIDATE_ID", "").strip()
    active = [
        c
        for c in enabled
        if bool(c.get("active") or c.get("selected"))
        or (active_id and str(c.get("candidate_id") or c.get("artifact_id") or "").strip() == active_id)
    ]
    if not active and enabled:
        active = [enabled[0]]
    out.update({"candidates": len(candidates), "enabled": len(enabled), "active": len(active)})
    if active:
        candidate = active[0]
        out["active_candidate_id"] = str(candidate.get("candidate_id") or candidate.get("artifact_id") or "")
        artifact_dir = str(candidate.get("artifact_dir") or "").strip()
        if artifact_dir and not (REPO_ROOT / artifact_dir).exists():
            out["reason"] = "ARTIFACT_NOT_FOUND"
        else:
            out["reason"] = "OK"
    elif candidates and all(str(c.get("disabled_reason") or "").upper() == "ARTIFACT_NOT_FOUND" for c in candidates):
        out["reason"] = "ARTIFACT_NOT_FOUND"
    elif candidates:
        out["reason"] = "ROUTER_DISABLED_NO_ACTIVE_CANDIDATE"
    return out


def _intraday_request() -> Dict[str, Any]:
    token = (
        os.getenv("DHAN_NIFTY_SECURITY_ID", "").strip()
        or get_dhan_underlying_security_id("NIFTY")
        or os.getenv("DHAN_UNDERLYING_SECURITY_ID", "").strip()
        or "13"
    )
    exchange = os.getenv("DHAN_NIFTY_EXCHANGE_SEGMENT", os.getenv("DHAN_UNDER_EXCHANGE_SEGMENT", "IDX_I")).strip().upper() or "IDX_I"
    interval = int(os.getenv("DHAN_INTRADAY_INTERVAL", "1") or "1")
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "security_id": token,
        "exchange_segment": exchange,
        "instrument_type": "INDEX" if exchange == "IDX_I" else "EQUITY",
        "from_date": os.getenv("DHAN_CANDLE_FROM_DATE", today).strip() or today,
        "to_date": os.getenv("DHAN_CANDLE_TO_DATE", today).strip() or today,
        "interval": interval,
        "oi": False,
    }


def main() -> int:
    client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
    token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    under_id = os.getenv("DHAN_UNDERLYING_SECURITY_ID", os.getenv("DHAN_UNDER_SECURITY_ID", "")).strip()
    if under_id:
        os.environ["DHAN_UNDERLYING_SECURITY_ID"] = under_id
        os.environ["DHAN_UNDER_SECURITY_ID"] = under_id

    selected_broker = os.getenv("SCALPER_BROKER", "mstock").strip().lower() or "mstock"
    report: Dict[str, Any] = {
        "credentials": {
            "client_id_present": bool(client_id),
            "access_token_present": bool(token),
            "access_token_masked": mask_token(token),
            "underlying_security_id_present": bool(under_id),
            "underlying_security_id": under_id or "(missing)",
            "nifty_security_id": os.getenv("DHAN_NIFTY_SECURITY_ID", "13"),
            "nifty_exchange_segment": os.getenv("DHAN_NIFTY_EXCHANGE_SEGMENT", "IDX_I"),
            "intraday_interval": os.getenv("DHAN_INTRADAY_INTERVAL", "1"),
        },
        "selected_broker": selected_broker,
        "chart_path": "dhan" if selected_broker == "dhan" else "mstock",
        "sdk_import": get_dhan_sdk_diagnostics(),
        "paper_forward_candidate": _pf_candidate_status(),
    }

    print(
        "[DHAN-AUTH] "
        f"source=env client_id_present={bool(client_id)} "
        f"token_present={bool(token)} underlying_id={under_id or '(missing)'}"
    )
    print(
        "[PF-CONFIG] "
        f"path={report['paper_forward_candidate']['path']} "
        f"candidates={report['paper_forward_candidate']['candidates']} "
        f"active={report['paper_forward_candidate']['active']} "
        f"enabled={report['paper_forward_candidate']['enabled']} "
        f"reason={report['paper_forward_candidate']['reason']}"
    )

    if not client_id or not token or not under_id:
        report["final"] = {"summary": "FAIL", "reason": "missing credentials"}
        print(json.dumps(report, indent=2, default=str))
        return 1

    client_status = _safe("client", lambda: DhanClient(load_strategy_config()))
    report["client_init"] = {k: v for k, v in client_status.items() if k != "result"}
    client = client_status.get("result")
    if client is None:
        report["final"] = {"summary": "FAIL", "reason": report["client_init"]["detail"]}
        print(json.dumps(report, indent=2, default=str))
        return 1

    report["auth_profile_check"] = _safe("login", lambda: client.login(interactive=False))

    request = _intraday_request()
    raw_status = _safe("raw_intraday", lambda: client._call_api_with_retry("diagnose_intraday_minute_data", client._raw.intraday_minute_data, **request))
    raw_payload = raw_status.get("result")
    report["raw_intraday_candle_response"] = {
        "ok": raw_status.get("ok"),
        "detail": raw_status.get("detail"),
        "request": request,
        "structure": _raw_structure(raw_payload) if raw_status.get("ok") else {},
    }
    normalized_rows = client.normalize_candle_payload(raw_payload) if raw_status.get("ok") else []
    normalized_candles = client._rows_to_candles(normalized_rows, limit=100)
    report["normalized_candles"] = {
        "rows": len(normalized_candles),
        "columns": ["timestamp", "open", "high", "low", "close", "volume"] if normalized_candles else _columns(normalized_rows),
    }

    ltp_status = _safe("ltp", lambda: client.get_ltp(os.getenv("DHAN_UNDERLYING", "NIFTY") or "NIFTY"))
    report["ltp_result"] = {k: v for k, v in ltp_status.items() if k != "result"}
    report["ltp_result"]["value_present"] = ltp_status.get("result") is not None

    chain_status = _safe("option_chain", lambda: client.get_option_chain(os.getenv("DHAN_UNDERLYING", "NIFTY") or "NIFTY"))
    chain_rows = chain_status.get("result") or []
    report["option_chain_result"] = {
        "ok": bool(chain_status.get("ok") and chain_rows),
        "detail": chain_status.get("detail"),
        "rows": len(chain_rows) if isinstance(chain_rows, list) else 0,
        "columns": _columns(chain_rows if isinstance(chain_rows, list) else []),
    }

    passed = bool(
        report["sdk_import"].get("import_ok")
        and report["client_init"].get("ok")
        and report["auth_profile_check"].get("ok")
        and report["normalized_candles"]["rows"] > 0
        and report["option_chain_result"]["ok"]
    )
    report["final"] = {"summary": "PASS" if passed else "FAIL"}
    print(json.dumps(report, indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
