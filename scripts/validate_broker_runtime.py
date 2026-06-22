from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
LOCAL_DHAN_SDK = REPO_ROOT / "DhanHQ-py-main" / "DhanHQ-py-main" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if LOCAL_DHAN_SDK.exists() and str(LOCAL_DHAN_SDK) not in sys.path:
    sys.path.insert(0, str(LOCAL_DHAN_SDK))

from config import load_api_config, load_strategy_config
from dhan_client import DhanClient, get_dhan_sdk_diagnostics, mask_token, validate_dhan_live_readiness
from mstock_client import MStockTypeBClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only broker runtime validation.")
    parser.add_argument("--broker", choices=["dhan", "mstock"], required=True)
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "reports"))
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _status(ok: bool, detail: str = "", **extra: Any) -> Dict[str, Any]:
    out = {"ok": bool(ok), "detail": detail}
    out.update(extra)
    return out


def _safe_call(name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        keys: List[str] = []
        if isinstance(result, dict):
            keys = list(result.keys())[:10]
        return _status(True, "ok", response_keys=keys, result=result)
    except Exception as exc:
        return _status(False, str(exc), response_keys=[])


def validate_dhan_runtime(symbol: str, *, dry_run: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {"broker": "dhan", "symbol": symbol, "dry_run": bool(dry_run)}
    report["sdk_diagnostics"] = get_dhan_sdk_diagnostics()
    client_id = str(os.getenv("DHAN_CLIENT_ID", "")).strip()
    access_token = str(os.getenv("DHAN_ACCESS_TOKEN", "")).strip()
    under_id = str(os.getenv("DHAN_UNDERLYING_SECURITY_ID", "") or os.getenv("DHAN_UNDER_SECURITY_ID", "")).strip()
    if under_id:
        os.environ["DHAN_UNDERLYING_SECURITY_ID"] = under_id
        os.environ["DHAN_UNDER_SECURITY_ID"] = under_id
    report["credentials"] = {
        "client_id_present": bool(client_id),
        "access_token_present": bool(access_token),
        "access_token_masked": mask_token(access_token),
        "underlying_security_id_present": bool(under_id),
    }
    if not client_id or not access_token or not under_id:
        report["final_verdict"] = "BROKER_RUNTIME_INVALID"
        return report
    runtime = _safe_call("build_client", DhanClient, load_strategy_config())
    report["auth_session_status"] = _status(runtime["ok"], runtime["detail"])
    if not runtime["ok"]:
        report["final_verdict"] = "BROKER_RUNTIME_INVALID"
        return report
    client: DhanClient = runtime["result"]
    expiry = _safe_call("expiry_list", client._resolve_expiry, symbol)
    report["expiry_list_status"] = _status(expiry["ok"], expiry["detail"], expiry=expiry.get("result"))
    readiness = validate_dhan_live_readiness(symbol, client=client)
    report["live_readiness"] = readiness
    report["option_chain_status"] = _status(bool(readiness.get("option_chain_ok")), readiness.get("last_error", "") or ("ok" if readiness.get("option_chain_ok") else "option chain unavailable"))
    report["atm_strike_detection"] = _status(bool(readiness.get("atm_detected")), "atm resolved" if readiness.get("atm_detected") else "atm not resolved")
    report["ce_pe_contract_extraction"] = _status(
        bool(readiness.get("selected_contract_from_live_chain")),
        "live contract extracted" if readiness.get("selected_contract_from_live_chain") else "live contract not validated",
    )
    report["ltp_status"] = _status(bool(readiness.get("ltp_ok")), "ok" if readiness.get("ltp_ok") else (readiness.get("last_error", "") or "ltp unavailable"))
    report["bid_ask_status"] = _status(bool(readiness.get("bid_ask_ok")), "ok" if readiness.get("bid_ask_ok") else (readiness.get("last_error", "") or "bid/ask unavailable"))
    report["security_resolution_status"] = _status(
        bool(readiness.get("selected_security_id")),
        "resolved" if readiness.get("selected_security_id") else "missing token",
        token=str(readiness.get("selected_security_id") or ""),
        symbol=str(readiness.get("selected_symbol") or ""),
    )
    report["positions_status"] = _status(bool(readiness.get("positions_ok")), "ok" if readiness.get("positions_ok") else (readiness.get("last_error", "") or "positions unavailable"))
    report["holdings_status"] = _status(bool(readiness.get("holdings_ok")), "ok" if readiness.get("holdings_ok") else (readiness.get("last_error", "") or "holdings unavailable"))
    report["paper_order_payload_construction"] = _status(
        True,
        "dry-run validation is read-only; no order placement attempted",
        live_order_blocked=True,
    )
    passed = bool(readiness.get("live_order_allowed"))
    client.mark_health_check_passed(passed)
    report["last_api_error"] = client.last_api_error() or str(readiness.get("last_error") or "")
    report["last_successful_api_timestamp"] = readiness.get("last_success_timestamp")
    report["final_verdict"] = "BROKER_RUNTIME_VALID" if passed else "BROKER_RUNTIME_INVALID"
    report["summary"] = "PASS" if passed else "FAIL"
    return report


def validate_mstock_runtime(symbol: str, *, dry_run: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {"broker": "mstock", "symbol": symbol, "dry_run": bool(dry_run)}
    api_key = str(os.getenv("MSTOCK_API_KEY", "")).strip()
    access_token = str(os.getenv("MSTOCK_ACCESS_TOKEN", "")).strip()
    report["credentials"] = {
        "api_key_present": bool(api_key),
        "access_token_present": bool(access_token),
        "access_token_masked": mask_token(access_token),
    }
    if not api_key or not access_token:
        report["final_verdict"] = "BROKER_RUNTIME_INVALID"
        return report
    client = MStockTypeBClient(load_api_config())
    report["auth_session_status"] = _safe_call("login", client.login, interactive=False)
    report["option_chain_status"] = _safe_call("option_chain", client.get_option_chain, symbol)
    report["ltp_status"] = _safe_call("ltp", client.get_ltp, symbol)
    chain_rows = report["option_chain_status"].get("result") if report["option_chain_status"].get("ok") else []
    if chain_rows:
        report["bid_ask_status"] = _safe_call("bid_ask", client.get_bid_ask, str(chain_rows[0].get("symbol") or symbol))
        report["security_resolution_status"] = _safe_call("resolve", client.resolve_exchange_token, str(chain_rows[0].get("symbol") or symbol))
    else:
        report["bid_ask_status"] = _status(False, "no chain rows available")
        report["security_resolution_status"] = _status(False, "no chain rows available")
    report["positions_status"] = _safe_call("positions", client.get_open_positions)
    report["holdings_status"] = _safe_call("holdings", client.get_holdings)
    report["final_verdict"] = "BROKER_RUNTIME_VALID" if all(report[name]["ok"] for name in ["auth_session_status", "option_chain_status", "ltp_status"]) else "BROKER_RUNTIME_INVALID"
    return report


def write_report(payload: Dict[str, Any], report_dir: Path) -> Dict[str, str]:
    stamp = _stamp()
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"broker_validation_{stamp}.json"
    md_path = report_dir / f"broker_validation_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Broker Validation Report",
        "",
        f"- Broker: `{payload.get('broker')}`",
        f"- Symbol: `{payload.get('symbol')}`",
        f"- Dry run: `{payload.get('dry_run')}`",
        f"- Final verdict: `{payload.get('final_verdict')}`",
        f"- Summary: `{payload.get('summary', 'n/a')}`",
        "",
    ]
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"## {key}")
            lines.append("")
            for sub_key, sub_val in value.items():
                if sub_key == "result":
                    continue
                lines.append(f"- `{sub_key}`: `{sub_val}`")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def main() -> None:
    args = parse_args()
    payload = validate_dhan_runtime(args.symbol, dry_run=args.dry_run) if args.broker == "dhan" else validate_mstock_runtime(args.symbol, dry_run=args.dry_run)
    payload["report_paths"] = write_report(payload, Path(args.report_dir))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
