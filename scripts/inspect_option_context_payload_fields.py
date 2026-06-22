from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


REPORT_FIELDS = {
    "underlying_spot": ["underlying_spot", "underlyingSpotPrice", "spot_price", "spotPrice", "spot", "underlying_value"],
    "expiry": ["expiry", "expiryDate", "expDate", "expiry_date"],
    "strike": ["strike", "strikePrice", "strike_price"],
    "option_type": ["option_type", "optionType", "opt_type", "instrument_type"],
    "ltp": ["ltp", "LTP", "lastPrice", "last_price", "LastTradedPrice", "last_price"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
    "volume": ["volume", "vol"],
    "open_interest": ["open_interest", "oi"],
    "change_in_oi": ["change_in_oi", "change_oi", "oi_change"],
    "iv": ["iv", "implied_volatility", "impliedVolatility"],
    "delta": ["delta"],
    "gamma": ["gamma"],
    "theta": ["theta"],
    "vega": ["vega"],
    "bid": ["bid", "bidPrice", "bestBidPrice"],
    "ask": ["ask", "askPrice", "bestAskPrice"],
    "spread": ["spread"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect captured broker payload fields for option-context availability.")
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args()


def walk_keys(obj: Any, keys: Set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(str(key))
            walk_keys(value, keys)
    elif isinstance(obj, list):
        for item in obj:
            walk_keys(item, keys)


def load_payload_files(payload_dir: Path) -> List[Path]:
    return sorted(payload_dir.glob("*.json"))


def field_presence(keys: Iterable[str]) -> Dict[str, bool]:
    key_set = {str(key) for key in keys}
    return {
        field: any(alias in key_set for alias in aliases)
        for field, aliases in REPORT_FIELDS.items()
    }


def verdict_from_presence(present: Dict[str, bool], payload_count: int) -> str:
    if payload_count == 0:
        return "NO_PAYLOAD_CAPTURED"
    has_iv = bool(present["iv"])
    has_bid_ask = bool(present["bid"] and present["ask"])
    has_full_greeks = all(bool(present[name]) for name in ["delta", "gamma", "theta", "vega"])
    if has_iv and has_bid_ask and has_full_greeks:
        return "BROKER_PAYLOAD_HAS_FULL_CONTEXT"
    if has_iv and has_bid_ask:
        return "BROKER_PAYLOAD_HAS_IV_AND_BID_ASK_ONLY"
    if has_iv:
        return "BROKER_PAYLOAD_HAS_IV_ONLY"
    if has_bid_ask:
        return "BROKER_PAYLOAD_HAS_BID_ASK_ONLY"
    return "BROKER_PAYLOAD_MISSING_REQUIRED_CONTEXT"


def main() -> None:
    args = parse_args()
    payload_dir = Path(args.payload_dir)
    files = load_payload_files(payload_dir)
    all_keys: Set[str] = set()
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        walk_keys(data, all_keys)
    present = field_presence(all_keys)
    verdict = verdict_from_presence(present, len(files))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"live_broker_payload_field_availability_{stamp}.json"
    md_path = report_dir / f"live_broker_payload_field_availability_{stamp}.md"
    payload = {
        "generated_at": stamp,
        "payload_dir": str(payload_dir),
        "payload_file_count": int(len(files)),
        "fields_present": present,
        "observed_keys_sample": sorted(all_keys)[:200],
        "final_verdict": verdict,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            ["# Live Broker Payload Field Availability", ""]
            + [f"- Payload dir: `{payload_dir}`", f"- Payload files: `{len(files)}`", f"- Final verdict: `{verdict}`", "", "## Field Presence", ""]
            + [f"- `{name}`: `{present[name]}`" for name in REPORT_FIELDS]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report_json": str(json_path), "report_md": str(md_path), "final_verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
