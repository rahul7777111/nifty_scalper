"""Debug Paper Forward live input readiness.

Read-only diagnostics only. This script never places broker orders.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_forward_engine import PaperForwardDataStatus  # noqa: E402
from ui import (  # noqa: E402
    derive_spot_from_option_chain_payload,
    validate_option_chain_config_for_active_broker,
)


CONFIG_PATH = REPO_ROOT / "config" / "paper_forward_candidates.json"


def _selected_broker() -> str:
    broker = str(os.getenv("SCALPER_BROKER", "mstock") or "mstock").strip().lower()
    if broker in {"mstocks", "m.stock", "m_stock"}:
        broker = "mstock"
    return broker if broker in {"mstock", "dhan"} else "mstock"


def _load_candidates() -> list[dict[str, Any]]:
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("candidates", payload if isinstance(payload, list) else [])
    return [row for row in rows if isinstance(row, dict)]


def _candidate_key(row: dict[str, Any]) -> tuple[str, ...]:
    threshold = row.get("threshold")
    if threshold in (None, ""):
        threshold = row.get("selected_threshold") or row.get("entry_threshold") or ""
    return (
        str(row.get("candidate_id") or ""),
        str(row.get("artifact_dir") or ""),
        str(row.get("model_name") or ""),
        str(row.get("preset_family") or ""),
        str(row.get("side_policy") or ""),
        str(threshold or ""),
    )


def _artifact_exists(row: dict[str, Any]) -> bool:
    value = row.get("artifact_dir")
    if not value:
        return False
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.exists()


def _try_fetch_chain(broker: str) -> tuple[list[dict[str, Any]], str]:
    cfg = validate_option_chain_config_for_active_broker(broker)
    if cfg.get("missing_keys"):
        return [], "broker_config_missing"
    try:
        if broker == "dhan":
            from dhan_client import DhanClient

            client = DhanClient()
            rows = client.get_option_chain(str(cfg.get("underlying") or "NIFTY"))
        else:
            from config import load_api_config
            from mstock_client import MStockTypeBClient

            client = MStockTypeBClient(load_api_config())
            token = os.getenv("MSTOCK_ACCESS_TOKEN", "")
            if token:
                setattr(client, "access_token", token)
            rows = client.get_option_chain(str(cfg.get("underlying") or "NIFTY"))
        return list(rows or []), "ok"
    except Exception as exc:
        return [], f"fetch_error:{type(exc).__name__}:{exc}"


def main() -> int:
    broker = _selected_broker()
    cfg = validate_option_chain_config_for_active_broker(broker)
    token_key = "DHAN_ACCESS_TOKEN" if broker == "dhan" else "MSTOCK_ACCESS_TOKEN"
    auth_status = "AUTH_TOKEN_PRESENT" if os.getenv(token_key) else "TOKEN_MISSING"

    chain, chain_status = _try_fetch_chain(broker)
    spot, spot_source = derive_spot_from_option_chain_payload(chain)
    candles = []
    candle_count = 0

    candidates = _load_candidates()
    seen: set[tuple[str, ...]] = set()
    duplicate_candidates = 0
    enabled_candidates = 0
    missing_artifact_candidates = 0
    for row in candidates:
        key = _candidate_key(row)
        if key in seen:
            duplicate_candidates += 1
        seen.add(key)
        if row.get("enabled", True):
            enabled_candidates += 1
        if not _artifact_exists(row):
            missing_artifact_candidates += 1

    status = PaperForwardDataStatus.from_snapshot(
        {
            "broker_name": broker,
            "broker_auth": "AUTH_OK" if auth_status == "AUTH_TOKEN_PRESENT" else "TOKEN_MISSING",
            "spot": spot,
            "candles": candles,
            "candle_count": candle_count,
            "option_chain_status": chain_status,
            "option_chain_error": "" if chain_status == "ok" else chain_status,
        },
        chain,
    )

    print(f"selected broker: {broker}")
    print(f"auth status: {auth_status}")
    print(f"missing config keys: {','.join(cfg.get('missing_keys') or []) or 'none'}")
    print(f"selected expiry: {cfg.get('selected_expiry')}")
    print(f"underlying: {cfg.get('underlying')}")
    print(f"spot source and value: {spot_source or 'none'} {spot if spot is not None else 'n/a'}")
    print(f"option chain row count: {len(chain)}")
    print(f"option chain status: {chain_status}")
    print(f"candle count: {candle_count}")
    print(f"candidate config path: {CONFIG_PATH}")
    print(f"total candidates: {len(candidates)}")
    print(f"enabled candidates: {enabled_candidates}")
    print(f"duplicate candidates: {duplicate_candidates}")
    print(f"missing artifact candidates: {missing_artifact_candidates}")
    print(f"final data_quality: {status.as_dict().get('data_quality_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
