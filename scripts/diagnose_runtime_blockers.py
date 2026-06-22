from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import matplotlib
except Exception:
    matplotlib = None  # type: ignore[assignment]

import ui


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve(path_value: Any) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _stale(path_value: Any) -> bool:
    raw = str(path_value or "").strip()
    if not raw:
        return False
    p = Path(raw)
    if not p.is_absolute():
        return False
    try:
        p.resolve().relative_to(REPO_ROOT)
        return False
    except Exception:
        return True


def _artifact_ok(candidate: Dict[str, Any]) -> bool:
    art = _resolve(candidate.get("artifact_dir"))
    model = _resolve(candidate.get("model_path"))
    if model and model.exists():
        return True
    if art and art.exists():
        return any(p.name == "model.pkl" or p.suffix == ".pkl" for p in art.glob("*.pkl"))
    return False


def _candidate_report() -> Dict[str, Any]:
    path = REPO_ROOT / "config" / "paper_forward_candidates.json"
    payload = _load_json(path)
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    rows: List[Dict[str, Any]] = []
    stale_roots = 0
    enabled = 0
    enabled_artifact_ok = 0
    disabled_missing = 0
    for c in candidates:
        if not isinstance(c, dict):
            continue
        art = _resolve(c.get("artifact_dir"))
        stale = _stale(c.get("artifact_dir"))
        stale_roots += int(stale)
        is_enabled = bool(c.get("enabled", False))
        ok = _artifact_ok(c)
        enabled += int(is_enabled)
        enabled_artifact_ok += int(is_enabled and ok)
        disabled_missing += int((not is_enabled) and str(c.get("disabled_reason") or "").upper() == "ARTIFACT_NOT_FOUND")
        rows.append(
            {
                "candidate_id": c.get("candidate_id"),
                "enabled": is_enabled,
                "artifact_dir": c.get("artifact_dir"),
                "resolved_artifact_dir": str(art) if art else "",
                "artifact_exists": bool(art and art.exists()),
                "artifact_ok": ok,
                "stale_root": stale,
                "reason": c.get("disabled_reason") or c.get("last_reason") or "",
            }
        )
    return {
        "path": str(path),
        "total": len(rows),
        "enabled": enabled,
        "artifact_ok_enabled": enabled_artifact_ok,
        "disabled_missing_artifacts": disabled_missing,
        "stale_artifact_roots_found": stale_roots,
        "rows": rows,
    }


def main() -> int:
    broker = ui._normalize_broker_name(os.getenv("SCALPER_BROKER", "mstock"))
    cfg = ui.validate_option_chain_config_for_active_broker(broker)
    candidates = _candidate_report()
    candle_rows = 0
    option_rows = 0
    readiness = "READY"
    if cfg.get("missing_keys"):
        readiness = "BLOCKED_BROKER_CONFIG"
    elif candle_rows <= 0:
        readiness = "BLOCKED_NO_CANDLES"
    elif option_rows <= 0:
        readiness = "BLOCKED_NO_OPTION_CHAIN"
    elif candidates["artifact_ok_enabled"] <= 0:
        readiness = "BLOCKED_NO_ACTIVE_ARTIFACT"
    report = {
        "repo_root": str(REPO_ROOT),
        "selected_broker": broker,
        "env_broker_keys": {
            "SCALPER_BROKER": os.getenv("SCALPER_BROKER", ""),
            "DHAN_CLIENT_ID_present": bool(os.getenv("DHAN_CLIENT_ID", "").strip()),
            "DHAN_ACCESS_TOKEN_present": bool(os.getenv("DHAN_ACCESS_TOKEN", "").strip()),
            "DHAN_UNDERLYING_SECURITY_ID_present": bool(
                os.getenv("DHAN_UNDERLYING_SECURITY_ID", "").strip()
                or os.getenv("DHAN_NIFTY_SECURITY_ID", "").strip()
            ),
            "MSTOCK_ACCESS_TOKEN_present": bool(os.getenv("MSTOCK_ACCESS_TOKEN", "").strip()),
        },
        "gui_runtime_broker_source": "env/script",
        "candles": {"rows": candle_rows, "source": "not_available_in_static_script"},
        "option_chain": {"rows": option_rows, "source": "not_available_in_static_script"},
        "broker_config_validation": cfg,
        "paper_forward_candidates": candidates,
        "router_readiness": readiness,
        "matplotlib_version": getattr(matplotlib, "__version__", "not_imported"),
        "chart_text_safety_enabled": hasattr(__import__("chart"), "_finite_plot_coord"),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if readiness in {"READY", "BLOCKED_NO_CANDLES", "BLOCKED_NO_OPTION_CHAIN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
