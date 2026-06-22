#!/usr/bin/env python3
"""
scripts/build_paper_forward_preset_registry.py

Discovers real dynamic / paper presets from multiple sources and builds
a canonical registry.

Writes: config/paper_forward_preset_registry.json

Sources (priority order for merging):
- config/dynamic_presets.json (global policy)
- config/presets.json
- config/paper_forward_presets.json
- reports/*dynamic*preset*.json , *preset*.json , *candidate*.json
- Per-candidate dynamic_presets.json inside artifacts (from previous registry)
- candidate_profile.json embedded dynamic_presets

For each preset name/family, records parameters useful for routing:
threshold, min_conf, spread_limit, liquidity, max_trades, side_policy, etc.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_BASE = ROOT / "artifacts" / "candidates"

def _safe_load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

def build_preset_registry() -> Dict[str, Any]:
    presets: Dict[str, Dict[str, Any]] = {}

    # 1. Global dynamic_presets.json (highest priority base)
    for rel in ("config/dynamic_presets.json", "config/presets.json", "config/paper_forward_presets.json"):
        p = ROOT / rel
        data = _safe_load(p)
        if data:
            base = data.get("presets", data.get("dynamic_presets", data))
            if isinstance(base, dict):
                for name, val in base.items():
                    if isinstance(val, dict):
                        presets[name] = {
                            "name": name,
                            "source": str(p.relative_to(ROOT)),
                            "params": val,
                            "side_policy": val.get("side_policy") or val.get("option_side_policy"),
                            "min_confidence": float(val.get("min_confidence") or val.get("effective_threshold_adjustment") or 0.0),
                            "threshold": float(val.get("threshold") or 0.5),
                            "max_trades_per_day": int(val.get("max_trades_per_day", 3)),
                        }

    # 2. Reports
    for pat in ("*dynamic*preset*.json", "*preset*.json", "*candidate*.json"):
        for rp in REPORTS_DIR.glob(pat):
            data = _safe_load(rp)
            if not data:
                continue
            items = data.get("candidates", data.get("presets", [])) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for it in items:
                if not isinstance(it, dict):
                    continue
                name = it.get("preset_family") or it.get("preset") or it.get("name")
                if name and isinstance(it.get("params") or it, dict):
                    val = it.get("params") or it
                    if name not in presets or "report" in str(rp).lower():
                        presets[name] = {
                            "name": name,
                            "source": str(rp.relative_to(ROOT)),
                            "params": val,
                            "side_policy": val.get("side_policy"),
                            "min_confidence": float(val.get("min_confidence", 0.0)),
                            "threshold": float(val.get("threshold", 0.5)),
                        }

    # 3. Per-artifact dynamic_presets.json (candidate specific)
    if ARTIFACTS_BASE.exists():
        for d in ARTIFACTS_BASE.rglob("dynamic_presets.json"):
            data = _safe_load(d)
            if isinstance(data, dict):
                for name, val in data.items():
                    if isinstance(val, dict):
                        cid = d.parent.name
                        key = f"{cid}__{name}"
                        presets[key] = {
                            "name": name,
                            "candidate_specific_for": cid,
                            "source": str(d.relative_to(ROOT)),
                            "params": val,
                            "side_policy": val.get("side_policy"),
                            "min_confidence": float(val.get("min_confidence") or val.get("effective_threshold_adjustment") or 0.0),
                            "threshold": float(val.get("entry_threshold") or val.get("threshold") or 0.5),
                        }

    out = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_presets": len(presets),
        "presets": presets,
    }
    (CONFIG_DIR / "paper_forward_preset_registry.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[PRESET-REGISTRY] Wrote config/paper_forward_preset_registry.json with {len(presets)} presets")
    return out

if __name__ == "__main__":
    build_preset_registry()
