#!/usr/bin/env python3
"""
scripts/build_paper_forward_artifact_registry.py

Scans the entire repo for candidate artifacts (model.pkl, profiles, manifests, feature schemas, etc.)
Builds a robust mapping from configured candidate_id (from paper_forward_candidates.json) 
to the best matching physical artifact location + files.

Writes: config/paper_forward_artifact_registry.json

Matching strategies (in priority):
1. Exact candidate_id match in metadata or folder name
2. Normalized (strip common suffixes like _t30_*, _2026*, _paper_*)
3. model + preset_family + side_policy
4. Substring / prefix match (if unique)
5. Content scan inside JSONs for candidate_id
6. Fallback to any dir that has model.pkl + feature_schema under artifacts/candidates

For each, records full paths to:
- model
- feature_schema
- candidate_profile / manifest
- dynamic_presets
- metrics / gates
- threshold (from metrics if present)
"""
from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
ARTIFACTS_BASE = ROOT / "artifacts" / "candidates"
REPORTS_DIR = ROOT / "reports"

def _safe_load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

def _normalize(cid: str) -> str:
    if not cid:
        return ""
    s = cid.lower()
    # strip common timestamp / variant suffixes
    s = re.sub(r'[_-](t\d+|paper|2026\d+_\d+|auto|balanced|conservative|directional|symmetric|volatility|cost_survivor).*', '', s)
    s = re.sub(r'[_-]\d{8}_\d{6}', '', s)
    return s.strip("_- ")

def find_all_artifacts() -> List[Dict[str, Any]]:
    """Deep scan for candidate artifact directories."""
    found = []
    if not ARTIFACTS_BASE.exists():
        return found
    for d in ARTIFACTS_BASE.rglob("*"):
        if not d.is_dir():
            continue
        has_model = any((d / n).exists() for n in ("model.pkl",)) or bool(list(d.glob("*.pkl")))
        has_profile = (d / "candidate_profile.json").exists() or (d / d.name / "candidate_profile.json").exists()
        has_manifest = (d / "candidate_manifest.json").exists() or (d / d.name / "candidate_manifest.json").exists()
        has_feature = (d / "feature_schema.json").exists() or (d / d.name / "feature_schema.json").exists()
        has_presets = (d / "dynamic_presets.json").exists() or (d / d.name / "dynamic_presets.json").exists()
        if has_model or has_profile or has_manifest or has_feature:
            cid = d.name
            # try to read cid from metadata
            for meta_name in ("candidate_profile.json", "candidate_manifest.json", "paper_forward_manifest.json"):
                for base in (d, d / d.name):
                    mp = base / meta_name
                    if mp.exists():
                        m = _safe_load(mp)
                        if m and m.get("candidate_id"):
                            cid = m["candidate_id"]
                            break
            found.append({
                "dir": str(d.relative_to(ROOT)),
                "absolute": str(d),
                "candidate_id_raw": d.name,
                "candidate_id": cid,
                "has_model": has_model,
                "has_profile": has_profile,
                "has_manifest": has_manifest,
                "has_feature": has_feature,
                "has_presets": has_presets,
                "files": [str(f.relative_to(ROOT)) for f in d.glob("*") if f.is_file()][:20],
            })
    return found

def build_registry() -> Dict[str, Any]:
    cfg_path = CONFIG_DIR / "paper_forward_candidates.json"
    declared = []
    if cfg_path.exists():
        cfg = _safe_load(cfg_path) or {}
        declared = cfg.get("candidates", cfg if isinstance(cfg, list) else [])

    all_artifacts = find_all_artifacts()

    # Also scan reports for any embedded mappings
    report_hits = {}
    for rp in REPORTS_DIR.glob("*.json"):
        data = _safe_load(rp)
        if not data:
            continue
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("candidates", data.get("selected", []) or [])
        for it in items:
            if isinstance(it, dict) and it.get("candidate_id"):
                report_hits.setdefault(it["candidate_id"], []).append(str(rp.name))

    registry: Dict[str, Dict[str, Any]] = {}
    for cand in declared:
        cid = cand.get("candidate_id") or cand.get("id")
        if not cid:
            continue
        model = cand.get("model_name", "unknown")
        preset = cand.get("preset_family", "")
        side = cand.get("side_policy", "BOTH")
        explicit = cand.get("artifact_dir", "")

        best = None
        best_score = -1
        reasons = []

        norm_cid = _normalize(cid)

        for art in all_artifacts:
            score = 0
            art_cid = art["candidate_id"]
            art_norm = _normalize(art_cid)

            if art_cid == cid:
                score += 100
                reasons.append("exact_id")
            if art_norm == norm_cid:
                score += 60
                reasons.append("normalized_id")
            if model.lower() in art_cid.lower() and preset.lower() in art_cid.lower():
                score += 30
                reasons.append("model+preset")
            if side.lower() in art_cid.lower():
                score += 10
            if cid in art["dir"] or art_cid in cid:
                score += 20
                reasons.append("substring")
            if art["has_model"] and art["has_feature"]:
                score += 15
            if art["has_profile"] or art["has_manifest"]:
                score += 10
            if art["has_presets"]:
                score += 5

            if score > best_score:
                best_score = score
                best = {
                    "dir": art["dir"],
                    "absolute": art["absolute"],
                    "score": score,
                    "match_reasons": reasons,
                    "has_model": art["has_model"],
                    "has_profile": art["has_profile"],
                    "has_feature": art["has_feature"],
                    "has_presets": art["has_presets"],
                }

        entry = {
            "candidate_id": cid,
            "model": model,
            "preset": preset,
            "side": side,
            "explicit_artifact_dir": explicit,
            "resolved_dir": best["dir"] if best else None,
            "model_path": str(Path(best["absolute"]) / "model.pkl") if best and best["has_model"] else None,
            "feature_list_path": str(Path(best["absolute"]) / "feature_schema.json") if best and best["has_feature"] else None,
            "metadata_path": str(Path(best["absolute"]) / "candidate_profile.json") if best and best["has_profile"] else None,
            "threshold": 0.5,  # will be refined later from metrics
            "threshold_path": None,
            "calibrator_path": None,
            "scaler_path": None,
            "dynamic_presets_path": str(Path(best["absolute"]) / "dynamic_presets.json") if best and best["has_presets"] else None,
            "source_files": [best["dir"]] if best else [],
            "load_status": "candidate_loaded_ok" if (best and best["has_model"] and (best["has_profile"] or best["has_feature"])) else "candidate_missing_artifact_paths",
            "match_score": best["score"] if best else 0,
            "match_reasons": best["match_reasons"] if best else ["no_match"],
            "report_sources": report_hits.get(cid, []),
        }
        # Try to extract threshold from metrics if present
        if best:
            mpath = Path(best["absolute"]) / "metrics.json"
            m = _safe_load(mpath)
            if m and "threshold" in str(m).lower():
                # crude extraction
                for k in ("threshold", "selected_threshold"):
                    if k in m:
                        entry["threshold"] = float(m[k])
                        entry["threshold_path"] = str(mpath.relative_to(ROOT))
                        break
        registry[cid] = entry

    out = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "num_declared": len(declared),
        "num_resolved_ok": sum(1 for e in registry.values() if e["load_status"] == "candidate_loaded_ok"),
        "candidates": registry,
    }
    (CONFIG_DIR / "paper_forward_artifact_registry.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[REGISTRY] Wrote config/paper_forward_artifact_registry.json with {len(registry)} entries, {out['num_resolved_ok']} resolved OK")
    return out

if __name__ == "__main__":
    build_registry()
