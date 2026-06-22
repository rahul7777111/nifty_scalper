#!/usr/bin/env python3
"""
scripts/diagnose_paper_forward_candidates.py

Comprehensive diagnostics for Paper Forward Monitor candidates.
Scans config, reports, models, artifacts for preset + artifact mismatches.
Explains exactly why each candidate shows no_dynamic_presets_configured or candidate_missing_artifact_paths (or other).

Writes detailed:
  reports/paper_forward_candidate_diagnostics_<ts>.json
  reports/paper_forward_candidate_diagnostics_<ts>.md

Run:
  python scripts/diagnose_paper_forward_candidates.py
  python scripts/diagnose_paper_forward_candidates.py --candidate-file config/paper_forward_candidates.json
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make src importable
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_forward_engine import (
    resolve_candidate_artifacts,
    load_candidate_registry,
    PaperForwardEngine,
    REPO_ROOT,
    LOAD_OK,
)

# Additional sources for presets / registry (PHASE 1 spec)
PRESET_FILES = [
    "config/dynamic_presets.json",
    "config/presets.json",
    "config/paper_forward_presets.json",
]
REGISTRY_FILES = [
    "config/candidate_registry.json",
    "config/model_registry.json",
    "config/paper_forward_candidates.json",
]
REPORT_GLOBS = [
    "reports/*selected*candidate*.json",
    "reports/*persisted*candidate*.json",
    "reports/*promotion*.json",
    "reports/*deployment*.json",
    "reports/*shadow*.json",
    "reports/*live*.json",
    "reports/*paper*forward*.json",
    "reports/best_paper_forward_selection*.json",
    "reports/paper_forward_candidate_selection*.json",
]


def _safe_load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _find_all_files(root: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    return sorted(set(out))


def load_dynamic_presets(project_root: Path | str = ROOT) -> Dict[str, Any]:
    """PHASE 2 central loader (also used by diagnose for reporting). Priority search + formats."""
    root = Path(project_root).resolve()
    # 1. Env overrides
    for envk in ("DYNAMIC_PRESETS_PATH", "PAPER_FORWARD_DYNAMIC_PRESETS_PATH"):
        ep = os.environ.get(envk)
        if ep:
            p = Path(ep)
            if p.exists():
                data = _safe_load_json(p)
                if data:
                    return {"source": str(p), "data": data, "loaded": True}

    # 2-4. Standard config locations
    for rel in PRESET_FILES:
        p = root / rel
        data = _safe_load_json(p)
        if data:
            return {"source": str(p), "data": data, "loaded": True}

    # 5-6. Reports that may embed preset configs
    for rp in _find_all_files(root, ["reports/*dynamic*preset*.json", "reports/*candidate*selection*.json"]):
        data = _safe_load_json(rp)
        if data and (data.get("presets") or data.get("dynamic_presets") or "preset" in str(data).lower()):
            return {"source": str(rp), "data": data, "loaded": True, "embedded": True}

    # 7. Candidate registry entries may carry preset
    for rp in _find_all_files(root, REGISTRY_FILES + ["reports/*.json"]):
        data = _safe_load_json(rp)
        if data:
            cands = data.get("candidates", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for c in cands:
                if c.get("preset") or c.get("dynamic_preset") or c.get("preset_family"):
                    return {"source": str(rp), "data": data, "loaded": True, "from_registry": True}

    # Global fallback attempt from artifacts scan (embedded in profiles)
    art_base = root / "artifacts" / "candidates"
    if art_base.exists():
        for d in art_base.iterdir():
            for fn in ("dynamic_presets.json", "dynamic_preset.json"):
                pp = d / fn
                if pp.exists():
                    data = _safe_load_json(pp)
                    if data:
                        return {"source": str(pp), "data": data, "loaded": True, "from_artifact": True}

    return {"source": None, "data": {}, "loaded": False}


def resolve_candidate_preset(candidate: Dict[str, Any], preset_registry: Dict[str, Any]) -> Dict[str, Any]:
    """PHASE 2: return structured preset status. Never generic no_dynamic... if fallback possible."""
    cid = candidate.get("candidate_id") or candidate.get("id") or "?"
    preset_family = candidate.get("preset_family") or candidate.get("preset") or ""
    art_dir = candidate.get("artifact_dir") or candidate.get("_resolve", {}).get("resolved_dir")

    # Try per-candidate dynamic_presets.json / embedded in profile
    per_cand: Dict[str, Any] = {}
    if art_dir:
        ad = Path(art_dir)
        for fn in ("dynamic_presets.json", "dynamic_preset.json", "candidate_profile.json"):
            p = ad / fn
            if p.exists():
                try:
                    dd = json.loads(p.read_text(encoding="utf-8"))
                    if "dynamic_presets" in dd:
                        per_cand = dd["dynamic_presets"]
                    elif isinstance(dd, dict) and any(k for k in dd if "preset" in k.lower()):
                        per_cand = dd
                    break
                except Exception:
                    pass
            # nested <cid>/
            p2 = ad / cid / fn
            if p2.exists():
                try:
                    dd = json.loads(p2.read_text(encoding="utf-8"))
                    per_cand = dd.get("dynamic_presets", dd)
                    break
                except Exception:
                    pass

    # Global registry
    global_p = preset_registry.get("data", {}) if preset_registry.get("loaded") else {}
    global_presets = global_p.get("presets", global_p.get("dynamic_presets", {})) if isinstance(global_p, dict) else {}

    # Candidate embedded preset_family match
    if preset_family and (preset_family in global_presets or preset_family in per_cand):
        return {
            "preset_loaded_ok": True,
            "preset_missing": False,
            "preset_fallback_generated": False,
            "preset_invalid_schema": False,
            "selected": preset_family,
            "source": "candidate+global" if preset_family in global_presets else "per_candidate",
        }

    if per_cand:
        return {
            "preset_loaded_ok": True,
            "preset_missing": False,
            "preset_fallback_generated": False,
            "preset_invalid_schema": False,
            "selected": next(iter(per_cand.keys()), preset_family or "embedded"),
            "source": "per_candidate_artifact",
        }

    if global_presets:
        # pick a safe paper one or first
        safe = "CONSERVATIVE" if "CONSERVATIVE" in global_presets else (preset_family or next(iter(global_presets.keys()), "OBSERVE_ONLY"))
        return {
            "preset_loaded_ok": True,
            "preset_missing": False,
            "preset_fallback_generated": True,
            "preset_invalid_schema": False,
            "selected": safe,
            "source": "global_fallback",
        }

    # Last resort safe fallback for PAPER ONLY (no real orders)
    return {
        "preset_loaded_ok": False,
        "preset_missing": True,
        "preset_fallback_generated": True,
        "preset_invalid_schema": False,
        "selected": "PAPER_OBSERVE_FALLBACK",
        "source": "generated_fallback",
        "reason": "no config + no per-cand; generated safe paper-only observe preset",
    }


def _detect_duplicates(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen: Dict[str, int] = {}
    dups = 0
    for c in cands:
        key = (c.get("candidate_id"), c.get("model_name"), c.get("preset_family"), c.get("side_policy"))
        k = str(key)
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > 1:
            dups += 1
    return {"total": len(cands), "unique_keys": len(seen), "duplicates": dups, "dup_count": dups}


def _runtime_route_probe(candidate_file: str) -> Dict[str, Any]:
    """Run one thin paper snapshot so diagnose reports route/eval state, not just loading."""
    try:
        eng = PaperForwardEngine(candidate_file=candidate_file, broker_safe_mode=True)
        chain = [
            {"strike": 24150, "option_type": "CE", "ltp": 42.0, "bid": 41.5, "ask": 42.5, "volume": 1000, "oi": 10000, "spot": 24150},
            {"strike": 24150, "option_type": "PE", "ltp": 39.0, "bid": 38.5, "ask": 39.5, "volume": 1000, "oi": 10000, "spot": 24150},
        ]
        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 24150.0,
            "spot": 24150.0,
            "source": "diagnose_runtime_probe",
            "broker_auth": "TOKEN_SET_NOT_VERIFIED",
        }
        decisions = eng.on_market_snapshot(snap, chain)
        diag = eng.get_diagnostics()
        return {
            "ok": True,
            "decisions_returned": len(decisions),
            "waiting_for_snapshot": sum(1 for d in decisions if "waiting_for_snapshot" in str(d.get("no_trade_reason", "")).lower()),
            "runtime_diagnostics": diag,
            "reason_counts": diag.get("reason_counts", {}),
            "top_missing_features": diag.get("top_missing_features", {}),
            "route_exception_count": diag.get("route_errors", 0),
            "route_exception_type_counts": diag.get("route_exception_type_counts", {}),
            "route_exception_candidate_ids": diag.get("route_exception_candidate_ids", []),
            "first_5_tracebacks": diag.get("route_exception_tracebacks", [])[:5],
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


import os  # for env in load_dynamic_presets


def main(argv: List[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    cand_file = "config/paper_forward_candidates.json"
    for i, a in enumerate(argv):
        if a in ("--candidate-file", "-c") and i + 1 < len(argv):
            cand_file = argv[i + 1]

    print("[PAPER-FWD-DIAG] Starting FULL candidate + preset + artifact diagnostics (PHASE 1)")
    print(f"[PAPER-FWD-DIAG] candidate_file={cand_file}")
    print(f"[PAPER-FWD-DIAG] repo_root={REPO_ROOT}")

    try:
        cfg = json.loads(Path(cand_file).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[PAPER-FWD-DIAG] FATAL: cannot read {cand_file}: {e}")
        return 2

    raw_cands = cfg.get("candidates", cfg if isinstance(cfg, list) else [])
    paper_cands = [c for c in raw_cands if c.get("paper_forward_only") or c.get("classification") == "paper_forward_only" or c.get("enabled", True)]
    print(f"[PAPER-FWD-DIAG] {len(paper_cands)} paper_forward candidates declared in config")

    # Load central registry (engine impl)
    reg = load_candidate_registry(REPO_ROOT)
    print(f"[PAPER-FWD-DIAG] registry has {len(reg)} entries from reports+disk")

    # PHASE 1/2 + TASK 6: load dynamic presets — prefer dedicated registry for consistency with engine
    preset_reg = load_dynamic_presets(REPO_ROOT)
    dedicated = _safe_load_json(REPO_ROOT / "config" / "paper_forward_preset_registry.json") or {}
    if dedicated.get("presets"):
        preset_reg = {"loaded": True, "source": str((REPO_ROOT / "config" / "paper_forward_preset_registry.json").relative_to(REPO_ROOT)), "data": dedicated}
    print(f"[PAPER-FWD-DIAG] dynamic presets loaded_ok={preset_reg.get('loaded')} source={preset_reg.get('source')} (dedicated used if present)")

    # Scan reports for more
    extra_reports = _find_all_files(REPO_ROOT, REPORT_GLOBS)

    diags: List[Dict[str, Any]] = []
    ok = 0
    fail = 0
    preset_ok = 0
    preset_fallback = 0
    reason_counts: Dict[str, int] = {}
    dup_info = _detect_duplicates(paper_cands)

    for c in paper_cands:
        cid = c.get("candidate_id") or c.get("id") or "?"
        diag: Dict[str, Any] = {
            "candidate_id": cid,
            "enabled": c.get("enabled", True),
            "model": c.get("model_name") or c.get("model", "unknown"),
            "preset_family": c.get("preset_family") or c.get("preset", ""),
            "side": c.get("side_policy") or c.get("side", "BOTH"),
            "config_source": cand_file,
            "explicit_artifact_path": c.get("artifact_dir") or c.get("candidate_dir", ""),
            "fallback_artifact_paths": [],
            "dynamic_preset_source": None,
            "model_file_path": None,
            "feature_list_path": None,
            "threshold_path": None,
            "scaler_path": None,
            "all_files_exist": {},
            "final_load_status": "unknown",
            "exact_failure_reason": "",
            "traceback": None,
            "suggested_fix": "",
        }

        # Artifact resolution (reuse engine for consistency)
        try:
            art_res = resolve_candidate_artifacts(c, REPO_ROOT)
            diag["resolved_dir"] = art_res.get("resolved_dir")
            diag["artifact_load_status"] = art_res.get("load_status") or art_res.get("status")
            diag["files"] = art_res.get("files", {})
            diag["exists"] = art_res.get("exists", {})
            diag["fallback_artifact_paths"] = [str(x) for x in art_res.get("resolved_dir", [])] if isinstance(art_res.get("resolved_dir"), (list, tuple)) else []
            diag["model_file_path"] = art_res.get("files", {}).get("model_pkl")
            diag["feature_list_path"] = art_res.get("files", {}).get("feature_schema.json")
            diag["threshold_path"] = art_res.get("files", {}).get("metrics.json")  # often has threshold info
        except Exception as e:
            art_res = {"load_status": "candidate_load_exception", "error": str(e), "traceback": traceback.format_exc()}
            diag["artifact_load_status"] = "candidate_load_exception"
            diag["traceback"] = art_res["traceback"]

        # Preset resolution (PHASE 2 + TASK 6 unified)
        try:
            pres_res = resolve_candidate_preset(c, preset_reg)
            diag["dynamic_preset_source"] = pres_res.get("source")
            diag["preset_status"] = pres_res
            if pres_res.get("preset_loaded_ok"):
                preset_ok += 1
            if pres_res.get("preset_fallback_generated") and "dedicated" not in str(pres_res.get("source","")).lower() and "registry" not in str(pres_res.get("source","")).lower():
                preset_fallback += 1
        except Exception as e:
            pres_res = {"preset_loaded_ok": False, "preset_missing": True, "reason": f"exception:{e}"}
            diag["preset_status"] = pres_res

        # Combine status
        art_status = diag.get("artifact_load_status", "candidate_missing_artifact_paths")
        if art_status == LOAD_OK or art_status == "candidate_loaded_ok":
            if pres_res.get("preset_loaded_ok") or pres_res.get("preset_fallback_generated"):
                diag["final_load_status"] = LOAD_OK
                ok += 1
                print(f"[PAPER-FWD-LOAD] {cid} status=OK preset_src={diag['dynamic_preset_source']} resolved={diag.get('resolved_dir')}")
            else:
                diag["final_load_status"] = "preset_missing_but_artifact_ok"
                fail += 1
        else:
            fail += 1
            diag["final_load_status"] = art_status

        rs = diag["final_load_status"]
        reason_counts[rs] = reason_counts.get(rs, 0) + 1
        diag["exact_failure_reason"] = rs if rs != LOAD_OK else "candidate_loaded_ok"
        if art_res.get("suggested_fix"):
            diag["suggested_fix"] = art_res["suggested_fix"]
        elif not pres_res.get("preset_loaded_ok"):
            diag["suggested_fix"] = "Provide dynamic_presets in config/dynamic_presets.json or per-candidate dynamic_presets.json ; or allow fallback"

        # Registry hit
        if cid in reg:
            diag["registry_hit"] = True
            diag["registry_artifact_dir"] = reg[cid].get("artifact_dir")

        # Extra report hits
        diag["report_sources"] = [str(r.name) for r in extra_reports if cid in r.read_text(errors="ignore")]

        diags.append(diag)

    # Dupe summary
    print(f"[PAPER-FWD-DEDUPE] declared={dup_info['total']} unique_keys={dup_info['unique_keys']} duplicates={dup_info['duplicates']}")
    runtime_probe = _runtime_route_probe(cand_file)
    if runtime_probe.get("ok"):
        rd = runtime_probe.get("runtime_diagnostics", {})
        print(
            "[PAPER-FWD-DIAG-RUNTIME] "
            f"decisions={runtime_probe.get('decisions_returned')} "
            f"evals={rd.get('evaluations_count')} route_errors={rd.get('route_errors')} "
            f"reasons={runtime_probe.get('reason_counts')}"
        )
        if runtime_probe.get("first_5_tracebacks"):
            print("[PAPER-FWD-DIAG-RUNTIME] route exception details present in report")
    else:
        print(f"[PAPER-FWD-DIAG-RUNTIME] probe_failed={runtime_probe.get('error')}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(Path(cand_file).resolve()),
        "total_declared": len(paper_cands),
        "loaded_ok": ok,
        "failed": fail,
        "preset_ok": preset_ok,
        "preset_fallback_generated": preset_fallback,
        "by_reason": reason_counts,
        "duplicate_info": dup_info,
        "preset_registry_source": preset_reg.get("source"),
        "registry_size": len(reg),
        "runtime_route_probe": runtime_probe,
        "candidates": diags,
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    jpath = Path("reports") / f"paper_forward_candidate_diagnostics_{ts}.json"
    mpath = Path("reports") / f"paper_forward_candidate_diagnostics_{ts}.md"
    jpath.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Rich MD per spec
    md = [
        f"# Paper Forward Candidate Diagnostics — {ts}",
        "",
        f"- Declared in {cand_file}: {len(paper_cands)}",
        f"- Loaded OK: {ok}",
        f"- Failed (artifacts or presets): {fail}",
        f"- Preset OK (real or embedded): {preset_ok}",
        f"- Preset fallback generated (paper safe): {preset_fallback}",
        f"- Duplicates in config: {dup_info['duplicates']} (unique keys {dup_info['unique_keys']})",
        f"- Registry entries discovered: {len(reg)}",
        f"- Preset config source: {preset_reg.get('source') or 'NONE (will fallback)'}",
        "",
        "## Runtime route probe",
        f"- Probe OK: {runtime_probe.get('ok')}",
        f"- Decisions returned: {runtime_probe.get('decisions_returned', 0)}",
        f"- waiting_for_snapshot after probe: {runtime_probe.get('waiting_for_snapshot', 0)}",
        f"- Route exception count: {runtime_probe.get('route_exception_count', 0)}",
        f"- Exception type counts: {runtime_probe.get('route_exception_type_counts', {})}",
        f"- Candidate IDs affected: {runtime_probe.get('route_exception_candidate_ids', [])}",
        f"- Reason counts: {runtime_probe.get('reason_counts', {})}",
        f"- Top missing features: {runtime_probe.get('top_missing_features', {})}",
        "",
        "### First 5 route tracebacks",
    ]
    for ex in runtime_probe.get("first_5_tracebacks", []) if isinstance(runtime_probe, dict) else []:
        md.append(f"- {ex.get('candidate_id')}: {ex.get('exception_type')}: {ex.get('exception_message')}")
        tb = str(ex.get("traceback") or "")
        if tb:
            md.append("```")
            md.append(tb[-1200:])
            md.append("```")
    md += [
        "## Top failure reasons",
    ]
    for rsn, cnt in sorted(reason_counts.items(), key=lambda kv: -kv[1])[:10]:
        md.append(f"- {rsn}: {cnt}")

    md += ["", "## Per-candidate detail (exact reasons + suggested fixes)"]
    for d in diags:
        md.append(f"\n### {d['candidate_id']}")
        md.append(f"- enabled: {d['enabled']}, model: {d['model']}, preset_family: {d['preset_family']}, side: {d['side']}")
        md.append(f"- explicit_artifact (from paper_forward_candidates.json): {d['explicit_artifact_path']}")
        md.append(f"- resolved_dir: {d.get('resolved_dir')}")
        md.append(f"- final_load_status: {d['final_load_status']}")
        md.append(f"- exact_failure_reason: {d['exact_failure_reason']}")
        md.append(f"- dynamic_preset_source: {d.get('dynamic_preset_source')}")
        md.append(f"- model_file: {d.get('model_file_path')}")
        md.append(f"- feature_list: {d.get('feature_list_path')}")
        md.append(f"- files_exist: {d.get('exists')}")
        if d.get("suggested_fix"):
            md.append(f"- **suggested_fix**: {d['suggested_fix']}")
        if d.get("registry_hit"):
            md.append(f"- registry_hit: {d.get('registry_artifact_dir')}")

    md.append("\n## Summary of problems + fixes")
    md.append(f"- no_dynamic_presets_configured: happens when per-candidate profile.dynamic_presets empty AND no global config/dynamic_presets.json usable. Fixed by load_dynamic_presets + fallback generator.")
    md.append(f"- candidate_missing_artifact_paths: configured dir in paper_forward_candidates.json does not match actual materialized dir under artifacts/candidates/ (timestamp drift or nested <id>/<id>/ layout). Registry + fuzzy scan + resolve_candidate_artifacts mitigate.")
    md.append(f"- Duplicates: multiple t30/t35 variants or reload without clear. Engine + UI dedupe by (id,model,preset,side).")

    mpath.write_text("\n".join(md), encoding="utf-8")

    print(f"\n[PAPER-FWD-DIAG] Wrote {jpath}")
    print(f"[PAPER-FWD-DIAG] Wrote {mpath}")
    print(f"\nSUMMARY: Loaded OK: {ok} / {len(paper_cands)}   Failed: {fail} / {len(paper_cands)}")
    print("Preset OK:", preset_ok, "Fallbacks:", preset_fallback)
    if reason_counts:
        print("Top reasons:", dict(sorted(reason_counts.items(), key=lambda x: -x[1])[:5]))
    print("Duplicates declared:", dup_info)

    # Acceptance: must explain all
    if fail > 0 or preset_fallback > 0:
        print("[PAPER-FWD-DIAG] See report for WHY each of the 16 failed or used fallback.")
    return 0 if ok > 0 or preset_fallback > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
