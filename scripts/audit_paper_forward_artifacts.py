#!/usr/bin/env python3
"""
scripts/audit_paper_forward_artifacts.py

Audit paper forward candidates against on-disk artifacts.
Prints report, auto-fixes config (updates paths for timestamp variants, disables invalid).
Run after changes to ensure only valid artifacts are enabled.

Usage: python scripts/audit_paper_forward_artifacts.py [--fix-config]
"""
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "paper_forward_candidates.json"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "candidates"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-config", action="store_true", help="Update config with valid paths and disable invalids")
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        print("No config")
        return 1

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cands = cfg.get("candidates", [])

    print("candidate_id | enabled | configured_artifact_dir | resolved_artifact_dir | model_exists | manifest_exists | feature_order_count | status | reason")
    print("-" * 160)

    updated_cands = []
    valid_count = 0
    disabled_count = 0

    for c in cands:
        cid = c.get("candidate_id", "")
        enabled = bool(c.get("enabled", True))
        conf_dir = c.get("artifact_dir", "")
        conf_name = Path(conf_dir).name if conf_dir else ""

        # find best on disk
        resolved = None
        model_exists = False
        manifest_exists = False
        fo_count = 0
        status = "MISSING"
        reason = "no matching dir with model+features"

        # exact
        p = Path(conf_dir)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists() and p.is_dir():
            has_model = (p / "model.pkl").exists() or bool(list(p.glob("*.pkl")))
            has_man = any((p / f).exists() for f in ["candidate_manifest.json", "paper_forward_manifest.json", "candidate_profile.json"])
            fs_n = 0
            fs = p / "feature_schema.json"
            if fs.exists():
                try:
                    fd = json.loads(fs.read_text(encoding="utf-8"))
                    if isinstance(fd, list):
                        fs_n = len(fd)
                    elif isinstance(fd, dict):
                        fs_n = len(fd.get("features") or fd.get("live_computable_features") or [])
                except: pass
            if has_model and fs_n > 0:
                resolved = str(p.relative_to(REPO_ROOT))
                model_exists = has_model
                manifest_exists = has_man
                fo_count = fs_n
                status = "OK"
                reason = "exact match"
                valid_count += 1
            else:
                resolved = str(p.relative_to(REPO_ROOT)) if p.exists() else ""
                reason = "exists but no model or features"

        if status != "OK":
            # search for variant with model+fs>0 matching prefix
            prefix = cid.split("_t")[0]
            for d in sorted(ARTIFACTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
                if d.is_dir() and prefix in d.name:
                    has_model = (d / "model.pkl").exists() or bool(list(d.glob("*.pkl")))
                    has_man = any((d / f).exists() for f in ["candidate_manifest.json", "paper_forward_manifest.json", "candidate_profile.json"])
                    fs_n = 0
                    fs = d / "feature_schema.json"
                    if fs.exists():
                        try:
                            fd = json.loads(fs.read_text(encoding="utf-8"))
                            if isinstance(fd, list): fs_n = len(fd)
                            elif isinstance(fd, dict): fs_n = len(fd.get("features") or fd.get("live_computable_features") or [])
                        except: pass
                    if has_model and fs_n > 0:
                        resolved = str(d.relative_to(REPO_ROOT))
                        model_exists = has_model
                        manifest_exists = has_man
                        fo_count = fs_n
                        status = "UPDATED_PATH"
                        reason = "timestamp variant with valid artifact"
                        valid_count += 1
                        if args.fix_config:
                            c["artifact_dir"] = resolved
                        break

        if status not in ("OK", "UPDATED_PATH"):
            # disable
            disabled_count += 1
            if args.fix_config:
                c["enabled"] = False
                c["disabled_reason"] = "ARTIFACT_MISSING_OR_FEATURE_ORDER_MISSING"
                reason = c["disabled_reason"]

        print(f"{cid[:50]:50} | {enabled} | {conf_dir[-40:]:40} | {resolved or '':40} | {model_exists} | {manifest_exists} | {fo_count} | {status} | {reason}")

        updated_cands.append(c)

    print(f"\nSummary: valid={valid_count} disabled={disabled_count} total={len(cands)}")

    if args.fix_config:
        cfg["candidates"] = updated_cands
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("Config updated (paths fixed, invalids disabled).")

    return 0

if __name__ == "__main__":
    sys.exit(main())
