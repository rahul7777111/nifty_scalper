from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    "artifacts/candidates",
    "models/candidates",
    "reports/candidates",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def safe_token(value: Any, default: str = "unknown", max_len: int = 42) -> str:
    text = str(value if value not in (None, "") else default)
    text = text.replace("cost_survivor_label_v2", "cost_survivor_v2")
    text = text.replace("profitable_trade_label", "profit_label")
    text = text.replace("logistic_regression", "logreg")
    text = text.replace("calibrated_logistic_regression", "cal_logreg")
    text = text.replace("random_forest", "rf")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    if not text:
        text = default
    return text[:max_len].strip("_") or default


def threshold_token(value: Any) -> str:
    try:
        return f"t{int(round(float(value) * 100)):02d}"
    except Exception:
        return "tNA"


def infer_side(meta: dict[str, Any], candidate_id: str) -> str:
    filter_meta = meta.get("filter") if isinstance(meta.get("filter"), dict) else {}
    side_policy = str(first_value(meta.get("side_policy"), filter_meta.get("side_policy"), "")).upper()
    filter_name = str(first_value(meta.get("filter_name"), filter_meta.get("filter_name"), "")).upper()
    cid = str(candidate_id).upper()

    if side_policy in {"PE_ONLY", "PE"}:
        return "PE_ONLY"
    if side_policy in {"CE_ONLY", "CE"}:
        return "CE_ONLY"
    if side_policy in {"BOTH", "BOTH_SYMMETRIC"}:
        return "BOTH"
    if side_policy in {"AUTO_DIRECTIONAL", "CE_PE_ROUTER"}:
        return "CE_PE_ROUTER"

    if re.search(r"(^|[_-])PE($|[_-])", filter_name) or filter_name.startswith("PE_"):
        return "PE_ONLY"
    if re.search(r"(^|[_-])CE($|[_-])", filter_name) or filter_name.startswith("CE_"):
        return "CE_ONLY"
    if "CE_AND_PE" in filter_name or "CE_PE" in filter_name:
        return "CE_PE_ROUTER"

    if re.search(r"(^|[_-])PE($|[_-])", cid) or "_PE_ONLY_" in cid:
        return "PE_ONLY"
    if re.search(r"(^|[_-])CE($|[_-])", cid) or "_CE_ONLY_" in cid:
        return "CE_ONLY"
    if "ROUTER" in cid or "CE_PE" in cid:
        return "CE_PE_ROUTER"
    if "BOTH" in cid:
        return "BOTH"
    return "ANY_SIDE"


def merge_metadata(candidate_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in ("candidate_manifest.json", "shadow_manifest.json", "candidate_profile.json", "metrics.json"):
        data = load_json(candidate_dir / name)
        if data:
            merged[name.removesuffix(".json")] = data
            for key, value in data.items():
                merged.setdefault(key, value)
    nested = candidate_dir / candidate_dir.name
    if nested.exists():
        for name in ("candidate_manifest.json", "candidate_profile.json", "metrics.json"):
            data = load_json(nested / name)
            if data:
                merged[f"nested_{name.removesuffix('.json')}"] = data
                for key, value in data.items():
                    merged.setdefault(key, value)
    return merged


def candidate_dirs(root: Path) -> list[Path]:
    dirs: set[Path] = set()
    for pkl in root.rglob("*.pkl"):
        if pkl.name.lower().endswith(".pkl"):
            dirs.add(pkl.parent)
    return sorted(dirs)


def build_record(candidate_dir: Path, model_pkl: Path) -> dict[str, Any]:
    meta = merge_metadata(candidate_dir)
    model_meta = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    filter_meta = meta.get("filter") if isinstance(meta.get("filter"), dict) else {}
    perf = meta.get("performance") if isinstance(meta.get("performance"), dict) else {}
    overall = meta.get("overall_metrics") if isinstance(meta.get("overall_metrics"), dict) else {}

    candidate_id = str(first_value(meta.get("candidate_id"), meta.get("model_id"), candidate_dir.name))
    model_name = first_value(meta.get("model_name"), model_meta.get("model_name"), model_meta.get("model_type"), "model")
    target = first_value(model_meta.get("target"), meta.get("target"), "target_unknown")
    filter_name = first_value(meta.get("filter_name"), filter_meta.get("filter_name"), meta.get("preset_family"), "filter_unknown")
    threshold = first_value(meta.get("selected_threshold"), model_meta.get("threshold"), meta.get("threshold"))
    status = first_value(meta.get("status"), meta.get("verdict"), perf.get("readiness"), "status_unknown")
    candidate_type = first_value(meta.get("candidate_type"), filter_meta.get("candidate_type"), "MODEL")
    timestamp = first_value(meta.get("persistence_timestamp"), meta.get("generated_at"))
    if not timestamp:
        match = re.search(r"(20\d{6}_\d{6})", candidate_id)
        timestamp = match.group(1) if match else datetime.fromtimestamp(model_pkl.stat().st_mtime).strftime("%Y%m%d_%H%M%S")

    side = infer_side(meta, candidate_id)
    pf = first_value(overall.get("mean_pf"), perf.get("mean_pf"), meta.get("mean_pf"))
    pf_token = "pfNA"
    try:
        pf_token = f"pf{float(pf):.2f}".replace(".", "p")
    except Exception:
        pass

    short_hash = hashlib.sha1(str(model_pkl.resolve()).encode("utf-8")).hexdigest()[:8]
    alias_stem = "__".join(
        safe_token(x)
        for x in [
            side,
            candidate_type,
            model_name,
            target,
            filter_name,
            threshold_token(threshold),
            safe_token(status, max_len=26),
            pf_token,
            timestamp,
            short_hash,
        ]
    )
    return {
        "alias_file": f"{alias_stem}.pkl",
        "display_name": alias_stem,
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "side": side,
        "model_name": model_name,
        "target": target,
        "filter_name": filter_name,
        "threshold": threshold,
        "status": status,
        "mean_pf": pf,
        "artifact_dir": str(candidate_dir.relative_to(REPO_ROOT)) if candidate_dir.is_relative_to(REPO_ROOT) else str(candidate_dir),
        "model_path": str(model_pkl.relative_to(REPO_ROOT)) if model_pkl.is_relative_to(REPO_ROOT) else str(model_pkl),
        "manifest_path": str((candidate_dir / "candidate_manifest.json").relative_to(REPO_ROOT))
        if (candidate_dir / "candidate_manifest.json").exists() and (candidate_dir / "candidate_manifest.json").is_relative_to(REPO_ROOT)
        else "",
        "feature_schema_path": str((candidate_dir / "feature_schema.json").relative_to(REPO_ROOT))
        if (candidate_dir / "feature_schema.json").exists() and (candidate_dir / "feature_schema.json").is_relative_to(REPO_ROOT)
        else "",
        "timestamp": str(timestamp),
    }


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create readable .pkl aliases and config for historical candidate backtests.")
    parser.add_argument("--output-dir", default="artifacts/backtest_candidate_pkls")
    parser.add_argument("--config-out", default="config/backtest_candidates_named.json")
    parser.add_argument("--index-csv", default="reports/backtest_candidate_named_index.csv")
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for root_text in args.roots:
        root = (REPO_ROOT / root_text).resolve()
        if not root.exists():
            continue
        for cdir in candidate_dirs(root):
            pkls = sorted(cdir.glob("*.pkl"), key=lambda p: (p.name != "model.pkl", p.name))
            if not pkls:
                continue
            record = build_record(cdir, pkls[0])
            alias_path = output_dir / record["alias_file"]
            record["alias_path"] = str(alias_path.relative_to(REPO_ROOT))
            record["alias_materialization"] = link_or_copy(pkls[0], alias_path)
            records.append(record)

    records.sort(key=lambda r: (str(r["side"]), str(r["model_name"]), str(r["filter_name"]), str(r["timestamp"]), str(r["candidate_id"])))

    config = {
        "mode": "historical_backtest_named_candidates",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Generated aliases are for human-readable selection. Original artifact_dir/model_path values remain canonical.",
        "candidates": [
            {
                "candidate_id": r["candidate_id"],
                "display_name": r["display_name"],
                "enabled": True,
                "artifact_dir": r["artifact_dir"],
                "model_path": r["model_path"],
                "named_model_path": r["alias_path"],
                "model_name": r["model_name"],
                "side_policy": r["side"],
                "filter_name": r["filter_name"],
                "target": r["target"],
                "selected_threshold": r["threshold"],
                "status": r["status"],
                "mean_pf": r["mean_pf"],
            }
            for r in records
        ],
    }

    config_path = (REPO_ROOT / args.config_out).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    index_path = (REPO_ROOT / args.index_csv).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()) if records else ["candidate_id"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Candidates indexed: {len(records)}")
    print(f"Named pickle folder: {output_dir}")
    print(f"Backtest config: {config_path}")
    print(f"CSV index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
