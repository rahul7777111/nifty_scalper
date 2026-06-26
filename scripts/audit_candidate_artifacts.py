#!/usr/bin/env python3
"""Offline audit for candidate configs and artifact directories."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from candidate_artifact_resolver import resolve_candidate_artifact
except Exception:  # noqa: BLE001
    resolve_candidate_artifact = None  # type: ignore[assignment]

try:
    from candidate_manifest import load_candidate_manifest, validate_candidate_manifest
except Exception:  # noqa: BLE001
    load_candidate_manifest = None  # type: ignore[assignment]
    validate_candidate_manifest = None  # type: ignore[assignment]


CONFIG_FILES = [
    REPO_ROOT / "config" / "paper_forward_candidates.json",
    REPO_ROOT / "config" / "shadow_candidates.json",
    REPO_ROOT / "config" / "live_candidate_whitelist.json",
    REPO_ROOT / "config" / "paper_forward_artifact_registry.json",
]
ARTIFACT_ROOTS = [
    REPO_ROOT / "artifacts" / "candidates",
    REPO_ROOT / "models" / "candidates",
]
MANIFEST_NAMES = [
    "candidate_manifest.json",
    "candidate_profile.json",
    "paper_forward_manifest.json",
    "manifest.json",
]


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (REPO_ROOT / path)


def _iter_candidate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("candidates"), list):
            return [row for row in payload["candidates"] if isinstance(row, dict)]
        if isinstance(payload.get("members"), list):
            return [row for row in payload["members"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("id") or row.get("model_id") or "").strip()


def _artifact_dir_from_row(row: dict[str, Any]) -> Path | None:
    for key in ("artifact_dir", "artifact_path", "candidate_dir"):
        path = _resolve_path(row.get(key))
        if path is not None:
            if path.suffix:
                return path.parent
            return path
    cid = _candidate_id(row)
    if cid:
        return REPO_ROOT / "artifacts" / "candidates" / cid
    return None


def _model_path_from_row(row: dict[str, Any], artifact_dir: Path | None) -> Path | None:
    for key in ("model_path", "model", "model_pkl"):
        path = _resolve_path(row.get(key))
        if path is not None:
            return path
    if artifact_dir is not None:
        return artifact_dir / "model.pkl"
    return None


def _find_manifest(artifact_dir: Path) -> Path | None:
    for name in MANIFEST_NAMES:
        candidate = artifact_dir / name
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    total_rows = 0
    total_artifacts = 0

    print("CANDIDATE ARTIFACT AUDIT")
    print(f"repo={REPO_ROOT}")

    for path in CONFIG_FILES:
        if not path.exists():
            warnings.append(f"missing optional config: {path.relative_to(REPO_ROOT)}")
            continue

        payload, error = _load_json(path)
        if error:
            failures.append(f"{path.relative_to(REPO_ROOT)}: invalid JSON: {error}")
            continue

        rows = _iter_candidate_rows(payload)
        total_rows += len(rows)
        print(f"{path.relative_to(REPO_ROOT)} candidates={len(rows)}")

        for row in rows:
            cid = _candidate_id(row) or "<missing>"
            artifact_dir = _artifact_dir_from_row(row)
            model_path = _model_path_from_row(row, artifact_dir)

            if cid == "<missing>":
                failures.append(f"{path.relative_to(REPO_ROOT)}: row missing candidate_id/model_id")

            if artifact_dir is None:
                warnings.append(f"{path.relative_to(REPO_ROOT)}:{cid}: no artifact directory")
                continue

            total_artifacts += 1
            if not artifact_dir.exists():
                failures.append(f"{path.relative_to(REPO_ROOT)}:{cid}: artifact dir missing: {artifact_dir}")
                continue
            if not artifact_dir.is_dir():
                failures.append(f"{path.relative_to(REPO_ROOT)}:{cid}: artifact path is not a directory: {artifact_dir}")
                continue

            if model_path is not None and not model_path.exists():
                failures.append(f"{path.relative_to(REPO_ROOT)}:{cid}: model path missing: {model_path}")

            manifest_path = _find_manifest(artifact_dir)
            if manifest_path is None:
                warnings.append(f"{path.relative_to(REPO_ROOT)}:{cid}: no manifest/profile sidecar in {artifact_dir}")
            elif manifest_path.name == "candidate_manifest.json" and load_candidate_manifest is not None:
                try:
                    manifest = load_candidate_manifest(manifest_path)
                    if validate_candidate_manifest is not None:
                        manifest_errors = validate_candidate_manifest(manifest)
                        for item in manifest_errors:
                            failures.append(f"{path.relative_to(REPO_ROOT)}:{cid}: manifest validation: {item}")
                except Exception as exc:  # noqa: BLE001
                    raw_manifest, raw_error = _load_json(manifest_path)
                    if raw_error:
                        failures.append(
                            f"{path.relative_to(REPO_ROOT)}:{cid}: candidate_manifest invalid JSON: {raw_error}"
                        )
                    elif not isinstance(raw_manifest, dict):
                        failures.append(
                            f"{path.relative_to(REPO_ROOT)}:{cid}: candidate_manifest is not a JSON object"
                        )
                    else:
                        warnings.append(
                            f"{path.relative_to(REPO_ROOT)}:{cid}: candidate_manifest strict load warning: {type(exc).__name__}: {exc}"
                        )
            else:
                data, sidecar_error = _load_json(manifest_path)
                if sidecar_error:
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}:{cid}: sidecar {manifest_path.name} invalid JSON: {sidecar_error}"
                    )
                elif not isinstance(data, dict):
                    warnings.append(
                        f"{path.relative_to(REPO_ROOT)}:{cid}: sidecar {manifest_path.name} is not a JSON object"
                    )

            if resolve_candidate_artifact is not None and cid != "<missing>":
                try:
                    resolution = resolve_candidate_artifact(
                        row,
                        project_root=REPO_ROOT,
                        strict=True,
                        allow_fallback=False,
                    )
                    status = str(getattr(resolution, "artifact_identity_status", ""))
                    if status != "EXACT_MATCH":
                        failures.append(
                            f"{path.relative_to(REPO_ROOT)}:{cid}: artifact identity status={status}"
                        )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}:{cid}: artifact resolver crashed: {type(exc).__name__}: {exc}"
                    )

    for root in ARTIFACT_ROOTS:
        if not root.exists():
            continue
        for candidate_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            total_artifacts += 1
            model_file = candidate_dir / "model.pkl"
            manifest_file = _find_manifest(candidate_dir)
            if not model_file.exists():
                warnings.append(f"{candidate_dir.relative_to(REPO_ROOT)}: missing model.pkl")
            if manifest_file is None:
                warnings.append(f"{candidate_dir.relative_to(REPO_ROOT)}: missing manifest/profile sidecar")

    print(f"scanned_candidate_rows={total_rows}")
    print(f"scanned_artifact_dirs={total_artifacts}")

    if warnings:
        print("warnings:")
        for item in warnings:
            print(f"  - {item}")

    if failures:
        print("AUDIT FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
