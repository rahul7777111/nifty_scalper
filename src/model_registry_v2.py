"""Model registry with deployment metadata and rollback support."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistryV2:
    def __init__(self, registry_path: Path, model_dir: Path, live_model_path: Path):
        self.registry_path = Path(registry_path)
        self.model_dir = Path(model_dir)
        self.live_model_path = Path(live_model_path)
        self.registry = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            if self.registry_path.exists():
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {
            "_schema": "niftyscalper_model_registry_v2",
            "current": None,
            "history": [],
        }

    def _save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")

    def register_model(
        self,
        *,
        filename: str,
        checksum: str,
        dataset_size: int,
        training_horizon: Optional[str] = None,
        effective_horizon: Optional[str] = None,
        requested_horizon: Optional[str] = None,
        trained_at: Optional[str] = None,
        dataset_start_timestamp: Optional[str] = None,
        dataset_end_timestamp: Optional[str] = None,
        real_trading_days: int = 0,
        feature_count: int = 0,
        sample_count: Optional[int] = None,
        model_type: str = "ensemble",
        roc_auc: float = 0.0,
        accuracy: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1: float = 0.0,
        sharpe: float = 0.0,
        sortino: float = 0.0,
        profit_factor: float = 0.0,
        drawdown: float = 0.0,
        deployed: bool = False,
        deployment_decision: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        max_drawdown: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        entry = {
            "filename": filename,
            "checksum": checksum,
            "trained_at": trained_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset_start_timestamp": dataset_start_timestamp,
            "dataset_end_timestamp": dataset_end_timestamp,
            "real_trading_days": int(real_trading_days or 0),
            "requested_horizon": requested_horizon or training_horizon,
            "effective_horizon": effective_horizon or training_horizon,
            "feature_count": int(feature_count or 0),
            "sample_count": int(sample_count if sample_count is not None else dataset_size or 0),
            "model_type": str(model_type or "ensemble"),
            "metrics": {
                "roc_auc": float(roc_auc or 0.0),
                "accuracy": float(accuracy or 0.0),
                "precision": float(precision or 0.0),
                "recall": float(recall or 0.0),
                "f1": float(f1 or 0.0),
                "profit_factor": float(profit_factor or 0.0),
                "sharpe": float(sharpe or 0.0),
                "sortino": float(sortino or 0.0),
                "max_drawdown": float(max_drawdown if max_drawdown is not None else drawdown or 0.0),
            },
            "deployment_decision": deployment_decision or ("deployed" if deployed else "rejected"),
            "rejection_reason": rejection_reason,
            "deployed": bool(deployed),
        }
        if kwargs:
            entry["extras"] = dict(kwargs)

        previous = self.registry.get("current")
        if previous:
            self.registry.setdefault("history", []).append(previous)
            self.registry["history"] = self.registry["history"][-25:]
        self.registry["current"] = entry
        self._save()
        return entry

    def rollback(self) -> bool:
        history = self.registry.get("history", [])
        if not history:
            print("[REGISTRY] Rollback failed: no model in history.")
            return False

        rollback_entry = history[-1]
        source_path = self.model_dir / str(rollback_entry.get("filename") or "")
        if not source_path.exists():
            print(f"[REGISTRY] Rollback failed: {source_path.name} missing.")
            return False

        expected_checksum = str(rollback_entry.get("checksum") or "").strip()
        if expected_checksum:
            actual_checksum = _checksum(source_path)
            if actual_checksum != expected_checksum:
                print("[REGISTRY] Rollback failed: checksum mismatch on rollback candidate.")
                return False

        try:
            current_entry = self.registry.get("current")
            shutil.copy2(source_path, self.live_model_path)
            deployed_checksum = _checksum(self.live_model_path)
            if expected_checksum and deployed_checksum != expected_checksum:
                raise ValueError("live model checksum mismatch after rollback copy")

            self.registry["current"] = dict(rollback_entry)
            self.registry["current"]["deployed"] = True
            self.registry["current"]["deployment_decision"] = "rolled_back"
            self.registry["history"].pop()
            if current_entry:
                self.registry.setdefault("history", []).append(current_entry)
                self.registry["history"] = self.registry["history"][-25:]
            self._save()
            print(f"[REGISTRY] Rollback successful: {rollback_entry['filename']}")
            return True
        except Exception as exc:
            print(f"[REGISTRY] Rollback failed during operation: {exc}")
            return False
