from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from ml_model_registry import ALLOWED_PAPER_VERDICTS, SHADOW_ONLY_VERDICTS, MLModelRegistry, feature_vector_hash
from ml_paper_risk_manager import MLPaperRiskManager


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def paper_mode_allowed(*, manifest_payload: Dict[str, Any], registry_health: Dict[str, Any], schema_ok: bool, paper_mode_enabled: bool, kill_switch: bool) -> tuple[bool, str]:
    if kill_switch:
        return False, "ml_disable_all"
    if not paper_mode_enabled:
        return False, "paper_mode_disabled"
    verdict = str(manifest_payload.get("paper_readiness_verdict") or "")
    if verdict in SHADOW_ONLY_VERDICTS:
        return False, "verdict_shadow_only"
    if verdict not in ALLOWED_PAPER_VERDICTS:
        return False, "manifest_verdict_invalid"
    if bool(manifest_payload.get("production_adoption_allowed")):
        return False, "production_adoption_not_allowed"
    if not schema_ok:
        return False, "schema_mismatch"
    if not registry_health.get("ok"):
        return False, "registry_unhealthy"
    return True, ""


class MLRuntimeEngine:
    def __init__(
        self,
        *,
        registry: MLModelRegistry,
        risk_manager: MLPaperRiskManager,
        log_dir: str | Path = "logs",
        kill_switch: bool = False,
        shadow_mode_enabled: bool = False,
        paper_mode_enabled: bool = False,
        min_confidence_threshold: float | None = None,
        max_predictions_per_day: int = 1000,
        log_feature_vector: bool = True,
        log_prediction_reason: bool = True,
        fail_closed_on_schema_mismatch: bool = True,
        real_order_callable: Callable[..., Any] | None = None,
    ) -> None:
        self.registry = registry
        self.risk_manager = risk_manager
        self.log_dir = Path(log_dir)
        self.kill_switch = bool(kill_switch)
        self.shadow_mode_enabled = bool(shadow_mode_enabled)
        self.paper_mode_enabled = bool(paper_mode_enabled)
        self.min_confidence_threshold = min_confidence_threshold
        self.max_predictions_per_day = int(max_predictions_per_day)
        self.log_feature_vector = bool(log_feature_vector)
        self.log_prediction_reason = bool(log_prediction_reason)
        self.fail_closed_on_schema_mismatch = bool(fail_closed_on_schema_mismatch)
        self.real_order_callable = real_order_callable
        self._prediction_count_today = 0
        self._last_prediction: Dict[str, Any] | None = None


    def _load_dynamic_preset(self) -> dict | None:
        """Load dynamic_preset.json from the registered model's candidate directory.

        Returns None if no preset found or file missing (safe — no preset applied).
        Preset contains only live-computable config: threshold adjustments,
        spread/liquidity/DTE/premium filters, regime conditions, time windows.
        NO future, PnL, return, label, or outcome data.
        """
        try:
            manifest = self.registry.manifest()
            if manifest is None:
                return None
            payload = manifest.payload if isinstance(manifest.payload, dict) else {}
            candidate_dir = (
                payload.get("source_manifest_dir")
                or payload.get("candidate_dir")
                or getattr(manifest.payload, "candidate_dir", None)
                or ""
            )
            if not candidate_dir:
                return None
            preset_path = Path(str(candidate_dir)) / "dynamic_preset.json"
            if preset_path.exists():
                with preset_path.open() as f2:
                    return json.load(f2)
            return None
        except Exception:
            return None

    def _date_suffix(self, ts: datetime) -> str:
        return ts.strftime("%Y%m%d")



    def evaluate_snapshot(self, snapshot: Mapping[str, Any], *, current_position_state: str = "FLAT", market_regime: str = "", symbol: str = "NIFTY") -> Dict[str, Any]:
        ts = snapshot.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts is None:
            ts = datetime.now()
        if self.kill_switch or not self.shadow_mode_enabled:
            status = {"shadow_decision": "ERROR", "reason": "ml_disabled", "production_order_sent": False}
            self._last_prediction = status
            return status
        if self._prediction_count_today >= self.max_predictions_per_day:
            decision = "BLOCKED_MAX_DAILY_TRADES"
            reason = "max_predictions_per_day"
            return self._log_shadow(snapshot, ts, symbol, current_position_state, market_regime, 0.0, decision, reason, [], [], {})
        health = self.registry.model_health_status()
        manifest = self.registry.manifest()
        if not health.get("ok") or manifest is None:
            return self._log_shadow(snapshot, ts, symbol, current_position_state, market_regime, 0.0, "ERROR", "registry_unhealthy", [], [], {})
        valid, missing, invalid = self.registry.validate_runtime_features(snapshot)
        if not valid:
            decision = "BLOCKED_SCHEMA" if self.fail_closed_on_schema_mismatch else "ERROR"
            return self._log_shadow(snapshot, ts, symbol, current_position_state, market_regime, 0.0, decision, "runtime_feature_validation_failed", missing, invalid, {})

        # --- Dynamic preset filter (live-computable only) ---
        # Load preset from candidate directory; skip if not present
        preset_config = self._load_dynamic_preset()
        if preset_config is not None:
            from candidate_filters import apply_dynamic_preset_filter
            preset_result = apply_dynamic_preset_filter(dict(snapshot), preset_config)
            if not preset_result.get("filter_passed", False):
                rejection = preset_result.get("rejection_reason", "preset_blocked")
                return self._log_shadow(
                    snapshot, ts, symbol, current_position_state, market_regime,
                    0.0, "BLOCKED_DYNAMIC_PRESET", rejection, [], [], {}
                )

        probability = self.registry.predict_proba(snapshot)
        threshold = float(manifest.payload.get("selected_threshold") or 0.5)
        min_threshold = threshold if self.min_confidence_threshold is None else max(threshold, float(self.min_confidence_threshold))
        if probability < min_threshold:
            return self._log_shadow(snapshot, ts, symbol, current_position_state, market_regime, probability, "BLOCKED_LOW_CONFIDENCE", "below_threshold", [], [], {})
        risk = self.risk_manager.evaluate(dict(snapshot))
        if not risk.get("allowed"):
            return self._log_shadow(snapshot, ts, symbol, current_position_state, market_regime, probability, "BLOCKED_RISK_FILTER", str(risk.get("blocking_reason") or "risk_block"), [], [], risk)
        decision = "WOULD_ENTER"
        row = self._log_shadow(snapshot, ts, symbol, current_position_state, market_regime, probability, decision, "threshold_and_risk_passed", [], [], risk)
        paper_allowed, paper_reason = paper_mode_allowed(
            manifest_payload=manifest.payload,
            registry_health=health,
            schema_ok=True,
            paper_mode_enabled=self.paper_mode_enabled,
            kill_switch=self.kill_switch,
        )
        if paper_allowed:
            row["paper_trade"] = self._create_paper_trade(snapshot, ts, probability, threshold, symbol)
        else:
            blocked_path = self.log_dir / f"ml_paper_mode_blocked_{self._date_suffix(ts)}.jsonl"
            _append_jsonl(blocked_path, {"timestamp": ts.isoformat(), "reason": paper_reason, "model_id": manifest.model_id, "production_order_sent": False})
        return row

    def _create_paper_trade(self, snapshot: Mapping[str, Any], ts: datetime, probability: float, threshold: float, symbol: str) -> Dict[str, Any]:
        if self.real_order_callable is not None:
            raise RuntimeError("ML paper/shadow mode must never call real broker place_order")
        ltp = float(snapshot.get("option_ltp") or snapshot.get("ltp") or 0.0)
        fill = ltp * 1.001 if ltp > 0 else 0.0
        trade = {
            "timestamp": ts.isoformat(),
            "paper_trade_id": f"paper-{int(ts.timestamp())}",
            "model_id": str((self.registry.manifest().payload if self.registry.manifest() else {}).get("model_id") or ""),
            "symbol": symbol,
            "option_symbol": snapshot.get("option_symbol") or snapshot.get("option_symbol") or "",
            "option_type": snapshot.get("option_type") or "",
            "strike": snapshot.get("strike"),
            "expiry": snapshot.get("expiry"),
            "side": "BUY",
            "entry_price": ltp,
            "simulated_fill_price": fill,
            "quantity": int(snapshot.get("quantity") or 1),
            "predicted_probability": probability,
            "threshold": threshold,
            "stop_loss": snapshot.get("stop_loss"),
            "target": snapshot.get("target"),
            "exit_time": None,
            "exit_price": None,
            "pnl": None,
            "return_pct": None,
            "status": "OPEN",
            "exit_reason": "",
            "costs_applied": {"slippage_bps": 10},
            "production_order_sent": False,
        }
        self.risk_manager.record_entry(ts)
        log_path = self.log_dir / f"ml_paper_trades_{self._date_suffix(ts)}.jsonl"
        _append_jsonl(log_path, trade)
        return trade

    def _log_shadow(self, snapshot: Mapping[str, Any], ts: datetime, symbol: str, current_position_state: str, market_regime: str, probability: float, decision: str, reason: str, missing: list[str], invalid: list[str], risk: Mapping[str, Any]) -> Dict[str, Any]:
        manifest = self.registry.manifest()
        feature_names = self.registry.get_model_features()
        row = {
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "option_symbol": snapshot.get("option_symbol") or "",
            "option_type": snapshot.get("option_type") or "",
            "strike": snapshot.get("strike"),
            "expiry": snapshot.get("expiry"),
            "spot": snapshot.get("spot"),
            "option_ltp": snapshot.get("option_ltp") or snapshot.get("ltp"),
            "model_id": manifest.model_id if manifest else "",
            "model_family": (manifest.payload.get("model_family") if manifest else ""),
            "feature_set_name": (manifest.payload.get("feature_set_name") if manifest else ""),
            "selected_threshold": (manifest.payload.get("selected_threshold") if manifest else None),
            "predicted_probability": probability,
            "shadow_decision": decision,
            "reason": reason if self.log_prediction_reason else "",
            "missing_features": missing,
            "invalid_features": invalid,
            "feature_vector_hash": feature_vector_hash(feature_names, snapshot) if feature_names else "",
            "feature_values": {name: snapshot.get(name) for name in feature_names} if self.log_feature_vector else None,
            "current_position_state": current_position_state,
            "market_regime": market_regime,
            "risk_filter_status": dict(risk or {}),
            "paper_mode_enabled": bool(self.paper_mode_enabled),
            "production_order_sent": False,
        }
        self._prediction_count_today += 1
        self._last_prediction = row
        path = self.log_dir / f"ml_shadow_predictions_{self._date_suffix(ts)}.jsonl"
        _append_jsonl(path, row)
        return row

    def status(self) -> Dict[str, Any]:
        return {
            "shadow_mode_enabled": self.shadow_mode_enabled and not self.kill_switch,
            "paper_mode_enabled": self.paper_mode_enabled and not self.kill_switch,
            "kill_switch": self.kill_switch,
            "prediction_count_today": self._prediction_count_today,
            "last_prediction": self._last_prediction,
            "model_health": self.registry.model_health_status(),
        }
