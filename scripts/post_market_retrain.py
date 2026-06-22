from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
CURRENT_MODEL_PATH = REPO_ROOT / "ml_signal_model.pkl"
sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".scalper.env")
except Exception:
    pass

from indicators import atr
from cost_model import CostModel
from label_policies import build_label_dataset
from market_data import Candle
from ml_pipeline import (
    LABEL_POLICY,
    TARGET_FEATURES,
    build_supervised_dataset_v2,
    verify_no_lookahead_leakage,
    walk_forward_validate_production,
)
from ml_signals import load_model, train_ensemble, WeightedEnsembleClassifier
from model_registry_v2 import ModelRegistryV2
from feature_audit import run_feature_audit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("post_market_retrain")


MIN_USABLE_SAMPLES = 1500
LOOKBACK_BARS = 30
TB_MAX_HORIZON_BARS = int(os.getenv("TB_MAX_HORIZON_BARS", "45"))
TB_PROFIT_TARGET_ATR = float(os.getenv("TB_PROFIT_TARGET_ATR", "1.5"))
TB_STOP_LOSS_ATR = float(os.getenv("TB_STOP_LOSS_ATR", "2.0"))
WF_N_FOLDS = 8
WF_GAP = 5
WF_MIN_TRAIN = 250
WF_MIN_TEST = 50
MAX_ACCEPTABLE_DRAWDOWN = float(os.getenv("RETRAIN_MAX_DRAWDOWN", "0.05"))

GATES = {
    "roc_auc": 0.55,
    "f1": 0.50,
    "profit_factor": 1.15,
    "sharpe": 1.0,
}

TELEGRAM_TOKEN = os.getenv("MSTOCK_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("MSTOCK_TELEGRAM_CHAT_ID", "")


def notify(msg: str, *, is_error: bool = False) -> None:
    level = "ERROR" if is_error else "OK"
    full_msg = f"[NiftyScalper ML] {level}\n{msg}"
    log.info("Notification: %s", msg)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            import urllib.parse
            import urllib.request

            payload = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": full_msg}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data=payload,
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            log.warning("Telegram notification failed: %s", exc)


def checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candles_from_data_dir() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("candles_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("candles") or [])
        except Exception as exc:
            log.warning("Failed to load %s: %s", path.name, exc)
    return rows


def fetch_todays_candles() -> List[Dict[str, Any]]:
    try:
        from collect_training_data import build_client
    except Exception:
        return []

    try:
        client = build_client()
        today = date.today()
        candles_raw = client.get_intraday_candles(
            symbol_token=os.getenv("COLLECT_SYMBOL_TOKEN", "26000"),
            exchange=os.getenv("COLLECT_EXCHANGE", "NSE"),
        )
        if not candles_raw:
            return []
        rows = []
        for candle in candles_raw:
            rows.append(
                {
                    "time": str(getattr(candle, "time", "")),
                    "open": float(getattr(candle, "open", 0.0) or 0.0),
                    "high": float(getattr(candle, "high", 0.0) or 0.0),
                    "low": float(getattr(candle, "low", 0.0) or 0.0),
                    "close": float(getattr(candle, "close", 0.0) or 0.0),
                    "volume": float(getattr(candle, "volume", 0.0) or 0.0),
                }
            )
        out_path = DATA_DIR / f"candles_{today.strftime('%Y%m%d')}.json"
        out_path.write_text(json.dumps({"date": str(today), "candles": rows}, indent=2), encoding="utf-8")
        return rows
    except Exception as exc:
        log.warning("fetch_todays_candles failed: %s", exc)
        return []


def dict_to_candle(raw: Dict[str, Any]) -> Optional[Candle]:
    raw_time = str(raw.get("time", "")).split(".")[0]
    dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw_time, fmt)
            break
        except Exception:
            continue
    if dt is None:
        return None
    try:
        return Candle(
            time=dt,
            open=float(raw.get("open", 0.0) or 0.0),
            high=float(raw.get("high", 0.0) or 0.0),
            low=float(raw.get("low", 0.0) or 0.0),
            close=float(raw.get("close", 0.0) or 0.0),
            volume=float(raw.get("volume", 0.0) or 0.0),
        )
    except Exception:
        return None


def run_static_leakage_audit() -> Dict[str, Any]:
    leakage_patterns = [
        "shift(-1)",
        "future_return",
        "future_close",
        "fit_transform(",
        "center=True",
        "center = True",
        "replicate_candles(",
    ]
    files = [
        REPO_ROOT / "src" / "ml_pipeline.py",
        REPO_ROOT / "src" / "ml_signals.py",
        REPO_ROOT / "scripts" / "post_market_retrain.py",
        REPO_ROOT / "scripts" / "train_window_benchmark.py",
    ]
    findings: List[str] = []
    for path in files:
        if not path.exists():
            continue
        try:
            in_pattern_block = False
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if "leakage_patterns = [" in line:
                    in_pattern_block = True
                    continue
                if in_pattern_block:
                    if "]" in line:
                        in_pattern_block = False
                    continue
                for pattern in leakage_patterns:
                    if pattern in line:
                        findings.append(f"{path.name}:{line_no}: {pattern}")
        except Exception as exc:
            findings.append(f"{path.name}: read_error: {exc}")
    return {
        "passed": len(findings) == 0,
        "findings": findings,
        "reason": "passed" if not findings else "forbidden_patterns_found",
    }


def compress_to_target_features(X: List[List[float]], feature_names: List[str]) -> tuple[List[List[float]], List[str]]:
    indices = [idx for idx, name in enumerate(feature_names) if name in TARGET_FEATURES]
    if not indices:
        return X, feature_names
    return [[row[idx] for idx in indices] for row in X], [feature_names[idx] for idx in indices]


def compute_labeling_params(candles: List[Candle]) -> Dict[str, float]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    latest_close = closes[-1] if closes else 1.0
    try:
        latest_atr = atr(highs, lows, closes, period=14) or 10.0
    except Exception:
        latest_atr = 10.0
    return {
        "tb_profit_target_pct": (TB_PROFIT_TARGET_ATR * latest_atr) / latest_close if latest_close else 0.01,
        "tb_stop_loss_pct": (TB_STOP_LOSS_ATR * latest_atr) / latest_close if latest_close else 0.005,
    }


def write_report(report: Dict[str, Any], report_timestamp: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"retraining_report_{report_timestamp}.json"
    md_path = REPORTS_DIR / f"retraining_report_{report_timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Retraining Report",
        "",
        f"- Timestamp: {report['run']['started_at']}",
        f"- Requested horizon: {report['dataset']['requested_horizon']}",
        f"- Effective horizon: {report['dataset']['effective_horizon']}",
        f"- Real trading days: {report['dataset']['real_trading_days']}",
        f"- Samples: {report['dataset']['sample_count']}",
        f"- Feature count: {report['dataset']['feature_count']}",
        f"- Static leakage audit: {report['audits']['static']['passed']}",
        f"- Dynamic leakage audit: {report['audits']['dynamic']['passed']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("roc_auc", "accuracy", "precision", "recall", "f1", "profit_factor", "sharpe", "sortino", "drawdown", "trades_count"):
        md_lines.append(f"| {key} | {report['validation']['metrics'].get(key)} |")
    md_lines.extend(
        [
            "",
            "## Deployment",
            "",
            f"- Decision: {report['deployment']['decision']}",
            f"- Reason: {report['deployment']['reason']}",
            f"- Model path: {report['deployment'].get('model_path')}",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def archive_old_models(keep_n: int = 15) -> None:
    pkls = sorted(MODEL_DIR.glob("ml_signal_model_*.pkl"))
    for old_path in pkls[:-keep_n]:
        try:
            old_path.unlink()
        except Exception:
            continue


def generate_model_comparison_report(registry: ModelRegistryV2) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def metrics_for_entry(entry: Dict[str, Any] | None) -> Dict[str, Any]:
        if not entry:
            return {}
        metrics = dict(entry.get("metrics") or {})
        metrics["model_name"] = entry.get("filename")
        metrics["deployment_decision"] = entry.get("deployment_decision")
        return metrics

    production_bundle = load_model(str(CURRENT_MODEL_PATH))
    production_metrics = dict(getattr(production_bundle, "metrics", {}) or {})
    current_entry = registry.registry.get("current")
    previous_entry = registry.registry.get("history", [])[-1] if registry.registry.get("history") else None

    rows = [
        ("current_production_model", production_metrics, getattr(production_bundle, "trained_at", "unknown"), CURRENT_MODEL_PATH.name),
        ("latest_retrained_model", metrics_for_entry(current_entry), current_entry.get("trained_at") if current_entry else "unknown", current_entry.get("filename") if current_entry else "n/a"),
        ("previous_retrained_model", metrics_for_entry(previous_entry), previous_entry.get("trained_at") if previous_entry else "unknown", previous_entry.get("filename") if previous_entry else "n/a"),
    ]

    lines = [
        "# Model Comparison",
        "",
        "| Model | Artifact | Trained At | ROC-AUC | F1 | Profit Factor | Sharpe | Drawdown | Decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, metrics, trained_at, artifact in rows:
        lines.append(
            f"| {label} | {artifact} | {trained_at} | "
            f"{float(metrics.get('roc_auc', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('f1', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('profit_factor', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('sharpe', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('max_drawdown', metrics.get('drawdown', 0.0)) or 0.0):.4f} | "
            f"{metrics.get('deployment_decision', 'n/a')} |"
        )
    out_path = REPORTS_DIR / "model_comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run(
    *,
    years: Optional[int] = None,
    no_deploy: bool = False,
    force_no_deploy: bool = False,
    report_only: bool = False,
    label_policy: str = "current_triple_barrier",
) -> bool:
    started_at = datetime.now(timezone.utc)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fetch_todays_candles()
    raw_rows = load_candles_from_data_dir()
    candles = [c for c in (dict_to_candle(row) for row in raw_rows) if c is not None]
    candles.sort(key=lambda candle: candle.time)

    if len(candles) < 1000:
        raise RuntimeError(f"Insufficient raw candles for retraining: {len(candles)}")

    active_days = sorted({candle.time.date() for candle in candles})
    real_trading_days = len(active_days)
    available_years = real_trading_days / 252.0 if real_trading_days else 0.0
    requested_horizon = f"{years}y" if years else "all_available"
    effective_days = real_trading_days

    if years:
        requested_days = years * 252
        if real_trading_days < requested_days:
            log.warning(
                "Requested %s years, but only %s trading days / %.2f years of real data are available. Using available real data only.",
                years,
                real_trading_days,
                available_years,
            )
        else:
            selected_days = set(active_days[-requested_days:])
            candles = [candle for candle in candles if candle.time.date() in selected_days]
            effective_days = len(selected_days)

    # Maintain compatibility with the current retraining pipeline.
    candles = candles[::10]
    if len(candles) <= LOOKBACK_BARS + TB_MAX_HORIZON_BARS:
        raise RuntimeError(f"Insufficient resampled candles for retraining: {len(candles)}")

    static_audit = run_static_leakage_audit()
    if not static_audit["passed"] and not report_only:
        raise RuntimeError("Static leakage audit failed")

    dynamic_audit = verify_no_lookahead_leakage(
        candles,
        build_supervised_dataset_v2,
        lookback=LOOKBACK_BARS,
        horizon=TB_MAX_HORIZON_BARS,
    )
    if not dynamic_audit["passed"] and not report_only:
        raise RuntimeError(f"Dynamic leakage audit failed: {dynamic_audit['reason']}")

    label_dataset = build_label_dataset(
        candles,
        policy_name=label_policy,
        lookback=LOOKBACK_BARS,
        horizon=TB_MAX_HORIZON_BARS,
        cost_model=CostModel(),
    )
    X, y, feature_names = label_dataset.X, label_dataset.y, label_dataset.feature_names
    X, feature_names = compress_to_target_features(X, feature_names)

    sample_count = len(X)
    positive_rate = (sum(y) / len(y)) if y else 0.0
    min_sample_pass = sample_count >= MIN_USABLE_SAMPLES

    wf_result = walk_forward_validate_production(
        X,
        y,
        n_splits=WF_N_FOLDS,
        gap=WF_GAP,
        min_train_samples=WF_MIN_TRAIN,
        min_test_samples=WF_MIN_TEST,
        gate_roc_auc=GATES["roc_auc"],
        gate_accuracy=0.50,
        gate_f1=GATES["f1"],
        model_factory=lambda: WeightedEnsembleClassifier(),
        candles=candles,
        lookback=LOOKBACK_BARS,
    )
    metrics = dict(wf_result.get("metrics") or {})

    gate_failures: List[str] = []
    if not static_audit["passed"]:
        gate_failures.append("static_leakage_audit_failed")
    if not dynamic_audit["passed"]:
        gate_failures.append(dynamic_audit["reason"])
    if not min_sample_pass:
        gate_failures.append(f"samples_below_minimum ({sample_count} < {MIN_USABLE_SAMPLES})")
    if not (0.35 <= positive_rate <= 0.65):
        gate_failures.append(f"label_imbalance ({positive_rate:.4f})")
    if not wf_result.get("passed", False):
        gate_failures.append(str(wf_result.get("reason") or "walk_forward_failed"))
    if float(metrics.get("roc_auc", 0.0) or 0.0) <= GATES["roc_auc"]:
        gate_failures.append(f"roc_auc <= {GATES['roc_auc']}")
    if float(metrics.get("f1", 0.0) or 0.0) <= GATES["f1"]:
        gate_failures.append(f"f1 <= {GATES['f1']}")
    if float(metrics.get("profit_factor", 0.0) or 0.0) <= GATES["profit_factor"]:
        gate_failures.append(f"profit_factor <= {GATES['profit_factor']}")
    if float(metrics.get("sharpe", 0.0) or 0.0) <= GATES["sharpe"]:
        gate_failures.append(f"sharpe <= {GATES['sharpe']}")
    if float(metrics.get("drawdown", 0.0) or 0.0) > MAX_ACCEPTABLE_DRAWDOWN:
        gate_failures.append(f"drawdown > {MAX_ACCEPTABLE_DRAWDOWN}")

    report_timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    versioned_model_path = MODEL_DIR / f"ml_signal_model_{report_timestamp}.pkl"

    bundle = train_ensemble(
        X,
        y,
        save_path=str(versioned_model_path),
        feature_names=feature_names,
        walk_forward_splits=WF_N_FOLDS,
        model_type="ensemble",
    )
    if bundle is None:
        raise RuntimeError("train_ensemble returned None")

    model_checksum = checksum_file(versioned_model_path)
    should_deploy = not gate_failures and not no_deploy and not force_no_deploy and not report_only
    deployment_reason = "passed_all_gates"
    if force_no_deploy:
        deployment_reason = "force_no_deploy"
    elif no_deploy:
        deployment_reason = "no_deploy"
    elif report_only:
        deployment_reason = "report_only"
    elif gate_failures:
        deployment_reason = "; ".join(gate_failures)

    if should_deploy:
        shutil.copy2(versioned_model_path, CURRENT_MODEL_PATH)

    registry = ModelRegistryV2(
        registry_path=MODEL_DIR / "model_registry_v2.json",
        model_dir=MODEL_DIR,
        live_model_path=CURRENT_MODEL_PATH,
    )
    registry.register_model(
        filename=versioned_model_path.name,
        checksum=model_checksum,
        dataset_size=sample_count,
        training_horizon=requested_horizon,
        requested_horizon=requested_horizon,
        effective_horizon=f"{effective_days} trading_days",
        dataset_start_timestamp=candles[0].time.isoformat(),
        dataset_end_timestamp=candles[-1].time.isoformat(),
        real_trading_days=effective_days,
        feature_count=len(feature_names),
        sample_count=sample_count,
        model_type="ensemble",
        roc_auc=float(metrics.get("roc_auc", 0.0) or 0.0),
        accuracy=float(metrics.get("accuracy", 0.0) or 0.0),
        precision=float(metrics.get("precision", 0.0) or 0.0),
        recall=float(metrics.get("recall", 0.0) or 0.0),
        f1=float(metrics.get("f1", 0.0) or 0.0),
        sharpe=float(metrics.get("sharpe", 0.0) or 0.0),
        sortino=float(metrics.get("sortino", 0.0) or 0.0),
        profit_factor=float(metrics.get("profit_factor", 0.0) or 0.0),
        drawdown=float(metrics.get("drawdown", 0.0) or 0.0),
        deployed=should_deploy,
        deployment_decision="deployed" if should_deploy else "rejected",
        rejection_reason=None if should_deploy else deployment_reason,
    )
    archive_old_models()
    feature_audit_summary = run_feature_audit(candles)
    model_comparison_path = generate_model_comparison_report(registry)

    report = {
        "run": {
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "label_policy": label_policy,
        },
        "dataset": {
            "raw_candles": len(raw_rows),
            "resampled_candles": len(candles),
            "requested_horizon": requested_horizon,
            "effective_horizon": f"{effective_days} trading_days",
            "real_trading_days": effective_days,
            "available_years": round(available_years, 4),
            "sample_count": sample_count,
            "feature_count": len(feature_names),
            "positive_rate": positive_rate,
        },
        "audits": {
            "static": static_audit,
            "dynamic": dynamic_audit,
        },
        "validation": {
            "passed": not gate_failures,
            "walk_forward": wf_result,
            "metrics": metrics,
            "gate_failures": gate_failures,
        },
        "deployment": {
            "decision": "DEPLOY" if should_deploy else "REJECT DEPLOYMENT",
            "reason": deployment_reason,
            "model_path": str(versioned_model_path),
            "current_model_path": str(CURRENT_MODEL_PATH) if should_deploy else None,
            "checksum": model_checksum,
        },
        "feature_governance": feature_audit_summary,
        "model_comparison_report": str(model_comparison_path),
    }
    json_path, md_path = write_report(report, report_timestamp)

    if should_deploy:
        notify(f"Retraining deployed successfully. Report: {json_path.name}")
    else:
        notify(f"Retraining completed without deployment. Reason: {deployment_reason}", is_error=bool(gate_failures))

    if gate_failures and not (report_only or no_deploy or force_no_deploy):
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Post-market retraining using real historical candles only.")
    parser.add_argument("--years", type=int, choices=[1, 2, 5, 10], help="Requested training horizon in years")
    parser.add_argument("--no-deploy", action="store_true", help="Train and validate, but do not deploy")
    parser.add_argument("--force-no-deploy", action="store_true", help="Force report-only training result without deployment")
    parser.add_argument("--rollback", action="store_true", help="Rollback to the previous deployed model")
    parser.add_argument("--report-only", action="store_true", help="Always exit zero after writing the report")
    parser.add_argument("--label-policy", default="current_triple_barrier", help="Versioned label policy name")
    args = parser.parse_args(argv)

    if args.rollback:
        registry = ModelRegistryV2(
            registry_path=MODEL_DIR / "model_registry_v2.json",
            model_dir=MODEL_DIR,
            live_model_path=CURRENT_MODEL_PATH,
        )
        return 0 if registry.rollback() else 1

    try:
        ok = run(
            years=args.years,
            no_deploy=args.no_deploy,
            force_no_deploy=args.force_no_deploy,
            report_only=args.report_only,
            label_policy=args.label_policy,
        )
    except Exception as exc:
        log.error("Retraining failed: %s", exc)
        if args.report_only:
            return 0
        return 1
    return 0 if ok or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
