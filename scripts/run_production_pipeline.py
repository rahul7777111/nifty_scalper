from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
CHECKPOINT_PATH = REPORTS_DIR / ".benchmark_checkpoint.json"
PROMOTION_DEGRADATION_MD = REPORTS_DIR / "promotion_degradation_report.md"
PRODUCTION_PROFILE_PATH = REPO_ROOT / "config" / "production_model_profile.json"
LIVE_MODEL_PATH = REPO_ROOT / "ml_signal_model.pkl"
LOG_PATH = REPORTS_DIR / "production_pipeline.log"
ENGINE_DIAGNOSTICS_PATH = REPO_ROOT / ".engine_diagnostics.json"
PREDICTION_DRIFT_REPORT_PATH = REPORTS_DIR / "prediction_drift_report.json"


def _configure_logging() -> logging.Logger:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_production_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


log = _configure_logging()


@dataclass
class PhaseResult:
    name: str
    returncode: int
    timed_out: bool = False


def _stream_subprocess_output(proc: subprocess.Popen[str], prefix: str) -> threading.Thread:
    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.info("%s | %s", prefix, line.rstrip())

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def run_subprocess_phase(
    *,
    name: str,
    cmd: Sequence[str],
    timeout_sec: Optional[float] = None,
) -> PhaseResult:
    log.info("Starting %s: %s", name, " ".join(cmd))
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    reader = _stream_subprocess_output(proc, name)
    timed_out = False
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        log.warning("%s timed out after %.1f seconds", name, float(timeout_sec or 0.0))
        proc.terminate()
        try:
            proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            log.warning("%s did not terminate gracefully; killing process", name)
            proc.kill()
            proc.wait(timeout=5.0)
    reader.join(timeout=2.0)
    return PhaseResult(name=name, returncode=int(proc.returncode or 0), timed_out=timed_out)


def prompt_retry_benchmark() -> bool:
    auto_retry = str(os.getenv("PIPELINE_AUTO_RETRY_BENCHMARK", "")).strip().lower()
    if auto_retry in {"1", "true", "yes", "y"}:
        log.info("Auto-retry benchmark enabled by environment.")
        return True
    if not sys.stdin.isatty():
        log.info("No interactive stdin available; benchmark retry declined.")
        return False
    while True:
        try:
            raw = input("Benchmark interrupted. Retry from checkpoint? [y/N]: ").strip().lower()
        except EOFError:
            return False
        if raw in {"y", "yes"}:
            return True
        if raw in {"", "n", "no"}:
            return False
        print("Please answer y or n.")


def refresh_offline_alignment() -> None:
    phases = [
        ("PHASE1_LABEL_DIAGNOSTICS", [sys.executable, str(REPO_ROOT / "tools" / "label_diagnostics.py")]),
        ("PHASE1_REPORT_REBUILD", [sys.executable, str(REPO_ROOT / "scripts" / "generate_label_redesign_reports.py")]),
    ]
    for name, cmd in phases:
        result = run_subprocess_phase(name=name, cmd=cmd, timeout_sec=300.0)
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def run_fault_tolerant_benchmark() -> None:
    while True:
        result = run_subprocess_phase(
            name="PHASE2_BENCHMARK",
            cmd=[sys.executable, str(REPO_ROOT / "scripts" / "train_window_benchmark.py")],
            timeout_sec=float(os.getenv("PIPELINE_BENCHMARK_TIMEOUT_SEC", "0") or 0.0) or None,
        )
        if result.returncode == 0:
            log.info("Benchmark phase completed successfully.")
            return
        if result.timed_out or result.returncode != 0:
            if CHECKPOINT_PATH.exists():
                log.warning(
                    "Benchmark phase interrupted (exit=%s, timed_out=%s). Checkpoint detected at %s.",
                    result.returncode,
                    result.timed_out,
                    CHECKPOINT_PATH,
                )
            else:
                log.warning(
                    "Benchmark phase interrupted (exit=%s, timed_out=%s) and no checkpoint file was found.",
                    result.returncode,
                    result.timed_out,
                )
            if prompt_retry_benchmark():
                log.info("Retrying benchmark phase from checkpoint-aware runner.")
                continue
        raise RuntimeError(f"Benchmark phase failed with exit code {result.returncode}")


def run_promotion_gate() -> None:
    result = run_subprocess_phase(
        name="PHASE3_PROMOTION",
        cmd=[sys.executable, str(REPO_ROOT / "scripts" / "promote_to_production.py")],
        timeout_sec=180.0,
    )
    if result.returncode == 0:
        if not PRODUCTION_PROFILE_PATH.exists() or not LIVE_MODEL_PATH.exists():
            raise RuntimeError("Promotion phase reported success, but production artifacts are missing.")
        return

    if PROMOTION_DEGRADATION_MD.exists():
        log.error("Promotion blocked. Printing degradation report:")
        content = PROMOTION_DEGRADATION_MD.read_text(encoding="utf-8")
        for line in content.splitlines():
            log.error("%s", line)
    raise RuntimeError("Promotion gate failed; aborting Phase 4.")


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            log.info("Loaded .env from %s", env_path)
    except Exception:
        log.info("python-dotenv unavailable or .env load skipped")


def _shadow_stage_worker(stop_event: threading.Event, exception_bucket: List[BaseException]) -> None:
    try:
        if str(REPO_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "src"))
        _load_env_file()
        from config import load_api_config, load_strategy_config
        from mstock_client import MStockTypeBClient
        from prediction_drift_monitor import PredictionDriftMonitor
        from strategy import NiftyScalper

        api_cfg = load_api_config()
        strat_cfg = load_strategy_config()
        client = MStockTypeBClient(api_cfg)

        log.info("Shadow stage: logging in to m.Stock")
        client.login(interactive=False)
        log.info("Shadow stage: broker login successful")

        scalper = NiftyScalper(client, strat_cfg)
        notifier = getattr(scalper, "notifier", None)
        drift_monitor = PredictionDriftMonitor(
            notifier=notifier,
            strategy_ref=scalper,
        )

        try:
            diag = scalper.run_enhancements_diagnostics()
            log.info("Shadow stage diagnostics: %s", diag)
        except Exception as exc:
            log.warning("Shadow stage diagnostics failed: %s", exc)

        try:
            import gpt

            gpt.health_check(print_result=True)
        except Exception as exc:
            log.warning("GPT health check failed: %s", exc)

        def _run_strategy() -> None:
            try:
                scalper.run_forever(stop_event=stop_event)
            except BaseException as exc:  # noqa: BLE001
                exception_bucket.append(exc)
                stop_event.set()

        def _run_monitor() -> None:
            try:
                drift_monitor.run(stop_event=stop_event, poll_interval_sec=float(os.getenv("PIPELINE_DRIFT_POLL_SEC", "1.0") or 1.0))
            except BaseException as exc:  # noqa: BLE001
                exception_bucket.append(exc)
                stop_event.set()

        strategy_thread = threading.Thread(target=_run_strategy, name="shadow-strategy", daemon=True)
        monitor_thread = threading.Thread(target=_run_monitor, name="shadow-drift-monitor", daemon=True)
        strategy_thread.start()
        monitor_thread.start()

        start = time.time()
        telemetry_confirmed = False
        while not stop_event.is_set():
            if exception_bucket:
                break
            if not telemetry_confirmed and (ENGINE_DIAGNOSTICS_PATH.exists() or PREDICTION_DRIFT_REPORT_PATH.exists()):
                telemetry_confirmed = True
                log.info(
                    "Shadow stage telemetry active | engine_diagnostics=%s drift_report=%s",
                    ENGINE_DIAGNOSTICS_PATH.exists(),
                    PREDICTION_DRIFT_REPORT_PATH.exists(),
                )
            if time.time() - start > float(os.getenv("PIPELINE_SHADOW_HEALTH_WARN_SEC", "30") or 30.0) and not telemetry_confirmed:
                telemetry_confirmed = True
                log.warning(
                    "Shadow stage threads are running, but telemetry files have not both appeared yet | engine_diagnostics=%s drift_report=%s",
                    ENGINE_DIAGNOSTICS_PATH.exists(),
                    PREDICTION_DRIFT_REPORT_PATH.exists(),
                )
            time.sleep(1.0)

        stop_event.set()
        strategy_thread.join(timeout=15.0)
        monitor_thread.join(timeout=15.0)
    except BaseException as exc:  # noqa: BLE001
        exception_bucket.append(exc)
        stop_event.set()


def launch_shadow_stage() -> None:
    stop_event = threading.Event()
    exceptions: List[BaseException] = []

    def _handle_signal(signum: int, _frame: object) -> None:
        log.warning("Received signal %s, stopping shadow stage.", signum)
        stop_event.set()

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        log.info("Starting Phase 4 shadow staging deployment.")
        worker = threading.Thread(
            target=_shadow_stage_worker,
            args=(stop_event, exceptions),
            name="shadow-stage-root",
            daemon=False,
        )
        worker.start()
        worker.join()
        if exceptions:
            raise RuntimeError(f"Shadow stage failed: {exceptions[0]}")
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> int:
    log.info("=== Production pipeline started ===")
    try:
        refresh_offline_alignment()
        run_fault_tolerant_benchmark()
        run_promotion_gate()
        launch_shadow_stage()
    except Exception as exc:  # noqa: BLE001
        log.error("Production pipeline failed: %s", exc)
        return 1
    log.info("=== Production pipeline completed cleanly ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
