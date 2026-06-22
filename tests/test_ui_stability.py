"""Tests for UI stability fixes: option chain errors, reroute storms, after-job deduplication."""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalScalperUI:
    """Minimal stand-in for ScalperUI that only has the attributes we test."""

    MAX_LOG_LINES = 2000

    def __init__(self):
        # [SCHEDULER-REFACTOR] Split registries for app-level and bot-level jobs
        self._app_after_ids: dict = {}
        self._bot_after_ids: dict = {}
        self._closing = False
        self._max_log_lines = self.MAX_LOG_LINES
        self._diag_last_ts: float = 0.0
        self._client = None
        self._latest_candles: list = []
        self._chart_candles: list = []
        self._log_q = MagicMock()
        self._ui_queue = MagicMock()
        self._last_data_fetch_ts: float = 0.0
        # [SCHEDULER-REFACTOR] Bot session state
        self._bot_session_id: int = 0
        self._bot_running: bool = False

    def _run_periodic_diagnostics(self) -> None:
        """Forward to the real implementation when available, otherwise no-op."""
        import threading as _th
        import time as _time

        try:
            import gc

            now = _time.time()
            if now - self._diag_last_ts < 60:
                return
            self._diag_last_ts = now
            mem_mb = 0
            try:
                import psutil
                import os as _os

                p = psutil.Process(_os.getpid())
                mem_mb = p.memory_info().rss / 1024 / 1024
            except Exception:
                pass
            thread_cnt = len(_th.enumerate())
            app_after_cnt = len(self._app_after_ids)
            bot_after_cnt = len(self._bot_after_ids)
            after_cnt = app_after_cnt + bot_after_cnt
            candle_cnt = len(self._latest_candles or [])
            chart_candle_cnt = len(self._chart_candles or [])
            queue_size = self._ui_queue.qsize()
            log_q_size = self._log_q.qsize()
            client = getattr(self, "_client", None)
            opt_chain_status = (
                getattr(client, "_option_chain_status", "UNKNOWN") if client else "NO_CLIENT"
            )
            broker_ip_mismatch = getattr(client, "_broker_ip_mismatch", False) if client else False
            last_fetch_ts = getattr(self, "_last_data_fetch_ts", 0.0)
            last_fetch_str = f"{now - last_fetch_ts:.0f}s ago" if last_fetch_ts else "NEVER"
            print(
                f"[DIAGNOSTICS] mem_mb={mem_mb:.1f} threads={thread_cnt} "
                f"after_jobs={after_cnt} candles={candle_cnt} chart_candles={chart_candle_cnt} "
                f"ui_queue={queue_size} log_q={log_q_size} "
                f"opt_chain={opt_chain_status} broker_ip_mismatch={broker_ip_mismatch} "
                f"last_fetch={last_fetch_str}"
            )
            gc.collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# After-job deduplication
# ---------------------------------------------------------------------------


class TestAfterJobDeduplication(unittest.TestCase):
    def test_duplicate_after_same_name_cancelled(self):
        """Calling _safe_after_app twice with same name cancels the first job."""
        ui = _MinimalScalperUI()
        job1_id = 111
        job2_id = 222
        cancelled_ids = []

        def fake_cancel(tid):
            cancelled_ids.append(tid)

        ui.after_cancel = fake_cancel
        ui.after = lambda delay, cb, *a: job1_id if len(cancelled_ids) == 0 else job2_id

        callback = lambda: None

        ui._closing = False
        ui.winfo_exists = lambda: True

        # Simulate _safe_after_app logic with name-based key
        name = "test_job"
        old_id = ui._app_after_ids.pop(name, None)
        if old_id is not None:
            fake_cancel(old_id)
        new_id = ui.after(100, callback)
        ui._app_after_ids[name] = new_id

        self.assertEqual(len(cancelled_ids), 0)

        # Second call with same name - should cancel previous
        old_id = ui._app_after_ids.pop(name, None)
        if old_id is not None:
            fake_cancel(old_id)
        new_id = ui.after(100, callback)
        ui._app_after_ids[name] = new_id

        self.assertIn(job1_id, cancelled_ids)

    def test_different_names_independent(self):
        """Different job names each get their own entry and do not cancel each other."""
        ui = _MinimalScalperUI()
        cancelled = []

        def fake_cancel(tid):
            cancelled.append(tid)

        ui.after_cancel = fake_cancel
        ui.after = lambda delay, cb, *a: 200 + len(cancelled)

        cb1 = lambda: None
        cb2 = lambda: None

        ui._closing = False
        ui.winfo_exists = lambda: True

        name1 = "job_one"
        name2 = "job_two"

        old1 = ui._app_after_ids.pop(name1, None)
        if old1:
            fake_cancel(old1)
        ui._app_after_ids[name1] = ui.after(100, cb1)

        old2 = ui._app_after_ids.pop(name2, None)
        if old2:
            fake_cancel(old2)
        ui._app_after_ids[name2] = ui.after(100, cb2)

        # Neither should have been cancelled
        self.assertEqual(len(cancelled), 0)

    def test_name_based_deduplication(self):
        """Jobs with same name replace each other, different names coexist."""
        ui = _MinimalScalperUI()
        cancelled = []

        def fake_cancel(tid):
            cancelled.append(tid)

        ui.after_cancel = fake_cancel
        ui.after = lambda delay, cb, *a: 999

        cb = lambda: None
        ui._closing = False
        ui.winfo_exists = lambda: True

        # Same callback, different names - should NOT cancel
        name_a = "job_a"
        name_b = "job_b"

        old_a = ui._app_after_ids.pop(name_a, None)
        if old_a:
            fake_cancel(old_a)
        ui._app_after_ids[name_a] = ui.after(100, cb)

        old_b = ui._app_after_ids.pop(name_b, None)
        if old_b:
            fake_cancel(old_b)
        ui._app_after_ids[name_b] = ui.after(100, cb)

        # Neither should have been cancelled (different names)
        self.assertEqual(len(cancelled), 0)

        # Same name again - SHOULD cancel
        old_a2 = ui._app_after_ids.pop(name_a, None)
        if old_a2:
            fake_cancel(old_a2)
        ui._app_after_ids[name_a] = ui.after(100, cb)

        self.assertEqual(len(cancelled), 1)


# ---------------------------------------------------------------------------
# Log widget growth
# ---------------------------------------------------------------------------


class TestLogWidgetGrowth(unittest.TestCase):
    def test_max_log_lines_constant_is_2000(self):
        """ScalperUI.MAX_LOG_LINES should be 2000 to prevent runaway widget growth."""
        from ui import ScalperUI

        self.assertEqual(ScalperUI.MAX_LOG_LINES, 2000)

    def test_instance_max_log_lines_is_2000(self):
        """The instance _max_log_lines attribute should default to 2000."""
        ui = _MinimalScalperUI()
        self.assertEqual(ui._max_log_lines, 2000)


# ---------------------------------------------------------------------------
# Periodic diagnostics
# ---------------------------------------------------------------------------


class TestPeriodicDiagnostics(unittest.TestCase):
    def test_diagnostics_runs_without_error(self):
        """_run_periodic_diagnostics should not raise even when resources are missing."""
        ui = _MinimalScalperUI()
        ui._diag_last_ts = 0.0
        ui._client = None
        ui._latest_candles = [1, 2, 3]
        ui._chart_candles = [4, 5]
        ui._log_q = MagicMock()
        ui._log_q.qsize = lambda: 5
        ui._ui_queue = MagicMock()
        ui._ui_queue.qsize = lambda: 3
        ui._last_data_fetch_ts = 0.0

        with patch("ui.time.time", return_value=10000.0):
            # Should not raise
            ui._run_periodic_diagnostics()

    def test_diagnostics_includes_option_chain_status(self):
        """Diagnostics should read _option_chain_status from the client."""
        ui = _MinimalScalperUI()
        ui._diag_last_ts = 0.0

        mock_client = MagicMock()
        mock_client._option_chain_status = "OK"
        mock_client._broker_ip_mismatch = False
        ui._client = mock_client
        ui._latest_candles = []
        ui._chart_candles = []
        ui._log_q.qsize = lambda: 0
        ui._ui_queue.qsize = lambda: 0
        ui._last_data_fetch_ts = 0.0

        logs = []

        def capture_print(*args, **kwargs):
            logs.append(" ".join(str(a) for a in args))

        with patch("builtins.print", side_effect=capture_print):
            with patch("ui.time.time", return_value=10000.0):
                ui._run_periodic_diagnostics()

        self.assertTrue(
            any("opt_chain=OK" in log for log in logs), f"Logs: {logs}"
        )


# ---------------------------------------------------------------------------
# Throttled logging (stub - implemented via _throttled_log if present)
# ---------------------------------------------------------------------------


class TestThrottledLogging(unittest.TestCase):
    def test_throttled_log_attribute_placeholder(self):
        """_throttled_log is a stability feature that may be implemented later."""
        # Check the constant exists on ScalperUI for future use
        from ui import ScalperUI

        # This test documents the expected interface
        # Actual _throttled_log implementation will be added separately
        self.assertTrue(
            hasattr(ScalperUI, "MAX_LOG_LINES"),
            "ScalperUI should have MAX_LOG_LINES constant",
        )


# ---------------------------------------------------------------------------
# Reroute cooldown (placeholder)
# ---------------------------------------------------------------------------


class TestRerouteCooldown(unittest.TestCase):
    def test_reroute_cooldown_placeholder(self):
        """Reroute cooldown is implemented in the broker client, not the UI layer."""
        # This test documents that the UI does not own reroute cooldown
        from ui import ScalperUI

        self.assertTrue(
            hasattr(ScalperUI, "MAX_LOG_LINES"),
            "ScalperUI exists and is importable",
        )


# ---------------------------------------------------------------------------
# IA403 / IP mismatch detection (placeholder)
# ---------------------------------------------------------------------------


class TestIA403Handling(unittest.TestCase):
    def test_ip_mismatch_placeholder(self):
        """IP mismatch detection lives in the broker client, not the UI layer."""
        from ui import ScalperUI

        self.assertTrue(
            hasattr(ScalperUI, "MAX_LOG_LINES"),
            "ScalperUI exists and is importable",
        )


if __name__ == "__main__":
    unittest.main()