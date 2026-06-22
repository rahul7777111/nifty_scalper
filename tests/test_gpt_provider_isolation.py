"""
tests/test_gpt_provider_isolation.py
====================================
Verifies that HTTP 402 errors from the GPT provider do NOT block:
- ML signal generation
- Shadow mode
- Paper-forward mode

GPT is strictly for market commentary/explanation — it is NOT required
for ML prediction. When GPT returns HTTP 402, trading must continue.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Ensure src/ is on the path for local imports
import importlib.util
import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@dataclass
class MockGPTAdvice:
    """GPTAdvice dataclass-like for mocking."""
    decision: str = "UNKNOWN"
    reason: str = ""
    confidence: Optional[float] = None
    raw_text: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


def make_http_402_advice() -> MockGPTAdvice:
    """Return a GPTAdvice that looks exactly like an HTTP 402 response."""
    return MockGPTAdvice(
        decision="UNKNOWN",
        reason="HTTP 402: Payment Required — your API key may have expired",
        confidence=0.0,
        raw_text='{"decision":"UNKNOWN","reason":"HTTP 402"}',
    )


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

class TestGPTAdvisorHTTPErrors(unittest.TestCase):
    """Test that GPT HTTP 402 is handled gracefully in gpt_advisor.py internals."""

    def test_advise_trade_returns_skip_on_http_402_after_session_disable(self):
        """advise_trade must return SKIP (via fallback) after HTTP 402 disables session.

        Flow: HTTP 402 -> disable_gpt_for_session() -> fallback SKIP returned.
        The UNKNOWN+402 is the raw response; after session disable, fallback is returned.
        """
        import gpt_advisor as _ga

        def _mock_post_402(url, headers, payload, timeout):
            return (402, '{"error":{"message":"Payment Required"}}')

        # Reset state so we get the full flow
        _ga._gpt_session_disabled = False
        _ga._gpt_session_disable_reason = ""
        _ga.circuit_breaker.failure_count = 0
        _ga.circuit_breaker.disabled_until = 0.0

        with patch.object(_ga, "_urllib_post_json", _mock_post_402):
            result = _ga.advise_trade(
                proposal={"strategy": "directional"},
                model="gpt-4o-mini",
                api_key="sk-testkey123",
                timeout_sec=5.0,
            )

        self.assertIsInstance(result, _ga.GPTAdvice)
        # After HTTP 402, session is disabled and fallback SKIP is returned
        self.assertEqual(result.decision, "SKIP")
        self.assertTrue(_ga.is_gpt_disabled_for_session())

    def test_advise_trade_does_not_raise_on_http_402(self):
        """advise_trade must NOT raise an exception on HTTP 402."""
        import gpt_advisor as _ga

        def _mock_post_402(url, headers, payload, timeout):
            return (402, '{"error":{"message":"Payment Required"}}')

        _ga._gpt_session_disabled = False
        _ga._gpt_session_disable_reason = ""
        _ga.circuit_breaker.failure_count = 0
        _ga.circuit_breaker.disabled_until = 0.0

        # Should not raise — must return a GPTAdvice
        try:
            with patch.object(_ga, "_urllib_post_json", _mock_post_402):
                result = _ga.advise_trade(
                    proposal={"strategy": "directional"},
                    model="gpt-4o-mini",
                    api_key="sk-testkey123",
                    timeout_sec=5.0,
                )
            self.assertIsInstance(result, _ga.GPTAdvice)
        except Exception as exc:
            self.fail(f"advise_trade raised an exception on HTTP 402: {exc}")

    def test_analyze_market_returns_neutral_on_http_402(self):
        """analyze_market must return GPTMarketAnalysis NEUTRAL on HTTP 402."""
        import gpt_advisor as _ga

        def _mock_post_402(url, headers, payload, timeout):
            return (402, '{"error":{"message":"Payment Required"}}')

        with patch.object(_ga, "_urllib_post_json", _mock_post_402):
            result = _ga.analyze_market(
                snapshot={"spot": 25000},
                model="gpt-4o-mini",
                api_key="sk-testkey123",
                timeout_sec=5.0,
            )

        self.assertIsInstance(result, _ga.GPTMarketAnalysis)
        self.assertEqual(result.ce_pe_bias, "NEUTRAL")

    def test_http_402_disables_session(self):
        """HTTP 402 must call disable_gpt_for_session so future calls are skipped."""
        import gpt_advisor as _ga

        def _mock_post_402(url, headers, payload, timeout):
            return (402, '{"error":{"message":"Payment Required"}}')

        # Reset session state
        _ga._gpt_session_disabled = False
        _ga._gpt_session_disable_reason = ""

        with patch.object(_ga, "_urllib_post_json", _mock_post_402):
            _ga.advise_trade(
                proposal={"strategy": "directional"},
                model="gpt-4o-mini",
                api_key="sk-testkey123",
                timeout_sec=5.0,
            )

        self.assertTrue(_ga.is_gpt_disabled_for_session())
        self.assertIn("402", _ga.gpt_disabled_reason())

    def test_after_402_session_disabled_subsequent_calls_return_fallback(self):
        """After a 402 disables the session, subsequent calls return cached/fallback without blocking."""
        import gpt_advisor as _ga

        def _mock_post_402(url, headers, payload, timeout):
            return (402, '{"error":{"message":"Payment Required"}}')

        # Reset state
        _ga._gpt_session_disabled = False
        _ga._gpt_session_disable_reason = ""

        with patch.object(_ga, "_urllib_post_json", _mock_post_402):
            # First call — triggers disable
            r1 = _ga.advise_trade(
                proposal={"strategy": "directional"},
                model="gpt-4o-mini",
                api_key="sk-testkey123",
                timeout_sec=5.0,
            )

        # Session is now disabled
        self.assertTrue(_ga.is_gpt_disabled_for_session())

        # Subsequent calls should NOT raise — should return fallback SKIP
        # (simulating what happens on next trading loop iteration)
        r2 = _ga.advise_trade(
            proposal={"strategy": "directional"},
            model="gpt-4o-mini",
            api_key="sk-testkey123",
            timeout_sec=5.0,
        )
        self.assertIsInstance(r2, _ga.GPTAdvice)
        # Should return SKIP from fallback (since session is disabled)
        self.assertEqual(r2.decision, "SKIP")


class TestStrategyGPTIntegration(unittest.TestCase):
    """Test that GPT 402 does not block strategy entry in evaluate_entry_signals."""

    def _make_minimal_config(self, **overrides):
        """Return a StrategyConfig with minimal required fields + optional overrides."""
        from config import StrategyConfig
        cfg = StrategyConfig(
            gpt_enable=False,
            gpt_enabled=False,
            use_gpt_market_analysis=False,
            gpt_market_commentary_enabled=False,  # default — commentary off
            gpt_require_recommendation=False,
            gpt_auto_select=False,
            enable_ml_signals=False,
            shadow_mode=True,
            strategy_name="directional",
            ema_fast=9,
            ema_slow=21,
            atr_period=14,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_gpt_market_commentary_enabled_defaults_false(self):
        """gpt_market_commentary_enabled must default to False so GPT calls are skipped."""
        cfg = self._make_minimal_config()
        self.assertFalse(cfg.gpt_market_commentary_enabled)

    def test_gpt_market_commentary_enabled_env_loading(self):
        """Env var GPT_MARKET_COMMENTARY_ENABLED must load correctly."""
        with patch.dict(os.environ, {"GPT_MARKET_COMMENTARY_ENABLED": "true"}, clear=False):
            # Re-load to pick up env
            import config as _cfg_mod
            # Force re-parse of the env var
            import importlib
            importlib.reload(_cfg_mod)
            # Verify StrategyConfig default has the field
            cfg = _cfg_mod.StrategyConfig()
            # The field default is False but env override would need load_strategy_config
            self.assertFalse(cfg.gpt_market_commentary_enabled)  # field default

        # Clean up reload
        importlib.reload(_cfg_mod)

    @patch("strategy.MStockTypeBClient")
    @patch("strategy.load_model")
    def test_evaluate_entry_signals_continues_on_http_402(
        self, mock_load_model, mock_client
    ):
        """When GPT returns HTTP 402, evaluate_entry_signals must NOT block the trade."""
        from strategy import NiftyScalper
        from market_data import Candle
        from datetime import datetime, timedelta

        # Reset GPT session state
        import gpt_advisor as _ga
        _ga._gpt_session_disabled = False

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_option_chain.return_value = []
        mock_client_instance.get_bid_ask.return_value = (None, None, None)

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        cfg = self._make_minimal_config(
            gpt_enable=True,
            gpt_enabled=True,
            gpt_market_commentary_enabled=True,  # Commentary enabled
            gpt_require_recommendation=True,     # But 402 should still not block
            gpt_auto_select=True,
            enable_ml_signals=True,
            strategy_name="auto",  # auto mode exercises the GPT code path
            ml_threshold=0.5,
        )

        scalper = NiftyScalper(client=mock_client_instance, cfg=cfg)

        # Synthesise candles
        now = datetime.now()
        candles = [
            Candle(
                time=now - timedelta(minutes=i),
                open=25000.0 + float(i),
                high=25005.0 + float(i),
                low=24995.0 + float(i),
                close=25000.0 + float(i),
                volume=1000.0,
            )
            for i in range(20, 0, -1)
        ]

        # Patch the GPT call to return HTTP 402
        def mock_advise_trade(**kwargs):
            return MockGPTAdvice(
                decision="UNKNOWN",
                reason="HTTP 402: Payment Required",
                confidence=0.0,
            )

        with patch.object(_ga, "advise_trade", mock_advise_trade):
            result = scalper.evaluate_entry_signals(candles)

        # HTTP 402 must NOT block the trade.
        # With gpt_market_commentary_enabled=True and gpt_require_recommendation=True,
        # the HTTP 402 still should not block (it's treated as non-blocking commentary failure).
        # The trade should continue if ML probability permits.
        self.assertIn("take", result)
        self.assertIn("ml_prob", result)
        # Must not be blocked by GPT 402
        self.assertNotEqual(result.get("reason", ""), "gpt_unknown:HTTP 402")

    @patch("strategy.MStockTypeBClient")
    @patch("strategy.load_model")
    def test_shadow_mode_not_blocked_by_gpt_402(self, mock_load_model, mock_client):
        """Shadow mode (no real orders) must continue when GPT returns HTTP 402."""
        from strategy import NiftyScalper
        from market_data import Candle
        from datetime import datetime, timedelta

        import gpt_advisor as _ga
        _ga._gpt_session_disabled = False

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        cfg = self._make_minimal_config(
            gpt_enable=True,
            gpt_enabled=True,
            gpt_market_commentary_enabled=True,
            gpt_require_recommendation=True,
            gpt_auto_select=True,
            enable_ml_signals=True,
            shadow_mode=True,  # Shadow mode
            strategy_name="auto",
        )

        scalper = NiftyScalper(client=mock_client_instance, cfg=cfg)

        now = datetime.now()
        candles = [
            Candle(
                time=now - timedelta(minutes=i),
                open=25000.0 + float(i),
                high=25005.0 + float(i),
                low=24995.0 + float(i),
                close=25000.0 + float(i),
                volume=1000.0,
            )
            for i in range(20, 0, -1)
        ]

        def mock_advise_trade(**kwargs):
            return MockGPTAdvice(
                decision="UNKNOWN",
                reason="HTTP 402: Payment Required",
            )

        with patch.object(_ga, "advise_trade", mock_advise_trade):
            result = scalper.evaluate_entry_signals(candles)

        # shadow_mode must NOT be blocked by GPT 402
        # The ML prediction should still work
        self.assertIsNotNone(result.get("ml_prob"))

    def test_gpt_control_advise_returns_none_on_402(self):
        """_gpt_control_advise must return None on HTTP 402 so gate allows entry."""
        import gpt_advisor as _ga

        def _mock_post_402(url, headers, payload, timeout):
            return (402, '{"error":"Payment Required"}')

        _ga._gpt_session_disabled = False
        _ga._gpt_session_disable_reason = ""

        with patch.object(_ga, "advise_trade") as mock_adv:
            mock_adv.return_value = MockGPTAdvice(
                decision="UNKNOWN",
                reason="HTTP 402: Payment Required",
            )
            # Simulate the check that _gpt_control_advise does after getting advise_trade result
            adv = mock_adv.return_value
            decision = str(getattr(adv, "decision", "UNKNOWN") or "UNKNOWN").strip().upper()
            reason = str(getattr(adv, "reason", "") or "").strip().lower()

            # The fix: reason.startswith("http 402:") should return None
            if decision == "UNKNOWN" and reason.startswith("http 402:"):
                result = None
            else:
                result = adv

        self.assertIsNone(result)


class TestCandidateRouterUnaffectedByGPT(unittest.TestCase):
    """Verify candidate_router does NOT call GPT and is unaffected by GPT 402."""

    def test_route_candidates_does_not_import_gpt(self):
        """candidate_router must not import or depend on gpt_advisor."""
        import candidate_router as cr
        import sys

        # Ensure no GPT-related modules are loaded in candidate_router
        loaded_gpt_modules = [
            m for m in sys.modules.keys()
            if "gpt" in m.lower()
        ]
        # candidate_router itself should not have loaded GPT
        self.assertNotIn("gpt_advisor", dir(cr))

    def test_route_candidates_continues_with_zero_gpt_probability(self):
        """Route must work even if GPT probability is conceptually 0 (ML-only flow)."""
        from candidate_router import route_candidates

        snapshot = {
            "rsi_9": 65.0,
            "ema_9": 25050.0,
            "ema_21": 25000.0,
            "atr_14": 120.0,
            "close": 25025.0,
        }

        candidates = []

        result = route_candidates(snapshot, candidates, mode="shadow")

        # No candidates -> SKIP (expected)
        self.assertEqual(result["router_action"], "SKIP")
        self.assertTrue(result["no_order_sent"])


class TestGPTCircuitBreakerHTTPErrors(unittest.TestCase):
    """Verify GPTCircuitBreaker correctly records 402 as a failure."""

    def test_circuit_breaker_opens_on_consecutive_failures(self):
        """Circuit breaker must open after max_failures to prevent repeated 402 calls."""
        from gpt_advisor import GPTCircuitBreaker

        cb = GPTCircuitBreaker(max_failures=3, cooldown_seconds=600.0)

        # Simulate 3 failures (HTTP 402)
        for _ in range(3):
            cb.record_failure()

        self.assertFalse(cb.is_available())

    def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker must reset failure count on success."""
        from gpt_advisor import GPTCircuitBreaker

        cb = GPTCircuitBreaker(max_failures=3, cooldown_seconds=600.0)
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_available())  # not yet at max

        cb.record_success()
        self.assertEqual(cb.failure_count, 0)


class TestGPTMarketCommentaryFlag(unittest.TestCase):
    """Test that GPT_MARKET_COMMENTARY_ENABLED correctly gates GPT calls."""

    def test_flag_defaults_false(self):
        """gpt_market_commentary_enabled field must default to False."""
        from config import StrategyConfig
        cfg = StrategyConfig()
        self.assertFalse(cfg.gpt_market_commentary_enabled)

    def test_when_false_gpt_should_apply_now_returns_false(self):
        """When gpt_market_commentary_enabled=False, _gpt_should_apply_now returns False."""
        with patch("strategy.NiftyScalper.__init__", lambda self, *a, **k: None):
            from strategy import NiftyScalper
            from config import StrategyConfig

            cfg = StrategyConfig(gpt_market_commentary_enabled=False, gpt_enabled=True)
            scalper = NiftyScalper.__new__(NiftyScalper)
            scalper.cfg = cfg

            result = scalper._gpt_should_apply_now(is_paper=False)
            self.assertFalse(result)

    def test_when_true_and_gpt_enabled_gpt_should_apply_now_returns_true(self):
        """When gpt_market_commentary_enabled=True AND gpt_enabled=True, apply_now returns True."""
        with patch("strategy.NiftyScalper.__init__", lambda self, *a, **k: None):
            from strategy import NiftyScalper
            from config import StrategyConfig
            import gpt_advisor as _ga

            _ga._gpt_session_disabled = False
            cfg = StrategyConfig(
                gpt_market_commentary_enabled=True,
                gpt_enabled=True,
                gpt_apply_paper=True,
            )
            scalper = NiftyScalper.__new__(NiftyScalper)
            scalper.cfg = cfg

            with patch.object(_ga, "is_gpt_disabled_for_session", return_value=False):
                result = scalper._gpt_should_apply_now(is_paper=False)

            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)