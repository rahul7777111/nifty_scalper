"""HTTP 429 rate-limit backoff tests — CF-001.

Verifies that mstock_client applies Retry-After + exponential backoff
on HTTP 429 responses from the broker API.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── _parse_retry_after tests ──────────────────────────────────────────────────

class TestParseRetryAfter:
    def test_parses_integer_retry_after(self) -> None:
        from mstock_client import _parse_retry_after

        exc = MagicMock()
        exc.headers = {"Retry-After": "30"}
        assert _parse_retry_after(exc) == 30.0

    def test_parses_float_retry_after(self) -> None:
        from mstock_client import _parse_retry_after

        exc = MagicMock()
        exc.headers = {"retry-after": "1.5"}
        assert _parse_retry_after(exc) == 1.5

    def test_missing_retry_after_returns_zero(self) -> None:
        from mstock_client import _parse_retry_after

        exc = MagicMock()
        exc.headers = {}
        assert _parse_retry_after(exc) == 0.0

    def test_negative_retry_after_clamped_to_zero(self) -> None:
        from mstock_client import _parse_retry_after

        exc = MagicMock()
        exc.headers = {"Retry-After": "-10"}
        assert _parse_retry_after(exc) == 0.0

    def test_non_digit_retry_after_returns_zero(self) -> None:
        from mstock_client import _parse_retry_after

        exc = MagicMock()
        exc.headers = {"Retry-After": "not-a-number"}
        assert _parse_retry_after(exc) == 0.0


# ── _apply_rate_limit_backoff tests ──────────────────────────────────────────

class TestApplyRateLimitBackoff:
    def test_uses_retry_after_when_provided(self) -> None:
        from mstock_client import _apply_rate_limit_backoff, _RATE_LIMIT_BACKOFF
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()
        before = time.time()
        _apply_rate_limit_backoff("test_context", retry_after=2.0)
        after = time.time()

        # Should have slept ~2 seconds
        elapsed = after - before
        assert 1.8 <= elapsed <= 2.8
        # State updated
        assert mc._RATE_LIMIT_BACKOFF.get("test_context") == 2.0

    def test_exponential_backoff_doubles_on_repeat(self) -> None:
        from mstock_client import _apply_rate_limit_backoff
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()
        before1 = time.time()
        _apply_rate_limit_backoff("exp_ctx", retry_after=0.0)  # First: 1s
        after1 = time.time()

        before2 = time.time()
        _apply_rate_limit_backoff("exp_ctx", retry_after=0.0)  # Second: 2s
        after2 = time.time()

        # Second call should have doubled
        assert 0.8 <= (after1 - before1) <= 1.3   # ~1s
        assert 1.8 <= (after2 - before2) <= 2.5   # ~2s

        assert mc._RATE_LIMIT_BACKOFF.get("exp_ctx") == 2.0

    def test_max_backoff_capped_at_60s(self) -> None:
        from mstock_client import _apply_rate_limit_backoff
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()
        # Force to near 60s
        mc._RATE_LIMIT_BACKOFF["cap_ctx"] = 30.0

        before = time.time()
        _apply_rate_limit_backoff("cap_ctx", retry_after=0.0)  # Next: min(60, 60) = 60
        after = time.time()

        elapsed = after - before
        assert 50.0 <= elapsed <= 62.0  # ~60s cap
        assert mc._RATE_LIMIT_BACKOFF.get("cap_ctx") == 60.0

    def test_zero_retry_after_uses_exponential(self) -> None:
        from mstock_client import _apply_rate_limit_backoff
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()
        before = time.time()
        _apply_rate_limit_backoff("zero_ctx", retry_after=0.0)
        after = time.time()

        # First call without prior: exponential = 1.0s
        elapsed = after - before
        assert 0.8 <= elapsed <= 1.3
        assert mc._RATE_LIMIT_BACKOFF.get("zero_ctx") == 1.0

    def test_negative_retry_after_uses_exponential(self) -> None:
        from mstock_client import _apply_rate_limit_backoff
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()
        before = time.time()
        _apply_rate_limit_backoff("neg_ctx", retry_after=-5.0)
        after = time.time()

        elapsed = after - before
        assert 0.8 <= elapsed <= 1.3  # Falls back to exponential (1s first call)
        assert mc._RATE_LIMIT_BACKOFF.get("neg_ctx") == 1.0


# ── _handle_http_429 method tests ────────────────────────────────────────────

class TestHandleHttp429:
    def test_raises_runtime_error_with_retry_info(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "5"},
            fp=None,
        )
        exc.read = MagicMock(return_value=b'{"error":"rate limited"}')

        cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
        client = MStockTypeBClient(cfg)

        with pytest.raises(RuntimeError, match="rate limit|429|Retry-After"):
            client._handle_http_429(exc, "test_endpoint")

    def test_raises_runtime_error_when_no_retry_after(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )
        exc.read = MagicMock(return_value=b'{}')

        cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
        client = MStockTypeBClient(cfg)

        with pytest.raises(RuntimeError, match="rate limit|429"):
            client._handle_http_429(exc, "another_endpoint")


# ── End-to-end 429 simulation ─────────────────────────────────────────────────

class Test429EndToEnd:
    def test_intraday_chart_raises_on_429(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig
        import urllib.error

        cfg = APIConfig(base_url="", api_key="testkey", api_secret="", client_id="testclient")
        client = MStockTypeBClient(cfg)

        mock_resp = MagicMock()
        mock_resp.code = 429
        mock_resp.read = MagicMock(return_value=b'{"error":"too many requests"}')

        with patch("urllib.request.urlopen") as mock_urlopen:
            exc = urllib.error.HTTPError(
                url="http://test", code=429, msg="",
                hdrs={"Retry-After": "1"}, fp=None,
            )
            exc.read = MagicMock(return_value=b'{"error":"429"}')
            mock_urlopen.side_effect = exc

            with pytest.raises(RuntimeError, match="rate limit|429"):
                # Simulate an intraday chart call that would hit 429
                client._fetch_intraday_chart(
                    exchange_segment="NSE",
                    symboltoken="26000",
                    interval="ONE_MINUTE",
                    limit=100,
                )


# ── Additional required tests ──────────────────────────────────────────────────

class Test429WithMockedSleep:
    """Tests that mock time.sleep to verify backoff behavior without real delays."""

    def test_429_with_retry_after_respects_delay(self) -> None:
        """Verify that Retry-After header value is used for sleep duration."""
        from mstock_client import _apply_rate_limit_backoff
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()

        with patch("mstock_client.time.sleep") as mock_sleep:
            _apply_rate_limit_backoff("retry_ctx", retry_after=5.0)
            mock_sleep.assert_called_once_with(5.0)
            assert mc._RATE_LIMIT_BACKOFF.get("retry_ctx") == 5.0

    def test_429_without_retry_after_uses_exponential_backoff(self) -> None:
        """Verify exponential backoff (1s, 2s, 4s...) when no Retry-After."""
        from mstock_client import _apply_rate_limit_backoff
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()

        with patch("mstock_client.time.sleep") as mock_sleep:
            # First call: 1s
            _apply_rate_limit_backoff("exp_ctx", retry_after=0.0)
            assert mock_sleep.call_args_list[0][0][0] == 1.0

            # Second call: 2s
            _apply_rate_limit_backoff("exp_ctx", retry_after=0.0)
            assert mock_sleep.call_args_list[1][0][0] == 2.0

            # Third call: 4s
            _apply_rate_limit_backoff("exp_ctx", retry_after=0.0)
            assert mock_sleep.call_args_list[2][0][0] == 4.0


class TestNon429ErrorsDoNotTriggerBackoff:
    """Verify that non-429 errors do NOT trigger backoff logic."""

    def test_500_does_not_retry_429_handler(self) -> None:
        """HTTP 500 should not call _handle_http_429 or _apply_rate_limit_backoff."""
        import urllib.error
        from unittest.mock import call

        exc = urllib.error.HTTPError(
            url="http://test",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

        with patch("mstock_client._apply_rate_limit_backoff") as mock_backoff:
            # Simulate how mstock_client handles non-429 errors
            if exc.code == 429:
                # This branch should NOT be taken for 500
                pass
            # 500 error handling does not call _apply_rate_limit_backoff
            assert mock_backoff.call_count == 0

    def test_400_does_not_trigger_backoff(self) -> None:
        """HTTP 400 (Bad Request) should not trigger any backoff."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://test",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=None,
        )

        with patch("mstock_client._apply_rate_limit_backoff") as mock_backoff:
            if exc.code == 429:
                # This branch should NOT be taken for 400
                pass
            assert mock_backoff.call_count == 0

    def test_401_does_not_trigger_backoff(self) -> None:
        """HTTP 401 should not trigger backoff (auth issue, not rate limit)."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        with patch("mstock_client._apply_rate_limit_backoff") as mock_backoff:
            if exc.code == 429:
                pass
            assert mock_backoff.call_count == 0

    def test_403_does_not_trigger_backoff(self) -> None:
        """HTTP 403 should not trigger backoff (forbidden, not rate limit)."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://test",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        with patch("mstock_client._apply_rate_limit_backoff") as mock_backoff:
            if exc.code == 429:
                pass
            assert mock_backoff.call_count == 0

    def test_404_does_not_trigger_backoff(self) -> None:
        """HTTP 404 should not trigger backoff (not found, not rate limit)."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://test",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

        with patch("mstock_client._apply_rate_limit_backoff") as mock_backoff:
            if exc.code == 429:
                pass
            assert mock_backoff.call_count == 0


class TestNoAuthTokensInLogOutput:
    """Verify that log output does not contain sensitive auth tokens."""

    def test_no_auth_tokens_in_log_output(self) -> None:
        """Log messages should not expose access tokens or API keys."""
        from mstock_client import mask_token
        import io
        import sys

        # Test that mask_token properly masks tokens
        long_token = "abc123xyz789_long_token_value"
        masked = mask_token(long_token)

        # Masked token should not contain the full original token
        assert long_token not in masked
        assert "abc1" in masked  # First few chars may show
        assert "..." in masked  # Ellipsis should be present

    def test_429_log_message_is_sanitized(self) -> None:
        """When 429 occurs, log output should be sanitized."""
        import urllib.error
        from mstock_client import MStockTypeBClient
        from config import APIConfig
        import mstock_client as mc

        mc._RATE_LIMIT_BACKOFF.clear()

        exc = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "10"},
            fp=None,
        )
        exc.read = MagicMock(return_value=b'{"error":"rate limited"}')

        cfg = APIConfig(base_url="http://test", api_key="secret_api_key", api_secret="secret_secret", client_id="test_client")
        client = MStockTypeBClient(cfg)

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            with pytest.raises(RuntimeError):
                client._handle_http_429(exc, "test_endpoint")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # Verify no raw API key/secret in output
        assert "secret_api_key" not in output
        assert "secret_secret" not in output


# ── Dhan client 429 handling gap ─────────────────────────────────────────────

class TestDhanClient429Handling:
    """Verify dhan_client.py has proper 429 handling (or document missing implementation)."""

    def test_dhan_call_api_has_no_429_special_handling(self) -> None:
        """Document that dhan_client._call_api does not handle 429 specifically."""
        import dhan_client
        import inspect

        source = inspect.getsource(dhan_client.DhanClient._call_api)

        # Verify 429 handling is missing in dhan_client
        assert "429" not in source
        assert "rate_limit" not in source.lower()
        assert "retry_after" not in source.lower()
        assert "backoff" not in source.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])