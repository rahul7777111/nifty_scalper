"""Tests for m.Stock JWT token expiry handling (CF-001).

Covers:
- decode_mstock_jwt: base64url JWT payload decoding
- mstock_token_expiry_epoch: exp claim extraction
- is_mstock_token_expiring: expiry-within-N-seconds check
- safe_refresh_mstock_token: TOTP refresh (mocked)
- Edge cases: empty, malformed, missing exp, expired, expiring soon
- No raw token in any log/print output
"""
from __future__ import annotations

import sys
import time
from unittest import mock

# Ensure src/ is on the path for auth module
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("MSTOCK_USERNAME", "testuser")
os.environ.setdefault("MSTOCK_PASSWORD", "testpass")
os.environ.setdefault("MSTOCK_API_KEY", "testkey")
os.environ.setdefault("MSTOCK_TOTP_SECRET", "TESTTESTTEST")

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jwt(payload: dict, header: str = "eyJhbGciOiJIUzI1NiJ9") -> str:
    """Build a signed-looking JWT (header.payload.signature) with a fixed header."""
    import base64, json

    def _b64(d: dict) -> str:
        raw = json.dumps(d, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    return f"{header}.{_b64(payload)}.FAKE_SIGNATURE"


def _make_exp_payload(exp_seconds_from_now: int) -> dict:
    """Return a JWT payload dict with an exp key set to now + exp_seconds_from_now."""
    return {"exp": int(time.time()) + exp_seconds_from_now, "sub": "testuser"}


# ── Tests: decode_mstock_jwt ──────────────────────────────────────────────────

class TestDecodeMStockJwt:
    def test_valid_jwt_returns_payload(self):
        from auth import decode_mstock_jwt

        payload = {"exp": int(time.time()) + 3600, "sub": "user@example.com"}
        token = _make_jwt(payload)
        result = decode_mstock_jwt(token)
        assert result == payload

    def test_valid_jwt_with_nested_dict(self):
        from auth import decode_mstock_jwt

        payload = {"exp": int(time.time()) + 7200, "data": {"key": "value"}}
        token = _make_jwt(payload)
        result = decode_mstock_jwt(token)
        assert result == payload

    def test_empty_string_returns_empty_dict(self):
        from auth import decode_mstock_jwt

        assert decode_mstock_jwt("") == {}
        assert decode_mstock_jwt("   ") == {}

    def test_none_token_returns_empty_dict(self):
        from auth import decode_mstock_jwt

        assert decode_mstock_jwt(None) == {}

    def test_malformed_not_three_parts(self):
        from auth import decode_mstock_jwt

        assert decode_mstock_jwt("notajwt") == {}
        assert decode_mstock_jwt("onlyone") == {}
        assert decode_mstock_jwt("two.parts.here.too.many") == {}

    def test_malformed_base64_in_payload(self):
        from auth import decode_mstock_jwt

        # Valid header, invalid base64 in payload
        result = decode_mstock_jwt("eyJhbGciOiJIUzI1NiJ9.invalid!!!base64.fake")
        assert result == {}

    def test_valid_header_invalid_payload_json(self):
        from auth import decode_mstock_jwt

        # Header is valid b64; payload is valid b64 but not JSON
        import base64

        bad_payload = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
        token = f"eyJhbGciOiJIUzI1NiJ9.{bad_payload}.FAKE"
        assert decode_mstock_jwt(token) == {}

    def test_payload_is_list_not_dict(self):
        from auth import decode_mstock_jwt

        import base64, json

        list_payload = base64.urlsafe_b64encode(
            json.dumps([1, 2, 3]).encode()
        ).rstrip(b"=").decode()
        token = f"eyJhbGciOiJIUzI1NiJ9.{list_payload}.FAKE"
        # The function checks isinstance(parsed, dict) and returns {} for non-dict
        assert decode_mstock_jwt(token) == {}


# ── Tests: mstock_token_expiry_epoch ─────────────────────────────────────────

class TestMStockTokenExpiryEpoch:
    def test_valid_token_with_exp_returns_epoch(self):
        from auth import mstock_token_expiry_epoch

        exp_time = int(time.time()) + 3600
        token = _make_jwt({"exp": exp_time})
        assert mstock_token_expiry_epoch(token) == exp_time

    def test_missing_exp_returns_none(self):
        from auth import mstock_token_expiry_epoch

        token = _make_jwt({"sub": "user"})  # no "exp" key
        assert mstock_token_expiry_epoch(token) is None

    def test_exp_is_string_returns_none(self):
        from auth import mstock_token_expiry_epoch

        token = _make_jwt({"exp": "not_an_integer"})
        assert mstock_token_expiry_epoch(token) is None

    def test_empty_token_returns_none(self):
        from auth import mstock_token_expiry_epoch

        assert mstock_token_expiry_epoch("") is None
        assert mstock_token_expiry_epoch(None) is None

    def test_malformed_token_returns_none(self):
        from auth import mstock_token_expiry_epoch

        assert mstock_token_expiry_epoch("not.a.jwt") is None


# ── Tests: is_mstock_token_expiring ──────────────────────────────────────────

class TestIsMStockTokenExpiring:
    def test_valid_jwt_not_expiring_returns_false(self):
        from auth import is_mstock_token_expiring

        # Token valid for 2 hours, check within 15 min → should be False
        token = _make_jwt(_make_exp_payload(7200))
        assert is_mstock_token_expiring(token, within_seconds=900) is False

    def test_valid_jwt_expiring_soon_returns_true(self):
        from auth import is_mstock_token_expiring

        # Token valid for only 5 minutes (300 s), check within 15 min → should be True
        token = _make_jwt(_make_exp_payload(300))
        assert is_mstock_token_expiring(token, within_seconds=900) is True

    def test_expired_jwt_returns_true(self):
        from auth import is_mstock_token_expiring

        # Token expired 60 seconds ago
        token = _make_jwt(_make_exp_payload(-60))
        assert is_mstock_token_expiring(token, within_seconds=900) is True

    def test_token_expiring_at_exactly_threshold_returns_true(self):
        from auth import is_mstock_token_expiring

        # Token expires in exactly 900 seconds
        token = _make_jwt(_make_exp_payload(900))
        # exp <= now + 900  →  True when exp == now + 900
        assert is_mstock_token_expiring(token, within_seconds=900) is True

    def test_token_expiring_just_beyond_threshold_returns_false(self):
        from auth import is_mstock_token_expiring

        # Token expires in 901 seconds — beyond the 900-s window
        token = _make_jwt(_make_exp_payload(901))
        assert is_mstock_token_expiring(token, within_seconds=900) is False

    def test_custom_within_seconds_param(self):
        from auth import is_mstock_token_expiring

        # Token expires in 200 seconds
        token = _make_jwt(_make_exp_payload(200))
        assert is_mstock_token_expiring(token, within_seconds=300) is True
        assert is_mstock_token_expiring(token, within_seconds=100) is False

    def test_malformed_jwt_returns_false(self):
        from auth import is_mstock_token_expiring

        # Fail-open for non-critical paths
        assert is_mstock_token_expiring("malformed.jwt.token", within_seconds=900) is False

    def test_empty_token_returns_false(self):
        from auth import is_mstock_token_expiring

        assert is_mstock_token_expiring("", within_seconds=900) is False
        assert is_mstock_token_expiring(None, within_seconds=900) is False

    def test_missing_exp_returns_false(self):
        from auth import is_mstock_token_expiring

        token = _make_jwt({"sub": "user"})  # no exp claim
        assert is_mstock_token_expiring(token, within_seconds=900) is False


# ── Tests: safe_refresh_mstock_token ─────────────────────────────────────────

class TestSafeRefreshMStockToken:
    def test_returns_none_when_env_not_configured(self, monkeypatch):
        from auth import safe_refresh_mstock_token

        monkeypatch.delenv("MSTOCK_USERNAME", raising=False)
        monkeypatch.delenv("MSTOCK_PASSWORD", raising=False)
        monkeypatch.delenv("MSTOCK_API_KEY", raising=False)
        monkeypatch.delenv("MSTOCK_TOTP_SECRET", raising=False)

        result = safe_refresh_mstock_token()
        assert result is None

    @mock.patch("auth.login_with_totp", return_value="new_fake_token_abc123")
    def test_returns_new_token_on_success(self, mock_login):
        from auth import safe_refresh_mstock_token

        result = safe_refresh_mstock_token()
        assert result == "new_fake_token_abc123"
        mock_login.assert_called_once()

    @mock.patch("auth.login_with_totp", return_value=None)
    def test_returns_none_on_login_failure(self, mock_login):
        from auth import safe_refresh_mstock_token

        result = safe_refresh_mstock_token()
        assert result is None

    @mock.patch("auth.login_with_totp")
    def test_raises_and_returns_none_on_exception(self, mock_login):
        from auth import safe_refresh_mstock_token

        mock_login.side_effect = RuntimeError("TOTP rejected")

        result = safe_refresh_mstock_token()
        assert result is None


# ── Tests: No raw token in logs/prints ───────────────────────────────────────

class TestNoTokenInLogs:
    def test_is_mstock_token_expiring_does_not_print_token(self):
        from auth import is_mstock_token_expiring

        token = _make_jwt(_make_exp_payload(300))

        with mock.patch("builtins.print") as mock_print:
            is_mstock_token_expiring(token)
            # Should not have called print at all
            mock_print.assert_not_called()

    def test_safe_refresh_mstock_token_logs_no_raw_token(self, monkeypatch):
        from auth import safe_refresh_mstock_token

        # Patch login to avoid real API calls
        with mock.patch("auth.login_with_totp", return_value="super_secret_raw_token"):
            with mock.patch("builtins.print") as mock_print:
                safe_refresh_mstock_token()
                printed = "\n".join(str(c) for c in mock_print.call_args_list)

                # Raw token must NOT appear in any printed output
                assert "super_secret_raw_token" not in printed
                assert "FAKE_SIGNATURE" not in printed
                # exp_display is fine
                assert "Safe-refresh succeeded" in printed or "Safe-refresh skipped" in printed

    def test_safe_refresh_mstock_token_skip_message_has_no_token(self, monkeypatch):
        from auth import safe_refresh_mstock_token

        monkeypatch.delenv("MSTOCK_USERNAME", raising=False)
        with mock.patch("builtins.print") as mock_print:
            safe_refresh_mstock_token()
            printed = "\n".join(str(c) for c in mock_print.call_args_list)
            assert "MSTOCK_ACCESS_TOKEN" not in printed
            assert "FAKE_SIGNATURE" not in printed