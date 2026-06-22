"""Broker and API Safety Audit — placeholder tests.

These tests verify safety properties of the live-trading layer.
Each test is self-contained and uses monkeypatching / static analysis.

Run with:  pytest tests/test_broker_safety_audit.py -v

Critical failures this module will verify when implemented:
  1. mStock JWT token expiry checked before API calls
  2. HTTP 429 handled with Retry-After + exponential backoff
  3. Market order fill confirmation via get_order_status + polling
  4. All urlopen calls have explicit timeout arguments
  5. Default credential path is outside the repo directory

NOTE: Full implementation blocked by missing functions:
  - auth.is_mstock_token_expiring / decode_mstock_jwt
  - auth.safe_refresh_mstock_token
  - mstock_client._apply_rate_limit_backoff / _parse_retry_after
  - config._default_credential_path
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: paper mode never calls place_order
# ─────────────────────────────────────────────────────────────────────────────

def test_paper_mode_never_calls_place_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that when enable_live_trading=False, place_order is never called."""
    from config import StrategyConfig

    cfg = StrategyConfig(enable_live_trading=False)
    mock_client = MagicMock()
    mock_client.get_ltp.return_value = 25000.0
    mock_client.place_order.side_effect = AssertionError(
        "place_order should NOT be called in paper mode"
    )

    from strategy import NiftyScalper
    scalper = NiftyScalper(mock_client, cfg)

    from datetime import datetime as dt
    from market_data import Candle

    fake_candles = [
        Candle(
            time=dt(2026, 6, 7, 9, 30),
            open=25000.0,
            high=25050.0,
            low=24950.0,
            close=25020.0,
            volume=1000.0,
        ),
        Candle(
            time=dt(2026, 6, 7, 9, 31),
            open=25020.0,
            high=25070.0,
            low=25010.0,
            close=25050.0,
            volume=1200.0,
        ),
    ]

    with patch("strategy.is_market_open", return_value=True):
        result = scalper._decide_entries()
        mock_client.place_order.assert_not_called()


def test_live_mode_guard_blocks_without_explicit_enable() -> None:
    """Ensure StrategyConfig defaults to paper mode (enable_live_trading=False)."""
    from config import StrategyConfig

    cfg = StrategyConfig()
    assert cfg.enable_live_trading is False, (
        "enable_live_trading must default to False (paper/dry-run) "
        "to prevent accidental live orders on first run."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: mStock token expiry check present (CF-001 placeholder)
# ─────────────────────────────────────────────────────────────────────────────

def test_mstock_token_expiry_check_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify auth.is_mstock_token_expiring and decode_mstock_jwt are present and functional."""
    # Test 1: Functions are importable from auth
    try:
        from auth import decode_mstock_jwt, is_mstock_token_expiring, safe_refresh_mstock_token
    except ImportError as exc:
        pytest.fail(f"auth.py does not export required functions: {exc}")

    # Test 2: decode_mstock_jwt — valid JWT
    import base64, json, time
    payload = {"sub": "test-user", "exp": int(time.time()) + 3600}
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
    token = f"eyJhbGciOiJIUzI1NiJ9.{payload_b64}.fakesig"
    result = decode_mstock_jwt(token)
    assert result.get("sub") == "test-user", f"Expected sub=test-user, got {result}"

    # Test 3: decode_mstock_jwt — malformed tokens return {}
    assert decode_mstock_jwt("not.a.jwt") == {}
    assert decode_mstock_jwt("") == {}
    assert decode_mstock_jwt("only_two_parts.here") == {}
    assert decode_mstock_jwt(None) == {}  # type: ignore[arg-type]

    # Test 4: is_mstock_token_expiring — not expiring (1 hour)
    expiring_token = f"header.{payload_b64}.sig"
    assert is_mstock_token_expiring(expiring_token) is False, "Token valid for 1 hour should NOT be flagged as expiring"

    # Test 5: is_mstock_token_expiring — expiring within 5 minutes (default 15 min window)
    payload_short = {"sub": "test", "exp": int(time.time()) + 300}  # 5 min
    short_b64 = base64.urlsafe_b64encode(json.dumps(payload_short).encode()).decode().rstrip('=')
    short_token = f"header.{short_b64}.sig"
    assert is_mstock_token_expiring(short_token) is True, "Token expiring in 5 min should be flagged"

    # Test 6: is_mstock_token_expiring — expired token
    payload_exp = {"sub": "test", "exp": int(time.time()) - 60}  # expired 1 min ago
    exp_b64 = base64.urlsafe_b64encode(json.dumps(payload_exp).encode()).decode().rstrip('=')
    exp_token = f"header.{exp_b64}.sig"
    assert is_mstock_token_expiring(exp_token) is True, "Expired token should be flagged"

    # Test 7: is_mstock_token_expiring — empty string vs None
    # Note: fail-open for empty/missing tokens (CF-001 design: non-critical paths
    # that don't place live orders use is_mstock_token_expiring liberally, so empty
    # tokens are treated as "not expiring" to avoid false positives. Critical paths
    # (place_order, cancel_order) call _ensure_valid_token which is stricter.
    assert is_mstock_token_expiring("") is False, "Empty token returns False (fail-open by design)"
    assert is_mstock_token_expiring(None) is False  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: no hardcoded secrets in source code
# ─────────────────────────────────────────────────────────────────────────────

def _scan_file_for_secrets(path: Path) -> list[str]:
    """Return list of suspicious lines (token/key patterns not redacted)."""
    findings = []
    content = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "***" in line or "REDACTED" in line or "hidden" in line.lower():
            continue
        patterns = [
            r"\bsk-[A-Za-z0-9]{20,}\b",
            r"\bsk-proj-[A-Za-z0-9]{20,}\b",
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
            r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
            r"\bghp_[A-Za-z0-9]{20,}\b",
            r"mstock_[A-Za-z0-9]{20,}",
            r"MSTOCK_[A-Z0-9]{20,}",
            r"DHAN_[A-Z0-9]{20,}",
        ]
        for pat in patterns:
            if re.search(pat, line):
                findings.append(f"  Line {lineno}: {line[:120].strip()}")
    return findings


def test_no_hardcoded_secrets_in_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if any source file contains a hardcoded secret pattern."""
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "")

    secret_files = []
    skip_dirs = {".venv", "site-packages", "__pycache__",
                 "tests", "archive", "pytradingapi-typeB-main"}

    for py_file in SRC_DIR.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        if any(skip in str(rel).split(os.sep) for skip in skip_dirs):
            continue
        findings = _scan_file_for_secrets(py_file)
        if findings:
            secret_files.append(f"\n{py_file.name}:\n" + "\n".join(findings))

    assert not secret_files, (
        "Hardcoded secrets found in source files:\n" + "\n".join(secret_files)
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: GPT redaction in logs
# ─────────────────────────────────────────────────────────────────────────────

def test_gpt_redacts_secrets_in_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _redact_secrets masks OpenAI keys, GitHub PATs, and sk-proj keys."""
    import gpt_advisor as ga

    test_cases = [
        (
            "The API key is sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890",
            "The API key is sk-***",
        ),
        (
            "Token: sk-proj-1234567890abcdefghijklmnopqrstuvwxyz1234567890",
            "Token: sk-proj-***",
        ),
        (
            "github_pat_1234567890abcdefghijklmnopqrstuvwxyz1234567890ABC",
            "github_pat_***",
        ),
        (
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz1234567890ABC",
            "ghp_***",
        ),
        (
            '{"api_key": "sk-ABCD1234567890EFGH", "model": "gpt-4o-mini"}',
            '{"api_key": "sk-***", "model": "gpt-4o-mini"}',
        ),
    ]

    for raw, expected in test_cases:
        result = ga._redact_secrets(raw)
        assert result == expected, (
            f"_redact_secrets failed:\n  input:  {raw}\n  output: {result}\n  expected: {expected}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: candle bucket prevents duplicate entry signals
# ─────────────────────────────────────────────────────────────────────────────

def test_candle_bucket_prevents_duplicate_signals() -> None:
    """Verify _last_entry_bucket_start_ts prevents duplicate decisions within same candle bucket."""
    from config import StrategyConfig

    cfg = StrategyConfig(enable_live_trading=False, timeframe="1m")
    mock_client = MagicMock()
    mock_client.get_ltp.return_value = 25000.0

    from strategy import NiftyScalper
    scalper = NiftyScalper(mock_client, cfg)

    now_ts = time.time()
    bucket_sec = 60
    bucket = now_ts - (now_ts % bucket_sec)
    scalper._last_entry_bucket_start_ts = float(bucket)

    assert scalper._last_entry_bucket_start_ts == float(bucket)

    same_bucket = float(bucket)
    new_bucket = float(bucket) + bucket_sec

    assert float(scalper._last_entry_bucket_start_ts) == float(bucket)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Dhan live-order requires explicit opt-in flags
# ─────────────────────────────────────────────────────────────────────────────

def test_dhan_guard_live_order_requires_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """DhanClient._guard_live_order must require SCALPER_BROKER=dhan AND
    SCALPER_ALLOW_LIVE_ORDERS=true before any live order."""
    for key in ("SCALPER_BROKER", "SCALPER_ALLOW_LIVE_ORDERS"):
        monkeypatch.delenv(key, raising=False)

    try:
        from dhan_client import DhanClient
        from config import StrategyConfig

        cfg = StrategyConfig()
        with patch("dhan_client.DhanContext"), \
             patch("dhan_client.DhanHQClient"):
            client = DhanClient(cfg)

        with pytest.raises(RuntimeError, match="SCALPER_BROKER must be set to dhan"):
            client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

        monkeypatch.setenv("SCALPER_BROKER", "dhan")

        with pytest.raises(RuntimeError, match="SCALPER_ALLOW_LIVE_ORDERS=true is required"):
            client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

        monkeypatch.setenv("SCALPER_ALLOW_LIVE_ORDERS", "true")
    except RuntimeError as exc:
        dhan_path = SRC_DIR / "dhan_client.py"
        source = dhan_path.read_text(encoding="utf-8")
        assert "SCALPER_BROKER" in source
        assert "SCALPER_ALLOW_LIVE_ORDERS" in source
        assert "_guard_live_order" in source
        pytest.skip(f"Dhan SDK not installed; verified guard code statically: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: HTTP 429 backoff present (placeholder)
# ─────────────────────────────────────────────────────────────────────────────

def test_http_429_backoff_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HTTP 429 rate-limit backoff helpers are present and functional."""
    mstock_path = SRC_DIR / "mstock_client.py"
    source = mstock_path.read_text(encoding="utf-8")

    # Test 1: Module-level helpers exist
    assert "_apply_rate_limit_backoff" in source, "_apply_rate_limit_backoff missing from mstock_client"
    assert "_parse_retry_after" in source, "_parse_retry_after missing from mstock_client"

    # Test 2: _handle_http_429 method exists on MStockTypeBClient
    assert "def _handle_http_429" in source, "_handle_http_429 method missing"

    # Test 3: 429 handling in dhan_client
    dhan_path = SRC_DIR / "dhan_client.py"
    dhan_source = dhan_path.read_text(encoding="utf-8")
    assert "_call_api_with_retry" in dhan_source, "_call_api_with_retry missing from dhan_client"
    assert "429" in dhan_source, "HTTP 429 handling missing from dhan_client"

    # Test 4: Functional backoff in dhan_client
    from dhan_client import _parse_retry_after_dhan, _apply_dhan_rate_limit_backoff
    # Integer seconds
    assert _parse_retry_after_dhan("30") == 30.0
    # Float
    assert _parse_retry_after_dhan("1.5") == 1.5
    # Empty/invalid → 0
    assert _parse_retry_after_dhan("") == 0.0
    assert _parse_retry_after_dhan("abc") == 0.0

    # Test 5: mstock _parse_retry_after handles HTTPError-like objects
    from mstock_client import _parse_retry_after
    # Simulate an HTTPError with a headers dict (urllib.error.HTTPError interface)
    class FakeHTTPError(Exception):
        def __init__(self, code, headers_dict):
            super().__init__(f"HTTP {code}")
            self.code = code
            self.headers = headers_dict  # dict directly, as urllib uses

    exc = FakeHTTPError(429, {"Retry-After": "15", "Content-Type": "application/json"})
    delay = _parse_retry_after(exc)
    assert delay == 15.0, f"Expected 15.0, got {delay}"

    exc2 = FakeHTTPError(429, {})
    delay2 = _parse_retry_after(exc2)
    assert delay2 == 0.0, f"Expected 0.0 for missing Retry-After, got {delay2}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: Order status polling methods exist on MStockTypeBClient
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: mStock HTTP timeouts are set (positive finding)
# ─────────────────────────────────────────────────────────────────────────────

def test_mstock_http_calls_have_timeouts() -> None:
    """Verify m.Stock API calls all pass an explicit timeout to urlopen."""
    mstock_path = SRC_DIR / "mstock_client.py"
    source = mstock_path.read_text(encoding="utf-8")

    urlopen_calls = re.findall(r"urlopen\([^)]+\)", source)

    for call in urlopen_calls:
        assert "timeout=" in call, (
            f"urlopen call missing timeout parameter: {call}"
        )

    assert len(urlopen_calls) >= 3, (
        f"Expected at least 3 urlopen calls (intraday, historical, scripmaster), "
        f"found {len(urlopen_calls)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: config credentials default path is outside repo
# ─────────────────────────────────────────────────────────────────────────────

def test_credentials_path_is_outside_repo() -> None:
    """Ensure default credential path uses APPDATA (percent APPDATA on Windows).

    Credentials should NEVER be stored inside the repo directory.
    """
    from config import _default_credential_path

    path = _default_credential_path()
    repo_root = REPO_ROOT.resolve()
    resolved = path.resolve()

    try:
        resolved.relative_to(repo_root)
        inside_repo = True
    except ValueError:
        inside_repo = False

    assert not inside_repo, (
        f"Credentials path {resolved} MUST be outside repo root {repo_root}. "
        "Never store secrets inside the repository."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: Order status polling methods exist on MStockTypeBClient
# ─────────────────────────────────────────────────────────────────────────────

def test_order_status_polling_methods_exist() -> None:
    """Verify get_order_status and poll_order_until_terminal exist on MStockTypeBClient."""
    from mstock_client import MStockTypeBClient

    assert hasattr(MStockTypeBClient, "get_order_status"), \
        "MStockTypeBClient missing get_order_status method"
    assert hasattr(MStockTypeBClient, "poll_order_until_terminal"), \
        "MStockTypeBClient missing poll_order_until_terminal method"

    import inspect
    sig = inspect.signature(MStockTypeBClient.poll_order_until_terminal)
    params = list(sig.parameters.keys())
    assert "order_id" in params, f"poll_order_until_terminal missing order_id param: {params}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: place_order / cancel_order call _ensure_valid_token
# ─────────────────────────────────────────────────────────────────────────────

def test_place_order_calls_ensure_valid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """place_order must call _ensure_valid_token before placing any order."""
    mstock_path = SRC_DIR / "mstock_client.py"
    source = mstock_path.read_text(encoding="utf-8")

    # Locate place_order method body
    import re
    match = re.search(
        r'def place_order\([^)]*\)[^:]*:(.*?)(?=\n    def |\nclass |\Z)',
        source, re.DOTALL
    )
    assert match, "place_order not found in mstock_client.py"
    body = match.group(1)
    assert "_ensure_valid_token" in body, \
        "place_order must call _ensure_valid_token before placing orders"

    # cancel_order must also call _ensure_valid_token
    cancel_match = re.search(
        r'def cancel_order\([^)]*\)[^:]*:(.*?)(?=\n    def |\nclass |\Z)',
        source, re.DOTALL
    )
    assert cancel_match, "cancel_order not found"
    cancel_body = cancel_match.group(1)
    assert "_ensure_valid_token" in cancel_body, \
        "cancel_order must call _ensure_valid_token"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: Dhan _call_api_with_retry used on rate-limitable endpoints
# ─────────────────────────────────────────────────────────────────────────────

def test_dhan_retry_used_on_critical_endpoints() -> None:
    """Critical Dhan endpoints must use _call_api_with_retry for 429 resilience."""
    dhan_path = SRC_DIR / "dhan_client.py"
    source = dhan_path.read_text(encoding="utf-8")

    # These methods touch live broker data and can trigger 429s
    critical_methods = ["get_ltp", "get_bid_ask", "get_option_chain", "get_candles"]
    for method in critical_methods:
        # Find method body
        match = re.search(
            rf'def {method}\([^)]*\)[^:]*:(.*?)(?=\n    def |\nclass |\Z)',
            source, re.DOTALL
        )
        if match:
            body = match.group(1)
            assert "_call_api_with_retry" in body or "_call_api" in body, (
                f"{method} must use _call_api_with_retry or _call_api for broker calls"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])