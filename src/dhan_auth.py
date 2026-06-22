from __future__ import annotations

import os
import sys
import json
import base64
import time
from pathlib import Path
from typing import Any, Dict


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DHAN_SDK = _REPO_ROOT / "DhanHQ-py-main" / "DhanHQ-py-main" / "src"
if _LOCAL_DHAN_SDK.exists() and str(_LOCAL_DHAN_SDK) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DHAN_SDK))

try:
    from dhanhq.auth import DhanLogin
except Exception:  # pragma: no cover
    DhanLogin = None  # type: ignore[assignment]

try:
    from auth import get_totp_from_secret
except Exception:  # pragma: no cover
    get_totp_from_secret = None  # type: ignore[assignment]


def _extract_dhan_access_token(payload: Dict[str, Any]) -> str:
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("accessToken", "access_token", "token", "jwtToken"):
            value = str(data.get(key) or "").strip()
            if value:
                return value
    if isinstance(data, str):
        value = str(data).strip()
        if value.count(".") >= 2:
            return value
    message = payload.get("message")
    if isinstance(message, dict):
        for key in ("accessToken", "access_token", "token", "jwtToken"):
            value = str(message.get(key) or "").strip()
            if value:
                return value
    if isinstance(message, str):
        value = str(message).strip()
        if value.count(".") >= 2:
            return value
    for key in ("accessToken", "access_token", "token", "jwtToken"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    status = str(payload.get("status") or "").strip().lower()
    if status and status not in {"success", "true", "ok"}:
        raise RuntimeError(str(message or payload))
    raise RuntimeError(f"Could not extract Dhan access token from response: {payload}")


def decode_dhan_jwt(access_token: str) -> Dict[str, Any]:
    token = str(access_token or "").strip()
    if not token:
        return {}
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def dhan_token_expiry_epoch(access_token: str) -> int | None:
    payload = decode_dhan_jwt(access_token)
    try:
        value = payload.get("exp")
        return int(value) if value is not None else None
    except Exception:
        return None


def is_dhan_token_expiring(access_token: str, *, within_seconds: int = 900) -> bool:
    exp = dhan_token_expiry_epoch(access_token)
    if exp is None:
        return False
    try:
        return exp <= int(time.time()) + int(within_seconds)
    except Exception:
        return False


def dhan_token_generated_at_epoch(value: str = "") -> int | None:
    raw = str(value or os.getenv("DHAN_ACCESS_TOKEN_GENERATED_AT", "") or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except Exception:
        return None


def is_dhan_token_refresh_due(
    access_token: str,
    *,
    within_seconds: int = 900,
    max_age_seconds: int = 24 * 60 * 60,
    generated_at: str = "",
) -> bool:
    token = str(access_token or "").strip()
    if not token:
        return True
    if is_dhan_token_expiring(token, within_seconds=within_seconds):
        return True
    issued_at = dhan_token_generated_at_epoch(generated_at)
    if issued_at is None:
        if dhan_token_expiry_epoch(token) is not None:
            return False
        return False
    try:
        return issued_at + int(max_age_seconds) <= int(time.time()) + int(within_seconds)
    except Exception:
        return False


def persist_dhan_access_token(access_token: str) -> int:
    token = str(access_token or "").strip()
    generated_at = int(time.time())
    os.environ["DHAN_ACCESS_TOKEN"] = token
    os.environ["DHAN_ACCESS_TOKEN_GENERATED_AT"] = str(generated_at)
    try:
        from config import persist_settings_env

        persist_settings_env(
            {
                "DHAN_ACCESS_TOKEN": token,
                "DHAN_ACCESS_TOKEN_GENERATED_AT": str(generated_at),
            },
            [],
        )
    except Exception:
        pass
    return generated_at


def _fresh_totp_from_secret(secret: str) -> str:
    if get_totp_from_secret is None:
        raise RuntimeError("TOTP helper is unavailable.")
    # Avoid sending a code that is about to roll over while the HTTP request is in flight.
    try:
        remaining = 30 - (int(time.time()) % 30)
        if remaining <= 3:
            time.sleep(remaining + 1)
    except Exception:
        pass
    return str(get_totp_from_secret(secret) or "").strip()


def _is_invalid_totp_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return "invalid totp" in msg or "totp" in msg and ("invalid" in msg or "reject" in msg)


def normalize_dhan_pin(pin: str) -> str:
    raw = str(pin or "").strip()
    normalized = raw.replace(" ", "").replace("-", "")
    if not normalized:
        raise RuntimeError("Dhan PIN is required.")
    if not normalized.isdigit():
        raise RuntimeError("Dhan PIN must contain digits only. Enter your Dhan app PIN, not the API secret or password.")
    if len(normalized) < 4 or len(normalized) > 8:
        raise RuntimeError("Dhan PIN length looks wrong. Enter your Dhan app PIN, not the API secret or password.")
    return normalized


def _is_invalid_pin_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return "invalid pin" in msg or ("pin" in msg and ("invalid" in msg or "incorrect" in msg or "wrong" in msg))


def generate_dhan_access_token(
    client_id: str,
    pin: str,
    *,
    totp_secret: str = "",
    totp_code: str = "",
) -> str:
    if DhanLogin is None:
        raise RuntimeError("Dhan SDK auth helper is unavailable.")
    client_id = str(client_id or "").strip()
    pin = normalize_dhan_pin(pin)
    direct_code = str(totp_code or "").strip()
    secret = str(totp_secret or "").strip() or str(os.getenv("DHAN_TOTP_SECRET", "")).strip()
    if not client_id:
        raise RuntimeError("Dhan Client ID is required.")
    if not direct_code:
        if not secret:
            raise RuntimeError("Provide either a Dhan TOTP code or Dhan TOTP secret.")
        direct_code = _fresh_totp_from_secret(secret)
    if not (direct_code.isdigit() and len(direct_code) == 6):
        raise RuntimeError("Dhan TOTP must be a 6-digit code.")

    login = DhanLogin(client_id)
    try:
        payload = login.generate_token(pin, direct_code)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Dhan token response type: {type(payload)}")
        return _extract_dhan_access_token(payload)
    except Exception as exc:
        if _is_invalid_pin_error(exc):
            raise RuntimeError(
                "Dhan rejected the PIN. Enter the Dhan app PIN/MPIN in the Dhan PIN field; "
                "do not use API Secret, trading password, or TOTP here."
            ) from exc
        if not secret or not _is_invalid_totp_error(exc):
            raise
        retry_code = _fresh_totp_from_secret(secret)
        if retry_code == direct_code:
            raise
        payload = login.generate_token(pin, retry_code)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Dhan token response type: {type(payload)}")
        return _extract_dhan_access_token(payload)
