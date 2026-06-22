from __future__ import annotations

import base64
import getpass
import json
import os
import sys
import time
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pyotp
except ImportError:
    pyotp = None

try:
    from tradingapi_b.mconnect import MConnectB
    from tradingapi_b import __config__
except ImportError:  # pragma: no cover
    _repo_root = Path(__file__).resolve().parent.parent
    _sdk_root = _repo_root / "pytradingapi-typeB-main"
    if _sdk_root.exists():
        sys.path.insert(0, str(_sdk_root))
    from tradingapi_b.mconnect import MConnectB
    from tradingapi_b import __config__


def _extract_request_token(payload: Dict[str, Any]) -> str:
    """Extract request/refresh token from the login API response."""
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"login response is not a dict: {type(payload)}. Response: {payload}"
        )
    
    data = payload.get("data")
    
    # The SDK's generate_session expects the 'refreshToken' from login as the request token.
    if isinstance(data, dict):
        for key in ["refreshToken", "request_token", "requestToken", "token", "refresh_token"]:
            if key in data and data[key]:
                return str(data[key])
    
    for key in ["refreshToken", "request_token", "requestToken", "token", "refresh_token"]:
        if key in payload and payload[key]:
            return str(payload[key])
    
    if payload.get("status") == "false" or payload.get("errorcode"):
        import json
        error_msg = payload.get("message", "Unknown error")
        error_code = payload.get("errorcode", "Unknown")
        raise RuntimeError(
            f"API returned an error during login: {error_msg} (errorcode: {error_code}). Full response: {json.dumps(payload)}"
        )
        
    raise RuntimeError(
        "Could not find refreshToken (or request_token) in login response. "
        f"Response keys: {list(payload.keys())}. "
        "Inspect the JSON and adjust _extract_request_token."
    )


def _extract_access_token(payload: Dict[str, Any]) -> str:
    """Extract access token from the generate_session/verify_totp API response."""
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"generate_session response is not a dict: {type(payload)}. Response: {payload}"
        )
    
    data = payload.get("data")
    
    if isinstance(data, dict):
        for key in ["jwtToken", "access_token", "accessToken", "token"]:
            if key in data and data[key]:
                return str(data[key])
            
    for key in ["jwtToken", "access_token", "accessToken", "token"]:
        if key in payload and payload[key]:
            return str(payload[key])
    
    if payload.get("status") == "false" or payload.get("errorcode"):
        import json
        error_msg = payload.get("message", "Unknown error")
        error_code = payload.get("errorcode", "Unknown")
        raise RuntimeError(
            f"API returned an error: {error_msg} (errorcode: {error_code}). Full response: {json.dumps(payload)}"
        )
        
    raise RuntimeError(
        "Could not find jwtToken (or access_token) in generate_session response. "
        f"Response keys: {list(payload.keys())}. "
        "Inspect the JSON and adjust _extract_access_token."
    )


def generate_totp_secret() -> str:
    """Generate a new TOTP secret for m.Stock."""
    if pyotp is None:
        raise RuntimeError(
            "pyotp is not installed. Install it with: pip install pyotp"
        )
    return pyotp.random_base32()


def get_totp_from_secret(secret: str) -> str:
    """Generate current TOTP code from secret after robust cleaning and URI query parsing."""
    if pyotp is None:
        raise RuntimeError(
            "pyotp is not installed. Install it with: pip install pyotp"
        )
    
    secret_str = str(secret or "").strip()
    
    # If the user pasted an otpauth:// URI or query string, extract the secret key
    if "secret=" in secret_str:
        try:
            parsed = urllib.parse.urlparse(secret_str)
            qs = urllib.parse.parse_qs(parsed.query)
            if "secret" in qs:
                secret_str = qs["secret"][0]
        except Exception:
            # Fallback to simple split
            try:
                secret_str = secret_str.split("secret=")[1].split("&")[0]
            except Exception:
                pass

    # Clean the secret: strip whitespace, remove spaces/hyphens, and convert to uppercase
    cleaned = secret_str.replace(" ", "").replace("-", "").upper()
    
    # Ensure it's valid base32 character set (A-Z, 2-7). Remove any other invalid characters
    cleaned = re.sub(r'[^A-Z2-7]', '', cleaned)
    
    if not cleaned:
        raise ValueError("TOTP secret contains no valid Base32 characters.")
        
    totp = pyotp.TOTP(cleaned)
    return totp.now()


def login_with_totp(username: str, password: str, api_key: str, totp_secret: str = None, totp_code: str = None) -> str:
    """Login and get access token using TOTP.
    
    Supports either manual entry of a 6-digit code or auto-generation using a TOTP secret.
    """
    # Step 1: Login to get refreshToken
    client = MConnectB()
    login_resp = client.login(username, password)
    
    try:
        login_json = login_resp.json()
    except ValueError as exc:
        raise RuntimeError(
            "Login response is not valid JSON: " + str(getattr(login_resp, 'text', ''))
        ) from exc
    
    # Check for error early
    if isinstance(login_json, dict) and (login_json.get("status") == "false" or login_json.get("errorcode")):
        import json
        error_msg = login_json.get("message", "Unknown error")
        error_code = login_json.get("errorcode", "Unknown")
        raise RuntimeError(
            f"API returned an error during login: {error_msg} (errorcode: {error_code}). Full response: {json.dumps(login_json)}"
        )

    # Clean totp_code or totp_secret input if it represents a direct 6-digit OTP
    direct_code = ""
    if totp_code and str(totp_code).strip().isdigit() and len(str(totp_code).strip()) == 6:
        direct_code = str(totp_code).strip()
    elif totp_secret and str(totp_secret).strip().isdigit() and len(str(totp_secret).strip()) == 6:
        direct_code = str(totp_secret).strip()

    if not direct_code and not totp_secret:
        totp_secret = os.getenv("MSTOCK_TOTP_SECRET", "").strip()
        if not totp_secret:
            raise RuntimeError(
                "Login succeeded, but neither a 6-digit TOTP code nor a TOTP secret was provided. "
                "Configure MSTOCK_TOTP_SECRET in .env or enter the code in the UI."
            )
    
    # Extract refresh token for TOTP verification
    refresh_token = _extract_request_token(login_json)
    
    # Generate/resolve TOTP code
    code_to_verify = direct_code if direct_code else get_totp_from_secret(totp_secret)
    
    # Verify TOTP and get access token
    client = MConnectB(api_key=api_key)
    verify_resp = client.verify_totp(api_key, refresh_token, code_to_verify)
    
    try:
        verify_json = verify_resp.json()
    except ValueError as exc:
        raise RuntimeError(
            "TOTP verification response is not valid JSON: " + str(getattr(verify_resp, 'text', ''))
        ) from exc
    
    # Extract access token from response
    if isinstance(verify_json, dict):
        # Check if the API returned an error first
        if verify_json.get("status") is False or verify_json.get("errorcode") == "MA400":
            raise RuntimeError(str(verify_json.get("message") or "TOTP verification rejected by broker."))
            
        try:
            return _extract_access_token(verify_json)
        except Exception:
            pass
    
    # Fallback: use the token captured by the SDK
    if getattr(client, "access_token", None):
        return str(client.access_token)
    
    raise RuntimeError(
        "Could not extract access token from TOTP verification response."
    )


def generate_access_token_via_totp() -> str:
    """Interactive helper to generate an access token using TOTP."""
    try:
        from config import load_saved_credentials
        saved = load_saved_credentials()
    except Exception:
        saved = {}

    username = (os.getenv("MSTOCK_USERNAME") or str(saved.get("username") or "")).strip() or input("Client code: ")
    password = os.getenv("MSTOCK_PASSWORD") or str(saved.get("password") or "") or getpass.getpass("Password: ")

    try:
        from config import get_sdk_default_api_key
        sdk_key = get_sdk_default_api_key()
    except Exception:
        sdk_key = ""

    api_key = (os.getenv("MSTOCK_API_KEY") or str(saved.get("api_key") or "") or sdk_key or __config__.API_KEY).strip()
    if not api_key or "ENTER_YOUR_API_KEY_HERE" in api_key:
        raise RuntimeError("API key is not configured. Set MSTOCK_API_KEY.")

    totp_secret = (os.getenv("MSTOCK_TOTP_SECRET") or str(saved.get("totp_secret") or "")).strip()
    if not totp_secret:
        totp_secret = getpass.getpass("TOTP secret (will not echo): ")

    print("\nLogging in and generating access token via TOTP...")
    access_token = login_with_totp(username, password, api_key, totp_secret)

    print("\nYour access token is:\n")
    print(access_token)
    print(
        "\nSet this in your shell before running the bot, e.g. (PowerShell):\n"
        "  $env:MSTOCK_ACCESS_TOKEN = '" + access_token + "'\n"
    )

    return access_token


# ─────────────────────────────────────────────────────────────────────────────
# m.Stock JWT token expiry helpers  (mirrors dhan_auth.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

def decode_mstock_jwt(token: str) -> Dict[str, Any]:
    """Decode a m.Stock JWT payload (base64url) without verifying the signature.

    Returns the decoded payload dict, or an empty dict on any decode error.
    Safe to call with malformed/empty tokens — never raises.
    """
    raw = str(token or "").strip()
    if not raw:
        return {}
    try:
        parts = raw.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        # PKCS7 padding for base64 decoder
        payload_b64 += "=" * (-len(payload_b64) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        parsed = json.loads(decoded.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def mstock_token_expiry_epoch(token: str) -> Optional[int]:
    """Return the UTC epoch (seconds) of the JWT exp claim, or None if absent."""
    payload = decode_mstock_jwt(token)
    try:
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


def is_mstock_token_expiring(token: str, *, within_seconds: int = 900) -> bool:
    """Return True when the m.Stock JWT expires within `within_seconds` (default 15 min).

    Returns False for empty/malformed tokens (fail-open for non-critical paths).
    Critical paths MUST call _ensure_valid_token which enforces stricter checks.
    """
    exp = mstock_token_expiry_epoch(token)
    if exp is None:
        return False
    try:
        return exp <= int(time.time()) + int(within_seconds)
    except Exception:
        return False


def safe_refresh_mstock_token() -> Optional[str]:
    """Attempt safe TOTP-based token refresh via the SDK.

    Returns a new access token string on success, or None on any failure.
    Logs warnings so operators can debug failures.

    Does NOT raise — callers must handle None and fail closed.
    """
    try:
        username = str(os.getenv("MSTOCK_USERNAME", "")).strip()
        password = os.getenv("MSTOCK_PASSWORD", "")
        api_key = str(
            os.getenv("MSTOCK_API_KEY", "")
            or getattr(__config__, "API_KEY", "")
            or ""
        ).strip()
        totp_secret = str(os.getenv("MSTOCK_TOTP_SECRET", "")).strip()

        if not (username and password and api_key and totp_secret):
            print(
                "[TOKEN] Safe-refresh skipped: MSTOCK_USERNAME/PASSWORD/API_KEY/TOTP_SECRET "
                "not all configured."
            )
            return None

        # reuse existing login helpers; login_with_totp updates env + returns token
        new_token = login_with_totp(username, password, api_key, totp_secret)
        if new_token and str(new_token).strip():
            print(f"[TOKEN] Safe-refresh succeeded, new token acquired.")
            return str(new_token).strip()
        return None
    except Exception as exc:
        print(f"[TOKEN] Safe-refresh failed: {exc}")
        return None


def main() -> None:
    generate_access_token_via_totp()


if __name__ == "__main__":
    main()
