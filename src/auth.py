from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path
from typing import Any, Dict

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
    data = payload.get("data") if isinstance(payload, dict) else None
    # The SDK's generate_session expects the 'refreshToken' from login as the request token.
    if isinstance(data, dict):
        if "refreshToken" in data:
            return str(data["refreshToken"])
        if "request_token" in data:
            return str(data["request_token"])
    
    if "refreshToken" in payload:
        return str(payload["refreshToken"])
    if "request_token" in payload:
        return str(payload["request_token"])
        
    raise RuntimeError(
        "Could not find refreshToken (or request_token) in login response. "
        "Inspect the JSON and adjust _extract_request_token."
    )


def _extract_access_token(payload: Dict[str, Any]) -> str:
    data = payload.get("data") if isinstance(payload, dict) else None
    # The SDK sets access_token from 'jwtToken' in the response data.
    if isinstance(data, dict):
        if "jwtToken" in data:
            return str(data["jwtToken"])
        if "access_token" in data:
            return str(data["access_token"])
            
    if "jwtToken" in payload:
        return str(payload["jwtToken"])
    if "access_token" in payload:
        return str(payload["access_token"])
        
    raise RuntimeError(
        "Could not find jwtToken (or access_token) in generate_session response. "
        "Inspect the JSON and adjust _extract_access_token."
    )


def request_sms_otp(username: str, password: str) -> tuple[str, str]:
    """Start login and trigger an SMS OTP.

    Returns (refresh_token, message).
    """
    client = MConnectB()

    login_resp = client.login(username, password)
    try:
        login_json = login_resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Login response is not valid JSON: {getattr(login_resp, 'text', '')}"
        ) from exc

    refresh_token = _extract_request_token(login_json)
    message = str(login_json.get("message") or "OTP requested") if isinstance(login_json, dict) else "OTP requested"
    return refresh_token, message


def verify_sms_otp(api_key: str, refresh_token: str, otp: str) -> str:
    """Complete login by verifying the SMS OTP and returning an access token."""
    if not api_key or "ENTER_YOUR_API_KEY_HERE" in api_key:
        raise RuntimeError(
            "API key is not configured. Set MSTOCK_API_KEY or update "
            "tradingapi_b.__config__.API_KEY with your key."
        )

    client = MConnectB()
    sess_resp = client.generate_session(api_key, refresh_token, otp)
    try:
        sess_json = sess_resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Session response is not valid JSON: {getattr(sess_resp, 'text', '')}"
        ) from exc

    return _extract_access_token(sess_json)


def generate_access_token(username: str, password: str, otp: str, api_key: str) -> str:
    """Non-interactive helper to generate an access token using SMS OTP."""
    refresh_token, _ = request_sms_otp(username, password)
    return verify_sms_otp(api_key, refresh_token, otp)


def generate_access_token_via_sms_otp() -> str:
    """Interactive helper to generate an access token using SMS OTP.

    Flow (using official pytradingapi-typeB endpoints):
    1. Call login(user_id, password) to receive a request token.
    2. Prompt for the SMS OTP you receive from m.Stock.
    3. Call generate_session(api_key, request_token, otp) to get access_token.

    The resulting access token is printed so you can set it as the
    MSTOCK_ACCESS_TOKEN environment variable before running the bot.
    """

    # Optional convenience: reuse credentials the UI saved locally.
    try:
        from config import load_saved_credentials

        saved = load_saved_credentials()
    except Exception:
        saved = {}

    username = (os.getenv("MSTOCK_USERNAME") or str(saved.get("username") or "")).strip() or input("Client code: ")
    # If MSTOCK_PASSWORD is set, use it; otherwise, prompt securely.
    password = os.getenv("MSTOCK_PASSWORD") or str(saved.get("password") or "") or getpass.getpass("Password: ")

    # Prefer environment override; then saved file; then SDK config constant.
    # Also allow vendored SDK config (pytradingapi-typeB-main) even if the SDK isn't installed.
    try:
        from config import get_sdk_default_api_key

        sdk_key = get_sdk_default_api_key()
    except Exception:
        sdk_key = ""

    api_key = (os.getenv("MSTOCK_API_KEY") or str(saved.get("api_key") or "") or sdk_key or __config__.API_KEY).strip()
    if not api_key or "ENTER_YOUR_API_KEY_HERE" in api_key:
        raise RuntimeError(
            "API key is not configured. Set MSTOCK_API_KEY or update "
            "tradingapi_b.__config__.API_KEY with your key."
        )

    print("\nRequesting SMS OTP via MConnectB.login()...")
    refresh_token, message = request_sms_otp(username, password)
    print(message)

    otp = input("Enter SMS OTP sent by m.Stock: ")

    print("\nVerifying OTP and generating session...")
    access_token = verify_sms_otp(api_key, refresh_token, otp)

    print("\nYour access token is:\n")
    print(access_token)
    print(
        "\nSet this in your shell before running the bot, e.g. "
        "(PowerShell):\n"
        "  $env:MSTOCK_ACCESS_TOKEN = '" + access_token + "'\n"
    )

    return access_token


def main() -> None:
    generate_access_token_via_sms_otp()


if __name__ == "__main__":
    main()
