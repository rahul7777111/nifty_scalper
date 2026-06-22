import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import verify_mstock_token_read_only


def test_verify_token_uses_profile_first():
    calls = []

    class Client:
        def get_profile(self):
            calls.append("profile")
            return {"user": "ok"}

        def get_ltp(self, symbol):
            calls.append("ltp")
            return 1

    result = verify_mstock_token_read_only(Client())

    assert result.ok is True
    assert result.status == "AUTH_OK"
    assert result.endpoint_used == "profile"
    assert calls == ["profile"]


def test_verify_token_falls_back_to_ltp():
    class Client:
        def get_profile(self):
            raise RuntimeError("profile unavailable")

        def get_ltp(self, symbol):
            return 23257.95

    result = verify_mstock_token_read_only(Client())

    assert result.ok is True
    assert result.status == "AUTH_OK"
    assert result.endpoint_used.startswith("get_ltp:")
    assert result.spot == 23257.95


def test_verify_token_classifies_expired():
    class Client:
        def get_ltp(self, symbol):
            raise RuntimeError("IA401 token expired")

    result = verify_mstock_token_read_only(Client())

    assert result.ok is False
    assert result.status == "SESSION_EXPIRED"
    assert "IA401" in result.error_message
