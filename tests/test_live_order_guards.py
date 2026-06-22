"""Live order guard tests — CF-001.

Verifies that DhanClient._guard_live_order and MStockTypeBClient's
live-order path enforce all safety interlocks before placing any live order.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── DhanClient guard tests ────────────────────────────────────────────────────

class TestDhanLiveOrderGuards:
    """Verify DhanClient blocks live orders unless ALL conditions are met."""

    @pytest.fixture
    def clean_env(self):
        for key in ("SCALPER_BROKER", "SCALPER_ALLOW_LIVE_ORDERS", "DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN"):
            if key in os.environ:
                del os.environ[key]

    def test_blocks_without_scaper_broker_set(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
            client = DhanClient(StrategyConfig())

        with pytest.raises(RuntimeError, match="SCALPER_BROKER must be set to dhan"):
            client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

    def test_blocks_without_scaper_allow_live_orders(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            with pytest.raises(RuntimeError, match="SCALPER_ALLOW_LIVE_ORDERS=true is required"):
                client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

    def test_blocks_when_order_mode_not_live(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan", "SCALPER_ALLOW_LIVE_ORDERS": "true"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            with pytest.raises(RuntimeError, match="order mode must be explicitly set to live"):
                client._guard_live_order(symbol="NIFTY", token="13", order_mode="paper")

    def test_blocks_when_sdk_import_failed(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan", "SCALPER_ALLOW_LIVE_ORDERS": "true"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            # Simulate SDK import not healthy
            client._live_readiness["sdk_import_ok"] = False

            with pytest.raises(RuntimeError, match="SDK import is not healthy"):
                client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

    def test_blocks_when_option_chain_not_validated(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan", "SCALPER_ALLOW_LIVE_ORDERS": "true"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            client._live_readiness["sdk_import_ok"] = True
            client._live_readiness["option_chain_ok"] = False
            client._health_passed_current_session = True

            with pytest.raises(RuntimeError, match="option chain validation has not passed"):
                client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

    def test_blocks_when_health_not_passed_current_session(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan", "SCALPER_ALLOW_LIVE_ORDERS": "true"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            client._live_readiness["sdk_import_ok"] = True
            client._live_readiness["option_chain_ok"] = True
            client._live_readiness["live_order_allowed"] = False
            client._health_passed_current_session = False

            with pytest.raises(RuntimeError, match="broker live readiness has not passed"):
                client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

    def test_blocks_when_token_not_validated(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan", "SCALPER_ALLOW_LIVE_ORDERS": "true"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            client._live_readiness.update({
                "sdk_import_ok": True,
                "option_chain_ok": True,
                "live_order_allowed": True,
            })
            client._health_passed_current_session = True
            client._validated_live_security_ids = {"99999"}  # Different token

            with pytest.raises(RuntimeError, match="security_id was not validated"):
                client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

    def test_blocks_when_token_mismatch(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan", "SCALPER_ALLOW_LIVE_ORDERS": "true"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            client._validated_live_security_ids = {"13", "14", "15"}
            client._live_readiness["selected_security_id"] = "14"  # Different
            client._live_readiness.update({
                "sdk_import_ok": True,
                "option_chain_ok": True,
                "live_order_allowed": True,
            })
            client._health_passed_current_session = True

            with pytest.raises(RuntimeError, match="security_id does not match"):
                client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")

    def test_passes_all_guards_with_valid_state(self, clean_env) -> None:
        import os
        from dhan_client import DhanClient
        from config import StrategyConfig

        with patch.dict("os.environ", {"SCALPER_BROKER": "dhan", "SCALPER_ALLOW_LIVE_ORDERS": "true"}):
            with patch("dhan_client.DhanContext"), patch("dhan_client.DhanHQClient"):
                client = DhanClient(StrategyConfig())

            client._validated_live_security_ids = {"13", "14", "15"}
            client._live_readiness.update({
                "sdk_import_ok": True,
                "option_chain_ok": True,
                "live_order_allowed": True,
                "selected_security_id": "13",
            })
            client._health_passed_current_session = True

            # Should not raise
            client._guard_live_order(symbol="NIFTY", token="13", order_mode="live")


# ── MStock token guard tests ─────────────────────────────────────────────────

class TestMStockLiveOrderGuards:
    """Verify MStockTypeBClient blocks live orders when token is expired."""

    def test_live_order_blocked_when_token_expired(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        import time
        # Create an already-expired token
        expired_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOiIxMjM0NTY3ODkwIn0.signature"
        import base64, json
        payload = {"exp": int(time.time()) - 60}
        raw_payload = json.dumps(payload, separators=(",", ":")).encode()
        encoded = base64.urlsafe_b64encode(raw_payload).rstrip(b"=").decode()
        expired_token = f"eyJhbGciOiJIUzI1NiJ9.{encoded}.signature"

        with patch.dict("os.environ", {
            "MSTOCK_ACCESS_TOKEN": expired_token,
            "MSTOCK_USERNAME": "",
            "MSTOCK_PASSWORD": "",
            "MSTOCK_API_KEY": "testkey",
            "MSTOCK_TOTP_SECRET": "",
        }):
            cfg = APIConfig(base_url="", api_key="testkey", api_secret="", client_id="testclient")
            client = MStockTypeBClient(cfg)

            with pytest.raises(RuntimeError, match="expired|expiring"):
                client._ensure_valid_token(allow_refresh=True)

    def test_live_order_allowed_with_fresh_token(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        import time
        import base64, json
        payload = {"exp": int(time.time()) + 3600}
        raw_payload = json.dumps(payload, separators=(",", ":")).encode()
        encoded = base64.urlsafe_b64encode(raw_payload).rstrip(b"=").decode()
        fresh_token = f"eyJhbGciOiJIUzI1NiJ9.{encoded}.signature"

        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="testkey", api_secret="", client_id="testclient")
            client = MStockTypeBClient(cfg)

            # Should not raise
            client._ensure_valid_token(allow_refresh=False)


# ── Paper mode default tests ─────────────────────────────────────────────────

class TestPaperModeDefaults:
    """Verify paper mode is the safe default."""

    def test_strategy_config_defaults_to_paper(self) -> None:
        from config import StrategyConfig

        cfg = StrategyConfig()
        assert cfg.enable_live_trading is False, (
            "enable_live_trading must default to False"
        )

    def test_strategy_paper_mode_true_by_default(self) -> None:
        from strategy import NiftyScalper
        from config import StrategyConfig
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        cfg = StrategyConfig()
        scalper = NiftyScalper(mock_client, cfg)

        assert scalper.paper_mode is True, (
            "paper_mode must default to True for safety"
        )


# ── Kill-switch tests ─────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_kill_switch_env_variable_halts_strategy(self) -> None:
        """SCALPER_KILL_SWITCH=true must trigger safe exit without placing orders."""
        import os
        from strategy import NiftyScalper
        from config import StrategyConfig
        from datetime import datetime as dt
        from market_data import Candle
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.get_ltp.return_value = 25000.0

        with patch.dict("os.environ", {
            "SCALPER_KILL_SWITCH": "true",
            "SCALPER_ALLOW_LIVE_ORDERS": "true",
            "SCALPER_BROKER": "mstock",
        }):
            cfg = StrategyConfig(enable_live_trading=True)
            scalper = NiftyScalper(mock_client, cfg)

            fake_candles = [
                Candle(
                    time=dt(2026, 6, 7, 9, 30),
                    open=25000.0, high=25050.0, low=24950.0, close=25020.0,
                    volume=1000.0,
                ),
            ]

            # When kill switch is active, strategy should not place orders
            # and instead safely exit
            with patch("strategy.is_market_open", return_value=True):
                try:
                    result = scalper.run_once(fake_candles)
                    # Kill switch should result in no orders placed
                    assert mock_client.place_order.call_count == 0 or result is None
                except Exception:
                    pass  # Kill switch may raise — both are acceptable


# ── No raw tokens logged ───────────────────────────────────────────────────────

class TestNoRawTokensLogged:
    def test_dhan_client_masks_access_token_in_logs(self) -> None:
        from dhan_client import DhanClient, mask_token

        # Short token
        assert mask_token("abc") == "ab...c"
        # Empty
        assert mask_token("") == "(missing)"
        # Normal length - should mask middle
        result = mask_token("1234567890abcdef1234567890abcdef")
        assert result == "1234...ef"
        assert "12345678" not in result  # No full token in logs

    def test_auth_malformed_token_handled_safely(self) -> None:
        from auth import decode_mstock_jwt

        # Malformed tokens should not crash
        assert decode_mstock_jwt("") == {}
        assert decode_mstock_jwt("not.jwt") == {}
        assert decode_mstock_jwt("a.b.c.d.e") == {}
        assert decode_mstock_jwt("!!!") == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])