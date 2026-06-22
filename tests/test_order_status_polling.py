"""Order status polling tests — CF-001.

Verifies that MStockTypeBClient correctly polls get_order_status
until terminal state, handles PARTIALLY_FILLED correctly, and respects
the timeout on rejected/partial orders.
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


# ── get_order_status tests ────────────────────────────────────────────────────

class TestGetOrderStatus:
    def test_returns_order_from_order_book(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            mock_raw = MagicMock()
            mock_raw.access_token = fresh_token
            mock_raw.order_book.return_value = {
                "data": [
                    {
                        "order_id": "ORD001",
                        "status": "FILLED",
                        "filled_quantity": 65,
                        "remaining_quantity": 0,
                        "message": "Order filled",
                    }
                ]
            }
            client._raw = mock_raw

            result = client.get_order_status("ORD001")
            assert result["order_id"] == "ORD001"
            assert result["status"] == "FILLED"
            assert result["filled_qty"] == 65
            assert result["remaining_qty"] == 0

    def test_returns_empty_when_order_not_found(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            mock_raw = MagicMock()
            mock_raw.access_token = fresh_token
            mock_raw.order_book.return_value = {"data": []}
            client._raw = mock_raw

            result = client.get_order_status("NOTFOUND")
            assert result["order_id"] == "NOTFOUND"
            assert result["status"] == ""

    def test_handles_list_response_format(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            mock_raw = MagicMock()
            mock_raw.access_token = fresh_token
            # Some SDKs return a list directly
            mock_raw.order_book.return_value = [
                {
                    "orderId": "ORD002",
                    "orderStatus": "OPEN",
                    "filledQty": 10,
                    "remainingQty": 55,
                }
            ]
            client._raw = mock_raw

            result = client.get_order_status("ORD002")
            assert result["order_id"] == "ORD002"
            assert result["status"] == "OPEN"
            assert result["filled_qty"] == 10

    def test_handles_sdk_error_gracefully(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            mock_raw = MagicMock()
            mock_raw.access_token = fresh_token
            mock_raw.order_book.side_effect = RuntimeError("SDK network error")
            client._raw = mock_raw

            result = client.get_order_status("ORD003")
            assert result["order_id"] == "ORD003"
            assert result["status"] == ""  # Error path returns empty status
            assert "SDK network error" in result["message"]


# ── _poll_for_order_fill tests ────────────────────────────────────────────────

class TestPollForOrderFill:
    def _make_mock_client(self) -> tuple:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)
            return client, fresh_token

    def test_returns_immediately_on_filled(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            call_count = [0]

            def mock_get_status(oid):
                call_count[0] += 1
                return {
                    "order_id": oid,
                    "status": "FILLED",
                    "filled_qty": 65,
                    "remaining_qty": 0,
                    "message": "OK",
                }

            client.get_order_status = mock_get_status

            start = time.time()
            result = client._poll_for_order_fill("ORD001", max_wait_seconds=10.0, poll_interval_seconds=1.0)
            elapsed = time.time() - start

            assert result["status"] == "FILLED"
            assert result["filled_qty"] == 65
            assert elapsed < 2.0  # Should return immediately (first poll)
            assert call_count[0] == 1

    def test_polls_until_rejected_state(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            call_count = [0]

            def mock_get_status(oid):
                call_count[0] += 1
                if call_count[0] < 3:
                    return {
                        "order_id": oid,
                        "status": "PARTIALLY_FILLED",
                        "filled_qty": 20,
                        "remaining_qty": 45,
                        "message": "Partial",
                    }
                return {
                    "order_id": oid,
                    "status": "REJECTED",
                    "filled_qty": 0,
                    "remaining_qty": 0,
                    "message": "Order rejected by exchange",
                }

            client.get_order_status = mock_get_status

            result = client._poll_for_order_fill(
                "ORD002",
                max_wait_seconds=10.0,
                poll_interval_seconds=0.5,
            )

            assert result["status"] == "REJECTED"
            assert call_count[0] == 3

    def test_respects_max_wait_timeout(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            call_count = [0]

            def mock_get_status(oid):
                call_count[0] += 1
                return {
                    "order_id": oid,
                    "status": "PARTIALLY_FILLED",  # Never terminal
                    "filled_qty": 10,
                    "remaining_qty": 55,
                    "message": "Still pending",
                }

            client.get_order_status = mock_get_status

            start = time.time()
            result = client._poll_for_order_fill(
                "ORD003",
                max_wait_seconds=2.0,
                poll_interval_seconds=0.5,
            )
            elapsed = time.time() - start

            # Should have timed out after ~2s
            assert 1.8 <= elapsed <= 3.0
            assert result["status"] == "PARTIALLY_FILLED"
            # Multiple polls occurred
            assert call_count[0] >= 3

    def test_terminates_on_cancelled_state(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            call_count = [0]

            def mock_get_status(oid):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {
                        "order_id": oid,
                        "status": "OPEN",
                        "filled_qty": 0,
                        "remaining_qty": 65,
                        "message": "",
                    }
                return {
                    "order_id": oid,
                    "status": "CANCELLED",
                    "filled_qty": 0,
                    "remaining_qty": 65,
                    "message": "User cancelled",
                }

            client.get_order_status = mock_get_status

            result = client._poll_for_order_fill(
                "ORD004",
                max_wait_seconds=10.0,
                poll_interval_seconds=1.0,
            )

            assert result["status"] == "CANCELLED"
            assert call_count[0] == 2

    def test_partial_fill_then_fill_transitions(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            call_count = [0]

            def mock_get_status(oid):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {
                        "order_id": oid, "status": "PARTIALLY_FILLED",
                        "filled_qty": 30, "remaining_qty": 35, "message": "",
                    }
                elif call_count[0] == 2:
                    return {
                        "order_id": oid, "status": "PARTIALLY_FILLED",
                        "filled_qty": 55, "remaining_qty": 10, "message": "",
                    }
                return {
                    "order_id": oid, "status": "FILLED",
                    "filled_qty": 65, "remaining_qty": 0, "message": "",
                }

            client.get_order_status = mock_get_status

            result = client._poll_for_order_fill(
                "ORD005",
                max_wait_seconds=10.0,
                poll_interval_seconds=0.5,
            )

            assert result["status"] == "FILLED"
            assert result["filled_qty"] == 65
            assert call_count[0] == 3


# ── Case-insensitivity tests ──────────────────────────────────────────────────

class TestStatusCaseInsensitive:
    def test_filled_uppercase_terminates(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            def mock_get_status(oid):
                return {"order_id": oid, "status": "FILLED", "filled_qty": 65, "remaining_qty": 0, "message": ""}

            client.get_order_status = mock_get_status
            result = client._poll_for_order_fill("ORD001", max_wait_seconds=5.0)
            assert result["status"] == "FILLED"

    def test_rejected_lowercase_terminates(self) -> None:
        from mstock_client import MStockTypeBClient
        from config import APIConfig

        fresh_token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjEwMTAsInN1YiI6InRlc3QifQ.signature"
        with patch.dict("os.environ", {"MSTOCK_ACCESS_TOKEN": fresh_token}):
            cfg = APIConfig(base_url="", api_key="k", api_secret="", client_id="c")
            client = MStockTypeBClient(cfg)

            def mock_get_status(oid):
                return {"order_id": oid, "status": "rejected", "filled_qty": 0, "remaining_qty": 0, "message": ""}

            client.get_order_status = mock_get_status
            result = client._poll_for_order_fill("ORD001", max_wait_seconds=5.0)
            assert result["status"] == "rejected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])