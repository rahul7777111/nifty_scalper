#!/usr/bin/env python3
"""
scripts/smoke_paper_forward_data_status.py

Deterministic smoke for PaperForwardDataStatus transitions. No broker calls, no orders.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_forward_engine import PaperForwardDataStatus
from ui import ScalperUI, verify_mstock_token_read_only


class _AuthOK:
    def get_ltp(self, symbol):
        return 23257.95


class _Expired:
    def get_ltp(self, symbol):
        raise RuntimeError("IA401 token expired")


def _assert(name: str, cond: bool, detail: str) -> bool:
    if cond:
        print(f"[SMOKE-DATA] OK {name}: {detail}")
        return True
    print(f"[SMOKE-DATA] FAIL {name}: {detail}")
    return False


def main() -> int:
    print("[SMOKE-DATA] Checking paper-forward data status states")
    bad = False

    unchecked = PaperForwardDataStatus(
        broker_auth="TOKEN_PRESENT_UNCHECKED",
        auth_validation_attempted=False,
        spot=23172.5,
        option_rows=0,
        candle_count=0,
    ).as_dict()
    bad |= not _assert("initial_token_state", unchecked["data_quality_status"] == "TOKEN_NOT_VERIFIED", str(unchecked))

    ok_result = verify_mstock_token_read_only(_AuthOK())
    bad |= not _assert("validation_success_terminal", ok_result.status == "AUTH_OK", str(ok_result))

    expired_result = verify_mstock_token_read_only(_Expired())
    bad |= not _assert("validation_expired_terminal", expired_result.status == "SESSION_EXPIRED", str(expired_result))

    failed = PaperForwardDataStatus(
        broker_auth="AUTH_FAILED:IA401",
        auth_error="IA401 token expired",
        auth_validation_attempted=True,
        auth_validation_endpoint="get_ltp:NIFTY",
        spot=23172.5,
    ).as_dict()
    bad |= not _assert("auth_failure_exact", failed["data_quality_status"] == "AUTH_FAILED", str(failed))

    empty = PaperForwardDataStatus(broker_auth="AUTH_OK", spot=23172.5, option_rows=0, candle_count=0).as_dict()
    bad |= not _assert("rows_zero_empty", empty["option_chain_status"] == "option_chain_empty" and empty["data_quality_status"] == "OPTION_CHAIN_EMPTY", str(empty))

    thin = PaperForwardDataStatus(broker_auth="AUTH_OK", spot=23172.5, option_rows=1, candle_count=0).as_dict()
    bad |= not _assert("rows_one_thin", thin["option_chain_status"] == "THIN_OPTION_CHAIN" and thin["data_quality_status"] == "THIN_OPTION_CHAIN", str(thin))

    no_candles = PaperForwardDataStatus(broker_auth="AUTH_OK", spot=23172.5, option_rows=24, candle_count=0).as_dict()
    bad |= not _assert("chain_ok_candles_empty", no_candles["option_chain_status"] == "DATA_OK" and no_candles["data_quality_status"] == "CANDLES_MISSING", str(no_candles))

    ok = PaperForwardDataStatus(
        broker_auth="AUTH_OK",
        spot=23172.5,
        option_rows=24,
        candle_count=100,
        candle_source="mstock_historical",
    ).as_dict()
    bad |= not _assert("data_ok", ok["data_quality_status"] == "DATA_OK" and ok["candle_source"] != "none", str(ok))

    bad |= not _assert("last_tick_format_iso", ScalperUI._pf_format_last_tick("2026-06-11T13:17:57.431702") == "13:17:57", "")
    bad |= not _assert("last_tick_format_invalid", ScalperUI._pf_format_last_tick("-11T13:17:57.431702") == "n/a", "")

    if bad:
        print("[SMOKE-DATA] FAIL")
        return 1
    print("[SMOKE-DATA] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
