import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_state_after_auth_attempt_is_not_verifying_forever():
    ui = object.__new__(ScalperUI)

    label = ui._pf_status_label_from_data_status({
        "broker_auth": "TOKEN_SET_NOT_VERIFIED",
        "auth_validation_attempted": True,
        "data_quality_status": "TOKEN_NOT_VERIFIED",
    })

    assert label != "VERIFYING_BROKER_TOKEN"
    assert label == "BROKER_AUTH_FAILED"


def test_auth_ok_chain_empty_waits_for_option_chain():
    ui = object.__new__(ScalperUI)

    label = ui._pf_status_label_from_data_status({
        "broker_auth": "AUTH_OK",
        "auth_validation_attempted": True,
        "data_quality_status": "OPTION_CHAIN_EMPTY",
    })

    assert label == "WAITING_FOR_OPTION_CHAIN"
