import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_login_totp_success_state_can_be_reused_by_paper_forward():
    ui = object.__new__(ScalperUI)
    ui._pf_last_auth_status = "AUTH_OK"
    ui._pf_last_auth_error = ""
    ui._pf_last_auth_endpoint = "login_totp"
    ui._pf_auth_validation_attempted = True
    ui._pf_auth_validation_in_progress = False

    label = ui._pf_status_label_from_data_status({
        "broker_auth": ui._pf_last_auth_status,
        "auth_validation_attempted": ui._pf_auth_validation_attempted,
        "auth_validation_in_progress": ui._pf_auth_validation_in_progress,
        "data_quality_status": "OPTION_CHAIN_EMPTY",
    })

    assert label == "WAITING_FOR_OPTION_CHAIN"
    assert ui._pf_last_auth_endpoint == "login_totp"
