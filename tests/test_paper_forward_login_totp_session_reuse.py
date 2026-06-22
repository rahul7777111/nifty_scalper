import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui import ScalperUI


def test_login_totp_reuse_state_is_auth_ok():
    ui = object.__new__(ScalperUI)
    ui._pf_last_auth_status = "AUTH_OK"
    ui._pf_last_auth_error = ""
    ui._pf_last_auth_endpoint = "login_totp"
    ui._pf_auth_validation_attempted = True
    ui._pf_auth_validation_in_progress = False

    fields = ui._pf_auth_status_fields()

    assert ui._pf_last_auth_status == "AUTH_OK"
    assert fields["auth_validation_attempted"] is True
    assert fields["auth_validation_endpoint"] == "login_totp"
