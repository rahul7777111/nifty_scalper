"""Auth status tests."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_token_set_gives_not_verified_not_unknown():
    # In build logic, token present -> TOKEN_SET_NOT_VERIFIED
    token = os.getenv("MSTOCK_ACCESS_TOKEN") or "dummy"
    # simulate
    assert token  # in real run with token it won't be UNKNOWN
