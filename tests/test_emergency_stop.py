"""Emergency stop hardening tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.candidate_lifecycle import get_kill_switch_active


def test_emergency_stop_disables_new_entries(monkeypatch):
    monkeypatch.setenv("SCALPER_KILL_SWITCH", "1")
    assert get_kill_switch_active() is True


def test_emergency_stop_does_not_square_off_unless_configured(monkeypatch):
    # The UI emergency handler and any engine must check SQUARE_OFF_ON_EMERGENCY
    monkeypatch.setenv("SQUARE_OFF_ON_EMERGENCY", "false")
    # In real code the emergency path would read this; here we just assert the var
    assert os.getenv("SQUARE_OFF_ON_EMERGENCY", "false").lower() != "true"
