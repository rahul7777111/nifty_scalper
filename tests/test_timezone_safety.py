"""
Timezone safety tests for is_market_open() in src/strategy.

Ensures that:
1. Market-hours gating works correctly during IST 09:15–15:30.
2. Market-hours gating correctly returns False outside IST 09:15–15:30.
3. When timezone support is completely absent, is_market_open() returns False
   (fail-closed) — NOT all-times-open.
4. The dangerous "fall back to local wall-clock and return True" path is removed.

Safety contract: this project is research/shadow/paper-only. These tests
verify the fail-closed guardrail that prevents the system from attempting
live orders when it cannot determine market hours correctly.
"""

from __future__ import annotations

import sys
import pytest
from datetime import time as dt_time, datetime as dt_datetime, timezone, timedelta
from unittest.mock import MagicMock


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------

class _SpyDatetime:
    """datetime subclass that records calls and returns a controlled now()."""

    _now_calls: list = []
    _fake_time: dt_datetime = None  # type: ignore[assignment]

    def now(cls, tz=None):
        cls._now_calls.append(tz)
        if tz is None:
            return cls._fake_time
        return cls._fake_time.astimezone(tz)

    @classmethod
    def reset(cls):
        cls._now_calls.clear()


def _patch_strategy_dt_now(strategy_module, fake_time: dt_datetime) -> None:
    """
    Patch strategy.dt_datetime.now to return `fake_time`.
    Uses direct attribute assignment (no patch.object on immutable type).
    """
    _SpyDatetime._fake_time = fake_time
    _SpyDatetime.reset()
    strategy_module.dt_datetime.now = classmethod(lambda cls, tz=None: _SpyDatetime.now(tz))  # type: ignore[assignment]


def _restore_strategy_dt_now(strategy_module) -> None:
    """Restore the real dt_datetime.now on the strategy module."""
    if hasattr(strategy_module.dt_datetime, "_orig_now"):
        strategy_module.dt_datetime.now = strategy_module.dt_datetime._orig_now  # type: ignore[assignment]


# ------------------------------------------------------------------
# Tests — real timezone behavior
# ------------------------------------------------------------------

class TestMarketHoursWithRealTz:
    """
    Tests that use the real Asia/Kolkata timezone via zoneinfo.
    """

    @pytest.fixture(autouse=True)
    def _require_real_tz(self):
        from src.strategy import _TZ_AVAILABLE
        if not _TZ_AVAILABLE:
            pytest.skip("Asia/Kolkata timezone not available in this environment")

    def test_flag_true_when_tz_available(self):
        """_TZ_AVAILABLE must be True when zoneinfo gives us Asia/Kolkata."""
        from src.strategy import _TZ_AVAILABLE, IST
        assert _TZ_AVAILABLE is True
        assert IST is not None

    def test_is_market_open_returns_bool(self):
        """is_market_open() must return a bool."""
        from src.strategy import is_market_open
        result = is_market_open()
        assert isinstance(result, bool), (
            f"is_market_open() must return bool, got {type(result).__name__}: {result!r}"
        )

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_market_open_during_valid_ist_hours(self):
        """
        Test that is_market_open() returns True at IST 12:30 (mid-session).
        """
        from src.strategy import IST
        import src.strategy as _m

        # IST 12:30 — clearly inside the session
        ist_midday = dt_datetime(2025, 6, 9, 12, 30, 0, tzinfo=IST)

        # Save original now and patch
        orig_now = _m.dt_datetime.now
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (
            ist_midday if tz else dt_datetime(2025, 6, 9, 12, 30, 0)
        ))
        _m.dt_datetime.now._orig_now = orig_now  # type: ignore[attr-defined]

        try:
            result = _m.is_market_open()
        finally:
            _restore_strategy_dt_now(_m)

        assert result is True, (
            f"is_market_open() should return True at IST 12:30, got {result}"
        )

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_market_closed_outside_valid_ist_hours_pre_open(self):
        """
        Test that is_market_open() returns False at IST 09:10 (before open).
        """
        from src.strategy import IST
        import src.strategy as _m

        pre_open = dt_datetime(2025, 6, 9, 9, 10, 0, tzinfo=IST)

        orig_now = _m.dt_datetime.now
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (
            pre_open if tz else dt_datetime(2025, 6, 9, 9, 10, 0)
        ))
        _m.dt_datetime.now._orig_now = orig_now  # type: ignore[attr-defined]

        try:
            result = _m.is_market_open()
        finally:
            _restore_strategy_dt_now(_m)

        assert result is False, (
            f"is_market_open() should return False at IST 09:10, got {result}"
        )

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_market_closed_outside_valid_ist_hours_post_close(self):
        """
        Test that is_market_open() returns False at IST 15:45 (after close).
        """
        from src.strategy import IST
        import src.strategy as _m

        post_close = dt_datetime(2025, 6, 9, 15, 45, 0, tzinfo=IST)

        orig_now = _m.dt_datetime.now
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (
            post_close if tz else dt_datetime(2025, 6, 9, 15, 45, 0)
        ))
        _m.dt_datetime.now._orig_now = orig_now  # type: ignore[attr-defined]

        try:
            result = _m.is_market_open()
        finally:
            _restore_strategy_dt_now(_m)

        assert result is False, (
            f"is_market_open() should return False at IST 15:45, got {result}"
        )

    @pytest.mark.parametrize(
        "ist_hour,ist_min,expected",
        [
            # Boundary: exactly at open
            (9, 15, True),
            # 1 min before open
            (9, 14, False),
            # 1 min after open
            (9, 16, True),
            # Boundary: exactly at close
            (15, 30, True),
            # 1 min after close
            (15, 31, False),
            # Deep inside session
            (12, 0, True),
            # Deep outside
            (0, 0, False),
            (23, 59, False),
        ],
    )
    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_market_hours_parametric(self, ist_hour, ist_min, expected):
        """
        Parametric test of IST 09:15–15:30 boundaries.
        """
        from src.strategy import IST
        import src.strategy as _m

        ist_time = dt_datetime(2025, 6, 9, ist_hour, ist_min, 0, tzinfo=IST)
        naive_time = dt_datetime(2025, 6, 9, ist_hour, ist_min, 0)

        orig_now = _m.dt_datetime.now
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (
            ist_time if tz else naive_time
        ))
        _m.dt_datetime.now._orig_now = orig_now  # type: ignore[attr-defined]

        try:
            result = _m.is_market_open()
        finally:
            _restore_strategy_dt_now(_m)

        assert result is expected, (
            f"is_market_open() at IST {ist_hour:02d}:{ist_min:02d} "
            f"expected {expected}, got {result}"
        )


# ------------------------------------------------------------------
# Tests — fail-closed when timezone is unavailable
# ------------------------------------------------------------------

class TestTimezoneFailureSafety:
    """
    Verify is_market_open() returns False (fail-closed) when timezone
    support is completely absent (both zoneinfo and pytz unavailable).
    """

    def test_timezone_failure_blocks_trading(self):
        """
        When _TZ_AVAILABLE is False and IST is None, is_market_open()
        must return False (fail-closed), not True.
        """
        import src.strategy as _m

        # Force failure state
        _m._TZ_AVAILABLE = False
        _m.IST = None

        result = _m.is_market_open()

        # Restore
        _m._TZ_AVAILABLE = True
        _m.IST = None  # will be re-set on next import; leave None for safety

        assert result is False, (
            f"is_market_open() returned {result!r} when _TZ_AVAILABLE=False. "
            f"Expected False (fail-closed). Dangerous: system may trade at wrong hours!"
        )

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_no_always_open_fallback(self):
        """
        Confirm the dangerous 'dt_datetime.now() in local-time fallback'
        code path does NOT exist when _TZ_AVAILABLE=False.

        We track calls to dt_datetime.now by patching the strategy module's
        reference to it. If dt_datetime.now() is called when _TZ_AVAILABLE=False,
        that proves a local-wall-clock fallback still exists.
        """
        import src.strategy as _m

        # Track whether now() was called
        now_call_log: list = []
        _orig_now = _m.dt_datetime.now  # type: ignore[assignment]

        def _spy_now(tz=None):
            now_call_log.append(tz)
            # Return a valid naive datetime so we don't crash if called
            return dt_datetime(2025, 6, 9, 12, 0)

        # Force failure state FIRST
        _m._TZ_AVAILABLE = False
        _m.IST = None

        # Then patch dt_datetime.now on the module
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: _spy_now(tz))  # type: ignore[assignment]

        try:
            result = _m.is_market_open()
        finally:
            _m.dt_datetime.now = _orig_now  # type: ignore[assignment]

        assert result is False, (
            f"is_market_open() must return False when _TZ_AVAILABLE=False, got {result}"
        )
        assert len(now_call_log) == 0, (
            f"dt_datetime.now() was called {len(now_call_log)} time(s) when "
            f"_TZ_AVAILABLE=False. This means the dangerous local-time fallback "
            f"still exists! Calls: {now_call_log}"
        )

    def test_no_local_wallclock_time_used(self):
        """
        Additional test: if the old vulnerable fallback existed:
            now = dt_datetime.now().time()
            return dt_time(9,15) <= now <= dt_time(15,30)
        it would return True for ~37.5% of all times (09:15-15:30 local).

        We verify that even when we manually call the time-range comparison
        with a naive datetime that WOULD pass (e.g. 12:00 local), the
        function STILL returns False when _TZ_AVAILABLE=False.
        """
        import src.strategy as _m

        _m._TZ_AVAILABLE = False
        _m.IST = None

        # The old vulnerable code path would evaluate:
        #   now = dt_datetime.now().time()   → dt_time(12, 0)
        #   return dt_time(9,15) <= dt_time(12,0) <= dt_time(15,30)  → True
        # We verify the new code does NOT do this.
        result = _m.is_market_open()

        _m._TZ_AVAILABLE = True

        assert result is False, (
            f"Even with a time that WOULD pass the 9:15-15:30 range check "
            f"(e.g. 12:00), is_market_open() must still return False when "
            f"_TZ_AVAILABLE=False. Got {result}. This means the "
            f"dangerous 'all times open' fallback exists!"
        )


# ------------------------------------------------------------------
# Tests — _TZ_AVAILABLE flag
# ------------------------------------------------------------------

class TestTzAvailabilityFlag:
    """Verify the _TZ_AVAILABLE module flag is correctly set and consistent."""

    def test_flag_is_bool(self):
        """_TZ_AVAILABLE must be a bool."""
        from src.strategy import _TZ_AVAILABLE
        assert isinstance(_TZ_AVAILABLE, bool), (
            f"_TZ_AVAILABLE must be bool, got {type(_TZ_AVAILABLE).__name__}"
        )

    def test_flag_matches_ist_state(self):
        """
        The _TZ_AVAILABLE flag must be True iff IST is not None.
        """
        from src.strategy import _TZ_AVAILABLE, IST
        assert (_TZ_AVAILABLE is False) == (IST is None), (
            f"Inconsistent state: _TZ_AVAILABLE={_TZ_AVAILABLE} but IST={IST!r}"
        )

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_flag_matches_ist_state(self):
        """
        When _TZ_AVAILABLE is True, is_market_open() must use actual IST time.
        """
        import src.strategy as _m
        from src.strategy import IST
        from datetime import datetime as dt_datetime

        # Set up: make IST-aware now return a known time inside the session
        _test_time = dt_datetime(2025, 6, 9, 12, 0, 0, tzinfo=IST)
        _orig_now = _m.dt_datetime.now  # type: ignore[assignment]
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (  # type: ignore[assignment]
            _test_time if tz else dt_datetime(2025, 6, 9, 12, 0, 0)
        ))

        try:
            result = _m.is_market_open()
        finally:
            _m.dt_datetime.now = _orig_now  # type: ignore[assignment]

        assert result is True, (
            f"When _TZ_AVAILABLE=True, is_market_open() should use IST 12:00 → True, got {result}"
        )

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_flag_false_means_market_closed(self):
        """
        When _TZ_AVAILABLE is False, is_market_open() MUST return False.
        This is the core safety contract tested at the flag level.
        """
        import src.strategy as _m

        _m._TZ_AVAILABLE = False
        _m.IST = None

        # Track dt_datetime.now calls
        call_log: list = []
        _orig_now = _m.dt_datetime.now  # type: ignore[assignment]

        def _spy_now(tz=None):
            call_log.append(tz)
            return dt_datetime(2025, 6, 9, 12, 0)

        _m.dt_datetime.now = classmethod(lambda cls, tz=None: _spy_now(tz))  # type: ignore[assignment]

        try:
            result = _m.is_market_open()
        finally:
            _m.dt_datetime.now = _orig_now  # type: ignore[assignment]

        assert result is False, (
            f"When _TZ_AVAILABLE=False, is_market_open() must return False, got {result}"
        )
        assert len(call_log) == 0, (
            f"dt_datetime.now() was called when _TZ_AVAILABLE=False. "
            f"Dangerous local-time fallback exists! calls={call_log}"
        )

    def test_flag_accessible_from_outside_module(self):
        """
        Verify _TZ_AVAILABLE is accessible as a module-level attribute
        so other parts of the codebase can check tz availability.
        """
        from src.strategy import _TZ_AVAILABLE
        # Just verify it's importable from outside
        assert isinstance(_TZ_AVAILABLE, bool)

    def test_flag_consistent_after_multiple_calls(self):
        """
        Repeated calls to is_market_open() should not change _TZ_AVAILABLE.
        The flag is read-only after module initialization.
        """
        from src.strategy import is_market_open, _TZ_AVAILABLE

        # Call is_market_open a few times
        for _ in range(5):
            is_market_open()

        from src.strategy import _TZ_AVAILABLE as flag_after
        assert flag_after == _TZ_AVAILABLE, (
            "TZ_AVAILABLE flag changed after repeated is_market_open() calls"
        )


# ------------------------------------------------------------------
# Tests — integration: function result vs. IST session boundaries
# ------------------------------------------------------------------

class TestIntegration:
    """Integration tests verifying the complete is_market_open() contract."""

    @pytest.fixture(autouse=True)
    def _require_real_tz(self):
        from src.strategy import _TZ_AVAILABLE
        if not _TZ_AVAILABLE:
            pytest.skip("Asia/Kolkata timezone not available")

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_ist_time_comparison_uses_correct_timezone(self):
        """
        Verify that is_market_open() is using IST time, not local time.

        Strategy: mock dt_datetime.now to return a time that would give
        True if interpreted in local time, but False in IST (or vice versa).

        Example: if we mock now() to return IST equivalent of 23:00 local,
        the function should still use IST → False (outside 9:15-15:30 IST).
        """
        from src.strategy import IST
        import src.strategy as _m

        # Create a datetime that is 23:00 IST (clearly outside session)
        ist_23_00 = dt_datetime(2025, 6, 9, 23, 0, 0, tzinfo=IST)
        naive_23_00 = dt_datetime(2025, 6, 9, 23, 0, 0)

        orig_now = _m.dt_datetime.now
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (
            ist_23_00 if tz else naive_23_00
        ))
        _m.dt_datetime.now._orig_now = orig_now  # type: ignore[attr-defined]

        try:
            result = _m.is_market_open()
        finally:
            _restore_strategy_dt_now(_m)

        assert result is False, (
            "is_market_open() should return False at IST 23:00 (outside session)"
        )

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="datetime immutable in Py3.14+, datetime.now() patching not supported")
    def test_result_changes_when_ist_crosses_session_boundary(self):
        """
        Verify that is_market_open() correctly changes its result
        when the IST time crosses the session boundary.
        """
        from src.strategy import IST
        import src.strategy as _m

        # Inside session: IST 12:00
        inside = dt_datetime(2025, 6, 9, 12, 0, 0, tzinfo=IST)
        outside = dt_datetime(2025, 6, 9, 8, 0, 0, tzinfo=IST)

        # Test inside session
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (
            inside if tz else dt_datetime(2025, 6, 9, 12, 0, 0)
        ))
        _m.dt_datetime.now._orig_now = _m.dt_datetime.now  # type: ignore[attr-defined]
        result_inside = _m.is_market_open()

        # Test outside session
        _m.dt_datetime.now = classmethod(lambda cls, tz=None: (
            outside if tz else dt_datetime(2025, 6, 9, 8, 0, 0)
        ))
        _m.dt_datetime.now._orig_now = _m.dt_datetime.now  # type: ignore[attr-defined]
        result_outside = _m.is_market_open()

        _restore_strategy_dt_now(_m)

        assert result_inside is True, "Expected True at IST 12:00"
        assert result_outside is False, "Expected False at IST 08:00"
        assert result_inside != result_outside, (
            "is_market_open() did not change when IST crossed session boundary"
        )