import time

from prentice.capture.clock import ClockAnchor


def test_elapsed_ms_starts_near_zero():
    anchor = ClockAnchor.now()
    assert anchor.elapsed_ms() < 50


def test_elapsed_ms_increases_monotonically():
    anchor = ClockAnchor.now()
    time.sleep(0.05)
    first = anchor.elapsed_ms()
    time.sleep(0.05)
    second = anchor.elapsed_ms()
    assert second > first
    assert first >= 40
