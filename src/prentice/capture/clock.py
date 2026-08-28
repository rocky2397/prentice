"""Shared clock anchor for correlating video frames with logged input events.

Both the video encoder (fixed output fps) and the event logger derive their
timing from the same anchor, so downstream stages can map an event's
``t_ms`` to a video frame index via ``round(t_ms / 1000 * fps)`` without
needing per-frame timestamps to be stored anywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ClockAnchor:
    """A fixed point in time, captured once at session start.

    ``t0_monotonic`` is the source of truth for all event offsets — it's
    immune to wall-clock adjustments (NTP sync, sleep/wake) during a
    recording. ``epoch0_utc`` is stored purely as a human-readable reference
    in the session manifest and is never used for offset math.
    """

    t0_monotonic: float
    epoch0_utc: float

    @classmethod
    def now(cls) -> ClockAnchor:
        return cls(t0_monotonic=time.monotonic(), epoch0_utc=time.time())

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.t0_monotonic) * 1000.0
