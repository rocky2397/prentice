"""Event-log path for Stage 2 (Segment): cluster logged input events into
action segments using timestamps and event type. Primary, exact
action-boundary signal per ARCHITECTURE.md §Stage 2 — no model call needed.

This is *temporal* grouping only (deciding where one action ends and the
next begins), not semantic interpretation of what happened — that's Stage 3
(Interpret). Fixing up any fragmentation this heuristic gets wrong (e.g.
merging a burst of keystrokes an app's autocomplete briefly split) is
explicitly Stage 4 (Refine)'s job per ARCHITECTURE.md, not this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from ..capture.schema import (
    CaptureEvent,
    KeyEvent,
    MouseClickEvent,
    MouseScrollEvent,
    WindowSwitchEvent,
)
from .schema import Segment

DEFAULT_DRAG_PIXEL_THRESHOLD = 8.0
DEFAULT_SCROLL_GAP_MS = 400.0
DEFAULT_TYPE_GAP_MS = 750.0
DEFAULT_FRAME_PAD_MS = 150.0


@dataclass(frozen=True)
class ClusteringParams:
    drag_pixel_threshold: float = DEFAULT_DRAG_PIXEL_THRESHOLD
    scroll_gap_ms: float = DEFAULT_SCROLL_GAP_MS
    type_gap_ms: float = DEFAULT_TYPE_GAP_MS
    frame_pad_ms: float = DEFAULT_FRAME_PAD_MS


# (start_ms, end_ms, action_hint, contributing events) — pre-frame-range, pre-id
_RawSegment = tuple[float, float, str, list[CaptureEvent]]


def _frame_index(t_ms: float, fps: float) -> int:
    return round(t_ms / 1000.0 * fps)


def _burst_cluster(events: list[CaptureEvent], gap_ms: float, hint: str, boundary_ts: list[float]) -> list[_RawSegment]:
    """Merge consecutive same-type events into bursts, splitting on either a
    gap exceeding ``gap_ms`` or a window-switch boundary falling between them
    — a window switch is a hard context break even if the gap is small."""
    if not events:
        return []
    bursts: list[_RawSegment] = []
    current = [events[0]]
    for prev, curr in pairwise(events):
        gap = curr.t_ms - prev.t_ms
        crosses_boundary = any(prev.t_ms < b < curr.t_ms for b in boundary_ts)
        if gap <= gap_ms and not crosses_boundary:
            current.append(curr)
        else:
            bursts.append((current[0].t_ms, current[-1].t_ms, hint, list(current)))
            current = [curr]
    bursts.append((current[0].t_ms, current[-1].t_ms, hint, list(current)))
    return bursts


def cluster_events(
    events: list[CaptureEvent],
    *,
    session_id: str,
    fps: float,
    duration_ms: float,
    params: ClusteringParams | None = None,
) -> list[Segment]:
    params = params or ClusteringParams()
    events = sorted(events, key=lambda e: e.t_ms)
    raw_segments: list[_RawSegment] = []

    # window switches: always their own hard-boundary segment
    window_events = [e for e in events if isinstance(e, WindowSwitchEvent)]
    for e in window_events:
        raw_segments.append((e.t_ms, e.t_ms, "window_switch", [e]))
    boundary_ts = sorted(e.t_ms for e in window_events)

    # clicks / drags: pair each press with the next release of the same button
    click_events = [e for e in events if isinstance(e, MouseClickEvent)]
    open_presses: dict[str, MouseClickEvent] = {}
    for e in click_events:
        if e.pressed:
            open_presses[e.button] = e
            continue
        press = open_presses.pop(e.button, None)
        if press is None:
            # release with no matching press in this window (e.g. press happened
            # just before recording started) — its own minimal segment
            raw_segments.append((e.t_ms, e.t_ms, "click", [e]))
            continue
        dx, dy = e.x - press.x, e.y - press.y
        hint = "drag" if (dx * dx + dy * dy) ** 0.5 > params.drag_pixel_threshold else "click"
        raw_segments.append((press.t_ms, e.t_ms, hint, [press, e]))
    for press in open_presses.values():
        # press with no matching release (e.g. recording stopped mid-press)
        raw_segments.append((press.t_ms, press.t_ms, "click", [press]))

    scroll_events = sorted((e for e in events if isinstance(e, MouseScrollEvent)), key=lambda e: e.t_ms)
    raw_segments.extend(_burst_cluster(scroll_events, params.scroll_gap_ms, "scroll", boundary_ts))

    key_events = sorted((e for e in events if isinstance(e, KeyEvent)), key=lambda e: e.t_ms)
    raw_segments.extend(_burst_cluster(key_events, params.type_gap_ms, "type", boundary_ts))

    raw_segments.sort(key=lambda s: s[0])

    segments: list[Segment] = []
    for i, (start_ms, end_ms, hint, contributing) in enumerate(raw_segments):
        pad_start = max(0.0, start_ms - params.frame_pad_ms)
        pad_end = min(duration_ms, end_ms + params.frame_pad_ms)
        segments.append(
            Segment(
                segment_id=f"{session_id}-{i:04d}",
                source="event_log",
                action_hint=hint,
                start_ms=start_ms,
                end_ms=end_ms,
                frame_start=_frame_index(pad_start, fps),
                frame_end=_frame_index(pad_end, fps),
                events=sorted(contributing, key=lambda e: e.t_ms),
            )
        )
    return segments
