"""Typed models for the capture session manifest and event log.

``events.jsonl`` is written one JSON object per line by the low-level
loggers in ``events.py`` / ``window_tracker.py`` without going through these
models on the hot path (listener callbacks should do as little work as
possible). These models exist so that Stage 1's *output* has a documented,
validated shape that Stage 2 (Segment) can load against.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class AXElement(BaseModel):
    """A snapshot of the accessibility-tree element under the cursor."""

    role: str | None = None
    title: str | None = None
    description: str | None = None
    value: str | None = None


class MouseClickEvent(BaseModel):
    type: Literal["mouse_click"] = "mouse_click"
    t_ms: float
    x: float
    y: float
    button: str
    pressed: bool
    ax_element: AXElement | None = None


class MouseScrollEvent(BaseModel):
    type: Literal["mouse_scroll"] = "mouse_scroll"
    t_ms: float
    x: float
    y: float
    dx: float
    dy: float


class KeyEvent(BaseModel):
    type: Literal["key"] = "key"
    t_ms: float
    key: str
    pressed: bool


class WindowSwitchEvent(BaseModel):
    type: Literal["window_switch"] = "window_switch"
    t_ms: float
    app_name: str | None = None
    bundle_id: str | None = None
    window_title: str | None = None


CaptureEvent = Annotated[
    MouseClickEvent | MouseScrollEvent | KeyEvent | WindowSwitchEvent,
    Field(discriminator="type"),
]

CaptureEventAdapter: TypeAdapter = TypeAdapter(CaptureEvent)


class _BaseSessionManifest(BaseModel):
    session_id: str
    fps: float
    video_width: int  # actual encoded pixel dimensions (from ffprobe), authoritative for both sources
    video_height: int
    video_path: str
    events_path: str
    has_events: bool  # whether an OS-level (or, later, synthesized) event log backs this session


class LiveCaptureManifest(_BaseSessionManifest):
    """A session produced by ``prentice-capture start`` on this machine."""

    source: Literal["live_capture"] = "live_capture"
    epoch0_utc: str  # ISO 8601, human-readable reference only — not used for offset math
    os_version: str
    screen_width: int  # logical points (NSScreen), NOT pixels — see backing_scale_factor
    screen_height: int
    backing_scale_factor: float  # informational only; video_width/height is the source of truth
    avfoundation_device_index: int
    avfoundation_device_name: str


class ImportedManifest(_BaseSessionManifest):
    """A session wrapped from a pre-recorded video via ``prentice-capture import``.

    Always has ``has_events=False``: there is no way to retroactively recover
    OS-level input events from a video that wasn't captured by this tool's
    listeners. Per ARCHITECTURE.md, that means Stage 2+ can't use the
    event-boundary signal for these sessions and must fall back to a
    weaker, vision-only path — a known, explicitly-flagged reliability gap,
    not a silent one.
    """

    source: Literal["imported"] = "imported"
    original_video_path: str  # absolute path to the source file, for reference only (never mutated)
    imported_at_utc: str


SessionManifest = Annotated[
    LiveCaptureManifest | ImportedManifest,
    Field(discriminator="source"),
]

SessionManifestAdapter: TypeAdapter = TypeAdapter(SessionManifest)


def load_events(path: str) -> list[CaptureEvent]:
    """Parse an ``events.jsonl`` file into validated event models, in order."""
    events: list[CaptureEvent] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(CaptureEventAdapter.validate_json(line))
    return events


def load_manifest(path: str) -> SessionManifest:
    with open(path, encoding="utf-8") as f:
        return SessionManifestAdapter.validate_json(f.read())
