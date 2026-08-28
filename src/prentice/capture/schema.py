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


class SessionManifest(BaseModel):
    session_id: str
    epoch0_utc: str  # ISO 8601, human-readable reference only — not used for offset math
    fps: int
    screen_width: int
    screen_height: int
    backing_scale_factor: float  # video pixel dims = screen dims * this factor (e.g. 2.0 on Retina)
    os_version: str
    video_path: str
    events_path: str
    avfoundation_device_index: int
    avfoundation_device_name: str


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
