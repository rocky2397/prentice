"""Shared output schema for Stage 2 (Segment): both the event-log clustering
path and the CLIP-fallback boundary-detection path emit the same ``Segment``
shape so Stage 3 can consume either uniformly. ``source`` is the field that
must survive untouched into later stages — it's how accuracy gets reported
separately per path in the eval harness, and later how low-confidence
(inferred) segments get flagged for extra verification.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from ..capture.schema import CaptureEvent
from ..io_utils import load_json, load_jsonl


class Segment(BaseModel):
    segment_id: str
    source: Literal["event_log", "inferred"]
    # "scene_change" is the only hint the CLIP path can ever produce — it knows a
    # boundary exists, not what kind of action happened there. The event-log path
    # always knows the concrete action type, derived directly from event type.
    action_hint: Literal["click", "drag", "scroll", "type", "window_switch", "scene_change"]
    start_ms: float
    end_ms: float
    frame_start: int  # inclusive; both bounds are indices into the same video's frame timeline
    frame_end: int
    events: list[CaptureEvent] = Field(default_factory=list)  # always empty when source="inferred"


SegmentAdapter: TypeAdapter = TypeAdapter(Segment)


class _BaseSegmentRunMeta(BaseModel):
    """Records the parameters a segmentation run used, alongside its output,
    for reproducibility — the project's eval story depends on being able to
    say which parameters produced which numbers."""

    session_id: str
    segment_count: int


class EventLogSegmentMeta(_BaseSegmentRunMeta):
    source: Literal["event_log"] = "event_log"
    drag_pixel_threshold: float
    scroll_gap_ms: float
    type_gap_ms: float
    frame_pad_ms: float


class InferredSegmentMeta(_BaseSegmentRunMeta):
    source: Literal["inferred"] = "inferred"
    clip_model_name: str
    clip_pretrained: str
    sample_fps: float
    similarity_threshold: float
    device: str


SegmentRunMeta = Annotated[
    EventLogSegmentMeta | InferredSegmentMeta,
    Field(discriminator="source"),
]

SegmentRunMetaAdapter: TypeAdapter = TypeAdapter(SegmentRunMeta)


def load_segments(path: str) -> list[Segment]:
    """Parse a ``segments.jsonl`` file into validated Segment models, in order."""
    return load_jsonl(path, SegmentAdapter)


def load_segment_meta(path: str) -> SegmentRunMeta:
    return load_json(path, SegmentRunMetaAdapter)
