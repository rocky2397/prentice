from prentice.capture.schema import MouseClickEvent
from prentice.segment.schema import (
    EventLogSegmentMeta,
    InferredSegmentMeta,
    Segment,
    SegmentAdapter,
    SegmentRunMetaAdapter,
)


def test_event_log_segment_roundtrip():
    segment = Segment(
        segment_id="s-0000",
        source="event_log",
        action_hint="click",
        start_ms=100.0,
        end_ms=150.0,
        frame_start=3,
        frame_end=5,
        events=[MouseClickEvent(t_ms=100.0, x=1.0, y=2.0, button="Button.left", pressed=True)],
    )
    parsed = SegmentAdapter.validate_json(segment.model_dump_json())
    assert parsed.source == "event_log"
    assert len(parsed.events) == 1


def test_inferred_segment_has_no_events_by_construction():
    segment = Segment(
        segment_id="s-0001",
        source="inferred",
        action_hint="scene_change",
        start_ms=0.0,
        end_ms=500.0,
        frame_start=0,
        frame_end=15,
    )
    assert segment.events == []
    parsed = SegmentAdapter.validate_json(segment.model_dump_json())
    assert parsed.events == []


def test_segment_run_meta_discriminates_by_source():
    event_log_meta = EventLogSegmentMeta(
        session_id="s",
        segment_count=3,
        drag_pixel_threshold=8.0,
        scroll_gap_ms=400.0,
        type_gap_ms=750.0,
        frame_pad_ms=150.0,
    )
    inferred_meta = InferredSegmentMeta(
        session_id="s",
        segment_count=5,
        clip_model_name="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
        sample_fps=2.0,
        similarity_threshold=0.9,
        device="cpu",
    )
    parsed_event_log = SegmentRunMetaAdapter.validate_json(event_log_meta.model_dump_json())
    parsed_inferred = SegmentRunMetaAdapter.validate_json(inferred_meta.model_dump_json())
    assert isinstance(parsed_event_log, EventLogSegmentMeta)
    assert isinstance(parsed_inferred, InferredSegmentMeta)
