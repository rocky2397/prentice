"""Tests for recovering capture-time grounding evidence (accessibility
identifier + coordinate fallback) by walking the traceability chain the
earlier stages left behind.
"""

from __future__ import annotations

from prentice.capture.schema import AXElement, KeyEvent, MouseClickEvent
from prentice.ground.provenance import evidence_for, load_segments_by_step_id
from prentice.interpret.schema import Step
from prentice.refine.schema import RefinedStep
from prentice.segment.schema import Segment


def _click(x: float, y: float, ax: AXElement | None = None) -> MouseClickEvent:
    return MouseClickEvent(t_ms=0.0, x=x, y=y, button="left", pressed=True, ax_element=ax)


def _segment(segment_id: str, events: list) -> Segment:
    return Segment(
        segment_id=segment_id,
        source="event_log",
        action_hint="click",
        start_ms=0.0,
        end_ms=10.0,
        frame_start=0,
        frame_end=1,
        events=events,
    )


def _refined(source_step_ids: list[str]) -> RefinedStep:
    return RefinedStep(
        step_id="refined-0000",
        source_step_ids=source_step_ids,
        source="event_log",
        intent="x",
        action_type="click",
        target_description="a button",
    )


def test_evidence_is_empty_without_events():
    """The imported-session case: no event log, so nothing to recover."""
    mapping = {"s-0000": _segment("seg-0000", [])}
    evidence = evidence_for(_refined(["s-0000"]), mapping)
    assert evidence.ax_role is None
    assert evidence.x is None


def test_evidence_is_empty_when_step_id_is_unknown():
    assert evidence_for(_refined(["missing"]), {}).x is None


def test_evidence_recovers_ax_element_and_coordinates():
    ax = AXElement(role="AXButton", title="Save", description="Save the file", value=None)
    mapping = {"s-0000": _segment("seg-0000", [_click(10.0, 20.0, ax)])}
    evidence = evidence_for(_refined(["s-0000"]), mapping)
    assert evidence.ax_role == "AXButton"
    assert evidence.ax_title == "Save"
    assert (evidence.x, evidence.y) == (10.0, 20.0)


def test_click_carrying_ax_wins_over_an_earlier_click_without_one():
    """The accessibility identifier is the strongest available signal, so a
    later click that has one beats an earlier click that doesn't."""
    ax = AXElement(role="AXButton", title="Save")
    mapping = {
        "s-0000": _segment("seg-0000", [_click(1.0, 1.0)]),
        "s-0001": _segment("seg-0001", [_click(9.0, 9.0, ax)]),
    }
    evidence = evidence_for(_refined(["s-0000", "s-0001"]), mapping)
    assert evidence.ax_title == "Save"
    assert (evidence.x, evidence.y) == (9.0, 9.0)


def test_coordinates_still_recovered_when_no_click_has_ax():
    mapping = {"s-0000": _segment("seg-0000", [_click(5.0, 6.0)])}
    evidence = evidence_for(_refined(["s-0000"]), mapping)
    assert evidence.ax_role is None
    assert (evidence.x, evidence.y) == (5.0, 6.0)


def test_non_click_events_are_ignored():
    mapping = {"s-0000": _segment("seg-0000", [KeyEvent(t_ms=0.0, key="a", pressed=True)])}
    assert evidence_for(_refined(["s-0000"]), mapping).x is None


def test_load_segments_by_step_id_maps_steps_to_their_segments(tmp_path):
    segment = _segment("seg-0000", [_click(1.0, 2.0)])
    (tmp_path / "segments.jsonl").write_text(segment.model_dump_json() + "\n")
    step = Step(
        step_id="s-0000",
        segment_id="seg-0000",
        source="event_log",
        intent="x",
        action_type="click",
        target_description="y",
    )
    (tmp_path / "steps.jsonl").write_text(step.model_dump_json() + "\n")

    mapping = load_segments_by_step_id(tmp_path)
    assert mapping["s-0000"].segment_id == "seg-0000"


def test_load_segments_by_step_id_tolerates_missing_files(tmp_path):
    """Grounding must still work off refined_steps.jsonl alone if the
    intermediate stage output was cleaned up."""
    assert load_segments_by_step_id(tmp_path) == {}
