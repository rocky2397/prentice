"""Recovers capture-time grounding evidence for a refined step.

ARCHITECTURE.md §Stage 5 requires a UI-replay action to store the
accessibility-tree identifier where one was captured, and permits raw
coordinates as a last-resort fallback hint. Stage 1 captures both (an
``ax_element`` on every click, plus the click's x/y), but Stage 3's ``Step``
has no field for either, so they don't reach Stage 5 by simply reading
``refined_steps.jsonl``.

They are still fully recoverable, because every stage kept its traceability
link: RefinedStep.source_step_ids -> Step.step_id -> Step.segment_id ->
Segment.events -> MouseClickEvent.ax_element / .x / .y. This module walks
that chain over the session directory's own on-disk output.

For an imported (video-only) session this correctly yields nothing at all:
``has_events`` is false, ``events.jsonl`` is empty, so every Segment carries
an empty event list and no AX identifier or coordinate exists to recover.
That's the known imported-session gap from ARCHITECTURE.md §2 showing up
exactly where it should, rather than being papered over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..capture.schema import MouseClickEvent
from ..interpret.schema import load_steps
from ..refine.schema import RefinedStep
from ..segment.schema import Segment, load_segments


@dataclass(frozen=True)
class StepEvidence:
    """Capture-time evidence backing one refined step, if any survives."""

    ax_role: str | None = None
    ax_title: str | None = None
    ax_description: str | None = None
    ax_value: str | None = None
    x: float | None = None
    y: float | None = None


EMPTY_EVIDENCE = StepEvidence()


def load_segments_by_step_id(session_dir: Path) -> dict[str, Segment]:
    """Map each Stage 3 step_id to the Segment it was interpreted from.

    Returns an empty mapping if either file is missing — grounding must still
    work (falling back to the semantic description alone) for a session whose
    intermediate stage output was cleaned up.
    """
    steps_path = session_dir / "steps.jsonl"
    segments_path = session_dir / "segments.jsonl"
    if not steps_path.is_file() or not segments_path.is_file():
        return {}

    segments_by_id = {s.segment_id: s for s in load_segments(str(segments_path))}
    return {
        step.step_id: segments_by_id[step.segment_id]
        for step in load_steps(str(steps_path))
        if step.segment_id in segments_by_id
    }


def evidence_for(refined: RefinedStep, segments_by_step_id: dict[str, Segment]) -> StepEvidence:
    """Best available AX identifier / coordinate hint for a refined step.

    A refined step can absorb several Stage 3 steps, each with its own
    segment and events. The first click event carrying an accessibility
    element wins, since that's the strongest grounding signal available; if
    none has one, the first click still supplies a coordinate fallback hint.
    """
    click_events = [
        event
        for step_id in refined.source_step_ids
        for event in getattr(segments_by_step_id.get(step_id), "events", [])
        if isinstance(event, MouseClickEvent)
    ]
    if not click_events:
        return EMPTY_EVIDENCE

    with_ax = next((e for e in click_events if e.ax_element is not None), None)
    chosen = with_ax or click_events[0]
    ax = chosen.ax_element
    return StepEvidence(
        ax_role=ax.role if ax else None,
        ax_title=ax.title if ax else None,
        ax_description=ax.description if ax else None,
        ax_value=ax.value if ax else None,
        x=chosen.x,
        y=chosen.y,
    )
