"""End-to-end tests for Stage 5.

Unlike the Stage 2/3 pipeline tests, these need no video and no ffmpeg:
grounding reads only the JSON/JSONL output of the earlier stages, so a
session directory can be faked from files alone.
"""

from __future__ import annotations

import json

from prentice.capture.schema import (
    AXElement,
    ImportedManifest,
    LiveCaptureManifest,
    MouseClickEvent,
)
from prentice.ground.pipeline import ground_session
from prentice.ground.schema import load_ground_meta, load_grounded_steps
from prentice.interpret.schema import Step
from prentice.refine.schema import RefinedStep
from prentice.segment.schema import Segment


def _write_jsonl(path, models) -> None:
    path.write_text("".join(m.model_dump_json() + "\n" for m in models))


def _imported_manifest(session_id: str) -> ImportedManifest:
    return ImportedManifest(
        session_id=session_id,
        fps=30.0,
        video_width=1920,
        video_height=1080,
        video_path="screen.mp4",
        events_path="events.jsonl",
        has_events=False,
        duration_ms=1000.0,
        original_video_path="/tmp/original.mov",
        imported_at_utc="2026-01-01T00:00:00+00:00",
    )


def _refined(step_id: str, source_step_ids: list[str], **overrides) -> RefinedStep:
    base = {
        "step_id": step_id,
        "source_step_ids": source_step_ids,
        "source": "inferred",
        "intent": "click something",
        "action_type": "click",
        "target_description": "a button in the toolbar",
        "parameters": {},
    }
    base.update(overrides)
    return RefinedStep(**base)


def _imported_session(tmp_path, refined_steps):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(_imported_manifest("sess-1").model_dump_json())
    _write_jsonl(session_dir / "refined_steps.jsonl", refined_steps)
    return session_dir


def test_ground_session_emits_skill_md_and_meta(tmp_path):
    session_dir = _imported_session(tmp_path, [_refined("r-0", ["s-0"]), _refined("r-1", ["s-1"])])

    skill_path = ground_session(session_dir)

    assert skill_path == session_dir / "skill" / "SKILL.md"
    assert skill_path.is_file()
    doc = skill_path.read_text()
    assert doc.startswith("---\n")
    assert "## Verification" in doc

    meta = load_ground_meta(str(session_dir / "ground_meta.json"))
    assert meta.input_step_count == 2
    assert meta.output_step_count == 2
    assert meta.scripted_step_count == 0
    assert meta.ui_replay_step_count == 2


def test_ground_session_is_one_to_one_with_refined_steps(tmp_path):
    """Unlike Stage 4, Stage 5 never merges or drops: every refined step
    becomes exactly one grounded step."""
    refined = [_refined(f"r-{i}", [f"s-{i}"]) for i in range(5)]
    session_dir = _imported_session(tmp_path, refined)

    ground_session(session_dir)

    grounded = load_grounded_steps(str(session_dir / "grounded_steps.jsonl"))
    assert len(grounded) == 5
    assert [g.step_id for g in grounded] == [f"grounded-{i:04d}" for i in range(5)]
    assert [g.source_step_ids for g in grounded] == [[f"s-{i}"] for i in range(5)]


def test_ground_session_writes_executable_scripts(tmp_path):
    refined = [
        _refined("r-0", ["s-0"], action_type="run_command", parameters={"command": "echo hi"}),
        _refined("r-1", ["s-1"]),
    ]
    session_dir = _imported_session(tmp_path, refined)

    ground_session(session_dir)

    scripts = sorted((session_dir / "skill" / "scripts").iterdir())
    assert len(scripts) == 1
    assert scripts[0].name.startswith("step-0001-")
    assert "echo hi" in scripts[0].read_text()
    assert scripts[0].stat().st_mode & 0o111  # executable

    meta = load_ground_meta(str(session_dir / "ground_meta.json"))
    assert meta.scripted_step_count == 1
    assert meta.ui_replay_step_count == 1


def test_no_scripts_directory_when_nothing_is_scriptable(tmp_path):
    session_dir = _imported_session(tmp_path, [_refined("r-0", ["s-0"])])
    ground_session(session_dir)
    assert not (session_dir / "skill" / "scripts").exists()


def test_imported_session_recovers_no_ax_evidence(tmp_path):
    """The known imported-session gap, showing up where it should: no event
    log means no accessibility identifier and no coordinates."""
    session_dir = _imported_session(tmp_path, [_refined("r-0", ["s-0"])])
    ground_session(session_dir)

    grounded = load_grounded_steps(str(session_dir / "grounded_steps.jsonl"))
    assert grounded[0].ui_target is not None
    assert grounded[0].ui_target.has_ax_identifier is False
    assert grounded[0].ui_target.fallback_x is None
    assert load_ground_meta(str(session_dir / "ground_meta.json")).ax_grounded_step_count == 0


def test_live_capture_session_recovers_ax_identifier_through_the_stage_chain(tmp_path):
    """The full traceability walk: refined step -> Stage 3 step -> segment ->
    click event -> accessibility element captured in Stage 1."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest = LiveCaptureManifest(
        session_id="sess-live",
        epoch0_utc="2026-01-01T00:00:00+00:00",
        fps=30.0,
        video_width=1920,
        video_height=1080,
        video_path="screen.mp4",
        events_path="events.jsonl",
        has_events=True,
        duration_ms=1000.0,
        os_version="test",
        screen_width=1920,
        screen_height=1080,
        backing_scale_factor=1.0,
        avfoundation_device_index=0,
        avfoundation_device_name="test",
    )
    (session_dir / "session.json").write_text(manifest.model_dump_json())

    click = MouseClickEvent(
        t_ms=0.0,
        x=42.0,
        y=84.0,
        button="left",
        pressed=True,
        ax_element=AXElement(role="AXButton", title="Save"),
    )
    _write_jsonl(
        session_dir / "segments.jsonl",
        [
            Segment(
                segment_id="seg-0",
                source="event_log",
                action_hint="click",
                start_ms=0.0,
                end_ms=10.0,
                frame_start=0,
                frame_end=1,
                events=[click],
            )
        ],
    )
    _write_jsonl(
        session_dir / "steps.jsonl",
        [
            Step(
                step_id="s-0",
                segment_id="seg-0",
                source="event_log",
                intent="save the file",
                action_type="click",
                target_description="the Save button",
            )
        ],
    )
    _write_jsonl(
        session_dir / "refined_steps.jsonl",
        [_refined("r-0", ["s-0"], source="event_log", target_description="the Save button")],
    )

    ground_session(session_dir)

    grounded = load_grounded_steps(str(session_dir / "grounded_steps.jsonl"))
    target = grounded[0].ui_target
    assert target is not None
    assert target.ax_role == "AXButton"
    assert target.ax_title == "Save"
    assert (target.fallback_x, target.fallback_y) == (42.0, 84.0)
    assert load_ground_meta(str(session_dir / "ground_meta.json")).ax_grounded_step_count == 1
    assert "**Accessibility identifier:**" in (session_dir / "skill" / "SKILL.md").read_text()


def test_flagged_steps_are_counted_and_marked(tmp_path):
    refined = [
        _refined("r-0", ["s-0"], needs_review=True, review_reason="chunk failed to parse: x"),
        _refined("r-1", ["s-1"], needs_review=True, review_reason="genuinely unsure"),
        _refined("r-2", ["s-2"]),
    ]
    session_dir = _imported_session(tmp_path, refined)
    ground_session(session_dir)

    meta = load_ground_meta(str(session_dir / "ground_meta.json"))
    assert meta.needs_review_count == 2
    assert meta.unrefined_passthrough_count == 1
    assert "**needs-review**" in (session_dir / "skill" / "SKILL.md").read_text()


def test_name_and_description_overrides_reach_the_frontmatter(tmp_path):
    session_dir = _imported_session(tmp_path, [_refined("r-0", ["s-0"])])
    ground_session(session_dir, skill_name="my-skill", description="Does a thing.")

    doc = (session_dir / "skill" / "SKILL.md").read_text()
    assert "name: 'my-skill'" in doc
    assert "description: 'Does a thing.'" in doc
    assert load_ground_meta(str(session_dir / "ground_meta.json")).skill_name == "my-skill"


def test_grounded_steps_jsonl_is_valid_json_per_line(tmp_path):
    session_dir = _imported_session(tmp_path, [_refined("r-0", ["s-0"]), _refined("r-1", ["s-1"])])
    ground_session(session_dir)

    lines = (session_dir / "grounded_steps.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["grounding"] in ("script", "ui_replay")
