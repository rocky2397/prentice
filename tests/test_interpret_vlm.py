"""Fast, real (non-mocked) tests of the pure prompt-building and
response-parsing logic in interpret/vlm.py — no model download needed for
these. The actual end-to-end VLM call is a separate, opt-in test below
(PRENTICE_TEST_VLM=1) since it downloads a large checkpoint and runs real
inference.
"""

from __future__ import annotations

import os
import shutil

import pytest

from prentice.interpret.schema import Step
from prentice.interpret.vlm import (
    DEFAULT_MODEL_NAME,
    InterpretParams,
    _build_prompt,
    _parse_step_response,
)
from prentice.segment.schema import Segment


def _inferred_segment() -> Segment:
    return Segment(
        segment_id="s-0000",
        source="inferred",
        action_hint="scene_change",
        start_ms=0.0,
        end_ms=1000.0,
        frame_start=0,
        frame_end=10,
    )


def _event_log_segment() -> Segment:
    from prentice.capture.schema import MouseClickEvent

    return Segment(
        segment_id="s-0000",
        source="event_log",
        action_hint="click",
        start_ms=0.0,
        end_ms=50.0,
        frame_start=0,
        frame_end=1,
        events=[MouseClickEvent(t_ms=0.0, x=10.0, y=20.0, button="Button.left", pressed=True)],
    )


def test_build_prompt_no_events_says_infer_from_screenshots():
    prompt = _build_prompt(_inferred_segment(), prior_steps=[])
    assert "No input event log exists" in prompt
    assert "This is the first step" in prompt


def test_build_prompt_includes_event_json():
    prompt = _build_prompt(_event_log_segment(), prior_steps=[])
    assert "mouse_click" in prompt
    assert '"x": 10.0' in prompt


def test_build_prompt_includes_bounded_prior_steps():
    prior = [
        Step(
            step_id=f"s-{i:04d}",
            segment_id=f"s-{i:04d}",
            source="event_log",
            intent=f"step {i}",
            action_type="click",
            target_description="a button",
        )
        for i in range(2)
    ]
    prompt = _build_prompt(_inferred_segment(), prior_steps=prior)
    assert "step 0" in prompt
    assert "step 1" in prompt


def test_parse_step_response_plain_json():
    text = (
        '{"intent": "save the file", "action_type": "click", '
        '"target_description": "the Save button", "parameters": {}}'
    )
    step = _parse_step_response(text, _inferred_segment(), "s-0000")
    assert step.action_type == "click"
    assert step.source == "inferred"
    assert step.segment_id == "s-0000"


def test_parse_step_response_strips_code_fence():
    text = (
        "```json\n"
        '{"intent": "type text", "action_type": "type", '
        '"target_description": "the search box", "parameters": {"text": "hello"}}\n'
        "```"
    )
    step = _parse_step_response(text, _inferred_segment(), "s-0000")
    assert step.action_type == "type"
    assert step.parameters == {"text": "hello"}


def test_parse_step_response_invalid_json_raises():
    with pytest.raises(ValueError, match="did not return valid JSON"):
        _parse_step_response("not json at all", _inferred_segment(), "s-0000")


def test_parse_step_response_unrecognized_action_type_raises():
    text = '{"intent": "x", "action_type": "teleport", "target_description": "y"}'
    with pytest.raises(ValueError, match="unrecognized action_type"):
        _parse_step_response(text, _inferred_segment(), "s-0000")


def test_parse_step_response_non_dict_parameters_falls_back_to_empty():
    # regression: a real run had the model return "parameters": "wi" (a bare
    # string, not an object) — this must not crash the whole step
    text = '{"intent": "x", "action_type": "click", "target_description": "y", "parameters": "wi"}'
    step = _parse_step_response(text, _inferred_segment(), "s-0000")
    assert step.parameters == {}


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not os.environ.get("PRENTICE_TEST_VLM"),
    reason="requires ffmpeg and PRENTICE_TEST_VLM=1 (downloads a large VLM checkpoint and runs real inference)",
)
def test_interpret_segment_real_inference(tmp_path):
    from _video_helpers import make_two_scene_test_video
    from prentice.interpret.keyframes import extract_keyframes
    from prentice.interpret.vlm import interpret_segment

    video_path = tmp_path / "two_scene.mp4"
    make_two_scene_test_video(video_path, scene_duration=1.5, fps=10)
    segment = Segment(
        segment_id="real-0000",
        source="inferred",
        action_hint="scene_change",
        start_ms=0.0,
        end_ms=1500.0,
        frame_start=0,
        frame_end=14,
    )
    keyframes = extract_keyframes(video_path, [segment], tmp_path / "keyframes")

    step = interpret_segment(
        segment,
        keyframes["real-0000"],
        prior_steps=[],
        step_id="real-0000",
        params=InterpretParams(model_name=DEFAULT_MODEL_NAME),
    )

    assert step.segment_id == "real-0000"
    assert step.source == "inferred"
    assert step.action_type in {"click", "type", "scroll", "drag", "navigate", "run_command"}
    assert step.intent
