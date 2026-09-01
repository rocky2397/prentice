"""Tests for Stage 5's scripted-vs-UI-replay decision.

The negative cases matter more than the positive ones here: the whole point
of grounding.py is that it refuses to invent a command from prose, so most
of these assert that a plausible-looking step does *not* become a script.
"""

from __future__ import annotations

import pytest

from prentice.ground.grounding import classify_confidence, derive_script, ground_step
from prentice.ground.provenance import StepEvidence
from prentice.refine.schema import RefinedStep


def _refined(**overrides) -> RefinedStep:
    base = {
        "step_id": "refined-0000",
        "source_step_ids": ["s-0000"],
        "source": "inferred",
        "intent": "do a thing",
        "action_type": "click",
        "target_description": "the YouTube homepage",
        "parameters": {},
    }
    base.update(overrides)
    return RefinedStep(**base)


# --- scriptable: only from concrete parameter values ---


def test_explicit_command_parameter_is_used_verbatim():
    step = _refined(action_type="run_command", parameters={"command": "git status"})
    assert derive_script(step) == "git status"


def test_explicit_url_becomes_an_open_command():
    step = _refined(action_type="navigate", parameters={"url": "https://example.com/a?b=1"})
    assert derive_script(step) == "open 'https://example.com/a?b=1'"


def test_explicit_url_is_found_under_any_key_name():
    """A full scheme-bearing URL is unambiguous evidence whatever it's called."""
    step = _refined(action_type="navigate", parameters={"destination": "https://example.com"})
    # shlex.quote leaves a value alone when it needs no quoting
    assert derive_script(step) == "open https://example.com"


def test_bare_domain_under_a_url_key_gets_a_scheme():
    step = _refined(action_type="navigate", parameters={"url": "youtube.com"})
    assert derive_script(step) == "open https://youtube.com"


def test_application_parameter_becomes_open_dash_a():
    step = _refined(action_type="navigate", parameters={"application": "Terminal.app"})
    assert derive_script(step) == "open -a Terminal.app"


def test_script_values_are_shell_quoted():
    step = _refined(parameters={"application": "Visual Studio Code.app"})
    assert derive_script(step) == "open -a 'Visual Studio Code.app'"


# --- NOT scriptable: the core safety property ---


def test_prose_target_description_is_never_evidence():
    """The single most important negative case: 'the YouTube homepage' must
    not become `open https://youtube.com`."""
    step = _refined(action_type="navigate", target_description="the YouTube homepage")
    assert derive_script(step) is None


def test_url_key_holding_prose_is_not_scripted():
    step = _refined(action_type="navigate", parameters={"url": "the address bar"})
    assert derive_script(step) is None


def test_empty_parameters_are_not_scriptable():
    assert derive_script(_refined(action_type="run_command")) is None


def test_non_string_parameter_values_are_ignored():
    step = _refined(parameters={"command": 42, "url": None})
    assert derive_script(step) is None


def test_action_type_alone_never_makes_a_step_scriptable():
    for action_type in ("click", "type", "scroll", "drag", "navigate", "run_command"):
        assert derive_script(_refined(action_type=action_type)) is None


# --- confidence classification ---


def test_classify_confidence_unflagged():
    assert classify_confidence(_refined()) == "ok"


def test_classify_confidence_chunk_parse_failure_is_passthrough():
    step = _refined(needs_review=True, review_reason="chunk failed to parse: bad JSON")
    assert classify_confidence(step) == "unrefined_passthrough"


def test_classify_confidence_omission_is_passthrough():
    step = _refined(
        needs_review=True,
        review_reason="model did not include this step in its refine response — passed through unrefined",
    )
    assert classify_confidence(step) == "unrefined_passthrough"


def test_classify_confidence_genuine_model_flag():
    step = _refined(needs_review=True, review_reason="unsure whether these are duplicates")
    assert classify_confidence(step) == "model_flagged"


def test_classify_confidence_over_merge_override_is_not_passthrough():
    """Stage 4's deterministic over-merge flag means the step *was* refined
    and judged bad — a different signal from never having been refined."""
    step = _refined(needs_review=True, review_reason="auto-flagged: this group absorbed 88 input steps")
    assert classify_confidence(step) == "model_flagged"


def test_passthrough_markers_still_match_stage_4s_real_wording():
    """Guards the string coupling to refine/llm.py. If Stage 4's passthrough
    wording changes, this fails loudly here rather than silently
    reclassifying every passthrough step as a genuine model judgement."""
    pytest.importorskip("mlx_vlm")
    from prentice.interpret.schema import Step
    from prentice.refine.llm import _passthrough_step

    step = Step(
        step_id="s-0000",
        segment_id="seg-0000",
        source="inferred",
        intent="x",
        action_type="click",
        target_description="y",
    )
    omitted = _passthrough_step(step, "model did not include this step in its refine response — passed through unrefined")
    chunk_failed = _passthrough_step(step, "chunk failed to parse: Expecting value")
    assert classify_confidence(omitted) == "unrefined_passthrough"
    assert classify_confidence(chunk_failed) == "unrefined_passthrough"


# --- ground_step end to end ---


def test_ground_step_scripted_carries_no_ui_target():
    grounded = ground_step(_refined(parameters={"command": "ls"}))
    assert grounded.grounding == "script"
    assert grounded.script == "ls"
    assert grounded.ui_target is None


def test_ground_step_ui_replay_uses_description_as_primary_target():
    grounded = ground_step(_refined(target_description="the Save button in the toolbar"))
    assert grounded.grounding == "ui_replay"
    assert grounded.script is None
    assert grounded.ui_target is not None
    assert grounded.ui_target.description == "the Save button in the toolbar"


def test_ground_step_folds_in_capture_evidence():
    evidence = StepEvidence(ax_role="AXButton", ax_title="Save", x=12.0, y=34.0)
    grounded = ground_step(_refined(), evidence)
    assert grounded.ui_target is not None
    assert grounded.ui_target.ax_title == "Save"
    assert grounded.ui_target.fallback_x == 12.0


def test_unscriptable_run_command_is_flagged():
    """The one action type that should always have been scriptable but
    wasn't — worth a human look regardless of Stage 4's verdict."""
    grounded = ground_step(_refined(action_type="run_command"))
    assert grounded.grounding == "ui_replay"
    assert grounded.needs_review is True
    assert "run_command" in (grounded.review_reason or "")


def test_unscriptable_run_command_keeps_an_existing_review_reason():
    grounded = ground_step(
        _refined(action_type="run_command", needs_review=True, review_reason="chunk failed to parse: x")
    )
    assert grounded.review_reason == "chunk failed to parse: x"
    assert grounded.confidence == "unrefined_passthrough"


def test_ground_step_preserves_traceability_and_source():
    step = _refined(source="mixed", source_step_ids=["s-0000", "s-0001"], variable_parameters=["filename"])
    grounded = ground_step(step, step_id="grounded-0007")
    assert grounded.step_id == "grounded-0007"
    assert grounded.source == "mixed"
    assert grounded.source_step_ids == ["s-0000", "s-0001"]
    assert grounded.variable_parameters == ["filename"]
