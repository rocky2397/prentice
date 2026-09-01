"""Tests for the SKILL.md / generated-script emission."""

from __future__ import annotations

from prentice.ground.schema import GroundedStep, UITarget
from prentice.ground.skill_md import (
    derive_skill_name,
    render_script,
    render_skill_md,
    required_inputs,
    script_filename,
)


def _step(**overrides) -> GroundedStep:
    base = {
        "step_id": "grounded-0000",
        "source_step_ids": ["refined-0000"],
        "source": "inferred",
        "intent": "click the Save button",
        "action_type": "click",
        "grounding": "ui_replay",
        "confidence": "ok",
        "ui_target": UITarget(description="the Save button in the toolbar"),
    }
    base.update(overrides)
    return GroundedStep(**base)


def _render(steps: list[GroundedStep]) -> str:
    return render_skill_md(steps, skill_name="demo", description="A demo.", session_id="sess-1")


def test_frontmatter_has_the_three_required_keys():
    doc = _render([_step()])
    assert doc.startswith("---\n")
    assert "name: 'demo'" in doc
    assert "description: 'A demo.'" in doc
    assert "required_inputs: []" in doc


def test_frontmatter_escapes_single_quotes():
    doc = render_skill_md(
        [_step()], skill_name="demo", description="Opens the user's 'Documents' folder.", session_id="s"
    )
    assert "description: 'Opens the user''s ''Documents'' folder.'" in doc


def test_frontmatter_lists_required_inputs():
    doc = _render([_step(variable_parameters=["filename", "folder"])])
    assert "  - 'filename'" in doc
    assert "  - 'folder'" in doc


def test_required_inputs_dedupes_and_preserves_first_seen_order():
    steps = [
        _step(variable_parameters=["b", "a"]),
        _step(variable_parameters=["a", "c"]),
    ]
    assert required_inputs(steps) == ["b", "a", "c"]


def test_ui_replay_step_renders_its_semantic_target():
    doc = _render([_step()])
    assert "the Save button in the toolbar" in doc
    assert "**Replay by locating:**" in doc


def test_ax_identifier_is_rendered_when_present():
    doc = _render([_step(ui_target=UITarget(description="Save", ax_role="AXButton", ax_title="Save"))])
    assert "**Accessibility identifier:**" in doc
    assert "role=" in doc


def test_coordinates_are_rendered_as_a_last_resort_only():
    doc = _render([_step(ui_target=UITarget(description="Save", fallback_x=12.4, fallback_y=99.6))])
    assert "**Fallback coordinates:** (12, 100)" in doc
    assert "last resort only" in doc


def test_coordinates_are_omitted_when_absent():
    assert "Fallback coordinates" not in _render([_step()])


def test_scripted_step_renders_the_command_and_its_script_path():
    doc = _render(
        [
            _step(
                grounding="script",
                action_type="run_command",
                ui_target=None,
                script="open -a Terminal.app",
                script_path="step-0001-open-terminal.sh",
            )
        ]
    )
    assert "scripts/step-0001-open-terminal.sh" in doc
    assert "open -a Terminal.app" in doc


def test_flagged_steps_are_marked_not_silently_included():
    doc = _render([_step(needs_review=True, confidence="model_flagged", review_reason="unsure")])
    assert "**needs-review**" in doc
    assert "**Review reason:** unsure" in doc


def test_vision_only_steps_are_marked():
    assert "_vision-only_" in _render([_step(source="inferred")])
    assert "_vision-only_" not in _render([_step(source="event_log")])


def test_reliability_summary_reports_proportions():
    steps = [_step(), _step(needs_review=True, confidence="unrefined_passthrough")]
    doc = _render(steps)
    assert "Flagged `needs-review` | 1 of 2 (50%)" in doc
    assert "Never refined (Stage 4 passthrough) | 1 of 2 (50%)" in doc


def test_every_step_carries_traceability():
    doc = _render([_step(source_step_ids=["refined-0000", "refined-0001"])])
    assert "**Traces to:** `refined-0000`, `refined-0001`" in doc


def test_verification_section_is_always_present():
    doc = _render([_step()])
    assert "## Verification" in doc
    assert "click the Save button" in doc


def test_empty_step_list_still_renders_a_valid_document():
    doc = _render([])
    assert doc.startswith("---\n")
    assert "## Verification" in doc
    assert "No steps were produced" in doc


def test_derive_skill_name_picks_distinctive_words():
    steps = [
        _step(ui_target=UITarget(description="the Recents item in the Finder sidebar")),
        _step(ui_target=UITarget(description="the Documents folder in the Finder sidebar")),
        _step(ui_target=UITarget(description="the AirDrop item in the Finder sidebar")),
    ]
    name = derive_skill_name(steps, "sess-1")
    assert name.endswith("-workflow")
    assert "finder" in name
    assert "the" not in name.split("-")


def test_derive_skill_name_falls_back_to_the_session_id():
    steps = [_step(ui_target=UITarget(description="the the the"), intent="")]
    assert derive_skill_name(steps, "abcdef123456") == "workflow-abcdef12"


def test_script_filename_is_slugified_and_numbered():
    name = script_filename(_step(intent="Open the Terminal application!"), 7)
    assert name.startswith("step-0007-")
    assert name.endswith(".sh")
    assert " " not in name


def test_render_script_is_a_safe_standalone_shell_script():
    script = render_script(
        _step(grounding="script", ui_target=None, script="open -a Terminal.app", script_path="x.sh")
    )
    assert script.startswith("#!/bin/sh\n")
    assert "set -eu" in script
    assert "open -a Terminal.app" in script
    assert "refined-0000" in script  # traceability back to the Stage 3/4 step
