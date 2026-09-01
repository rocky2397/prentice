from prentice.ground.schema import GroundedStep, GroundedStepAdapter, GroundRunMeta, UITarget


def _grounded(**overrides) -> GroundedStep:
    base = {
        "step_id": "grounded-0000",
        "source_step_ids": ["refined-0000"],
        "source": "inferred",
        "intent": "open the terminal",
        "action_type": "run_command",
        "grounding": "script",
        "confidence": "ok",
        "script": "open -a Terminal.app",
        "script_path": "step-0001-open-the-terminal.sh",
    }
    base.update(overrides)
    return GroundedStep(**base)


def test_grounded_step_roundtrip():
    step = _grounded()
    assert GroundedStepAdapter.validate_json(step.model_dump_json()) == step


def test_grounded_step_defaults():
    step = _grounded()
    assert step.needs_review is False
    assert step.review_reason is None
    assert step.parameters == {}
    assert step.variable_parameters == []
    assert step.ui_target is None


def test_ui_replay_step_roundtrip_preserves_target():
    step = _grounded(
        grounding="ui_replay",
        action_type="click",
        script=None,
        script_path=None,
        ui_target=UITarget(description="the Save button", ax_role="AXButton", ax_title="Save"),
    )
    parsed = GroundedStepAdapter.validate_json(step.model_dump_json())
    assert parsed == step
    assert parsed.ui_target is not None
    assert parsed.ui_target.ax_title == "Save"


def test_ui_target_has_ax_identifier():
    assert UITarget(description="x").has_ax_identifier is False
    assert UITarget(description="x", ax_role="AXButton").has_ax_identifier is True
    assert UITarget(description="x", ax_value="hello").has_ax_identifier is True


def test_ui_target_coordinates_are_optional_and_independent_of_ax():
    """Coordinates are a last-resort hint, not part of the AX identifier —
    a target with only coordinates must not claim to be AX-grounded."""
    target = UITarget(description="x", fallback_x=100.0, fallback_y=200.0)
    assert target.has_ax_identifier is False
    assert target.fallback_x == 100.0


def test_ground_run_meta_roundtrip():
    meta = GroundRunMeta(
        session_id="s",
        skill_name="finder-sidebar-workflow",
        input_step_count=10,
        output_step_count=10,
        scripted_step_count=1,
        ui_replay_step_count=9,
        ax_grounded_step_count=0,
        needs_review_count=8,
        unrefined_passthrough_count=7,
    )
    assert GroundRunMeta.model_validate_json(meta.model_dump_json()) == meta
