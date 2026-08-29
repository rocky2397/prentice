from prentice.interpret.schema import InterpretRunMeta, Step, StepAdapter


def test_step_roundtrip():
    step = Step(
        step_id="s-0000",
        segment_id="s-0000",
        source="event_log",
        intent="save the current document",
        action_type="click",
        target_description="the Save button in the toolbar",
        parameters={},
    )
    parsed = StepAdapter.validate_json(step.model_dump_json())
    assert parsed == step


def test_step_defaults_empty_parameters():
    step = Step(
        step_id="s-0000",
        segment_id="s-0000",
        source="inferred",
        intent="switch to a different window",
        action_type="navigate",
        target_description="the Finder window",
    )
    assert step.parameters == {}


def test_interpret_run_meta_roundtrip():
    meta = InterpretRunMeta(
        session_id="s",
        step_count=3,
        model_name="mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit",
        context_window=5,
        max_tokens=512,
        temperature=0.0,
    )
    parsed = InterpretRunMeta.model_validate_json(meta.model_dump_json())
    assert parsed == meta
