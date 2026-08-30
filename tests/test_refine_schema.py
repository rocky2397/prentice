from prentice.refine.schema import RefinedStep, RefinedStepAdapter, RefineRunMeta


def test_refined_step_roundtrip():
    step = RefinedStep(
        step_id="refined-0000",
        source_step_ids=["s-0000", "s-0001", "s-0002"],
        source="inferred",
        intent="navigate to youtube",
        action_type="navigate",
        target_description="the address bar",
        parameters={"url": "youtube.com"},
        variable_parameters=["url"],
    )
    parsed = RefinedStepAdapter.validate_json(step.model_dump_json())
    assert parsed == step


def test_refined_step_defaults():
    step = RefinedStep(
        step_id="refined-0000",
        source_step_ids=["s-0000"],
        source="event_log",
        intent="x",
        action_type="click",
        target_description="y",
    )
    assert step.parameters == {}
    assert step.variable_parameters == []
    assert step.needs_review is False
    assert step.review_reason is None


def test_refined_step_mixed_source():
    step = RefinedStep(
        step_id="refined-0000",
        source_step_ids=["s-0000", "s-0001"],
        source="mixed",
        intent="x",
        action_type="click",
        target_description="y",
        needs_review=True,
        review_reason="ambiguous merge",
    )
    assert step.source == "mixed"
    assert step.needs_review is True


def test_refine_run_meta_roundtrip():
    meta = RefineRunMeta(
        session_id="s",
        input_step_count=10,
        output_step_count=6,
        model_name="mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit",
        max_tokens=4096,
        temperature=0.0,
        repetition_penalty=1.3,
        repetition_context_size=200,
        chunk_size=20,
    )
    parsed = RefineRunMeta.model_validate_json(meta.model_dump_json())
    assert parsed == meta
