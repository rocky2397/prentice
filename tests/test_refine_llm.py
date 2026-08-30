"""Fast, real (non-mocked) tests of the pure prompt-building and
response-parsing logic in refine/llm.py — no model call needed for these.
The actual end-to-end LLM call is a separate, opt-in test below
(PRENTICE_TEST_VLM=1) — it reuses the same checkpoint Stage 3's test does,
so no extra download, but it's still real inference.
"""

from __future__ import annotations

import os

import pytest

from prentice.interpret.schema import Step
from prentice.refine.llm import _build_refine_prompt, _build_refined_step, _parse_refine_response


def _step(step_id: str, source: str = "inferred", action_type: str = "click", intent: str = "x") -> Step:
    return Step(
        step_id=step_id,
        segment_id=step_id,
        source=source,
        intent=intent,
        action_type=action_type,
        target_description="a button",
    )


def test_build_refine_prompt_includes_all_step_ids():
    steps = [_step("s-0000"), _step("s-0001")]
    prompt = _build_refine_prompt(steps)
    assert "s-0000" in prompt
    assert "s-0001" in prompt


def test_build_refined_step_resolves_source_from_inputs():
    input_by_id = {"s-0000": _step("s-0000", source="event_log"), "s-0001": _step("s-0001", source="event_log")}
    item = {
        "source_step_ids": ["s-0000", "s-0001"],
        "action_type": "click",
        "intent": "click the button",
        "target_description": "the button",
    }
    step = _build_refined_step(item, input_by_id, set(), "refined-0000")
    assert step.source == "event_log"
    assert step.source_step_ids == ["s-0000", "s-0001"]


def test_build_refined_step_mixed_source():
    input_by_id = {"s-0000": _step("s-0000", source="event_log"), "s-0001": _step("s-0001", source="inferred")}
    item = {
        "source_step_ids": ["s-0000", "s-0001"],
        "action_type": "click",
        "intent": "x",
        "target_description": "y",
    }
    step = _build_refined_step(item, input_by_id, set(), "refined-0000")
    assert step.source == "mixed"


def test_build_refined_step_drops_unknown_step_ids_but_keeps_valid_ones():
    input_by_id = {"s-0000": _step("s-0000")}
    item = {
        "source_step_ids": ["s-0000", "s-9999"],
        "action_type": "click",
        "intent": "x",
        "target_description": "y",
    }
    step = _build_refined_step(item, input_by_id, set(), "refined-0000")
    assert step.source_step_ids == ["s-0000"]


def test_build_refined_step_all_unknown_step_ids_raises():
    input_by_id = {"s-0000": _step("s-0000")}
    item = {"source_step_ids": ["s-9999"], "action_type": "click", "intent": "x", "target_description": "y"}
    with pytest.raises(ValueError, match="match no unclaimed input step"):
        _build_refined_step(item, input_by_id, set(), "refined-0000")


def test_build_refined_step_unrecognized_action_type_raises():
    input_by_id = {"s-0000": _step("s-0000")}
    item = {"source_step_ids": ["s-0000"], "action_type": "teleport", "intent": "x", "target_description": "y"}
    with pytest.raises(ValueError, match="unrecognized action_type"):
        _build_refined_step(item, input_by_id, set(), "refined-0000")


def test_build_refined_step_oversized_group_auto_flags_needs_review():
    # regression: a real run produced one group absorbing 88 of 126 steps —
    # this must be forced into review regardless of what the model claimed
    from prentice.refine.llm import MAX_REASONABLE_GROUP_SIZE

    ids = [f"s-{i:04d}" for i in range(MAX_REASONABLE_GROUP_SIZE + 1)]
    input_by_id = {sid: _step(sid) for sid in ids}
    item = {
        "source_step_ids": ids,
        "action_type": "click",
        "intent": "x",
        "target_description": "y",
        "needs_review": False,
    }
    step = _build_refined_step(item, input_by_id, set(), "refined-0000")
    assert step.needs_review is True
    assert "over-merge" in step.review_reason


def test_build_refined_step_needs_review_carries_reason():
    input_by_id = {"s-0000": _step("s-0000")}
    item = {
        "source_step_ids": ["s-0000"],
        "action_type": "click",
        "intent": "x",
        "target_description": "y",
        "needs_review": True,
        "review_reason": "ambiguous target",
    }
    step = _build_refined_step(item, input_by_id, set(), "refined-0000")
    assert step.needs_review is True
    assert step.review_reason == "ambiguous target"


def test_parse_refine_response_later_group_cannot_reclaim_already_claimed_step():
    # regression: a real run produced a legitimate set of well-sized groups,
    # then one final "catch-all" group re-listing 123 of the same 126
    # steps already correctly covered — every one of those must be stripped
    # from the later group, not double-counted
    steps = [_step("s-0000"), _step("s-0001"), _step("s-0002")]
    text = (
        '[{"source_step_ids": ["s-0000"], "action_type": "click", "intent": "a", "target_description": "x"},'
        ' {"source_step_ids": ["s-0001"], "action_type": "click", "intent": "b", "target_description": "y"},'
        ' {"source_step_ids": ["s-0000", "s-0001", "s-0002"], "action_type": "click", "intent": "catch-all",'
        ' "target_description": "z"}]'
    )
    refined = _parse_refine_response(text, steps)
    assert len(refined) == 3
    assert refined[2].source_step_ids == ["s-0002"]


def test_parse_refine_response_omitted_step_is_passed_through_not_lost():
    # regression: a real run silently dropped genuine distinct actions
    # (Save click, Run click) just by never mentioning their step_ids —
    # indistinguishable from deliberate noise-dropping. A step the model
    # never mentions at all must still survive, flagged for review.
    steps = [_step("s-0000"), _step("s-0001", intent="click Save"), _step("s-0002")]
    text = (
        '[{"source_step_ids": ["s-0000"], "action_type": "click", "intent": "a", "target_description": "x"},'
        ' {"source_step_ids": ["s-0002"], "action_type": "click", "intent": "c", "target_description": "z"}]'
    )
    refined = _parse_refine_response(text, steps)
    assert len(refined) == 3
    passthrough = next(r for r in refined if r.source_step_ids == ["s-0001"])
    assert passthrough.needs_review is True
    assert passthrough.intent == "click Save"
    # chronological order preserved despite being appended after parsing
    assert [r.source_step_ids[0] for r in refined] == ["s-0000", "s-0001", "s-0002"]


def test_parse_refine_response_rejected_group_does_not_falsely_claim_its_ids():
    # regression: claiming ids before validating action_type meant a group
    # that got rejected (bad action_type) still marked its ids as claimed,
    # so they were never recovered by the no-silent-loss passthrough either
    steps = [_step("s-0000")]
    text = '[{"source_step_ids": ["s-0000"], "action_type": "teleport", "intent": "a", "target_description": "x"}]'
    refined = _parse_refine_response(text, steps)
    assert len(refined) == 1
    assert refined[0].source_step_ids == ["s-0000"]
    assert refined[0].needs_review is True


def test_parse_refine_response_group_left_with_no_unclaimed_ids_is_dropped():
    steps = [_step("s-0000")]
    text = (
        '[{"source_step_ids": ["s-0000"], "action_type": "click", "intent": "a", "target_description": "x"},'
        ' {"source_step_ids": ["s-0000"], "action_type": "click", "intent": "duplicate", "target_description": "y"}]'
    )
    refined = _parse_refine_response(text, steps)
    assert len(refined) == 1
    assert refined[0].intent == "a"


def test_parse_refine_response_skips_one_bad_element_keeps_rest():
    steps = [_step("s-0000"), _step("s-0001")]
    text = (
        '[{"source_step_ids": ["s-0000"], "action_type": "click", "intent": "a", "target_description": "x"},'
        ' "not an object",'
        ' {"source_step_ids": ["s-0001"], "action_type": "click", "intent": "b", "target_description": "y"}]'
    )
    refined = _parse_refine_response(text, steps)
    assert len(refined) == 2
    assert [r.intent for r in refined] == ["a", "b"]


def test_parse_refine_response_invalid_json_raises():
    with pytest.raises(ValueError, match="did not return valid JSON"):
        _parse_refine_response("not json at all", [_step("s-0000")])


def test_parse_refine_response_non_array_raises():
    with pytest.raises(TypeError, match="did not return a JSON array"):
        _parse_refine_response('{"not": "an array"}', [_step("s-0000")])


@pytest.mark.skipif(
    not os.environ.get("PRENTICE_TEST_VLM"),
    reason="requires PRENTICE_TEST_VLM=1 — reuses the Stage 3 checkpoint (no extra download) but runs real inference",
)
def test_refine_session_steps_real_inference():
    from prentice.refine.llm import RefineParams, refine_session_steps

    steps = [
        _step("s-0000", intent="click the Save button in the toolbar", action_type="click"),
        _step("s-0001", intent="click the Save button in the toolbar", action_type="click"),
        _step("s-0002", intent="type the filename report.txt", action_type="type"),
    ]
    refined = refine_session_steps(steps, params=RefineParams())

    assert len(refined) >= 1
    all_absorbed = {sid for r in refined for sid in r.source_step_ids}
    assert all_absorbed <= {"s-0000", "s-0001", "s-0002"}
    for r in refined:
        assert r.action_type in {"click", "type", "scroll", "drag", "navigate", "run_command"}
