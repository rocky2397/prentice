"""Local-LLM refine pass for Stage 4, via mlx-vlm's text-only path: reasons
over Stage 3's step sequence in bounded chunks (not one step at a time like
Stage 3, and not the whole session in one call either — merging duplicates
and reconciling parameters need to see more than one step at a time, but
real testing found asking for a whole long session in a single generation
degrades badly: malformed JSON, garbled English, and repetition loops
appearing partway through the response, worse the longer the output runs)
and asks for a consolidated sequence back per chunk, per ARCHITECTURE.md's
Stage 4 spec: merge fragmented/duplicate steps, drop noise, distinguish
fixed vs. variable parameter values, flag steps it can't resolve
confidently.

Reuses the same Qwen3-VL checkpoint Stage 3 already downloaded and cached —
called text-only (no images), which satisfies ARCHITECTURE.md's "at least a
separate call" without a second multi-GB model download. A dedicated
text-reasoning model (e.g. via mlx-lm) is the documented next lever if
refine quality demands it, not the starting default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template

from ..interpret.schema import VALID_ACTION_TYPES, Step
from ..interpret.vlm import DEFAULT_MODEL_NAME
from ..llm_json import strip_code_fence
from ..vlm_model import load_vlm_model
from .schema import RefinedStep

DEFAULT_MAX_TOKENS = 32768  # model's context is 262k tokens, plenty of headroom — a thorough,
# fine-grained refine of a 126-step session genuinely needs more than a few thousand tokens
DEFAULT_TEMPERATURE = 0.0
# Greedy (temperature=0) decoding can fall into a degenerate loop, repeating
# the same output block verbatim until it hits max_tokens — observed in real
# testing (a "Debug Console" step repeated dozens of times, truncated mid-
# word). A repetition penalty is the standard mitigation.
DEFAULT_REPETITION_PENALTY = 1.3
DEFAULT_REPETITION_CONTEXT_SIZE = 200

# Deterministic safety net, not just prompt wording: a merge group this large
# is almost certainly the model collapsing an entire multi-action stretch
# into one step (observed in real testing — an 88-of-126 catch-all group).
# Force needs_review rather than trust the model to have flagged it itself.
MAX_REASONABLE_GROUP_SIZE = 15

# Real testing found whole-session single-shot generation unreliable — 2 of
# 5 real sessions (63 and 93 raw steps) produced totally unparseable output,
# even after prompt tightening and a repetition penalty. Chunking is the
# structural fix: keep each call's output short enough that degradation
# doesn't have room to set in. 20 is comfortably below every chunk size that
# has worked reliably in testing so far.
DEFAULT_CHUNK_SIZE = 20


@dataclass(frozen=True)
class RefineParams:
    model_name: str = DEFAULT_MODEL_NAME
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    repetition_context_size: int = DEFAULT_REPETITION_CONTEXT_SIZE
    chunk_size: int = DEFAULT_CHUNK_SIZE


_EXAMPLE_INPUT = [
    {"step_id": "s-01", "action_type": "click", "target_description": "the Save button", "intent": "save the file"},
    {"step_id": "s-02", "action_type": "click", "target_description": "the Save button", "intent": "save the file"},
    {"step_id": "s-03", "action_type": "navigate", "target_description": "the file browser", "intent": "open the file browser"},
    {"step_id": "s-04", "action_type": "click", "target_description": "the report.txt file", "intent": "open report.txt"},
]
_EXAMPLE_OUTPUT = [
    {
        "source_step_ids": ["s-01", "s-02"],
        "intent": "save the file",
        "action_type": "click",
        "target_description": "the Save button",
        "parameters": {},
        "variable_parameters": [],
        "needs_review": False,
        "review_reason": None,
    },
    {
        "source_step_ids": ["s-03"],
        "intent": "open the file browser",
        "action_type": "navigate",
        "target_description": "the file browser",
        "parameters": {},
        "variable_parameters": [],
        "needs_review": False,
        "review_reason": None,
    },
    {
        "source_step_ids": ["s-04"],
        "intent": "open report.txt",
        "action_type": "click",
        "target_description": "the report.txt file",
        "parameters": {"filename": "report.txt"},
        "variable_parameters": ["filename"],
        "needs_review": False,
        "review_reason": None,
    },
]


def _build_refine_prompt(steps: list[Step]) -> str:
    steps_json = json.dumps([s.model_dump() for s in steps], default=str)
    example_input_json = json.dumps(_EXAMPLE_INPUT)
    example_output_json = json.dumps(_EXAMPLE_OUTPUT)
    return (
        "You are refining a sequence of recorded desktop-workflow steps. Each "
        "step was interpreted independently, in isolation, so some are "
        "duplicates or fragments of the same real action, and some may be "
        "noise with no real effect.\n\n"
        "Example. Given this input:\n"
        f"{example_input_json}\n"
        "The correct output is:\n"
        f"{example_output_json}\n"
        "Notice: only the two truly-duplicate Save clicks (s-01, s-02) "
        "merged into one step — navigating to the file browser (s-03) and "
        "opening a specific file (s-04) stayed separate, even though they "
        "are part of the same broader task, because they are different "
        "real actions. Also notice \"report.txt\" was marked as a variable "
        "parameter, since a different run of this workflow would likely "
        "open a different file.\n\n"
        f"Now refine this real step sequence, in order, as JSON:\n{steps_json}\n\n"
        "Produce a refined step sequence as a JSON array. For each output "
        "step, include:\n"
        '- "source_step_ids": the step_id(s) from the input this absorbed (a list)\n'
        '- "intent", "action_type", "target_description": the consolidated '
        "description of the real action (action_type must be one of click, "
        "type, scroll, drag, navigate, run_command)\n"
        '- "parameters": an object with any concrete values, using '
        "consistent keys across similar actions\n"
        '- "variable_parameters": which parameter keys (if any) should be '
        "variable inputs the automation exposes as a parameter, rather than "
        "a value fixed by this recording\n"
        '- "needs_review": see rule below\n'
        '- "review_reason": a short reason if needs_review is true, else null\n\n'
        "Guidelines:\n"
        "- Only merge input steps that describe the SAME immediate real-world "
        "action — e.g. several consecutive entries all describing one "
        "click, or one page load split into several scene-change "
        "fragments. Do NOT merge steps just because they happen close "
        "together in time or belong to the same broader task, exactly as "
        "in the example above\n"
        f"- A merge group must NEVER contain more than {MAX_REASONABLE_GROUP_SIZE} "
        "input step_ids. If you find yourself wanting to group more than "
        "that, you are merging different real actions together — stop and "
        "split it into multiple smaller, more specific groups instead\n"
        "- Drop input steps that look like noise (no real effect) — omit "
        "them from the output entirely\n"
        "- Keep the output in the same chronological order as the input\n"
        "- Every kept input step_id must appear in exactly one output "
        "step's source_step_ids\n"
        "- needs_review should be false for most steps. Only set it true "
        "when you are genuinely unsure of the action_type, or unsure "
        "whether a group of inputs really are duplicates of the same "
        "action — not merely because their wording differs slightly\n\n"
        "Output format — this matters: respond with a single line of "
        "compact, minified JSON — no indentation, no line breaks, no extra "
        "whitespace, no markdown formatting (no ** or backticks) anywhere, "
        "including inside string values. Keep every field name exactly as "
        "specified above, spelled and cased exactly the same, in every "
        "object. Respond with ONLY that one line of JSON, no other text."
    )


def _resolve_source(valid_ids: list[str], input_by_id: dict[str, Step]) -> str:
    sources = {input_by_id[sid].source for sid in valid_ids}
    return "mixed" if len(sources) > 1 else next(iter(sources))


def _passthrough_step(step: Step, reason: str) -> RefinedStep:
    """A 1:1, unrefined stand-in for a step the model didn't usefully
    account for — used both when a step is individually omitted from an
    otherwise-valid response, and when a whole chunk's response fails to
    parse at all. Never silently dropped either way."""
    return RefinedStep(
        step_id=f"refined-passthrough-{step.step_id}",
        source_step_ids=[step.step_id],
        source=step.source,
        intent=step.intent,
        action_type=step.action_type,
        target_description=step.target_description,
        parameters=step.parameters,
        variable_parameters=[],
        needs_review=True,
        review_reason=reason,
    )


def _build_refined_step(
    item: dict[str, Any], input_by_id: dict[str, Step], claimed: set[str], step_id: str
) -> RefinedStep:
    source_step_ids = item.get("source_step_ids")
    if not isinstance(source_step_ids, list) or not source_step_ids:
        raise ValueError("missing or empty source_step_ids")
    # First group to claim an input step_id wins — a later group re-listing
    # an already-covered id (observed in real testing: a redundant 123-item
    # "catch-all" group re-covering steps already in 100+ earlier, correctly
    # sized groups) gets those ids silently stripped, not double-counted.
    valid_ids = [sid for sid in source_step_ids if sid in input_by_id and sid not in claimed]
    if not valid_ids:
        raise ValueError(f"source_step_ids {source_step_ids!r} match no unclaimed input step")

    action_type = item.get("action_type")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"unrecognized action_type {action_type!r}")

    parameters = item.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}

    variable_parameters = item.get("variable_parameters")
    variable_parameters = (
        [v for v in variable_parameters if isinstance(v, str)] if isinstance(variable_parameters, list) else []
    )

    needs_review = bool(item.get("needs_review", False))
    review_reason = item.get("review_reason") if needs_review else None

    if len(valid_ids) > MAX_REASONABLE_GROUP_SIZE:
        # Deterministic override, not just prompt wording — a group this
        # large is far more likely to be an over-merge than a real single
        # action, regardless of whether the model itself flagged it.
        needs_review = True
        review_reason = (
            f"auto-flagged: this group absorbed {len(valid_ids)} input steps, "
            f"more than the {MAX_REASONABLE_GROUP_SIZE} sanity threshold — likely an over-merge"
        )

    # Only claim once every validation has passed — claiming earlier and
    # then raising (e.g. on a bad action_type) would falsely mark these ids
    # as accounted-for even though this group was ultimately discarded,
    # defeating the no-silent-loss passthrough this claims set feeds into.
    claimed.update(valid_ids)

    return RefinedStep(
        step_id=step_id,
        source_step_ids=valid_ids,
        source=_resolve_source(valid_ids, input_by_id),
        intent=item.get("intent", ""),
        action_type=action_type,
        target_description=item.get("target_description", ""),
        parameters=parameters,
        variable_parameters=variable_parameters,
        needs_review=needs_review,
        review_reason=review_reason,
    )


def _parse_refine_response(text: str, input_steps: list[Step]) -> list[RefinedStep]:
    text = strip_code_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {text!r}") from exc
    if not isinstance(data, list):
        raise TypeError(f"model did not return a JSON array: {text!r}")

    input_by_id = {s.step_id: s for s in input_steps}
    index_by_id = {s.step_id: i for i, s in enumerate(input_steps)}
    claimed: set[str] = set()
    refined: list[RefinedStep] = []
    for i, item in enumerate(data):
        try:
            if not isinstance(item, dict):
                raise TypeError(f"array element is not an object: {item!r}")
            refined.append(_build_refined_step(item, input_by_id, claimed, step_id=f"refined-{i:04d}"))
        except (ValueError, TypeError, KeyError) as exc:
            # one malformed element must not lose every other step in the
            # batch — same lesson as the Stage 3 parameters-type crash
            print(f"[prentice] refine: skipping malformed output element {i}: {exc}")

    # The model can drop a step as noise just by omitting it — but real
    # testing found genuine actions (a Save click, a Run click) going
    # missing this way too, indistinguishable from deliberate noise-drops.
    # Never let a step vanish silently: pass it through unrefined, flagged.
    for step_id, step in input_by_id.items():
        if step_id in claimed:
            continue
        refined.append(
            _passthrough_step(
                step, "model did not include this step in its refine response — passed through unrefined"
            )
        )

    refined.sort(key=lambda r: min(index_by_id[sid] for sid in r.source_step_ids))
    return refined


def _refine_chunk(
    chunk: list[Step], model: Any, processor: Any, config: Any, params: RefineParams
) -> list[RefinedStep]:
    prompt = _build_refine_prompt(chunk)
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=0)
    result = generate(
        model,
        processor,
        formatted_prompt,
        image=None,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        repetition_penalty=params.repetition_penalty,
        repetition_context_size=params.repetition_context_size,
        verbose=False,
    )
    try:
        return _parse_refine_response(result.text, chunk)
    except (ValueError, TypeError) as exc:
        # A whole chunk's response can still fail to parse outright (e.g.
        # garbled JSON partway through). Same no-silent-loss rule applies at
        # the chunk level as at the individual-step level: pass every step
        # in this chunk through unrefined rather than losing it.
        print(f"[prentice] refine: chunk failed to parse ({exc}) — passing its {len(chunk)} steps through unrefined")
        return [_passthrough_step(s, f"chunk failed to parse: {exc}") for s in chunk]


def refine_session_steps(steps: list[Step], params: RefineParams | None = None) -> list[RefinedStep]:
    """Refines in bounded chunks rather than the whole session at once —
    see module docstring: real testing found long single-shot generations
    degrade into malformed JSON, garbled text, or repetition loops."""
    params = params or RefineParams()
    if not steps:
        return []

    model, processor, config = load_vlm_model(params.model_name)

    all_refined: list[RefinedStep] = []
    for chunk_start in range(0, len(steps), params.chunk_size):
        chunk = steps[chunk_start : chunk_start + params.chunk_size]
        all_refined.extend(_refine_chunk(chunk, model, processor, config, params))

    # Chunks are processed and appended in chronological order already, and
    # each chunk's own results are internally ordered — just needs a single
    # consistent id sequence across the whole, now-combined session.
    return [r.model_copy(update={"step_id": f"refined-{i:04d}"}) for i, r in enumerate(all_refined)]
