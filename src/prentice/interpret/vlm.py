"""Qwen3-VL wrapper for Stage 3 (Interpret), via mlx-vlm: for each segment,
sends the before/after keyframes plus the raw logged event(s) plus a bounded
window of recent prior steps, and asks for a structured step back — per
ARCHITECTURE.md's Stage 3 spec.

Local-only for now: ARCHITECTURE.md also calls for a separate frontier-model
benchmark (e.g. Gemini) to establish a quality ceiling, but that needs an
external API key and costs money per call, so it's intentionally out of scope
here — a separate, explicitly-scoped piece of work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from ..segment.schema import Segment
from .keyframes import SegmentKeyframes
from .schema import Step

# A ~3B-active-param MoE despite the "30B" name, so it stays fast on unified
# memory. See README for the mlx-community model card and disk footprint.
DEFAULT_MODEL_NAME = "mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit"
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0

_VALID_ACTION_TYPES = {"click", "type", "scroll", "drag", "navigate", "run_command"}


@dataclass(frozen=True)
class InterpretParams:
    model_name: str = DEFAULT_MODEL_NAME
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE


@lru_cache(maxsize=2)
def _load_vlm_model(model_name: str):
    """Cached so a batch interpret run over many sessions loads the checkpoint once."""
    model, processor = load(model_name)
    config = load_config(model_name)
    return model, processor, config


def _build_prompt(segment: Segment, prior_steps: list[Step]) -> str:
    if segment.events:
        event_json = json.dumps([e.model_dump() for e in segment.events], default=str)
        event_context = f"The OS-level input event(s) logged during this step: {event_json}"
    else:
        event_context = (
            "No input event log exists for this step (it was detected purely from a "
            "visual scene change) — infer the action from the before/after screenshots alone."
        )

    if prior_steps:
        lines = [f"{i + 1}. {s.action_type} — {s.intent}" for i, s in enumerate(prior_steps)]
        prior_context = "Recent prior steps, for context only:\n" + "\n".join(lines)
    else:
        prior_context = "This is the first step in the recording."

    return (
        "You are analyzing one recorded step of a desktop workflow, shown as a "
        "BEFORE screenshot (first image) and an AFTER screenshot (second image).\n\n"
        f"{event_context}\n\n"
        f"{prior_context}\n\n"
        "Describe this step as a JSON object with exactly these fields:\n"
        '- "intent": why this step happened, in one short sentence\n'
        '- "action_type": one of click, type, scroll, drag, navigate, run_command\n'
        '- "target_description": the UI element or content this step acted on, in words '
        '(e.g. "the Save button in the toolbar")\n'
        '- "parameters": an object with any concrete values (typed text, scroll amount, '
        "drag destination, etc.), or {} if none apply\n\n"
        "Respond with ONLY the JSON object, no other text."
    )


def _parse_step_response(text: str, segment: Segment, step_id: str) -> Step:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON for segment {segment.segment_id}: {text!r}") from exc

    if data.get("action_type") not in _VALID_ACTION_TYPES:
        raise ValueError(
            f"model returned an unrecognized action_type {data.get('action_type')!r} "
            f"for segment {segment.segment_id}"
        )

    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        # parameters is best-effort extra detail (typed text, scroll amount, etc.) —
        # a malformed value here (e.g. a stray string) shouldn't invalidate the whole step
        parameters = {}

    return Step(
        step_id=step_id,
        segment_id=segment.segment_id,
        source=segment.source,
        intent=data.get("intent", ""),
        action_type=data["action_type"],
        target_description=data.get("target_description", ""),
        parameters=parameters,
    )


def interpret_segment(
    segment: Segment,
    keyframes: SegmentKeyframes,
    prior_steps: list[Step],
    step_id: str,
    params: InterpretParams | None = None,
) -> Step:
    params = params or InterpretParams()
    model, processor, config = _load_vlm_model(params.model_name)

    bounded_context = prior_steps[-params.context_window :] if params.context_window > 0 else []
    prompt = _build_prompt(segment, bounded_context)
    image_paths = [str(keyframes.before_path), str(keyframes.after_path)]

    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image_paths))
    result = generate(
        model,
        processor,
        formatted_prompt,
        image_paths,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        verbose=False,
    )
    return _parse_step_response(result.text, segment, step_id)
