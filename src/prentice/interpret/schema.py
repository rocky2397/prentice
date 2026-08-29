"""Output schema for Stage 3 (Interpret): the structured step each segment
gets turned into by the VLM, per ARCHITECTURE.md's Stage 3 spec — intent,
action type, target description, and parameters. ``source`` carries straight
through from the originating Segment, unmodified, so later stages (and eval
reporting) can still tell which segments came from the exact event-log path
vs. the inferred CLIP path.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from ..io_utils import load_json, load_jsonl


class Step(BaseModel):
    step_id: str
    segment_id: str
    source: Literal["event_log", "inferred"]
    intent: str
    action_type: Literal["click", "type", "scroll", "drag", "navigate", "run_command"]
    target_description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


StepAdapter: TypeAdapter = TypeAdapter(Step)


class InterpretRunMeta(BaseModel):
    """Records the parameters an interpret run used, alongside its output —
    same reproducibility rationale as segment_meta.json in Stage 2."""

    session_id: str
    step_count: int
    model_name: str
    context_window: int
    max_tokens: int
    temperature: float


InterpretRunMetaAdapter: TypeAdapter = TypeAdapter(InterpretRunMeta)


def load_steps(path: str) -> list[Step]:
    """Parse a ``steps.jsonl`` file into validated Step models, in order."""
    return load_jsonl(path, StepAdapter)


def load_interpret_meta(path: str) -> InterpretRunMeta:
    return load_json(path, InterpretRunMetaAdapter)
