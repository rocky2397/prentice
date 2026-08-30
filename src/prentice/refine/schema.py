"""Output schema for Stage 4 (Refine): the consolidated step sequence
produced by merging/dropping/reconciling Stage 3's raw steps.jsonl.

Deliberately NOT 1:1 with steps.jsonl — merging fragmented duplicates and
dropping noise both legitimately change the count, per ARCHITECTURE.md's
Stage 4 spec. Each RefinedStep keeps source_step_ids for traceability back
to exactly which Stage 3 steps it absorbed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from ..interpret.schema import ActionType
from ..io_utils import load_json, load_jsonl


class RefinedStep(BaseModel):
    step_id: str
    source_step_ids: list[str]
    source: Literal["event_log", "inferred", "mixed"]
    intent: str
    action_type: ActionType
    target_description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    # which parameter keys (if any) should be variable inputs the automation
    # exposes as a parameter, rather than a value fixed by this recording
    variable_parameters: list[str] = Field(default_factory=list)
    needs_review: bool = False
    review_reason: str | None = None


RefinedStepAdapter: TypeAdapter = TypeAdapter(RefinedStep)


class RefineRunMeta(BaseModel):
    """Records the parameters a refine run used, alongside its output — same
    reproducibility rationale as segment_meta.json / interpret_meta.json."""

    session_id: str
    input_step_count: int
    output_step_count: int
    model_name: str
    max_tokens: int
    temperature: float
    repetition_penalty: float
    repetition_context_size: int
    chunk_size: int


RefineRunMetaAdapter: TypeAdapter = TypeAdapter(RefineRunMeta)


def load_refined_steps(path: str) -> list[RefinedStep]:
    return load_jsonl(path, RefinedStepAdapter)


def load_refine_meta(path: str) -> RefineRunMeta:
    return load_json(path, RefineRunMetaAdapter)
