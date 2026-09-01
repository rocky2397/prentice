"""Output schema for Stage 5 (Ground & output): the per-step decision of how
a refined step should actually be replayed, per ARCHITECTURE.md's Stage 5
spec — a **scripted action** wherever one genuinely exists (preferred: it
doesn't suffer from UI-grounding failures at all), otherwise a **UI-replay
action** carrying the semantic target description as the primary handle.

``source`` continues to ride through unmodified, same propagation rule as
Stages 2-4, so the event-log and inferred paths stay separable in eval.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from ..interpret.schema import ActionType
from ..io_utils import load_json, load_jsonl

# How a step will be replayed. "script" is strictly preferred per
# ARCHITECTURE.md §Stage 5 — it's the single biggest reliability lever
# available — but is only ever chosen on concrete evidence (see grounding.py).
GroundingKind = Literal["script", "ui_replay"]

# Why a step is or isn't trustworthy. Deliberately finer-grained than Stage
# 4's single needs_review bool: on real sessions the overwhelming majority
# of flags are Stage 4 *passthrough* artifacts (a chunk whose JSON failed to
# parse, or a step the model silently omitted), which say "this step was
# never actually refined" — a very different thing from the model inspecting
# a step and reporting genuine uncertainty about it. Collapsing the two would
# make the flag useless precisely when it matters.
Confidence = Literal["ok", "model_flagged", "unrefined_passthrough"]


class UITarget(BaseModel):
    """How to find the element again at replay time.

    ``description`` is the primary handle and is always present. The
    accessibility identifier is the strongest signal when it exists
    (ARCHITECTURE.md §7 cites GUI-360°'s ~3% -> ~37% grounding gain from
    accessibility-tree metadata over vision-only), and coordinates are a
    last-resort hint only, never the primary target — layouts shift, windows
    resize, resolutions differ.
    """

    description: str
    ax_role: str | None = None
    ax_title: str | None = None
    ax_description: str | None = None
    ax_value: str | None = None
    # Last-resort fallback hint ONLY. Recorded in the source video's pixel
    # coordinate space; meaningless on a different display geometry.
    fallback_x: float | None = None
    fallback_y: float | None = None

    @property
    def has_ax_identifier(self) -> bool:
        return any((self.ax_role, self.ax_title, self.ax_description, self.ax_value))


class GroundedStep(BaseModel):
    step_id: str
    source_step_ids: list[str]  # traceability all the way back to Stage 3 steps
    source: Literal["event_log", "inferred", "mixed"]
    intent: str
    action_type: ActionType
    grounding: GroundingKind
    confidence: Confidence
    needs_review: bool = False
    review_reason: str | None = None
    # exactly one of these is set, per `grounding`
    script: str | None = None  # the shell command, when grounding == "script"
    script_path: str | None = None  # relative path of the emitted script file
    ui_target: UITarget | None = None  # when grounding == "ui_replay"
    parameters: dict[str, Any] = Field(default_factory=dict)
    variable_parameters: list[str] = Field(default_factory=list)


GroundedStepAdapter: TypeAdapter = TypeAdapter(GroundedStep)


class GroundRunMeta(BaseModel):
    """Records what a grounding run produced, alongside its output — same
    reproducibility rationale as segment_meta / interpret_meta / refine_meta.

    No model name or sampling parameters here, unlike Stages 3 and 4: Stage 5
    is fully deterministic and makes no model call at all, so the run is
    reproducible from its inputs alone.
    """

    session_id: str
    skill_name: str
    input_step_count: int
    output_step_count: int
    scripted_step_count: int
    ui_replay_step_count: int
    ax_grounded_step_count: int  # UI-replay steps that carry an accessibility identifier
    needs_review_count: int
    unrefined_passthrough_count: int


GroundRunMetaAdapter: TypeAdapter = TypeAdapter(GroundRunMeta)


def load_grounded_steps(path: str) -> list[GroundedStep]:
    return load_jsonl(path, GroundedStepAdapter)


def load_ground_meta(path: str) -> GroundRunMeta:
    return load_json(path, GroundRunMetaAdapter)
