"""The Stage 5 per-step decision: scripted action or UI-replay action.

ARCHITECTURE.md §Stage 5 makes scripted actions strictly preferable — Claude
Code can execute them directly and they don't suffer from UI-grounding
failures at all — and calls this the biggest reliability lever in the whole
pipeline. The temptation that follows is to try hard to turn prose into
commands. This module deliberately does the opposite.

**A script is only ever emitted from a concrete executable value that
actually appears in a step's ``parameters``.** A target description is prose
written by a vision model and is never treated as evidence: turning "the
YouTube homepage" into ``open https://youtube.com`` would be a guess that
looks exactly as confident as a real one, and a wrong scripted action is
strictly worse than an honest UI-replay step, because it runs without ever
showing a human the thing it got wrong. On the five real sessions in this
repo, this rule scripts almost nothing — the VLM populated no URLs at all
(see README) — and that low number is the correct, honest result rather
than a shortfall to engineer around.

Fully deterministic and model-free, so a grounding run is reproducible from
its inputs alone.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from ..refine.schema import RefinedStep
from .provenance import EMPTY_EVIDENCE, StepEvidence
from .schema import Confidence, GroundedStep, UITarget

# Parameter keys whose *name* asserts the value is of a given kind. Matching
# on the key, not on a loose sniff of the value, keeps the rule evidence-based.
URL_KEYS = frozenset({"url", "uri", "link", "href", "address", "web_address", "page_url"})
APP_KEYS = frozenset({"application", "app", "app_name", "program"})
COMMAND_KEYS = frozenset({"command", "shell_command", "cmd", "shell", "terminal_command"})

_EXPLICIT_URL = re.compile(r"^https?://\S+$", re.IGNORECASE)
# A bare hostname (optionally with a path): labels separated by dots, ending
# in an alphabetic TLD of at least two characters, and no whitespace anywhere.
_BARE_DOMAIN = re.compile(
    r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*\.[a-z]{2,}(/\S*)?$",
    re.IGNORECASE,
)

# Substrings that identify a Stage 4 *passthrough* review reason. Coupled by
# wording to refine/llm.py's `_passthrough_step` / `_refine_chunk` messages;
# tests/test_ground_grounding.py asserts that coupling against the real
# Stage 4 constructor, so a reworded reason there fails loudly here rather
# than silently downgrading every passthrough step to "model_flagged".
PASSTHROUGH_MARKERS = ("chunk failed to parse", "passed through unrefined")


def classify_confidence(step: RefinedStep) -> Confidence:
    """Split Stage 4's single needs_review bool into a usable signal.

    ``unrefined_passthrough`` means Stage 4 never successfully refined this
    step at all (its chunk's JSON failed to parse, or the model omitted it
    and the no-silent-loss net caught it). ``model_flagged`` means the step
    *was* refined and something judged it low-confidence anyway — either the
    model itself, or Stage 4's deterministic over-merge override.
    """
    if not step.needs_review:
        return "ok"
    reason = step.review_reason or ""
    if any(marker in reason for marker in PASSTHROUGH_MARKERS):
        return "unrefined_passthrough"
    return "model_flagged"


def _first_matching_value(parameters: dict[str, Any], keys: frozenset[str]) -> str | None:
    """The first non-empty string value under any of ``keys`` (case-insensitive)."""
    for key, value in parameters.items():
        if key.strip().lower() in keys and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _explicit_url_anywhere(parameters: dict[str, Any]) -> str | None:
    """Any parameter value carrying an explicit http(s) scheme.

    Checked regardless of key name: a full scheme-bearing URL is unambiguous
    evidence on its own, whatever the model chose to call the field.
    """
    for value in parameters.values():
        if isinstance(value, str) and _EXPLICIT_URL.match(value.strip()):
            return value.strip()
    return None


def derive_script(step: RefinedStep) -> str | None:
    """The shell command this step should run, or None if it isn't scriptable.

    Only concrete values from ``parameters`` are considered — never the
    action type alone, and never the target description.
    """
    command = _first_matching_value(step.parameters, COMMAND_KEYS)
    if command:
        return command

    explicit_url = _explicit_url_anywhere(step.parameters)
    if explicit_url:
        return f"open {shlex.quote(explicit_url)}"

    url_value = _first_matching_value(step.parameters, URL_KEYS)
    if url_value and _BARE_DOMAIN.match(url_value):
        # The key explicitly names this a URL and the value is a well-formed
        # hostname, so supplying the scheme is a safe completion rather than
        # an invention. A URL-named key whose value is *not* domain-shaped
        # (e.g. {"url": "the address bar"}) deliberately falls through.
        return f"open {shlex.quote(f'https://{url_value}')}"

    app = _first_matching_value(step.parameters, APP_KEYS)
    if app:
        return f"open -a {shlex.quote(app)}"

    return None


def _ui_target(step: RefinedStep, evidence: StepEvidence) -> UITarget:
    return UITarget(
        description=step.target_description,
        ax_role=evidence.ax_role,
        ax_title=evidence.ax_title,
        ax_description=evidence.ax_description,
        ax_value=evidence.ax_value,
        fallback_x=evidence.x,
        fallback_y=evidence.y,
    )


def ground_step(
    step: RefinedStep, evidence: StepEvidence = EMPTY_EVIDENCE, *, step_id: str | None = None
) -> GroundedStep:
    confidence = classify_confidence(step)
    needs_review = step.needs_review
    review_reason = step.review_reason

    script = derive_script(step)

    if script is None and step.action_type == "run_command":
        # The interpreter believed a command ran here but captured no command
        # text, so the one action type that should always have been scriptable
        # can't be. That's worth a human look regardless of Stage 4's verdict.
        needs_review = True
        review_reason = review_reason or (
            "action_type is run_command but no command text was captured — "
            "cannot be scripted, falling back to UI replay"
        )

    if script is not None:
        return GroundedStep(
            step_id=step_id or step.step_id,
            source_step_ids=step.source_step_ids,
            source=step.source,
            intent=step.intent,
            action_type=step.action_type,
            grounding="script",
            confidence=confidence,
            needs_review=needs_review,
            review_reason=review_reason,
            script=script,
            parameters=step.parameters,
            variable_parameters=step.variable_parameters,
        )

    return GroundedStep(
        step_id=step_id or step.step_id,
        source_step_ids=step.source_step_ids,
        source=step.source,
        intent=step.intent,
        action_type=step.action_type,
        grounding="ui_replay",
        confidence=confidence,
        needs_review=needs_review,
        review_reason=review_reason,
        ui_target=_ui_target(step, evidence),
        parameters=step.parameters,
        variable_parameters=step.variable_parameters,
    )
