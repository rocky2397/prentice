"""Group consecutive terminal commands into one script.

Each scripted step becomes its own script file, and each script runs in its
own shell. For terminal commands that's wrong: `cd` in one script doesn't
carry over to `ls` in the next, so a replay would run later commands in the
wrong folder. A person typing them ran them in one shell, in order.

So consecutive steps whose commands come from the event log (typed in a
terminal app and ended with Enter, see provenance.py) are merged into one
scripted step, run as one script. Between two commands only steps that are
nothing but focus changes within a terminal app are absorbed (a terminal's
window title changes when `cd` changes the folder, which the capture logs as
a window switch). Anything else ends the group: a click, typing elsewhere,
another app. The merged step keeps every source step id, so traceability back
to Stage 3 is intact.

Known limit: two different windows of the same terminal app can't be told
apart from the event log, so commands typed into two terminal windows with no
other step in between would be grouped.
"""

from __future__ import annotations

from .provenance import StepEvidence
from .schema import GroundedStep

_CONFIDENCE_ORDER = {"ok": 0, "model_flagged": 1, "unrefined_passthrough": 2}


def _merge(steps: list[GroundedStep]) -> GroundedStep:
    commands = [s for s in steps if s.grounding == "script"]
    script = "\n".join(s.script for s in commands if s.script)
    lines = script.splitlines()
    reasons = [s.review_reason for s in steps if s.review_reason]
    first = commands[0]
    return first.model_copy(
        update={
            "source_step_ids": [sid for s in steps for sid in s.source_step_ids],
            "intent": "Run in one terminal session: " + "; ".join(f"`{line}`" for line in lines),
            "action_type": "run_command",
            "grounding": "script",
            "confidence": max((s.confidence for s in steps), key=_CONFIDENCE_ORDER.__getitem__),
            "needs_review": any(s.needs_review for s in steps),
            "review_reason": reasons[0] if reasons else None,
            "script": script,
            "parameters": {"commands": lines},
            "variable_parameters": [],
        }
    )


def group_terminal_commands(
    steps: list[GroundedStep], evidence: list[StepEvidence]
) -> list[GroundedStep]:
    """Merge runs of event-log terminal commands; everything else is unchanged."""
    assert len(steps) == len(evidence)
    out: list[GroundedStep] = []
    i = 0
    while i < len(steps):
        if not evidence[i].terminal_command:
            out.append(steps[i])
            i += 1
            continue
        group = [steps[i]]
        pending_focus: list[GroundedStep] = []
        j = i + 1
        while j < len(steps):
            if evidence[j].terminal_command:
                group += pending_focus + [steps[j]]
                pending_focus = []
            elif evidence[j].terminal_focus_only:
                pending_focus.append(steps[j])
            else:
                break
            j += 1
        # focus changes after the last command aren't part of the session
        out.append(_merge(group) if len(group) > 1 else group[0])
        out.extend(pending_focus)
        i = j
    return out
