"""Scoring refined steps against a hand-verified answer key (eval/ground_truth).

The answer key is the step sequence a human saw in the recording, written as
intent-level steps ("search for X"), each with the action types that count as
right, keywords that identify it, and optionally a concrete value the
pipeline should have extracted (a URL, a command, a typed name).

Matching is deliberately simple and transparent, with no model involved:

- A predicted step *can* stand for an answer-key step when any of the key's
  keywords appears, as a whole word, in the step's intent, target description
  or parameters.
- Steps are paired by an order-preserving alignment that pairs as many as
  possible (ties prefer the right action type), so each predicted step and
  each answer-key step is used at most once.

Reported:

- **recall:** required answer-key steps that found a partner (optional steps
  never count against it);
- **precision:** predicted steps that found a partner (the rest are extra or
  noise, e.g. switching back to the recorder to stop it);
- **action type:** of the pairs, how many have an allowed action type;
- **values captured:** of the answer-key steps with an expected value, how
  many have it in the partner's parameters (what Stage 5 needs to script it).

Known limits, stated in the report: keywords are a proxy for "this is the
same step" and can be fooled by a step that mentions the right words for the
wrong reason; Stage 4 may merge two real steps into one, which then counts as
one found and one missed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from prentice.interpret.schema import ActionType
from prentice.refine.schema import RefinedStep, load_refined_steps


class AnswerKeyStep(BaseModel):
    id: int
    what: str
    action_types: list[ActionType]
    keywords: list[str] = Field(min_length=1)
    expect_param: str | None = None
    optional: bool = False


class AnswerKey(BaseModel):
    task: str
    status: str  # "draft" until a human checked it against the recording, then "verified"
    verified_against: str | None = None
    steps: list[AnswerKeyStep]


def load_answer_key(path: str | Path) -> AnswerKey:
    return AnswerKey.model_validate_json(Path(path).read_text())


def _word(term: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive: "ls" must not match "tools"."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])")


def _step_text(step: RefinedStep) -> str:
    return " ".join([step.intent, step.target_description, json.dumps(step.parameters)]).lower()


def can_pair(key: AnswerKeyStep, step: RefinedStep) -> bool:
    text = _step_text(step)
    return any(_word(k).search(text) for k in key.keywords)


def type_ok(key: AnswerKeyStep, step: RefinedStep) -> bool:
    return step.action_type in key.action_types


def value_captured(key: AnswerKeyStep, step: RefinedStep) -> bool:
    if key.expect_param is None:
        return False
    return bool(_word(key.expect_param).search(json.dumps(step.parameters).lower()))


def align(keys: list[AnswerKeyStep], steps: list[RefinedStep]) -> list[tuple[int, int]]:
    """Order-preserving pairing with the most pairs; ties prefer right action types.

    Returns (answer-key index, predicted index) pairs.
    """
    n, m = len(keys), len(steps)
    # best[i][j]: (pairs, right types) for keys[i:] and steps[j:]
    best = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            options = [best[i + 1][j], best[i][j + 1]]
            if can_pair(keys[i], steps[j]):
                pairs, types = best[i + 1][j + 1]
                options.append((pairs + 1, types + int(type_ok(keys[i], steps[j]))))
            best[i][j] = max(options)
    out, i, j = [], 0, 0
    while i < n and j < m:
        if can_pair(keys[i], steps[j]):
            pairs, types = best[i + 1][j + 1]
            if (pairs + 1, types + int(type_ok(keys[i], steps[j]))) == best[i][j]:
                out.append((i, j))
                i, j = i + 1, j + 1
                continue
        if best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


@dataclass
class StepRow:
    key_id: int
    what: str
    optional: bool
    partner: str | None  # predicted step_id
    type_ok: bool | None
    value_expected: bool
    value_captured: bool | None


@dataclass
class StepScore:
    task: str
    answer_key_status: str
    required_steps: int
    required_found: int
    optional_found: int
    predicted_steps: int
    predicted_paired: int
    recall: float
    precision: float
    action_type_right: int
    values_expected: int
    values_captured: int
    rows: list[StepRow] = field(default_factory=list)


def score(key: AnswerKey, steps: list[RefinedStep]) -> StepScore:
    pairs = dict(align(key.steps, steps))
    rows = []
    for i, k in enumerate(key.steps):
        j = pairs.get(i)
        partner = steps[j] if j is not None else None
        rows.append(
            StepRow(
                key_id=k.id,
                what=k.what,
                optional=k.optional,
                partner=partner.step_id if partner else None,
                type_ok=type_ok(k, partner) if partner else None,
                value_expected=k.expect_param is not None,
                value_captured=value_captured(k, partner) if partner and k.expect_param else None,
            )
        )
    required = [r for r in rows if not r.optional]
    found = [r for r in required if r.partner]
    return StepScore(
        task=key.task,
        answer_key_status=key.status,
        required_steps=len(required),
        required_found=len(found),
        optional_found=sum(1 for r in rows if r.optional and r.partner),
        predicted_steps=len(steps),
        predicted_paired=len(pairs),
        recall=len(found) / len(required) if required else 0.0,
        precision=len(pairs) / len(steps) if steps else 0.0,
        action_type_right=sum(1 for r in rows if r.type_ok),
        values_expected=sum(1 for r in rows if r.value_expected),
        values_captured=sum(1 for r in rows if r.value_captured),
        rows=rows,
    )


def report(s: StepScore, ground_meta: dict | None) -> str:
    lines = [f"Task {s.task}"]
    if s.answer_key_status != "verified":
        lines.append(
            "  ! answer key is a DRAFT: verify it against the recording before trusting these numbers"
        )
    lines += [
        f"  steps found (recall):   {s.required_found}/{s.required_steps} = {s.recall:.0%}"
        + (f" (+{s.optional_found} optional)" if s.optional_found else ""),
        f"  steps that pair (precision): {s.predicted_paired}/{s.predicted_steps} = {s.precision:.0%}",
        f"  right action type:      {s.action_type_right}/{s.predicted_paired}",
        f"  values captured:        {s.values_captured}/{s.values_expected}",
        "",
    ]
    for r in s.rows:
        mark = "found " if r.partner else ("skip  " if r.optional else "MISSED")
        extra = []
        if r.partner:
            extra.append("type ok" if r.type_ok else "WRONG type")
            if r.value_expected:
                extra.append("value ok" if r.value_captured else "NO value")
        lines.append(
            f"  {mark} #{r.key_id} {r.what}" + (f"  [{', '.join(extra)}]" if extra else "")
        )
    if ground_meta:
        lines += [
            "",
            (
                f"  grounding: {ground_meta.get('scripted_step_count')} scripted, "
                f"{ground_meta.get('ui_replay_step_count')} UI replay, "
                f"{ground_meta.get('ax_grounded_step_count')} with an accessibility id, "
                f"{ground_meta.get('needs_review_count')} flagged for review"
            ),
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score a session's refined steps against an answer key."
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("answer_key", type=Path, help="eval/ground_truth/<task>/steps.json")
    parser.add_argument("--out", type=Path, default=None, help="write the full score as JSON here")
    args = parser.parse_args(argv)

    refined = args.session_dir / "refined_steps.jsonl"
    if not refined.is_file():
        sys.exit(f"{refined} not found: run prentice-segment, -interpret and -refine first")
    meta_path = args.session_dir / "ground_meta.json"
    ground_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None

    s = score(load_answer_key(args.answer_key), load_refined_steps(str(refined)))
    print(report(s, ground_meta))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"session": args.session_dir.name, **asdict(s), "grounding": ground_meta}
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
