"""Scoring refined steps against an answer key (prentice.evaluate.steps)."""

import json

from prentice.evaluate.steps import AnswerKey, AnswerKeyStep, align, can_pair, main, score
from prentice.refine.schema import RefinedStep


def step(sid, action, target, intent="", **params):
    return RefinedStep(
        step_id=sid,
        source_step_ids=[sid],
        source="event_log",
        intent=intent,
        action_type=action,
        target_description=target,
        parameters=params,
    )


KEY = AnswerKey(
    task="terminal-git",
    status="draft",
    steps=[
        AnswerKeyStep(
            id=1,
            what="cd",
            action_types=["run_command", "type"],
            keywords=["cd"],
            expect_param="cd",
        ),
        AnswerKeyStep(
            id=2,
            what="ls",
            action_types=["run_command", "type"],
            keywords=["ls"],
            expect_param="ls",
        ),
        AnswerKeyStep(
            id=3,
            what="git log",
            action_types=["run_command"],
            keywords=["git log"],
            expect_param="git log",
        ),
    ],
)


def test_a_perfect_run_scores_full_marks():
    steps = [
        step("s1", "run_command", "terminal", command="cd ~/x"),
        step("s2", "run_command", "terminal", command="ls"),
        step("s3", "run_command", "terminal", command="git log --oneline -3"),
    ]
    s = score(KEY, steps)
    assert (s.recall, s.precision) == (1.0, 1.0)
    assert s.action_type_right == 3
    assert s.values_captured == 3


def test_missing_extra_and_wrong_type_are_counted():
    steps = [
        step("s1", "click", "terminal window", intent="focus the terminal"),  # noise
        step("s2", "type", "prompt", intent="change directory with cd"),  # no value captured
        step("s3", "click", "git log output", intent="look at git log"),  # wrong type
    ]
    s = score(KEY, steps)
    assert s.required_found == 2 and s.recall == 2 / 3
    assert s.predicted_paired == 2 and s.precision == 2 / 3
    assert s.action_type_right == 1
    assert s.values_captured == 0
    assert [r.partner for r in s.rows] == ["s2", None, "s3"]


def test_keywords_match_whole_words_only():
    ls = KEY.steps[1]
    assert not can_pair(ls, step("s", "type", "tools menu"))
    assert can_pair(ls, step("s", "type", "prompt", command="ls -la"))


def test_alignment_keeps_order():
    # "git log" mentioned first must not steal the later git-log step's place
    steps = [
        step("a", "type", "prompt", intent="prepare to run git log after cd"),
        step("b", "run_command", "terminal", command="cd ~/x"),
        step("c", "run_command", "terminal", command="git log"),
    ]
    pairs = align(KEY.steps, steps)
    assert pairs == [(0, 0), (2, 2)] or pairs == [(0, 1), (2, 2)]
    assert (2, 2) in pairs


def test_optional_steps_never_hurt_recall():
    key = AnswerKey(
        task="t",
        status="verified",
        steps=[
            AnswerKeyStep(id=1, what="a", action_types=["click"], keywords=["alpha"]),
            AnswerKeyStep(id=2, what="b", action_types=["click"], keywords=["beta"], optional=True),
        ],
    )
    s = score(key, [step("s1", "click", "alpha button")])
    assert s.recall == 1.0 and s.optional_found == 0


def test_cli_reports_and_writes_json(tmp_path, capsys):
    session = tmp_path / "sess"
    session.mkdir()
    steps = [
        step("s1", "run_command", "t", command="cd x"),
        step("s2", "run_command", "t", command="ls"),
    ]
    (session / "refined_steps.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in steps))
    key_path = tmp_path / "steps.json"
    key_path.write_text(KEY.model_dump_json())
    out = tmp_path / "result.json"
    main([str(session), str(key_path), "--out", str(out)])
    printed = capsys.readouterr().out
    assert "DRAFT" in printed and "2/3" in printed
    assert json.loads(out.read_text())["required_found"] == 2


def test_longer_keywords_also_match_word_endings():
    key = AnswerKeyStep(id=1, what="scroll", action_types=["scroll"], keywords=["scroll"])
    assert can_pair(key, step("s", "scroll", "page", intent="The user scrolled down"))
    assert not can_pair(
        key, step("s", "scroll", "page", intent="the bankscroll")
    )  # must start a word
