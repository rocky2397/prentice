"""The parts of prentice-scripted-run that don't drive the screen."""

import pytest

from prentice.evaluate import scripted
from prentice.evaluate.steps import AnswerKey


def test_answer_key_loads_in_the_scorer():
    plan = scripted.Plan("terminal-git")
    plan.steps.append(scripted.KeyStep("ls", ["run_command", "type"], ["ls"], "ls", at_unix_ms=1))
    key = AnswerKey.model_validate(plan.answer_key("sess-1", success=True))
    assert key.status == "scripted"
    assert key.verified_against == "sess-1"
    assert key.steps[0].expect_param == "ls"


def test_sandbox_is_created_and_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(scripted, "SANDBOX", tmp_path / "prentice-sandbox")
    scripted.prepare_sandbox()
    assert (scripted.SANDBOX / scripted.DUMMY).is_file()
    (scripted.SANDBOX / "prentice-test").mkdir()
    (scripted.SANDBOX / "prentice-test" / "prentice-renamed.txt").write_text("test\n")
    scripted.remove_sandbox()
    assert not scripted.SANDBOX.exists()


def test_sandbox_with_foreign_files_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(scripted, "SANDBOX", tmp_path / "prentice-sandbox")
    scripted.prepare_sandbox()
    (scripted.SANDBOX / "my-notes.txt").write_text("keep me\n")
    with pytest.raises(RuntimeError, match="my-notes.txt"):
        scripted.remove_sandbox()
    assert (scripted.SANDBOX / "my-notes.txt").is_file()
