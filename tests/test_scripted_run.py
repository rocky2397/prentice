"""The parts of prentice-scripted-run that don't drive the screen."""

import sys

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


def _fake_recorder(tmp_path, body):
    # A shell wrapper: a "#!" line can't hold a Python path with spaces in it.
    script = tmp_path / "fake_capture.py"
    script.write_text(f"import time\n{body}\n")
    exe = tmp_path / "fake-capture"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    exe.chmod(0o755)
    return exe


def test_recorder_output_through_a_pipe_is_seen(tmp_path):
    # print() without flush: only works because the child runs unbuffered
    exe = _fake_recorder(
        tmp_path,
        'print("[prentice] session: s-1")\nprint("[prentice] press Ctrl+C to stop")\ntime.sleep(30)',
    )
    proc, session = scripted.start_capture(timeout=10, exe=exe, settle=0)
    try:
        assert session == "s-1"
    finally:
        proc.kill()


def test_a_recorder_that_never_starts_fails_with_its_output(tmp_path):
    exe = _fake_recorder(tmp_path, 'print("[prentice] error: no screen device")\ntime.sleep(30)')
    with pytest.raises(RuntimeError, match="no screen device"):
        scripted.start_capture(timeout=1.5, exe=exe, settle=0)
