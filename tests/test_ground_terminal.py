"""Terminal commands rebuilt from the event log (provenance + grounding)."""

from prentice.capture.schema import KeyEvent, MouseClickEvent, WindowSwitchEvent
from prentice.ground.grounding import ground_step
from prentice.ground.provenance import StepEvidence, evidence_for, typed_commands
from prentice.ground.schema import GroundedStep, UITarget
from prentice.ground.terminal_session import group_terminal_commands
from prentice.refine.schema import RefinedStep
from prentice.segment.schema import Segment


def keys(text, t0=1000.0):
    """Press/release events for a string; "\n" is Enter, "\b" backspace, "\t" Tab."""
    names = {" ": "Key.space", "\n": "Key.enter", "\b": "Key.backspace", "\t": "Key.tab"}
    out, t = [], t0
    for ch in text:
        k = names.get(ch, ch)
        out += [KeyEvent(t_ms=t, key=k, pressed=True), KeyEvent(t_ms=t + 1, key=k, pressed=False)]
        t += 10
    return out


def segment(sid, events, start=1000.0, end=2000.0):
    return Segment(
        segment_id=sid,
        source="event_log",
        action_hint="type",
        start_ms=start,
        end_ms=end,
        frame_start=0,
        frame_end=10,
        events=events,
    )


def refined(*step_ids, action="type", **params):
    return RefinedStep(
        step_id="r1",
        source_step_ids=list(step_ids),
        source="event_log",
        intent="typed something",
        action_type=action,
        target_description="the Terminal window",
        parameters=params,
    )


TERMINAL = [WindowSwitchEvent(t_ms=10, app_name="Terminal", bundle_id="com.apple.Terminal")]
FINDER = [WindowSwitchEvent(t_ms=10, app_name="Finder", bundle_id="com.apple.finder")]


def test_typed_text_with_enter_is_the_command():
    assert typed_commands(keys('cd "/a b"\n')) == 'cd "/a b"'
    assert typed_commands(keys("lss\bs\n")) == "lss"
    assert typed_commands(keys("ls\ngit log\n")) == "ls\ngit log"


def test_uncertain_typing_gives_no_command():
    assert typed_commands(keys("cd /Vol\t\n")) is None  # Tab completion
    assert typed_commands(keys("ls")) is None  # never ran: no Enter
    assert typed_commands(keys("\n")) is None  # empty line


def test_shift_does_not_change_the_text():
    events = [KeyEvent(t_ms=1, key="Key.shift", pressed=True), *keys('"x"\n')]
    assert typed_commands(events) == '"x"'


def test_evidence_needs_a_terminal_app():
    segs = {"s1": segment("s1", keys("ls\n"))}
    assert evidence_for(refined("s1"), segs, TERMINAL).terminal_command == "ls"
    assert evidence_for(refined("s1"), segs, FINDER).terminal_command is None
    assert evidence_for(refined("s1"), segs, []).terminal_command is None


def test_a_window_switch_or_click_while_typing_gives_no_command():
    segs = {"s1": segment("s1", keys("ls\n"))}
    switched = [*TERMINAL, WindowSwitchEvent(t_ms=1500, bundle_id="com.apple.finder")]
    assert evidence_for(refined("s1"), segs, switched).terminal_command is None
    clicked = {
        "s1": segment(
            "s1", [MouseClickEvent(t_ms=1001, x=1, y=1, button="left", pressed=True), *keys("ls\n")]
        )
    }
    assert evidence_for(refined("s1"), clicked, TERMINAL).terminal_command is None


def test_grounding_scripts_the_logged_command_over_the_models_text():
    segs = {"s1": segment("s1", keys('cd "/Volumes/X Y"\n'))}
    step = refined("s1", text='cd "Volumes/X Y"')  # the model dropped the slash
    grounded = ground_step(step, evidence_for(step, segs, TERMINAL))
    assert grounded.grounding == "script"
    assert grounded.script == 'cd "/Volumes/X Y"'
    assert grounded.action_type == "run_command"


def test_without_evidence_nothing_changes():
    step = refined("s1", text="ls")
    grounded = ground_step(step, evidence_for(step, {}, TERMINAL))
    assert grounded.grounding == "ui_replay"
    assert grounded.action_type == "type"


# ---- grouping consecutive commands into one script (terminal_session.py)


def CMD(command):
    return StepEvidence(terminal_command=command)


FOCUS = StepEvidence(terminal_focus_only=True)
OTHER = StepEvidence()


def g(sid, script=None):
    common = {
        "step_id": sid,
        "source_step_ids": [sid],
        "source": "event_log",
        "intent": f"step {sid}",
        "confidence": "ok",
    }
    if script:
        return GroundedStep(**common, action_type="run_command", grounding="script", script=script)
    return GroundedStep(
        **common, action_type="navigate", grounding="ui_replay", ui_target=UITarget(description="x")
    )


def test_commands_with_terminal_focus_changes_between_become_one_script():
    steps = [g("a", "cd /x"), g("b"), g("c", "ls"), g("d", "git log")]
    out = group_terminal_commands(steps, [CMD("cd /x"), FOCUS, CMD("ls"), CMD("git log")])
    assert len(out) == 1
    assert out[0].script == "cd /x\nls\ngit log"
    assert out[0].source_step_ids == ["a", "b", "c", "d"]
    assert out[0].intent == "Run in one terminal session: `cd /x`; `ls`; `git log`"


def test_anything_else_ends_the_session():
    steps = [g("a", "cd /x"), g("b"), g("c", "ls")]
    out = group_terminal_commands(steps, [CMD("cd /x"), OTHER, CMD("ls")])
    assert [s.step_id for s in out] == ["a", "b", "c"]


def test_focus_changes_after_the_last_command_stay_separate():
    steps = [g("f"), g("a", "cd /x"), g("c", "ls"), g("b")]
    out = group_terminal_commands(steps, [FOCUS, CMD("cd /x"), CMD("ls"), FOCUS])
    assert [s.source_step_ids for s in out] == [["f"], ["a", "c"], ["b"]]


def test_a_single_command_is_left_as_it_is():
    steps = [g("a", "ls")]
    assert group_terminal_commands(steps, [CMD("ls")]) == steps
