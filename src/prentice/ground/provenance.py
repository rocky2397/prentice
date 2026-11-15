"""Recovers capture-time grounding evidence for a refined step.

ARCHITECTURE.md §Stage 5 requires a UI-replay action to store the
accessibility-tree identifier where one was captured, and permits raw
coordinates as a last-resort fallback hint. Stage 1 captures both (an
``ax_element`` on every click, plus the click's x/y), but Stage 3's ``Step``
has no field for either, so they don't reach Stage 5 by simply reading
``refined_steps.jsonl``.

They are still fully recoverable, because every stage kept its traceability
link: RefinedStep.source_step_ids -> Step.step_id -> Step.segment_id ->
Segment.events -> MouseClickEvent.ax_element / .x / .y. This module walks
that chain over the session directory's own on-disk output.

For an imported (video-only) session this correctly yields nothing at all:
``has_events`` is false, ``events.jsonl`` is empty, so every Segment carries
an empty event list and no AX identifier or coordinate exists to recover.
That's the known imported-session gap from ARCHITECTURE.md §2 showing up
exactly where it should, rather than being papered over.

**Terminal commands.** The same chain also yields the exact keystrokes behind
a step. When they were typed in a terminal app (the last window switch before
the step names one) and every line ends with Enter, the typed text *is* the
command that ran: concrete evidence from the event log, not the vision
model's transcription of it. Anything that makes the text uncertain (Tab
completion, arrow keys, shortcuts, an unfinished line, a window switch in
between) yields no command, and the step stays a UI-replay step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..capture.schema import KeyEvent, MouseClickEvent, WindowSwitchEvent, load_events
from ..interpret.schema import load_steps
from ..refine.schema import RefinedStep
from ..segment.schema import Segment, load_segments


@dataclass(frozen=True)
class StepEvidence:
    """Capture-time evidence backing one refined step, if any survives."""

    ax_role: str | None = None
    ax_title: str | None = None
    ax_description: str | None = None
    ax_value: str | None = None
    x: float | None = None
    y: float | None = None
    terminal_command: str | None = None  # exact typed command(s), one per line
    terminal_focus_only: bool = False  # nothing but window switches within a terminal app
    focus_only: bool = False  # nothing but window switches (any app): context, not an action


EMPTY_EVIDENCE = StepEvidence()

# Bundle ids of terminal apps: typing + Enter there runs a shell command.
TERMINAL_BUNDLES = frozenset(
    {
        "com.apple.Terminal",
        "com.googlecode.iterm2",
        "dev.warp.Warp-Stable",
        "com.mitchellh.ghostty",
        "net.kovidgoyal.kitty",
        "io.alacritty",
        "com.github.wez.wezterm",
    }
)
# Keys that don't change the typed text on their own.
_NEUTRAL_KEYS = frozenset({"Key.shift", "Key.shift_l", "Key.shift_r", "Key.caps_lock"})


def typed_commands(keys: list[KeyEvent]) -> str | None:
    """The command lines typed by these key presses, or None if unsure.

    Printable characters, space and backspace edit the line; Enter ends it.
    Any other key (Tab completion, arrows, a Cmd/Ctrl shortcut) means the
    final text can't be known from the keys alone, so nothing is returned;
    so does a line left without Enter.
    """
    line: list[str] = []
    commands: list[str] = []
    for event in keys:
        if not event.pressed:
            continue
        key = event.key
        if len(key) == 1:
            line.append(key)
        elif key == "Key.space":
            line.append(" ")
        elif key == "Key.backspace":
            if line:
                line.pop()
        elif key == "Key.enter":
            command = "".join(line).strip()
            if not command:
                return None
            commands.append(command)
            line = []
        elif key in _NEUTRAL_KEYS:
            continue
        else:
            return None
    if line or not commands:
        return None
    return "\n".join(commands)


def load_window_switches(session_dir: Path) -> list[WindowSwitchEvent]:
    """All window switches of a live session, in order (empty for imported ones)."""
    path = session_dir / "events.jsonl"
    if not path.is_file():
        return []
    return [e for e in load_events(str(path)) if isinstance(e, WindowSwitchEvent)]


def _terminal_command(
    segments: list[Segment], window_switches: list[WindowSwitchEvent]
) -> str | None:
    if not segments or not window_switches:
        return None
    start, end = segments[0].start_ms, segments[-1].end_ms
    before = [w for w in window_switches if w.t_ms <= start]
    if not before or before[-1].bundle_id not in TERMINAL_BUNDLES:
        return None
    if any(start < w.t_ms <= end for w in window_switches):
        return None  # focus moved while typing: keys may have gone elsewhere
    keys = [e for s in segments for e in s.events if isinstance(e, KeyEvent)]
    if any(isinstance(e, MouseClickEvent) for s in segments for e in s.events):
        return None  # a click could have moved the cursor or pasted
    return typed_commands(keys)


def load_segments_by_step_id(session_dir: Path) -> dict[str, Segment]:
    """Map each Stage 3 step_id to the Segment it was interpreted from.

    Returns an empty mapping if either file is missing — grounding must still
    work (falling back to the semantic description alone) for a session whose
    intermediate stage output was cleaned up.
    """
    steps_path = session_dir / "steps.jsonl"
    segments_path = session_dir / "segments.jsonl"
    if not steps_path.is_file() or not segments_path.is_file():
        return {}

    segments_by_id = {s.segment_id: s for s in load_segments(str(segments_path))}
    return {
        step.step_id: segments_by_id[step.segment_id]
        for step in load_steps(str(steps_path))
        if step.segment_id in segments_by_id
    }


def evidence_for(
    refined: RefinedStep,
    segments_by_step_id: dict[str, Segment],
    window_switches: list[WindowSwitchEvent] | None = None,
) -> StepEvidence:
    """Best available AX identifier / coordinate hint for a refined step.

    A refined step can absorb several Stage 3 steps, each with its own
    segment and events. The first click event carrying an accessibility
    element wins, since that's the strongest grounding signal available; if
    none has one, the first click still supplies a coordinate fallback hint.
    """
    segments = sorted(
        (segments_by_step_id[sid] for sid in refined.source_step_ids if sid in segments_by_step_id),
        key=lambda s: s.start_ms,
    )
    command = _terminal_command(segments, window_switches or [])
    events = [e for s in segments for e in s.events]
    focus_only = bool(events) and all(isinstance(e, WindowSwitchEvent) for e in events)
    terminal_focus_only = focus_only and all(e.bundle_id in TERMINAL_BUNDLES for e in events)
    click_events = [
        event
        for segment in segments
        for event in segment.events
        if isinstance(event, MouseClickEvent)
    ]
    if not click_events:
        if command or focus_only:
            return StepEvidence(
                terminal_command=command,
                terminal_focus_only=terminal_focus_only,
                focus_only=focus_only,
            )
        return EMPTY_EVIDENCE

    with_ax = next((e for e in click_events if e.ax_element is not None), None)
    chosen = with_ax or click_events[0]
    ax = chosen.ax_element
    return StepEvidence(
        ax_role=ax.role if ax else None,
        ax_title=ax.title if ax else None,
        ax_description=ax.description if ax else None,
        ax_value=ax.value if ax else None,
        x=chosen.x,
        y=chosen.y,
        terminal_command=command,
        terminal_focus_only=terminal_focus_only,
        focus_only=focus_only,
    )
