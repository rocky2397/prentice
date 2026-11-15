"""A scripted live test: the computer does the eval tasks while prentice records.

    uv run prentice-scripted-run            # all three tasks
    uv run prentice-scripted-run --dry-run  # check everything, touch nothing

For each task this starts a real live capture (`prentice-capture start`),
drives the apps with synthetic mouse and keyboard input (the same OS events a
person produces, so the event logger and accessibility lookups see them), stops
the capture, checks whether the task succeeded, and writes an exact answer key
into the session folder (`answer_key.json`, status "scripted"). Nobody has to
label anything: the script knows which steps it did.

Honest limit: a script clicks and types more cleanly than a person (no
hesitation, no corrections, no wandering mouse), so these numbers are on the
optimistic side of what a human recording would give.

Safety: press Esc at any time to abort; the capture is stopped and nothing
else happens. The Finder task works only inside ~/prentice-sandbox, which this
script creates and removes. Nothing typed is a secret, but the whole screen is
recorded: close or hide private windows first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RECORDINGS = REPO / "eval" / "recordings"
SANDBOX = Path.home() / "prentice-sandbox"
DUMMY = "prentice-dummy.txt"
PAUSE = 1.0  # seconds between actions: about a person's pace


class Aborted(Exception):
    """The person pressed Esc."""


# ---------------------------------------------------------------- answer keys


@dataclass
class KeyStep:
    what: str
    action_types: list[str]
    keywords: list[str]
    expect_param: str | None = None
    optional: bool = False
    at_unix_ms: int | None = None  # when the script did it


@dataclass
class Plan:
    task: str
    steps: list[KeyStep] = field(default_factory=list)

    def answer_key(self, session: str, success: bool) -> dict:
        return {
            "task": self.task,
            "status": "scripted",
            "verified_against": session,
            "note": "Written by prentice-scripted-run: exactly the steps the script performed.",
            "task_succeeded": success,
            "steps": [
                {
                    "id": i + 1,
                    "what": s.what,
                    "action_types": s.action_types,
                    "keywords": s.keywords,
                    "expect_param": s.expect_param,
                    "optional": s.optional,
                    "at_unix_ms": s.at_unix_ms,
                }
                for i, s in enumerate(self.steps)
            ],
        }


# ------------------------------------------------------------- input driving


class Driver:
    """Synthetic input through pynput, with an Esc watchdog."""

    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self.kb = keyboard.Controller()
        self.mouse = mouse.Controller()
        self.Key = keyboard.Key
        self.Button = mouse.Button
        self.aborted = threading.Event()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key) -> None:
        if key == self.Key.esc:
            self.aborted.set()

    def stop(self) -> None:
        self._listener.stop()

    def check(self) -> None:
        if self.aborted.is_set():
            raise Aborted

    def wait(self, seconds: float = PAUSE) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.check()
            time.sleep(0.05)

    def type(self, text: str) -> None:
        for ch in text:
            self.check()
            self.kb.type(ch)
            time.sleep(0.04)

    def press(self, key) -> None:
        self.check()
        self.kb.press(key)
        self.kb.release(key)

    def combo(self, *keys) -> None:
        """e.g. combo(Key.cmd, "l"): hold all but the last, tap the last."""
        self.check()
        *mods, last = keys
        for m in mods:
            self.kb.press(m)
        self.kb.press(last)
        self.kb.release(last)
        for m in reversed(mods):
            self.kb.release(m)

    def move(self, x: float, y: float, steps: int = 20) -> None:
        sx, sy = self.mouse.position
        for i in range(1, steps + 1):
            self.check()
            self.mouse.position = (sx + (x - sx) * i / steps, sy + (y - sy) * i / steps)
            time.sleep(0.015)

    def click(self, x: float, y: float, count: int = 1) -> None:
        self.move(x, y)
        self.wait(0.2)
        self.mouse.click(self.Button.left, count)

    def drag(self, src: tuple[float, float], dst: tuple[float, float]) -> None:
        self.move(*src)
        self.wait(0.3)
        self.mouse.press(self.Button.left)
        self.wait(0.4)  # Finder needs a moment before a press becomes a drag
        self.move(*dst, steps=40)
        self.wait(0.6)
        self.mouse.release(self.Button.left)

    def scroll(self, ticks: int) -> None:
        for _ in range(ticks):
            self.check()
            self.mouse.scroll(0, -1)
            time.sleep(0.12)


def osascript(script: str) -> str:
    out = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=30, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"AppleScript failed: {out.stderr.strip()}")
    return out.stdout.strip()


def find_center(
    app: str, match: Callable[[str], bool], timeout: float = 6.0
) -> tuple[float, float]:
    """Screen centre of the first element in app's main window whose text matches."""
    from AppKit import NSWorkspace
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCreateApplication,
        AXValueGetValue,
        kAXValueCGPointType,
        kAXValueCGSizeType,
    )

    def attr(el, name):
        err, val = AXUIElementCopyAttributeValue(el, name, None)
        return val if err == 0 else None

    running = [
        a for a in NSWorkspace.sharedWorkspace().runningApplications() if a.localizedName() == app
    ]
    if not running:
        raise RuntimeError(f"{app} is not running")
    root = AXUIElementCreateApplication(running[0].processIdentifier())
    deadline = time.time() + timeout
    while time.time() < deadline:
        start = attr(root, "AXMainWindow") or root
        queue = [(start, 0)]
        while queue:
            el, depth = queue.pop(0)
            for name in ("AXValue", "AXTitle", "AXFilename"):
                text = attr(el, name)
                if isinstance(text, str) and match(text):
                    pos, size = attr(el, "AXPosition"), attr(el, "AXSize")
                    if pos is not None and size is not None:
                        _, p = AXValueGetValue(pos, kAXValueCGPointType, None)
                        _, s = AXValueGetValue(size, kAXValueCGSizeType, None)
                        if s.width > 0 and s.height > 0:
                            return (p.x + s.width / 2, p.y + s.height / 2)
            if depth < 14:
                queue.extend((c, depth + 1) for c in (attr(el, "AXChildren") or []))
        time.sleep(0.3)
    raise RuntimeError(f"could not find an element matching in {app}")


# ---------------------------------------------------------------- the tasks


def prepare_terminal() -> None:
    """Open the task's Terminal window before recording starts.

    The shell gets a neutral prompt ("prentice %" instead of user@host) and a
    cleared screen, so the recording shows no user name, computer name, home
    path or login line. AppleScript passes this to the shell directly; it
    isn't typed, so it isn't part of the recorded task.
    """
    osascript('tell application "Terminal" to do script "export PROMPT=\'%1~ %# \'; clear"')
    osascript('tell application "Terminal" to set bounds of front window to {120, 120, 1000, 640}')
    time.sleep(1.5)


def terminal_git(d: Driver, plan: Plan) -> bool:
    osascript('tell application "Terminal" to activate')
    d.wait(1.5)
    for what, command, keywords, expect in [
        ("cd into the prentice folder", f'cd "{REPO}"', ["cd", "prentice"], "cd"),
        ("List the folder with ls", "ls", ["ls", "list"], "ls"),
        ("Show the last three commits", "git log --oneline -3", ["git log", "commit"], "git log"),
    ]:
        plan.steps.append(
            KeyStep(what, ["run_command", "type"], keywords, expect, at_unix_ms=_now())
        )
        d.type(command)
        d.press(d.Key.enter)
        d.wait(1.5)
    contents = osascript(
        'tell application "Terminal" to get contents of selected tab of front window'
    )
    return "--oneline" in contents and "fatal" not in contents


def prepare_browser(browser: str) -> None:
    """Open the task's browser window before recording starts.

    Chrome gets a guest window: no history (so typing a URL shows no
    suggestions from your browsing), no shortcuts, bookmarks or extensions,
    and its data is deleted when it closes. Safari has no guest mode, so a
    Safari recording can show your favourites and history; prefer Chrome for
    anything you'll share.
    """
    if browser == "Safari":
        osascript('tell application "Safari" to make new document')
    else:
        subprocess.run(["open", "-na", browser, "--args", "--guest"], check=True)
        time.sleep(3)
    osascript(
        f'tell application "{browser}" to set bounds of front window to {{140, 90, 1280, 860}}'
    )
    time.sleep(1)


def close_task_window(browser: str) -> None:
    """Close the window this task used, matched by its page title, never another one."""
    if browser == "Safari":
        return
    try:
        osascript(
            f'tell application "{browser}" to close (every window whose title contains "Large language model")'
        )
    except RuntimeError:
        pass  # already closed, or the title didn't match: leave everything as it is


def browser_wikipedia(d: Driver, plan: Plan, browser: str) -> bool:
    osascript(f'tell application "{browser}" to activate')
    d.wait(1.5)

    plan.steps.append(
        KeyStep(
            "Go to en.wikipedia.org",
            ["navigate", "type"],
            ["wikipedia"],
            "wikipedia.org",
            at_unix_ms=_now(),
        )
    )
    d.combo(d.Key.cmd, "l")
    d.wait(0.5)
    d.type("en.wikipedia.org")
    d.press(d.Key.enter)
    d.wait(4)

    plan.steps.append(
        KeyStep(
            'Search for "Large language model"',
            ["type", "click"],
            ["search", "large language model"],
            "large language model",
            at_unix_ms=_now(),
        )
    )
    d.combo(d.Key.ctrl, d.Key.alt, "f")  # Wikipedia's access key for the search field
    d.wait(0.8)
    d.type("Large language model")
    d.press(d.Key.enter)
    d.wait(4)

    plan.steps.append(
        KeyStep(
            "Scroll down the article",
            ["scroll"],
            ["scroll", "history", "article"],
            at_unix_ms=_now(),
        )
    )
    d.move(700, 500)
    d.scroll(25)
    d.wait(1.5)

    if browser == "Safari":
        url = osascript('tell application "Safari" to get URL of current tab of front window')
    else:
        url = osascript(f'tell application "{browser}" to get URL of active tab of front window')
    return "Large_language_model" in url


def finder_folder(d: Driver, plan: Plan) -> bool:
    osascript(
        f'tell application "Finder" to make new Finder window to (POSIX file "{SANDBOX}" as alias)'
    )
    osascript('tell application "Finder" to set current view of front Finder window to list view')
    osascript(
        'tell application "Finder" to set bounds of front Finder window to {160, 140, 1060, 640}'
    )
    osascript('tell application "Finder" to activate')
    d.wait(1.5)

    plan.steps.append(
        KeyStep(
            "Create a new folder", ["type", "click"], ["new folder", "folder"], at_unix_ms=_now()
        )
    )
    d.combo(d.Key.cmd, d.Key.shift, "n")
    d.wait(1.2)

    plan.steps.append(
        KeyStep(
            "Name the folder prentice-test",
            ["type"],
            ["prentice-test", "name"],
            "prentice-test",
            at_unix_ms=_now(),
        )
    )
    d.type("prentice-test")
    d.press(d.Key.enter)
    d.wait(1.2)

    plan.steps.append(
        KeyStep(
            "Drag prentice-dummy.txt into prentice-test",
            ["drag"],
            ["prentice-dummy", "drag", "move"],
            at_unix_ms=_now(),
        )
    )
    src = find_center("Finder", lambda t: t.startswith("prentice-dummy"))
    dst = find_center("Finder", lambda t: t == "prentice-test")
    d.drag(src, dst)
    d.wait(1.2)

    plan.steps.append(
        KeyStep("Open prentice-test", ["click"], ["prentice-test", "open"], at_unix_ms=_now())
    )
    d.click(*find_center("Finder", lambda t: t == "prentice-test"), count=2)
    d.wait(1.5)

    plan.steps.append(
        KeyStep(
            "Rename the file to prentice-renamed.txt",
            ["type", "click"],
            ["rename", "prentice-renamed"],
            "prentice-renamed",
            at_unix_ms=_now(),
        )
    )
    d.click(*find_center("Finder", lambda t: t.startswith("prentice-dummy")))
    d.wait(0.6)
    d.press(d.Key.enter)
    d.wait(0.8)
    d.type("prentice-renamed")
    d.press(d.Key.enter)
    d.wait(1.2)

    osascript('tell application "Finder" to close front Finder window')
    return (SANDBOX / "prentice-test" / "prentice-renamed.txt").is_file()


def _now() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------- the run


def prepare_sandbox() -> None:
    if SANDBOX.exists():
        remove_sandbox()
    SANDBOX.mkdir()
    (SANDBOX / DUMMY).write_text("test\n")


def remove_sandbox() -> None:
    """Delete ~/prentice-sandbox, but only if it holds nothing this script didn't make."""
    allowed = {DUMMY, "prentice-test", "prentice-renamed.txt", ".DS_Store"}
    for p in SANDBOX.rglob("*"):
        if p.name not in allowed:
            raise RuntimeError(
                f"{SANDBOX} contains {p.name}, which this script didn't create; left in place"
            )
    shutil.rmtree(SANDBOX)


def start_capture(
    timeout: float = 20.0, exe: Path | None = None, settle: float = 2.5
) -> tuple[subprocess.Popen, str]:
    """Start `prentice-capture start`; fail with its output if it isn't recording in time."""
    import queue

    exe = exe or Path(sys.executable).parent / "prentice-capture"
    proc = subprocess.Popen(
        [str(exe), "start", "--output-dir", str(RECORDINGS)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Without this the child's prints sit in a pipe buffer and we'd wait forever.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def drain() -> None:  # keeps reading for the whole capture, so the pipe never fills
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=drain, daemon=True).start()
    seen: list[str] = []
    session = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is None:
            break  # the recorder exited
        seen.append(line.rstrip())
        if line.startswith("[prentice] session:"):
            session = line.split(":", 1)[1].strip()
        if "press Ctrl+C to stop" in line and session:
            time.sleep(settle)  # let the screen recorder settle before the first action
            return proc, session
    proc.kill()
    raise RuntimeError(
        "prentice-capture did not start recording. Its output:\n  "
        + "\n  ".join(seen or ["(nothing)"])
    )


def stop_capture(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


TASKS = ["terminal-git", "browser-wikipedia", "finder-folder"]


def run(tasks: list[str], browser: str) -> list[dict]:
    d = Driver()
    results = []
    try:
        for task in tasks:
            if task == "finder-folder":
                prepare_sandbox()
            plan = Plan(task)
            if task == "terminal-git":
                prepare_terminal()
            elif task == "browser-wikipedia":
                prepare_browser(browser)
            proc, session = start_capture()
            success = False
            try:
                if task == "terminal-git":
                    success = terminal_git(d, plan)
                elif task == "browser-wikipedia":
                    success = browser_wikipedia(d, plan, browser)
                else:
                    success = finder_folder(d, plan)
            finally:
                stop_capture(proc)
                session_dir = RECORDINGS / session
                (session_dir / "answer_key.json").write_text(
                    json.dumps(plan.answer_key(session, success), indent=2) + "\n"
                )
                if task == "finder-folder" and SANDBOX.exists():
                    remove_sandbox()
                if task == "browser-wikipedia":
                    close_task_window(browser)
            results.append({"task": task, "session": session, "succeeded": success})
            print(
                f"[scripted] {task}: {'succeeded' if success else 'DID NOT succeed'} -> {session}"
            )
            d.wait(2)
    except Aborted:
        print("[scripted] aborted with Esc; the capture of the current task was stopped")
    finally:
        d.stop()
    return results


TRUST_HELP = """\
This terminal app isn't trusted for Accessibility, so the script can't send or
watch mouse and keyboard input. In System Settings > Privacy & Security, make
sure the terminal you're using is switched ON under Accessibility and under
Input Monitoring (if it's already on, switch it off and on again), then quit the
terminal completely (Command-Q), reopen it and run this again."""


def trusted() -> bool:
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:
        return False
    return bool(AXIsProcessTrusted())


def dry_run(browser: str, tasks: list[str] = TASKS) -> int:
    ok = True

    def check(label: str, passed: bool, hint: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(
            f"[{' OK ' if passed else 'FAIL'}] {label}"
            + (f": {hint}" if hint and not passed else "")
        )

    check("prentice-capture installed", (Path(sys.executable).parent / "prentice-capture").exists())
    check("ffmpeg on PATH", shutil.which("ffmpeg") is not None, "brew install ffmpeg")
    check("Accessibility granted to this terminal", trusted(), "\n" + TRUST_HELP)
    for app in ["Terminal", "Finder", browser]:
        found = (
            subprocess.run(
                ["osascript", "-e", f'id of application "{app}"'], capture_output=True, check=False
            ).returncode
            == 0
        )
        check(f"{app} installed", found)
    check(f"{SANDBOX} free to use", not SANDBOX.exists() or SANDBOX.is_dir())
    print()
    print("Would run, in order:", ", ".join(tasks), f"(browser: {browser}); nothing was touched.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Let the computer do the eval tasks while prentice records."
    )
    parser.add_argument(
        "--task", choices=TASKS, action="append", help="run only this task (repeatable)"
    )
    parser.add_argument(
        "--browser", default="Google Chrome", help='"Google Chrome" (default) or "Safari"'
    )
    parser.add_argument("--dry-run", action="store_true", help="check prerequisites, touch nothing")
    args = parser.parse_args(argv)

    if args.dry_run:
        sys.exit(dry_run(args.browser, args.task or TASKS))

    if not trusted():
        sys.exit(TRUST_HELP)

    print(
        "[scripted] in 5 seconds the computer takes over mouse and keyboard for about 3 minutes.\n"
        "[scripted] don't touch anything; press Esc to abort. Private windows should be closed or hidden.\n"
        "[scripted] the browser task uses a Chrome guest window (no history, bookmarks or extensions)."
    )
    time.sleep(5)
    results = run(args.task or TASKS, args.browser)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
