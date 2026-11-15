"""Active app / window title tracking via polling.

Polling (every ``poll_interval`` seconds) rather than AXObserver callbacks
trades up to one poll interval of detection latency for a much simpler,
harder-to-deadlock implementation — an acceptable tradeoff for Stage 1
given how infrequently window focus changes relative to click/key events.
Known limitation: two switches faster than ``poll_interval`` apart may
coalesce into one logged event.

The app comes from the on-screen window list (front to back, always
current): the owner of the frontmost normal window. The title comes from the
accessibility API's focused window. ``NSWorkspace.frontmostApplication``
looked simpler, but in a command-line process without a running AppKit event
loop it is never refreshed: every switch was logged as the terminal the
recorder ran in, even when the title said "… - Google Chrome". The AX focused
application can't stand in for it either: it fails outright in some contexts
(kAXErrorCannotComplete), and then the app would be lost.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from AppKit import NSRunningApplication

from .accessibility import _copy_attr, _get_attr

try:
    from ApplicationServices import (
        AXUIElementCreateSystemWide,
        kAXFocusedApplicationAttribute,
        kAXFocusedWindowAttribute,
        kAXTitleAttribute,
    )

    _AX_AVAILABLE = True
except ImportError:
    _AX_AVAILABLE = False


def _app_for_pid(pid: int) -> tuple[str | None, str | None]:
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if app is None:
        return None, None
    return app.localizedName(), app.bundleIdentifier()


def _front_window() -> dict | None:
    """The frontmost normal (layer 0) on-screen window, as the window server lists it."""
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID
    )
    return next((w for w in windows or [] if w.get("kCGWindowLayer") == 0), None)


def _ax_focused_title() -> str | None:
    if not _AX_AVAILABLE:
        return None
    app = _copy_attr(AXUIElementCreateSystemWide(), kAXFocusedApplicationAttribute)
    if app is None:
        return None
    window = _copy_attr(app, kAXFocusedWindowAttribute)
    return _get_attr(window, kAXTitleAttribute) if window is not None else None


def focused_app_and_title() -> tuple[str | None, str | None, str | None]:
    """(app name, bundle id, window title) of the window in front."""
    front = _front_window()
    if front is None:
        return None, None, _ax_focused_title()
    name, bundle = _app_for_pid(int(front["kCGWindowOwnerPID"]))
    title = _ax_focused_title() or front.get("kCGWindowName") or None
    return name or front.get("kCGWindowOwnerName"), bundle, title


class WindowTracker:
    """Polls the frontmost app + its focused window title and reports changes."""

    def __init__(self, on_change: Callable[[dict[str, Any]], None], poll_interval: float = 0.5):
        self.on_change = on_change
        self.poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last: tuple[str | None, str | None, str | None] | None = None

    def _poll_once(self) -> None:
        app_name, bundle_id, window_title = focused_app_and_title()
        current = (app_name, bundle_id, window_title)
        if current != self._last:
            self._last = current
            self.on_change(
                {
                    "type": "window_switch",
                    "app_name": app_name,
                    "bundle_id": bundle_id,
                    "window_title": window_title,
                }
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self.poll_interval)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
