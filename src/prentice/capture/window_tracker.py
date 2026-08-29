"""Active app / window title tracking via polling.

Polling (every ``poll_interval`` seconds) rather than AXObserver callbacks
trades up to one poll interval of detection latency for a much simpler,
harder-to-deadlock implementation — an acceptable tradeoff for Stage 1
given how infrequently window focus changes relative to click/key events.
Known limitation: two switches faster than ``poll_interval`` apart may
coalesce into one logged event.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from AppKit import NSWorkspace

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


def _focused_window_title() -> str | None:
    """The focused window's title, via the same best-effort AX accessors
    accessibility.py uses for the under-cursor element lookup."""
    if not _AX_AVAILABLE:
        return None
    system_wide = AXUIElementCreateSystemWide()
    app = _copy_attr(system_wide, kAXFocusedApplicationAttribute)
    if app is None:
        return None
    window = _copy_attr(app, kAXFocusedWindowAttribute)
    if window is None:
        return None
    return _get_attr(window, kAXTitleAttribute)


class WindowTracker:
    """Polls the frontmost app + its focused window title and reports changes."""

    def __init__(self, on_change: Callable[[dict[str, Any]], None], poll_interval: float = 0.5):
        self.on_change = on_change
        self.poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last: tuple[str | None, str | None, str | None] | None = None

    def _poll_once(self) -> None:
        workspace = NSWorkspace.sharedWorkspace()
        app_info = workspace.frontmostApplication()
        app_name = app_info.localizedName() if app_info else None
        bundle_id = app_info.bundleIdentifier() if app_info else None
        window_title = _focused_window_title()
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
