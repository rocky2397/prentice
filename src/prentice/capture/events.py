"""Global mouse/keyboard event logging via pynput, timestamped against a ClockAnchor.

pynput's mouse and keyboard listeners each run on their own OS thread and
invoke these callbacks directly, so writes are serialized behind a lock to
keep JSONL lines from interleaving.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from pynput import keyboard, mouse

from .accessibility import describe_element_at
from .clock import ClockAnchor


class EventLogger:
    def __init__(self, events_path: Path, anchor: ClockAnchor, capture_ax_on_click: bool = True):
        self.events_path = events_path
        self.anchor = anchor
        self.capture_ax_on_click = capture_ax_on_click
        events_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(events_path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._mouse_listener: mouse.Listener | None = None
        self._keyboard_listener: keyboard.Listener | None = None

    def _write(self, record: dict[str, Any]) -> None:
        record = {"t_ms": round(self.anchor.elapsed_ms(), 3), **record}
        with self._lock:
            self._file.write(json.dumps(record) + "\n")

    def write_event(self, record: dict[str, Any]) -> None:
        """Public hook for non-pynput sources (e.g. WindowTracker) to log into the same stream."""
        self._write(record)

    def _on_click(self, x: float, y: float, button: Any, pressed: bool) -> None:
        record: dict[str, Any] = {
            "type": "mouse_click",
            "x": x,
            "y": y,
            "button": str(button),
            "pressed": pressed,
        }
        if pressed and self.capture_ax_on_click:
            element = describe_element_at(x, y)
            if element is not None:
                record["ax_element"] = element
        self._write(record)

    def _on_scroll(self, x: float, y: float, dx: float, dy: float) -> None:
        self._write({"type": "mouse_scroll", "x": x, "y": y, "dx": dx, "dy": dy})

    def _on_key_press(self, key: Any) -> None:
        self._write({"type": "key", "key": _key_repr(key), "pressed": True})

    def _on_key_release(self, key: Any) -> None:
        self._write({"type": "key", "key": _key_repr(key), "pressed": False})

    def start(self) -> None:
        self._mouse_listener = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll)
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self._mouse_listener.start()
        self._keyboard_listener.start()

    def stop(self) -> None:
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        with self._lock:
            self._file.flush()
            self._file.close()


def _key_repr(key: Any) -> str:
    try:
        return key.char if key.char is not None else str(key)
    except AttributeError:
        return str(key)
