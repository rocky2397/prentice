"""Orchestrates Stage 1 (Capture): wires video + events + window tracking
together behind one explicit start/stop session, per ARCHITECTURE.md — no
always-on capture.
"""

from __future__ import annotations

import argparse
import platform
import signal
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

from AppKit import NSScreen

from .clock import ClockAnchor
from .events import EventLogger
from .schema import SessionManifest
from .video import FFmpegScreenRecorder, find_screen_device, list_avfoundation_video_devices
from .window_tracker import WindowTracker

DEFAULT_SESSIONS_DIR = Path("eval/recordings")


def _screen_geometry() -> tuple[int, int, float]:
    """Logical point size + backing scale factor of the main screen.

    pynput and the AX API both report coordinates in points, but avfoundation
    captures at native pixel resolution (e.g. 2x on Retina displays) — so
    ``video_width == screen_width * backing_scale_factor``, not
    ``video_width == screen_width``. Downstream stages need this factor to
    map an event's (x, y) onto the matching video frame.
    """
    screen = NSScreen.mainScreen()
    frame = screen.frame()
    return int(frame.size.width), int(frame.size.height), float(screen.backingScaleFactor())


def start_session(output_dir: Path, fps: int = 30) -> Path:
    session_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    video_path = session_dir / "screen.mp4"
    events_path = session_dir / "events.jsonl"
    manifest_path = session_dir / "session.json"

    device = find_screen_device()
    width, height, backing_scale_factor = _screen_geometry()
    anchor = ClockAnchor.now()

    manifest = SessionManifest(
        session_id=session_id,
        epoch0_utc=datetime.fromtimestamp(anchor.epoch0_utc, tz=UTC).isoformat(),
        fps=fps,
        screen_width=width,
        screen_height=height,
        backing_scale_factor=backing_scale_factor,
        os_version=platform.mac_ver()[0],
        video_path=video_path.name,
        events_path=events_path.name,
        avfoundation_device_index=device.index,
        avfoundation_device_name=device.name,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    recorder = FFmpegScreenRecorder(video_path, device.index, fps=fps)
    logger = EventLogger(events_path, anchor)
    tracker = WindowTracker(on_change=logger.write_event)

    print(f"[prentice] session: {session_id}")
    print(f"[prentice] recording to: {session_dir}")
    print("[prentice] press Ctrl+C to stop")

    recorder.start()
    logger.start()
    tracker.start()

    stop_flag = {"stop": False}

    def _handle_sigint(signum: int, frame: FrameType | None) -> None:
        stop_flag["stop"] = True

    previous_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not stop_flag["stop"]:
            time.sleep(0.1)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        print("[prentice] stopping...")
        tracker.stop()
        logger.stop()
        recorder.stop()
        print(f"[prentice] saved session to {session_dir}")

    return session_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prentice-capture")
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start", help="start a capture session (Ctrl+C to stop)")
    start_parser.add_argument("--fps", type=int, default=30)
    start_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SESSIONS_DIR)

    sub.add_parser("devices", help="list avfoundation video devices")

    args = parser.parse_args(argv)

    if args.command == "start":
        start_session(args.output_dir, fps=args.fps)
    elif args.command == "devices":
        for d in list_avfoundation_video_devices():
            print(f"[{d.index}] {d.name}")


if __name__ == "__main__":
    main()
