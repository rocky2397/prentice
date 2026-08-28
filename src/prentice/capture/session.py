"""Orchestrates Stage 1 (Capture): wires video + events + window tracking
together behind one explicit start/stop session, per ARCHITECTURE.md — no
always-on capture.

Also handles the video-only import path: wrapping a pre-recorded video
(from any source — QuickTime, OBS, a phone, etc.) into a session directory
with the same shape as a live capture, so later stages can consume either
uniformly. An imported session always has ``has_events=False`` — there is
no way to retroactively recover OS-level input events from a video this
tool didn't capture. Synthesizing a usable event log for such sessions
(via a vision model) is a separate, not-yet-built piece of Stage 2/3, not
part of this import step.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import signal
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

from AppKit import NSScreen

from .clock import ClockAnchor
from .events import EventLogger
from .schema import ImportedManifest, LiveCaptureManifest
from .video import (
    FFmpegNotFoundError,
    FFmpegScreenRecorder,
    find_screen_device,
    list_avfoundation_video_devices,
    probe_video,
)
from .window_tracker import WindowTracker

DEFAULT_SESSIONS_DIR = Path("eval/recordings")


def _screen_geometry() -> tuple[int, int, float]:
    """Logical point size + backing scale factor of the main screen.

    pynput and the AX API both report coordinates in points, but avfoundation
    captures at native pixel resolution (e.g. 2x on Retina displays). Kept
    here as informational metadata explaining the video's pixel dimensions —
    ``video_width``/``video_height`` in the manifest (from probing the actual
    recorded file) are the authoritative values downstream stages should use.
    """
    screen = NSScreen.mainScreen()
    frame = screen.frame()
    return int(frame.size.width), int(frame.size.height), float(screen.backingScaleFactor())


def _new_session_id(suffix: str = "") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}{suffix}"


def start_session(output_dir: Path, fps: int = 30) -> Path:
    session_id = _new_session_id()
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    video_path = session_dir / "screen.mp4"
    events_path = session_dir / "events.jsonl"
    manifest_path = session_dir / "session.json"

    device = find_screen_device()
    screen_width, screen_height, backing_scale_factor = _screen_geometry()
    anchor = ClockAnchor.now()

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

    probe = probe_video(video_path)
    manifest = LiveCaptureManifest(
        session_id=session_id,
        epoch0_utc=datetime.fromtimestamp(anchor.epoch0_utc, tz=UTC).isoformat(),
        fps=probe.fps,
        video_width=probe.width,
        video_height=probe.height,
        video_path=video_path.name,
        events_path=events_path.name,
        has_events=True,
        os_version=platform.mac_ver()[0],
        screen_width=screen_width,
        screen_height=screen_height,
        backing_scale_factor=backing_scale_factor,
        avfoundation_device_index=device.index,
        avfoundation_device_name=device.name,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    print(f"[prentice] saved session to {session_dir}")

    return session_dir


def import_session(video_path: Path, output_dir: Path) -> Path:
    """Wrap an existing (pre-recorded) video into a valid session directory.

    Copies the video in (never touches or moves the original) and writes an
    empty ``events.jsonl`` alongside an ``ImportedManifest`` — explicitly
    flagged ``has_events=False`` rather than silently pretending an
    event-quality log exists.
    """
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"no such video file: {video_path}")

    probe = probe_video(video_path)

    session_id = _new_session_id(suffix="-imported")
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dest_video = session_dir / f"screen{video_path.suffix}"
    shutil.copy2(video_path, dest_video)

    events_path = session_dir / "events.jsonl"
    events_path.touch()

    manifest = ImportedManifest(
        session_id=session_id,
        fps=probe.fps,
        video_width=probe.width,
        video_height=probe.height,
        video_path=dest_video.name,
        events_path=events_path.name,
        has_events=False,
        original_video_path=str(video_path),
        imported_at_utc=datetime.now(UTC).isoformat(),
    )
    (session_dir / "session.json").write_text(manifest.model_dump_json(indent=2))
    print(f"[prentice] imported session: {session_dir}")
    print("[prentice] no event log — Stage 2+ will need to fall back to a vision-only path for this session")

    return session_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prentice-capture")
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start", help="start a capture session (Ctrl+C to stop)")
    start_parser.add_argument("--fps", type=int, default=30)
    start_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SESSIONS_DIR)

    import_parser = sub.add_parser(
        "import", help="wrap a pre-recorded video (no event log) into a session directory"
    )
    import_parser.add_argument("video_path", type=Path)
    import_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SESSIONS_DIR)

    sub.add_parser("devices", help="list avfoundation video devices")

    args = parser.parse_args(argv)

    try:
        if args.command == "start":
            start_session(args.output_dir, fps=args.fps)
        elif args.command == "import":
            import_session(args.video_path, args.output_dir)
        elif args.command == "devices":
            for d in list_avfoundation_video_devices():
                print(f"[{d.index}] {d.name}")
    except (FFmpegNotFoundError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"[prentice] error: {exc}") from None


if __name__ == "__main__":
    main()
