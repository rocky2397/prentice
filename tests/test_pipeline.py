import json
import shutil
from pathlib import Path

import pytest

from _video_helpers import make_test_video
from prentice.capture.schema import KeyEvent, LiveCaptureManifest, MouseClickEvent
from prentice.segment.pipeline import segment_session

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _write_session(
    session_dir: Path,
    events: list,
    fps: float = 10.0,
    size: tuple[int, int] = (320, 240),
    duration: float = 3.0,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    video_path = session_dir / "screen.mp4"
    make_test_video(video_path, duration=duration, fps=int(fps), size=f"{size[0]}x{size[1]}")

    events_path = session_dir / "events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        f.writelines(e.model_dump_json() + "\n" for e in events)

    manifest = LiveCaptureManifest(
        session_id=session_dir.name,
        epoch0_utc="2026-01-01T00:00:00+00:00",
        fps=fps,
        video_width=size[0],
        video_height=size[1],
        video_path="screen.mp4",
        events_path="events.jsonl",
        has_events=True,
        os_version="test",
        screen_width=size[0],
        screen_height=size[1],
        backing_scale_factor=1.0,
        avfoundation_device_index=0,
        avfoundation_device_name="test",
    )
    (session_dir / "session.json").write_text(manifest.model_dump_json(indent=2))


def test_segment_session_event_log_path(tmp_path):
    events = [
        MouseClickEvent(t_ms=500.0, x=10.0, y=10.0, button="Button.left", pressed=True),
        MouseClickEvent(t_ms=520.0, x=10.0, y=10.0, button="Button.left", pressed=False),
        KeyEvent(t_ms=1500.0, key="a", pressed=True),
        KeyEvent(t_ms=1520.0, key="a", pressed=False),
    ]
    session_dir = tmp_path / "session1"
    _write_session(session_dir, events)

    segments_path = segment_session(session_dir)
    assert segments_path == session_dir / "segments.jsonl"

    lines = segments_path.read_text().strip().splitlines()
    assert len(lines) == 2
    segs = [json.loads(line) for line in lines]
    assert all(s["source"] == "event_log" for s in segs)
    assert {s["action_hint"] for s in segs} == {"click", "type"}

    meta = json.loads((session_dir / "segment_meta.json").read_text())
    assert meta["source"] == "event_log"
    assert meta["segment_count"] == 2


def test_segment_session_no_events_writes_empty_segments(tmp_path):
    session_dir = tmp_path / "session2"
    _write_session(session_dir, events=[])

    segments_path = segment_session(session_dir)
    assert segments_path.read_text().strip() == ""

    meta = json.loads((session_dir / "segment_meta.json").read_text())
    assert meta["source"] == "event_log"
    assert meta["segment_count"] == 0
