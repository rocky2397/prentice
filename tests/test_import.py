import json
import shutil
import subprocess

import pytest

from prentice.capture.session import import_session

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_test_video(path, duration=1, fps=10, size="320x240"):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size={size}:rate={fps}",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_import_session_wraps_pre_recorded_video(tmp_path):
    source = tmp_path / "source.mp4"
    _make_test_video(source)

    session_dir = import_session(source, tmp_path / "sessions")

    manifest = json.loads((session_dir / "session.json").read_text())
    assert manifest["source"] == "imported"
    assert manifest["has_events"] is False
    assert manifest["video_width"] == 320
    assert manifest["video_height"] == 240
    assert manifest["original_video_path"] == str(source)

    events_path = session_dir / manifest["events_path"]
    assert events_path.exists()
    assert events_path.stat().st_size == 0

    video_path = session_dir / manifest["video_path"]
    assert video_path.exists()
    # the source file must be left untouched, not moved
    assert source.exists()


def test_import_session_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_session(tmp_path / "does-not-exist.mp4", tmp_path / "sessions")
