import shutil

import pytest
from PIL import Image

from _video_helpers import make_test_video
from prentice.interpret.keyframes import extract_keyframes
from prentice.segment.schema import Segment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _segment(segment_id: str, frame_start: int, frame_end: int) -> Segment:
    return Segment(
        segment_id=segment_id,
        source="inferred",
        action_hint="scene_change",
        start_ms=frame_start * 100.0,
        end_ms=frame_end * 100.0,
        frame_start=frame_start,
        frame_end=frame_end,
    )


def test_extract_keyframes_writes_before_and_after_jpegs(tmp_path):
    video_path = tmp_path / "video.mp4"
    make_test_video(video_path, duration=1.0, fps=10)
    segments = [_segment("s-0000", 0, 4), _segment("s-0001", 4, 9)]

    keyframes = extract_keyframes(video_path, segments, tmp_path / "keyframes")

    assert set(keyframes.keys()) == {"s-0000", "s-0001"}
    for kf in keyframes.values():
        assert kf.before_path.exists()
        assert kf.after_path.exists()
        # a real, openable JPEG — not just a file that happens to exist
        Image.open(kf.before_path).verify()
        Image.open(kf.after_path).verify()


def test_extract_keyframes_out_of_range_frame_raises(tmp_path):
    video_path = tmp_path / "video.mp4"
    make_test_video(video_path, duration=1.0, fps=10)
    segments = [_segment("s-0000", 0, 999)]

    with pytest.raises(RuntimeError):
        extract_keyframes(video_path, segments, tmp_path / "keyframes")


def test_extract_keyframes_no_segments_returns_empty(tmp_path):
    video_path = tmp_path / "video.mp4"
    make_test_video(video_path, duration=1.0, fps=10)

    keyframes = extract_keyframes(video_path, [], tmp_path / "keyframes")

    assert keyframes == {}
