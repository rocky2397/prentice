import shutil

import pytest

from _video_helpers import make_test_video
from prentice.video_frames import extract_frames_at, iter_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_iter_frames_yields_every_frame_in_order(tmp_path):
    video_path = tmp_path / "video.mp4"
    make_test_video(video_path, duration=1.0, fps=10)

    frames = list(iter_frames(video_path))

    assert len(frames) == 10
    assert [index for index, _ in frames] == list(range(10))
    for _, image in frames:
        assert image.size == (320, 240)


def test_iter_frames_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        list(iter_frames(tmp_path / "does-not-exist.mp4"))


def test_extract_frames_at_returns_only_requested_indices(tmp_path):
    video_path = tmp_path / "video.mp4"
    make_test_video(video_path, duration=1.0, fps=10)

    frames = extract_frames_at(video_path, {0, 3, 9})

    assert set(frames.keys()) == {0, 3, 9}
    for image in frames.values():
        assert image.size == (320, 240)


def test_extract_frames_at_index_beyond_video_length_is_absent(tmp_path):
    video_path = tmp_path / "video.mp4"
    make_test_video(video_path, duration=1.0, fps=10)

    frames = extract_frames_at(video_path, {5, 999})

    assert set(frames.keys()) == {5}


def test_extract_frames_at_empty_request_does_not_decode(tmp_path):
    frames = extract_frames_at(tmp_path / "does-not-exist.mp4", set())
    assert frames == {}
