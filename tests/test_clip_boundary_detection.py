"""Real, non-mocked test of the CLIP fallback path: generates a synthetic
two-scene video (a moving pattern, then a hard cut to a solid color) and
checks detect_boundaries finds a boundary near the cut.

Downloads a CLIP checkpoint (~600MB) on first run and does real inference,
so this is opt-in rather than part of the default `pytest` run — set
PRENTICE_TEST_CLIP=1 to include it.
"""

from __future__ import annotations

import os
import shutil

import pytest

from _video_helpers import make_two_scene_test_video
from prentice.segment.clip_boundary_detection import BoundaryDetectionParams, detect_boundaries

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not os.environ.get("PRENTICE_TEST_CLIP"),
    reason="requires ffmpeg and PRENTICE_TEST_CLIP=1 (downloads a CLIP checkpoint and runs real inference)",
)


def test_detect_boundaries_finds_the_scene_cut(tmp_path):
    fps = 10
    scene_duration = 1.5
    video_path = tmp_path / "two_scene.mp4"
    make_two_scene_test_video(video_path, scene_duration=scene_duration, fps=fps)

    segments, resolved_params = detect_boundaries(
        video_path,
        session_id="clip-test",
        video_fps=fps,
        params=BoundaryDetectionParams(sample_fps=fps),  # sample every frame given the short duration
    )

    assert len(segments) >= 2
    assert all(s.source == "inferred" for s in segments)
    assert all(s.action_hint == "scene_change" for s in segments)
    assert resolved_params.device in {"cpu", "mps", "cuda"}

    cut_ms = scene_duration * 1000.0
    boundary_times = [s.start_ms for s in segments[1:]]
    assert any(abs(t - cut_ms) < 500.0 for t in boundary_times)
