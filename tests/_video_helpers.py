"""Synthetic-video generators shared by capture/segment tests, so tests
don't depend on real screen recordings or macOS permissions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def make_test_video(path: Path, duration: float = 1.0, fps: int = 10, size: str = "320x240") -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size={size}:rate={fps}",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def make_two_scene_test_video(
    path: Path, scene_duration: float = 1.5, fps: int = 10, size: str = "320x240"
) -> None:
    """A moving test pattern for the first half, then a hard cut to a static
    solid color for the second half — an unambiguous visual boundary, used
    to sanity-check CLIP-based boundary detection end to end.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={scene_duration}:size={size}:rate={fps}",
            "-f", "lavfi", "-i", f"color=c=blue:duration={scene_duration}:size={size}:rate={fps}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
            "-map", "[outv]",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
