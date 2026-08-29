"""Shared low-level video-frame decoding — sequential access only, reused by
Stage 2's fixed-rate CLIP sampling and Stage 3's specific-index keyframe
extraction, so the decode loop exists in exactly one place.

Sequential decode throughout, never seek-based: seeking a compressed video to
an arbitrary timestamp risks landing off-keyframe and decoding a smeared or
stale frame, which would corrupt whatever reads that frame downstream.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
from PIL import Image


def iter_frames(video_path: Path) -> Iterator[tuple[int, Image.Image]]:
    """Yield (frame_index, image) for every frame, in order, decoding once."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for frame decoding: {video_path}")
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_index, Image.fromarray(frame_rgb)
            frame_index += 1
    finally:
        cap.release()


def extract_frames_at(video_path: Path, frame_indices: set[int]) -> dict[int, Image.Image]:
    """Pull specific frames by index in a single decode pass.

    One pass regardless of how many indices are requested — cheaper than
    decoding the video once per index, which matters when many segments each
    need a frame from the same video (Stage 3's before/after keyframes).
    """
    wanted = set(frame_indices)
    found: dict[int, Image.Image] = {}
    if not wanted:
        return found
    for frame_index, image in iter_frames(video_path):
        if frame_index in wanted:
            found[frame_index] = image
            wanted.discard(frame_index)
            if not wanted:
                break
    return found
