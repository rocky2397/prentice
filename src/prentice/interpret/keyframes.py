"""Before/after keyframe extraction for Stage 3 (Interpret): pulls each
segment's frame_start/frame_end frame in a single decode pass over the video
(video_frames.extract_frames_at), then writes them out as JPEGs — mlx-vlm's
generate() takes image file paths, not in-memory images. Kept under the
session directory rather than a temp dir, so what the VLM actually saw for a
given step stays inspectable afterward — same reproducibility rationale as
segment_meta.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..segment.schema import Segment
from ..video_frames import extract_frames_at


@dataclass(frozen=True)
class SegmentKeyframes:
    before_path: Path
    after_path: Path


def extract_keyframes(
    video_path: Path, segments: list[Segment], output_dir: Path
) -> dict[str, SegmentKeyframes]:
    wanted_indices = {s.frame_start for s in segments} | {s.frame_end for s in segments}
    frames = extract_frames_at(video_path, wanted_indices)

    keyframes: dict[str, SegmentKeyframes] = {}
    for segment in segments:
        if segment.frame_start not in frames or segment.frame_end not in frames:
            raise RuntimeError(
                f"segment {segment.segment_id} references frames "
                f"{segment.frame_start}/{segment.frame_end} not found while decoding "
                f"{video_path} (video shorter than expected, or frame indices out of range)"
            )
        segment_dir = output_dir / segment.segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        before_path = segment_dir / "before.jpg"
        after_path = segment_dir / "after.jpg"
        frames[segment.frame_start].save(before_path)
        frames[segment.frame_end].save(after_path)
        keyframes[segment.segment_id] = SegmentKeyframes(before_path=before_path, after_path=after_path)
    return keyframes
