"""CLIP-embedding fallback path for Stage 2 (Segment): detect action
boundaries directly from video when no event log exists (an imported,
pre-recorded session). ARCHITECTURE.md §Stage 2 describes CLIP-embedding
scene-change detection as an optional *secondary* signal alongside events;
here it's promoted to the only signal available, since no event-boundary
data exists for these sessions at all.

Deliberately the smallest reasonable CLIP checkpoint (ViT-B-32): this is a
cheap signal by design, not where quality effort should go. Upgrading to a
larger checkpoint (e.g. ViT-L-14) is the documented next lever if eval
numbers demand it — not the starting default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import cv2
import open_clip
import torch
from PIL import Image

from .schema import Segment

DEFAULT_CLIP_MODEL_NAME = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_SAMPLE_FPS = 2.0
# Literature/practice-informed starting point — NOT yet calibrated against
# real eval ground truth (none exists in this repo yet). Recalibrate with
# scripts/tune_segment_threshold.py once eval/ground_truth has real data.
DEFAULT_SIMILARITY_THRESHOLD = 0.90


@dataclass(frozen=True)
class BoundaryDetectionParams:
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED
    sample_fps: float = DEFAULT_SAMPLE_FPS
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    device: str | None = None  # None -> auto-detect (mps > cpu)


@dataclass(frozen=True)
class SampledFrame:
    t_ms: float
    frame_index: int
    image: Image.Image


def select_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_frames(video_path: Path, video_fps: float, sample_fps: float) -> list[SampledFrame]:
    """Decode sequentially and keep every Nth frame to hit ``sample_fps``.

    Sequential decode rather than seeking: seeking a compressed video to an
    arbitrary timestamp risks landing off-keyframe and decoding a smeared or
    stale frame, which would corrupt the embedding for that sample.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for frame sampling: {video_path}")
    stride = max(1, round(video_fps / sample_fps))
    frames: list[SampledFrame] = []
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                t_ms = frame_index / video_fps * 1000.0
                frames.append(SampledFrame(t_ms=t_ms, frame_index=frame_index, image=image))
            frame_index += 1
    finally:
        cap.release()
    return frames


@lru_cache(maxsize=4)
def _load_clip_model(clip_model_name: str, clip_pretrained: str, device: str):
    """Cached so a batch eval run over many sessions loads each checkpoint once."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained, device=device
    )
    model.eval()
    return model, preprocess


def embed_frames(
    frames: list[SampledFrame], *, clip_model_name: str, clip_pretrained: str, device: str
) -> torch.Tensor:
    """L2-normalized CLIP image embeddings, one row per frame, in input order."""
    model, preprocess = _load_clip_model(clip_model_name, clip_pretrained, device)
    with torch.no_grad():
        batch = torch.stack([preprocess(f.image) for f in frames]).to(device)
        embeddings = model.encode_image(batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def compute_similarities(embeddings: torch.Tensor) -> list[float]:
    """Cosine similarity between each pair of consecutive (already L2-normalized) embeddings."""
    return (embeddings[:-1] * embeddings[1:]).sum(dim=-1).tolist()


def segments_from_similarities(
    frames: list[SampledFrame], similarities: list[float], threshold: float, session_id: str
) -> list[Segment]:
    """Build segments from an already-computed similarity sequence and a threshold.

    Factored out of detect_boundaries() so a threshold sweep can reuse one
    sample+embed pass (the expensive part) across many threshold values —
    classifying boundaries from precomputed similarities is nearly free.
    """
    boundary_indices = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
    cut_points = [0, *boundary_indices, len(frames) - 1]

    segments: list[Segment] = []
    for i in range(len(cut_points) - 1):
        start_frame = frames[cut_points[i]]
        end_frame = frames[cut_points[i + 1]]
        segments.append(
            Segment(
                segment_id=f"{session_id}-{i:04d}",
                source="inferred",
                action_hint="scene_change",
                start_ms=start_frame.t_ms,
                end_ms=end_frame.t_ms,
                frame_start=start_frame.frame_index,
                frame_end=end_frame.frame_index,
                events=[],
            )
        )
    return segments


def detect_boundaries(
    video_path: Path,
    *,
    session_id: str,
    video_fps: float,
    params: BoundaryDetectionParams | None = None,
) -> tuple[list[Segment], BoundaryDetectionParams]:
    """Returns the detected segments and the params actually used (with
    ``device`` resolved to a concrete value), so callers can record exactly
    what produced the output without re-deriving the auto-detected device.
    """
    params = params or BoundaryDetectionParams()
    resolved_params = replace(params, device=select_device(params.device))

    frames = sample_frames(video_path, video_fps, resolved_params.sample_fps)
    if not frames:
        return [], resolved_params

    similarities: list[float] = []
    if len(frames) > 1:
        embeddings = embed_frames(
            frames,
            clip_model_name=resolved_params.clip_model_name,
            clip_pretrained=resolved_params.clip_pretrained,
            device=resolved_params.device,
        )
        similarities = compute_similarities(embeddings)

    segments = segments_from_similarities(frames, similarities, resolved_params.similarity_threshold, session_id)
    return segments, resolved_params
