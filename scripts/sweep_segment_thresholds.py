"""Sweep the CLIP similarity threshold over the imported (no-event-log)
sessions in eval/recordings/, without scoring against ground truth — there is
none, and per-project decision it isn't being hand-labeled. This just produces
each threshold's output, kept separate and distinguishable, for later manual
comparison (see scripts/tune_segment_threshold.py for scored calibration once
real ground truth exists).

Frame sampling + CLIP embedding is done once per video and reused across every
threshold — only the cheap boundary classification differs per threshold — so
this stays fast even over several threshold values.

Each threshold's output goes to its own subdirectory:

    eval/recordings/<session_id>/threshold_sweep/t<threshold>/segments.jsonl
    eval/recordings/<session_id>/threshold_sweep/t<threshold>/segment_meta.json

Usage:
    uv run python scripts/sweep_segment_thresholds.py
    uv run python scripts/sweep_segment_thresholds.py --thresholds 0.85 0.90 0.95
    uv run python scripts/sweep_segment_thresholds.py eval/recordings/<session-id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prentice.capture.schema import load_manifest
from prentice.segment.clip_boundary_detection import (
    BoundaryDetectionParams,
    compute_similarities,
    embed_frames,
    sample_frames,
    segments_from_similarities,
    select_device,
)
from prentice.segment.schema import InferredSegmentMeta

DEFAULT_THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.93, 0.95]
DEFAULT_RECORDINGS_DIR = Path("eval/recordings")


def _all_sessions(recordings_dir: Path) -> list[Path]:
    return sorted(p for p in recordings_dir.iterdir() if p.is_dir() and (p / "session.json").exists())


def sweep_session(session_dir: Path, thresholds: list[float]) -> None:
    manifest = load_manifest(str(session_dir / "session.json"))
    if manifest.has_events:
        print(f"[sweep] skipping {session_dir.name} — has an event log, CLIP sweep doesn't apply")
        return

    video_path = session_dir / manifest.video_path
    params = BoundaryDetectionParams()
    device = select_device(params.device)

    print(f"[sweep] {session_dir.name}: sampling + embedding once (device={device})")
    frames = sample_frames(video_path, manifest.fps, params.sample_fps)
    similarities: list[float] = []
    if len(frames) > 1:
        embeddings = embed_frames(
            frames, clip_model_name=params.clip_model_name, clip_pretrained=params.clip_pretrained, device=device
        )
        similarities = compute_similarities(embeddings)

    for threshold in thresholds:
        segments = segments_from_similarities(frames, similarities, threshold, manifest.session_id)
        out_dir = session_dir / "threshold_sweep" / f"t{threshold:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)

        segments_path = out_dir / "segments.jsonl"
        with open(segments_path, "w", encoding="utf-8") as f:
            f.writelines(segment.model_dump_json() + "\n" for segment in segments)

        meta = InferredSegmentMeta(
            session_id=manifest.session_id,
            segment_count=len(segments),
            clip_model_name=params.clip_model_name,
            clip_pretrained=params.clip_pretrained,
            sample_fps=params.sample_fps,
            similarity_threshold=threshold,
            device=device,
        )
        (out_dir / "segment_meta.json").write_text(meta.model_dump_json(indent=2))

        print(f"[sweep]   t={threshold:.2f}: {len(segments)} segments -> {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "session_dirs",
        type=Path,
        nargs="*",
        help="specific session directories (default: all sessions under eval/recordings/)",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args(argv)

    session_dirs = args.session_dirs or _all_sessions(DEFAULT_RECORDINGS_DIR)
    for session_dir in session_dirs:
        sweep_session(session_dir, args.thresholds)


if __name__ == "__main__":
    main()
