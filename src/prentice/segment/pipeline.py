"""Stage 2 (Segment) entrypoint: branches on the session manifest's
``has_events`` to pick the event-log clustering path or the CLIP
boundary-detection fallback, and writes ``segments.jsonl`` +
``segment_meta.json`` into the session directory. Per ARCHITECTURE.md,
Stage 3 onward must be able to consume either path's output identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..capture.schema import load_events, load_manifest
from .clip_boundary_detection import BoundaryDetectionParams, detect_boundaries
from .event_clustering import ClusteringParams, cluster_events
from .schema import EventLogSegmentMeta, InferredSegmentMeta, Segment, SegmentRunMeta


def segment_session(session_dir: Path) -> Path:
    manifest = load_manifest(str(session_dir / "session.json"))
    video_path = session_dir / manifest.video_path

    segments: list[Segment]
    meta: SegmentRunMeta

    if manifest.has_events:
        events = load_events(str(session_dir / manifest.events_path))
        params = ClusteringParams()
        segments = cluster_events(
            events,
            session_id=manifest.session_id,
            fps=manifest.fps,
            duration_ms=manifest.duration_ms,
            params=params,
        )
        meta = EventLogSegmentMeta(
            session_id=manifest.session_id,
            segment_count=len(segments),
            drag_pixel_threshold=params.drag_pixel_threshold,
            scroll_gap_ms=params.scroll_gap_ms,
            type_gap_ms=params.type_gap_ms,
            frame_pad_ms=params.frame_pad_ms,
        )
    else:
        segments, resolved_params = detect_boundaries(
            video_path,
            session_id=manifest.session_id,
            video_fps=manifest.fps,
            params=BoundaryDetectionParams(),
        )
        meta = InferredSegmentMeta(
            session_id=manifest.session_id,
            segment_count=len(segments),
            clip_model_name=resolved_params.clip_model_name,
            clip_pretrained=resolved_params.clip_pretrained,
            sample_fps=resolved_params.sample_fps,
            similarity_threshold=resolved_params.similarity_threshold,
            device=resolved_params.device,
        )

    segments_path = session_dir / "segments.jsonl"
    with open(segments_path, "w", encoding="utf-8") as f:
        f.writelines(segment.model_dump_json() + "\n" for segment in segments)

    meta_path = session_dir / "segment_meta.json"
    meta_path.write_text(meta.model_dump_json(indent=2))

    print(f"[prentice] wrote {len(segments)} segments (source={meta.source}) to {segments_path}")
    return segments_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prentice-segment")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args(argv)
    segment_session(args.session_dir)


if __name__ == "__main__":
    main()
