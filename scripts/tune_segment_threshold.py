"""Sweep the CLIP similarity threshold against hand-labeled ground-truth
segment boundaries for one imported (no-event-log) session, and report
precision/recall/F1 per threshold.

This is scaffolding: DEFAULT_SIMILARITY_THRESHOLD in clip_boundary_detection.py
is a literature-informed guess, not a validated number, because eval/ground_truth
has no real hand-labeled recordings yet (see eval/README.md). Run this once
that exists to pick and record an actual default.

Usage:
    uv run python scripts/tune_segment_threshold.py \\
        eval/recordings/<imported-session-id> \\
        eval/ground_truth/<task>/boundaries.json

``boundaries.json`` is a JSON list of ground-truth boundary timestamps in
milliseconds: [1234.0, 5678.0, ...].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prentice.capture.schema import load_manifest
from prentice.segment.boundary_eval import score_boundaries
from prentice.segment.clip_boundary_detection import (
    DEFAULT_CLIP_MODEL_NAME,
    DEFAULT_CLIP_PRETRAINED,
    DEFAULT_SAMPLE_FPS,
    compute_similarities,
    embed_frames,
    sample_frames,
    select_device,
)

TOLERANCE_MS = 500.0
THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]  # 0.80 .. 0.99


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path, help="an imported session directory to tune against")
    parser.add_argument("ground_truth_json", type=Path, help="JSON list of ground-truth boundary timestamps (ms)")
    args = parser.parse_args()

    manifest = load_manifest(str(args.session_dir / "session.json"))
    video_path = args.session_dir / manifest.video_path
    ground_truth = json.loads(args.ground_truth_json.read_text())

    device = select_device(None)
    print(f"[tune] device: {device}")
    frames = sample_frames(video_path, manifest.fps, DEFAULT_SAMPLE_FPS)
    print(f"[tune] sampled {len(frames)} frames at {DEFAULT_SAMPLE_FPS} fps")
    embeddings = embed_frames(
        frames,
        clip_model_name=DEFAULT_CLIP_MODEL_NAME,
        clip_pretrained=DEFAULT_CLIP_PRETRAINED,
        device=device,
    )
    similarities = compute_similarities(embeddings)

    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    best_threshold, best_score = None, None
    for threshold in THRESHOLDS:
        predicted = [frames[i + 1].t_ms for i, sim in enumerate(similarities) if sim < threshold]
        score = score_boundaries(predicted, ground_truth, TOLERANCE_MS)
        print(f"{threshold:>10.2f} {score.precision:>10.3f} {score.recall:>10.3f} {score.f1:>10.3f}")
        if best_score is None or score.f1 > best_score.f1:
            best_threshold, best_score = threshold, score

    print(f"\nbest threshold: {best_threshold} (F1={best_score.f1:.3f})")


if __name__ == "__main__":
    main()
