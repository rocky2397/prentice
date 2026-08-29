"""Stage 3 (Interpret) entrypoint: reads a session's segments.jsonl in order,
extracts before/after keyframes for each, calls the local VLM with a bounded
window of recent prior steps as context, and writes steps.jsonl +
interpret_meta.json into the session directory — same shape and
reproducibility-record pattern as Stage 2's pipeline.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..capture.schema import load_manifest
from ..segment.schema import load_segments
from .keyframes import extract_keyframes
from .schema import InterpretRunMeta, Step
from .vlm import InterpretParams, interpret_segment


def interpret_session(session_dir: Path, params: InterpretParams | None = None) -> Path:
    params = params or InterpretParams()
    manifest = load_manifest(str(session_dir / "session.json"))
    segments = load_segments(str(session_dir / "segments.jsonl"))

    video_path = session_dir / manifest.video_path
    keyframes = extract_keyframes(video_path, segments, session_dir / "keyframes")

    steps: list[Step] = []
    for i, segment in enumerate(segments):
        step = interpret_segment(
            segment,
            keyframes[segment.segment_id],
            prior_steps=steps,
            step_id=f"{manifest.session_id}-{i:04d}",
            params=params,
        )
        steps.append(step)

    steps_path = session_dir / "steps.jsonl"
    with open(steps_path, "w", encoding="utf-8") as f:
        f.writelines(step.model_dump_json() + "\n" for step in steps)

    meta = InterpretRunMeta(
        session_id=manifest.session_id,
        step_count=len(steps),
        model_name=params.model_name,
        context_window=params.context_window,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
    )
    (session_dir / "interpret_meta.json").write_text(meta.model_dump_json(indent=2))

    print(f"[prentice] wrote {len(steps)} steps to {steps_path}")
    return steps_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prentice-interpret")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args(argv)
    interpret_session(args.session_dir)


if __name__ == "__main__":
    main()
