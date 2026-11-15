"""Stage 3 (Interpret) entrypoint: reads a session's segments.jsonl in order,
extracts before/after keyframes for each, calls the local VLM with a bounded
window of recent prior steps as context, and writes steps.jsonl +
interpret_meta.json into the session directory — same shape and
reproducibility-record pattern as Stage 2's pipeline.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..capture.schema import load_manifest
from ..segment.schema import load_segments
from .keyframes import extract_keyframes
from .schema import InterpretRunMeta, Step
from .vlm import InterpretParams, fallback_step, interpret_segment, window_switch_step


def progress_line(done: int, total: int, durations: list[float]) -> str:
    """One status line per segment, with a rough time left.

    The first segment also loads the model (and on a first run downloads it),
    so the estimate uses the later segments only.
    """
    line = f"[prentice] interpreted {done}/{total} segments ({done / total:.0%})"
    if done == total:
        return line
    if len(durations) < 2:
        return line + ", the first one includes loading the model"
    per_segment = sum(durations[1:]) / len(durations[1:])
    left = per_segment * (total - done)
    estimate = f"about {left / 60:.0f} min left" if left >= 90 else f"about {left:.0f} s left"
    return f"{line}, {estimate}"


def interpret_session(session_dir: Path, params: InterpretParams | None = None) -> Path:
    params = params or InterpretParams()
    manifest = load_manifest(str(session_dir / "session.json"))
    segments = load_segments(str(session_dir / "segments.jsonl"))

    video_path = session_dir / manifest.video_path
    keyframes = extract_keyframes(
        video_path, segments, session_dir / "keyframes", max_width=params.keyframe_max_width
    )

    steps: list[Step] = []
    durations: list[float] = []
    model_calls = from_event_log = uninterpreted = 0
    for i, segment in enumerate(segments):
        started = time.monotonic()
        step_id = f"{manifest.session_id}-{i:04d}"
        step = window_switch_step(segment, step_id)
        if step is not None:
            from_event_log += 1
        else:
            model_calls += 1
            try:
                step = interpret_segment(
                    segment,
                    keyframes[segment.segment_id],
                    prior_steps=steps,
                    step_id=step_id,
                    params=params,
                )
            except ValueError as exc:
                # One unusable answer must not lose every segment done so far.
                uninterpreted += 1
                reason = str(exc).split(":")[0]
                print(
                    f"[prentice] interpret: segment {segment.segment_id}: {reason}; kept as a fallback step",
                    flush=True,
                )
                step = fallback_step(segment, step_id, reason)
        steps.append(step)
        durations.append(time.monotonic() - started)
        # flush: the line must show up live even when output goes to a pipe or file
        print(progress_line(i + 1, len(segments), durations), flush=True)

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
        keyframe_max_width=params.keyframe_max_width,
        model_calls=model_calls,
        from_event_log=from_event_log,
        uninterpreted=uninterpreted,
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
