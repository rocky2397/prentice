"""Stage 4 (Refine) entrypoint: reads a session's steps.jsonl and sends it
to the local LLM in bounded chunks (see refine/llm.py — a single call over a
whole session degrades badly) to merge fragmented duplicates, drop noise,
distinguish fixed vs. variable parameters, and flag low-confidence steps.
Writes refined_steps.jsonl + refine_meta.json — NOT 1:1 with steps.jsonl by
design (see refine/schema.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..capture.schema import load_manifest
from ..interpret.schema import load_steps
from .llm import RefineParams, refine_session_steps
from .schema import RefineRunMeta


def refine_session(session_dir: Path, params: RefineParams | None = None) -> Path:
    params = params or RefineParams()
    manifest = load_manifest(str(session_dir / "session.json"))
    steps = load_steps(str(session_dir / "steps.jsonl"))

    failures: list[dict] = []
    refined = refine_session_steps(steps, params=params, failures=failures)

    # Full model output of chunks that failed to parse, for debugging; the
    # steps themselves only carry a short reason.
    failures_path = session_dir / "refine_failures.jsonl"
    if failures:
        with open(failures_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(failure) + "\n" for failure in failures)
    elif failures_path.exists():
        failures_path.unlink()

    refined_path = session_dir / "refined_steps.jsonl"
    with open(refined_path, "w", encoding="utf-8") as f:
        f.writelines(step.model_dump_json() + "\n" for step in refined)

    meta = RefineRunMeta(
        session_id=manifest.session_id,
        input_step_count=len(steps),
        output_step_count=len(refined),
        model_name=params.model_name,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        repetition_penalty=params.repetition_penalty,
        repetition_context_size=params.repetition_context_size,
        chunk_size=params.chunk_size,
    )
    (session_dir / "refine_meta.json").write_text(meta.model_dump_json(indent=2))

    print(
        f"[prentice] wrote {len(refined)} refined steps (from {len(steps)} raw) to {refined_path}"
    )
    return refined_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prentice-refine")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args(argv)
    refine_session(args.session_dir)


if __name__ == "__main__":
    main()
