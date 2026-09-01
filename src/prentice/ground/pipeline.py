"""Stage 5 (Ground & output) entrypoint: reads a session's
``refined_steps.jsonl``, decides per step whether it replays as a scripted
action or a UI-replay action, and emits the pipeline's actual deliverable —
a ``SKILL.md`` plus any generated scripts — into ``<session>/skill/``.

Same shape and reproducibility-record pattern as Stages 2-4, except that
this stage makes no model call at all, so ``ground_meta.json`` records only
what was produced, not how a model was sampled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..capture.schema import load_manifest
from ..refine.schema import load_refined_steps
from .grounding import ground_step
from .provenance import evidence_for, load_segments_by_step_id
from .schema import GroundedStep, GroundRunMeta
from .skill_md import (
    SCRIPTS_DIR_NAME,
    SKILL_DIR_NAME,
    derive_skill_name,
    render_script,
    render_skill_md,
    script_filename,
)


def _default_description(steps: list[GroundedStep], skill_name: str) -> str:
    scripted = sum(1 for s in steps if s.grounding == "script")
    flagged = sum(1 for s in steps if s.needs_review)
    return (
        f"Replays the recorded {skill_name.replace('-', ' ')}: {len(steps)} steps "
        f"({scripted} scripted, {len(steps) - scripted} UI-replay, {flagged} needing review). "
        "Auto-generated from a screen recording and not yet verified by a successful replay."
    )


def ground_session(
    session_dir: Path, *, skill_name: str | None = None, description: str | None = None
) -> Path:
    manifest = load_manifest(str(session_dir / "session.json"))
    refined_steps = load_refined_steps(str(session_dir / "refined_steps.jsonl"))
    segments_by_step_id = load_segments_by_step_id(session_dir)

    grounded: list[GroundedStep] = [
        ground_step(
            step,
            evidence_for(step, segments_by_step_id),
            step_id=f"grounded-{i:04d}",
        )
        for i, step in enumerate(refined_steps)
    ]

    # Script paths are assigned here rather than in grounding.py so the
    # decision logic stays a pure function of a step, with no filesystem
    # knowledge — the numbering depends on position in the whole session.
    grounded = [
        step.model_copy(update={"script_path": script_filename(step, i + 1)})
        if step.grounding == "script"
        else step
        for i, step in enumerate(grounded)
    ]

    skill_dir = session_dir / SKILL_DIR_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)

    resolved_name = skill_name or derive_skill_name(grounded, manifest.session_id)
    resolved_description = description or _default_description(grounded, resolved_name)

    scripted = [s for s in grounded if s.grounding == "script"]
    if scripted:
        scripts_dir = skill_dir / SCRIPTS_DIR_NAME
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for step in scripted:
            assert step.script_path is not None  # set immediately above
            script_file = scripts_dir / step.script_path
            script_file.write_text(render_script(step))
            script_file.chmod(0o755)

    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        render_skill_md(
            grounded,
            skill_name=resolved_name,
            description=resolved_description,
            session_id=manifest.session_id,
        )
    )

    grounded_path = session_dir / "grounded_steps.jsonl"
    with open(grounded_path, "w", encoding="utf-8") as f:
        f.writelines(step.model_dump_json() + "\n" for step in grounded)

    meta = GroundRunMeta(
        session_id=manifest.session_id,
        skill_name=resolved_name,
        input_step_count=len(refined_steps),
        output_step_count=len(grounded),
        scripted_step_count=len(scripted),
        ui_replay_step_count=len(grounded) - len(scripted),
        ax_grounded_step_count=sum(
            1 for s in grounded if s.ui_target and s.ui_target.has_ax_identifier
        ),
        needs_review_count=sum(1 for s in grounded if s.needs_review),
        unrefined_passthrough_count=sum(
            1 for s in grounded if s.confidence == "unrefined_passthrough"
        ),
    )
    (session_dir / "ground_meta.json").write_text(meta.model_dump_json(indent=2))

    print(
        f"[prentice] wrote {len(grounded)} grounded steps "
        f"({len(scripted)} scripted, {len(grounded) - len(scripted)} UI-replay, "
        f"{meta.needs_review_count} needing review) to {skill_path}"
    )
    return skill_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prentice-ground")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--name", default=None, help="skill name (default: derived from the steps)")
    parser.add_argument(
        "--description", default=None, help="skill description (default: derived from the steps)"
    )
    args = parser.parse_args(argv)
    ground_session(args.session_dir, skill_name=args.name, description=args.description)


if __name__ == "__main__":
    main()
