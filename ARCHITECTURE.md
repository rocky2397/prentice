# Workflow Capture Framework — Architecture & Build Instructions

## 1. Goal

A framework that records a user's on-screen desktop work during an explicit
record/stop session and automatically — with **zero manual annotation** —
converts the recording into a workflow specification that Claude Code can
use to replicate the task: a **SKILL.md** plus supporting scripts.

This is a personal/portfolio project, not a commercial product. Priorities,
in order: **reliability** (it should genuinely work on the workflows it
claims to support) > breadth of app support > polish.

## 2. Pipeline overview

Five stages, each with a single responsibility. Downstream stages should be
swappable (different VLM, different refiner) without touching upstream
stages.

```
Capture  →  Segment  →  Interpret  →  Refine  →  Ground & output
(video +    (split by    (VLM: what     (LLM:      (SKILL.md +
 events)     events)      happened)      verify)    scripts)
```

### Stage 1 — Capture

Record **screen video** (30 fps is a reasonable default) **and OS-level
input events** in parallel, both timestamped against the same clock:

- Mouse: click coordinates, button, drag start/end, scroll
- Keyboard: keystrokes (batch into typed strings where possible)
- Active window / application title on every switch
- Where available: the accessibility-tree element under the cursor at the
  time of each click (macOS Accessibility API / Windows UI Automation)

**Why events, not just video:** relying on a VLM to infer *that* and *when*
a click happened from pixels alone is the single biggest source of error in
every 2024–2025 system that tried it. Logging the event turns that into a
solved problem and lets the VLM focus on the part it's actually good at:
explaining *why* the action happened and *what it means*.

Capture is explicit start/stop, not always-on. Warn users in the README not
to record sessions containing secrets — no system here redacts sensitive
on-screen content.

### Stage 2 — Segment

Use the logged input events as the primary action-boundary signal (free,
exact — no model call needed). Optionally add a cheap secondary signal
(CLIP-embedding or pixel-diff based scene-change detection) to catch UI
state changes that happen without direct input (dialogs opening, pages
loading, async content appearing).

Output: an ordered list of segments, each with a start/end timestamp, the
triggering event(s), and the associated video frames.

### Stage 3 — Interpret

For each segment, send a strong multimodal model:

- Before/after keyframes for the segment
- The raw logged event (coordinates, keystrokes, window title)
- A short window of recent prior steps as context (running memory, not the
  full history — keep the prompt bounded)

Ask it to produce a structured step: **intent** (why), **action type**
(click / type / scroll / drag / navigate / run-command), **target
description** (in words — "the Save button in the toolbar"), and
**parameters** (typed text, scroll amount, drag destination, etc).

Model choice: prototype with a self-hosted open-weight VLM (e.g.
Qwen3-VL) to iterate cheaply, and separately benchmark against a frontier
model (e.g. Gemini 3 Pro) to establish a quality ceiling. Report both
numbers in the README rather than picking one blind.

### Stage 4 — Refine

A second pass, ideally a different (reasoning-oriented) model or at least a
separate call, that:

- Merges redundant/fragmented steps (e.g. multiple keystrokes → one "type"
  action)
- Drops noise (mouse jitter, accidental clicks with no effect)
- Infers which values are **fixed** vs. **variable inputs** the automation
  should expose as parameters
- Flags steps it cannot resolve confidently rather than guessing

This stage is not optional — every system that ablated an equivalent step
saw a significant accuracy drop. Budget for it.

### Stage 5 — Ground & output

Decide, per step, whether it should become:

1. **A scripted action** (preferred whenever possible) — a shell command,
   API call, or CLI invocation. Claude Code can execute these directly and
   they don't suffer from UI-grounding failures at all. This is the
   biggest reliability lever available and is worth real engineering
   effort in Stage 4/5 to detect when a step qualifies.
2. **A UI-replay action** — only when no scripted equivalent exists. Store
   the **semantic target description** and, if captured, the
   **accessibility-tree identifier**. Do *not* store raw pixel coordinates
   as the primary target — layouts shift, windows resize, resolutions
   differ. Coordinates may be kept as a last-resort fallback hint only.

Emit a `SKILL.md` with YAML frontmatter (name, description / when to use,
required inputs) followed by the ordered step list and a verification
section, plus any generated scripts in a `scripts/` subfolder alongside it.

Any step the Refine stage flagged as low-confidence should be marked
`needs-review` in the output rather than silently included — a framework
that knows what it doesn't know is more trustworthy than one that's
silently wrong sometimes.

## 3. Reliability: build an eval harness, don't just assert it

Given reliability is the stated priority, this project should include a
`eval/` folder from early on:

- Record a small fixed set of sample workflows yourself (2–3 target apps
  for v1 — pick a browser, a file manager, and one dev tool, for example).
- Hand-verify the ground-truth step sequence for each **once**, as a test
  oracle (this is separate from the automated-annotation pipeline itself —
  it's how you measure the pipeline, not how the pipeline runs).
- Report, per pipeline version: step-extraction precision/recall against
  ground truth, and end-to-end replay success rate when Claude Code
  actually executes the generated SKILL.md.
- Track these numbers over time in the README. This is what turns "I built
  a thing" into a portfolio piece that demonstrates ML engineering rigor.

## 4. V1 scope

- 2–3 target applications, chosen for being either scriptable (so you can
  prove out the "prefer scripts" path) or having good accessibility-tree
  support (so you can prove out the grounding path). Avoid apps that need
  both raw pixel inference and no accessibility support for v1.
- Single OS to start (pick whichever you develop on).
- No attempt at control-flow inference (loops, conditionals) in v1 — call
  this out explicitly as a known limitation rather than attempting a weak
  version of it.

## 5. Known limitations (state these explicitly in the README)

As of this project's design, none of the following are solved anywhere in
the published literature — don't over-promise on them:

- Inferring control flow (loops, conditionals, branching) from a single
  demonstration
- Disambiguating near-identical UI elements (e.g. two similar icons)
- Precise drag and text-selection boundaries
- Generalizing one recorded demonstration to a *family* of similar tasks
  without over- or under-fitting to the specific recording
- Non-deterministic or dynamically-loaded UI content at replay time

## 6. Suggested stack

- Orchestration: LangGraph — model the five stages as an explicit graph
  with retry/branch logic per stage.
- Capture: platform-native APIs (macOS Accessibility API / Windows UI
  Automation + a standard screen-recording library) rather than
  reverse-engineering input logging from video.
- VLM: Qwen3-VL (self-hosted, cheap iteration) as the working default;
  Gemini 3 Pro or GPT-5 as the benchmarked upper bound.

## 7. Background / prior art this design is based on

- Sharingan (2024) — established that direct-frame VLM interpretation
  outperforms explicit frame-differencing, and that a verification/
  correction pass materially improves accuracy.
- ShowUI-Aloha, OpenAI Codex Record & Replay, Microsoft Skill Recorder
  (2026) — established the event-log-augmented capture pattern and
  runtime re-grounding over coordinate playback.
- GUI-360° (2025) — quantified the accuracy gain from accessibility-tree
  metadata over vision-only grounding (~3% → ~37% in their benchmark).
- Video2GUI / WildGUI (2026) — established the separate spatial-grounding
  stage as distinct from trajectory/step extraction.
