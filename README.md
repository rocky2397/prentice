# prentice

Records a user's desktop workflow — screen video + OS-level input events,
explicit start/stop — and converts it into a `SKILL.md` that Claude Code can
use to replicate the task. Personal portfolio project; see
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full five-stage pipeline design,
the reasoning behind it, and known limitations.

## Status

v1, macOS only. All five stages are implemented: **Stage 1 (Capture)**,
**Stage 2 (Segment)**, **Stage 3 (Interpret)**, **Stage 4 (Refine)**, and
**Stage 5 (Ground & output)** — so the pipeline now runs end to end and
produces a `SKILL.md`. What is *not* done is measuring it: there is still no
hand-labeled ground truth and no replay-success harness, so no accuracy
number is reported anywhere (see [Eval](#eval)).

Two gaps worth knowing before reading any results below. Every recording in
this repo is an **imported** video, so the event-log path — Stage 2's
primary, exact, no-model path, and the reason the design logs input events
at all — has only ever run under unit tests, never on a real session. And
because imported sessions carry no event log, no step anywhere has an
accessibility identifier attached.

## Setup

Requires macOS, [Homebrew](https://brew.sh), and
[`uv`](https://docs.astral.sh/uv/):

```sh
brew install ffmpeg uv
uv sync
```

### Model checkpoints

Stage 2's CLIP model and Stage 3's VLM are both downloaded from Hugging Face
on first use. Rather than Hugging Face's own default (the home-directory
cache — easy to forget about, and not necessarily the disk you meant),
`src/prentice/__init__.py` defaults `HF_HOME` to `.model_cache/` inside the
repo, so checkpoints live on the same disk as the project and don't
silently pile up somewhere else. This is only a default: set `HF_HOME`
yourself before running anything if you want a different location.

### Permissions

Capture needs three separate macOS TCC grants, given to whichever terminal
app you run it from (System Settings > Privacy & Security):

- **Screen Recording** — for the video capture itself
- **Accessibility** — for reading the accessibility-tree element under the
  cursor on click, and the focused window title
- **Input Monitoring** — for global mouse/keyboard event logging

Restart the terminal app after granting each. Check your current status
with:

```sh
uv run python scripts/check_permissions.py
```

## Usage

```sh
# list available avfoundation screen-capture devices
uv run prentice-capture devices

# start a session; Ctrl+C to stop (explicit start/stop only, never always-on)
uv run prentice-capture start

# wrap an existing (pre-recorded) video into a session directory instead
uv run prentice-capture import path/to/existing-recording.mov
```

Both commands write a session directory to `eval/recordings/<session_id>/`:

- `screen.<ext>` — the video (fixed-frame-rate `screen.mp4` for a live
  capture; whatever the source file's format is for an import)
- `events.jsonl` — one JSON object per line: mouse clicks/scrolls, key
  presses/releases, window switches — each timestamped as `t_ms` since
  session start, using the same clock anchor as the video's frame clock.
  **Empty for imported sessions** (see below).
- `session.json` — manifest: `source` (`"live_capture"` or `"imported"`),
  `has_events`, fps, video pixel dimensions, and source-specific metadata

### Importing a pre-recorded video

`prentice-capture import` accepts a video from any source (QuickTime, OBS, a
phone, a screen-recording app you already used) and wraps it into a session
directory with the same shape as a live capture — it never touches or moves
the original file, and needs none of the TCC permissions above (it only
shells out to `ffprobe`).

The one thing it *cannot* do is retroactively recover OS-level input events
— those only exist if this tool's listeners were running during the
recording. An imported session's manifest is stamped `has_events: false`
precisely so nothing downstream silently assumes event-quality data exists
for it. Per [`ARCHITECTURE.md`](ARCHITECTURE.md) §2, event logs are what
make segmentation exact; an imported video without one falls back to a
weaker, vision-only signal in Stage 2 (below) — a known, explicitly-flagged
gap, not a hidden one. (Synthesizing an event log for imported video via a
vision model is a separate piece of future work, not part of this import
step.)

**Do not record or import sessions containing secrets** — nothing in this
project redacts sensitive on-screen content.

### Stage 2 — Segment

```sh
uv run prentice-segment eval/recordings/<session_id>
```

Reads a session directory's `session.json` and branches automatically:

- **`has_events: true`** (a live capture) — clusters the logged input events
  into action segments using timestamps and event type: click vs. drag by
  press/release coordinate delta, scroll/type bursts merged within a gap
  threshold, window switches as hard boundaries. No model call — exact.
- **`has_events: false`** (an imported video) — samples frames at a fixed
  rate, embeds them with CLIP (`open_clip`, `ViT-B-32` /
  `laion2b_s34b_b79k`), and flags a boundary wherever cosine similarity
  between consecutive sampled frames drops below a threshold. The starting
  threshold (`0.90`) is a literature-informed guess, **not yet calibrated**
  — `scripts/tune_segment_threshold.py` sweeps thresholds against
  hand-labeled ground truth once that exists in `eval/ground_truth/`.

Both paths write the same output shape to the session directory:

- `segments.jsonl` — ordered segments, each tagged
  `source: "event_log" | "inferred"` and `action_hint` (the concrete action
  type for the event-log path; always `"scene_change"` for the inferred
  path, since vision alone can't say *what* changed, only *that* it did)
- `segment_meta.json` — the parameters that run used, for reproducibility

### Stage 3 — Interpret

```sh
uv run prentice-interpret eval/recordings/<session_id>
```

Reads `segments.jsonl` in order and, for each segment, sends a local VLM
(Qwen3-VL, via [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) — runs
entirely on-device on Apple Silicon, no API key or cost) the segment's
before/after keyframes, the raw logged event(s) if any (empty for the
inferred path — the model is told explicitly that no event log exists rather
than being given misleading empty data), and a bounded window of the last 5
interpreted steps as context. Asks for a structured JSON step back: intent,
action type, target description, parameters.

Per [`ARCHITECTURE.md`](ARCHITECTURE.md) §Stage 3, a separate frontier-model
(e.g. Gemini) benchmark to establish a quality ceiling is intentionally out
of scope here — that needs an external API key and costs money per call, so
it's a distinct, separately-scoped piece of work, not bundled into this one.

Writes `steps.jsonl` (each step still carries the originating segment's
`source: "event_log" | "inferred"`, unmodified, same propagation rule as
Stage 2) and `interpret_meta.json` (model/params used). Also writes
`keyframes/<segment_id>/{before,after}.jpg` into the session directory, kept
(not a temp dir) so what the model actually saw for a given step stays
inspectable.

**First run downloads the model checkpoint** (tens of GB, from Hugging Face
— free, but slow and disk-hungry) — the default,
`mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit`, is a MoE with only ~3B active
params/token despite the "30B" name, chosen to stay fast on unified memory.

### Stage 4 — Refine

```sh
uv run prentice-refine eval/recordings/<session_id>
```

Reads `steps.jsonl` and sends it in **bounded chunks of 20 steps** (unlike
Stage 3, which reasons per-segment) to the same already-cached Qwen3-VL,
called text-only — no images, since this stage reasons over Stage 3's
structured output, not pixels. This satisfies `ARCHITECTURE.md` §Stage 4's
"at least a separate call" without a second model download. Per
`ARCHITECTURE.md`, this pass: merges fragmented/duplicate steps, drops
noise, distinguishes fixed vs. variable parameter values, and flags steps it
can't resolve confidently.

Chunking (rather than one call per session) was a deliberate fix, not the
starting design — real testing found a single long generation over a whole
session reliably degrades into malformed JSON, garbled English, or
repetition loops partway through, and this got *more* likely the longer the
session, not just occasionally. Smaller chunks make each individual
generation short enough that this has much less room to happen. It doesn't
eliminate the problem (see below), but it makes the failure mode local to
one chunk instead of catastrophic to the whole session.

Writes `refined_steps.jsonl` + `refine_meta.json`. **Deliberately not 1:1
with `steps.jsonl`** — merging and noise-dropping both change the count.
Each `RefinedStep` keeps `source_step_ids` (traceability to exactly which
Stage 3 steps it absorbed) and a unioned `source`
(`event_log` | `inferred` | `mixed`).

**Reliability guarantees, enforced in code, not just prompt wording** —
every one of these was found necessary by testing against real sessions,
not assumed in advance:
- A merge group larger than 15 input steps is automatically flagged
  `needs_review`, regardless of what the model claims — real testing found
  the model sometimes collapses an entire multi-action stretch into one
  step (up to 123 of 126 steps in one group).
- An input step already claimed by an earlier group can never be claimed
  again by a later one — real testing found the model sometimes appends a
  redundant "catch-all" group re-listing steps already correctly covered
  elsewhere.
- **No step can be silently dropped**, at either the individual-step or
  whole-chunk level. The model can decline to mention a step (intended for
  genuine noise, like an accidental click), but real testing found it doing
  this to genuine distinct actions too (a Save click, a Run click,
  vanishing entirely). Any step the model doesn't account for — including
  every step in a chunk whose response fails to parse outright — is passed
  through unrefined and flagged for review. Never silently lost.

**Known limitation, validated against all 5 real imported sessions**: these
guarantees mean Stage 4 never corrupts or loses data — every session
produces complete output with 100% of raw steps traceable exactly once —
but merge *quality* still varies a lot, and per-chunk parse failures remain
common even at 20-step granularity:

| Session | Raw steps | Refined | Chunks that failed to parse |
|---|---|---|---|
| Chrome (322b4ab2) | 79 | 79 | all 4 — output is 100% passthrough |
| Chrome (88b72c3b) | 126 | 124 | 5 of 7 |
| VS Code (99f1919f) | 63 | 62 | 3 of 4 |
| debugger (eca285c3) | 30 | 30 | both — 100% passthrough |
| Finder (c9bc0323) | 93 | 67 | 0 of 5 — real merging happened |

So chunking fixed the *severity* (no more total session loss, which
happened on 3 of 5 sessions before this fix) but not the *frequency* of
individual chunks producing bad JSON — that's still common. The safety net
means a bad chunk degrades gracefully to "flagged, unrefined" rather than
"lost," which is the property that actually matters, but the refining
value-add is inconsistent session to session. Next levers, not yet
attempted: a smaller chunk size, a stricter/more constrained output schema,
or a different (non-VLM) text-reasoning model for this stage specifically.

### Stage 5 — Ground & output

```sh
uv run prentice-ground eval/recordings/<session_id>
```

Reads `refined_steps.jsonl` and decides, per step, how it should actually be
replayed — then emits the pipeline's real deliverable to
`<session_dir>/skill/`: a `SKILL.md` with YAML frontmatter (name,
description, required inputs), the ordered step list, and a verification
section, plus a `scripts/` subfolder for any generated scripts. Also writes
`grounded_steps.jsonl` + `ground_meta.json`.

Unlike Stages 3 and 4, this stage makes **no model call at all**. Two
reasons: the grounding decision is a rule, not a judgment, and Stage 4's
measured failure rate (5 of 7 chunks on one session) is a poor argument for
putting this model in the path that produces the one artifact meant to be
trusted. A run is therefore reproducible from its inputs alone, which is
also why `ground_meta.json` records no sampling parameters.

**A step becomes a scripted action only when a concrete executable value
appears in its `parameters`** — a `command`, a URL with an explicit scheme
(under any key) or a well-formed hostname under a URL-named key, or an
application name. The prose `target_description` is *never* treated as
evidence. Turning `"the YouTube homepage"` into `open https://youtube.com`
would produce a command that looks exactly as confident as a real one and
runs without ever showing a human what it guessed. A wrong scripted action
is strictly worse than an honest UI-replay step, so the rule refuses to
guess. Everything else becomes a UI-replay action carrying the semantic
target description as the primary handle, the accessibility identifier when
one was captured, and raw coordinates only as an explicitly-labelled
last-resort hint — never as the primary target, per
[`ARCHITECTURE.md`](ARCHITECTURE.md) §Stage 5.

The accessibility identifier and coordinates don't exist on `Step`, so they
can't be read straight out of `refined_steps.jsonl`. They're recovered by
walking the traceability chain every earlier stage maintained:
`RefinedStep.source_step_ids` → `Step.segment_id` → `Segment.events` →
`MouseClickEvent.ax_element`.

Stage 4's single `needs_review` bool is also split into a finer signal,
because on real sessions it is dominated by one cause that isn't a quality
judgment at all: `unrefined_passthrough` (Stage 4's chunk failed to parse,
or the model omitted the step) versus `model_flagged` (the step *was*
refined and something judged it low-confidence anyway). Collapsing the two
would make the flag useless exactly where it matters.

**Results on the 5 real sessions** — this table mostly measures the quality
of what Stages 3 and 4 hand over, not Stage 5's decision logic:

| Session | Steps | Scripted | UI-replay | With AX id | Flagged |
|---|---|---|---|---|---|
| Chrome (322b4ab2) | 79 | 0 | 79 | 0 | 79 |
| Chrome (88b72c3b) | 124 | 0 | 124 | 0 | 119 |
| VS Code (99f1919f) | 62 | 0 | 62 | 0 | 60 |
| debugger (eca285c3) | 30 | 0 | 30 | 0 | 30 |
| Finder (c9bc0323) | 67 | 1 | 66 | 0 | 45 |
| **Total** | **362** | **1** | **361** | **0** | **333** |

One step out of 362 was scriptable: a `{"application": "Terminal.app"}`
parameter becoming `open -a Terminal.app`. Every `navigate` step that
plainly *was* a navigation — to YouTube, to Gmail, to a Reddit link — came
out of Stage 3 with empty `parameters`, so no URL exists to script, and the
rule above correctly declines to invent one. Zero steps carry an
accessibility identifier because no imported session has an event log. The
"prefer scripts" path that `ARCHITECTURE.md` calls the biggest reliability
lever is therefore built and tested but essentially unexercised by this
data — the lever it depends on is upstream parameter extraction, which is
where the effort belongs next, not in loosening this rule.

## Development

```sh
uv sync --extra dev
uv run pytest
```

The CLIP-path and VLM-path integration tests each download a real checkpoint
(~600MB and tens of GB respectively) and run real inference, so both are
opt-in rather than part of the default run:

```sh
PRENTICE_TEST_CLIP=1 uv run pytest tests/test_clip_boundary_detection.py
PRENTICE_TEST_VLM=1 uv run pytest tests/test_interpret_vlm.py tests/test_refine_llm.py
```

Stage 5's tests need neither a checkpoint nor ffmpeg — grounding reads only
the earlier stages' JSON output, so a session directory can be faked from
files alone — and run as part of the default suite.

## Eval

See [`eval/README.md`](eval/README.md) for the reliability eval harness
plan. There is still no hand-labeled ground truth, so no precision/recall or
replay-success number is reported anywhere in this README — every figure
above is a count of what the pipeline produced, never a measure of whether
it was right. `scripts/tune_segment_threshold.py` is ready to calibrate the
CLIP threshold once ground truth exists.

The two pieces of measurement machinery that `ARCHITECTURE.md` §3 calls for
are also still missing, not just the labels: `segment/boundary_eval.py`
scores segment boundaries only, and there is no step-extraction
precision/recall scorer and no end-to-end replay harness that executes a
generated `SKILL.md` and reports whether it worked.
