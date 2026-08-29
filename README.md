# prentice

Records a user's desktop workflow — screen video + OS-level input events,
explicit start/stop — and converts it into a `SKILL.md` that Claude Code can
use to replicate the task. Personal portfolio project; see
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full five-stage pipeline design,
the reasoning behind it, and known limitations.

## Status

v1, macOS only. Currently implemented: **Stage 1 (Capture)**, **Stage 2
(Segment)**, and **Stage 3 (Interpret)**. Stages 4–5 (Refine, Ground &
output) are not built yet.

## Setup

Requires macOS, [Homebrew](https://brew.sh), and
[`uv`](https://docs.astral.sh/uv/):

```sh
brew install ffmpeg uv
uv sync
```

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
PRENTICE_TEST_VLM=1 uv run pytest tests/test_interpret_vlm.py
```

## Eval

See [`eval/README.md`](eval/README.md) for the reliability eval harness plan
— still no recordings or ground truth yet, so accuracy numbers for either
Stage 2 path aren't reported here. `scripts/tune_segment_threshold.py` is
ready to calibrate the CLIP threshold once ground truth exists.
