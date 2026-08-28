# prentice

Records a user's desktop workflow — screen video + OS-level input events,
explicit start/stop — and converts it into a `SKILL.md` that Claude Code can
use to replicate the task. Personal portfolio project; see
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full five-stage pipeline design,
the reasoning behind it, and known limitations.

## Status

v1, macOS only. Currently implemented: **Stage 1 (Capture)** only. Stages
2–5 (Segment, Interpret, Refine, Ground & output) are not built yet.

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
make Stage 2+ reliable; an imported video without one will have to fall
back to a weaker, vision-only signal once that's built — a known,
explicitly-flagged gap, not a hidden one. (Synthesizing an event log for
imported video via a vision model is a separate piece of future work, not
part of this import step.)

**Do not record or import sessions containing secrets** — nothing in this
project redacts sensitive on-screen content.

## Development

```sh
uv sync --extra dev
uv run pytest
```

## Eval

See [`eval/README.md`](eval/README.md) for the reliability eval harness plan
— currently an empty skeleton, populated once Stage 2+ exist to evaluate.
