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
```

Each session writes to `eval/recordings/<session_id>/`:

- `screen.mp4` — fixed-frame-rate screen recording
- `events.jsonl` — one JSON object per line: mouse clicks/scrolls, key
  presses/releases, window switches — each timestamped as `t_ms` since
  session start, using the same clock anchor as the video's frame clock
- `session.json` — manifest: fps, screen size, OS version, capture device

**Do not record sessions containing secrets** — nothing in this project
redacts sensitive on-screen content.

## Development

```sh
uv sync --extra dev
uv run pytest
```

## Eval

See [`eval/README.md`](eval/README.md) for the reliability eval harness plan
— currently an empty skeleton, populated once Stage 2+ exist to evaluate.
