# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
"""Build the README's input/output GIF from one processed recording.

The recording plays on the left. On the right, each grounded step from
SKILL.md appears when the action it came from starts: grounded step ->
source step -> segment start_ms, which is on the same clock as screen.mp4.

    uv run scripts/build_demo_gif.py <session dir> <out dir> [start s] [task steps]

`task steps` is a JSON list of the step numbers that match a real action of
the task (from prentice-score); the others are drawn grey as "extra". Needs
ffmpeg and macOS system fonts. Writes <out dir>/frames/ as scratch.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

session = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
fps = 10
crop = (280, 180, 1560, 1540)  # x, y, w, h in video pixels: the browser window only
left_w = 580
left_h = round(crop[3] * left_w / crop[2])
right_w = 560
pad = 16
head = 44
W, H = left_w + right_w + 3 * pad, left_h + head + 2 * pad
start_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
hold_s = 3.5

BG = (250, 250, 248)
INK = (30, 30, 34)
MUTED = (110, 110, 118)
ACCENT = (37, 99, 235)
CARD = (255, 255, 255)
LINE = (222, 222, 228)
NEW = (232, 240, 254)
GOOD = (22, 128, 61)

sans = "/System/Library/Fonts/SFNS.ttf"
mono = "/System/Library/Fonts/SFNSMono.ttf"
f_head = ImageFont.truetype(sans, 18)
f_small = ImageFont.truetype(sans, 13)
f_title = ImageFont.truetype(sans, 16)
f_mono = ImageFont.truetype(mono, 13)

seg_start = {}
for line in (session / "segments.jsonl").read_text().splitlines():
    s = json.loads(line)
    seg_start[s["segment_id"]] = s["start_ms"] / 1000
step_seg = {}
for line in (session / "steps.jsonl").read_text().splitlines():
    s = json.loads(line)
    step_seg[s["step_id"]] = s["segment_id"]

steps = []
for line in (session / "grounded_steps.jsonl").read_text().splitlines():
    g = json.loads(line)
    t = min(seg_start[step_seg[sid]] for sid in g["source_step_ids"])
    steps.append((t, g))
real = set(json.loads(sys.argv[4])) if len(sys.argv) > 4 else set()
shown = [(t, g, i in real) for i, (t, g) in enumerate(steps, 1)]


def detail(g: dict) -> str:
    p = {k: v for k, v in (g.get("parameters") or {}).items() if v not in (None, "", [])}
    parts = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in p.items()]
    return f"{g['action_type']}  " + "  ".join(parts)


def card_lines(n: int, g: dict, is_real: bool) -> list[tuple[str, ImageFont.FreeTypeFont, tuple]]:
    intent = g["intent"].rstrip(".")
    if not is_real:
        short = textwrap.shorten(f"{n}. {intent}", 66, placeholder="...")
        return [(short, f_small, MUTED)]
    wrapped = textwrap.wrap(f"{n}. {intent}", 52)
    if len(wrapped) > 2:
        wrapped = [wrapped[0], textwrap.shorten(" ".join(wrapped[1:]), 52, placeholder="...")]
    lines = [(t, f_title, INK) for t in wrapped]
    lines += [(t, f_mono, ACCENT) for t in textwrap.wrap(detail(g), 60)[:1]]
    return lines


def lh(fnt) -> int:
    return {id(f_title): 20, id(f_mono): 17, id(f_small): 17}[id(fnt)]


video_len = json.loads((session / "session.json").read_text())["duration_ms"] / 1000
tmp = out_dir / "frames"
tmp.mkdir(parents=True, exist_ok=True)
for f in tmp.glob("*.png"):
    f.unlink()
x, y, w, h = crop
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        str(start_s),
        "-i",
        str(session / "screen.mp4"),
        "-vf",
        f"fps={fps},crop={w}:{h}:{x}:{y},scale={left_w}:{left_h}:flags=lanczos",
        str(tmp / "v%04d.png"),
    ],
    check=True,
)
vframes = sorted(tmp.glob("v*.png"))
n_hold = int(hold_s * fps)

for i in range(len(vframes) + n_hold):
    t = start_s + min(i, len(vframes) - 1) / fps
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, pad + 6), "Input: you do the task once", font=f_head, fill=INK)
    rx = left_w + 2 * pad
    d.text((rx, pad + 6), "Output: the steps prentice writes to SKILL.md", font=f_head, fill=INK)
    frame = Image.open(vframes[min(i, len(vframes) - 1)])
    img.paste(frame, (pad, pad + head))
    d.rectangle([pad - 1, pad + head - 1, pad + left_w, pad + head + left_h], outline=LINE)

    cy = pad + head
    d.rectangle([rx, cy, rx + right_w - 1, cy + left_h], fill=CARD, outline=LINE)
    cy += 12
    visible = [x for x in shown if x[0] <= t]
    for n, (tt, g, is_real) in enumerate(visible, 1):
        lines = card_lines(n, g, is_real)
        bh = sum(lh(fnt) for _, fnt, _ in lines) + (14 if is_real else 8)
        if cy + bh > pad + head + left_h - 8:
            break
        fresh = t - tt < 1.2 and i < len(vframes)
        d.rectangle(
            [rx + 10, cy, rx + right_w - 11, cy + bh],
            fill=NEW if fresh else CARD,
            outline=ACCENT if fresh else LINE,
        )
        if is_real:
            d.rectangle([rx + 10, cy, rx + 14, cy + bh], fill=GOOD)
        tag, col = ("task step", GOOD) if is_real else ("extra", MUTED)
        tw = d.textlength(tag, font=f_small)
        d.text((rx + right_w - 20 - tw, cy + 5), tag, font=f_small, fill=col)
        ly = cy + (7 if is_real else 4)
        for text, fnt, col in lines:
            d.text((rx + 20, ly), text, font=fnt, fill=col)
            ly += lh(fnt)
        cy += bh + 6
    fy = pad + head + left_h - 52
    d.line([rx + 10, fy - 8, rx + right_w - 11, fy - 8], fill=LINE)
    d.rectangle([rx + 12, fy + 3, rx + 16, fy + 15], fill=GOOD)
    d.text(
        (rx + 24, fy), "Task step: matches an action the task really had", font=f_small, fill=INK
    )
    d.text(
        (rx + 24, fy + 20),
        "Extra: a line prentice wrote that isn't a real step (mostly title changes)",
        font=f_small,
        fill=MUTED,
    )
    if not visible:
        d.text((rx + 20, cy + 4), "Waiting for the first action...", font=f_title, fill=MUTED)
    img.save(tmp / f"c{i:04d}.png")

gif = out_dir / "wikipedia-input-output.gif"
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(tmp / "c%04d.png"),
        "-vf",
        "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        "-loop",
        "0",
        str(gif),
    ],
    check=True,
)
mp4 = out_dir / "wikipedia-input-output.mp4"
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(tmp / "c%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(mp4),
    ],
    check=True,
)
print(gif, gif.stat().st_size // 1024, "KB;", len(vframes) + n_hold, "frames;", W, "x", H)
for tt, g in steps:
    print(f"{tt:6.2f}s  {g['action_type']:10} {g['intent'][:90]}")
