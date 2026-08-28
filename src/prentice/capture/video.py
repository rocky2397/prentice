"""Screen recording via ffmpeg's avfoundation input on macOS.

ffmpeg (shelled out to, not a Python binding) rather than ScreenCaptureKit:
keeps the whole capture stack in Python with one well-understood external
dependency, instead of adding a Swift toolchain for Stage 1 alone. Output is
encoded at a fixed frame rate (``-r fps``) specifically so that frame index
can be derived from an event's ``t_ms`` by simple arithmetic in Stage 2,
without needing to store a timestamp per frame.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class AVFoundationDevice:
    index: int
    name: str


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    duration_s: float


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError("ffmpeg not found on PATH. Install it with `brew install ffmpeg`.")
    return path


def _require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise FFmpegNotFoundError("ffprobe not found on PATH. Install it with `brew install ffmpeg`.")
    return path


def probe_video(path: Path) -> VideoProbe:
    """Read a video's actual encoded dimensions/fps/duration via ffprobe.

    Used as the single source of truth for pixel dimensions on both capture
    paths: a just-recorded avfoundation session (rather than trusting a
    logical-points screen size times an assumed scale factor) and an
    imported pre-recorded video (which has no other source for this at all).
    """
    ffprobe = _require_ffprobe()
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num_str, _, den_str = stream["r_frame_rate"].partition("/")
    num, den = float(num_str), float(den_str or 1)
    fps = num / den if den else num
    duration_s = float(data.get("format", {}).get("duration", 0.0))
    return VideoProbe(width=width, height=height, fps=fps, duration_s=duration_s)


def list_avfoundation_video_devices() -> list[AVFoundationDevice]:
    """Parse the device list ffmpeg prints to stderr for `-f avfoundation -list_devices true`."""
    ffmpeg = _require_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        check=False,
    )
    devices: list[AVFoundationDevice] = []
    in_video_section = False
    pattern = re.compile(r"\[(\d+)\]\s+(.*)")
    for line in proc.stderr.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            in_video_section = False
            continue
        if in_video_section:
            match = pattern.search(line)
            if match:
                devices.append(AVFoundationDevice(index=int(match.group(1)), name=match.group(2).strip()))
    return devices


def find_screen_device(devices: list[AVFoundationDevice] | None = None) -> AVFoundationDevice:
    devices = devices if devices is not None else list_avfoundation_video_devices()
    for device in devices:
        if "capture screen" in device.name.lower():
            return device
    raise RuntimeError(
        "No 'Capture screen' device found via avfoundation. "
        f"Devices seen: {[d.name for d in devices]}. "
        "Grant Screen Recording permission to this terminal app in "
        "System Settings > Privacy & Security > Screen Recording, then restart the terminal."
    )


class FFmpegScreenRecorder:
    """Wraps an ffmpeg subprocess recording one screen via avfoundation."""

    def __init__(self, output_path: Path, device_index: int, fps: int = 30):
        self._ffmpeg = _require_ffmpeg()
        self.output_path = output_path
        self.device_index = device_index
        self.fps = fps
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("recorder already started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg,
            "-y",
            "-f", "avfoundation",
            "-framerate", str(self.fps),
            "-capture_cursor", "1",
            "-i", f"{self.device_index}:none",
            "-r", str(self.fps),
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(self.output_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Ask ffmpeg to quit gracefully (so the mp4 container finalizes cleanly)."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(b"q")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
        self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
