"""Checks the macOS TCC grants and binaries Stage 1 capture needs, and
explains how to fix whatever's missing. Run this before `prentice-capture
start` the first time (or after switching terminal apps).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prentice.capture.video import (
    FFmpegNotFoundError,
    list_avfoundation_video_devices,
)

try:
    from ApplicationServices import AXIsProcessTrusted
except ImportError:
    AXIsProcessTrusted = None


def main() -> int:
    ok = True

    if shutil.which("ffmpeg") is None:
        print("[FAIL] ffmpeg not found on PATH. Install with: brew install ffmpeg")
        ok = False
    else:
        print("[ OK ] ffmpeg found")
        try:
            devices = list_avfoundation_video_devices()
            screen_devices = [d for d in devices if "capture screen" in d.name.lower()]
            if screen_devices:
                print(f"[ OK ] screen capture device(s) found: {[d.name for d in screen_devices]}")
            else:
                print(
                    "[FAIL] no 'Capture screen' avfoundation device found — grant Screen "
                    "Recording permission to this terminal app in System Settings > "
                    "Privacy & Security > Screen Recording, then restart the terminal."
                )
                ok = False
        except FFmpegNotFoundError as exc:
            print(f"[FAIL] {exc}")
            ok = False

    if AXIsProcessTrusted is None:
        print(
            "[WARN] pyobjc ApplicationServices not importable — "
            "install project dependencies first (`uv sync`)."
        )
    elif not AXIsProcessTrusted():
        print(
            "[FAIL] Accessibility permission not granted to this terminal app. Grant it in "
            "System Settings > Privacy & Security > Accessibility, then restart the terminal."
        )
        ok = False
    else:
        print("[ OK ] Accessibility permission granted")

    print()
    print(
        "Note: Input Monitoring permission (for keyboard/mouse event logging) can't be "
        "checked programmatically — macOS only prompts for it the first time a listener "
        "actually starts. If capture starts but no key/click events are logged, grant it "
        "in System Settings > Privacy & Security > Input Monitoring and restart the terminal."
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
