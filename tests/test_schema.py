import json

from prentice.capture.schema import (
    CaptureEventAdapter,
    ImportedManifest,
    LiveCaptureManifest,
    MouseClickEvent,
    SessionManifestAdapter,
    WindowSwitchEvent,
)


def test_mouse_click_roundtrip():
    event = MouseClickEvent(t_ms=123.4, x=10, y=20, button="Button.left", pressed=True)
    parsed = CaptureEventAdapter.validate_json(event.model_dump_json())
    assert isinstance(parsed, MouseClickEvent)
    assert parsed.x == 10


def test_discriminated_union_parses_by_type():
    line = json.dumps(
        {
            "type": "window_switch",
            "t_ms": 5.0,
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "window_title": "Example",
        }
    )
    parsed = CaptureEventAdapter.validate_json(line)
    assert isinstance(parsed, WindowSwitchEvent)
    assert parsed.app_name == "Safari"


def test_imported_manifest_has_no_events_by_construction():
    manifest = ImportedManifest(
        session_id="abc",
        fps=30.0,
        video_width=1920,
        video_height=1080,
        video_path="screen.mov",
        events_path="events.jsonl",
        has_events=False,
        duration_ms=60_000.0,
        original_video_path="/tmp/source.mov",
        imported_at_utc="2026-01-01T00:00:00+00:00",
    )
    assert manifest.has_events is False
    parsed = SessionManifestAdapter.validate_json(manifest.model_dump_json())
    assert isinstance(parsed, ImportedManifest)


def test_manifest_union_discriminates_by_source():
    live = LiveCaptureManifest(
        session_id="xyz",
        epoch0_utc="2026-01-01T00:00:00+00:00",
        fps=30.0,
        video_width=3456,
        video_height=2234,
        video_path="screen.mp4",
        events_path="events.jsonl",
        has_events=True,
        duration_ms=90_000.0,
        os_version="26.5.1",
        screen_width=1728,
        screen_height=1117,
        backing_scale_factor=2.0,
        avfoundation_device_index=4,
        avfoundation_device_name="Capture screen 0",
    )
    parsed = SessionManifestAdapter.validate_json(live.model_dump_json())
    assert isinstance(parsed, LiveCaptureManifest)
    assert parsed.source == "live_capture"
