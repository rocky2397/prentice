import json

from prentice.capture.schema import CaptureEventAdapter, MouseClickEvent, WindowSwitchEvent


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
