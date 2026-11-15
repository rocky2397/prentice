"""Stage 3 speed-ups and crash safety (no model needed)."""

import json
from types import SimpleNamespace

from PIL import Image

from prentice import vlm_model
from prentice.capture.schema import KeyEvent, WindowSwitchEvent
from prentice.interpret import pipeline
from prentice.interpret.keyframes import fit_width
from prentice.interpret.vlm import fallback_step, window_switch_step
from prentice.segment.schema import Segment


def seg(sid, hint, events):
    return Segment(
        segment_id=sid,
        source="event_log",
        action_hint=hint,
        start_ms=0,
        end_ms=1,
        frame_start=0,
        frame_end=1,
        events=events,
    )


SWITCH = WindowSwitchEvent(
    t_ms=0, app_name="Finder", bundle_id="com.apple.finder", window_title="prentice-sandbox"
)


def test_a_window_switch_is_described_from_the_event_log():
    step = window_switch_step(seg("s1", "window_switch", [SWITCH]), "id-1")
    assert step.action_type == "navigate"
    assert step.intent == 'Switched to Finder ("prentice-sandbox").'
    assert step.parameters == {"application": "Finder", "window_title": "prentice-sandbox"}


def test_anything_but_a_pure_window_switch_still_goes_to_the_model():
    key = KeyEvent(t_ms=0, key="a", pressed=True)
    assert window_switch_step(seg("s1", "window_switch", [SWITCH, key]), "id") is None
    assert window_switch_step(seg("s1", "type", [key]), "id") is None
    assert window_switch_step(seg("s1", "window_switch", []), "id") is None


def test_fallback_step_keeps_the_kind_of_action_and_says_why():
    step = fallback_step(seg("s1", "drag", []), "id-1", "model did not return valid JSON")
    assert step.action_type == "drag"
    assert step.intent.startswith("Could not interpret this step (model did not return valid JSON)")


def test_screenshots_are_scaled_down_never_up():
    retina = Image.new("RGB", (3456, 2234))
    assert fit_width(retina, 1280).size == (1280, 827)
    small = Image.new("RGB", (800, 500))
    assert fit_width(small, 1280).size == (800, 500)
    assert fit_width(retina, None).size == (3456, 2234)


def test_a_cached_model_loads_without_asking_the_hub(monkeypatch):
    import huggingface_hub

    calls = []

    def fake(name, local_files_only=False):
        calls.append(local_files_only)
        return "/cache/model"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    monkeypatch.delenv("PRENTICE_UPDATE_MODEL", raising=False)
    assert vlm_model.resolve_model("org/model") == "/cache/model"
    assert calls == [True]


def test_an_uncached_model_or_an_update_request_uses_the_name(monkeypatch):
    import huggingface_hub

    def missing(name, local_files_only=False):
        raise FileNotFoundError

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    assert vlm_model.resolve_model("org/model") == "org/model"
    monkeypatch.setenv("PRENTICE_UPDATE_MODEL", "1")
    assert vlm_model.resolve_model("org/model") == "org/model"


def test_one_bad_model_answer_does_not_stop_the_run(tmp_path, monkeypatch):
    segments = [
        seg("s0", "window_switch", [SWITCH]),
        seg("s1", "type", [KeyEvent(t_ms=0, key="a", pressed=True)]),
        seg("s2", "click", []),
    ]
    monkeypatch.setattr(
        pipeline, "load_manifest", lambda p: SimpleNamespace(session_id="sess", video_path="v.mp4")
    )
    monkeypatch.setattr(pipeline, "load_segments", lambda p: segments)
    monkeypatch.setattr(
        pipeline, "extract_keyframes", lambda *a, **k: {s.segment_id: None for s in segments}
    )
    asked = []

    def model(segment, keyframes, prior_steps, step_id, params):
        asked.append(segment.segment_id)
        if segment.segment_id == "s1":
            raise ValueError("model did not return valid JSON for segment s1: '{...'")
        return fallback_step(segment, step_id, "stand-in for a good answer")

    monkeypatch.setattr(pipeline, "interpret_segment", model)
    pipeline.interpret_session(tmp_path)

    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(steps) == 3
    assert asked == ["s1", "s2"]  # the window switch never reached the model
    assert steps[1]["intent"].startswith(
        "Could not interpret this step (model did not return valid JSON"
    )
    meta = json.loads((tmp_path / "interpret_meta.json").read_text())
    assert (meta["model_calls"], meta["from_event_log"], meta["uninterpreted"]) == (2, 1, 1)
    assert meta["keyframe_max_width"] == 1280
