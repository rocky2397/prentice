from prentice.capture.schema import KeyEvent, MouseClickEvent, MouseScrollEvent, WindowSwitchEvent
from prentice.segment.event_clustering import ClusteringParams, cluster_events

FPS = 30.0
DURATION_MS = 20_000.0


def test_click_below_drag_threshold():
    events = [
        MouseClickEvent(t_ms=1000.0, x=100.0, y=100.0, button="Button.left", pressed=True),
        MouseClickEvent(t_ms=1050.0, x=102.0, y=99.0, button="Button.left", pressed=False),
    ]
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS)
    assert len(segments) == 1
    assert segments[0].action_hint == "click"
    assert segments[0].source == "event_log"
    assert len(segments[0].events) == 2


def test_click_above_drag_threshold_is_a_drag():
    events = [
        MouseClickEvent(t_ms=1000.0, x=100.0, y=100.0, button="Button.left", pressed=True),
        MouseClickEvent(t_ms=1200.0, x=300.0, y=150.0, button="Button.left", pressed=False),
    ]
    params = ClusteringParams(drag_pixel_threshold=8.0)
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS, params=params)
    assert len(segments) == 1
    assert segments[0].action_hint == "drag"


def test_scroll_burst_merges_within_gap():
    events = [
        MouseScrollEvent(t_ms=t, x=10.0, y=10.0, dx=0.0, dy=1.0)
        for t in (1000.0, 1100.0, 1200.0, 1300.0)
    ]
    params = ClusteringParams(scroll_gap_ms=400.0)
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS, params=params)
    assert len(segments) == 1
    assert segments[0].action_hint == "scroll"
    assert segments[0].start_ms == 1000.0
    assert segments[0].end_ms == 1300.0
    assert len(segments[0].events) == 4


def test_scroll_burst_splits_on_large_gap():
    events = [
        MouseScrollEvent(t_ms=1000.0, x=10.0, y=10.0, dx=0.0, dy=1.0),
        MouseScrollEvent(t_ms=1100.0, x=10.0, y=10.0, dx=0.0, dy=1.0),
        MouseScrollEvent(t_ms=5000.0, x=10.0, y=10.0, dx=0.0, dy=1.0),
    ]
    params = ClusteringParams(scroll_gap_ms=400.0)
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS, params=params)
    assert len(segments) == 2
    assert [s.action_hint for s in segments] == ["scroll", "scroll"]


def test_type_burst_merges_keystrokes():
    events = []
    for i, ch in enumerate("hello"):
        base = 1000.0 + i * 100
        events.append(KeyEvent(t_ms=base, key=ch, pressed=True))
        events.append(KeyEvent(t_ms=base + 20, key=ch, pressed=False))
    params = ClusteringParams(type_gap_ms=750.0)
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS, params=params)
    assert len(segments) == 1
    assert segments[0].action_hint == "type"
    assert len(segments[0].events) == 10


def test_window_switch_hard_breaks_type_burst_even_within_gap():
    events = [
        KeyEvent(t_ms=1000.0, key="a", pressed=True),
        KeyEvent(t_ms=1020.0, key="a", pressed=False),
        WindowSwitchEvent(t_ms=1100.0, app_name="Safari", bundle_id="com.apple.Safari", window_title=None),
        KeyEvent(t_ms=1150.0, key="b", pressed=True),
        KeyEvent(t_ms=1170.0, key="b", pressed=False),
    ]
    # gap between the two keystrokes is well under the default 750ms type_gap_ms,
    # but the window switch between them must still force a split
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS)
    type_segments = [s for s in segments if s.action_hint == "type"]
    window_segments = [s for s in segments if s.action_hint == "window_switch"]
    assert len(type_segments) == 2
    assert len(window_segments) == 1


def test_window_switch_between_press_and_release_splits_the_click():
    events = [
        MouseClickEvent(t_ms=1000.0, x=10.0, y=10.0, button="Button.left", pressed=True),
        WindowSwitchEvent(t_ms=1050.0, app_name="Safari", bundle_id="com.apple.Safari", window_title=None),
        MouseClickEvent(t_ms=1100.0, x=10.0, y=10.0, button="Button.left", pressed=False),
    ]
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS)
    click_segments = [s for s in segments if s.action_hint == "click"]
    # press and release must not be paired into one segment spanning the window switch
    assert len(click_segments) == 2
    assert all(s.start_ms == s.end_ms for s in click_segments)


def test_unmatched_press_and_release_become_minimal_segments():
    events = [
        MouseClickEvent(t_ms=500.0, x=1.0, y=1.0, button="Button.left", pressed=False),  # release, no press
        MouseClickEvent(t_ms=2000.0, x=1.0, y=1.0, button="Button.right", pressed=True),  # press, no release
    ]
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS)
    assert len(segments) == 2
    assert all(s.action_hint == "click" for s in segments)
    assert all(s.start_ms == s.end_ms for s in segments)


def test_frame_range_padded_and_clamped_to_video_bounds():
    events = [
        MouseClickEvent(t_ms=50.0, x=1.0, y=1.0, button="Button.left", pressed=True),
        MouseClickEvent(t_ms=60.0, x=1.0, y=1.0, button="Button.left", pressed=False),
    ]
    params = ClusteringParams(frame_pad_ms=150.0)
    segments = cluster_events(events, session_id="s", fps=FPS, duration_ms=DURATION_MS, params=params)
    assert len(segments) == 1
    # padding would push start below 0ms — must clamp, not go negative
    assert segments[0].frame_start == 0


def test_segment_ids_are_session_prefixed_and_chronological():
    events = [
        MouseClickEvent(t_ms=2000.0, x=1.0, y=1.0, button="Button.left", pressed=True),
        MouseClickEvent(t_ms=2010.0, x=1.0, y=1.0, button="Button.left", pressed=False),
        MouseClickEvent(t_ms=1000.0, x=1.0, y=1.0, button="Button.left", pressed=True),
        MouseClickEvent(t_ms=1010.0, x=1.0, y=1.0, button="Button.left", pressed=False),
    ]
    segments = cluster_events(events, session_id="mysession", fps=FPS, duration_ms=DURATION_MS)
    assert [s.segment_id for s in segments] == ["mysession-0000", "mysession-0001"]
    assert segments[0].start_ms < segments[1].start_ms
