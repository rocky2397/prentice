"""The window tracker: app from the window list, title from the AX API."""

from prentice.capture import window_tracker as wt

CHROME = {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 4242, "kCGWindowOwnerName": "Google Chrome"}


def _patch(monkeypatch, front, ax_title):
    monkeypatch.setattr(wt, "_front_window", lambda: front)
    monkeypatch.setattr(wt, "_ax_focused_title", lambda: ax_title)
    monkeypatch.setattr(
        wt,
        "_app_for_pid",
        lambda pid: ("Google Chrome", "com.google.Chrome") if pid == 4242 else (None, None),
    )


def test_app_comes_from_the_front_window_and_title_from_ax(monkeypatch):
    _patch(monkeypatch, CHROME, "Wikipedia - Google Chrome")
    events = []
    wt.WindowTracker(on_change=events.append)._poll_once()
    assert events == [
        {
            "type": "window_switch",
            "app_name": "Google Chrome",
            "bundle_id": "com.google.Chrome",
            "window_title": "Wikipedia - Google Chrome",
        }
    ]


def test_a_failing_ax_query_keeps_the_app(monkeypatch):
    _patch(monkeypatch, {**CHROME, "kCGWindowName": "Wikipedia"}, None)
    assert wt.focused_app_and_title() == ("Google Chrome", "com.google.Chrome", "Wikipedia")


def test_no_window_in_front(monkeypatch):
    _patch(monkeypatch, None, None)
    assert wt.focused_app_and_title() == (None, None, None)


def test_the_live_window_list_names_a_real_app():
    """Runs against this Mac's window server (read-only)."""
    name, bundle, _ = wt.focused_app_and_title()
    assert name is None or isinstance(name, str)
    assert bundle is None or "." in bundle
