"""The per-segment progress line of prentice-interpret."""

from prentice.interpret.pipeline import progress_line


def test_first_segment_says_it_includes_loading_the_model():
    assert progress_line(1, 19, [40.0]) == (
        "[prentice] interpreted 1/19 segments (5%), the first one includes loading the model"
    )


def test_estimate_ignores_the_slow_first_segment():
    # 600 s for the first (model load), then 20 s each: 17 left -> 340 s
    assert progress_line(2, 19, [600.0, 20.0]).endswith("about 6 min left")


def test_short_estimates_are_in_seconds():
    assert progress_line(17, 19, [600.0] + [20.0] * 16).endswith("about 40 s left")


def test_last_segment_has_no_estimate():
    assert progress_line(19, 19, [1.0] * 19) == "[prentice] interpreted 19/19 segments (100%)"
