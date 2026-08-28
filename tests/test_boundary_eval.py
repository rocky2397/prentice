from prentice.segment.boundary_eval import score_boundaries


def test_perfect_match():
    score = score_boundaries([1000.0, 2000.0], [1000.0, 2000.0], tolerance_ms=100)
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_false_positive_and_false_negative():
    # predicted has an extra boundary (FP) and misses one (FN)
    score = score_boundaries([1000.0, 3000.0], [1000.0, 2000.0], tolerance_ms=100)
    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert 0.0 < score.precision < 1.0
    assert 0.0 < score.recall < 1.0


def test_within_tolerance_counts_as_match():
    score = score_boundaries([1050.0], [1000.0], tolerance_ms=100)
    assert score.true_positives == 1
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_outside_tolerance_does_not_match():
    score = score_boundaries([1200.0], [1000.0], tolerance_ms=100)
    assert score.true_positives == 0
    assert score.precision == 0.0
    assert score.recall == 0.0


def test_empty_predicted():
    score = score_boundaries([], [1000.0], tolerance_ms=100)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_empty_ground_truth_and_empty_predicted():
    score = score_boundaries([], [], tolerance_ms=100)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_each_ground_truth_boundary_claimed_at_most_once():
    # two predictions both near the same single ground-truth boundary
    score = score_boundaries([1000.0, 1010.0], [1000.0], tolerance_ms=100)
    assert score.true_positives == 1
    assert score.false_positives == 1
