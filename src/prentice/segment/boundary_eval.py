"""Scoring predicted segment boundaries against hand-labeled ground truth.

Used by scripts/tune_segment_threshold.py to sweep the CLIP similarity
threshold, and intended for reuse by the full pipeline eval harness later
(ARCHITECTURE.md §3) — same "predicted boundaries vs. ground truth, within
a tolerance window" scoring applies to both.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryScore:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def score_boundaries(
    predicted_ms: list[float], ground_truth_ms: list[float], tolerance_ms: float
) -> BoundaryScore:
    """Greedy nearest-available match within ``tolerance_ms``, each ground-truth
    boundary claimable by at most one predicted boundary."""
    matched_gt: set[int] = set()
    true_positives = 0
    for p in predicted_ms:
        candidates = [
            (abs(p - g), i) for i, g in enumerate(ground_truth_ms) if i not in matched_gt and abs(p - g) <= tolerance_ms
        ]
        if candidates:
            _, best_i = min(candidates)
            matched_gt.add(best_i)
            true_positives += 1

    false_positives = len(predicted_ms) - true_positives
    false_negatives = len(ground_truth_ms) - true_positives

    precision = true_positives / (true_positives + false_positives) if predicted_ms else 0.0
    recall = true_positives / (true_positives + false_negatives) if ground_truth_ms else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return BoundaryScore(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
