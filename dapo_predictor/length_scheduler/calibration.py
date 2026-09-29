"""Monotone score-to-token calibration for Linear/ListMLE."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MonotoneCalibration:
    """Piecewise-linear interpolation over monotone calibration knots."""

    scores: tuple[float, ...]
    tokens: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.tokens) or len(self.scores) < 2:
            raise ValueError("calibration requires at least two aligned knots")
        if not all(math.isfinite(value) for value in (*self.scores, *self.tokens)):
            raise ValueError("calibration knots must be finite")
        if any(right <= left for left, right in zip(self.scores, self.scores[1:], strict=False)):
            raise ValueError("calibration scores must be strictly increasing")
        if any(right < left for left, right in zip(self.tokens, self.tokens[1:], strict=False)):
            raise ValueError("calibration tokens must be non-decreasing")
        if any(value <= 0 for value in self.tokens):
            raise ValueError("calibration tokens must be positive")

    def __call__(self, score: float) -> float:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        if value <= self.scores[0]:
            return self.tokens[0]
        if value >= self.scores[-1]:
            return self.tokens[-1]
        upper = bisect_right(self.scores, value)
        lower = upper - 1
        span = self.scores[upper] - self.scores[lower]
        weight = (value - self.scores[lower]) / span
        return self.tokens[lower] + weight * (self.tokens[upper] - self.tokens[lower])


def fit_pava(scores: Iterable[float], token_targets: Iterable[float]) -> MonotoneCalibration:
    """Fit a non-decreasing score-to-token map with PAVA.

    Calibration data must come from training/validation history, never from the
    rollout batch currently being scheduled.
    """

    pairs = sorted((float(score), float(target)) for score, target in zip(scores, token_targets, strict=True))
    if len(pairs) < 2:
        raise ValueError("PAVA requires at least two observations")
    if not all(math.isfinite(score) and math.isfinite(target) and target > 0 for score, target in pairs):
        raise ValueError("scores must be finite and token targets must be finite and positive")

    # Equal scores describe one x coordinate and must receive one fitted value.
    # Aggregate them before weighted PAVA instead of fitting separate values and
    # arbitrarily retaining one during knot de-duplication.
    unique_scores: list[float] = []
    target_sums: list[float] = []
    target_counts: list[float] = []
    for score, target in pairs:
        if unique_scores and score == unique_scores[-1]:
            target_sums[-1] += target
            target_counts[-1] += 1.0
        else:
            unique_scores.append(score)
            target_sums.append(target)
            target_counts.append(1.0)

    blocks: list[list[float]] = []  # [start, end, weight, mean]
    for index, (target_sum, weight) in enumerate(zip(target_sums, target_counts, strict=True)):
        blocks.append([float(index), float(index), weight, target_sum / weight])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[2] + right[2]
            mean = (left[2] * left[3] + right[2] * right[3]) / weight
            blocks.append([left[0], right[1], weight, mean])

    fitted = [0.0] * len(unique_scores)
    for start, end, _, mean in blocks:
        for index in range(int(start), int(end) + 1):
            fitted[index] = max(float(mean), 1.0)

    knot_scores = unique_scores
    knot_tokens = fitted
    if len(knot_scores) < 2:
        center = knot_scores[0]
        token = knot_tokens[0]
        knot_scores = [center - 1.0, center + 1.0]
        knot_tokens = [token, token]
    return MonotoneCalibration(tuple(knot_scores), tuple(knot_tokens))
