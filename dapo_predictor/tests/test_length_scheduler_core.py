from __future__ import annotations

import math

import pytest

from dapo_predictor.length_scheduler.calibration import MonotoneCalibration, fit_pava
from dapo_predictor.length_scheduler.lifecycle import ActivationConfig, PredictorLifecycle
from dapo_predictor.length_scheduler.scheduler import EPWSWaitingPool
from dapo_predictor.length_scheduler.types import WaitingRequest


def test_pava_is_monotone_and_interpolates() -> None:
    calibration = fit_pava([0.0, 1.0, 2.0, 3.0], [10.0, 30.0, 20.0, 50.0])

    assert list(calibration.tokens) == sorted(calibration.tokens)
    assert calibration(-1.0) == calibration.tokens[0]
    assert calibration(4.0) == calibration.tokens[-1]
    assert calibration(2.5) >= calibration(2.0)


def test_pava_aggregates_equal_scores_before_isotonic_fit() -> None:
    calibration = fit_pava([0.0, 0.0, 1.0], [10.0, 30.0, 40.0])

    assert calibration.scores == (0.0, 1.0)
    assert calibration.tokens == (20.0, 40.0)


def test_calibration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        MonotoneCalibration((0.0, math.inf), (1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        MonotoneCalibration((0.0, 1.0), (1.0, 2.0))(math.nan)


def test_lifecycle_requires_epoch_samples_and_ready() -> None:
    lifecycle = PredictorLifecycle(ActivationConfig(min_epoch=10, min_samples=100))

    lifecycle.update(epoch=10, samples=100)
    assert not lifecycle.active
    lifecycle.update(epoch=10, samples=100, ready=True)
    assert lifecycle.active


def test_lifecycle_progress_is_monotone() -> None:
    lifecycle = PredictorLifecycle()
    lifecycle.update(epoch=2, samples=10, ready=True)
    with pytest.raises(ValueError, match="monotone"):
        lifecycle.update(epoch=1, samples=10)


def test_waiting_pool_is_fcfs_before_activation() -> None:
    lifecycle = PredictorLifecycle(ActivationConfig(min_epoch=10, min_samples=100))
    lifecycle.update(epoch=0, samples=0, ready=False)
    pool = EPWSWaitingPool(lifecycle)
    pool.add(
        [
            WaitingRequest("late-long", 1, "p1", 1000.0),
            WaitingRequest("early-short", 0, "p0", 10.0),
        ]
    )

    decision = pool.pop_next()
    assert decision.request_id == "early-short"
    assert decision.reason == "fcfs_fallback"


def test_waiting_pool_is_long_first_after_activation() -> None:
    lifecycle = PredictorLifecycle(ActivationConfig(min_epoch=0, min_samples=0))
    lifecycle.update(epoch=0, samples=0, ready=True)
    pool = EPWSWaitingPool(lifecycle)
    pool.add(
        [
            WaitingRequest("short", 0, "p0", 10.0),
            WaitingRequest("long", 1, "p1", 1000.0),
        ]
    )

    first = pool.pop_next()
    second = pool.pop_next()
    assert [first.request_id, second.request_id] == ["long", "short"]
    assert first.reason == "predicted_long_first"


def test_missing_prediction_fails_closed_to_fcfs() -> None:
    lifecycle = PredictorLifecycle(ActivationConfig(min_epoch=0, min_samples=0))
    lifecycle.update(epoch=0, samples=0, ready=True)
    pool = EPWSWaitingPool(lifecycle)
    pool.add(
        [
            WaitingRequest("early", 0, "p0", None),
            WaitingRequest("late", 1, "p1", 1000.0),
        ]
    )

    decision = pool.pop_next()
    assert decision.request_id == "early"
    assert decision.reason == "fcfs_fallback"


def test_request_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        WaitingRequest("r", 0, "p", math.inf)
    with pytest.raises(ValueError, match="non-negative"):
        WaitingRequest("r", -1, "p", 1.0)
