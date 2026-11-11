"""Predictor activation state with deterministic FCFS fallback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationConfig:
    """Activate the preselected backend only after both gates pass."""

    min_epoch: int = 10
    min_samples: int = 2560
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.min_epoch < 0 or self.min_samples < 0:
            raise ValueError("activation thresholds must be non-negative")


class PredictorLifecycle:
    """Track monotone training progress for one preselected backend."""

    def __init__(self, config: ActivationConfig | None = None):
        self.config = config or ActivationConfig()
        self.epoch = 0
        self.samples = 0
        self.ready = False

    @property
    def active(self) -> bool:
        return (
            self.config.enabled
            and self.ready
            and self.epoch >= self.config.min_epoch
            and self.samples >= self.config.min_samples
        )

    def update(self, *, epoch: int, samples: int, ready: bool | None = None) -> None:
        if epoch < self.epoch or samples < self.samples:
            raise ValueError("training progress must be monotone")
        self.epoch = int(epoch)
        self.samples = int(samples)
        if ready is not None:
            self.ready = bool(ready)
