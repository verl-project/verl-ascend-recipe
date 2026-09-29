"""Backend-neutral request and admission types."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WaitingRequest:
    """One rollout request waiting for admission."""

    request_id: str
    arrival_index: int
    prompt_id: str
    predicted_work: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.prompt_id:
            raise ValueError("prompt_id must be non-empty")
        if self.arrival_index < 0:
            raise ValueError("arrival_index must be non-negative")
        if self.predicted_work is not None:
            value = float(self.predicted_work)
            if not math.isfinite(value) or value < 0:
                raise ValueError("predicted_work must be finite and non-negative")


@dataclass(frozen=True)
class DispatchDecision:
    """One deterministic waiting-pool admission decision."""

    request_id: str
    reason: str
    predicted_work: float | None
