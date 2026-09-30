"""No-anchor Event-driven Pending/Waiting-pool Scheduler (EPWS)."""

from __future__ import annotations

from collections.abc import Iterable

from .lifecycle import PredictorLifecycle
from .types import DispatchDecision, WaitingRequest


class EPWSWaitingPool:
    """Choose the next request whenever the runtime exposes a free slot.

    Before predictor activation this is stable FCFS. After activation it is
    longest-predicted-work-first. Engine placement remains owned by verl's
    existing global request load balancer.
    """

    def __init__(self, lifecycle: PredictorLifecycle):
        self.lifecycle = lifecycle
        self._waiting: dict[str, WaitingRequest] = {}

    def __len__(self) -> int:
        return len(self._waiting)

    def add(self, requests: Iterable[WaitingRequest]) -> None:
        for request in requests:
            if request.request_id in self._waiting:
                raise ValueError(f"duplicate request_id={request.request_id}")
            self._waiting[request.request_id] = request

    def pop_next(self) -> DispatchDecision:
        if not self._waiting:
            raise IndexError("waiting pool is empty")

        use_prediction = self.lifecycle.active and all(
            request.predicted_work is not None for request in self._waiting.values()
        )
        if use_prediction:
            request = min(
                self._waiting.values(),
                key=lambda row: (-float(row.predicted_work), row.arrival_index, row.request_id),
            )
            reason = "predicted_long_first"
        else:
            request = min(
                self._waiting.values(),
                key=lambda row: (row.arrival_index, row.request_id),
            )
            reason = "fcfs_fallback"

        self._waiting.pop(request.request_id)
        return DispatchDecision(
            request_id=request.request_id,
            reason=reason,
            predicted_work=request.predicted_work,
        )
