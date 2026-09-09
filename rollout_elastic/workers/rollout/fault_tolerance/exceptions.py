# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fault tolerance exceptions and the `is_transient_fault` equivalence class.

Per spec §硬规则: production code must catch via `is_transient_fault(exc)`,
not by specific Ray exception class — Ray wraps/subclasses unpredictably across
versions.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import ray.exceptions


def _resolve_ray_exc(name: str):
    """Some Ray exception classes (e.g. ActorUnavailableError) are version-gated."""
    return getattr(ray.exceptions, name, None)


# Equivalence class: any exception in here means "transient fault, retry"
RAY_FAULT_EXCEPTIONS: tuple = tuple(
    e
    for e in [
        _resolve_ray_exc("RayActorError"),
        _resolve_ray_exc("ActorDiedError"),
        _resolve_ray_exc("ActorUnavailableError"),
        _resolve_ray_exc("WorkerCrashedError"),
        _resolve_ray_exc("OutOfMemoryError"),
        _resolve_ray_exc("GetTimeoutError"),
        ConnectionError,
        asyncio.TimeoutError,
    ]
    if e is not None
)


def is_transient_fault(exc: BaseException) -> bool:
    """Whether this exception should trigger transient fault retry.

    Catches direct instances of RAY_FAULT_EXCEPTIONS plus Ray's wrapper
    `RayTaskError` whose `.cause` / `.__cause__` is a transient fault.
    """
    if isinstance(exc, RAY_FAULT_EXCEPTIONS):
        return True
    ray_task_error = _resolve_ray_exc("RayTaskError")
    if ray_task_error is not None and isinstance(exc, ray_task_error):
        cause = getattr(exc, "cause", None) or getattr(exc, "__cause__", None)
        if cause is not None and isinstance(cause, RAY_FAULT_EXCEPTIONS):
            return True
    return False


class ServerUnavailable(Exception):
    """A single server is unavailable; client should switch and retry.

    Raised by L2 (`LLMServerClient.generate`); caught by L3 (`FullyLLMServerClient`).
    """

    def __init__(self, server_id: str, cause: Optional[BaseException] = None) -> None:
        self.server_id = server_id
        self.cause = cause
        super().__init__(f"server {server_id} unavailable: {cause!r}")

    def __reduce__(self):
        # Ray propagates exceptions via pickle. Default Exception.__reduce__ uses
        # `self.args` which we overwrote with a formatted string in __init__, so
        # without this override an unpickled instance loses .server_id / .cause.
        return (type(self), (self.server_id, self.cause))


class AllServersFailed(Exception):
    """L3 retry budget exhausted across all available servers.

    Raised by L3; caught by L4 (`AgentLoopWorker`) which decides batch policy.
    """


class BatchMostlyFailed(Exception):
    """L4 saw too many per-prompt failures (below min_ok_ratio).

    Raised by L4; propagates to L5+. Trainer treats as a skipped step.
    """


class BuildGroupPartialFailure(Exception):
    """CheckpointEngineManager.build_process_group dropped dead members.

    Carries the dropped worker handles so the caller can clean up.
    """

    def __init__(self, dead_workers: list, stage: str | None = None) -> None:
        self.dead_workers = list(dead_workers)
        self.stage = stage
        super().__init__(f"build_process_group dropped {len(self.dead_workers)} dead members")

    def __reduce__(self):
        return (type(self), (self.dead_workers, self.stage))


class WeightSyncStageFailure(Exception):
    """A weight-sync stage ended without a committable result.

    The exception contains only small, serializable failure summaries.  The
    checkpoint-engine manager uses rollout failures to prune the failed
    replica and retry the whole transaction with a fresh member snapshot;
    trainer failures remain terminal for the current update.
    """

    def __init__(
        self,
        stage: str,
        attempt_id: int,
        failures: tuple,
        membership_changed: bool = False,
    ) -> None:
        self.stage = stage
        self.attempt_id = attempt_id
        self.failures = tuple(failures)
        self.membership_changed = membership_changed
        super().__init__(
            f"weight-sync stage {stage!r} failed in attempt {attempt_id} with {len(self.failures)} member failure(s)"
        )

    @property
    def trainer_failed(self) -> bool:
        return any(getattr(getattr(failure, "member", None), "side", None) == "trainer" for failure in self.failures)

    @property
    def failed_replica_ids(self) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for failure in self.failures:
            member = getattr(failure, "member", None)
            replica_id = getattr(member, "replica_id", None)
            if getattr(member, "side", None) == "rollout" and replica_id is not None and replica_id not in seen:
                seen.add(replica_id)
                result.append(replica_id)
        return tuple(result)

    @property
    def retryable(self) -> bool:
        return self.membership_changed or (
            bool(self.failures)
            and not self.trainer_failed
            and all(getattr(failure, "retryable", False) for failure in self.failures)
        )

    def __reduce__(self):
        return (
            type(self),
            (self.stage, self.attempt_id, self.failures, self.membership_changed),
        )
