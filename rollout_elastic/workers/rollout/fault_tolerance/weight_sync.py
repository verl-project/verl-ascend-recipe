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
"""Contracts and bounded waiters for checkpoint-engine weight synchronisation.

Weight synchronisation is a multi-stage transaction.  The helpers in this
module deliberately keep stage results separate from recovery policy: they
preserve member identity, bound every wait, and classify failures so the
checkpoint-engine manager can decide whether to retry a rollout replica or
abort the update because the trainer side failed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Sequence

from verl.workers.rollout.fault_tolerance.exceptions import is_transient_fault


class WeightSyncStage(str, Enum):
    """Observable stages of one weight synchronisation attempt."""

    ABORT_REQUESTS = "abort_requests"
    SLEEP = "sleep"
    WAKE_UP = "wake_up"
    RESUME = "resume"
    PREPARE = "prepare"
    RESET_GROUP = "reset_group"
    INIT_PROCESS_GROUP = "init_process_group"
    TRANSFER = "transfer"
    FINALIZE = "finalize"
    VERIFY_VERSION = "verify_version"
    POST_COMMIT = "post_commit"


class StaleWeightSyncAttempt(RuntimeError):
    """A bucket belongs to an attempt that has already been superseded."""

    def __init__(self, attempt_id: int | None, target_version: int | None) -> None:
        self.attempt_id = attempt_id
        self.target_version = target_version
        super().__init__(f"stale weight-sync attempt: attempt_id={attempt_id}, target_version={target_version}")


class WeightSyncFence:
    """Serialize model writes and reject older attempt contexts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempt_id: int | None = None
        self._target_version: int | None = None

    def begin(self, attempt_id: int, target_version: int | None) -> bool:
        with self._lock:
            if self._attempt_id is not None and attempt_id < self._attempt_id:
                return False
            if self._attempt_id == attempt_id and self._target_version != target_version:
                return False
            self._attempt_id = attempt_id
            self._target_version = target_version
            return True

    def is_current(self, attempt_id: int | None, target_version: int | None) -> bool:
        with self._lock:
            return attempt_id is not None and self._attempt_id == attempt_id and self._target_version == target_version

    def apply_if_current(self, attempt_id: int, target_version: int | None, callback: Callable[[], Any]) -> bool:
        # Keep the fence held through the actual model write.  A newer attempt
        # can begin only after the old bucket callback has finished.
        with self._lock:
            if self._attempt_id != attempt_id or self._target_version != target_version:
                return False
            callback()
            return True


@dataclass(frozen=True)
class SyncMember:
    """Stable identity attached to one remote operation."""

    side: Literal["trainer", "rollout"]
    worker_index: int
    replica_id: str | None = None


@dataclass(frozen=True)
class StageRef:
    """A Ray object reference together with the member that owns it."""

    ref: Any
    member: SyncMember


@dataclass(frozen=True)
class ReplicaSnapshot:
    """Immutable member view used by one weight-sync attempt."""

    replica_id: str
    replica: Any
    workers: tuple[Any, ...]


@dataclass(frozen=True)
class WeightSyncAttempt:
    """Versioned transaction input captured before remote work starts."""

    attempt_id: int
    target_version: int | None
    membership_epoch: int
    replicas: tuple[ReplicaSnapshot, ...]


@dataclass(frozen=True)
class StageFailure:
    """Serializable summary of one failed stage operation."""

    member: SyncMember
    kind: str
    error_type: str
    error_message: str
    retryable: bool


@dataclass(frozen=True)
class StageResult:
    """Results collected in input order plus per-member failures.

    Successful values are retained even when another member failed so that
    tests and diagnostics can identify exactly what completed.  The manager
    must treat any non-empty ``failures`` as an abandoned stage and must not
    commit the partial result.
    """

    values: tuple[Any, ...]
    failures: tuple[StageFailure, ...]
    # Membership invalidates the whole attempt.  It is not a Trainer or
    # rollout member failure and must therefore not be expanded per ref.
    membership_changed: bool = False

    @property
    def trainer_failed(self) -> bool:
        return any(failure.member.side == "trainer" for failure in self.failures)

    @property
    def failed_replica_ids(self) -> tuple[str, ...]:
        """Return failed rollout replicas once, retaining first-seen order."""

        result: list[str] = []
        seen: set[str] = set()
        for failure in self.failures:
            replica_id = failure.member.replica_id
            if failure.member.side == "rollout" and replica_id is not None and replica_id not in seen:
                seen.add(replica_id)
                result.append(replica_id)
        return tuple(result)

    @property
    def retryable(self) -> bool:
        """Whether every failure is a retryable rollout-side failure."""

        return self.membership_changed or (
            bool(self.failures) and not self.trainer_failed and all(failure.retryable for failure in self.failures)
        )


def _failure(member: SyncMember, kind: str, error: BaseException, *, retryable: bool | None = None) -> StageFailure:
    if retryable is None:
        retryable = member.side == "rollout" and is_transient_fault(error)
    return StageFailure(
        member=member,
        kind=kind,
        error_type=type(error).__name__,
        error_message=str(error),
        retryable=retryable,
    )


def _classify_error(member: SyncMember, error: BaseException) -> StageFailure:
    if isinstance(error, asyncio.TimeoutError):
        return _failure(member, "timeout", error, retryable=member.side == "rollout")
    if is_transient_fault(error):
        return _failure(
            member,
            "transient_actor_failure",
            error,
            retryable=member.side == "rollout",
        )
    return _failure(member, "remote_error", error, retryable=False)


async def wait_ray_stage(
    refs: Sequence[StageRef],
    *,
    stage: WeightSyncStage,
    timeout_s: float,
    ray_wait_fn: Callable[..., tuple[list[Any], list[Any]]],
    ray_get_fn: Callable[[Any], Any],
    attempt_valid: Callable[[], bool],
) -> StageResult:
    """Resolve Ray refs with a deadline while retaining member identity.

    ``ray.wait`` does not promise input ordering and a single ``ray.get`` on
    all refs can hide which member failed.  We therefore wait in short rounds,
    resolve each ready ref individually, and construct values/failures in the
    original input order.  A membership change invalidates the attempt before
    its result can be used by the caller.
    """

    del stage  # The stage is part of the caller's context; failures carry member identity.
    entries = list(refs)
    if not entries:
        return StageResult((), ())

    successes: dict[int, Any] = {}
    failures: dict[int, StageFailure] = {}
    unresolved: dict[int, StageRef] = {id(entry.ref): entry for entry in entries}
    deadline = time.monotonic() + max(0.0, timeout_s)

    while unresolved:
        if not attempt_valid():
            return StageResult((), (), membership_changed=True)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for ref_id, entry in unresolved.items():
                failures[ref_id] = _failure(
                    entry.member,
                    "timeout",
                    asyncio.TimeoutError(f"{entry.member} stage wait timed out"),
                    retryable=entry.member.side == "rollout",
                )
            break

        pending_refs = [entry.ref for entry in unresolved.values()]
        try:
            ready, _ = await asyncio.to_thread(
                ray_wait_fn,
                pending_refs,
                num_returns=len(pending_refs),
                timeout=min(remaining, 1.0),
            )
        except BaseException as error:
            for ref_id, entry in unresolved.items():
                failures[ref_id] = _classify_error(entry.member, error)
            break

        progressed = False
        for ref in ready:
            ref_id = id(ref)
            entry = unresolved.pop(ref_id, None)
            if entry is None:
                continue
            progressed = True
            try:
                successes[ref_id] = await asyncio.to_thread(ray_get_fn, ref)
            except BaseException as error:
                failures[ref_id] = _classify_error(entry.member, error)

        if not unresolved:
            break
        if not progressed:
            # Injected waiters and real Ray can both return before the deadline;
            # avoid a tight loop while still allowing membership callbacks to run.
            await asyncio.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    if not attempt_valid():
        # A completed remote call is still stale if the membership epoch changed
        # before the stage returned.  The caller must discard all stage values.
        return StageResult((), (), membership_changed=True)

    values = tuple(successes[id(entry.ref)] for entry in entries if id(entry.ref) in successes)
    ordered_failures = tuple(failures[id(entry.ref)] for entry in entries if id(entry.ref) in failures)
    return StageResult(values, ordered_failures)


async def wait_replica_stage(
    operations: Sequence[tuple[SyncMember, Awaitable[Any]]],
    *,
    stage: WeightSyncStage,
    timeout_s: float,
    attempt_valid: Callable[[], bool],
) -> StageResult:
    """Await replica-side operations with one bounded deadline."""

    del stage
    entries = list(operations)
    if not entries:
        return StageResult((), ())

    tasks = [asyncio.ensure_future(operation) for _, operation in entries]
    task_to_index = {task: index for index, task in enumerate(tasks)}
    successes: dict[int, Any] = {}
    failures: dict[int, StageFailure] = {}
    pending = set(tasks)
    deadline = time.monotonic() + max(0.0, timeout_s)

    try:
        while pending:
            if not attempt_valid():
                return StageResult((), (), membership_changed=True)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for task in pending:
                    index = task_to_index[task]
                    failures[index] = _failure(
                        entries[index][0],
                        "timeout",
                        asyncio.TimeoutError("replica stage wait timed out"),
                        retryable=entries[index][0].side == "rollout",
                    )
                break

            done, pending = await asyncio.wait(pending, timeout=min(remaining, 1.0))
            for task in done:
                index = task_to_index[task]
                try:
                    successes[index] = task.result()
                except BaseException as error:
                    failures[index] = _classify_error(entries[index][0], error)

        if not attempt_valid():
            return StageResult((), (), membership_changed=True)

        values = tuple(successes[index] for index in range(len(entries)) if index in successes)
        ordered_failures = tuple(failures[index] for index in sorted(failures))
        return StageResult(values, ordered_failures)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
