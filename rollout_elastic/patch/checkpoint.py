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
"""Patch ``verl.checkpoint_engine`` for elastic-rollout fault tolerance.

This module rewrites the weight-synchronization path into a transactional,
fault-tolerant one:

- ``CheckpointEngineManager`` becomes a transaction manager: membership-aware
  bounded stage waits, per-attempt weight sync, NCCL communicator generation
  resets, and pending-replica promotion after a successful commit.
- ``CheckpointEngineWorker`` learns how to forward ``attempt_id`` /
  ``target_version`` to the server adapter, to destroy NCCL groups, and to
  report replica rank / leader status to the supervisor.
- ``CheckpointEngine`` gains a best-effort ``destroy_nccl_group`` helper.
- ``NCCLCheckpointEngine`` / ``HCCLCheckpointEngine`` gain communicator
  generation invalidation (``force_destroy_nccl_group``) and, for NCCL, a
  frozen transfer context so a late finalize cannot clobber the active
  communicator.

Everything is applied with ``@patch`` / ``@add`` / ``@wrap`` at import time;
no verl source file is modified on disk.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from typing import Any, Optional

import ray
import torch

from verl import checkpoint_engine as _ckpt_engine
from verl.checkpoint_engine.base import _worker_cls
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.fault_tolerance import (
    ReplicaSnapshot,
    StageFailure,
    StageRef,
    SyncMember,
    WeightSyncAttempt,
    WeightSyncStage,
    WeightSyncStageFailure,
    lenient_gather,
    wait_ray_stage,
    wait_replica_stage,
)
from verl.workers.rollout.replica import RolloutReplica

from ._core import add, patch, wrap

logger = logging.getLogger(__name__)

CheckpointEngine = _ckpt_engine.CheckpointEngine
CheckpointEngineWorker = _ckpt_engine.CheckpointEngineWorker
CheckpointEngineManager = _ckpt_engine.CheckpointEngineManager
NCCLCheckpointEngine = _ckpt_engine.NCCLCheckpointEngine
HCCLCheckpointEngine = _ckpt_engine.HCCLCheckpointEngine


# ---------------------------------------------------------------------------
# CheckpointEngine (abstract base)
# ---------------------------------------------------------------------------
@add(CheckpointEngine, "destroy_nccl_group")
def destroy_nccl_group(self) -> bool:
    """Best-effort local destroy of this engine's NCCL collective group."""
    group_name = getattr(self, "group_name", None)
    if group_name is None:
        return True
    try:
        import ray.util.collective as collective

        if collective.is_group_initialized(group_name):
            collective.destroy_collective_group(group_name)
        self._group_initialized = False
        return True
    except Exception as exc:
        logger.warning("[FT] destroy_nccl_group(%s) failed: %r", group_name, exc)
        self._group_initialized = False
        return False


# ---------------------------------------------------------------------------
# CheckpointEngineWorker
# ---------------------------------------------------------------------------
@patch(CheckpointEngineWorker, "update_weights")
@register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
async def update_weights(
    self,
    global_steps: int = None,
    attempt_id: int | None = None,
    target_version: int | None = None,
):
    weights = self.checkpoint_engine.receive_weights()
    if attempt_id is None:
        # Preserve the pre-FT adapter call shape so legacy sync paths keep
        # deriving the visible version from global_steps.
        await self.server_adapter.update_weights(weights, global_steps=global_steps)
    else:
        await self.server_adapter.update_weights(
            weights,
            global_steps=global_steps,
            attempt_id=attempt_id,
            target_version=target_version,
        )


@add(CheckpointEngineWorker, "destroy_nccl_group")
@register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
def _worker_destroy_nccl_group(self) -> bool:
    return self.checkpoint_engine.destroy_nccl_group()


@add(CheckpointEngineWorker, "get_replica_rank")
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def get_replica_rank(self) -> int:
    """Get replica rank from the underlying rollout server adapter."""
    return self.server_adapter.replica_rank


@add(CheckpointEngineWorker, "is_leader_rank")
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def is_leader_rank(self) -> bool:
    """Get leader rank flag from the underlying rollout server adapter."""
    return self.server_adapter.is_leader_rank


# ---------------------------------------------------------------------------
# CheckpointEngineManager — transactional weight sync
# ---------------------------------------------------------------------------
@wrap(CheckpointEngineManager, "__init__")
def manager_init(
    orig,
    self,
    config,
    trainer: RayWorkerGroup,
    replicas: list[RolloutReplica],
    fault_tolerance: Optional[Any] = None,
    load_balancer_handle: Optional[Any] = None,
    sync_failure_reporter: Optional[Any] = None,
    replica_promotion_reporter: Optional[Any] = None,
) -> None:
    orig(self, config, trainer, replicas)
    self.fault_tolerance = fault_tolerance
    self.load_balancer_handle = load_balancer_handle
    self.sync_failure_reporter = sync_failure_reporter
    self.replica_promotion_reporter = replica_promotion_reporter
    self.membership_changed: bool = False
    self._dead_lock: threading.Lock = threading.Lock()
    # This lock serializes complete sync transactions even when auto_await
    # is called from different event loops/threads.
    self._sync_lock: threading.Lock = threading.Lock()
    self._sync_in_progress: bool = False
    # Epoch changes only when an active member is removed.  An attempt
    # carrying an older epoch is stale and cannot commit its weights.
    self._membership_epoch: int = 0
    self._next_attempt_id: int = 0
    self._pending_replicas: list[RolloutReplica] = []
    # Bumped on each _force_reset; broadcast so workers share group_name@resetN.
    self._reset_generation: int = 0


@add(CheckpointEngineManager, "_ft_enabled")
def _ft_enabled(self) -> bool:
    return self.fault_tolerance is not None and self.fault_tolerance.enabled


@add(CheckpointEngineManager, "set_sync_failure_reporter")
def set_sync_failure_reporter(self, reporter: Optional[Any]) -> None:
    """Set the non-blocking Supervisor notification callback."""
    self.sync_failure_reporter = reporter


@add(CheckpointEngineManager, "set_replica_promotion_reporter")
def set_replica_promotion_reporter(self, reporter: Optional[Any]) -> None:
    """Set the Supervisor-owned serving-admission callback."""
    self.replica_promotion_reporter = reporter


@add(CheckpointEngineManager, "add_pending_replicas")
def add_pending_replicas(self, replicas: list[RolloutReplica]) -> None:
    """Register healthy replacements without admitting them to sync/LB yet."""
    with self._dead_lock:
        known_ids = {self._replica_id(replica) for replica in tuple(self.replicas) + tuple(self._pending_replicas)}
        for replica in replicas:
            replica_id = self._replica_id(replica)
            if replica_id not in known_ids:
                self._pending_replicas.append(replica)
                known_ids.add(replica_id)


@add(CheckpointEngineManager, "_replica_id")
def _replica_id(self, replica: RolloutReplica) -> str:
    """Return the identity shared by CKE, Supervisor, and the load balancer."""
    server_id = getattr(replica, "_server_address", None)
    if server_id is not None:
        return str(server_id)
    return f"replica-{getattr(replica, 'replica_rank', 'unknown')}"


@add(CheckpointEngineManager, "_snapshot_replica")
def _snapshot_replica(self, replica: RolloutReplica) -> ReplicaSnapshot:
    """Freeze worker handles so a later list mutation cannot remap ranks."""
    return ReplicaSnapshot(
        replica_id=self._replica_id(replica),
        replica=replica,
        workers=tuple(replica.workers),
    )


@add(CheckpointEngineManager, "_replicas_from_members")
def _replicas_from_members(self, members) -> tuple[ReplicaSnapshot, ...]:
    if members is None:
        with self._dead_lock:
            replicas = tuple(self.replicas) + tuple(self._pending_replicas)
        return tuple(self._snapshot_replica(replica) for replica in replicas)
    return tuple(
        member if isinstance(member, ReplicaSnapshot) else self._snapshot_replica(member) for member in members
    )


@add(CheckpointEngineManager, "_create_attempt")
def _create_attempt(self, global_steps: int | None, retry_members=None) -> WeightSyncAttempt:
    """Capture the target version, epoch, and fixed rollout workers."""
    members = self._replicas_from_members(retry_members)
    with self._dead_lock:
        membership_epoch = self._membership_epoch
        self._next_attempt_id += 1
        attempt_id = self._next_attempt_id
    return WeightSyncAttempt(
        attempt_id=attempt_id,
        target_version=global_steps,
        membership_epoch=membership_epoch,
        replicas=members,
    )


@add(CheckpointEngineManager, "_is_attempt_valid")
def _is_attempt_valid(self, attempt: WeightSyncAttempt) -> bool:
    """Reject work if membership changed or a fixed member disappeared."""
    with self._dead_lock:
        if self._membership_epoch != attempt.membership_epoch:
            return False
        current_by_id = {
            self._replica_id(replica): replica for replica in tuple(self.replicas) + tuple(self._pending_replicas)
        }
    return all(current_by_id.get(snapshot.replica_id) is snapshot.replica for snapshot in attempt.replicas)


@add(CheckpointEngineManager, "_build_rollout_worker_group")
def _build_rollout_worker_group(self, replicas=None) -> RayWorkerGroup:
    if replicas is None:
        with self._dead_lock:
            replicas_snapshot = tuple(self.replicas)
        workers = [worker for replica in replicas_snapshot for worker in replica.workers]
    elif isinstance(replicas, WeightSyncAttempt):
        workers = [worker for snapshot in replicas.replicas for worker in snapshot.workers]
    else:
        members = self._replicas_from_members(replicas)
        workers = [worker for snapshot in members for worker in snapshot.workers]
    return RayWorkerGroup(
        worker_handles=workers,
        ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
    )


@add(CheckpointEngineManager, "_replicas_for_dead_rollout_workers")
def _replicas_for_dead_rollout_workers(self, dead_workers: list, replicas=None) -> list[RolloutReplica]:
    dead_rollout_workers = {idx for side, idx in dead_workers if side == "rollout"}
    if not dead_rollout_workers:
        return []

    dead_replicas = []
    offset = 0
    members = self._replicas_from_members(replicas)
    for snapshot in members:
        next_offset = offset + len(snapshot.workers)
        if any(offset <= idx < next_offset for idx in dead_rollout_workers):
            dead_replicas.append(snapshot.replica)
        offset = next_offset
    return dead_replicas


@add(CheckpointEngineManager, "_remove_replicas_and_servers")
async def _remove_replicas_and_servers(self, replicas: list[RolloutReplica]):
    """Prune local sync membership; Supervisor exclusively owns LB isolation."""
    self.remove_replicas(replicas)


@add(CheckpointEngineManager, "_force_reset_nccl_groups")
async def _force_reset_nccl_groups(self, attempt: WeightSyncAttempt) -> None:
    """FT: broadcast force_destroy_nccl_group to all surviving CheckpointEngineWorkers.

    Called when CKE.membership_changed=True (Supervisor pruned a replica).
    Each worker invalidates its local communicator generation without calling
    a peer-coordinated destroy.  The next build_process_group then initializes
    a fresh N-1 communicator group.

    Every surviving worker must acknowledge the same generation before a
    new attempt can build a communicator. Proceeding after a partial reset
    would allow old and new collectives to share mutable engine state.
    """
    rollout = self._build_rollout_worker_group(attempt)
    self._reset_generation += 1
    gen = self._reset_generation
    refs = self.trainer.execute_checkpoint_engine(
        method=["force_destroy_nccl_group"] * self.trainer.world_size,
        generation=[gen] * self.trainer.world_size,
    ) + rollout.execute_checkpoint_engine(
        method=["force_destroy_nccl_group"] * rollout.world_size,
        generation=[gen] * rollout.world_size,
    )
    if not refs:
        return
    logger.warning(
        "[FT] _force_reset_nccl_groups: broadcasting destroy to %d surviving CKE workers",
        len(refs),
    )
    stage_refs = self._stage_refs(refs, attempt)
    result = await wait_ray_stage(
        stage_refs,
        stage=WeightSyncStage.RESET_GROUP,
        timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
        ray_wait_fn=ray.wait,
        ray_get_fn=ray.get,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    self._raise_stage_failure(attempt, WeightSyncStage.RESET_GROUP, result)
    rejected = tuple(
        StageFailure(
            member=entry.member,
            kind="reset_rejected",
            error_type="ResetRejected",
            error_message="checkpoint engine rejected communicator reset",
            retryable=entry.member.side == "rollout",
        )
        for entry, value in zip(stage_refs, result.values, strict=True)
        if value is False
    )
    if rejected:
        raise WeightSyncStageFailure(
            WeightSyncStage.RESET_GROUP.value,
            attempt.attempt_id,
            rejected,
        )


@add(CheckpointEngineManager, "_stage_refs")
def _stage_refs(self, refs: list, attempt: WeightSyncAttempt) -> list[StageRef]:
    """Attach fixed replica IDs to the flattened Ray worker order."""
    members = [StageRef(ref, SyncMember("trainer", index)) for index, ref in enumerate(refs[: self.trainer.world_size])]
    rollout_refs = refs[self.trainer.world_size :]
    rollout_index = 0
    for snapshot in attempt.replicas:
        for worker_index in range(len(snapshot.workers)):
            if rollout_index >= len(rollout_refs):
                break
            members.append(
                StageRef(
                    rollout_refs[rollout_index],
                    SyncMember("rollout", rollout_index, snapshot.replica_id),
                )
            )
            rollout_index += 1
    if rollout_index != len(rollout_refs):
        raise RuntimeError(
            "weight-sync rollout refs do not match the fixed replica worker snapshot: "
            f"mapped={rollout_index}, refs={len(rollout_refs)}"
        )
    return members


@add(CheckpointEngineManager, "_raise_stage_failure")
def _raise_stage_failure(self, attempt: WeightSyncAttempt, stage: WeightSyncStage, result) -> None:
    if result.failures or result.membership_changed:
        raise WeightSyncStageFailure(
            stage.value,
            attempt.attempt_id,
            result.failures,
            membership_changed=result.membership_changed,
        )


@add(CheckpointEngineManager, "_wait_worker_stage")
async def _wait_worker_stage(
    self,
    refs: list,
    *,
    attempt: WeightSyncAttempt,
    stage: WeightSyncStage,
    timeout_s: float,
) -> tuple[Any, ...]:
    result = await wait_ray_stage(
        self._stage_refs(refs, attempt),
        stage=stage,
        timeout_s=timeout_s,
        ray_wait_fn=ray.wait,
        ray_get_fn=ray.get,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    self._raise_stage_failure(attempt, stage, result)
    return result.values


@patch(CheckpointEngineManager, "build_process_group")
async def build_process_group(
    self,
    rollout: RayWorkerGroup | None = None,
    attempt: WeightSyncAttempt | None = None,
):
    """Build a process group, using member-aware bounded waits in FT mode."""
    trainer = self.trainer
    ft_on = self._ft_enabled()
    if rollout is None:
        rollout = self._build_rollout_worker_group(attempt)
    if not ft_on:
        prepare_refs = trainer.execute_checkpoint_engine(
            ["prepare"] * trainer.world_size
        ) + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        metadata = ray.get(prepare_refs)
        trainer_kwargs, rollout_kwargs = self.backend_cls.build_topology(
            trainer.world_size, rollout.world_size, metadata
        )
        trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size
        init_refs = trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(
            **rollout_kwargs
        )
        ray.get(init_refs)
        return

    if attempt is None:
        attempt = self._create_attempt(global_steps=None)
    timeout_s = self.fault_tolerance.weight_sync_member_timeout_s

    # Prepare and init are separate stages so a failure is attributed to
    # the exact remote member before the transaction is retried.
    prepare_refs = trainer.execute_checkpoint_engine(
        ["prepare"] * trainer.world_size
    ) + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
    metadata = await self._wait_worker_stage(
        prepare_refs,
        attempt=attempt,
        stage=WeightSyncStage.PREPARE,
        timeout_s=timeout_s,
    )

    trainer_kwargs, rollout_kwargs = self.backend_cls.build_topology(
        trainer.world_size, rollout.world_size, list(metadata)
    )
    for key, values in trainer_kwargs.items():
        assert len(values) == trainer.world_size, f"trainer_kwargs[{key}] must have length of {trainer.world_size}"
    for key, values in rollout_kwargs.items():
        assert len(values) == rollout.world_size, f"rollout_kwargs[{key}] must have length of {rollout.world_size}"

    trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
    rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size
    init_refs = trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(
        **rollout_kwargs
    )
    await self._wait_worker_stage(
        init_refs,
        attempt=attempt,
        stage=WeightSyncStage.INIT_PROCESS_GROUP,
        timeout_s=timeout_s,
    )


@patch(CheckpointEngineManager, "add_replicas")
def add_replicas(self, replicas: list[RolloutReplica]):
    """Register rollout replicas, deferring arrivals during a sync attempt.

    Args:
        replicas: The list of rollout replicas to add.
    """
    with self._dead_lock:
        if self._ft_enabled() and self._sync_in_progress:
            known_ids = {self._replica_id(replica) for replica in tuple(self.replicas) + tuple(self._pending_replicas)}
            for replica in replicas:
                replica_id = self._replica_id(replica)
                if replica_id in known_ids:
                    continue
                self._pending_replicas.append(replica)
                known_ids.add(replica_id)
            logger.info(
                "[FT] queued %d replacement replica(s) until the next successful weight sync",
                len(replicas),
            )
        else:
            self.replicas.extend(replicas)


@patch(CheckpointEngineManager, "remove_replicas")
def remove_replicas(self, replicas: list[RolloutReplica]):
    """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

    Args:
        replicas: The list of rollout replicas to remove.
    """
    remove_ids = {self._replica_id(replica) for replica in replicas}
    with self._dead_lock:
        active_before = {self._replica_id(replica) for replica in self.replicas}
        self.replicas[:] = [r for r in self.replicas if self._replica_id(r) not in remove_ids]
        self._pending_replicas[:] = [r for r in self._pending_replicas if self._replica_id(r) not in remove_ids]
        if self._ft_enabled() and active_before - {self._replica_id(replica) for replica in self.replicas}:
            self._membership_epoch += 1
            self.membership_changed = True


@add(CheckpointEngineManager, "on_replica_dead")
async def on_replica_dead(self, replica_id: str) -> None:
    """Remove a dead rollout replica and mark NCCL membership dirty.

    Called from the ThreadedSupervisor's sub-thread. The lock pairs with
    the read/clear in ``update_weights`` so the (replicas, membership_changed)
    pair stays consistent across threads.
    """
    with self._dead_lock:
        matches = lambda replica: (
            self._replica_id(replica) == str(replica_id) or str(getattr(replica, "replica_rank", "")) == str(replica_id)
        )
        dead_replicas = [replica for replica in self.replicas if matches(replica)]
        pending_replicas = [replica for replica in self._pending_replicas if matches(replica)]
        if not dead_replicas and not pending_replicas:
            logger.warning("[FT] on_replica_dead(%s): no matching rollout replica", replica_id)
            return
        dead_ids = {self._replica_id(replica) for replica in dead_replicas + pending_replicas}
        self.replicas[:] = [r for r in self.replicas if self._replica_id(r) not in dead_ids]
        self._pending_replicas[:] = [r for r in self._pending_replicas if self._replica_id(r) not in dead_ids]
        if dead_replicas:
            self._membership_epoch += 1
            self.membership_changed = True
        epoch = self._membership_epoch
    logger.warning(
        "[FT] on_replica_dead(%s): pruned %d active and %d pending replica(s); epoch=%d",
        replica_id,
        len(dead_replicas),
        len(pending_replicas),
        epoch,
    )


@add(CheckpointEngineManager, "_membership_stage_failure")
def _membership_stage_failure(self, attempt: WeightSyncAttempt, stage: WeightSyncStage) -> WeightSyncStageFailure:
    """Build a stale-attempt error without attributing it to a wrong worker."""
    return WeightSyncStageFailure(
        stage.value,
        attempt.attempt_id,
        (),
        membership_changed=True,
    )


@add(CheckpointEngineManager, "_collect_replica_stage")
async def _collect_replica_stage(
    self,
    attempt: WeightSyncAttempt,
    stage: WeightSyncStage,
    operation_factory,
    timeout_s: float,
):
    """Run one operation per fixed replica and retain all member outcomes."""
    operations = []
    for index, snapshot in enumerate(attempt.replicas):
        try:
            operation = operation_factory(snapshot.replica)
        except BaseException as error:

            async def failed_operation(error=error):
                raise error

            operation = failed_operation()
        operations.append((SyncMember("rollout", index, snapshot.replica_id), operation))
    result = await wait_replica_stage(
        operations,
        stage=stage,
        timeout_s=timeout_s,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    return result


@add(CheckpointEngineManager, "_run_replica_stage")
async def _run_replica_stage(
    self,
    attempt: WeightSyncAttempt,
    stage: WeightSyncStage,
    operation_factory,
    timeout_s: float,
) -> tuple[Any, ...]:
    """Run one fixed-membership replica stage and raise on partial success."""
    result = await self._collect_replica_stage(
        attempt,
        stage,
        operation_factory,
        timeout_s,
    )
    self._raise_stage_failure(attempt, stage, result)
    return result.values


@add(CheckpointEngineManager, "_execute_attempt")
async def _execute_attempt(self, attempt: WeightSyncAttempt) -> None:
    """Execute all pre-commit stages for one fixed membership snapshot."""
    member_timeout_s = self.fault_tolerance.weight_sync_member_timeout_s
    await self._run_replica_stage(
        attempt,
        WeightSyncStage.ABORT_REQUESTS,
        lambda replica: replica.abort_all_requests(),
        member_timeout_s,
    )
    await self._run_replica_stage(
        attempt,
        WeightSyncStage.SLEEP,
        lambda replica: replica.sleep(),
        member_timeout_s,
    )

    rollout = self._build_rollout_worker_group(attempt)
    await self.build_process_group(rollout=rollout, attempt=attempt)
    await self._begin_target_version(attempt)

    trainer_refs = self.trainer.update_weights(global_steps=attempt.target_version)
    rollout_refs = rollout.update_weights(
        global_steps=attempt.target_version,
        attempt_id=attempt.attempt_id,
        target_version=attempt.target_version,
    )
    transfer_refs = trainer_refs + rollout_refs
    transfer_result = await wait_ray_stage(
        self._stage_refs(transfer_refs, attempt),
        stage=WeightSyncStage.TRANSFER,
        timeout_s=self.fault_tolerance.weight_sync_transfer_timeout_s,
        ray_wait_fn=ray.wait,
        ray_get_fn=ray.get,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    self._raise_stage_failure(attempt, WeightSyncStage.TRANSFER, transfer_result)

    finalize_kwargs = {}
    if self.backend_cls.__name__ in {"NCCLCheckpointEngine", "HCCLCheckpointEngine"}:
        # A late finalize from an older generation must not destroy the
        # communicator opened for the current retry.
        generation = self._reset_generation
        finalize_kwargs = {"reset_generation": [generation] * self.trainer.world_size}
        trainer_finalize_refs = self.trainer.execute_checkpoint_engine(
            method=["finalize"] * self.trainer.world_size,
            **finalize_kwargs,
        )
        rollout_finalize_kwargs = {"reset_generation": [generation] * rollout.world_size}
        rollout_finalize_refs = rollout.execute_checkpoint_engine(
            method=["finalize"] * rollout.world_size,
            **rollout_finalize_kwargs,
        )
    else:
        trainer_finalize_refs = self.trainer.execute_checkpoint_engine(["finalize"] * self.trainer.world_size)
        rollout_finalize_refs = rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
    finalize_refs = trainer_finalize_refs + rollout_finalize_refs
    finalize_result = await wait_ray_stage(
        self._stage_refs(finalize_refs, attempt),
        stage=WeightSyncStage.FINALIZE,
        timeout_s=member_timeout_s,
        ray_wait_fn=ray.wait,
        ray_get_fn=ray.get,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    self._raise_stage_failure(attempt, WeightSyncStage.FINALIZE, finalize_result)
    await self._commit_target_version(attempt)
    await self._verify_target_version(attempt)


@add(CheckpointEngineManager, "_report_sync_failure")
def _report_sync_failure(self, replica_ids: tuple[str, ...], stage: str) -> None:
    """Notify Supervisor asynchronously after local sync-side pruning."""
    if self.sync_failure_reporter is None:
        return
    for replica_id in replica_ids:
        try:
            result = self.sync_failure_reporter(replica_id, f"weight_sync:{stage}")
        except Exception as error:
            logger.warning("[FT] sync failure reporter rejected %s: %r", replica_id, error)
            continue
        if inspect.isawaitable(result):

            async def drain_report(result=result, replica_id=replica_id):
                try:
                    await result
                except Exception as error:
                    logger.warning("[FT] async sync failure report for %s failed: %r", replica_id, error)

            asyncio.create_task(drain_report())


@add(CheckpointEngineManager, "_handle_stage_failure")
def _handle_stage_failure(
    self, attempt: WeightSyncAttempt, failure: WeightSyncStageFailure
) -> tuple[ReplicaSnapshot, ...]:
    """Prune failed rollout replicas and return the next retry snapshot."""
    if failure.trainer_failed or not failure.retryable:
        raise failure

    membership_changed = failure.membership_changed or any(
        item.kind == "membership_changed" for item in failure.failures
    )
    failed_ids = set(failure.failed_replica_ids)
    if not failed_ids and not membership_changed:
        raise failure

    with self._dead_lock:
        if membership_changed:
            # Supervisor may have removed the member already.  Reuse only
            # members still registered, without deleting healthy replicas.
            current_ids = {
                self._replica_id(replica) for replica in tuple(self.replicas) + tuple(self._pending_replicas)
            }
        else:
            active_before = {self._replica_id(replica) for replica in self.replicas}
            self.replicas[:] = [replica for replica in self.replicas if self._replica_id(replica) not in failed_ids]
            self._pending_replicas[:] = [
                replica for replica in self._pending_replicas if self._replica_id(replica) not in failed_ids
            ]
            active_removed = active_before - {self._replica_id(replica) for replica in self.replicas}
            if active_removed:
                self._membership_epoch += 1
                self.membership_changed = True
            current_ids = {
                self._replica_id(replica) for replica in tuple(self.replicas) + tuple(self._pending_replicas)
            }

        surviving = tuple(
            snapshot
            for snapshot in attempt.replicas
            if snapshot.replica_id in current_ids and (membership_changed or snapshot.replica_id not in failed_ids)
        )

    if not surviving:
        raise failure
    return surviving


@add(CheckpointEngineManager, "_reset_surviving_groups")
async def _reset_surviving_groups(self, attempt: WeightSyncAttempt) -> None:
    """Destroy stale communicators once before replaying the next attempt."""
    await self._force_reset_nccl_groups(attempt=attempt)
    with self._dead_lock:
        if self._membership_epoch == attempt.membership_epoch:
            self.membership_changed = False


@add(CheckpointEngineManager, "_server_operations")
def _server_operations(self, attempt: WeightSyncAttempt, method_name: str, *args, **kwargs):
    """Build stable server operations for every node in the attempt snapshot."""
    operations = []
    for snapshot in attempt.replicas:
        for server_index, server in enumerate(getattr(snapshot.replica, "servers", ())):
            method = getattr(server, method_name, None)
            if method is None:
                continue
            operations.append(
                (
                    SyncMember("rollout", server_index, snapshot.replica_id),
                    method.remote(*args, **kwargs),
                )
            )
    return operations


@add(CheckpointEngineManager, "_begin_target_version")
async def _begin_target_version(self, attempt: WeightSyncAttempt) -> None:
    """Fence every node-level server before any bucket for this attempt."""
    operations = self._server_operations(
        attempt,
        "begin_weight_sync",
        attempt.attempt_id,
        attempt.target_version,
    )
    result = await wait_replica_stage(
        operations,
        stage=WeightSyncStage.TRANSFER,
        timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    self._raise_stage_failure(attempt, WeightSyncStage.TRANSFER, result)
    rejected = tuple(
        StageFailure(
            member=member,
            kind="stale_attempt",
            error_type="StaleWeightSyncAttempt",
            error_message="server rejected the weight-sync attempt",
            retryable=True,
        )
        for (member, _), accepted in zip(operations, result.values, strict=True)
        if accepted is False
    )
    if rejected:
        raise WeightSyncStageFailure(
            WeightSyncStage.TRANSFER.value,
            attempt.attempt_id,
            rejected,
        )


@add(CheckpointEngineManager, "_commit_target_version")
async def _commit_target_version(self, attempt: WeightSyncAttempt) -> None:
    """Expose the target version only after all checkpoint engines finalize."""
    if attempt.target_version is None:
        return
    operations = self._server_operations(
        attempt,
        "set_global_steps",
        attempt.target_version,
        attempt_id=attempt.attempt_id,
    )
    result = await wait_replica_stage(
        operations,
        stage=WeightSyncStage.VERIFY_VERSION,
        timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    self._raise_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION, result)


@add(CheckpointEngineManager, "_verify_target_version")
async def _verify_target_version(self, attempt: WeightSyncAttempt) -> None:
    """Confirm every available vLLM server applied the requested version."""
    if attempt.target_version is None:
        if not self._is_attempt_valid(attempt):
            raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)
        return

    operations = self._server_operations(attempt, "get_global_steps")
    result = await wait_replica_stage(
        operations,
        stage=WeightSyncStage.VERIFY_VERSION,
        timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
        attempt_valid=lambda: self._is_attempt_valid(attempt),
    )
    self._raise_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION, result)
    if len(result.values) != len(operations):
        raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)

    failures = []
    value_index = 0
    for snapshot in attempt.replicas:
        for server_index, server in enumerate(getattr(snapshot.replica, "servers", ())):
            if getattr(server, "get_global_steps", None) is None:
                continue
            actual = result.values[value_index]
            value_index += 1
            if actual != attempt.target_version:
                failures.append(
                    StageFailure(
                        member=SyncMember("rollout", server_index, snapshot.replica_id),
                        kind="version_mismatch",
                        error_type="VersionMismatch",
                        error_message=(f"expected version {attempt.target_version}, got {actual}"),
                        retryable=True,
                    )
                )
    if failures:
        raise WeightSyncStageFailure(
            WeightSyncStage.VERIFY_VERSION.value,
            attempt.attempt_id,
            tuple(failures),
        )


@add(CheckpointEngineManager, "_commit_attempt")
def _commit_attempt(self, attempt: WeightSyncAttempt) -> None:
    """Commit only an epoch-valid attempt; serving admission remains separate."""
    if not self._is_attempt_valid(attempt):
        raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)

    attempt_ids = {snapshot.replica_id for snapshot in attempt.replicas}
    with self._dead_lock:
        current_ids = {self._replica_id(replica) for replica in tuple(self.replicas) + tuple(self._pending_replicas)}
        if self._membership_epoch != attempt.membership_epoch or not attempt_ids.issubset(current_ids):
            raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)

        self.membership_changed = False


@add(CheckpointEngineManager, "_publish_promoted_replicas")
async def _publish_promoted_replicas(self, attempt: WeightSyncAttempt) -> None:
    """Move synced pending replicas to active only after LB acknowledgement."""
    attempt_ids = {snapshot.replica_id for snapshot in attempt.replicas}
    with self._dead_lock:
        promoted = [replica for replica in self._pending_replicas if self._replica_id(replica) in attempt_ids]
    if not promoted:
        return
    acknowledged = list(promoted)
    if self.replica_promotion_reporter is not None:
        acknowledged = []
        for replica in promoted:
            replica_id = self._replica_id(replica)
            address = getattr(replica, "_server_address", None)
            handle = getattr(replica, "_server_handle", None)
            servers = {address: handle} if address is not None and handle is not None else {}
            try:
                result = self.replica_promotion_reporter(
                    replica_id,
                    servers,
                    attempt.attempt_id,
                    attempt.target_version,
                )
                if inspect.isawaitable(result):
                    result = await result
            except Exception as error:
                logger.warning(
                    "[FT] keeping synced replacement %s pending after Supervisor publish failed: %r",
                    replica_id,
                    error,
                )
                continue
            if result:
                acknowledged.append(replica)
    else:
        servers = {
            replica._server_address: replica._server_handle
            for replica in promoted
            if getattr(replica, "_server_address", None) is not None
            and getattr(replica, "_server_handle", None) is not None
        }
    if self.replica_promotion_reporter is None and servers and self.load_balancer_handle is not None:
        try:
            await self.load_balancer_handle.add_servers.remote(servers)
        except Exception as error:
            logger.warning(
                "[FT] keeping %d synced replacement replica(s) pending after LB publish failed: %r",
                len(promoted),
                error,
            )
            return
    promoted_ids = {self._replica_id(replica) for replica in acknowledged}
    with self._dead_lock:
        current = [replica for replica in self._pending_replicas if self._replica_id(replica) in promoted_ids]
        current_ids = {self._replica_id(replica) for replica in current}
        self._pending_replicas[:] = [
            replica for replica in self._pending_replicas if self._replica_id(replica) not in current_ids
        ]
        active_ids = {self._replica_id(replica) for replica in self.replicas}
        self.replicas.extend(replica for replica in current if self._replica_id(replica) not in active_ids)


@add(CheckpointEngineManager, "_wake_and_resume")
async def _wake_and_resume(self, attempt: WeightSyncAttempt) -> None:
    """Resume healthy committed members even when another member fails to wake."""
    member_timeout_s = self.fault_tolerance.weight_sync_member_timeout_s
    wake_result = await self._collect_replica_stage(
        attempt,
        WeightSyncStage.WAKE_UP,
        lambda replica: replica.wake_up(),
        member_timeout_s,
    )
    failed_ids = set(wake_result.failed_replica_ids)
    survivors = tuple(snapshot for snapshot in attempt.replicas if snapshot.replica_id not in failed_ids)
    resume_result = None
    if survivors:
        resume_attempt = WeightSyncAttempt(
            attempt_id=attempt.attempt_id,
            target_version=attempt.target_version,
            membership_epoch=attempt.membership_epoch,
            replicas=survivors,
        )
        resume_result = await self._collect_replica_stage(
            resume_attempt,
            WeightSyncStage.RESUME,
            lambda replica: replica.resume_generation(),
            member_timeout_s,
        )
    failures = wake_result.failures + (() if resume_result is None else resume_result.failures)
    if failures:
        raise WeightSyncStageFailure(
            WeightSyncStage.POST_COMMIT.value,
            attempt.attempt_id,
            failures,
        )


@patch(CheckpointEngineManager, "sleep_replicas")
@auto_await
async def sleep_replicas(self):
    """Sleep all rollout replicas: free weight and kv_cache device memory.

    FT path: lenient gather so a dead replica doesn't crash weight sync
    before abort_all_requests / build_process_group get a chance to react.
    """
    if self._ft_enabled():
        await lenient_gather(
            [r.sleep() for r in self.replicas],
            op_name="sleep_replicas",
            logger=logger,
        )
    else:
        await asyncio.gather(*[r.sleep() for r in self.replicas])


@patch(CheckpointEngineManager, "wake_up_replicas")
@auto_await
async def wake_up_replicas(self):
    """Resume all rollout replicas: recover kv_cache and weights device memory."""
    if self._ft_enabled():
        await lenient_gather(
            [r.wake_up() for r in self.replicas],
            op_name="wake_up_replicas",
            logger=logger,
        )
    else:
        await asyncio.gather(*[r.wake_up() for r in self.replicas])


@patch(CheckpointEngineManager, "update_weights")
@auto_await
async def _mgr_update_weights(self, global_steps: int = None):
    """Update weights from trainer to rollout replicas.

    Args:
        global_steps: The global steps of the trainer.
    """

    # 0. update weights for sync training with colocated trainer and rollout
    if self.backend == "naive":
        ray.get(self.trainer.update_weights(global_steps=global_steps))
        return

    if not self._ft_enabled():
        # Keep the non-FT path unchanged: strict, unbounded gathers retain
        # the original failure semantics when fault tolerance is disabled.
        await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])
        rollout = self._build_rollout_worker_group()
        await self.sleep_replicas()
        await self.build_process_group(rollout)
        rollout = self._build_rollout_worker_group()
        ray.get(
            self.trainer.update_weights(global_steps=global_steps) + rollout.update_weights(global_steps=global_steps)
        )
        ray.get(
            self.trainer.execute_checkpoint_engine(["finalize"] * self.trainer.world_size)
            + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        )
        await self.wake_up_replicas()
        await asyncio.gather(*[r.resume_generation() for r in self.replicas])
        return

    if not self._sync_lock.acquire(blocking=False):
        raise RuntimeError("another weight synchronization transaction is already running")

    with self._dead_lock:
        # Capture the initial set while publishing the in-progress flag.
        # A replacement callback that arrives after this point is pending
        # for the next transaction, even if it becomes healthy immediately.
        self._sync_in_progress = True
        initial_members = tuple(self.replicas) + tuple(self._pending_replicas)
        # A pending replacement changes communicator world size even when
        # a prior shrink transaction already cleared membership_changed.
        needs_reset = self.membership_changed or bool(self._pending_replicas)
    try:
        if not initial_members:
            raise RuntimeError("cannot synchronize weights without a rollout replica")

        retry_members = tuple(self._snapshot_replica(replica) for replica in initial_members)
        max_retries = max(0, int(getattr(self.fault_tolerance, "max_weight_sync_retries", 2)))
        for retry_index in range(max_retries + 1):
            attempt = self._create_attempt(global_steps, retry_members)
            try:
                if needs_reset:
                    await self._reset_surviving_groups(attempt)
                    needs_reset = False
                await self._execute_attempt(attempt)
                self._commit_attempt(attempt)
            except WeightSyncStageFailure as failure:
                if failure.trainer_failed or not failure.retryable or retry_index >= max_retries:
                    raise

                retry_members = self._handle_stage_failure(attempt, failure)
                if not failure.membership_changed and not any(
                    item.kind == "membership_changed" for item in failure.failures
                ):
                    failed_ids = set(failure.failed_replica_ids)
                    self._report_sync_failure(failure.failed_replica_ids, failure.stage)
                    failed_replicas = [
                        snapshot.replica for snapshot in attempt.replicas if snapshot.replica_id in failed_ids
                    ]
                    if failed_replicas:
                        await self._remove_replicas_and_servers(failed_replicas)
                if not retry_members:
                    raise

                # The next attempt owns reset and must receive every ACK
                # before it can build a new communicator generation.
                needs_reset = True
            else:
                # Commit is the transaction boundary.  A wake/resume
                # failure after commit must not replay the full model or
                # create a second committed attempt; report the rollout
                # failure and let the caller handle the post-commit error.
                try:
                    await self._wake_and_resume(attempt)
                except WeightSyncStageFailure as failure:
                    if failure.retryable:
                        self._report_sync_failure(failure.failed_replica_ids, failure.stage)
                        failed_ids = set(failure.failed_replica_ids)
                        failed_replicas = [
                            snapshot.replica for snapshot in attempt.replicas if snapshot.replica_id in failed_ids
                        ]
                        if failed_replicas:
                            await self._remove_replicas_and_servers(failed_replicas)
                        remaining_ids = {
                            self._replica_id(replica)
                            for replica in tuple(self.replicas) + tuple(self._pending_replicas)
                        }
                        if not remaining_ids:
                            raise
                    else:
                        raise
                await self._publish_promoted_replicas(attempt)
                return
    finally:
        with self._dead_lock:
            self._sync_in_progress = False
        self._sync_lock.release()


# ---------------------------------------------------------------------------
# NCCLCheckpointEngine — communicator generation invalidation
# ---------------------------------------------------------------------------
if NCCLCheckpointEngine is not None:

    @wrap(NCCLCheckpointEngine, "__init__")
    def nccl_engine_init(
        orig,
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: Any = None,
    ) -> None:
        if rollout_dtype is None:
            rollout_dtype = torch.bfloat16
        orig(self, bucket_size, group_name, rebuild_group, is_master, rollout_dtype)
        self._reset_count = 0
        self._group_initialized = False
        self.topic = "bucket_metadata@reset0"

    @patch(NCCLCheckpointEngine, "finalize")
    def nccl_engine_finalize(self, reset_generation: int | None = None):
        """Destroy the NCCL process group if rebuild_group is True."""
        if reset_generation is not None and reset_generation < getattr(self, "_reset_count", 0):
            return True
        if self.rebuild_group:
            if getattr(self, "rank", None) is not None and self.rank >= 0:
                import ray.util.collective as collective

                collective.destroy_collective_group(self.group_name)
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None

        import torch

        torch.cuda.empty_cache()

    @add(NCCLCheckpointEngine, "force_destroy_nccl_group")
    def nccl_force_destroy_nccl_group(self, generation: int = None) -> bool:
        """Invalidate the local communicator generation without peer sync.

        Reset requests are idempotent and monotonic.  This matters when the
        Supervisor and a sync-stage failure report the same replica: an older
        callback must not move a worker back to a stale communicator name.
        """
        last_generation = getattr(self, "_reset_count", 0)
        if generation is not None and int(generation) <= last_generation:
            return True
        self.rank = None
        self.world_size = None
        if generation is not None:
            self._reset_count = int(generation)
        else:
            self._reset_count = last_generation + 1
        base = self.group_name.split("@reset", 1)[0]
        self.group_name = f"{base}@reset{self._reset_count}"
        self.topic = f"bucket_metadata@reset{self._reset_count}"
        self._group_initialized = False
        return True

    @add(NCCLCheckpointEngine, "_capture_transfer_context")
    def nccl_capture_transfer_context(self):
        """Freeze mutable generation state for one complete transfer coroutine."""
        return self.group_name, self.topic, self.socket

    @patch(NCCLCheckpointEngine, "send_weights")
    @torch.no_grad()
    async def nccl_engine_send_weights(self, weights):
        """Send the weights of the model, freezing the communicator context."""
        import time

        import torch

        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."

        # For trainer rank other than 0, consume weights without sending.
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        group_name, topic, socket = self._capture_transfer_context()
        broadcast_op = None

        start_time = time.time()
        bucket_meta: dict[str, Any] = {}
        offset = 0
        async for tensor_meta, chunk in _split_weight_chunks(weights, self.bucket_size):
            # fill the tensor bucket
            if offset + tensor_meta.chunk_size > self.bucket_size:
                torch.cuda.synchronize()

                # wait previous broadcast op finish
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = _NCCLBroadcastOperation(
                    rank=self.rank,
                    group_name=group_name,
                    bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False},
                    socket=socket,
                    topic=topic,
                )

                # swap send_buf and recv_buf
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size] = _as_cupy(chunk)
            offset += tensor_meta.chunk_size

        # broadcast last bucket
        torch.cuda.synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = _NCCLBroadcastOperation(
            rank=self.rank,
            group_name=group_name,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=socket,
            topic=topic,
        )
        await broadcast_op.wait_for_complete()
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @patch(NCCLCheckpointEngine, "_receive_weight_chunks")
    async def nccl_engine_receive_weight_chunks(self):
        """Receive the weight chunks of the model, freezing the communicator context."""
        import time

        import torch

        assert self.rank > 0, "Rank 0 should not receive weights."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        group_name, topic, socket = self._capture_transfer_context()
        total_bytes, total_params = 0, 0

        # receive first bucket
        start_time = time.time()
        broadcast_op = _NCCLBroadcastOperation(
            rank=self.rank,
            group_name=group_name,
            bucket=recv_buf,
            metadata=None,
            socket=socket,
            topic=topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # swap send_buf and recv_buf
        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            # 1. receive next bucket
            broadcast_op = _NCCLBroadcastOperation(
                rank=self.rank,
                group_name=group_name,
                bucket=recv_buf,
                metadata=None,
                socket=socket,
                topic=topic,
            )

            # 2. yield tensor from send_buf
            for name, tensor_meta in metadata["bucket_meta"].items():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor

            # 3. wait for next bucket broadcast finish
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            # 4. swap send_buf and recv_buf
            torch.cuda.synchronize()  # sync non-blocking copy
            send_buf, recv_buf = recv_buf, send_buf

        # yield tensor from send_buf
        for name, tensor_meta in metadata["bucket_meta"].items():
            tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            yield tensor_meta, tensor

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )


# ---------------------------------------------------------------------------
# HCCLCheckpointEngine — communicator generation invalidation
# ---------------------------------------------------------------------------
if HCCLCheckpointEngine is not None:

    @wrap(HCCLCheckpointEngine, "__init__")
    def hccl_engine_init(
        orig,
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: Any = None,
    ) -> None:
        if rollout_dtype is None:
            import torch

            rollout_dtype = torch.bfloat16
        orig(self, bucket_size, group_name, rebuild_group, is_master, rollout_dtype)
        self._reset_count = 0

    @patch(HCCLCheckpointEngine, "finalize")
    def hccl_engine_finalize(self, reset_generation: int | None = None):
        """Destroy the HCCL process group if rebuild_group is True."""
        if reset_generation is not None and reset_generation < getattr(self, "_reset_count", 0):
            return True
        if self.rebuild_group:
            if getattr(self, "rank", None) is not None and self.rank >= 0 and self.pyhccl is not None:
                self.pyhccl.destroyComm(self.pyhccl.comm)
                self.pyhccl = None
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None

        import torch

        torch.npu.empty_cache()

    @add(HCCLCheckpointEngine, "force_destroy_nccl_group")
    def hccl_force_destroy_nccl_group(self, generation: int = None) -> bool:
        """Invalidate the local HCCL communicator generation without peer sync."""
        last_generation = getattr(self, "_reset_count", 0)
        if generation is not None and int(generation) <= last_generation:
            return True
        self.rank = None
        self.world_size = None
        if generation is not None:
            self._reset_count = int(generation)
        else:
            self._reset_count = last_generation + 1
        base = self.group_name.split("@reset", 1)[0]
        self.group_name = f"{base}@reset{self._reset_count}"
        self.pyhccl = None
        return True

    # Keep the historical misspelling as a compatibility alias for older
    # callers; new Manager code uses the correctly named method above.
    @add(HCCLCheckpointEngine, "force_destory_nccl_group")
    def hccl_force_destory_nccl_group(self, generation: int = None) -> bool:
        return self.force_destroy_nccl_group(generation=generation)


# ---------------------------------------------------------------------------
# NCCL send/receive helpers (imported lazily to avoid cupy at patch time)
# ---------------------------------------------------------------------------
def _as_cupy(tensor):
    import cupy as cp

    return cp.asarray(tensor)


def _split_weight_chunks(weights, bucket_size):
    from verl.checkpoint_engine.base import split_weight_chunks

    return split_weight_chunks(weights, bucket_size)


def _NCCLBroadcastOperation(*args, **kwargs):
    from verl.checkpoint_engine.nccl_checkpoint_engine import BroadcastOperation

    return BroadcastOperation(*args, **kwargs)
