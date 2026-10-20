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
"""Process-group membership helpers for fault-tolerant build_process_group.

Spec §横向系统合同 (CheckpointEngineMgr):
    build_process_group 改 per-ref ray.get(timeout=T) + dead 剔除后用 alive rebuild.

This module exposes a pure helper that bounds the wait on a list of ObjectRefs
and classifies any timed-out refs by side (trainer vs rollout). The caller
(CheckpointEngineManager) then decides whether to fail terminally or — once
P2 Supervisor lands — rebuild with the alive subset.

Rationale (validated via /codex:rescue preview):
  - Surviving ranks like 0,2,3 break NCCL's contiguous-rank assumption, so we
    cannot use partial metadata to keep building. On any timeout, return only
    the dead-index lists and abandon the generation.
  - `ray.wait`'s `ready` list is unordered. When all refs ready, we call
    `ray.get(original_refs)` to preserve ordering required by `build_topology`.
  - `ray.wait` / `ray.get` are injected so this module is testable without
    `ray.init()`.
"""

from __future__ import annotations

from typing import Any, Callable


def split_refs_by_timeout(
    refs: list[Any],
    *,
    trainer_world_size: int,
    timeout_s: float,
    ray_wait_fn: Callable[..., tuple[list[Any], list[Any]]],
    ray_get_fn: Callable[[list[Any]], list[Any]],
) -> tuple[list[Any], list[int], list[int]]:
    """Wait on `refs` with a deadline; classify failures.

    Layout convention: `refs[:trainer_world_size]` are trainer refs, the rest
    are rollout refs.

    Returns:
        On full success — `(values_in_input_order, [], [])` where values are
        the resolved results of every ref.

        On any timeout — `([], trainer_dead_indices, rollout_dead_indices)`.
        `trainer_dead_indices` index into the trainer-side slice (0..trainer_world_size-1);
        `rollout_dead_indices` index into the rollout-side slice (0..rollout_world_size-1).
        Values are NOT returned because surviving metadata would have holes
        that violate NCCL's contiguous-rank invariant.
    """
    if not refs:
        return [], [], []

    ready, not_ready = ray_wait_fn(refs, num_returns=len(refs), timeout=timeout_s)

    if not not_ready:
        return list(ray_get_fn(list(refs))), [], []

    not_ready_set = {id(r) for r in not_ready}
    trainer_dead: list[int] = [i for i in range(trainer_world_size) if id(refs[i]) in not_ready_set]
    rollout_dead: list[int] = [
        i - trainer_world_size for i in range(trainer_world_size, len(refs)) if id(refs[i]) in not_ready_set
    ]
    return [], trainer_dead, rollout_dead
