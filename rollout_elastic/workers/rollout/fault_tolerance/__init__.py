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
"""Fault tolerance for verl asynchronous rollout (P1/P2).

See `experiments/docs/elastic_rollout_spec.md` and
`experiments/docs/fully_async_recovery_design.md` for the design.
"""

from verl.workers.rollout.fault_tolerance.aio import lenient_gather
from verl.workers.rollout.fault_tolerance.batch import filter_partial_batch
from verl.workers.rollout.fault_tolerance.exceptions import (
    RAY_FAULT_EXCEPTIONS,
    AllServersFailed,
    BatchMostlyFailed,
    BuildGroupPartialFailure,
    ServerUnavailable,
    WeightSyncStageFailure,
    is_transient_fault,
)
from verl.workers.rollout.fault_tolerance.group_membership import split_refs_by_timeout
from verl.workers.rollout.fault_tolerance.progress import (
    CheckPointPayLoad,
    GCStats,
    LoadFailure,
    LoadResult,
    ModelVersionPolicy,
    ProgressConfig,
    ProgressContext,
    ResumeOutcome,
    ResumeResult,
    RolloutProgressStoreActor,
    VLLMProgressCheckPoint,
)
from verl.workers.rollout.fault_tolerance.supervisor import (
    HeartbeatTracker,
    Supervisor,
    make_on_dead,
)
from verl.workers.rollout.fault_tolerance.supervisor_thread import ThreadedSupervisor
from verl.workers.rollout.fault_tolerance.types import FaultToleranceConfig
from verl.workers.rollout.fault_tolerance.weight_sync import (
    ReplicaSnapshot,
    StageFailure,
    StageRef,
    StageResult,
    StaleWeightSyncAttempt,
    SyncMember,
    WeightSyncAttempt,
    WeightSyncFence,
    WeightSyncStage,
    wait_ray_stage,
    wait_replica_stage,
)

__all__ = [
    "AllServersFailed",
    "BatchMostlyFailed",
    "BuildGroupPartialFailure",
    "CheckPointPayLoad",
    "FaultToleranceConfig",
    "GCStats",
    "HeartbeatTracker",
    "LoadFailure",
    "LoadResult",
    "ModelVersionPolicy",
    "ProgressConfig",
    "ProgressContext",
    "RAY_FAULT_EXCEPTIONS",
    "ReplicaSnapshot",
    "ResumeOutcome",
    "ResumeResult",
    "RolloutProgressStoreActor",
    "ServerUnavailable",
    "StageFailure",
    "StageRef",
    "StageResult",
    "StaleWeightSyncAttempt",
    "SyncMember",
    "Supervisor",
    "ThreadedSupervisor",
    "VLLMProgressCheckPoint",
    "WeightSyncStage",
    "WeightSyncFence",
    "WeightSyncAttempt",
    "WeightSyncStageFailure",
    "filter_partial_batch",
    "is_transient_fault",
    "lenient_gather",
    "make_on_dead",
    "split_refs_by_timeout",
    "wait_ray_stage",
    "wait_replica_stage",
]
