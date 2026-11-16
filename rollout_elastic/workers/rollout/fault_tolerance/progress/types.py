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
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


@dataclass
class ProgressConfig:
    enabled: bool = False
    persist_root: str = "checkpoints/rollout_progress"
    flush_token_interval: int = 64
    gc_delay_s: float = 300.0
    gc_period_s: float = 60.0
    gc_retry_backoff_s: float = 30.0
    model_version_policy: ModelVersionPolicy = field(default_factory=lambda: ModelVersionPolicy(mode="exact"))
    write_timeout_s: float = 30.0
    max_pending_writes_per_recovery: int = 8
    stats_log_interval_s: float = 60.0


@dataclass
class ModelVersionPolicy:
    mode: str = "exact"

    def check(self, payload_version: Optional[str], request_version: Optional[str]) -> bool:
        if self.mode == "relaxed":
            return True
        if self.mode == "exact":
            return payload_version is not None and payload_version == request_version
        if self.mode == "compatible":
            if payload_version is None or request_version is None:
                return True
            return payload_version == request_version
        raise ValueError(f"Unknown model version policy mode: {self.mode}")


@dataclass
class CheckPointPayLoad:
    run_id: str
    recovery_id: str
    attempt_id: int
    prompt_token_ids: list[int]
    sampling_params: dict
    model_weight_version: Optional[str]
    inherited_prefix_len: int
    cumulative_token_ids: list[int]
    cumulative_log_probs: Optional[list[float]]
    cumulative_routed_experts: Optional[list[Any]]
    num_preempted: int
    finished: bool
    finished_reason: Optional[str]
    created_at: float
    updated_at: float
    schema_version: int = 1


@dataclass
class ProgressContext:
    checkpoint: Any


class ResumeOutcome(Enum):
    RESUMED = auto()
    DEGRADED_FRESH = auto()


@dataclass
class ResumeResult:
    checkpoint: Any
    outcome: ResumeOutcome
    attempt_id: int
    inherited_prefix_len: int
    failure_detail: Optional[str] = None


class LoadFailure(Enum):
    NOT_FOUND = auto()
    RUN_ID_MISMATCH = auto()
    ATTEMPT_STALE = auto()
    LENGTH_MISMATCH = auto()
    MODEL_VERSION_BLOCKED = auto()
    CORRUPTED = auto()


@dataclass
class LoadResult:
    payload: Optional[CheckPointPayLoad]
    failure: Optional[LoadFailure]
    detail: str
    prev_attempt_id: Optional[int] = None


@dataclass
class GCStats:
    collected_count: int = 0
    collected_bytes: int = 0
    failed_count: int = 0
    failed_paths: list[str] = field(default_factory=list)
    orphan_tmp_cleaned: int = 0
    last_collect_at: float = 0.0


__all__ = [
    "CheckPointPayLoad",
    "GCStats",
    "LoadFailure",
    "LoadResult",
    "ModelVersionPolicy",
    "ProgressConfig",
    "ProgressContext",
    "ResumeOutcome",
    "ResumeResult",
]
