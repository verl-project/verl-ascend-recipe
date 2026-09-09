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
"""Configuration types for fault tolerance.

Per spec §硬规则.可关性: `enabled=False` must yield bit-exact existing behavior.
All defaults are non-intrusive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.workers.rollout.fault_tolerance.progress.types import ProgressConfig


@dataclass
class FaultToleranceConfig:
    """Knobs for verl asynchronous rollout fault tolerance.

    Mounted under `config.async_training.fault_tolerance`. See spec §6 实现路径.
    """

    enabled: bool = False
    """Master switch. When False, all FT code paths are skipped."""

    # ----- L2/L3 request-level retry -----
    max_request_retries: int = 3
    """Max times L3 will switch server and retry a single prompt."""

    request_timeout_s: float = 600.0
    """Outer budget for the whole multi-server retry chain of one prompt."""

    server_call_timeout_s: float = 120.0
    """Per-server-call timeout in L2. Triggers retry on hang."""

    # ----- L4 batch-level partial return -----
    min_ok_ratio: float = 0.5
    """Min fraction of a batch that must succeed; below → BatchMostlyFailed."""

    # ----- Weight sync -----
    weight_sync_member_timeout_s: float = 60.0
    """Per-ref timeout in build_process_group; dropping dead members."""

    weight_sync_transfer_timeout_s: float = 600.0
    """Total timeout for transferring one complete model to rollout replicas."""

    max_weight_sync_retries: int = 2
    """Max N-1 replay attempts after a transient rollout-side sync failure."""

    # ----- P2 Supervisor -----
    heartbeat_interval_s: float = 5.0
    """Supervisor heartbeat period."""

    heartbeat_miss_threshold: int = 3
    """Consecutive misses before marking a replica dead."""

    replace_dead_replicas: bool = True
    """If True (standalone mode only), supervisor spawns replacement on dead."""

    # ----- Token continuation sub-config -----
    progress: ProgressConfig = field(default_factory=lambda: _default_progress_config())
    """Token 续推子配置. 当 ``progress.enabled=True`` 且 ``enabled=True`` 时启用."""


def _default_progress_config() -> ProgressConfig:
    """Late import to avoid circular dependency at module load time."""
    from verl.workers.rollout.fault_tolerance.progress.types import ProgressConfig

    return ProgressConfig()
