# Copyright 2026
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

"""Pass the verl rollout-step identifier through vLLM SamplingParams."""

from __future__ import annotations

from typing import Any

import ray

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

from .step_scope import sampling_params_with_step


class AdaptiveSpecAgentLoopWorker(AgentLoopWorker):
    """Inject step scope while preserving the configured agent loop."""

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        **kwargs,
    ):
        scoped = sampling_params_with_step(sampling_params, trajectory["step"])
        return await super()._run_agent_loop(scoped, trajectory, **kwargs)


class AdaptiveSpecAgentLoopManager(AgentLoopManager):
    """Create agent-loop workers that attach rollout-step metadata."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(AdaptiveSpecAgentLoopWorker)
        super().__init__(*args, **kwargs)
