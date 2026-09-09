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
"""Patch ``verl.workers.rollout.replica.RolloutReplica``.

- ``get_ray_class_with_init_args``: forward the replica's ``name_suffix`` to the
  ``CheckpointEngineWorker`` actor so replacement replicas get a unique prefix.
- ``health``: new method used by the fault-tolerance supervisor to probe every
  rollout server actor of a replica.
"""

from __future__ import annotations

import asyncio

from verl.single_controller.ray.base import RayClassWithInitArgs
from verl.workers.rollout.replica import RolloutReplica

from ._core import add, wrap


@wrap(RolloutReplica, "get_ray_class_with_init_args")
def get_ray_class_with_init_args(orig, self) -> RayClassWithInitArgs:
    """Get rollout worker actor class, propagating the replica name suffix."""
    ret = orig(self)
    ret.kwargs["name_suffix"] = self.name_suffix
    return ret


@add(RolloutReplica, "health")
async def health(self) -> bool:
    """Return whether all rollout server actors in this replica respond."""
    if not self.servers:
        return False
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[server.health.remote() for server in self.servers]), timeout=2.0
        )
    except Exception:
        return False
    return all(bool(result) for result in results)
