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
"""Orchestrate the rollout_elastic decorator patches.

``install()`` is the single entry point that turns a vanilla verl@dfc01f85
install into the elastic-rollout flavor:

1. Mount the recipe-owned ``fault_tolerance`` package into
   ``verl.workers.rollout.fault_tolerance``.
2. Import every area patch module. Each module carries ``@patch`` / ``@add`` /
   ``@wrap`` decorators that rewrite verl classes at import time, so merely
   importing the module is enough to apply the patch.

All area modules are imported lazily (``importlib``) so that the patches take
effect before the patched verl modules are used, and so importing this package
from within ``verl/__init__.py`` (via ``VERL_USE_EXTERNAL_MODULES``) does not
create import cycles.
"""

from __future__ import annotations

import importlib
import logging
from typing import Iterable

from ._mount import mount_fault_tolerance

logger = logging.getLogger(__file__)

# Area patch modules, in dependency order. Each module applies its decorators
# as a side effect of being imported.
_AREA_MODULES: tuple[str, ...] = (
    "replica",  # RolloutReplica.health / get_ray_class_with_init_args
    "attention",  # attention_utils._get_attention_functions fallback
    "backends",  # sglang / trtllm ServerAdapter
    "vllm",  # vllm_rollout / vllm_async_server / bucketed_weight_transfer / vllm_utils
    "llm_server",  # LLMServerClient / LLMServerManager / load balancer / FullyLLMServerClient
    "checkpoint",  # CheckpointEngineManager / CheckpointEngineWorker / hccl / nccl
    "experimental",  # fully_async / one_step_off / separation / agent_loop
)


def _apply_area(name: str) -> None:
    importlib.import_module(f"{__package__}.{name}")


def install(areas: Iterable[str] | None = None) -> None:
    """Apply all rollout_elastic patches to the running verl process.

    Args:
        areas: Optional subset of area names to apply (defaults to all).
    """
    mount_fault_tolerance()
    for name in _AREA_MODULES if areas is None else areas:
        try:
            _apply_area(name)
        except Exception:
            logger.exception("rollout_elastic: failed to apply patch area %r", name)
            raise
