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
"""Patch the SGLang / TRT-LLM rollout ``ServerAdapter`` constructors.

The elastic-rollout flow constructs rollout server adapters with an extra
``name_suffix`` keyword (used to give replacement replicas unique Ray actor
names). verl@dfc01f85 does not accept it yet, so we extend the two ``__init__``
signatures to swallow the argument.

The backends are optional: if a backend is not installed the corresponding
patch is skipped so installing the recipe never breaks a vLLM-only run.
"""

from __future__ import annotations

import logging

from ._core import wrap

logger = logging.getLogger(__file__)

try:
    from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter as _SGLangServerAdapter
except ImportError:
    _SGLangServerAdapter = None

try:
    from verl.workers.rollout.trtllm_rollout.trtllm_rollout import ServerAdapter as _TRTLLMServerAdapter
except ImportError:
    _TRTLLMServerAdapter = None


if _SGLangServerAdapter is not None:

    @wrap(_SGLangServerAdapter, "__init__")
    def sglang_server_adapter_init(
        orig, self, config, model_config, device_mesh, replica_rank: int = -1, name_suffix: str = ""
    ):
        """SGLang ServerAdapter accepting the recipe's ``name_suffix`` kwarg."""
        return orig(self, config, model_config, device_mesh, replica_rank=replica_rank)


if _TRTLLMServerAdapter is not None:

    @wrap(_TRTLLMServerAdapter, "__init__")
    def trtllm_server_adapter_init(
        orig, self, config, model_config, device_mesh, replica_rank: int = -1, name_suffix: str = ""
    ):
        """TRT-LLM ServerAdapter accepting the recipe's ``name_suffix`` kwarg."""
        return orig(self, config, model_config, device_mesh, replica_rank=replica_rank)
