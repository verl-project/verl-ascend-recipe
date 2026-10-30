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

"""Versioned positive-depth AdaptiveSpec entrypoint."""

from __future__ import annotations

import importlib
import logging
import os

from .compatibility import installed_adapter_name

logger = logging.getLogger(__name__)
_CONFIG_ENV = "VERL_ADAPTIVE_SPEC_CONFIG"


def adaptive_spec_plugin() -> bool:
    adapter, vllm_version, ascend_version = installed_adapter_name()
    kv_installed = False
    if os.environ.get("VERL_MTP_KV_PREEMPT_MODE"):
        if adapter != "v0_23":
            raise RuntimeError("KV mode requires a source-verified supported backend")
        kv_module = importlib.import_module(f"{__name__}.kv_{adapter}")
        kv_installed = bool(kv_module.install_from_environment())

    config_path = os.environ.get(_CONFIG_ENV)
    if not config_path:
        logger.info("%s is unset; adaptive-spec is disabled", _CONFIG_ENV)
        return kv_installed
    if adapter is None:
        logger.warning(
            "No adaptive-spec adapter for vLLM=%s and vLLM-Ascend=%s; preserving upstream path",
            vllm_version,
            ascend_version,
        )
        return False
    return bool(importlib.import_module(f"{__name__}.{adapter}").install(config_path)) or kv_installed
