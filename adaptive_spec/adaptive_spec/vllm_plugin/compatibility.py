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

"""Strict adapter selection for private vLLM-Ascend interfaces."""

from __future__ import annotations

from importlib import metadata


def adapter_name(vllm_version: str, ascend_version: str) -> str | None:
    if str(vllm_version).startswith("0.23.0") and str(ascend_version).startswith("0.23.0"):
        return "v0_23"
    return None


def installed_adapter_name() -> tuple[str | None, str, str]:
    try:
        vllm_version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        vllm_version = "missing"
    try:
        ascend_version = metadata.version("vllm-ascend")
    except metadata.PackageNotFoundError:
        try:
            ascend_version = metadata.version("vllm_ascend")
        except metadata.PackageNotFoundError:
            ascend_version = "missing"
    return adapter_name(vllm_version, ascend_version), vllm_version, ascend_version
