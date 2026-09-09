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
"""rollout_elastic patch entry — loaded via ``VERL_USE_EXTERNAL_MODULES``.

Setting ``VERL_USE_EXTERNAL_MODULES=rollout_elastic.patch`` makes verl's
``__init__.py`` import this module (through ``import_external_libs``) at
startup. Importing this module applies every decorator patch, so no source
files of verl are modified on disk.
"""

from __future__ import annotations

from .install import install

__all__ = ["install"]

# Applying the patches is the whole point of this module: importing it must
# immediately patch verl, matching how verl triggers external modules.
install()
