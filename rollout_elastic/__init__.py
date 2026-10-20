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
"""Elastic-rollout recipe for verl (training-inference separation with FT).

rollout_elastic extends a vanilla ``verl`` install entirely through Python
decorators that rewrite verl's native classes at import time — no verl source
file is modified on disk and no ``.patch`` file is shipped.

To activate the extension, make this repository importable (e.g. add it to
``PYTHONPATH``) and point verl at the patch entry::

    export VERL_USE_EXTERNAL_MODULES=rollout_elastic.patch

verl's ``__init__`` then imports ``rollout_elastic.patch``, whose side effect is
calling :func:`rollout_elastic.patch.install` to mount the recipe-owned
``fault_tolerance`` package into ``verl.workers.rollout`` and apply every
decorator patch. See ``rollout_elastic/README.md`` for details.
"""

from __future__ import annotations

from .patch import install

__all__ = ["install"]
