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
"""Mount the recipe-owned ``fault_tolerance`` package into verl's namespace.

``rollout_elastic`` ships a brand-new ``workers/rollout/fault_tolerance``
package (verl@dfc01f85 does not have one). The patched verl code references it
as ``verl.workers.rollout.fault_tolerance``, so before applying any decorator
patch we alias the recipe package into that name inside ``sys.modules``.

Because the package is registered with its real ``__path__``, the package's own
absolute imports (``from verl.workers.rollout.fault_tolerance.aio import ...``)
and its subpackages (``...fault_tolerance.progress``) resolve against the
recipe directory without copying anything into the verl tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_VERL_FT_NAME = "verl.workers.rollout.fault_tolerance"


def mount_fault_tolerance() -> None:
    """Alias the recipe fault_tolerance package into ``verl.workers.rollout``."""
    if _VERL_FT_NAME in sys.modules:
        return

    root = Path(__file__).resolve().parent.parent / "workers" / "rollout" / "fault_tolerance"
    spec = importlib.util.spec_from_file_location(
        _VERL_FT_NAME, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    assert spec is not None and spec.loader is not None, f"cannot create spec for {_VERL_FT_NAME}"
    package = importlib.util.module_from_spec(spec)
    sys.modules[_VERL_FT_NAME] = package
    spec.loader.exec_module(package)
