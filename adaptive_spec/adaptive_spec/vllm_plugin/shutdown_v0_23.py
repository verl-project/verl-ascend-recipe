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

"""Allow Ascend workers to finish resource teardown before termination."""

import sys
import time
from functools import wraps
from importlib.machinery import PathFinder

_TARGET = "vllm.v1.executor.multiproc_executor"
_MARKER = "_adaptive_spec_cleanup_grace"
_finder = None


def patch_executor(cls):
    original = cls._ensure_worker_termination
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def terminate(workers):
        deadline = time.monotonic() + 30
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        return original(workers)

    setattr(terminate, _MARKER, True)
    cls._ensure_worker_termination = staticmethod(terminate)


class _Loader:
    def __init__(self, loader, finder):
        self.loader, self.finder = loader, finder

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        try:
            self.loader.exec_module(module)
            patch_executor(module.MultiprocExecutor)
        finally:
            if self.finder in sys.meta_path:
                sys.meta_path.remove(self.finder)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, self)
        return spec


def install():
    global _finder
    module = sys.modules.get(_TARGET)
    if module is not None:
        patch_executor(module.MultiprocExecutor)
    elif _finder is None:
        _finder = _Finder()
        sys.meta_path.insert(0, _finder)
