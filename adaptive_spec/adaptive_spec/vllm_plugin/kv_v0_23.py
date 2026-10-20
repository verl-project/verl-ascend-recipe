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

"""Optional source-guarded targeted KV preemption."""

import ast
import hashlib
import inspect
import os
from functools import wraps
from pathlib import Path

from .kv_source_transform import patch_scheduler_source

BALANCE_SHA256 = "d6a77d35afc1a981edd4423409cb3bd0b8c6e3527392f0d43d9452ab27c9389c"
SOURCE_SHA256 = "9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9"


def patch_scheduler_class(cls, mode):
    if mode not in ("original", "targeted"):
        raise ValueError("KV mode must be original or targeted")
    previous = cls.__dict__.get("_verl_kv_mode_v023")
    if previous is not None:
        if previous != mode:
            raise RuntimeError("KV mode cannot change inside a live engine")
        return
    original = cls.schedule
    source_path = Path(inspect.getsourcefile(original))
    source = source_path.read_text()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_hash == BALANCE_SHA256:
        if cls.__name__ != "BalanceScheduler" or len(cls.__bases__) != 1:
            raise RuntimeError("Unexpected BalanceScheduler inheritance")
        patch_scheduler_class(cls.__bases__[0], mode)

        @wraps(original)
        def balance_disabled(self, *args, **kwargs):
            if self._balance_enabled:
                raise RuntimeError("KV adapter has not validated enabled DP balance scheduling")
            return original(self, *args, **kwargs)

        cls.schedule = balance_disabled
        cls._verl_kv_mode_v023 = mode
        return
    if source_hash != SOURCE_SHA256:
        raise RuntimeError("KV adapter refuses unaudited Scheduler source")
    implementation = original
    if mode == "targeted":
        patched = patch_scheduler_source(source)
        klass = next(n for n in ast.parse(patched).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        node = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
        if node.decorator_list:
            raise RuntimeError("Unexpected decorated Scheduler")
        body = [ast.parse("from __future__ import annotations").body[0], node]
        local = {}
        exec(
            compile(ast.Module(body=body, type_ignores=[]), "<verl-recipe-kv-targeted-v023>", "exec"),
            original.__globals__,
            local,
        )
        implementation = wraps(original)(local["schedule"])
    cls.schedule = implementation
    cls._verl_kv_mode_v023 = mode


def install_from_environment():
    mode = os.environ.get("VERL_MTP_KV_PREEMPT_MODE")
    if not mode:
        return False
    from .compatibility import installed_adapter_name

    if installed_adapter_name()[0] != "v0_23":
        raise RuntimeError("KV adapter requires verified v0.23 stack")
    from vllm.v1.core.sched.scheduler import Scheduler

    patch_scheduler_class(Scheduler, mode)
    return True
