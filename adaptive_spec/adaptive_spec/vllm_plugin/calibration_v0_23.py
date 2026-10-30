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

"""Offline-only observer for the source-guarded synchronous 0.23 runner.

No depth control is installed. Sampling returns raw valid token lists, so Mamba
state-advance counters and delayed next-step attribution are not involved.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from functools import wraps
from importlib import machinery
from pathlib import Path
from typing import Any

from .mtp_copy_v0_23 import _COPY_SHAS, _guard
from .v0_23 import _is_eager

_TARGET = "vllm_ascend.worker.model_runner_v1"
_MARKER = "_verl_adaptive_spec_calibration"
_FINDER = None


def step_record(plan: dict, snapshot: dict, latency_ns: int, counters: dict) -> dict | None:
    """Validate a pure decode snapshot and exclude warmup, partial drafts and drain."""
    state_index, repeat = snapshot["state_index"], snapshot["repeat"]
    if state_index is None or repeat is None:
        return None
    state = plan["states"][state_index]
    b, k = state["batch_size"], plan["k"]
    if snapshot["discarded"] or len(snapshot["draft_counts"]) != b or any(d != k for d in snapshot["draft_counts"]):
        return None
    if snapshot["scheduled_tokens"] != b * (k + 1):
        return None
    emitted, lengths = snapshot["emitted_counts"], snapshot["context_lengths"]
    if len(emitted) != b or len(lengths) != b or any(not 1 <= count <= k + 1 for count in emitted):
        raise RuntimeError("calibration raw verification counts do not match physical K")
    if any(
        not state["context_length"] <= length <= state["context_length"] + plan["length_window_tokens"]
        for length in lengths
    ):
        return None
    key = (state_index, repeat)
    counters[key] = counters.get(key, 0) + 1
    if counters[key] <= plan["warmup_steps"]:
        return None
    return {
        "clean": True,
        "state_index": state_index,
        "repeat": repeat,
        "k": k,
        "batch_size": b,
        "context_lengths": lengths,
        "accepted_draft_tokens": [count - 1 for count in emitted],
        "iteration_latency_ns": latency_ns,
    }


def patch_class(cls: Any, plan: dict, output: Path) -> None:
    if getattr(cls, _MARKER, False):
        if getattr(cls, _MARKER) != (plan, str(output)):
            raise RuntimeError("conflicting offline calibration installs")
        return
    if getattr(cls, "_verl_adaptive_spec_v0_23", False):
        raise RuntimeError("offline calibration cannot coexist with online depth control")
    _guard(cls.execute_model, _COPY_SHAS)
    _guard(cls._bookkeeping_sync, _COPY_SHAS)
    _guard(cls.sample_tokens, _COPY_SHAS)
    original_execute, original_bookkeeping = cls.execute_model, cls._bookkeeping_sync
    original_sample = cls.sample_tokens
    signature = inspect.signature(original_bookkeeping)

    @wraps(original_bookkeeping)
    def bookkeeping(self, *args, **kwargs):
        arguments = signature.bind(self, *args, **kwargs).arguments
        metadata = arguments.get("spec_decode_metadata")
        ids = list(self.input_batch.req_ids)
        lengths = [int(v) for v in self.input_batch.num_tokens_no_spec[: len(ids)]]
        tags = []
        for identity in ids:
            extra = self.requests[identity].sampling_params.extra_args or {}
            tags.append((extra.get("adaptive_spec_calibration_state"), extra.get("adaptive_spec_calibration_repeat")))
        result = original_bookkeeping(self, *args, **kwargs)
        state_index, repeat = tags[0] if tags and len(set(tags)) == 1 else (None, None)
        discarded = list(result[-1]) + list(self.discard_request_indices.np[: self.num_discarded_requests])
        self._verl_calibration_snapshot = {
            "state_index": state_index,
            "repeat": repeat,
            "context_lengths": lengths,
            "draft_counts": [] if metadata is None else list(metadata.num_draft_tokens),
            "emitted_counts": [len(tokens) for tokens in result[1]],
            "discarded": bool(discarded),
            "scheduled_tokens": arguments["num_scheduled_tokens"],
        }
        return result

    @wraps(original_execute)
    def execute(self, scheduler_output, intermediate_tensors=None):
        if self.use_async_scheduling or not _is_eager(self):
            raise RuntimeError("offline measurement requires eager synchronous execution")
        if int(self.num_spec_tokens) != plan["k"] or self.speculative_config.method != "mtp":
            raise RuntimeError("offline calibration requires a constant physical serial MTP K")
        if getattr(self.drafter, "parallel_drafting", False) or self.pcp_size != 1 or self.dcp_size != 1:
            raise RuntimeError("offline calibration requires serial MTP and CP1")
        if getattr(scheduler_output, "preempted_req_ids", ()):
            raise RuntimeError("KV preemption invalidates this calibration probe; preserve it and reduce the grid")
        # Keep this intentionally synchronized timing strictly outside production jobs.
        torch = inspect.unwrap(original_execute).__globals__["torch"]
        self._verl_calibration_snapshot = None
        self._verl_calibration_execute_ns = None
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        result = original_execute(self, scheduler_output, intermediate_tensors)
        torch.npu.synchronize()
        self._verl_calibration_execute_ns = time.perf_counter_ns() - start
        return result

    @wraps(original_sample)
    def sample(self, *args, **kwargs):
        # 0.23 splits target forward and sampling/drafting into separate calls.
        # Measuring execute_model alone would omit drafting and all acceptance evidence.
        torch = inspect.unwrap(original_sample).__globals__["torch"]
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        result = original_sample(self, *args, **kwargs)
        torch.npu.synchronize()
        sample_ns = time.perf_counter_ns() - start
        snapshot = self._verl_calibration_snapshot
        execute_ns = self._verl_calibration_execute_ns
        self._verl_calibration_execute_ns = None
        if snapshot is None:
            return result
        if execute_ns is None:
            raise RuntimeError("calibration sampling has no corresponding target forward")
        latency = execute_ns + sample_ns
        rank = int(self.vllm_config.parallel_config.rank)
        if snapshot is not None and rank == 0:
            if not hasattr(self, "_verl_calibration_counters"):
                self._verl_calibration_counters = {}
            record = step_record(plan, snapshot, latency, self._verl_calibration_counters)
            if record is not None:
                # Serialization/filesystem time is outside the measured interval.
                with (output / f"steps.{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        return result

    cls._bookkeeping_sync, cls.execute_model, cls.sample_tokens = bookkeeping, execute, sample
    setattr(cls, _MARKER, (plan, str(output)))


class _Loader:
    def __init__(self, loader, finder):
        self.loader, self.finder = loader, finder

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        try:
            patch_class(module.NPUModelRunner, self.finder.plan, self.finder.output)
        finally:
            sys.meta_path.remove(self.finder)


class _Finder:
    def __init__(self, plan, output):
        self.plan, self.output = plan, output

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, self)
        return spec


def calibration_plugin() -> bool:
    global _FINDER
    directory = os.environ.get("VERL_ADAPTIVE_SPEC_CALIBRATION_DIR")
    if not directory:
        return False
    if os.environ.get("VERL_ADAPTIVE_SPEC_CONFIG"):
        raise RuntimeError("offline calibration and online control are mutually exclusive")
    from .compatibility import installed_adapter_name

    if installed_adapter_name()[0] != "v0_23":
        raise RuntimeError("offline observer requires the supported 0.23 backend")
    plan = json.loads(os.environ["VERL_ADAPTIVE_SPEC_CALIBRATION_PLAN"])
    output = Path(directory)
    if not output.is_dir():
        raise ValueError("calibration output directory must already exist")
    module = sys.modules.get(_TARGET)
    if module is not None:
        patch_class(module.NPUModelRunner, plan, output)
    elif _FINDER is None:
        _FINDER = _Finder(plan, output)
        sys.meta_path.insert(0, _FINDER)
    elif (_FINDER.plan, _FINDER.output) != (plan, output):
        raise RuntimeError("conflicting offline calibration configuration")
    return True
