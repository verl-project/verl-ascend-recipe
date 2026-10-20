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

"""Pinned 0.23 MTP worker-local trim and optimistic-count restoration."""

from copy import copy
from functools import wraps

from .mtp_copy_v0_23 import _COPY_SHAS, _guard


def prepare_worker_schedule(runner, schedule):
    if not runner.use_async_scheduling:
        return schedule, {}
    drafts = getattr(runner, "_draft_token_ids", None)
    if drafts is None:
        return schedule, {}
    if not hasattr(drafts, "shape") or len(drafts.shape) != 2:
        raise RuntimeError("Expected physical tensor drafts")
    rows, width = map(int, drafts.shape)
    if width < 1 or int(runner.num_spec_tokens) != width:
        raise RuntimeError("Previous physical draft stride mismatch")
    cached = schedule.scheduled_cached_reqs
    excluded = (
        set(getattr(cached, "resumed_req_ids", ()))
        | set(getattr(schedule, "preempted_req_ids", ()) or ())
        | set(getattr(schedule, "finished_req_ids", ()))
    )
    restore = {}
    edits = []
    for request_id in cached.req_ids:
        index = runner.input_batch.req_id_to_index.get(request_id)
        tokens = schedule.scheduled_spec_decode_tokens.get(request_id)
        if index is None or tokens is None or request_id in excluded:
            continue
        if not 0 <= index < rows:
            raise RuntimeError("Request index outside preceding draft batch")
        allocated = len(tokens)
        count = min(allocated, width)
        if count == allocated:
            continue
        removed = allocated - count
        if schedule.num_scheduled_tokens[request_id] <= removed:
            raise RuntimeError("Trim would remove non-speculative work")
        restore[request_id] = allocated
        edits.append((request_id, tokens, count, removed))
    if not edits:
        return schedule, restore
    worker = copy(schedule)
    worker.num_scheduled_tokens = dict(schedule.num_scheduled_tokens)
    worker.scheduled_spec_decode_tokens = dict(schedule.scheduled_spec_decode_tokens)
    for request_id, tokens, count, removed in edits:
        worker.total_num_scheduled_tokens -= removed
        worker.num_scheduled_tokens[request_id] -= removed
        worker.scheduled_spec_decode_tokens[request_id] = tokens[:count]
    return worker, restore


_MARKER = "_verl_mtp_async_reconcile_v023"
_STATE = "_verl_mtp_async_restore_v023"


def wrap_methods(original_execute, original_update):
    @wraps(original_execute)
    def execute(self, scheduler_output, intermediate_tensors=None):
        if getattr(self, _STATE, None) is not None:
            raise RuntimeError("Overlapping MTP worker state reconciliation")
        worker, restore = prepare_worker_schedule(self, scheduler_output)
        if restore:
            spec = self.speculative_config
            mode = getattr(self.compilation_config, "cudagraph_mode", None)
            if spec is None or spec.method != "mtp" or getattr(mode, "name", str(mode).split(".")[-1]) != "NONE":
                raise RuntimeError("0.23 dynamic reconciliation requires eager MTP")
            if (
                getattr(self, "pcp_size", 1) != 1
                or getattr(self, "dcp_size", 1) != 1
                or self.vllm_config.parallel_config.pipeline_parallel_size != 1
            ):
                raise RuntimeError("0.23 dynamic reconciliation requires PP1/CP1")
        state = {"restore": restore, "updated": False}
        setattr(self, _STATE, state)
        try:
            return original_execute(self, worker, intermediate_tensors)
        finally:
            setattr(self, _STATE, None)

    @wraps(original_update)
    def update(self, scheduler_output):
        state = getattr(self, _STATE, None)
        if state is not None and state["updated"]:
            raise RuntimeError("Repeated MTP state update within one execute call")
        result = original_update(self, scheduler_output)
        if state is not None:
            for request_id, count in state["restore"].items():
                request = self.requests.get(request_id)
                if request is None:
                    raise RuntimeError("Trimmed request disappeared during state update")
                request.prev_num_draft_len = count
            state["updated"] = True
        return result

    setattr(execute, _MARKER, True)
    setattr(update, _MARKER, True)
    return execute, update


def patch_class(cls):
    execute, update = cls.execute_model, cls._update_states
    marked = (getattr(execute, _MARKER, False), getattr(update, _MARKER, False))
    if any(marked):
        if not all(marked):
            raise RuntimeError("Partial 0.23 async reconciliation installation")
        return False
    _guard(execute, _COPY_SHAS)
    _guard(update, _COPY_SHAS)
    cls.execute_model, cls._update_states = wrap_methods(execute, update)
    return True
