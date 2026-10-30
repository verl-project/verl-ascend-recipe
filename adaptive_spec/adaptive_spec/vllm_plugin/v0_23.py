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

"""Adapter for vLLM 0.23.0 and vLLM-Ascend 0.23.0.

This module patches three narrow model-runner methods. It never writes telemetry
and it refuses to change K when eager execution or required accepted-token
buffers are unavailable. Bookkeeping is observed so preempted or otherwise
invalid rows cannot be misclassified as draft-token rejections.
"""

from __future__ import annotations

import inspect
import logging
import sys
from functools import wraps
from importlib import machinery
from typing import Any

from ..config import AdaptiveSpecConfig
from ..controller import RequestContext

logger = logging.getLogger(__name__)
_PATCH_MARKER = "_verl_adaptive_spec_v0_23"
_STATE_ATTR = "_verl_adaptive_spec_state"
_PENDING_ATTR = "_verl_adaptive_spec_pending"
_CURRENT_ATTR = "_verl_adaptive_spec_current"
_INVALID_ATTR = "_verl_adaptive_spec_invalid_indices"
_REQUESTS_ATTR = "_verl_adaptive_spec_requests"
_LAST_DRAFT_K_ATTR = "_verl_adaptive_spec_last_draft_k"
_TARGET_MODULE = "vllm_ascend.worker.model_runner_v1"
_DEFERRED_CONFIG: AdaptiveSpecConfig | None = None
_DEFERRED_FINDER: Any = None


def _scalar(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _sampling_payload(params: Any) -> tuple[dict[str, Any], int]:
    names = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "ignore_eos",
        "stop_token_ids",
        "stop",
        "max_tokens",
        "min_tokens",
        "logit_bias",
        "allowed_token_ids",
    )
    payload = {name: getattr(params, name) for name in names if hasattr(params, name)}
    extra = getattr(params, "extra_args", None) or {}
    rollout_step = int(extra.get("adaptive_spec_rollout_step", 0))
    return (payload, rollout_step)


def _is_eager(runner: Any) -> bool:
    mode = getattr(getattr(runner, "compilation_config", None), "cudagraph_mode", None)
    return getattr(mode, "name", str(mode).split(".")[-1]) == "NONE"


def _apply_engine_k(runner: Any, selected_k: int, *, require_eager: bool) -> bool:
    from numbers import Integral

    drafter = getattr(runner, "drafter", None)
    if drafter is None or not hasattr(drafter, "num_speculative_tokens"):
        return False
    if isinstance(selected_k, bool) or not isinstance(selected_k, Integral):
        raise ValueError("Physical MTP K must be an integer")
    if not _is_eager(runner):
        raise RuntimeError("0.23 dynamic MTP requires eager execution")
    vllm_config = getattr(runner, "vllm_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    if spec is None or spec.method != "mtp" or getattr(drafter, "parallel_drafting", False):
        raise RuntimeError("0.23 dynamic depth requires a serial MTP drafter")
    if (
        parallel is None
        or parallel.pipeline_parallel_size != 1
        or getattr(runner, "pcp_size", 1) != 1
        or (getattr(runner, "dcp_size", 1) != 1)
    ):
        raise RuntimeError("0.23 dynamic MTP requires PP1/CP1")
    capacity = int(spec.num_speculative_tokens)
    kmax = getattr(runner, "_verl_adaptive_spec_kmax", capacity)
    if int(kmax) != capacity:
        raise RuntimeError("Startup MTP capacity changed after control began")
    selected_k = int(selected_k)
    if not 1 <= selected_k <= capacity or selected_k == 11:
        raise ValueError(f"Unsupported mainline MTP K={selected_k}, startup capacity={capacity}")
    runner._verl_adaptive_spec_kmax = capacity
    drafter.num_speculative_tokens = selected_k
    if hasattr(drafter, "decode_threshold"):
        drafter.decode_threshold = 1 + selected_k
    runner.num_spec_tokens = selected_k
    runner.uniform_decode_query_len = 1 + selected_k
    return True


def _logical_draft_count(raw_count: int, previous_k: int) -> int:
    return min(max(0, int(raw_count)), int(previous_k))


def _runtime_state(runner: Any, config: AdaptiveSpecConfig) -> dict[str, Any] | None:
    if hasattr(runner, _STATE_ATTR):
        return getattr(runner, _STATE_ATTR)
    if not config.enabled or config.phase_map_path is None:
        state = None
    else:
        group = _tensor_parallel_group(runner)
        leader = group is None or group.rank_in_group == 0
        state = {
            "controller": config.build_controller() if leader else None,
            "rollout_step": None,
            "tp_group": group,
            "decision_round": 0,
        }
    setattr(runner, _STATE_ATTR, state)
    setattr(runner, _REQUESTS_ATTR, {})
    return state


def _tensor_parallel_group(runner: Any) -> Any:
    parallel = runner.vllm_config.parallel_config
    size = int(parallel.tensor_parallel_size)
    if size == 1:
        return None
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size != size or not 0 <= group.rank_in_group < size:
        raise RuntimeError("AdaptiveSpec TP group does not match the Engine configuration")
    return group


def _select_engine_k(state: dict[str, Any], contexts: list[RequestContext]) -> int:
    """Use one authority per Engine, including filesystem evidence visibility."""
    group = state.get("tp_group")
    if group is None:
        return state["controller"].decide_engine(contexts).physical_k
    packet = None
    decision_round = state["decision_round"]
    if group.rank_in_group == 0:
        try:
            k = state["controller"].decide_engine(contexts).physical_k
            packet = (decision_round, int(k), None)
        except Exception as error:
            # All ranks must leave this collective before reporting a policy failure.
            packet = (decision_round, None, f"{type(error).__name__}: {error}")
    packet = group.broadcast_object(packet, src=0)
    if not isinstance(packet, tuple) or len(packet) != 3 or packet[0] != decision_round:
        raise RuntimeError("AdaptiveSpec received an out-of-order TP decision")
    if packet[2] is not None:
        raise RuntimeError(f"AdaptiveSpec TP leader decision failed: {packet[2]}")
    if type(packet[1]) is not int:
        raise RuntimeError("AdaptiveSpec TP leader did not broadcast an integer K")
    state["decision_round"] += 1
    return packet[1]


def _request_contexts(runner: Any) -> list[RequestContext]:
    request_ids = list(getattr(runner.input_batch, "req_ids", ()))
    lengths = getattr(runner.input_batch, "num_tokens_no_spec", ())
    contexts = []
    registry = getattr(runner, _REQUESTS_ATTR)
    for index, request_id in enumerate(request_ids):
        request_state = runner.requests.get(request_id)
        if request_state is None or request_state.prompt_token_ids is None:
            raise RuntimeError(f"adaptive-spec request state is unavailable: {request_id}")
        sampling_params, rollout_step = _sampling_payload(request_state.sampling_params)
        context = RequestContext(
            request_id=str(request_id),
            prompt_token_ids=tuple(int(value) for value in request_state.prompt_token_ids),
            sampling_params=sampling_params,
            context_length=_scalar(lengths[index]),
            rollout_step=rollout_step,
        )
        contexts.append(context)
        registry[context.request_id] = context
    return contexts


def _accepted_counts(runner: Any, count: int) -> list[int]:
    spec = getattr(runner, "speculative_config", None)
    if bool(getattr(spec, "disable_padded_drafter_batch", False)):
        raise RuntimeError("0.23 evidence requires the native padded-drafter count path")
    event = getattr(runner, "valid_sampled_token_count_event", None)
    buffer = getattr(runner, "valid_sampled_token_count_cpu", None)
    previous = getattr(runner.input_batch, "prev_sampled_token_ids", None)
    if event is None or buffer is None or previous is None:
        raise RuntimeError("adaptive-spec raw sampled-token count contract unavailable")
    if int(previous.shape[0]) != count:
        raise RuntimeError("adaptive-spec raw count/request snapshot length mismatch")
    event.synchronize()
    values = buffer[:count].tolist()
    if len(values) != count or any(int(value) < 0 for value in values):
        raise RuntimeError("adaptive-spec raw sampled-token counts invalid")
    return [int(value) for value in values]


def _finalize_previous(runner: Any, scheduler_output: Any, state: dict[str, Any]) -> None:
    controller = state["controller"]
    pending = getattr(runner, _PENDING_ATTR, None)
    if pending is not None:
        rows = pending["rows"]
        emitted_counts = _accepted_counts(runner, len(rows)) if controller is not None else ()
        for row, emitted in zip(rows, emitted_counts, strict=False):
            if row["discarded"]:
                continue
            if not 1 <= int(emitted) <= row["draft_tokens"] + 1:
                raise RuntimeError("Verification count exceeds bound of the physical draft")
            accepted = int(emitted) - 1
            if row["draft_tokens"]:
                controller.observe_round(
                    row["context"],
                    active_batch_size=pending["batch_size"],
                    active_k=row["draft_tokens"],
                    accepted_draft_tokens=accepted,
                )
        setattr(runner, _PENDING_ATTR, None)
    registry = getattr(runner, _REQUESTS_ATTR)
    for request_id in getattr(scheduler_output, "finished_req_ids", ()):
        context = registry.pop(str(request_id), None)
        if context is not None and controller is not None:
            controller.finish_request(context)


def _pending_snapshot(
    contexts: list[RequestContext], spec_decode_metadata: Any, previous_k: int | None
) -> dict[str, Any] | None:
    raw_counts = getattr(spec_decode_metadata, "num_draft_tokens", None)
    if raw_counts is None:
        return None
    raw = [_scalar(value) for value in raw_counts]
    if any(count > 0 for count in raw) and previous_k is None:
        raise RuntimeError("adaptive-spec draft evidence has no previous-K binding")
    if len(raw) < len(contexts):
        raw.extend([0] * (len(contexts) - len(raw)))
    return {
        "previous_k": previous_k,
        "rows": [
            {"context": context, "draft_tokens": _logical_draft_count(count, previous_k or 0), "discarded": False}
            for context, count in zip(contexts, raw, strict=False)
        ],
    }


def _install_pending(runner: Any, pending: dict[str, Any], invalid_indices: set[int]) -> None:
    if getattr(runner, _PENDING_ATTR, None) is not None:
        raise RuntimeError("adaptive-spec pending snapshot was not finalized")
    for index, row in enumerate(pending["rows"]):
        row["discarded"] = index in invalid_indices
    pending["batch_size"] = sum(row["draft_tokens"] > 0 and (not row["discarded"]) for row in pending["rows"])
    setattr(runner, _PENDING_ATTR, pending)


def _validate_contract(model_runner_cls: Any) -> None:
    required = {
        "execute_model": ("scheduler_output",),
        "_bookkeeping_sync": ("scheduler_output", "spec_decode_metadata"),
        "propose_draft_token_ids": ("scheduler_output", "spec_decode_metadata", "spec_decode_common_attn_metadata"),
    }
    for method_name, parameters in required.items():
        method = getattr(model_runner_cls, method_name, None)
        if method is None:
            raise RuntimeError(f"vLLM-Ascend contract is missing {method_name}")
        signature = inspect.signature(method)
        for parameter in parameters:
            if parameter not in signature.parameters:
                raise RuntimeError(f"vLLM-Ascend {method_name} is missing parameter {parameter}")


def _patch_class(model_runner_cls: Any, config: AdaptiveSpecConfig) -> bool:
    if getattr(model_runner_cls, _PATCH_MARKER, False):
        return True
    _validate_contract(model_runner_cls)
    from .mtp_copy_v0_23 import patch_copy_and_read

    patch_copy_and_read(model_runner_cls)
    from .async_reconcile_v0_23 import patch_class as patch_async_reconcile

    patch_async_reconcile(model_runner_cls)
    original_execute = model_runner_cls.execute_model
    original_bookkeeping = model_runner_cls._bookkeeping_sync
    original_propose = model_runner_cls.propose_draft_token_ids

    @wraps(original_execute)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        state = _runtime_state(self, config)
        if state is not None:
            _finalize_previous(self, scheduler_output, state)
            if getattr(self, _CURRENT_ATTR, None) is not None:
                raise RuntimeError("adaptive-spec current snapshot was not reconciled")
            setattr(self, _INVALID_ATTR, None)
        result = original_execute(self, scheduler_output, intermediate_tensors)
        if state is not None and getattr(self, _CURRENT_ATTR, None) is not None:
            raise RuntimeError("adaptive-spec bookkeeping did not publish current snapshot")
        return result

    @wraps(original_bookkeeping)
    def bookkeeping_sync(self, *args, **kwargs):
        result = original_bookkeeping(self, *args, **kwargs)
        invalid_indices = {int(index) for index in result[-1]}
        discarded = self.discard_request_indices.np[: self.num_discarded_requests]
        invalid_indices.update(int(index) for index in discarded)
        setattr(self, _INVALID_ATTR, invalid_indices)
        pending = getattr(self, _CURRENT_ATTR, None)
        if pending is not None:
            _install_pending(self, pending, invalid_indices)
            setattr(self, _CURRENT_ATTR, None)
        return result

    @wraps(original_propose)
    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids,
        sampling_metadata,
        scheduler_output,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        positions,
        num_scheduled_tokens,
        hidden_states,
        aux_hidden_states=None,
        sample_hidden_states=None,
        target_model_batch_desc=None,
    ):
        state = _runtime_state(self, config)
        can_control = (
            state is not None
            and (not config.require_eager or _is_eager(self))
            and hasattr(getattr(self, "drafter", None), "num_speculative_tokens")
        )
        contexts = _request_contexts(self) if can_control else []
        selected_k = None
        pending = None
        if contexts:
            pending = _pending_snapshot(contexts, spec_decode_metadata, getattr(self, _LAST_DRAFT_K_ATTR, None))
            steps = {context.rollout_step for context in contexts}
            if len(steps) != 1:
                raise RuntimeError("one Engine batch cannot mix adaptive-spec rollout steps")
            step = steps.pop()
            if state["rollout_step"] != step:
                if state["controller"] is not None:
                    state["controller"].reset_step()
                getattr(self, _REQUESTS_ATTR).clear()
                for context in contexts:
                    getattr(self, _REQUESTS_ATTR)[context.request_id] = context
                state["rollout_step"] = step
            physical_k = _select_engine_k(state, contexts)
            if _apply_engine_k(self, physical_k, require_eager=config.require_eager):
                selected_k = physical_k
        result = original_propose(
            self,
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            aux_hidden_states,
            sample_hidden_states,
            target_model_batch_desc,
        )
        if selected_k is not None:
            shape = getattr(result, "shape", None)
            if shape is None or len(shape) != 2 or int(shape[1]) != int(selected_k):
                raise RuntimeError("Physical MTP draft width differs from selected K")
            if int(shape[0]) != len(contexts):
                raise RuntimeError("Physical MTP draft rows differ from request snapshot")
        if selected_k is not None:
            setattr(self, _LAST_DRAFT_K_ATTR, int(selected_k))
            if pending is not None:
                invalid_indices = getattr(self, _INVALID_ATTR, None)
                if invalid_indices is None:
                    if getattr(self, _CURRENT_ATTR, None) is not None:
                        raise RuntimeError("adaptive-spec current snapshot was already staged")
                    setattr(self, _CURRENT_ATTR, pending)
                else:
                    _install_pending(self, pending, invalid_indices)
        return result

    model_runner_cls.execute_model = execute_model
    model_runner_cls._bookkeeping_sync = bookkeeping_sync
    model_runner_cls.propose_draft_token_ids = propose_draft_token_ids
    setattr(model_runner_cls, _PATCH_MARKER, True)
    logger.info("Installed adaptive-spec vLLM-Ascend 0.23 adapter")
    return True


class _DeferredPatchLoader:
    def __init__(self, loader: Any, finder: _DeferredPatchFinder) -> None:
        self._loader = loader
        self._finder = finder

    def create_module(self, spec: Any) -> Any:
        create_module = getattr(self._loader, "create_module", None)
        return None if create_module is None else create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._loader.exec_module(module)
        model_runner_cls = getattr(module, "NPUModelRunner", None)
        if model_runner_cls is None:
            self._finder.remove()
            raise RuntimeError("vLLM-Ascend target module has no NPUModelRunner")
        try:
            _patch_class(model_runner_cls, self._finder.config)
        finally:
            self._finder.remove()


class _DeferredPatchFinder:
    def __init__(self, config: AdaptiveSpecConfig) -> None:
        self.config = config

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        del target
        if fullname != _TARGET_MODULE:
            return None
        spec = machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _DeferredPatchLoader(spec.loader, self)
        return spec

    def remove(self) -> None:
        global _DEFERRED_CONFIG, _DEFERRED_FINDER
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        if _DEFERRED_FINDER is self:
            _DEFERRED_CONFIG = None
            _DEFERRED_FINDER = None


def install(config_path: str) -> bool:
    """Install now when possible, otherwise patch after the runner import.

    vLLM loads general plugins while its own package import is still active.
    Importing the Ascend runner from that callback creates a circular import, so
    this function registers a one-shot loader wrapper instead.
    """
    global _DEFERRED_CONFIG, _DEFERRED_FINDER
    config = AdaptiveSpecConfig.from_path(config_path)
    if config.build_controller() is None:
        logger.info("adaptive-spec config is disabled or has no phase map; preserving upstream path")
        return False
    target_module = sys.modules.get(_TARGET_MODULE)
    if target_module is not None:
        model_runner_cls = getattr(target_module, "NPUModelRunner", None)
        if model_runner_cls is None:
            raise RuntimeError("vLLM-Ascend runner module is only partially initialized")
        return _patch_class(model_runner_cls, config)
    if _DEFERRED_FINDER is not None:
        if _DEFERRED_CONFIG != config:
            raise RuntimeError("adaptive-spec deferred install received conflicting configs")
        return True
    _DEFERRED_CONFIG = config
    _DEFERRED_FINDER = _DeferredPatchFinder(config)
    sys.meta_path.insert(0, _DEFERRED_FINDER)
    logger.info("Registered deferred adaptive-spec vLLM-Ascend 0.23 adapter")
    return True
