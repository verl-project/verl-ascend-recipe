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
"""Patch ``verl.workers.rollout.llm_server`` for elastic inference.

This area extends the four native classes of ``llm_server.py``:

- ``GlobalRequestLoadBalancer`` (Ray actor): rewritten as a thin forwarder over
  the recipe-owned ``_LoadBalancerCore`` state machine (kept in
  ``fault_tolerance.load_balancer``), which adds fault tolerance
  (``mark_failed``, dead-set routing) and real ``add_servers`` /
  ``remove_servers`` for elastic scaling.
- ``LLMServerClient``: accepts ``run_id`` / ``progress_store`` / ``max_model_len``,
  and gains the single-attempt primitive ``_generate_once`` with transient-fault
  translation (``ServerUnavailable``) and ``_mark_server_failed``.
- ``FullyLLMServerClient``: its ``generate`` loop becomes fault-tolerant
  (retry on a fresh server) and token-continuation aware (mode C).
- ``LLMServerManager``: tracks a process-level ``run_id``, can assemble the
  ``RolloutProgressStoreActor``, and can spawn replacement replicas
  (``spawn_replacement`` / ``_reclaim_ray_resources``).

``_LoadBalancerCore`` and ``RetryLLMServerClient`` are brand-new classes and are
kept as recipe files under ``fault_tolerance``; everything else is expressed as
decorators against the native classes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.llm_server import (
    DEFAULT_ROUTING_CACHE_SIZE,
    FullyLLMServerClient,
    GlobalRequestLoadBalancer,
    LLMServerClient,
    LLMServerManager,
)
from verl.workers.rollout.replica import RolloutReplica, TokenOutput

from ._core import add, patch, wrap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GlobalRequestLoadBalancer (Ray actor) — thin forwarder over _LoadBalancerCore
# ---------------------------------------------------------------------------

_ORIG_LB = "_rollout_elastic_lb_core"


@wrap(GlobalRequestLoadBalancer, "__init__")
def _lb_init(
    orig,
    self,
    servers: dict[str, ray.actor.ActorHandle],
    max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
    enable_fault_tolerance: bool = False,
):
    """Initialize the LB around the recipe-owned ``_LoadBalancerCore``."""
    from verl.workers.rollout.fault_tolerance.load_balancer import _LoadBalancerCore

    orig(self, servers, max_cache_size=max_cache_size)
    self._core = _LoadBalancerCore(
        servers=servers,
        max_cache_size=max_cache_size,
        enable_fault_tolerance=enable_fault_tolerance,
    )
    setattr(self, _ORIG_LB, True)


def _lb_core(self):
    core = getattr(self, _ORIG_LB, None)
    if core is None:
        # Safety net for actors constructed before the patch was installed.
        from verl.workers.rollout.fault_tolerance.load_balancer import _LoadBalancerCore

        core = _LoadBalancerCore(servers=self._server)
        setattr(self, _ORIG_LB, core)
    return core


@patch(GlobalRequestLoadBalancer, "acquire_server")
def acquire_server(self, request_id: str) -> str:
    """Acquire a server for the given request, skipping dead servers."""
    return _lb_core(self).acquire_server(request_id)


@patch(GlobalRequestLoadBalancer, "release_server")
def release_server(self, server_id: str) -> None:
    """Release a server after a request completes, decrementing its inflight count."""
    _lb_core(self).release_server(server_id)


@add(GlobalRequestLoadBalancer, "mark_failed")
def mark_failed(self, server_id: str) -> None:
    """Mark a server as dead; subsequent acquires skip it. Idempotent."""
    _lb_core(self).mark_failed(server_id)


@add(GlobalRequestLoadBalancer, "set_fault_tolerance")
def set_fault_tolerance(self, enabled: bool) -> None:
    """Enable or disable fault tolerance for the load balancer."""
    _lb_core(self)._ft = bool(enabled)


@patch(GlobalRequestLoadBalancer, "add_servers")
def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
    """Add new servers to the server handles. Idempotent; resurrects dead ids."""
    _lb_core(self).add_servers(servers)


@patch(GlobalRequestLoadBalancer, "remove_servers")
def remove_servers(self, server_ids: list[str]) -> None:
    """Remove servers from the server handles."""
    _lb_core(self).remove_servers(server_ids)


@add(GlobalRequestLoadBalancer, "get_server_handle")
def get_server_handle(self, server_id: str):
    """Return the Ray actor handle for ``server_id``, or None if unknown."""
    return _lb_core(self).get_server_handle(server_id)


# ---------------------------------------------------------------------------
# LLMServerClient — FT helpers + single-attempt primitive
# ---------------------------------------------------------------------------


@wrap(LLMServerClient, "__init__")
def _client_init(
    orig,
    self,
    config: DictConfig,
    servers: dict[str, ray.actor.ActorHandle],
    load_balancer_handle: ray.actor.ActorHandle,
    run_id: Optional[str] = None,
    progress_store: Optional[ray.actor.ActorHandle] = None,
    max_model_len: Optional[int] = None,
):
    """LLMServerClient accepting the recipe's FT/progress kwargs."""
    orig(self, config, servers, load_balancer_handle)
    self._run_id = run_id
    self._progress_store = progress_store
    self.max_model_len = max_model_len


@add(LLMServerClient, "_ft_enabled")
def _ft_enabled(self) -> bool:
    """Whether the fault-tolerance master switch is on for this config."""
    try:
        return bool(self.config.async_training.fault_tolerance.enabled)
    except (AttributeError, KeyError):
        return False


@add(LLMServerClient, "_ft_call_timeout_s")
def _ft_call_timeout_s(self) -> float:
    """Per-call timeout used to bound a single server generate call."""
    try:
        return float(self.config.async_training.fault_tolerance.server_call_timeout_s)
    except (AttributeError, KeyError):
        return 120.0


@add(LLMServerClient, "_ft_max_request_retries")
def _ft_max_request_retries(self) -> int:
    """Max retries across servers for a single logical request."""
    try:
        return int(self.config.async_training.fault_tolerance.max_request_retries)
    except (AttributeError, KeyError):
        return 3


@add(LLMServerClient, "_call_server")
async def _call_server(
    self,
    server_id: str,
    server: ray.actor.ActorHandle,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]],
    video_data: Optional[list[Any]],
    kwargs: dict[str, Any],
    call_timeout_s: Optional[float],
) -> TokenOutput:
    """Issue a single ``server.generate.remote`` call, optionally bounded by a timeout."""
    call = server.generate.remote(
        request_id=uuid4().hex,
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        image_data=image_data,
        video_data=video_data,
        **kwargs,
    )
    if call_timeout_s is not None:
        return await asyncio.wait_for(call, timeout=call_timeout_s)
    return await call


@add(LLMServerClient, "_generate_once")
async def _generate_once(
    self,
    request_id: str,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]],
    video_data: Optional[list[Any]],
    kwargs: dict[str, Any],
    call_timeout_s: Optional[float],
    translate_fault: bool,
    progress_ctx: Optional[Any] = None,
) -> tuple[TokenOutput, str]:
    """Single acquire -> call -> release on one server.

    Returns ``(TokenOutput, server_id)``. When ``translate_fault`` is True,
    transient faults are translated to ``ServerUnavailable``; otherwise
    exceptions propagate as-is (bit-exact with the pre-FT path).
    """
    from verl.workers.rollout.fault_tolerance import AllServersFailed, ServerUnavailable, is_transient_fault

    try:
        server_id, server = await self._acquire_server(request_id)
    except AllServersFailed:
        raise
    except Exception as e:
        if translate_fault and is_transient_fault(e):
            raise ServerUnavailable("<lb>", cause=e) from e
        raise

    try:
        try:
            if progress_ctx is not None:
                kwargs = {**kwargs, "progress_ctx": progress_ctx}
            output = await self._call_server(
                server_id,
                server,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                kwargs=kwargs,
                call_timeout_s=call_timeout_s,
            )
            return output, server_id
        except Exception as e:
            if translate_fault and is_transient_fault(e):
                await self._mark_server_failed(server_id)
                raise ServerUnavailable(server_id, cause=e) from e
            raise
    finally:
        self._release_server(server_id)


@patch(LLMServerClient, "_acquire_server")
async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
    server_id = await self._load_balancer.acquire_server.remote(request_id=request_id)
    handle = self._server_id_to_handle.get(server_id)
    if handle is None:
        # Spawn-back replica not yet in the local cache; lazy-fetch from LB.
        handle = await self._load_balancer.get_server_handle.remote(server_id=server_id)
        if handle is None:
            raise RuntimeError(f"Unknown server_id returned by load balancer: {server_id}")
        self._server_id_to_handle[server_id] = handle
    return server_id, handle


@patch(LLMServerClient, "_release_server")
def _release_server(self, server_id: str) -> None:
    # Fire-and-forget: release is just a counter decrement, no need to await.
    # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
    try:
        self._load_balancer.release_server.remote(server_id=server_id)
    except Exception:
        # R10: LB itself may be unhealthy; don't let release leak into the caller path.
        pass


@add(LLMServerClient, "_mark_server_failed")
async def _mark_server_failed(self, server_id: str) -> None:
    """Fire-and-forget notify LB that a server is dead. Must never block."""
    try:
        await asyncio.wait_for(self._load_balancer.mark_failed.remote(server_id=server_id), timeout=3.0)
    except Exception:
        logger.warning("[FT] _mark_server_failed: mark server %s failed", server_id)


@patch(LLMServerClient, "generate")
@rollout_trace_op
async def generate(
    self,
    request_id,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
    progress_ctx: Optional[Any] = None,
    **kwargs: Any,
) -> TokenOutput:
    """Generate tokens from prompt ids.

    Args:
        request_id (str): request id for sticky session.
        prompt_ids (List[int]): List of prompt token ids.
        sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
        progress_ctx (Optional[Any]): Token-continuation context (ProgressContext)
            forwarded to the server's ``generate`` for in-stream ``ingest``.
            None when FT/progress is off (mode A/B).

    Returns:
        TokenOutput: token output

    Raises:
        ServerUnavailable: (FT-only) the chosen server died / hung. Caller (L3) should retry.
        AllServersFailed: (FT-only) the LB has no live servers to hand out.
    """
    ft_on = self._ft_enabled()
    output, _ = await self._generate_once(
        request_id,
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        image_data=image_data,
        video_data=video_data,
        progress_ctx=progress_ctx,
        kwargs=kwargs,
        call_timeout_s=self._ft_call_timeout_s() if ft_on else None,
        translate_fault=ft_on,
    )
    output.extra_fields["llm_generate_attempts"] = 1
    return output


# ---------------------------------------------------------------------------
# FullyLLMServerClient — FT + token continuation in the resume loop
# ---------------------------------------------------------------------------


@add(FullyLLMServerClient, "_ft_max_request_retries")
def _fully_ft_max_request_retries(self) -> int:
    try:
        return int(self.config.async_training.fault_tolerance.max_request_retries)
    except (AttributeError, KeyError):
        return 3


@add(FullyLLMServerClient, "_progress_enabled")
def _progress_enabled(self) -> bool:
    """True only when FT master switch AND progress sub-switch are both on."""
    if not self._ft_enabled():
        return False
    if self._progress_store is None or self._run_id is None:
        return False
    try:
        return bool(OmegaConf.select(self.config, "async_training.fault_tolerance.progress.enabled", default=False))
    except (AttributeError, KeyError, TypeError):
        return False


@add(FullyLLMServerClient, "_flush_token_interval")
def _flush_token_interval(self) -> int:
    try:
        return int(
            OmegaConf.select(
                self.config,
                "async_training.fault_tolerance.progress.flush_token_interval",
                default=64,
            )
        )
    except (AttributeError, KeyError, TypeError):
        return 64


@add(FullyLLMServerClient, "_model_version_policy")
def _model_version_policy(self):
    from verl.workers.rollout.fault_tolerance import ModelVersionPolicy

    try:
        node = OmegaConf.select(self.config, "async_training.fault_tolerance.progress.model_version_policy")
        if node is not None:
            mode = node.get("mode", "exact") if hasattr(node, "get") else "exact"
            return ModelVersionPolicy(mode=mode)
    except (AttributeError, KeyError, TypeError):
        pass
    return ModelVersionPolicy(mode="exact")


@patch(FullyLLMServerClient, "generate")
@rollout_trace_op
async def _fully_generate(
    self,
    request_id,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
) -> TokenOutput:
    """Generate tokens from prompt ids with FT retry and token continuation.

    The while-loop keeps the native abort path (partial tokens kept, resume from
    ``prompt + token_ids``) and adds a fault path: ``ServerUnavailable`` resets to
    the original prompt and retries on a fresh server. With mode C the prefix is
    decided by the persisted checkpoint (``create_or_resume``).
    """
    import copy

    from verl.workers.rollout.fault_tolerance import AllServersFailed, ServerUnavailable

    ft_on = self._ft_enabled()
    progress_on = self._progress_enabled()
    max_retries = self._ft_max_request_retries()

    original_prompt = normalize_token_ids(prompt_ids)
    original_sampling = copy.deepcopy(sampling_params)

    limit_key = None
    if "max_tokens" in sampling_params:
        limit_key = "max_tokens"
    elif "max_new_tokens" in sampling_params:
        limit_key = "max_new_tokens"
    original_max_tokens = sampling_params.get(limit_key) if limit_key else None

    final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)
    min_global_steps, max_global_steps, global_steps = None, None, None
    retries = 0
    total_llm_generate_attempts = 0
    current_request_id = request_id
    recovery_id = request_id
    model_weight_version = None

    while True:
        # ----- decide prefix + progress_ctx for this attempt -----
        checkpoint = None
        progress_ctx = None
        if progress_on:
            from verl.workers.rollout.fault_tolerance import (
                ProgressContext,
                VLLMProgressCheckPoint,
                is_transient_fault,
            )

            try:
                result = await VLLMProgressCheckPoint.create_or_resume(
                    store=self._progress_store,
                    run_id=self._run_id,
                    recovery_id=recovery_id,
                    prompt_token_ids=original_prompt,
                    sampling_params=original_sampling,
                    model_weight_version=model_weight_version,
                    flush_token_interval=self._flush_token_interval(),
                    model_version_policy=self._model_version_policy(),
                )
            except Exception as e:
                if not is_transient_fault(e):
                    raise
                print(
                    f"FullyLLMServerClient: progress store unavailable"
                    f"({type(e).__name__}), degrading to fresh attempt"
                    f"(run={self._run_id}, recovery_id={recovery_id})",
                    flush=True,
                )
                prefix_for_call = original_prompt
                call_sampling = sampling_params
            else:
                checkpoint = result.checkpoint
                progress_ctx = ProgressContext(checkpoint=checkpoint)
                prefix_for_call = checkpoint.resume_prefix_token_ids()
                call_sampling = copy.deepcopy(original_sampling)
                if limit_key is not None:
                    call_sampling[limit_key] = checkpoint.remaining_max_tokens()
                # Seed final_output with inherited cumulative so that after fault reset
                # (where final_output was cleared) the persisted tokens are not lost.
                # On the abort path this is idempotent (cumulative == existing final_output).
                if checkpoint.inherited_prefix_len > 0:
                    final_output = TokenOutput(
                        token_ids=list(checkpoint.cumulative_token_ids),
                        log_probs=list(checkpoint.cumulative_log_probs) if checkpoint.cumulative_log_probs else [],
                        routed_experts=checkpoint.cumulative_routed_experts,
                        num_preempted=checkpoint.num_preempted,
                    )
        else:
            progress_ctx = None
            prefix_for_call = original_prompt + final_output.token_ids
            call_sampling = sampling_params

        # 1. generate tokens — catch ServerUnavailable to retry on a fresh server
        try:
            output, _ = await self._generate_once(
                request_id=current_request_id,
                prompt_ids=prefix_for_call,
                sampling_params=call_sampling,
                image_data=image_data,
                video_data=video_data,
                kwargs={},
                call_timeout_s=self._ft_call_timeout_s() if ft_on else None,
                translate_fault=ft_on,
                progress_ctx=progress_ctx,
            )
            total_llm_generate_attempts += 1
        except ServerUnavailable as e:
            if not ft_on or e.server_id == "<lb>":
                raise
            retries += 1
            total_llm_generate_attempts += 1
            if retries >= max_retries:
                raise AllServersFailed(f"FullyLLMServerClient: retries exhausted after {retries} attempts") from None
            final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)
            sampling_params = copy.deepcopy(original_sampling)
            min_global_steps, max_global_steps, global_steps = None, None, None
            current_request_id = uuid4().hex
            model_weight_version = None
            continue

        # 2. merge output into final_output
        if progress_on and checkpoint is not None:
            final_output.token_ids = list(checkpoint.cumulative_token_ids) + list(output.token_ids)
            if checkpoint.cumulative_log_probs is not None:
                final_output.log_probs = list(checkpoint.cumulative_log_probs) + list(output.log_probs or [])
            else:
                final_output.log_probs = list(output.log_probs or [])
            if checkpoint.cumulative_routed_experts is not None:
                if output.routed_experts is not None:
                    final_output.routed_experts = torch.cat(
                        [checkpoint.cumulative_routed_experts, output.routed_experts], dim=0
                    )
                else:
                    final_output.routed_experts = checkpoint.cumulative_routed_experts
            else:
                final_output.routed_experts = output.routed_experts
            final_output.num_preempted = (checkpoint.num_preempted or 0) + (output.num_preempted or 0)
            final_output.stop_reason = output.stop_reason
        else:
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

        global_steps = output.extra_fields.get("global_steps", None)
        if min_global_steps is None:
            min_global_steps = global_steps
        max_global_steps = global_steps
        model_weight_version = global_steps

        # 3. update max_new_tokens; truncate (FT only) to the original budget
        if original_max_tokens is not None and limit_key is not None:
            if not progress_on:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
            if len(final_output.token_ids) >= original_max_tokens:
                if ft_on and len(final_output.token_ids) > original_max_tokens:
                    final_output.token_ids = final_output.token_ids[:original_max_tokens]
                    if final_output.log_probs:
                        final_output.log_probs = final_output.log_probs[:original_max_tokens]
                final_output.stop_reason = "length"
                break

        # 4. check stop reason
        partial_rollout_enabled = False
        try:
            partial_rollout_enabled = bool(self.config.async_training.partial_rollout)
        except (AttributeError, KeyError):
            pass
        if output.stop_reason not in ("aborted", "abort") or not partial_rollout_enabled:
            break

    final_output.extra_fields["global_steps"] = global_steps
    final_output.extra_fields["min_global_steps"] = min_global_steps
    final_output.extra_fields["max_global_steps"] = max_global_steps
    final_output.extra_fields["llm_generate_attempts"] = total_llm_generate_attempts
    final_output.extra_fields["llm_generate_retries"] = retries
    return final_output


# ---------------------------------------------------------------------------
# LLMServerManager — run_id / progress store / replacement spawning
# ---------------------------------------------------------------------------


@wrap(LLMServerManager, "__init__")
def _manager_init(orig, self, config, worker_group=None, rollout_resource_pool=None):
    """LLMServerManager tracking a process-level run_id for token continuation."""
    orig(self, config, worker_group=worker_group, rollout_resource_pool=rollout_resource_pool)
    self.run_id = f"{socket.gethostname()}_{os.getpid()}_{uuid4().hex[:8]}"
    self._progress_store = None
    self.max_model_len = self._resolve_max_model_len()


@add(LLMServerManager, "_resolve_max_model_len")
def _resolve_max_model_len(self) -> Optional[int]:
    """Best-effort context length for the client's token-continuation budget."""
    try:
        mml = self.rollout_config.get("max_model_length", None)
        if mml is not None:
            return int(mml)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        mpe = getattr(self.model_config, "max_position_embeddings", None)
        if mpe is not None:
            return int(mpe)
    except (AttributeError, TypeError, ValueError):
        pass
    return None


@add(LLMServerManager, "_init_progress_store")
async def _init_progress_store(self, progress_cfg) -> None:
    """Mode C: create and initialise the StoreActor. Called during FT assembly."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from verl.workers.rollout.fault_tolerance import RolloutProgressStoreActor

    try:
        node_id = ray.get_runtime_context().get_node_id()
        scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=True)
    except Exception:
        scheduling_strategy = None
    options = {"max_restarts": 3, "max_task_retries": 3}
    if scheduling_strategy is not None:
        options["scheduling_strategy"] = scheduling_strategy
    self._progress_store = RolloutProgressStoreActor.options(**options).remote()
    await self._progress_store.init.remote(progress_cfg)


@patch(LLMServerManager, "_init_global_load_balancer")
async def _init_global_load_balancer(self) -> None:
    ft_on = False
    try:
        ft_on = bool(self.config.async_training.fault_tolerance.enabled)
    except (AttributeError, KeyError):
        pass
    self.global_load_balancer = GlobalRequestLoadBalancer.remote(
        servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
    )
    if ft_on:
        await self.global_load_balancer.set_fault_tolerance.remote(True)


@patch(LLMServerManager, "get_client")
def get_client(self, fully_async: bool = False, retry: bool = False) -> LLMServerClient:
    """Get the LLMServerClient to request LLM server replicas.

    Args:
        fully_async (bool): Whether to return the FullyLLMServerClient.
        retry (bool): Whether to retry on server unavailability.
    """
    servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
    common = dict(
        config=self.config,
        servers=servers,
        load_balancer_handle=self.global_load_balancer,
        run_id=self.run_id,
        progress_store=self._progress_store,
        max_model_len=self.max_model_len,
    )
    if fully_async:
        return FullyLLMServerClient(**common)
    if retry:
        from verl.workers.rollout.fault_tolerance.retry_client import RetryLLMServerClient

        return RetryLLMServerClient(**common)
    return LLMServerClient(config=self.config, servers=servers, load_balancer_handle=self.global_load_balancer)


@add(LLMServerManager, "spawn_replacement")
async def spawn_replacement(self, dead_id: str) -> RolloutReplica:
    """Spawn a replacement RolloutReplica for ``dead_id`` (standalone mode only).

    Caller (on_spawn_success) wires the new replica into CKE / LB / Supervisor.
    """
    log = logging.getLogger(__name__)
    if self.worker_group is not None:
        raise RuntimeError("spawn_replacement is standalone-mode only")

    if dead_id not in self.server_addresses:
        raise ValueError(f"unknown dead_id={dead_id!r}; have {self.server_addresses}")
    dead_rank = self.server_addresses.index(dead_id)
    log.warning("[FT] spawn_replacement: dead_id=%s rank=%s — beginning respawn", dead_id, dead_rank)

    self.rollout_replicas = [r for r in self.rollout_replicas if r._server_address != dead_id]
    kept = {r._server_address for r in self.rollout_replicas}
    self.server_addresses = [a for a in self.server_addresses if a in kept]
    self.server_handles = [r._server_handle for r in self.rollout_replicas]

    await self._reclaim_ray_resources(dead_rank, log)

    # A killed named Ray actor can remain in Ray's name registry briefly.
    # Do not reuse the dead replica's fixed name prefix: on multi-node replicas
    # that window can otherwise surface as ActorAlreadyExistsError while
    # creating a worker on a non-zero local rank.
    recovery_suffix = f"recovery_{uuid4().hex}"
    new_replica = self.rollout_replica_class(
        replica_rank=dead_rank,
        config=self.rollout_config,
        model_config=self.model_config,
        gpus_per_node=self.rollout_config.n_gpus_per_node,
        name_suffix=recovery_suffix,
    )
    await new_replica.init_standalone()
    if not bool(await asyncio.wait_for(new_replica.health(), timeout=30.0)):
        raise RuntimeError(f"new replica for {dead_id} (rank={dead_rank}) failed health check")

    self.rollout_replicas.append(new_replica)
    self.server_addresses.append(new_replica._server_address)
    self.server_handles.append(new_replica._server_handle)
    log.warning(
        "[FT] spawn_replacement: new replica %s (rank=%s) up and healthy",
        new_replica._server_address,
        dead_rank,
    )
    return new_replica


@add(LLMServerManager, "_reclaim_ray_resources")
async def _reclaim_ray_resources(self, dead_rank: int, log: logging.Logger) -> None:
    """Kill stale actors, remove their placement groups, then await a full replica's resources."""
    from ray.util.placement_group import get_placement_group, placement_group_table, remove_placement_group

    from verl.utils.device import get_resource_name

    nnodes = int(self.rollout_config.nnodes)
    local_world_size = int(self.rollout_config.n_gpus_per_node)
    candidate_names = (
        [
            f"rollout_standalone_{dead_rank}{cls}{pg_idx}:{local_rank}"
            for cls in ("CheckpointEngineWorker", "vLLMHttpServer")
            for pg_idx in range(nnodes)
            for local_rank in range(local_world_size)
        ]
        + [f"vllm_server_{dead_rank}_{i}" for i in range(nnodes)]
        + [f"rollout_standalone_{dead_rank}"]
    )
    for name in candidate_names:
        try:
            handle = ray.get_actor(name)
        except ValueError:
            continue
        ray.kill(handle, no_restart=True)
        log.warning("[FT] spawn_replacement: ray.kill stale actor %s", name)

    pg_prefix = f"rollout_pool_{dead_rank}"
    for pg_id, info in placement_group_table().items():
        name = info.get("name", "")
        if not name.startswith(pg_prefix):
            continue
        try:
            pg = get_placement_group(name)
        except ValueError:
            continue
        log.warning("[FT] spawn_replacement: removing leftover placement_group %s", name)
        remove_placement_group(pg)

    resource_name = get_resource_name()
    rollout_world_size = (
        self.rollout_config.tensor_model_parallel_size
        * self.rollout_config.data_parallel_size
        * self.rollout_config.pipeline_model_parallel_size
    ) // nnodes
    required_accelerators = rollout_world_size
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 180.0
    last_seen = -1.0
    while True:
        accelerator_avail = float(ray.available_resources().get(resource_name, 0.0))
        if accelerator_avail >= required_accelerators:
            log.warning(
                "[FT] spawn_replacement: %.1f/%d %s resources free; proceeding to init",
                accelerator_avail,
                required_accelerators,
                resource_name,
            )
            return
        if loop.time() > deadline:
            raise RuntimeError(
                "dead replica resources not freed after 180s "
                f"(available {resource_name}={accelerator_avail}, required={required_accelerators})"
            )
        if abs(accelerator_avail - last_seen) > 0.01:
            log.warning(
                "[FT] spawn_replacement: waiting for %d %s resources, available=%.2f",
                required_accelerators,
                resource_name,
                accelerator_avail,
            )
            last_seen = accelerator_avail
        await asyncio.sleep(2)
