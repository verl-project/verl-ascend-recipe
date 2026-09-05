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
"""Cross-server retry client for elastic rollout.

``RetryLLMServerClient`` is a new class kept in the recipe (it does not exist in
verl@dfc01f85). It subclasses verl's native ``LLMServerClient``; the recipe's
``patch.llm_server`` decorators extend the base class with the FT helpers
(``_ft_enabled``, ``_generate_once``, ...) that this class relies on.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig, OmegaConf

from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.fault_tolerance.exceptions import is_transient_fault
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


class RetryLLMServerClient(LLMServerClient):
    """Cross-server retry for a single prompt, independent of fault tolerance.

    When a server dies mid-generation, the partial tokens produced so far are
    lost with the server. This client switches to a fresh server and regenerates
    the response from the original prompt, up to ``max_retries`` times before
    raising ``AllServersFailed``.
    """

    def __init__(
        self,
        config: DictConfig,
        servers: dict[str, Any],
        load_balancer_handle: Any,
        run_id: Optional[str] = None,
        progress_store: Optional[Any] = None,
        max_model_len: Optional[int] = None,
    ):
        super().__init__(
            config,
            servers,
            load_balancer_handle,
            run_id=run_id,
            progress_store=progress_store,
            max_model_len=max_model_len,
        )
        self.max_retries = self._ft_max_request_retries()
        self.call_timeout_s = self._ft_call_timeout_s()
        self._last_global_step: Optional[int] = None

    def _progress_enabled(self) -> bool:
        if not self._ft_enabled():
            return False
        if self._run_id is None or self._progress_store is None:
            return False
        try:
            return bool(OmegaConf.select(self.config, "async_training.fault_tolerance.progress.enabled", default=False))
        except (AttributeError, KeyError, TypeError):
            return False

    def _flush_token_interval(self) -> int:
        try:
            return int(
                OmegaConf.select(
                    self.config, "async_training.fault_tolerance.progress.flush_token_interval", default=64
                )
            )
        except (AttributeError, KeyError, TypeError):
            return 64

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

    def _resolve_original_max_tokens(self, sampling_params: dict, original_prompt: list[int]) -> int:
        if "max_tokens" in sampling_params:
            raw = sampling_params["max_tokens"]
        elif "max_new_tokens" in sampling_params:
            raw = sampling_params["max_new_tokens"]
        else:
            rollout_cfg = self.config.actor_rollout_ref.rollout
            raw = min(
                rollout_cfg.response_length,
                rollout_cfg.prompt_length + rollout_cfg.response_length - len(original_prompt),
            )
        raw = int(raw)
        if self.max_model_len is not None:
            raw = min(raw, self.max_model_len - len(original_prompt))
        return max(0, raw)

    async def _weights_version(self) -> Optional[str]:
        if self._last_global_step is not None:
            return str(self._last_global_step)
        for handle in list(self._server_id_to_handle.values()):
            try:
                gs = await asyncio.wait_for(handle.get_global_steps.remote(), timeout=5.0)
                if gs is not None:
                    self._last_global_step = int(gs)
                    return str(gs)
            except Exception:
                continue
        return None

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        from verl.workers.rollout.fault_tolerance import (
            AllServersFailed,
            ProgressContext,
            ServerUnavailable,
            VLLMProgressCheckPoint,
        )

        progress_on = self._progress_enabled()
        max_retries = self.max_retries

        original_prompt = normalize_token_ids(prompt_ids)
        original_sampling = copy.deepcopy(sampling_params)

        recovery_id = f"{request_id}:{uuid4().hex[:8]}"

        retries = 0
        while True:
            progress_ctx = None
            prefix_for_call = original_prompt
            call_sampling = original_sampling
            if progress_on:
                sp_for_checkpoint = dict(original_sampling)
                sp_for_checkpoint["max_tokens"] = self._resolve_original_max_tokens(original_sampling, original_prompt)
                try:
                    result = await VLLMProgressCheckPoint.create_or_resume(
                        store=self._progress_store,
                        run_id=self._run_id,
                        recovery_id=recovery_id,
                        prompt_token_ids=original_prompt,
                        sampling_params=sp_for_checkpoint,
                        model_weight_version=await self._weights_version(),
                        flush_token_interval=self._flush_token_interval(),
                        model_version_policy=self._model_version_policy(),
                    )
                except Exception as e:
                    if not is_transient_fault(e):
                        raise
                    print(
                        f"RetryLLMServerClient: progress store unavailable"
                        f"({type(e).__name__}), degrading to fresh attempt"
                        f"(run={self._run_id}, recovery_id={recovery_id})",
                        flush=True,
                    )
                else:
                    checkpoint = result.checkpoint
                    progress_ctx = ProgressContext(checkpoint=checkpoint)
                    prefix_for_call = checkpoint.resume_prefix_token_ids()
                    call_sampling = copy.deepcopy(original_sampling)
                    call_sampling["max_tokens"] = checkpoint.remaining_max_tokens()
                    logger.info(
                        "[progress] run=%s, rid=%s, attempt=%d, outcome=%s, inherited_len=%d"
                        "prefix_len=%d remaining_max_tokens=%d (%s)",
                        self._run_id,
                        recovery_id,
                        result.attempt_id,
                        result.outcome.name,
                        result.inherited_prefix_len,
                        len(prefix_for_call),
                        checkpoint.remaining_max_tokens(),
                        result.failure_detail or "-",
                    )
            try:
                output, server_id = await self._generate_once(
                    request_id,
                    prompt_ids=prefix_for_call,
                    sampling_params=call_sampling,
                    image_data=image_data,
                    video_data=video_data,
                    kwargs=dict(kwargs),
                    call_timeout_s=self.call_timeout_s,
                    translate_fault=True,
                    progress_ctx=progress_ctx,
                )
            except AllServersFailed:
                raise
            except ServerUnavailable as e:
                if e.server_id == "<lb>":
                    # The LB itself is unavailable; there is no healthy server to retry
                    raise
                retries += 1
                if retries > max_retries:
                    raise AllServersFailed(
                        f"RetryLLMServerClient: retries exhausted after {retries} attempts"
                    ) from None
                logger.warning(
                    "RetryLLMServerClient: server %s failed (%s), retries %d/%d",
                    e.server_id,
                    type(e.cause).__name__ if e.cause is not None else "server-fault",
                    retries,
                    max_retries,
                )
                continue

            gs = output.extra_fields.get("global_steps", None)
            if gs is not None:
                self._last_global_step = gs
            if progress_on and progress_ctx is not None:
                cp = progress_ctx.checkpoint
                inherited_ids = list(cp.cumulative_token_ids)
                new_ids = list(output.token_ids)
                lp_inherited = list(cp.cumulative_log_probs) if cp.cumulative_log_probs else []
                lp_new = list(output.log_probs) if output.log_probs else []
                re_inherited = cp.cumulative_routed_experts
                re_new = output.routed_experts
                if re_inherited is not None and re_new is not None:
                    routed_experts = torch.cat([re_inherited, re_new], dim=0)
                elif re_new is not None:
                    routed_experts = re_new
                else:
                    routed_experts = re_inherited
                final = TokenOutput(
                    token_ids=inherited_ids + new_ids,
                    log_probs=(lp_inherited + lp_new) or None,
                    routed_experts=routed_experts,
                    num_preempted=(cp.num_preempted or 0) + (output.num_preempted or 0),
                    stop_reason=output.stop_reason,
                    extra_fields=dict(output.extra_fields),
                )
            else:
                final = output
            output.extra_fields["llm_generate_attempts"] = retries + 1
            output.extra_fields["llm_generate_retries"] = retries
            return final
