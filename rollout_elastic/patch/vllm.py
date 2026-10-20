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
"""Patch the vLLM rollout stack for elastic inference.

Covers ``verl.workers.rollout.vllm_rollout.{vllm_rollout,vllm_async_server,
bucketed_weight_transfer,utils}``:

- ``ServerAdapter``: accept a replica ``name_suffix`` so replacement replicas get
  unique Ray actor names, and carry the weight-sync transaction context
  (``attempt_id`` / ``target_version``) across the IPC weight update.
- ``vLLMHttpServer``: server-side monotonic weight-sync fence (``begin_weight_sync``
  / ``set_global_steps``), health probes, and token-continuation support in
  ``generate`` via a ``progress_ctx``.
- ``BucketedWeightSender`` / ``BucketedWeightReceiver``: attach the transaction
  context to every bucket envelope and reject stale attempts.
- ``vLLMColocateWorkerExtension``: process-local ``WeightSyncFence`` so a late
  bucket from an old attempt is never applied to the model.

The whole stack is vLLM-only: if vLLM is not installed the area is skipped so a
vLLM-free run (sglang / trtllm / naive) still installs cleanly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_torch_device
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.fault_tolerance.weight_sync import StaleWeightSyncAttempt, WeightSyncFence
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import ensure_async_iterator, qwen2_5_vl_dedup_image_tokens

from ._core import add, patch, wrap

logger = logging.getLogger(__file__)

try:
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
        BucketedWeightReceiver,
        BucketedWeightSender,
        rebuild_ipc,
    )
except ImportError:
    BucketedWeightReceiver = None  # type: ignore[assignment,misc]
    BucketedWeightSender = None  # type: ignore[assignment,misc]
    rebuild_ipc = None  # type: ignore[assignment,misc]

try:
    from verl.workers.rollout.vllm_rollout.utils import (
        VLLM_LORA_INT_ID,
        VLLM_LORA_NAME,
        VLLM_LORA_PATH,
        extract_prompt_logprobs,
        is_fp8_model,
        patch_vllm_moe_model_weight_loader,
        vLLMColocateWorkerExtension,
    )
except ImportError:
    VLLM_LORA_INT_ID = VLLM_LORA_NAME = VLLM_LORA_PATH = None  # type: ignore[assignment]
    extract_prompt_logprobs = is_fp8_model = None  # type: ignore[assignment]
    patch_vllm_moe_model_weight_loader = None  # type: ignore[assignment]
    vLLMColocateWorkerExtension = None  # type: ignore[assignment]

try:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
except ImportError:
    vLLMHttpServer = None  # type: ignore[assignment,misc]

try:
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
except ImportError:
    ServerAdapter = None  # type: ignore[assignment,misc]


if ServerAdapter is not None:
    # ------------------------------------------------------------------
    # vllm_rollout.ServerAdapter
    # ------------------------------------------------------------------

    @wrap(ServerAdapter, "__init__")
    def __init__(orig, self, config, model_config, device_mesh, replica_rank: int = -1, name_suffix: str = ""):
        """ServerAdapter accepting the recipe's ``name_suffix`` kwarg."""
        orig(self, config, model_config, device_mesh, replica_rank=replica_rank)
        self.name_suffix = name_suffix

    @patch(ServerAdapter, "_execute_method")
    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.
        """
        if self.rollout_rank != 0:
            return None

        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            actor_name = f"{prefix}server_{self.replica_rank}_{self.node_rank}{self.name_suffix}"
            self.server_handle = ray.get_actor(actor_name)

        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        return future if non_block else await future

    @patch(ServerAdapter, "update_weights")
    @torch.no_grad()
    async def update_weights(self, weights, global_steps: int = None, **kwargs):
        """Update model weights via CUDA IPC (fallback to shared memory if IPC not supported) to inference workers."""
        start_time = time.time()
        attempt_id = kwargs.get("attempt_id")
        target_version = kwargs.get("target_version")
        if target_version is None:
            target_version = global_steps

        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
            attempt_id=attempt_id,
            target_version=target_version,
        )
        await sender.async_send_weights(weights)

        if future is not None:
            await future

        # reset prefix cache after updating weights
        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()
            if attempt_id is None and target_version is not None:
                # The FT path commits the visible version from the Manager only
                # after every checkpoint-engine member has finalized.
                await self.server_handle.set_global_steps.remote(target_version)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(f"update_weights done, time cost: {time.time() - start_time:.2f}s")


if vLLMHttpServer is not None:
    # ------------------------------------------------------------------
    # vllm_async_server.vLLMHttpServer
    # ------------------------------------------------------------------

    @wrap(vLLMHttpServer, "__init__")
    def __init__(
        orig,
        self,
        config,
        model_config,
        rollout_mode,
        workers,
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        """vLLMHttpServer with a server-side weight-sync version fence."""
        orig(
            self,
            config,
            model_config,
            rollout_mode,
            workers,
            replica_rank,
            node_rank,
            gpus_per_node,
            nnodes,
            cuda_visible_devices,
        )
        # Server-side version fence prevents a late old adapter from changing
        # the visible model version after a newer sync attempt has started.
        self._weight_sync_attempt_id: int | None = None
        self._weight_sync_target_version: int | None = None

    @add(vLLMHttpServer, "health")
    async def health(self) -> bool:
        """Health probe for the fault-tolerance supervisor."""
        return True

    @add(vLLMHttpServer, "get_global_steps")
    async def get_global_steps(self):
        """Return the currently committed model weights version."""
        return self.global_steps

    @patch(vLLMHttpServer, "generate")
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data=None,
        video_data=None,
        priority: int = 0,
        progress_ctx=None,
    ):
        """Generate sequence with token-in-token-out.

        Args:
            progress_ctx (Optional[Any]): Token-continuation context (ProgressContext).
                When not None, each streaming ``RequestOutput`` is fed to
                ``checkpoint.ingest`` which accumulates tokens and triggers
                fire-and-forget flushes. None in mode A/B (no checkpoint overhead).
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.lora.request import LoRARequest

        prompt_ids = normalize_token_ids(prompt_ids)

        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            # Cap max_tokens by response_length to ensure tensor alignment,
            # and by remaining budget to prevent OOM in multi-turn rollouts.
            max_tokens = min(
                self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids)
            )

        # Clamp max_tokens to the valid range [0, max_possible_tokens]
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # Get final response
        final_res = None
        async for output in generator:
            final_res = output
            if progress_ctx is not None:
                progress_ctx.checkpoint.ingest(output)
        assert final_res is not None

        # Final flush: ensure terminal fields (finished/finish_reason) are persisted.
        # ``ingest`` already force-flushes on finished=True, but guard against the
        # edge case where the last chunk did not set finished before the loop exited.
        if progress_ctx is not None:
            progress_ctx.checkpoint.ingest(final_res)
            progress_ctx.checkpoint._maybe_flush(force=True)

        extra_fields = {"global_steps": self.global_steps}
        extract_prompt_logprobs(
            output=final_res,
            num_prompt_logprobs=sampling_params.prompt_logprobs,
            result_dict=extra_fields,
        )
        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        routed_experts = None
        num_preempted = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts
        if hasattr(final_res.outputs[0], "num_preempted"):
            num_preempted = final_res.outputs[0].num_preempted

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    @add(vLLMHttpServer, "begin_weight_sync")
    async def begin_weight_sync(self, attempt_id: int, target_version: int | None = None) -> bool:
        """Open a monotonic sync context for this server actor."""
        if self._weight_sync_attempt_id is not None and attempt_id < self._weight_sync_attempt_id:
            return False
        if self._weight_sync_attempt_id == attempt_id and self._weight_sync_target_version != target_version:
            return False
        self._weight_sync_attempt_id = attempt_id
        self._weight_sync_target_version = target_version
        return True

    @patch(vLLMHttpServer, "set_global_steps")
    async def set_global_steps(self, global_steps: int, attempt_id: int | None = None):
        """Set the global steps of the model weights."""
        if attempt_id is not None and (
            attempt_id != self._weight_sync_attempt_id or global_steps != self._weight_sync_target_version
        ):
            raise StaleWeightSyncAttempt(attempt_id, global_steps)
        self.global_steps = global_steps


if BucketedWeightSender is not None:
    # ------------------------------------------------------------------
    # bucketed_weight_transfer.BucketedWeightSender
    # ------------------------------------------------------------------

    @wrap(BucketedWeightSender, "__init__")
    def __init__(
        orig,
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
        attempt_id=None,
        target_version=None,
    ):
        """BucketedWeightSender carrying the weight-sync transaction context."""
        orig(self, zmq_handle, bucket_size_mb=bucket_size_mb, use_shm=use_shm)
        self.attempt_id = attempt_id
        self.target_version = target_version

    @patch(BucketedWeightSender, "async_send_weights")
    async def async_send_weights(self, weights):
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, Any] = {}
            async for name, weight in ensure_async_iterator(weights):
                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    get_torch_device().synchronize()
                    self._send_bucket(bucket_meta, is_last=False)
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                        f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                    )
                    self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes

            # send the last bucket
            get_torch_device().synchronize()
            self._send_bucket(bucket_meta, is_last=True)
        finally:
            self._cleanup()

    @patch(BucketedWeightSender, "_direct_send_large_weight")
    def _direct_send_large_weight(self, name: str, weight: torch.Tensor):
        """Send a weight larger than the bucket size via cuda ipc or share memory."""
        logger.debug(f"Direct sending large weight {name}({weight.shape}, {weight.dtype})")
        # TODO: support fallback to shared memory
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, Any] = {}
        bucket_meta[name] = {
            "name": name,
            "shape": weight.shape,
            "dtype": weight.dtype,
            "offset": 0,
            "handle": handle,
        }
        self._send_bucket(bucket_meta, is_last=False)

    @add(BucketedWeightSender, "_send_bucket")
    def _send_bucket(self, bucket_meta: dict[str, Any], *, is_last: bool) -> None:
        """Attach the transaction context to every bucket envelope."""
        self.socket.send_pyobj(
            {
                "attempt_id": self.attempt_id,
                "target_version": self.target_version,
                "bucket_meta": bucket_meta,
                "is_last": is_last,
            }
        )
        self.socket.recv()


if BucketedWeightReceiver is not None:
    # ------------------------------------------------------------------
    # bucketed_weight_transfer.BucketedWeightReceiver
    # ------------------------------------------------------------------

    @patch(BucketedWeightReceiver, "receive_weights")
    def receive_weights(
        self,
        on_bucket_received: Callable[[list[tuple[str, torch.Tensor]]], Any],
        *,
        attempt_id: int | None = None,
        target_version: int | None = None,
        attempt_validator: Callable[[int | None, int | None], bool] | None = None,
    ):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        stale_attempt = False
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                context_matches = attempt_id is None or (
                    metadata.get("attempt_id") == attempt_id and metadata.get("target_version") == target_version
                )
                if attempt_validator is not None:
                    context_matches = context_matches and attempt_validator(attempt_id, target_version)
                if not context_matches:
                    stale_attempt = True

                weights, tensor = [], None
                if not stale_attempt:
                    for name, meta in metadata["bucket_meta"].items():
                        shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                        if handle is not None:
                            tensor = rebuild_ipc(handle, self.device.index)
                            weights.append((name, tensor))
                            continue
                        size = dtype.itemsize * shape.numel()
                        tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                        if self.use_shm:
                            tensor = tensor.to(self.device)
                        weights.append((name, tensor))
                    applied = on_bucket_received(weights)
                    if applied is False:
                        stale_attempt = True
                    if not stale_attempt:
                        get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
            if stale_attempt:
                raise StaleWeightSyncAttempt(attempt_id, target_version)
        finally:
            self._cleanup()


if vLLMColocateWorkerExtension is not None:
    # ------------------------------------------------------------------
    # vllm_rollout.utils.vLLMColocateWorkerExtension
    # ------------------------------------------------------------------

    @add(vLLMColocateWorkerExtension, "_get_weight_sync_fence")
    def _get_weight_sync_fence(self):
        """Lazily create the process-local fence shared by bucket callbacks."""
        if not hasattr(self, "_weight_sync_fence"):
            self._weight_sync_fence = WeightSyncFence()
        return self._weight_sync_fence

    @patch(vLLMColocateWorkerExtension, "update_weights_from_ipc")
    def update_weights_from_ipc(
        self,
        peft_config: dict = None,
        base_sync_done=False,
        use_shm: bool = False,
        attempt_id: int | None = None,
        target_version: int | None = None,
    ):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        fence = None
        if attempt_id is not None:
            fence = self._get_weight_sync_fence()
            if not fence.begin(attempt_id, target_version):
                raise StaleWeightSyncAttempt(attempt_id, target_version)

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            prepare_qat_for_load_weights(self.model_runner.model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            patch_vllm_moe_model_weight_loader(self.model_runner.model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )

        def on_bucket_received(weights):
            if fence is None:
                self._update_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done)
                return None
            return fence.apply_if_current(
                attempt_id,
                target_version,
                lambda: self._update_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done),
            )

        receiver.receive_weights(
            on_bucket_received=on_bucket_received,
            attempt_id=attempt_id,
            target_version=target_version,
            attempt_validator=fence.is_current if fence is not None else None,
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            manual_process_weights_after_loading(self.model_runner.model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model = self.model_runner.model
            model_config = self.model_runner.vllm_config.model_config
            process_weights_after_loading(model, model_config, self.device)
