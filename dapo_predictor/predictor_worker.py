"""Recipe-side worker extensions for predictor-driven prompt reordering."""

from __future__ import annotations

import os

import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.profiler import DistProfiler
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .length_scheduler.calibration import MonotoneCalibration, fit_pava
from .length_scheduler.predictor import CensoredLogNormalPredictor


class PredictorDataParallelPPOActor(DataParallelPPOActor):
    """PPO actor with an attached linear predictor scorer for prompt reordering.

    The predictor scorer maps last-token hidden states to a scalar score.
    It is initialized from rank 0 and broadcast across all processes.
    """

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        hidden_size = getattr(getattr(actor_module, "config", None), "hidden_size", None)
        if hidden_size is None and hasattr(actor_module, "module"):
            hidden_size = getattr(getattr(actor_module.module, "config", None), "hidden_size", None)
        hidden_size = hidden_size or 4096
        self.predictor_scorer = nn.Linear(hidden_size, 1, bias=False).to(next(actor_module.parameters()).device)
        self.predictor_calibration: MonotoneCalibration | None = None
        self.predictor_calibration_hidden = torch.empty((0, hidden_size), dtype=torch.float32)
        self.predictor_calibration_tokens = torch.empty((0,), dtype=torch.float32)
        self.distribution_predictor: CensoredLogNormalPredictor | None = None
        if torch.distributed.is_initialized():
            for param in self.predictor_scorer.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def extract_hidden_states(self, data: DataProto):
        """Extract last-token hidden states from the actor model for predictor scoring."""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=torch.distributed.group.WORLD,
            )
        else:
            micro_batches = data.split(micro_batch_size)

        hidden_states_list = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                hidden_states = self._forward_predictor_micro_batch(model_inputs, temperature)
                # hidden_states = self. _forward_micro_batch(model_inputs, temperature)
                hidden_states_list.append(hidden_states)

        # Concatenate hidden states from all micro batches. Dynamic batching
        # rearranges rows by sequence length, so restore the caller's row order
        # before returning prompt-aligned predictions.
        all_hidden_states = torch.concat(hidden_states_list, dim=0)  # [total_batch_size, hidden_size]
        if use_dynamic_bsz:
            all_hidden_states = restore_dynamic_batch(all_hidden_states, batch_idx_list)

        return all_hidden_states

    def extract_hidden_taps(self, data: DataProto, layer_indices: tuple[int, ...]) -> dict[int, torch.Tensor]:
        """Extract last-token hidden states for the configured D2 decoder taps."""
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must be non-empty and unique")
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        data = data.select(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["multi_modal_inputs"] if has_multi_modal_inputs else [],
        )
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=torch.distributed.group.WORLD,
            )
        else:
            micro_batches = data.split(micro_batch_size)

        by_layer: dict[int, list[torch.Tensor]] = {index: [] for index in layer_indices}
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                hidden = self._forward_predictor_micro_batch(model_inputs, temperature, layer_indices=layer_indices)
            for index in layer_indices:
                by_layer[index].append(hidden[index])
        result = {index: torch.cat(rows, dim=0) for index, rows in by_layer.items()}
        if use_dynamic_bsz:
            result = {index: restore_dynamic_batch(rows, batch_idx_list) for index, rows in result.items()}
        return result

    def _forward_predictor_micro_batch(self, micro_batch, temperature, layer_indices: tuple[int, ...] | None = None):
        """Process a single micro batch following the full dp_actor forward pass."""
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            # Handle multi-modal inputs
            multi_modal_inputs = {}
            if "multi_modal_inputs" in micro_batch.keys():
                if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = torch.cat(
                            [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                        )

            # Handle position_ids dimensionality for Qwen2VL mrope
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                # Apply remove-padding optimization
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad position_ids
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # Unpad multi-modal inputs
                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # Ulysses sequence parallel processing
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    **extra_args,
                )
                selected_layers = layer_indices or (-1,)
                full_hidden_by_layer = {}
                for layer_index in selected_layers:
                    layer_hidden = output.hidden_states[layer_index].squeeze(0)
                    if self.use_ulysses_sp:
                        layer_hidden = gather_outputs_and_unpad(
                            layer_hidden,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    full_hidden_by_layer[layer_index] = pad_input(
                        hidden_states=layer_hidden.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    ).squeeze(-1)
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
                selected_layers = layer_indices or (-1,)
                full_hidden_by_layer = {index: output.hidden_states[index] for index in selected_layers}
            # Select the last non-padding token by tensor position. This works
            # for both ordinary 2-D position ids and multi-axis MRoPE ids.
            token_positions = torch.arange(seqlen, device=attention_mask.device).unsqueeze(0)
            eos_mask_idx = torch.argmax(token_positions * attention_mask, dim=-1)
            last_token_by_layer = {
                index: hidden[
                    torch.arange(batch_size, device=hidden.device),
                    eos_mask_idx.to(hidden.device),
                ]
                for index, hidden in full_hidden_by_layer.items()
            }
            return last_token_by_layer if layer_indices is not None else last_token_by_layer[-1]

    @staticmethod
    def listmle_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        eps: float = 1e-10,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute ListMLE loss with random feature shuffling for regularization.

        Shuffles hidden dimensions, sorts by true labels, then computes the
        ListMLE loss to train the predictor scorer to predict response length.
        """
        random_indices = torch.randperm(y_pred.shape[-1], generator=generator).to(y_pred.device)
        y_pred_shuffled = y_pred[:, random_indices].float()
        y_true_shuffled = y_true[:, random_indices].float()
        _, indices = y_true_shuffled.sort(descending=True, dim=-1)
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
        return torch.mean(torch.sum(observation_loss, dim=1))


class PredictorAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Async worker that adds predictor score computation and update RPCs.

    Extends the standard worker with three new entry points:
    - compute_predictor_score: forward pass to score prompts
    - update_predictor: train the predictor with ListMLE loss
    - update_actor: overridden to keep actor loaded for predictor when needed
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self._pending_offload_param_restore = None
        if self._is_actor:
            self.actor = PredictorDataParallelPPOActor(
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
            cfg = self._predictor_cfg()
            backend = str(cfg.get("backend", "linear_listmle"))
            if backend == "censored_lognormal":
                checkpoint = cfg.get("d2_checkpoint")
                if not checkpoint:
                    raise ValueError("predictor_reorder.d2_checkpoint is required for censored_lognormal")
                self.actor.distribution_predictor = CensoredLogNormalPredictor.from_checkpoint(checkpoint)
            elif backend != "linear_listmle":
                raise ValueError("predictor_reorder.backend must be 'linear_listmle' or 'censored_lognormal'")

    def _predictor_cfg(self):
        """Resolve predictor config from worker or trainer level."""
        # Worker-side config is usually rooted at `actor_rollout_ref`, so `trainer.*`
        # is not always available here.
        cfg = OmegaConf.select(self.config, "predictor_reorder", default=None)
        if cfg is None:
            cfg = OmegaConf.select(self.config, "trainer.predictor_reorder", default=None)
        return cfg or {}

    def _actor_params_are_offloaded(self) -> bool:
        """Check whether FSDP actor parameters are currently on CPU."""
        return next(self.actor_module_fsdp.parameters()).device.type == "cpu"

    def _sync_predictor_scorer_device(self):
        """Ensure predictor scorer lives on the same device as the actor."""
        actor_device = next(self.actor_module_fsdp.parameters()).device
        if next(self.actor.predictor_scorer.parameters()).device != actor_device:
            self.actor.predictor_scorer = self.actor.predictor_scorer.to(actor_device)

    def _finish_predictor_actor_access(self) -> None:
        """Restore configured actor offload state after predictor work."""
        if self._pending_offload_param_restore is not None:
            self._is_offload_param = self._pending_offload_param_restore
            self._pending_offload_param_restore = None
        if self._is_offload_param and not self._actor_params_are_offloaded():
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_predictor(self, path: str) -> None:
        """Persist the small predictor/calibration state beside a verl checkpoint."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            calibration = self.actor.predictor_calibration
            torch.save(
                {
                    "weight": self.actor.predictor_scorer.weight.detach().cpu(),
                    "length_calibration": (
                        None if calibration is None else {"scores": calibration.scores, "tokens": calibration.tokens}
                    ),
                    "calibration_hidden": self.actor.predictor_calibration_hidden,
                    "calibration_tokens": self.actor.predictor_calibration_tokens,
                    "schema_version": 1,
                },
                path,
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_predictor(self, path: str) -> None:
        """Restore predictor state on every actor rank."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._sync_predictor_scorer_device()
        self.actor.predictor_scorer.weight.data.copy_(
            checkpoint["weight"].to(self.actor.predictor_scorer.weight.device)
        )
        calibration = checkpoint.get("length_calibration")
        self.actor.predictor_calibration = (
            None
            if calibration is None
            else MonotoneCalibration(tuple(calibration["scores"]), tuple(calibration["tokens"]))
        )
        self.actor.predictor_calibration_hidden = checkpoint.get(
            "calibration_hidden", torch.empty((0, self.actor.predictor_scorer.in_features))
        ).float()
        self.actor.predictor_calibration_tokens = checkpoint.get("calibration_tokens", torch.empty(0)).float()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """Override to keep actor loaded on GPU when predictor needs it afterward."""
        cfg = self._predictor_cfg()
        keep_actor_loaded = bool(cfg.get("predictor_keep_actor_loaded", False))

        if not (keep_actor_loaded and self._is_offload_param):
            return super().update_actor(data)
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self._sync_predictor_scorer_device()

        original_is_offload_param = self._is_offload_param
        self._is_offload_param = False
        self._pending_offload_param_restore = original_is_offload_param

        return super().update_actor(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="purple", role="predictor_compute_score")
    def compute_predictor_score(self, data: DataProto):
        """Score each prompt by running predictor scorer on sampled hidden states.

        Takes one sample per prompt group (stride=n), extracts hidden states,
        scores them, and broadcasts the score back to all samples of that prompt.
        """
        assert self._is_actor
        loaded_actor_for_predictor = self._is_offload_param and self._actor_params_are_offloaded()
        if loaded_actor_for_predictor:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self._sync_predictor_scorer_device()

        data = data.to(get_device_id())
        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        n = self.config.rollout.n
        batch_size = data.batch["input_ids"].shape[0]
        sample_indices = list(range(0, batch_size, n))
        sampled_non_tensors = {}
        for key, val in data.non_tensor_batch.items():
            sampled_non_tensors[key] = val[sample_indices] if isinstance(val, np.ndarray) else val
        # Create sampled data proto for one sample per prompt group
        sampled_data = DataProto.from_dict(
            {
                "input_ids": data.batch["input_ids"][sample_indices],
                "attention_mask": data.batch["attention_mask"][sample_indices],
                "position_ids": data.batch["position_ids"][sample_indices],
            },
            non_tensors=sampled_non_tensors,
        )
        sampled_data.meta_info = data.meta_info.copy()
        cfg = self._predictor_cfg()
        backend = str(cfg.get("backend", "linear_listmle"))
        with self.ulysses_sharding_manager:
            if backend == "censored_lognormal":
                predictor = self.actor.distribution_predictor
                if predictor is None:
                    raise RuntimeError("censored_lognormal predictor was not initialized")
                feature_keys = predictor.prior.feature_keys
                try:
                    layer_indices = tuple(int(key.removeprefix("layer_")) for key in feature_keys)
                except ValueError as error:
                    raise ValueError("D2 feature keys must use the layer_<index> schema") from error
                hidden_by_index = self.actor.extract_hidden_taps(sampled_data, layer_indices)
                predictions = predictor.predict_distribution_batch(
                    {key: hidden_by_index[index] for key, index in zip(feature_keys, layer_indices, strict=True)}
                )
                risk_weight = float(cfg.get("risk_weight", 0.5))
                actor_device = next(self.actor_module_fsdp.parameters()).device
                predicted_work_per_prompt = torch.tensor(
                    [row.as_length_prediction().risk(risk_weight) for row in predictions],
                    device=actor_device,
                    dtype=torch.float32,
                )
                scores = predicted_work_per_prompt
                predictor_ready = True
            else:
                sampled_hidden_states = self.actor.extract_hidden_states(data=sampled_data)
                scores = self.actor.predictor_scorer(sampled_hidden_states).squeeze(-1)
                if self.actor.predictor_calibration is None:
                    predicted_work_per_prompt = torch.full_like(scores, float("nan"))
                else:
                    predicted_work_per_prompt = torch.tensor(
                        [self.actor.predictor_calibration(float(score)) for score in scores.detach().cpu()],
                        device=scores.device,
                        dtype=scores.dtype,
                    )
                predictor_ready = self.actor.predictor_calibration is not None

        if backend == "linear_listmle" and self.actor.predictor_calibration is None:
            predicted_work_per_prompt = torch.full_like(scores, float("nan"))
        else:
            predicted_work_per_prompt = predicted_work_per_prompt.to(scores.device, dtype=scores.dtype)

        predictor_scores = torch.zeros(batch_size, device=scores.device, dtype=scores.dtype)
        predicted_work = torch.zeros(batch_size, device=scores.device, dtype=scores.dtype)
        for i, sample_idx in enumerate(sample_indices):
            predictor_scores[sample_idx : min(sample_idx + n, batch_size)] = scores[i]
            predicted_work[sample_idx : min(sample_idx + n, batch_size)] = predicted_work_per_prompt[i]

        output = DataProto.from_dict(
            tensors={"predictor_scores": predictor_scores, "epws_predicted_work": predicted_work},
            meta_info={"predictor_ready": predictor_ready, "predictor_backend": backend},
        ).to("cpu")
        if self._pending_offload_param_restore is not None:
            self._is_offload_param = self._pending_offload_param_restore
            self._pending_offload_param_restore = None
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="orange", role="predictor_update")
    def update_predictor(self, prompt_batch: DataProto, response_batch: DataProto):
        """Train the predictor scorer to predict response length via ListMLE loss.

        Gathers hidden states and response lengths across all GPUs,
        trains the predictor for `epochs` steps, and returns metrics.
        """
        assert self._is_actor
        cfg = self._predictor_cfg()
        if not cfg.get("enable", False):
            return DataProto(meta_info={"metrics": {}})

        if str(cfg.get("backend", "linear_listmle")) == "censored_lognormal":
            n = self.config.rollout.n
            local_prompts = len(response_batch) // n
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            sp_size = self.config.actor.ulysses_sequence_parallel_size
            global_prompts = local_prompts * (world_size // sp_size)
            output = DataProto(
                meta_info={
                    "metrics": {
                        "predictor/total_samples": global_prompts,
                        "predictor/observed_rollouts": global_prompts * n,
                        "predictor/calibration_samples": global_prompts,
                        "predictor/calibration_ready": 1.0,
                        "predictor/update_time_s": 0.0,
                    }
                }
            )
            self._finish_predictor_actor_access()
            return output

        loaded_actor_for_predictor = self._is_offload_param and self._actor_params_are_offloaded()
        if loaded_actor_for_predictor:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self._sync_predictor_scorer_device()

        prompt_batch = prompt_batch.to(get_device_id())
        prompt_batch.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        prompt_batch.meta_info["temperature"] = self.config.rollout.temperature
        prompt_batch.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        prompt_batch.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        n = self.config.rollout.n
        batch_size = prompt_batch.batch["input_ids"].shape[0]
        sp_size = self.config.actor.ulysses_sequence_parallel_size
        sample_indices = list(range(0, batch_size, n))
        sampled_non_tensors = {}
        for key, val in prompt_batch.non_tensor_batch.items():
            sampled_non_tensors[key] = val[sample_indices] if isinstance(val, np.ndarray) else val
        sampled_prompt = DataProto(
            batch=prompt_batch.batch[sample_indices],
            non_tensor_batch=sampled_non_tensors,
            meta_info=prompt_batch.meta_info.copy(),
        )

        with self.ulysses_sharding_manager:
            hidden_states = self.actor.extract_hidden_states(sampled_prompt)

        response_batch = response_batch.to(get_device_id())
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        response_lengths = (response_batch.batch["responses"] != pad_token_id).sum(dim=1)
        response_lengths = response_lengths.view(-1, n).max(dim=1).values.float()

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

            gathered_hidden = [torch.empty_like(hidden_states) for _ in range(world_size)]
            gathered_lengths = [torch.empty_like(response_lengths) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_hidden, hidden_states)
            torch.distributed.all_gather(gathered_lengths, response_lengths)
            hidden_states = torch.cat(gathered_hidden, dim=0)
            response_lengths = torch.cat(gathered_lengths, dim=0)

            if sp_size > 1:
                rank_data_len = len(response_lengths) // world_size
                dp_world_size = world_size // sp_size

                # Reshape and extract SP rank 0 data
                reshaped_response = response_lengths.view(dp_world_size, sp_size, rank_data_len)
                reshaped_hidden = hidden_states.view(dp_world_size, sp_size, rank_data_len, -1)

                response_lengths = reshaped_response[:, 0, :].flatten()
                hidden_states = reshaped_hidden[:, 0, :, :].flatten(0, 1)

        token_lengths = response_lengths.detach().cpu().float()
        label_group_size = max(1, self.config.rollout.response_length // 40)
        response_lengths = response_lengths // label_group_size

        predictor = self.actor.predictor_scorer.float()
        optimizer = torch.optim.AdamW(
            predictor.parameters(),
            lr=cfg.get("lr", 3e-5),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        dataset = TensorDataset(hidden_states.float(), response_lengths.float())
        predictor_seed = int(cfg.get("seed", 1))
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(predictor_seed)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.get("batch_size", 32),
            shuffle=True,
            drop_last=False,
            generator=dataloader_generator,
        )
        listmle_generator = torch.Generator()
        listmle_generator.manual_seed(predictor_seed)
        epochs = cfg.get("epochs", 10)

        kendalltau = None
        try:
            from scipy.stats import kendalltau
        except ImportError:
            pass

        metrics = {}
        with Timer(name="predictor_update", logger=None) as timer:
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                epoch_kendall_taus = []
                for batch_hidden, batch_lengths in dataloader:
                    preds = predictor(batch_hidden).squeeze(-1).unsqueeze(0)
                    labels = batch_lengths.unsqueeze(0)
                    loss = self.actor.listmle_loss(preds, labels, generator=listmle_generator)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                    if epoch == 0 or epoch == epochs - 1:
                        pred_numpy = preds.squeeze().detach().cpu().float().numpy()
                        true_numpy = labels.squeeze().detach().cpu().float().numpy()

                        if len(pred_numpy) > 1 and kendalltau is not None:
                            kendall_tau, _ = kendalltau(pred_numpy, true_numpy)
                            if not np.isnan(kendall_tau):
                                epoch_kendall_taus.append(kendall_tau)

                if epoch == 0 or epoch == epochs - 1:
                    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                    avg_kendall_tau = np.mean(epoch_kendall_taus) if epoch_kendall_taus else 0.0
                    metrics[f"predictor/epoch_{epoch}_loss"] = avg_epoch_loss
                    metrics[f"predictor/epoch_{epoch}_kendall_tau"] = avg_kendall_tau

                metrics["predictor/final_loss"] = epoch_loss / max(num_batches, 1)

        calibration_max_samples = int(cfg.get("calibration_max_samples", 4096))
        if calibration_max_samples < 2:
            raise ValueError("predictor_reorder.calibration_max_samples must be at least 2")
        replay_hidden = torch.cat(
            [self.actor.predictor_calibration_hidden, hidden_states.detach().cpu().float()], dim=0
        )[-calibration_max_samples:]
        replay_tokens = torch.cat([self.actor.predictor_calibration_tokens, token_lengths], dim=0)[
            -calibration_max_samples:
        ]
        if len(replay_tokens) >= 2:
            predictor_device = next(predictor.parameters()).device
            with torch.inference_mode():
                replay_scores = predictor(replay_hidden.to(predictor_device)).squeeze(-1).detach().cpu().tolist()
            self.actor.predictor_calibration = fit_pava(replay_scores, replay_tokens.tolist())
        self.actor.predictor_calibration_hidden = replay_hidden
        self.actor.predictor_calibration_tokens = replay_tokens

        metrics["predictor/epochs"] = epochs
        metrics["predictor/update_time_s"] = timer.last
        metrics["predictor/total_samples"] = len(dataset)
        metrics["predictor/observed_rollouts"] = len(dataset) * n
        metrics["predictor/calibration_samples"] = len(replay_tokens)
        metrics["predictor/calibration_ready"] = float(self.actor.predictor_calibration is not None)

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        self._finish_predictor_actor_access()
        return output
