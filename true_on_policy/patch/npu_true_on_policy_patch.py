from dataclasses import dataclass
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch.nn.parameter import Parameter
from vllm.config import VllmConfig
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.v1.sample.sampler import Sampler
from vllm_ascend import ascend_forward_context
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.activation import AscendSiluAndMul, AscendSwigluOAIAndMul
from vllm_ascend.ops.fused_moe import experts_selector, moe_comm_method, moe_mlp
from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV

from verl.workers.rollout.vllm_rollout import utils


@dataclass(frozen=True, slots=True)
class MoEAllToAllCombineMetadata:
    input_splits: np.ndarray
    output_splits: np.ndarray
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor | None
    reversed_local_input_permutation_mapping: torch.Tensor
    reversed_global_input_permutation_mapping: torch.Tensor | None
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size


def run_silu_and_mul_with_weight_prefetch_train_infer_consistent(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
    from vllm_ascend.utils import get_weight_prefetch_method

    weight_prefetch_method = get_weight_prefetch_method()
    weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(
        weight_prefetch_method.MLP_DOWN,
        x,
    )

    half_hidden_size = x.shape[-1] // 2
    output = F.silu(x[..., :half_hidden_size]) * x[..., half_hidden_size:]

    weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(output)
    return output


def apply_unquantized_grouped_mlp_train_infer_consistent(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
    swiglu_limit: float = 0.0,
    lora_context=None,
    expanded_row_idx: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None]:
    # topk_ids/expanded_row_idx are only consumed by the MoE LoRA branch of
    # vllm-ascend's unquant_apply_mlp, so they are inert while lora_context is
    # None. swiglu_limit and an active LoRA context would change the numerics,
    # so keep rejecting exactly those.
    if swiglu_limit or lora_context is not None:
        raise ValueError(
            "apply_unquantized_grouped_mlp_train_infer_consistent does not support "
            "swiglu_limit/lora_context. The Megatron-style grouped MLP path cannot "
            "honor them; disable true_on_policy for this model or extend the patch."
        )

    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_output = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    if activation == "swigluoai":
        _, _, hidden_size = w1.shape
        gate_up_output = AscendSwigluOAIAndMul.swiglu_oai_forward(
            gate_up_output.view(-1, hidden_size),
        )
    else:
        gate_hidden_size = gate_up_output.shape[-1] // 2
        gate_up_output = F.silu(gate_up_output[..., :gate_hidden_size]) * gate_up_output[..., gate_hidden_size:]

    if topk_scales is not None:
        gate_up_output *= topk_scales

    down_output = torch_npu.npu_grouped_matmul(
        x=[gate_up_output],
        weight=[w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    # vllm-ascend 0.23.0's fused_experts unpacks (mlp_output, before_gmm2_evt);
    # the unquantized path has no gmm2 event, matching upstream's `return ..., None`.
    return down_output, None


def select_experts_with_torch_topk_train_infer_consistent(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    global_num_experts: int = -1,
    tid2eid: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = hidden_states, renormalize, scoring_func, global_num_experts, input_ids

    if tid2eid is not None:
        raise ValueError(
            "select_experts_with_torch_topk_train_infer_consistent does not support "
            "tid2eid (sqrtsoftplus token-to-expert routing). The torch.topk routing "
            "path cannot honor it; disable true_on_policy for this model or extend "
            "the patch."
        )

    num_tokens, num_experts = router_logits.shape

    if use_grouped_topk and topk_group is not None and num_expert_group is not None:
        if e_score_correction_bias is not None:
            if e_score_correction_bias.dtype != router_logits.dtype:
                e_score_correction_bias = e_score_correction_bias.to(router_logits.dtype)
            routing_scores = router_logits + e_score_correction_bias.unsqueeze(0)
        else:
            routing_scores = router_logits

        num_groups = num_expert_group
        experts_per_group = num_experts // num_groups
        k_per_group = max(1, top_k // topk_group)

        group_scores = (
            routing_scores.view(num_tokens, num_groups, experts_per_group).topk(k_per_group, dim=-1)[0].sum(dim=-1)
        )

        selected_group_indices = torch.topk(
            group_scores,
            k=topk_group,
            dim=-1,
            sorted=True,
        )[1]
        selected_group_mask = torch.zeros_like(group_scores)
        selected_group_mask.scatter_(1, selected_group_indices, 1)

        selected_expert_mask = (
            selected_group_mask.unsqueeze(-1).expand(num_tokens, num_groups, experts_per_group).reshape(num_tokens, -1)
        )
        masked_routing_scores = routing_scores.masked_fill(
            ~selected_expert_mask.bool(),
            float("-inf"),
        )

        if e_score_correction_bias is not None:
            _, topk_ids = torch.topk(
                masked_routing_scores.to(torch.float32),
                k=top_k,
                dim=-1,
                sorted=True,
            )
            topk_logits = router_logits.gather(1, topk_ids)
        else:
            topk_logits, topk_ids = torch.topk(
                masked_routing_scores.to(torch.float32),
                k=top_k,
                dim=-1,
                sorted=True,
            )
    else:
        if e_score_correction_bias is not None:
            if e_score_correction_bias.dtype != router_logits.dtype:
                e_score_correction_bias = e_score_correction_bias.to(router_logits.dtype)
            routing_scores = router_logits + e_score_correction_bias.unsqueeze(0)
            _, topk_ids = torch.topk(
                routing_scores.to(torch.float32),
                k=top_k,
                dim=-1,
                sorted=True,
            )
            topk_logits = router_logits.gather(1, topk_ids)
        else:
            topk_logits, topk_ids = torch.topk(
                router_logits.to(torch.float32),
                k=top_k,
                dim=-1,
                sorted=True,
            )

    topk_weights = torch.softmax(
        topk_logits,
        dim=-1,
        dtype=torch.float32,
    ).type_as(router_logits)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids.to(torch.int32)


def dispatch_tokens_with_all_to_all_train_infer_consistent(
    self,
    token_dispatch_input: MoETokenDispatchInput,
) -> MoETokenDispatchOutput:
    use_quant = token_dispatch_input.quant.is_int_quant
    hidden_states = token_dispatch_input.hidden_states
    topk_weights = token_dispatch_input.topk_weights
    topk_ids = token_dispatch_input.topk_ids

    (
        permuted_local_tokens,
        reversed_local_permutation_mapping,
        tokens_per_expert,
        input_splits,
        output_splits,
        local_expert_indices,
        hidden_shape,
        hidden_shape_before_permute,
    ) = self._dispatch_preprocess(hidden_states, topk_ids)

    num_permuted_tokens = permuted_local_tokens.shape[0]
    dynamic_scale_after_all_to_all = None

    if use_quant:
        permuted_local_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
            permuted_local_tokens,
        )
        _, dynamic_scale_after_all_to_all, dynamic_scale_handle = async_all_to_all(
            dynamic_scale,
            output_splits,
            input_splits,
            self.ep_group,
        )
        dynamic_scale_handle.wait()
        dynamic_scale.untyped_storage().resize_(0)

    _, global_tokens, tokens_handle = async_all_to_all(
        permuted_local_tokens,
        output_splits,
        input_splits,
        self.ep_group,
    )
    tokens_handle.wait()
    permuted_local_tokens.untyped_storage().resize_(0)

    flat_topk_weights = topk_weights.view(-1)
    permuted_weights = torch.zeros(
        num_permuted_tokens,
        1,
        dtype=flat_topk_weights.dtype,
        device=flat_topk_weights.device,
    )
    permuted_weights.scatter_(
        0,
        reversed_local_permutation_mapping.unsqueeze(-1).long(),
        flat_topk_weights.unsqueeze(-1),
    )

    (
        global_tokens,
        final_dynamic_scale,
        reversed_global_permutation_mapping,
    ) = self._dispatch_postprocess(
        global_tokens,
        dynamic_scale_after_all_to_all,
        local_expert_indices,
        use_quant,
        # vllm-ascend 0.23.0 added dst_type/scale_type (used only by the fp8
        # e8m0 quant branch); keep the 0.18.0 defaults for the unquantized path.
        torch.bfloat16,
        torch.bfloat16,
    )

    _, global_weights, weights_handle = async_all_to_all(
        permuted_weights,
        output_splits,
        input_splits,
        self.ep_group,
    )
    weights_handle.wait()
    permuted_weights.untyped_storage().resize_(0)

    if self.num_local_experts > 1 and local_expert_indices is not None:
        global_weights, _ = torch_npu.npu_moe_token_permute(
            global_weights,
            local_expert_indices,
        )

    return MoETokenDispatchOutput(
        hidden_states=global_tokens,
        dynamic_scale=final_dynamic_scale,
        group_list=tokens_per_expert,
        group_list_type=1,
        combine_metadata=MoEAllToAllCombineMetadata(
            input_splits=input_splits,
            output_splits=output_splits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            reversed_local_input_permutation_mapping=reversed_local_permutation_mapping,
            reversed_global_input_permutation_mapping=reversed_global_permutation_mapping,
            hidden_shape=hidden_shape,
            hidden_shape_before_permute=hidden_shape_before_permute,
        ),
        topk_scales=global_weights,
    )


def build_expert_major_scatter_indices_train_infer_consistent(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    num_tokens = topk_ids.shape[0]
    routing_map = torch.zeros(
        num_tokens,
        num_experts,
        dtype=torch.bool,
        device=topk_ids.device,
    )
    routing_map.scatter_(1, topk_ids.long(), True)

    token_indices = (
        torch.arange(num_tokens, device=topk_ids.device)
        .unsqueeze(0)
        .expand(
            num_experts,
            -1,
        )
    )
    return token_indices.masked_select(routing_map.T.contiguous())


def unpermute_tokens_with_megatron_scatter_train_infer_consistent(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    _, hidden_size = restore_shape
    input_dtype = permuted_tokens.dtype
    scatter_indices = sorted_indices.to(torch.int64).reshape(-1)

    assert permuted_tokens.shape[0] == scatter_indices.numel(), (
        f"permuted rows {permuted_tokens.shape[0]} != indices {scatter_indices.numel()}"
    )

    output = torch.zeros(
        restore_shape,
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    output.scatter_add_(
        0,
        scatter_indices.unsqueeze(1).expand(-1, hidden_size),
        permuted_tokens,
    )
    return output.to(dtype=input_dtype)


def combine_tokens_after_all_to_all_train_infer_consistent(
    self,
    permutated_local_input_tokens: torch.Tensor,
    combine_metadata: MoEAllToAllCombineMetadata,
) -> torch.Tensor:
    permuted_local_tokens = permutated_local_input_tokens
    scatter_indices = build_expert_major_scatter_indices_train_infer_consistent(
        combine_metadata.topk_ids,
        num_experts=self.num_experts,
    )
    output = unpermute_tokens_with_megatron_scatter_train_infer_consistent(
        permuted_tokens=permuted_local_tokens,
        sorted_indices=scatter_indices,
        restore_shape=combine_metadata.hidden_shape_before_permute,
    )
    return output.view(combine_metadata.hidden_shape)


def patch_compute_logits_passthrough_train_infer_consistent(
    model,
    vocab_size: int,
    banned_token_ids=None,
) -> None:
    _ = vocab_size, banned_token_ids
    original_compute_logits = model.compute_logits

    def compute_logits(self, *args, **kwargs) -> torch.Tensor:
        _ = self
        return original_compute_logits(*args, **kwargs)

    model.compute_logits = MethodType(compute_logits, model)


@staticmethod
def compute_logprobs_from_logits_train_infer_consistent(
    logits: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits_shifted = logits - logits_max
    return logits_shifted - logits_shifted.exp().sum(dim=-1, keepdim=True).log()


def run_row_parallel_linear_with_padded_reduce_scatter_train_infer_consistent(
    self,
    input_,
) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
    if self.input_is_parallel:
        input_parallel = input_
    else:
        split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
        input_parallel = split_input[self.tp_rank].contiguous()

    assert self.quant_method is not None

    bias = None if self.tp_rank > 0 or self.skip_bias_add else self.bias
    output_parallel = self.quant_method.apply(self, input_parallel, bias)

    if self.reduce_results and self.tp_size > 1:
        pad_size = (self.tp_size - output_parallel.shape[0] % self.tp_size) % self.tp_size
        if pad_size > 0:
            output_parallel = F.pad(output_parallel, (0, 0, 0, pad_size))

        scattered = tensor_model_parallel_reduce_scatter(output_parallel, dim=0)
        output = tensor_model_parallel_all_gather(scattered, dim=0)

        if pad_size > 0:
            output = output[:-pad_size]
    else:
        output = output_parallel

    output_bias = self.bias if self.skip_bias_add else None
    if not self.return_bias:
        return output
    return output, output_bias


def select_all_to_all_moe_comm_method_train_infer_consistent(
    num_tokens: int,
    vllm_config: VllmConfig,
    is_draft_model=False,
) -> MoECommType:
    _ = num_tokens, vllm_config, is_draft_model
    return MoECommType.ALLTOALL


# Captured at import time, before any monkey patching, so re-applying the
# patches never wraps the wrapper itself.
_ORIG_SETUP_MOE_COMM_METHOD = moe_comm_method.setup_moe_comm_method
_ORIG_GET_MOE_COMM_METHOD = moe_comm_method.get_moe_comm_method


def setup_moe_comm_method_with_alltoall_backfill_train_infer_consistent(moe_config) -> None:
    """Register the ALLTOALL impl even when expert parallel is disabled.

    Since vllm-ascend 0.23.0, setup_moe_comm_method registers ALLTOALL only
    when ep_size > 1, but true_on_policy forces MoECommType.ALLTOALL (see the
    select patch above) for train-infer consistency. Without this backfill the
    forced selection resolves to None and the MoE forward crashes with
    "AttributeError: 'NoneType' object has no attribute 'prepare'".
    """
    _ORIG_SETUP_MOE_COMM_METHOD(moe_config)
    if MoECommType.ALLTOALL not in moe_comm_method._MoECommMethods:
        moe_comm_method._MoECommMethods[MoECommType.ALLTOALL] = moe_comm_method.AlltoAllCommImpl(moe_config)


def get_moe_comm_method_with_alltoall_fallback_train_infer_consistent(moe_comm_type):
    """Resolve ALLTOALL even when it was never registered (ep_size == 1).

    Belt-and-braces companion to the setup backfill above:
    set_ascend_forward_context lazy-imports get_moe_comm_method inside the
    context manager on every forward, so patching the module attribute here
    takes effect regardless of from-import binding order anywhere else.
    """
    method = _ORIG_GET_MOE_COMM_METHOD(moe_comm_type)
    if method is None and moe_comm_type == MoECommType.ALLTOALL:
        allgather = moe_comm_method._MoECommMethods.get(MoECommType.ALLGATHER)
        if allgather is not None:
            method = moe_comm_method.AlltoAllCommImpl(allgather.moe_config)
            moe_comm_method._MoECommMethods[MoECommType.ALLTOALL] = method
    return method


def apply_batch_consistency_patches() -> None:
    """Apply batch-consistency monkey patches for vLLM Ascend."""

    AscendSiluAndMul.forward_oot = run_silu_and_mul_with_weight_prefetch_train_infer_consistent
    moe_mlp.unquant_apply_mlp = apply_unquantized_grouped_mlp_train_infer_consistent
    experts_selector._select_experts_with_fusion_ops = select_experts_with_torch_topk_train_infer_consistent

    TokenDispatcherWithAll2AllV.token_dispatch = dispatch_tokens_with_all_to_all_train_infer_consistent
    TokenDispatcherWithAll2AllV._combine_postprocess = combine_tokens_after_all_to_all_train_infer_consistent

    utils.monkey_patch_compute_logits = patch_compute_logits_passthrough_train_infer_consistent
    Sampler.compute_logprobs = compute_logprobs_from_logits_train_infer_consistent
    RowParallelLinear.forward = run_row_parallel_linear_with_padded_reduce_scatter_train_infer_consistent
    ascend_forward_context.select_moe_comm_method = select_all_to_all_moe_comm_method_train_infer_consistent
    moe_comm_method.setup_moe_comm_method = setup_moe_comm_method_with_alltoall_backfill_train_infer_consistent
    moe_comm_method.get_moe_comm_method = get_moe_comm_method_with_alltoall_fallback_train_infer_consistent
