# SPDX-License-Identifier: Apache-2.0
"""Explicit Megatron expert weighting and token accumulation boundaries."""

import torch


def permute_router_weights(weights: torch.Tensor, reverse_mapping: torch.Tensor) -> torch.Tensor:
    """NPU permutation returns original-slot -> sorted-slot indices."""
    flat = weights.reshape(-1)
    result = torch.empty_like(flat)
    result.scatter_(0, reverse_mapping.long(), flat)
    return result.reshape(-1, 1)


def combine_training_experts(
    expert_output: torch.Tensor,
    reverse_mapping: torch.Tensor,
    topk_ids: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """Sum already weighted experts in native ascending-expert order.

    Megatron's unfused dispatcher scatters BF16 expert outputs into a BF16
    destination. Keep that actual operator and ordering, rather than invoking
    a fused unpermute which also multiplies router probabilities.
    """
    token_count, top_k = topk_ids.shape
    order = torch.argsort(topk_ids.reshape(-1), stable=True)
    sorted_output = expert_output.index_select(0, reverse_mapping.long().index_select(0, order))
    token_indices = torch.div(order, top_k, rounding_mode="floor")
    result = expert_output.new_zeros(restore_shape)
    if torch.are_deterministic_algorithms_enabled():
        result.index_add_(0, token_indices, sorted_output)
    else:
        result.scatter_add_(0, token_indices[:, None].expand_as(sorted_output), sorted_output)
    return result
