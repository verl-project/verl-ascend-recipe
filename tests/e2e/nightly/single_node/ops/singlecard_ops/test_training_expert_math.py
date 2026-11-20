# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu

from vllm_ascend.ops.fused_moe.training_math import combine_training_experts, permute_router_weights


@pytest.mark.parametrize('tokens', [1, 7, 144])
def test_training_dispatch_probability_and_sum(tokens):
    torch.manual_seed(111)
    device = 'npu'
    experts, top_k, hidden = 32, 8, 64
    ids = torch.rand(tokens, experts, device=device).topk(top_k, dim=-1).indices.int()
    probabilities = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1).bfloat16()
    # Unique token values reveal the actual NPU permutation contract.
    x = torch.arange(tokens, device=device).bfloat16()[:, None].expand(tokens, hidden).contiguous()
    permuted, reverse = torch_npu.npu_moe_token_permute(x, ids)
    scales = permute_router_weights(probabilities, reverse)
    sorted_ids = torch.empty_like(ids.reshape(-1)).scatter_(0, reverse.long(), ids.reshape(-1))
    expected_scales = []
    for row, expert in zip(permuted[:, 0].cpu().int().tolist(), sorted_ids.cpu().tolist()):
        slot = (ids[row] == expert).nonzero()[0, 0]
        expected_scales.append(probabilities[row, slot])
    assert torch.equal(scales[:, 0], torch.stack(expected_scales))

    # Independent expert-major native dispatcher construction, including BF16
    # accumulation. Slot values are distinct so inverse permutations matter.
    values = torch.randn(tokens, top_k, hidden, device=device).bfloat16()
    sorted_values = torch.empty_like(values.reshape(-1, hidden))
    sorted_values.index_copy_(0, reverse.long(), values.reshape(-1, hidden))
    result = combine_training_experts(sorted_values, reverse, ids, torch.Size([tokens, hidden]))
    native_indices, native_values = [], []
    for expert in range(experts):
        positions = (ids == expert).nonzero()
        native_indices.extend(positions[:, 0].unbind())
        native_values.extend(values[positions[:, 0], positions[:, 1]].unbind())
    index = torch.stack(native_indices).long()
    source = torch.stack(native_values)
    expected = torch.zeros_like(result)
    expected.scatter_add_(0, index[:, None].expand_as(source), source)
    assert torch.equal(result, expected)
    for token in (0, tokens - 1):
        _, single_reverse = torch_npu.npu_moe_token_permute(x[token:token+1], ids[token:token+1])
        single_sorted = torch.empty_like(values[token])
        single_sorted.index_copy_(0, single_reverse.long(), values[token])
        single = combine_training_experts(single_sorted, single_reverse, ids[token:token+1], torch.Size([1, hidden]))
        assert torch.equal(single[0], result[token])
