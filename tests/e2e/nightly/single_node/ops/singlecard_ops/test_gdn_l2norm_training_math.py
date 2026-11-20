# SPDX-License-Identifier: Apache-2.0
"""Validate the epsilon contract and invariance to token/head batching."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.gdn_batch_invariant import gdn_l2norm_training_math


@pytest.mark.parametrize('length', [1, 63, 65, 144])
@torch.no_grad()
def test_l2norm_is_exact_across_token_and_head_partitions(length):
    torch.manual_seed(42)
    x = torch.randn(1, length, 16, 128).bfloat16().npu()
    full = gdn_l2norm_training_math(x)
    tokenwise = torch.cat([gdn_l2norm_training_math(row) for row in x.split(1, dim=1)], dim=1)
    headwise = torch.cat([gdn_l2norm_training_math(head) for head in x.split(2, dim=2)], dim=2)
    assert torch.equal(full, tokenwise)
    assert torch.equal(full, headwise)
    assert torch.isfinite(full).all()


@torch.no_grad()
def test_l2norm_clamps_norm_instead_of_adding_epsilon_to_squared_norm():
    x = torch.zeros(3, 128, dtype=torch.bfloat16, device='npu')
    x[1, 0] = 1e-8
    x[2, 0] = 1e-4
    expected = torch.zeros_like(x)
    expected[1, 0] = (x[1, 0].float() / 1e-6).bfloat16()
    expected[2, 0] = 1
    assert torch.equal(gdn_l2norm_training_math(x), expected)
