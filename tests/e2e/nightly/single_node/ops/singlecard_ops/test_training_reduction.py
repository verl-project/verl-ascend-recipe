# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.training_reduction import fixed_vocab_sum


@pytest.mark.parametrize('width', [1025, 124160])
def test_fixed_vocabulary_sum_math_and_batch_independence(width):
    torch.manual_seed(71)
    row = (torch.randn(1, width, dtype=torch.bfloat16).float() - 4).exp().npu()
    expected = row.cpu().double().sum(-1)
    single = fixed_vocab_sum(row)
    assert torch.allclose(single.cpu().double(), expected, rtol=5e-7, atol=0)
    for count in (2, 9, 143):
        batch = row.repeat(count, 1)
        assert torch.equal(fixed_vocab_sum(batch), single.expand(count))
        assert torch.equal(fixed_vocab_sum(batch[:, None, :], keepdim=True).flatten(), single.expand(count))
    assert torch.equal(row.cpu().double().sum(-1), expected)
