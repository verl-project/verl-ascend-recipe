# SPDX-License-Identifier: Apache-2.0
"""Pinned Xing4 mHC rounding and native-training gradient regression checks."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.patch.worker.xing4.mhc_kernels import MhcMatmul, matmul_mix
from vllm_ascend.patch.worker.xing4.sinkhorn_kernel import sinkhorn


@pytest.mark.parametrize("tokens", [1, 4, 65, 529])
@pytest.mark.parametrize("scale", [0.1, 3.0, 10.0])
def test_sinkhorn_exact(tokens, scale):
    torch.manual_seed(42)
    logits = torch.randn(tokens, 4, 4, device="npu") * scale
    value = torch.exp(logits - logits.amax(-1, keepdim=True))
    expected = value
    for _ in range(20):
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-6)
        expected = expected / (expected.sum(-2, keepdim=True) + 1e-6)
    actual = sinkhorn(value)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("tokens", [1, 4, 65, 257])
def test_mhc_mix_forward_and_backward(tokens):
    torch.manual_seed(42)
    residual = torch.randn(tokens, 4, 3584, device="npu", dtype=torch.bfloat16, requires_grad=True)
    comb = torch.randn(tokens, 4, 4, device="npu", dtype=torch.bfloat16).transpose(-1, -2).requires_grad_()
    expected = torch.bmm(comb, residual)
    actual = MhcMatmul.apply(comb, residual)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(actual, matmul_mix(comb, residual))
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    assert torch.equal(comb.grad, torch.bmm(gradient, residual.detach().transpose(-1, -2)))
    assert torch.equal(residual.grad, torch.bmm(comb.detach().transpose(-1, -2), gradient))
