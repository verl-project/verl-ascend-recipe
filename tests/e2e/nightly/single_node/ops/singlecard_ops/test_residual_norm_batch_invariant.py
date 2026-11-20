# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm


@pytest.mark.parametrize('norm_type', ['rms', 'gemma', 'rms_bias'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_residual_norm_batch_invariant(monkeypatch, norm_type, dtype):
    """Custom fused residual norm must not bypass the BI split-add path."""
    monkeypatch.setenv('VLLM_BATCH_INVARIANT', '1')
    torch.manual_seed(42)
    x = (torch.randn(144, 2048) * 0.01).to(device='npu', dtype=dtype)
    residual = (torch.randn(144, 2048) * 0.01).to(device='npu', dtype=dtype)
    weight = torch.randn(2048).to(device='npu', dtype=dtype)
    bias = torch.randn(2048).to(device='npu', dtype=dtype) if norm_type == 'rms_bias' else None
    layer = SimpleNamespace(weight=weight, bias=bias, bias_loaded=bias is not None, variance_epsilon=1e-6)
    forward = AscendGemmaRMSNorm.forward_oot if norm_type == 'gemma' else AscendRMSNorm.forward_oot
    with torch.no_grad():
        whole, whole_residual = forward(layer, x, residual)
        parts = [forward(layer, x[:128], residual[:128])]
        parts.extend(forward(layer, x[i:i + 1], residual[i:i + 1]) for i in range(128, 144))
        expected_residual = x + residual
        expected_weight = 1.0 + weight.float() if norm_type == 'gemma' else weight
        expected, _ = torch_npu.npu_rms_norm(expected_residual, expected_weight, layer.variance_epsilon)
        if bias is not None:
            expected = expected + bias
    assert torch.equal(whole, torch.cat([item[0] for item in parts]))
    assert torch.equal(whole_residual, torch.cat([item[1] for item in parts]))
    assert torch.equal(whole, expected)
    assert torch.equal(whole_residual, expected_residual)


@pytest.mark.parametrize('hidden,heads', [(256, 8), (2048, 1)])
def test_gemma_without_residual_token_and_head_invariance(monkeypatch, hidden, heads):
    monkeypatch.setenv('VLLM_BATCH_INVARIANT', '1')
    torch.manual_seed(25)
    x = torch.randn(144, heads, hidden, device='npu', dtype=torch.bfloat16)
    weight = torch.randn(hidden, device='npu', dtype=torch.bfloat16)
    layer = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    actual = AscendGemmaRMSNorm.forward_oot(layer, x)
    expected = torch_npu.npu_rms_norm(x, 1 + weight.float(), 1e-6)[0]
    assert torch.equal(actual, expected)
    head_parts = [AscendGemmaRMSNorm.forward_oot(layer, x[:, i:i+1]) for i in range(heads)]
    assert torch.equal(actual, torch.cat(head_parts, dim=1))
    assert torch.equal(actual[-1:], AscendGemmaRMSNorm.forward_oot(layer, x[-1:]))
