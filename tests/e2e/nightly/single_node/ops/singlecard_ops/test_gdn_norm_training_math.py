# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.gdn_batch_invariant import GDNTrainingRMSNorm


@pytest.mark.parametrize('length', [1, 63, 65, 144])
def test_gdn_norm_checkpoint_gamma_and_rounding(length):
    torch.manual_seed(42)
    delta = torch.linspace(-0.4, -0.01, 128).bfloat16().npu()
    checkpoint_gamma = 1.0 + delta.float()
    assert not torch.equal(checkpoint_gamma, checkpoint_gamma.bfloat16().float())
    source = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(128, device='npu', dtype=torch.bfloat16)),
                             eps=1e-6, activation='silu', group_size=None, norm_before_gate=True)
    parameter = source.weight
    norm = GDNTrainingRMSNorm(source)
    norm.load_state_dict({'weight': checkpoint_gamma})
    assert norm.weight is parameter
    assert norm.weight.dtype == torch.float32
    assert torch.equal(norm.weight, checkpoint_gamma)
    x = torch.randn(length, 128, device='npu', dtype=torch.bfloat16)
    z = torch.randn_like(x)
    with torch.no_grad():
        actual = norm(x, z)
        native, _ = torch_npu.npu_rms_norm(x, 1.0 + delta.float(), source.eps)
        expected = (native * torch.nn.functional.silu(z.float())).to(x.dtype)
        split = torch.cat([norm(x[i:i+1], z[i:i+1]) for i in range(length)])
    assert torch.equal(actual, expected)
    assert torch.equal(actual, split)


@pytest.mark.parametrize('group_size,norm_before_gate,activation', [(64, True, 'silu'), (None, False, 'silu'), (None, True, 'sigmoid')])
def test_gdn_norm_rejects_unsupported_contract(group_size, norm_before_gate, activation):
    source = SimpleNamespace(group_size=group_size, norm_before_gate=norm_before_gate, activation=activation)
    with pytest.raises(ValueError):
        GDNTrainingRMSNorm(source)
