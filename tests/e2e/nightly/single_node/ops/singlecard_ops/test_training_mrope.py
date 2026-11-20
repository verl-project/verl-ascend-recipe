# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.training_mrope import training_mrope


@pytest.mark.parametrize('length', [1, 65, 144])
def test_mrope_matches_native_absolute_frequency_contract(length):
    torch.manual_seed(42)
    q = torch.randn(length, 8, 256, dtype=torch.bfloat16, device='npu')
    k = torch.randn(length, 1, 256, dtype=torch.bfloat16, device='npu')
    positions = torch.stack([torch.arange(length), torch.arange(length) * 2, torch.arange(length) * 3]).npu()
    actual_q, actual_k = training_mrope(q, k, positions, 64, 10000000, [11, 11, 10])
    # Native bridge creates absolute frequencies using a broadcast matmul and
    # picks axes before cosine/sine. Exercise distinct spatial/time positions.
    inv = 1 / (10000000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device='npu') / 64))
    freqs = (inv[None, :, None].expand(3, -1, -1) @ positions[:, None, :].float()).transpose(1, 2)
    index = torch.zeros(32, dtype=torch.long, device='npu')
    index[1:33:3] = 1
    index[2:30:3] = 2
    chosen = torch.stack([freqs[int(index[i]), :, i] for i in range(32)], dim=-1)
    emb = torch.cat([chosen, chosen], -1)[:, None, :]
    for x, actual in ((q, actual_q), (k, actual_k)):
        t = x[..., :64]
        rotated = torch.cat([-t[..., 32:], t[..., :32]], -1)
        expected = torch.cat([t * emb.cos().to(x.dtype) + rotated * emb.sin().to(x.dtype), x[..., 64:]], -1)
        assert torch.equal(actual, expected)
        assert torch.equal(actual[..., 64:], x[..., 64:])
    last_q, last_k = training_mrope(q[-1:], k[-1:], positions[:, -1:], 64, 10000000, [11, 11, 10])
    assert torch.equal(last_q, actual_q[-1:])
    assert torch.equal(last_k, actual_k[-1:])
    parts = [training_mrope(q[:, i:i+2], k, positions, 64, 10000000, [11, 11, 10])[0] for i in range(0, 8, 2)]
    assert torch.equal(torch.cat(parts, dim=1), actual_q)
