# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import pytest
import torch
import torch_npu
from vllm_ascend.ops.training_attention import training_causal_attention


@pytest.mark.parametrize('length', [65, 257, 1024])
@pytest.mark.parametrize('layout', ['LBNHC', 'LBHNC'])
def test_cached_decode_matches_training_prefill(length, layout):
    torch.manual_seed(777)
    q = torch.randn(length, 2, 256, device='npu', dtype=torch.bfloat16)
    k = torch.randn(length, 1, 256, device='npu', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(2048, 2048, dtype=torch.bool, device='npu').triu(1)
    reference = torch_npu.npu_fusion_attention(q, k, v, 2, 'TND', atten_mask=mask,
        scale=0.0625, pre_tockens=2147483647, next_tockens=0, keep_prob=1.0,
        inner_precise=0, sparse_mode=3, actual_seq_qlen=(length,), actual_seq_kvlen=(length,))[0]
    block_size = 32
    blocks = (length + block_size - 1) // block_size
    kc = torch.zeros(blocks, block_size, 1, 256, dtype=q.dtype, device='npu')
    vc = torch.zeros_like(kc)
    # Reverse physical blocks, so a linear cache read cannot accidentally pass.
    mapping = torch.arange(blocks - 1, -1, -1, device='npu')
    padded_k, padded_v = torch.zeros_like(kc).flatten(0, 1), torch.zeros_like(vc).flatten(0, 1)
    padded_k[:length], padded_v[:length] = k, v
    kc.index_copy_(0, mapping, padded_k.view_as(kc)); vc.index_copy_(0, mapping, padded_v.view_as(vc))
    if layout == 'LBHNC':
        kc, vc = kc.transpose(1, 2).contiguous(), vc.transpose(1, 2).contiguous()
    for count in (1, min(7, length)):
        meta = SimpleNamespace(actual_seq_lengths_q=[count], seq_lens_list=[length],
            attn_state=SimpleNamespace(name='DecodeOnly'), block_tables=mapping[None, :])
        output = training_causal_attention(q[-count:], None, None, kc, vc, meta, 0.0625, mask, layout)
        assert torch.equal(output, reference[-count:])
