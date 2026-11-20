# SPDX-License-Identifier: Apache-2.0
"""Check FP32 gate evaluation and post-top-k softmax against native formulas."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.linear_op import TrainingRouterLinearOp


def _router():
    return SimpleNamespace(scoring_func='softmax', renormalize=True, use_grouped_topk=False,
                           e_score_correction_bias=None, bias_vl=None, tid2eid=None,
                           routed_scaling_factor=1.0, top_k=8)


@pytest.mark.parametrize('length', [1, 65, 144])
@torch.no_grad()
def test_training_router_uses_fp32_products_and_topk_softmax(monkeypatch, length):
    import vllm_ascend.ascend_config as config

    monkeypatch.setattr(config, 'get_ascend_config', lambda: SimpleNamespace(
        rl_config=SimpleNamespace(training_router_math=True)))
    torch.manual_seed(42)
    x = torch.randn(length, 256).bfloat16().npu()
    weight = torch.randn(32, 256).bfloat16().npu()
    gate = SimpleNamespace(quant_method=UnquantizedLinearMethod(), bias=None,
                           layer=SimpleNamespace(weight=weight))
    logits, bias = TrainingRouterLinearOp.apply_impl(gate, x)
    expected_logits = (x.cpu().double() @ weight.cpu().double().t()).float()
    assert logits.dtype == torch.float32 and bias is None
    torch.testing.assert_close(logits.cpu(), expected_logits, atol=2e-5, rtol=2e-5)
    scores, ids = torch.topk(logits, 8, dim=-1, sorted=False)
    expected = torch.softmax(scores, dim=-1)
    probs, actual_ids = AscendFusedTopKRouter._compute_routing(_router(), x, logits, None)
    assert torch.equal(actual_ids.long(), ids)
    assert torch.equal(probs, expected)
    short_probs, short_ids = AscendFusedTopKRouter._compute_routing(_router(), x[:1], logits[:1], None)
    assert torch.equal(short_probs, probs[:1])
    assert torch.equal(short_ids, actual_ids[:1])
    with pytest.raises(ValueError, match='FP32'):
        AscendFusedTopKRouter._compute_routing(_router(), x, logits.bfloat16(), None)
