# SPDX-License-Identifier: Apache-2.0
import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.training_logprobs import training_logprobs_tp2


def test_training_logprobs_vocabulary_scale_and_batch_invariance():
    torch.manual_seed(17)
    logits = (torch.randn(9, 248320, dtype=torch.bfloat16) * 3).npu()
    original = logits.clone()
    result = training_logprobs_tp2(logits)
    # Independent high precision mathematical reference and normalization.
    reference = logits.cpu().double().log_softmax(-1).float()
    assert torch.allclose(result.cpu(), reference, rtol=0, atol=4e-6)
    assert torch.allclose(result.exp().sum(-1), torch.ones(9, device='npu'), rtol=0, atol=2e-6)
    assert torch.equal(result, torch.cat([training_logprobs_tp2(x) for x in logits.split(1)]))
    assert torch.equal(logits, original)
    assert result.dtype == torch.float32
