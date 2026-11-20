# SPDX-License-Identifier: Apache-2.0
"""Match the TP2 native vocabulary cross-entropy reduction in FP32."""

import torch

from vllm_ascend.ops.training_reduction import fixed_vocab_sum


def training_logprobs_tp2(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] % 2:
        raise ValueError("TP2 training logprobs require an evenly split vocabulary")
    values = logits.float()
    shifted = values - values.max(dim=-1, keepdim=True).values
    left, right = shifted.chunk(2, dim=-1)
    # Keep the two local reductions and one FP32 sum used by the trainer's
    # TP all-reduce. Native CE forms log(sum_exp) - selected_shifted_logit,
    # then verl negates that loss to obtain log-probabilities.
    log_sum = (fixed_vocab_sum(left.exp()) + fixed_vocab_sum(right.exp())).log()
    return -(log_sum.unsqueeze(-1) - shifted)
