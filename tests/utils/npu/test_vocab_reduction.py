# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.training_reduction import fixed_vocab_sum

from verl.utils.kernel.npu.vocab_reduction import NpuVocabReductionMode


def test_scoped_dispatch_and_native_sum_autograd():
    torch.manual_seed(31)
    value = torch.rand(3, 124160, device="npu", requires_grad=True)
    reference = fixed_vocab_sum(value.detach())
    with NpuVocabReductionMode(124160):
        result = value.sum(-1)
        kept = value.sum(dim=1, keepdim=True)
        cpu = torch.arange(5).sum(-1)
    assert torch.equal(result, reference)
    assert torch.equal(kept[:, 0], reference)
    assert cpu.item() == 10
    result.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))
    # Exiting the context must restore the ordinary backend dispatch.
    ordinary = value.detach().sum(-1)
    assert torch.equal(ordinary, torch.ops.aten.sum.dim_IntList(value.detach(), [-1]))
