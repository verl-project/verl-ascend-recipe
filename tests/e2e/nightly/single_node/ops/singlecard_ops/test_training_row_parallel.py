# SPDX-License-Identifier: Apache-2.0
"""Exercise training reduction boundaries and fresh operands after weight updates."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops.linear_op import TrainingRowParallelOp, TrainingSharedRowParallelOp
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent


class GatheredOperands:
    def __init__(self, x, weight):
        self.x = x
        self.weight = weight
        self.calls = 0

    def all_gather(self, local, dim):
        assert dim == -1
        self.calls += 1
        return self.x if self.calls % 2 else self.weight


class RowOpForTest(TrainingRowParallelOp):
    @property
    def comm_group(self):
        return self.test_group


class SharedRowOpForTest(TrainingSharedRowParallelOp):
    @property
    def comm_group(self):
        return self.test_group


@pytest.mark.parametrize("op_class,reduced", [(RowOpForTest, True), (SharedRowOpForTest, False)])
@torch.no_grad()
def test_row_reduction_matches_tp2_and_reads_updated_weights(op_class, reduced):
    torch.manual_seed(42)
    x = torch.randn(65, 1024, dtype=torch.bfloat16, device='npu')
    weight = torch.randn(256, 1024, dtype=torch.bfloat16, device='npu')
    op = object.__new__(op_class)
    op.quant_method = UnquantizedLinearMethod()
    op.reduce_results = reduced
    op.bias = None
    op.input_is_parallel = True
    op.layer = SimpleNamespace(weight=weight[:, :128])
    op.test_group = GatheredOperands(x, weight)
    previous = None
    for update in (0, 1):
        if update:
            weight.add_(0.25)
        expected = sum(linear_persistent(a, b) for a, b in zip(x.chunk(2, -1), weight.chunk(2, -1)))
        actual, bias = op.apply_impl(x[:, :128])
        assert bias is None
        assert torch.equal(actual, expected)
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual.clone()
    assert op.test_group.calls == 4
