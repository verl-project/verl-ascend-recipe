# SPDX-License-Identifier: Apache-2.0
"""Fixed-order FP32 row reduction using the existing Ascend BI GEMM."""

import torch
import torch.nn.functional as F


def fixed_vocab_sum(values: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Sum each vocabulary row independently of the number of token rows.

    Two fixed GEMM reductions avoid both CANN's batch-dependent reduction tree
    and a long sequential FP32 accumulation over the entire vocabulary. This
    primitive is shared with native Megatron CE via scoped operator dispatch;
    native CE still owns its softmax, collectives, loss, and backward.
    """
    from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent

    if values.dtype != torch.float32 or values.device.type != "npu":
        raise ValueError("fixed_vocab_sum requires an NPU FP32 tensor")
    width = values.shape[-1]
    if width == 0:
        raise ValueError("fixed_vocab_sum requires a nonempty vocabulary")
    tile = 1024
    chunks = (width + tile - 1) // tile
    flat = values.reshape(-1, width)
    if flat.shape[0] == 0:
        shape = (*values.shape[:-1], 1) if keepdim else values.shape[:-1]
        return values.new_empty(shape)
    padded = F.pad(flat, (0, chunks * tile - width))
    partial = linear_persistent(
        padded.reshape(-1, tile), values.new_ones((1, tile))
    ).reshape(-1, chunks)
    result = linear_persistent(partial, values.new_ones((1, chunks)))
    shape = (*values.shape[:-1], 1) if keepdim else values.shape[:-1]
    return result.reshape(shape)
