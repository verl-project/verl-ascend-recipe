# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in NPU GEMM dispatch shared with the Ascend rollout linear kernel."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def enable_batch_invariant_matmul():
    """Keep native Megatron modules/autograd and replace only NPU GEMM dispatch.

    Ascend rollout linear uses a fixed K=256 accumulation. Its separate mm
    fallback uses K=64 and is not numerically interchangeable. Route dense
    matrix products to the linear kernel on both sides. No CPU/CUDA dispatch
    or fused attention/normalization operator is changed.
    """
    from vllm_ascend.ops.triton.batch_invariant.matmul import linear_batch_invariant, linear_persistent

    def matmul(a, b, *, out=None):
        if a.ndim == 0 or b.ndim == 0:
            raise ValueError("matmul operands must have at least one dimension")
        left_vector, right_vector = a.ndim == 1, b.ndim == 1
        left = a.unsqueeze(0) if left_vector else a
        right = b.unsqueeze(-1) if right_vector else b
        if right.ndim == 2:
            result = linear_persistent(left.reshape(-1, left.shape[-1]), right.transpose(0, 1))
            result = result.reshape(*left.shape[:-1], right.shape[-1])
        else:
            batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
            rows, inner, columns = left.shape[-2], left.shape[-1], right.shape[-1]
            left = left.expand(*batch_shape, rows, inner).reshape(-1, rows, inner)
            right = right.expand(*batch_shape, inner, columns).reshape(-1, inner, columns)
            if left.shape[0] == 0:
                result = a.new_empty((*batch_shape, rows, columns))
            else:
                result = torch.stack(
                    [linear_persistent(x, y.transpose(0, 1)) for x, y in zip(left, right, strict=False)]
                ).reshape(*batch_shape, rows, columns)
        if left_vector:
            result = result.squeeze(-2)
        if right_vector:
            result = result.squeeze(-1)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def addmm(input, mat1, mat2, *, beta=1, alpha=1):
        result = matmul(mat1, mat2)
        if alpha != 1:
            result = result * alpha
        if beta != 0:
            result = result + input * beta
        return result

    def differentiable_linear(input, weight, bias=None):
        # The generated linear backward is unavailable on this NPU stack.
        # Native matmul/add autograd supplies the full input/weight/bias gradients.
        result = torch.matmul(input, weight.transpose(0, 1))
        return result if bias is None else result + bias

    library = torch.library.Library("aten", "IMPL")
    library.impl("mm", matmul, "NPU")
    library.impl("matmul", matmul, "NPU")
    library.impl("linear", linear_batch_invariant, "NPU")
    library.impl("linear", differentiable_linear, "AutogradPrivateUse1")
    library.impl("addmm", addmm, "NPU")
    return library
