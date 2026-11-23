"""Batched Xing mHC post with the reference BF16 GEMM rounding boundaries."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mix_kernel(
    A,
    R,
    Output,
    T: tl.constexpr,
    H: tl.constexpr,
    AST: tl.constexpr,
    ASR: tl.constexpr,
    ASC: tl.constexpr,
    RST: tl.constexpr,
    RSR: tl.constexpr,
    RSC: tl.constexpr,
    GRID: tl.constexpr,
    BLOCK_N: tl.constexpr = 256,
    BLOCK_K: tl.constexpr = 256,
):
    nblocks = tl.cdiv(H, BLOCK_N)
    for task in range(tl.program_id(0), T * nblocks, GRID):
        token = task // nblocks
        col = (task % nblocks) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, 4)
        inner = tl.arange(0, BLOCK_K)
        acc = tl.zeros((4, BLOCK_N), dtype=tl.float32)
        a = tl.load(A + token * AST + rows[:, None] * ASR + inner[None, :] * ASC, mask=inner[None, :] < 4, other=0.0)
        b = tl.load(
            R + token * RST + col[:, None] * RSC + inner[None, :] * RSR,
            mask=(col[:, None] < H) & (inner[None, :] < 4),
            other=0.0,
        )
        product = tl.dot(a, tl.trans(b))
        acc += product
        tl.store(Output + token * 4 * H + rows[:, None] * H + col[None, :], acc, mask=col[None, :] < H)


def matmul_mix(comb, residual):
    assert residual.shape[-2] == 4 and residual.dtype == comb.dtype == torch.bfloat16
    original_shape = residual.shape
    hidden = residual.shape[-1]
    residual = residual.reshape(-1, 4, hidden)
    comb = comb.reshape(-1, 4, 4)
    tokens = residual.shape[0]
    if tokens == 0:
        return torch.empty(original_shape, device=residual.device, dtype=residual.dtype)
    out = torch.zeros((tokens, 4, hidden), device=residual.device, dtype=residual.dtype)
    grid = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"] // 2
    _mix_kernel[(grid,)](comb, residual, out, tokens, hidden, *comb.stride(), *residual.stride(), grid)
    return out.reshape(original_shape)


def post_mix(comb, residual, post, value):
    mixed = matmul_mix(comb, residual)
    return post * value.unsqueeze(-2) + mixed


class MhcMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comb, residual):
        ctx.save_for_backward(comb, residual)
        return matmul_mix(comb, residual)

    @staticmethod
    def backward(ctx, grad):
        comb, residual = ctx.saved_tensors
        hidden = residual.shape[-1]
        g = grad.reshape(-1, 4, hidden)
        a = comb.reshape(-1, 4, 4)
        r = residual.reshape(-1, 4, hidden)
        da = torch.bmm(g, r.transpose(-1, -2)).reshape_as(comb) if ctx.needs_input_grad[0] else None
        dr = torch.bmm(a.transpose(-1, -2), g).reshape_as(residual) if ctx.needs_input_grad[1] else None
        return da, dr
