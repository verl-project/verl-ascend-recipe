"""Exact native mHC Sinkhorn: balanced row sums, sequential column sums."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _sinkhorn(X, Y, T: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr = 64):
    t = tl.program_id(0) * B + tl.arange(0, B)
    x00 = tl.load(X + t * 16 + 0, mask=t < T, other=0.0)
    x01 = tl.load(X + t * 16 + 1, mask=t < T, other=0.0)
    x02 = tl.load(X + t * 16 + 2, mask=t < T, other=0.0)
    x03 = tl.load(X + t * 16 + 3, mask=t < T, other=0.0)
    x10 = tl.load(X + t * 16 + 4, mask=t < T, other=0.0)
    x11 = tl.load(X + t * 16 + 5, mask=t < T, other=0.0)
    x12 = tl.load(X + t * 16 + 6, mask=t < T, other=0.0)
    x13 = tl.load(X + t * 16 + 7, mask=t < T, other=0.0)
    x20 = tl.load(X + t * 16 + 8, mask=t < T, other=0.0)
    x21 = tl.load(X + t * 16 + 9, mask=t < T, other=0.0)
    x22 = tl.load(X + t * 16 + 10, mask=t < T, other=0.0)
    x23 = tl.load(X + t * 16 + 11, mask=t < T, other=0.0)
    x30 = tl.load(X + t * 16 + 12, mask=t < T, other=0.0)
    x31 = tl.load(X + t * 16 + 13, mask=t < T, other=0.0)
    x32 = tl.load(X + t * 16 + 14, mask=t < T, other=0.0)
    x33 = tl.load(X + t * 16 + 15, mask=t < T, other=0.0)
    for _ in range(ITERS):
        d0 = (x00 + x01) + (x02 + x03) + EPS
        x00 = x00 / d0
        x01 = x01 / d0
        x02 = x02 / d0
        x03 = x03 / d0
        d1 = (x10 + x11) + (x12 + x13) + EPS
        x10 = x10 / d1
        x11 = x11 / d1
        x12 = x12 / d1
        x13 = x13 / d1
        d2 = (x20 + x21) + (x22 + x23) + EPS
        x20 = x20 / d2
        x21 = x21 / d2
        x22 = x22 / d2
        x23 = x23 / d2
        d3 = (x30 + x31) + (x32 + x33) + EPS
        x30 = x30 / d3
        x31 = x31 / d3
        x32 = x32 / d3
        x33 = x33 / d3
        e0 = ((x00 + x10) + x20) + x30 + EPS
        x00 = x00 / e0
        x10 = x10 / e0
        x20 = x20 / e0
        x30 = x30 / e0
        e1 = ((x01 + x11) + x21) + x31 + EPS
        x01 = x01 / e1
        x11 = x11 / e1
        x21 = x21 / e1
        x31 = x31 / e1
        e2 = ((x02 + x12) + x22) + x32 + EPS
        x02 = x02 / e2
        x12 = x12 / e2
        x22 = x22 / e2
        x32 = x32 / e2
        e3 = ((x03 + x13) + x23) + x33 + EPS
        x03 = x03 / e3
        x13 = x13 / e3
        x23 = x23 / e3
        x33 = x33 / e3
    tl.store(Y + t * 16 + 0, x00, mask=t < T)
    tl.store(Y + t * 16 + 1, x01, mask=t < T)
    tl.store(Y + t * 16 + 2, x02, mask=t < T)
    tl.store(Y + t * 16 + 3, x03, mask=t < T)
    tl.store(Y + t * 16 + 4, x10, mask=t < T)
    tl.store(Y + t * 16 + 5, x11, mask=t < T)
    tl.store(Y + t * 16 + 6, x12, mask=t < T)
    tl.store(Y + t * 16 + 7, x13, mask=t < T)
    tl.store(Y + t * 16 + 8, x20, mask=t < T)
    tl.store(Y + t * 16 + 9, x21, mask=t < T)
    tl.store(Y + t * 16 + 10, x22, mask=t < T)
    tl.store(Y + t * 16 + 11, x23, mask=t < T)
    tl.store(Y + t * 16 + 12, x30, mask=t < T)
    tl.store(Y + t * 16 + 13, x31, mask=t < T)
    tl.store(Y + t * 16 + 14, x32, mask=t < T)
    tl.store(Y + t * 16 + 15, x33, mask=t < T)


def sinkhorn(x, iters=20, eps=1e-6, mode=0):
    assert x.dtype == torch.float32 and x.shape[-2:] == (4, 4)
    x = x.contiguous()
    out = torch.empty_like(x)
    t = x.numel() // 16
    _sinkhorn[(triton.cdiv(t, 64),)](x, out, t, iters, eps, enable_fp_fusion=False)
    return out
