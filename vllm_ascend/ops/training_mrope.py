# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MRoPE with the native trainer's BF16 intermediate boundaries."""

import torch


def training_mrope(query, key, positions, rotary_dim, base, mrope_section):
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("Qwen training MRoPE expects text or three-axis positions")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or rotary_dim % 2:
        raise ValueError("Qwen training MRoPE requires BF16 and an even rotary dimension")
    # Compute on the same device as the trainer, without a CPU-built trig cache.
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=query.device) / rotary_dim))
    freqs = positions.to(device=query.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    selected = freqs[0].clone()
    for axis, offset in enumerate((1, 2), start=1):
        section = slice(offset, mrope_section[axis] * 3, 3)
        selected[..., section] = freqs[axis, ..., section]
    cos = selected.cos().to(query.dtype).repeat(1, 2).unsqueeze(1)
    sin = selected.sin().to(query.dtype).repeat(1, 2).unsqueeze(1)

    def apply(x):
        rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
        left, right = rotated.chunk(2, dim=-1)
        half = torch.cat((-right, left), dim=-1)
        # Each multiply and the addition round to BF16, as in native MCore.
        return torch.cat((rotated * cos + half * sin, passthrough), dim=-1)

    return apply(query), apply(key)
