# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""GDN chunk forward with the native MindSpeed training reduction order.

This inference primitive consumes already normalized Q/K and computed gates.
It does not install hooks or enable a model path on import. Callers must preserve
chunk boundaries and chunk-entry states when implementing incremental decode.
"""

import torch

CHUNK_SIZE = 64


def chunk_gdn_training_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return sequence-major output and K/V-major final state.

    Q/K/V have shape [B,T,H,D]; g and beta have shape [B,T,Hv].
    This is an inference-only entry. The training counterpart keeps its native
    autograd Function and uses the same decomposed primitive operations.
    Optional MindSpeed/FLA dependencies are loaded only when called.
    """
    from fla_npu.ops import ascendc
    from mindspeed_ops.api.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from mindspeed_ops.api.triton.cumsum import chunk_local_cumsum
    from mindspeed_ops.api.triton.solve_tril import solve_tril

    tensors = (q, k, v, g, beta) + (() if initial_state is None else (initial_state,))
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("chunk_gdn_training_math is inference-only; use the native training autograd path")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or q.shape[:2] != v.shape[:2]:
        raise ValueError("Expected sequence-major Q/K/V with matching batch and token dimensions")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != torch.bfloat16:
        raise ValueError("The verified GDN contract requires BF16 Q/K/V")
    if v.shape[2] % q.shape[2]:
        raise ValueError("Value heads must be divisible by key heads")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Packed GDN requires batch dimension 1")
    repeats = v.shape[2] // q.shape[2]
    if repeats != 1:
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, head_first=False)
    a = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE, output_dtype=torch.float32
    )
    if cu_seqlens is None:
        # The installed Ascend merge kernels can corrupt the terminal matrix
        # block in a cold worker. An unused zero block moves that boundary
        # beyond real rows. Chunk-local triangular systems are independent;
        # discard the guard before recomputing W/U or recurrent state.
        length = a.shape[1]
        guard = a.new_zeros((a.shape[0], CHUNK_SIZE, a.shape[2], a.shape[3]))
        padded = torch.cat((a, guard), dim=1)
        a = solve_tril(A=padded, output_dtype=k.dtype)[:, :length].contiguous()
    else:
        a = solve_tril(A=a, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    lengths = cu_seqlens.tolist() if cu_seqlens is not None else None
    indices = None
    if lengths is not None:
        indices = [
            item
            for seq, (start, end) in enumerate(zip(lengths[:-1], lengths[1:]))
            for chunk in range((end - start + CHUNK_SIZE - 1) // CHUNK_SIZE)
            for item in (seq, chunk)
        ]
    q, k, v, g, a = (t.transpose(1, 2).contiguous() for t in (q, k, v, g, a))
    beta = beta.transpose(1, 2).contiguous().float()
    w, u = ascendc.npu_recompute_w_u_fwd(
        k, v, beta, a, CHUNK_SIZE, g=g, gk=None, cu_seqlens=lengths, chunk_indices=indices
    )
    h, v_new, final_state = ascendc.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u, g, initial_state=initial_state, cu_seqlens=lengths,
        chunk_indices=indices, output_final_state=True, chunk_size=CHUNK_SIZE,
    )
    out = ascendc.npu_chunk_fwd_o(
        q, k, v_new, h, scale, g=g, cu_seqlens=lengths,
        chunk_indices=indices, chunk_size=CHUNK_SIZE,
    )
    return out.transpose(1, 2).contiguous(), final_state


class GDNChunkReplay:
    """One sequence's chunk-entry state and at most 63 cached input tokens.

    A caller owns one instance per live sequence and must discard it on slot
    reuse or policy reset. This eager helper is not a cache-manager integration.
    It processes an arbitrary prefill append in at most two chunk calls, and a
    single-token decode append in one call. No per-token prefill loop is used.
    """

    def __init__(self, scale: float):
        self.scale = scale
        self.entry_state: torch.Tensor | None = None
        self.tail: tuple[torch.Tensor, ...] | None = None

    def append(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        g: torch.Tensor, beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs = (q, k, v, g, beta)
        if q.shape[0] != 1 or q.shape[1] == 0:
            raise ValueError("Replay append requires one non-empty sequence")
        old_length = 0 if self.tail is None else self.tail[0].shape[1]
        combined = inputs if self.tail is None else tuple(
            torch.cat((old, new), dim=1) for old, new in zip(self.tail, inputs)
        )
        output, final_state = chunk_gdn_training_math(
            *combined, scale=self.scale, initial_state=self.entry_state,
        )
        total = combined[0].shape[1]
        remaining = total % CHUNK_SIZE
        boundary = total - remaining
        if remaining == 0:
            self.entry_state = final_state
            self.tail = None
        else:
            if boundary:
                _, self.entry_state = chunk_gdn_training_math(
                    *(t[:, :boundary].contiguous() for t in combined),
                    scale=self.scale, initial_state=self.entry_state,
                )
            # clone releases the full prefill allocation, including when the
            # suffix view is already contiguous and contiguous() would alias it.
            self.tail = tuple(t[:, boundary:].clone() for t in combined)
        return output[:, old_length:], final_state



def gdn_l2norm_training_math(x: torch.Tensor) -> torch.Tensor:
    """Match the native Ascend adaptor: FP32 norm, clamp epsilon, then BF16.

    FLA's rsqrt(sum(x*x) + eps) has a different epsilon convention and
    rounding sequence from F.normalize, including for ordinary nonzero rows.
    """
    return torch.nn.functional.normalize(x.float(), p=2, dim=-1, eps=1e-6).to(x.dtype)


def gdn_gates_training_math(
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match native Megatron GDN casts and unfused exp/softplus rounding."""
    g = -a_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = b.sigmoid()
    return g.unsqueeze(0), beta.unsqueeze(0)


class GDNTrainingRMSNorm(torch.nn.Module):
    """Preserve checkpoint gamma and the native BF16 norm/gate rounding boundary.

    Construct before checkpoint loading. The source parameter object and its
    loader attributes are retained, with FP32 storage for the checkpoint gamma.
    Native Megatron stores gamma-minus-one and adds one in FP32; the HF export
    already contains the effective gamma consumed here.
    """

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        if not norm.norm_before_gate or norm.group_size is not None:
            raise ValueError("GDN training math requires ungrouped norm-before-gate")
        if norm.activation not in ("silu", "swish"):
            raise ValueError("GDN training math currently supports SiLU gating")
        self.eps = norm.eps
        self.weight = norm.weight
        self.weight.data = self.weight.data.to(torch.float32)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        import torch_npu

        normalized, _ = torch_npu.npu_rms_norm(x, self.weight, self.eps)
        if z is None:
            return normalized
        return (normalized * torch.nn.functional.silu(z.float())).to(x.dtype)
