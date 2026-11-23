"""Xing mHC arithmetic for the explicitly enabled numerical recipe."""

import torch
from torch.nn import functional as F


def pre(self, hidden_states):
    dtype = hidden_states.dtype
    n = self.n
    residual = hidden_states.reshape(-1, n, self.hidden_size)
    flat = residual.flatten(start_dim=1).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
    from vllm import envs

    if envs.VLLM_BATCH_INVARIANT:
        from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent

        projected = linear_persistent(flat.to(dtype), self.hc_fn.to(dtype))
    else:
        projected = F.linear(flat.to(dtype), self.hc_fn.to(dtype))
    pre_w, post_w, comb_w = projected.float().split([n, n, n * n], dim=-1)
    pre_b, post_b, comb_b = self.hc_base.to(dtype).split([n, n, n * n])
    pre_s, post_s, comb_s = self.hc_scale.to(dtype).unbind(0)
    pre_mix = torch.sigmoid(pre_w * pre_s + pre_b)
    post_mix = 2 * torch.sigmoid(post_w * post_s + post_b)
    logits = comb_w.view(-1, n, n) * comb_s + comb_b.view(n, n)
    logits = torch.clamp(logits, min=self.h_res_clamp_min, max=self.h_res_clamp_max)
    comb = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    from .sinkhorn_kernel import sinkhorn

    comb = sinkhorn(comb, iters=self.sinkhorn_iterations, eps=self.norm_eps)
    collapsed = (pre_mix.unsqueeze(-1).to(dtype) * residual).sum(dim=1)
    return collapsed.to(dtype), comb.to(dtype).transpose(-1, -2).contiguous(), post_mix.to(dtype).unsqueeze(-1)


def post(self, h_res, original_residual, h_post, layer_output_with_bias):
    value, bias = layer_output_with_bias
    if bias is not None:
        raise ValueError("Xing checkpoint has no attention/MLP output bias")
    residual = original_residual.reshape(-1, self.n, self.hidden_size)
    from .mhc_kernels import post_mix

    return post_mix(h_res.transpose(-1, -2), residual, h_post, value).reshape_as(original_residual)


def install():
    from vllm_xing4_0_plugin.xing4_0 import Xing4_0MHC

    Xing4_0MHC.forward = pre
    Xing4_0MHC.fused_h_res_h_post_bda_inference = post
