"""Xing-only native FP32 router GEMM and sigmoid/group-1 top-k arithmetic."""


def install():
    import torch
    from vllm.config import get_current_vllm_config

    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear
    from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
    from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent

    original_gate = AscendGateLinear.forward
    if getattr(original_gate, "_xing_native_gate", False):
        return

    def enabled():
        cfg = get_current_vllm_config()
        rl = get_ascend_config().rl_config
        return cfg.model_config.hf_text_config.model_type == "xing4_0" and rl.enabled and rl.training_expert_math

    def gate(self, x):
        if enabled():
            if self.weight.dtype != torch.bfloat16 or self.out_dtype != torch.float32 or self.bias is not None:
                raise ValueError("Xing native gate requires bias-free BF16 weights and FP32 logits")
            return linear_persistent(x.reshape(-1, x.shape[-1]).float(), self.weight.float()).reshape(
                *x.shape[:-1], self.weight.shape[0]
            ), None
        return original_gate(self, x)

    gate._xing_native_gate = True
    AscendGateLinear.forward = gate
    original_route = AscendFusedTopKRouter._compute_routing

    def route(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        if enabled():
            if not (
                self.scoring_func == "sigmoid"
                and self.renormalize
                and self.num_expert_group == 1
                and self.topk_group == 1
                and self.e_score_correction_bias is not None
                and self.bias_vl is None
                and self.tid2eid is None
                and router_logits.dtype == torch.float32
            ):
                raise ValueError(
                    "Xing native route requires FP32 sigmoid, one group, correction bias and normalization"
                )
            scores = torch.sigmoid(router_logits.float())
            indices = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1, sorted=True).indices
            selected = scores.gather(1, indices)
            weights = selected / (selected.sum(-1, keepdim=True) + 1e-20) if self.top_k > 1 else selected
            weights = weights * self.routed_scaling_factor
            return weights, indices.to(torch.int32 if indices_type is None else indices_type)
        return original_route(self, hidden_states, router_logits, indices_type, input_ids=input_ids)

    AscendFusedTopKRouter._compute_routing = route
