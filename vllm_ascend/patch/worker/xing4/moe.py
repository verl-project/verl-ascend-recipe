"""Xing-only eager EP candidate using native expert weighting boundaries."""


def install():
    import sys

    from vllm import envs
    from vllm.config import get_current_vllm_config

    from vllm_ascend import ascend_forward_context as ctx
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ops import linear, linear_op

    old_comm = ctx.select_moe_comm_method
    if getattr(old_comm, "_xing_expert_candidate", False):
        return

    def select_comm(num_tokens, vllm_config, *args, **kwargs):
        rl = get_ascend_config().rl_config
        if vllm_config.model_config.hf_text_config.model_type == "xing4_0" and rl.training_expert_math:
            p = vllm_config.parallel_config
            if not (
                rl.enabled
                and envs.VLLM_BATCH_INVARIANT
                and vllm_config.model_config.enforce_eager
                and p.enable_expert_parallel
                and p.data_parallel_size == 1
            ):
                raise ValueError("Xing training expert candidate requires eager RL, BI, EP and DP1")
            return ctx.MoECommType.ALLTOALL
        return old_comm(num_tokens, vllm_config, *args, **kwargs)

    select_comm._xing_expert_candidate = True
    ctx.select_moe_comm_method = select_comm
    for name in ["vllm_ascend.worker.model_runner_v1", "vllm_ascend.worker.v2.model_runner"]:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "select_moe_comm_method", None) is old_comm:
            module.select_moe_comm_method = select_comm
    old_row = linear_op._get_row_parallel_op

    def row(prefix, layer):
        cfg = get_current_vllm_config()
        rl = get_ascend_config().rl_config
        if (
            cfg.model_config.hf_text_config.model_type == "xing4_0"
            and rl.training_expert_math
            and prefix.endswith(".shared_experts.down_proj")
        ):
            if rl.training_tensor_parallel_size != 2:
                raise ValueError("Xing shared expert candidate requires training TP2")
            return linear_op.TrainingSharedRowParallelOp(layer)
        return old_row(prefix, layer)

    linear_op._get_row_parallel_op = row
    old_rep = linear_op.get_replicated_op

    def replicated(disable_tp, prefix, layer):
        cfg = get_current_vllm_config()
        rl = get_ascend_config().rl_config
        if (
            cfg.model_config.hf_text_config.model_type == "xing4_0"
            and rl.training_expert_math
            and prefix.endswith(".mlp.gate")
        ):
            return linear_op.TrainingRouterLinearOp(layer), 0, 1
        return old_rep(disable_tp, prefix, layer)

    linear_op.get_replicated_op = replicated
    if linear.get_replicated_op is old_rep:
        linear.get_replicated_op = replicated


def install_activation():
    import torch
    from vllm.config import get_current_vllm_config

    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ops.fused_moe import moe_mlp

    original = moe_mlp._unified_apply_activation
    if getattr(original, "_xing_unfused_activation", False):
        return

    def activation(inp, hidden, quant_method):
        if (
            get_current_vllm_config().model_config.hf_text_config.model_type == "xing4_0"
            and get_ascend_config().rl_config.training_expert_math
        ):
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation

            if inp.activation != MoEActivation.SILU or inp.quant.is_quant or inp.swiglu_limit > 0:
                raise ValueError("Xing training experts require unquantized unclamped SiLU")
            gate, up = hidden.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up
        return original(inp, hidden, quant_method)

    activation._xing_unfused_activation = True
    moe_mlp._unified_apply_activation = activation
