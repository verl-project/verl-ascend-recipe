"""Xing dense MLP row reduction follows native TP2 BF16 boundaries."""


def install():
    from vllm.config import get_current_vllm_config

    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ops import linear_op

    original = linear_op._get_row_parallel_op
    if getattr(original, "_xing_dense_tp2", False):
        return

    def select(prefix, layer):
        config = get_current_vllm_config()
        rl = get_ascend_config().rl_config
        if (
            config.model_config.hf_text_config.model_type == "xing4_0"
            and rl.training_tensor_parallel_size == 2
            and prefix.endswith(".mlp.down_proj")
        ):
            if not rl.enabled:
                raise ValueError("Xing TP2 dense reduction requires RL mode")
            return linear_op.TrainingRowParallelOp(layer)
        return original(prefix, layer)

    select._xing_dense_tp2 = True
    linear_op._get_row_parallel_op = select
