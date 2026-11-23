"""Xing dense SwiGLU with the native unfused BF16 activation boundary."""


def install():
    import torch
    from torch import nn
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP

    from vllm_ascend.ascend_config import get_ascend_config

    original = DeepseekV2MLP.__init__
    if getattr(original, "_xing_training_activation", False):
        return

    class XingTrainingSiluAndMul(nn.Module):
        def forward(self, x):
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        config = get_current_vllm_config()
        rl = get_ascend_config().rl_config
        if (
            config.model_config.hf_text_config.model_type == "xing4_0"
            and rl.enabled
            and rl.training_tensor_parallel_size == 2
        ):
            self.act_fn = XingTrainingSiluAndMul()

    init._xing_training_activation = True
    DeepseekV2MLP.__init__ = init
