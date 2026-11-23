"""Xing-only BF16 RoPE operation boundaries matching native MLA training."""


def install():
    import torch
    from vllm.config import get_current_vllm_config

    from vllm_ascend.ops.rotary_embedding import AscendDeepseekScalingRotaryEmbedding as C

    original_init = C.__init__
    original_forward = C.forward
    if getattr(original_forward, "_xing_native_rounding", False):
        return

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._xing_native_rounding_enabled = (
            get_current_vllm_config().model_config.hf_text_config.model_type == "xing4_0"
        )

    def forward(self, positions, query, key, offsets=None):
        if not self._xing_native_rounding_enabled:
            return original_forward(self, positions, query, key, offsets)
        if offsets is not None:
            raise ValueError("Xing native RoPE does not support offsets")
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
            raise ValueError("Xing native RoPE requires BF16")
        if key.ndim == 2:
            key = key[:, None, :]
        cos = self.cos_cached.index_select(0, positions).unsqueeze(1).to(query.dtype)
        sin = self.sin_cached.index_select(0, positions).unsqueeze(1).to(query.dtype)

        def rotate(x):
            if not self.is_neox_style:
                x = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
            a, b = x.chunk(2, dim=-1)
            return x * cos + torch.cat((-b, a), dim=-1) * sin

        return rotate(query), rotate(key)

    forward._xing_native_rounding = True
    C.__init__ = init
    C.forward = forward
