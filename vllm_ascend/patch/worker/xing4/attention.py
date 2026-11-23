"""Xing-only eager expanded MLA diagnostic using training FA math."""


def install():
    import torch
    from vllm import envs

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.ops.training_attention import training_causal_attention

    original = AscendAttentionBackendImpl.forward
    if getattr(original, "_xing_training_candidate", False):
        return

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, **kwargs):
        if self.vllm_config.model_config.hf_text_config.model_type != "xing4_0":
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output=output, **kwargs)
        if output is None:
            raise ValueError("Xing candidate requires allocated output")
        if attn_metadata is None:
            return output.fill_(0)
        if not (self.vllm_config.model_config.enforce_eager and envs.VLLM_BATCH_INVARIANT and attn_metadata.causal):
            raise ValueError("Xing training attention candidate requires eager, BI, causal")
        if self.pcp_enabled or self.sliding_window is not None or self.sinks is not None:
            raise ValueError("Unsupported Xing attention topology")
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        n = query.shape[0]
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if key is not None and value is not None:
            query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
        if not hasattr(self, "_xing_causal_mask"):
            self._xing_causal_mask = torch.ones(2048, 2048, dtype=torch.bool, device=query.device).triu(1)
        out = training_causal_attention(
            query,
            key,
            value,
            self.key_cache,
            self.value_cache,
            attn_metadata,
            self.scale,
            self._xing_causal_mask,
            envs.VLLM_KV_CACHE_LAYOUT,
        )
        output[:n] = out[:n]
        return output

    forward._xing_training_candidate = True
    AscendAttentionBackendImpl.forward = forward
