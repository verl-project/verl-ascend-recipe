"""Native Megatron bridge registered explicitly by the pinned Xing recipe."""

# Required before Megatron imports, including standalone launcher validation.
import megatron_adaptor  # noqa: F401  # isort: skip
from copy import copy
from dataclasses import fields

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from mindspeed_bridge.models.glm_vl.modelling_glm_vl.mhc import (
    GLM53FlashHyperConnection,
    GLM53FlashHyperHeadRMSNorm,
    GLM53FlashTransformerLayer,
    GLM53FlashTransformerLayerSubmodules,
)
from torch.nn import functional as F


class XingHyperConnection(GLM53FlashHyperConnection):
    def compute_mappings(self, hidden_streams):
        x = self.as_streams(hidden_streams)
        dtype = x.dtype
        n = self.num_residual_streams
        flat = x.flatten(start_dim=-2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.input_norm_eps)
        a, b, c = F.linear(flat.to(dtype), self.mapping_proj.weight.to(dtype)).float().split([n, n, n * n], dim=-1)
        ab, bb, cb = self.bias.split([n, n, n * n])
        aa, ba, ca = self.scale.unbind(0)
        pre = torch.sigmoid(a * aa + ab)
        post = 2 * torch.sigmoid(b * ba + bb)
        logits = c.view(*c.shape[:-1], n, n) * ca + cb.view(n, n)
        logits = logits.clamp(self.config.xing_mhc_clamp_min, self.config.xing_mhc_clamp_max)
        comb = torch.exp(logits - logits.amax(-1, keepdim=True))
        for _ in range(self.sinkhorn_iterations):
            comb = comb / (comb.sum(-1, keepdim=True) + self.mhc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.mhc_eps)
        return post.to(dtype), comb.to(dtype), (pre.to(dtype).unsqueeze(-1) * x).sum(-2)

    def merge(self, residual_streams, sublayer_output, post, residual_mapping):
        from vllm_ascend.patch.worker.xing4.mhc_kernels import MhcMatmul

        return post.unsqueeze(-1) * sublayer_output.unsqueeze(-2) + MhcMatmul.apply(residual_mapping, residual_streams)


class XingHeadNorm(GLM53FlashHyperHeadRMSNorm):
    def forward(self, x):
        from mindspeed_bridge.models.glm_vl.modelling_glm_vl.mhc import GLM53FlashHyperHead

        x = GLM53FlashHyperHead.forward(self, x)
        dtype = x.dtype
        x = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def layer_spec(config, vp_stage=None):
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    for layer in spec.layer_specs:
        old = layer.submodules
        values = {f.name: getattr(old, f.name) for f in fields(old)}
        values["self_attention_hyper_connection"] = ModuleSpec(module=XingHyperConnection)
        values["mlp_hyper_connection"] = ModuleSpec(module=XingHyperConnection)
        layer.module = GLM53FlashTransformerLayer
        layer.submodules = GLM53FlashTransformerLayerSubmodules(**values)
    spec.layer_norm = XingHeadNorm
    return spec


@MegatronModelBridge.register_bridge(
    source="Xing4_0ForCausalLM", target=GPTModel, provider=MLAModelProvider, model_type="xing4_0"
)
class XingBridge(DeepSeekV3Bridge):
    def provider_bridge(self, hf_pretrained):
        p = super().provider_bridge(hf_pretrained)
        c = hf_pretrained.config
        p.transformer_layer_spec = layer_spec
        p.num_residual_streams = c.hc_mult
        p.mhc_sinkhorn_iterations = c.hc_sinkhorn_iters
        p.mhc_eps = c.hc_eps
        p.xing_mhc_clamp_min = c.mhc_h_res_clamp_min
        p.xing_mhc_clamp_max = c.mhc_h_res_clamp_max
        p.use_fused_mhc = False
        p.enable_hyper_connections = True
        p.layernorm_epsilon = c.rms_norm_eps
        p.mtp_num_layers = None
        p.make_vocab_size_divisible_by = 1
        p.moe_aux_loss_coeff = 0.0
        p.hidden_dropout = 0.0
        p.attention_dropout = 0.0
        p.moe_router_load_balancing_type = "none"
        p.moe_router_topk_scaling_factor = c.routed_scaling_factor
        p.moe_router_num_groups = c.n_group
        p.moe_router_group_topk = c.topk_group
        p.bias_activation_fusion = False
        p.bias_dropout_fusion = False
        p.gradient_accumulation_fusion = False
        p.cross_entropy_loss_fusion = False
        return p

    def mapping_registry(self):
        c = copy(self.hf_config)
        c.num_nextn_predict_layers = 0
        mappings = get_common_mapping_list(hf_config=c)
        mappings.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.expert_bias",
                hf_param="model.layers.*.mlp.gate.e_score_correction_bias",
            )
        )
        for target, source in [("self_attention_hyper_connection", "attn_hc"), ("mlp_hyper_connection", "ffn_hc")]:
            for m, h in [("mapping_proj.weight", "hc_fn"), ("bias", "hc_base"), ("scale", "hc_scale")]:
                mappings.append(
                    ReplicatedMapping(
                        megatron_param=f"decoder.layers.*.{target}.{m}", hf_param=f"model.layers.*.{source}.{h}"
                    )
                )
        return MegatronMappingRegistry(*mappings)
