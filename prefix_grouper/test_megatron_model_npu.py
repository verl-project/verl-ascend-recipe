# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=1 test_megatron_model_npu.py.

Use two ranks and --tp 2 to exercise sharded Q/K/V heads and vocabulary loss.
No checkpoint download is needed. The reference uses the installed Megatron
attention with one causal forward per trajectory. All parameters are reused
unchanged when the production Ascend PrefixGrouper adapter is installed.
"""

import argparse
import os
import tempfile

import torch
import torch.nn.functional as F


def make_model(tp, backend, recompute):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers import Qwen3Config

    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    qwen = Qwen3Config(
        vocab_size=256,
        num_hidden_layers=2,
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        architectures=["Qwen3ForCausalLM"],
        tie_word_embeddings=False,
    )
    # Construct the real engine to initialize its model-parallel groups, RNG
    # tracker and NPU compatibility patches. Only HF metadata is loaded here.
    with tempfile.TemporaryDirectory() as config_dir:
        qwen.save_pretrained(config_dir)
        engine = MegatronEngineWithLMHead(
            HFModelConfig(path=config_dir, load_tokenizer=False),
            McoreEngineConfig(
                tensor_model_parallel_size=tp,
                sequence_parallel=False,
                use_dynamic_bsz=False,
                use_remove_padding=False,
                use_prefix_grouper=True,
                seed=422,
            ),
            McoreOptimizerConfig(),
            CheckpointConfig(),
        )
    cfg = TransformerConfig(
        num_layers=qwen.num_hidden_layers,
        hidden_size=qwen.hidden_size,
        ffn_hidden_size=qwen.intermediate_size,
        num_attention_heads=qwen.num_attention_heads,
        num_query_groups=qwen.num_key_value_heads,
        kv_channels=qwen.head_dim,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=qwen.rms_norm_eps,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
        tensor_model_parallel_size=tp,
        attention_backend=AttnBackend.flash if backend == "te" else AttnBackend.local,
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
    )
    if backend == "te":
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    else:
        spec = get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")
    model = GPTModel(
        cfg,
        spec,
        qwen.vocab_size,
        qwen.max_position_embeddings,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=1_000_000,
    ).to(device=torch.device("npu", torch.npu.current_device()), dtype=torch.bfloat16)
    return engine, model.train()


def loss_for_branch(logits, tokens, len_prompt, branch):
    # This is the original verl vocabulary-parallel CE. A TP rank owns only a
    # vocabulary shard. Each branch gives the shared final prompt a different label.
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    targets = tokens[len_prompt:]
    preds = logits[len_prompt - 1 : -1].float().contiguous()
    log_probs = vocab_parallel_log_probs_from_logits(preds, targets)
    weights = torch.arange(1, targets.numel() + 1, device=log_probs.device, dtype=torch.float32)
    return -(log_probs * weights * (branch + 1)).mean()


def capture_gradients(model):
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                raise AssertionError(f"Missing gradient on {name}")
            gradients[name] = param.grad.detach().float().cpu()
    return gradients


def check_multichunk_backward(engine, model):
    """Replay the FP32 logits/temperature path that saves the divisor for backward."""
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    device = next(model.parameters()).device
    prompt = torch.arange(1, 73, device=device).repeat(2, 1)
    responses = torch.tensor([[101, 102, 103, 104, 105], [115, 116, 117, 118, 119]], device=device)
    ids = torch.cat((prompt, responses), dim=1)
    batch = tu.get_tensordict(
        {
            "input_ids": ids,
            "prompts": prompt,
            "responses": responses,
            "attention_mask": torch.ones_like(ids),
            "response_mask": torch.ones_like(responses),
            "position_ids": torch.arange(ids.shape[1], device=device).expand_as(ids),
            "temperature": torch.tensor([0.75, 1.25], device=device, dtype=torch.float32),
            "uid": ["group", "group"],
        }
    )
    batch = left_right_2_no_padding(batch)
    assert batch["input_ids"].values().numel() == 154  # Scoring runs chunks of 128 and 26.
    model.zero_grad(set_to_none=True)
    # Warm the exact dtype that exposed the bug in the 8B run. BF16 logits would
    # allocate a separate divisor with .to(logits.dtype) and hide the stale view.
    handle = model.register_forward_hook(lambda module, args, logits: logits.float())
    try:
        output, _ = engine.forward_step(
            iter([batch]),
            model,
            logits_processor_func=None,
            postprocess_micro_batch_func=engine.postprocess_micro_batch_func,
        )
        response_scores = no_padding_2_padding(output["log_probs"], batch)
        assert response_scores.shape == responses.shape and response_scores.dtype == torch.float32
        assert torch.isfinite(response_scores).all().item()
        loss = -response_scores.mean()
        loss.backward()
    finally:
        handle.remove()
    grads = capture_gradients(model)
    if not all(grad.isfinite().all().item() for grad in grads.values()) or not any(
        torch.count_nonzero(grad).item() for grad in grads.values()
    ):
        raise AssertionError("Expected finite, nonzero gradients after both logits chunks")
    print(
        f"rank={torch.distributed.get_rank()} PASS FP32 temperature multichunk backward: "
        f"154 active tokens, 2 chunks, 2 temperatures, {len(grads)} parameter gradients",
        flush=True,
    )


def run(tp, backend, recompute):
    from verl.models.mcore.megatron_prefix_grouper import pack_shared_prefix
    from verl.models.mcore.prefix_grouper_forward import _check_model, _grouped_logits
    from verl.models.mcore.util import _build_npu_attn_mask

    engine, model = make_model(tp, backend, recompute)
    device = next(model.parameters()).device
    prompt_width = 8
    ids = torch.tensor(
        [
            [0, 0, 3, 5, 7, 9, 11, 13, 31, 32, 33, 34],
            [0, 0, 0, 0, 0, 15, 17, 19, 51, 52, 0, 0],
            [0, 0, 3, 5, 7, 9, 11, 13, 43, 45, 47, 0],
        ],
        device=device,
    )
    valid = ids.ne(0)
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    packed = pack_shared_prefix(ids, valid, positions, ids.shape[1] - prompt_width, ["A", "B", "A"])
    if packed.original_token_count - packed.input_ids.numel() != 6:
        raise AssertionError("Expected the six-token A prompt to be computed once")
    rows = [row[mask] for row, mask in zip(ids, valid, strict=True)]
    prompt_lengths = valid[:, :prompt_width].sum(-1).tolist()
    cases = [(0,), (1,), (2,), (0, 1, 2)]
    ref_logits, ref_grads = {}, {}
    for case in cases:
        model.zero_grad(set_to_none=True)
        losses = []
        for branch in case:
            tokens = rows[branch]
            mask = _build_npu_attn_mask(torch.ones_like(tokens[None], dtype=torch.bool))
            output = model(
                input_ids=tokens[None],
                position_ids=torch.arange(len(tokens), device=device)[None],
                attention_mask=mask,
            )[0]
            ref_logits[branch] = output.detach().float().cpu()
            losses.append(loss_for_branch(output, tokens, prompt_lengths[branch], branch))
        torch.stack(losses).mean().backward()
        ref_grads[case] = capture_gradients(model)

    # Keep the identical parameters and transition only the attention and RoPE
    # dispatch. Check each branch by itself and then the combined gradient.
    _check_model(engine, model)
    rank = torch.distributed.get_rank()
    for case in cases:
        model.zero_grad(set_to_none=True)
        state = model._prefix_grouper_attention
        previous_calls = state["calls"]
        output = _grouped_logits(model, model, packed)[0]
        assert state["calls"] - previous_calls == len(model.decoder.layers)
        losses = []
        for branch in case:
            index = packed.restore_indices[branch, valid[branch]]
            values = output.index_select(0, index)
            # Compare every active position, every logit in this TP shard.
            torch.testing.assert_close(values.detach().float().cpu(), ref_logits[branch], atol=0.04, rtol=0.04)
            losses.append(loss_for_branch(values, rows[branch], prompt_lengths[branch], branch))
        torch.stack(losses).mean().backward()
        recomputations = state["calls"] - previous_calls - len(model.decoder.layers)
        if recomputations != (len(model.decoder.layers) if recompute else 0):
            raise AssertionError(f"Unexpected number of masked attention recomputations: {recomputations}")
        actual = capture_gradients(model)
        reference = ref_grads.pop(case)
        assert actual.keys() == reference.keys()
        worst = 0.0
        for name, expected in reference.items():
            grad = actual[name]
            torch.testing.assert_close(
                grad, expected, atol=0.035, rtol=0.06, msg=lambda msg, name=name: f"{name}: {msg}"
            )
            # Also test tensorwise relative L2 so small parameter gradients cannot
            # disappear behind an absolute tolerance. BF16 accumulation can differ.
            relative_l2 = (grad - expected).norm().item() / max(expected.norm().item(), 1e-7)
            if not relative_l2 < 0.065:
                raise AssertionError(f"{name}: relative gradient L2 error {relative_l2:.6g}")
            worst = max(worst, relative_l2)
        print(
            f"rank={rank} branches={case} parameters={len(actual)} max_grad_relative_l2={worst:.6g} "
            f"recomputed_layers={recomputations}",
            flush=True,
        )
    check_multichunk_backward(engine, model)
    torch.distributed.barrier()
    if rank == 0:
        print(f"PASS actual Megatron GPTModel, backend={backend}, TP={tp}, BF16, recompute={recompute}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--backend", choices=("te", "local"), default="te")
    parser.add_argument("--no-recompute", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 0)) != args.tp:
        parser.error("Launch with torchrun --standalone --nproc-per-node equal to --tp")
    # Preserve runtime registration order before importing Megatron-Core.
    # isort: off
    import torch_npu  # noqa: F401
    import megatron_adaptor  # noqa: F401  # install Ascend patches before Megatron-Core imports

    # isort: on
    from megatron.core import parallel_state

    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    initialize_global_process_group(timeout_second=180)
    try:
        run(args.tp, args.backend, not args.no_recompute)
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        destroy_global_process_group()


if __name__ == "__main__":
    main()
