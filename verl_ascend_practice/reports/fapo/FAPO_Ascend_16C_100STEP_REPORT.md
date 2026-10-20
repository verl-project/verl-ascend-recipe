# FAPO on Ascend: 16-card 100-step validation

Last updated: 10/19/2026

This report records the validation run for issue #16:
Qwen2.5-Math-7B + FAPO-GenRM-4B, Megatron training, vLLM-Ascend rollout,
and a reward model in a separate resource pool on the same 16-card Ascend 910C node.

The paths below describe the original deployment. See `../../../fapo/README.md` for
the mapping to this repository's scripts, reward function and bundled core patch.
This report is historical evidence, not a rerun of the relocated entry points.

## Implemented changes

| Area | File or component | Change | Reason |
|---|---|---|---|
| FAPO launcher | `examples/ascend_extras/fapo_trainer/run_qwen2_5_math_7b_megatron.sh` | Add a Megatron + vLLM-Ascend launcher for Qwen2.5-Math-7B and FAPO-GenRM-4B | Provide the issue-specific training entry point |
| Resource layout | FAPO launcher | Configure separate Actor/Ref and GenRM resource pools; expose device counts, TP, memory, recompute, and offload controls | Fit the actor, rollout, and 4B GenRM on Ascend |
| Reward wiring | FAPO launcher and `recipe/fapo` | Use the `reward.*` configuration namespace and connect the GenRM through `/v1/completions` | Ensure the reward loop actually invokes the vLLM-backed GenRM |
| GenRM observability | `recipe/fapo/reward_fn_reasoning.py` | Log every GenRM call, response status, response length, and `is_flawed_positive` verdict | Make reward-model execution auditable from training logs |
| Transformer compatibility | `recipe/fapo/reward_fn_reasoning.py` | Unwrap `BatchEncoding` returned by newer Transformers versions | Keep prompt IDs JSON serializable |
| GenRM memory | FAPO launcher and `recipe/fapo` | Cap GenRM generation length and vLLM request concurrency | Avoid invalid requests and reduce the reward-model prefill spike |
| Megatron/NPU conversion | `scripts/converter_hf_to_mcore.py` | Select the distributed backend through `get_nccl_backend()` | Allow HF-to-Megatron conversion on NPU |
| Conversion verification | `scripts/verify_hf_to_mcore.py` | Add elementwise HF-to-Megatron checkpoint verification with QKV/MLP/other buckets | Detect weight-layout errors that an in-memory conversion check can miss |
| Conversion tests | `tests/utils/test_mcore_qkv_unfuse_on_cpu.py` | Cover interleaved GQA QKV round trips and invalid geometry | Protect Qwen2.5-Math-7B conversion |
| NPU vLLM patching | `verl/utils/vllm/npu_vllm_patch.py` | Centralize vLLM version selection and support 0.13+ API variants, including 0.18 | Prevent silent gaps in Ascend patch activation |
| NPU vLLM tests | `tests/utils/test_npu_vllm_patch_version_on_cpu.py` | Test version ranges, local version suffixes, and partially initialized `vllm` modules | Cover startup-time version-gating regressions |
| Device API sanity | `.pre-commit-config.yaml`, `tests/special_sanity/check_device_api_usage.py` | Scan `scripts/` as well as `verl/`, with multi-directory support | Keep the new conversion utility covered by the device API check |
| Context length | FAPO launcher | Derive Actor `max_position_embeddings` from the prompt/response budget | Match Qwen2.5-Math-7B's 4096-token configuration and avoid vLLM startup failure |

## 16-card topology

| Role | Devices | Parallelism |
|---|---:|---|
| Actor / Ref | 12 | Megatron TP=4, PP=1, CP=1 |
| Rollout | Actor-side devices | vLLM-Ascend TP=4 |
| FAPO-GenRM-4B | 4 | vLLM-Ascend TP=2 |

## Validation result

Full logs, all 100 step metrics and regenerated curves are linked in
[the report index](README.md). The first/last 10-step mean reward is
-1.065026 / -0.447369. Time-weighted throughput across the full run is
119.435 tokens/s per allocated NPU (1910.966 tokens/s aggregated over 16 NPUs).
Some steps are below 100 tokens/s/NPU; the full-run mean is not a per-step guarantee.

![Training curves](assets/training_curves.png)

| Item | Result |
|---|---|
| PyTorchJob | `1280184` |
| Hardware | 1 node, 16 x Ascend 910C |
| Training steps | 100/100 |
| Training-loop time (excludes initialization and scheduling) | 2 hours 2 minutes 5 seconds |
| Average step time | about 73 seconds |
| Step 100 throughput | 98.3 tokens/s/NPU |
| Step 100 reward mean | -0.414 |
| Step 100 entropy | 0.210 |
| Step 100 rollout/training Pearson correlation | 0.9991 |
| Step 100 response length mean | 812 tokens |
| Step 100 aborted ratio | 0 |
| Actor max allocated memory | about 27.35 GiB |
| GenRM calls | Observed throughout the run, including both flawed-positive outcomes |
| Job result | Succeeded; no OOM or training crash |

The run used `TEST_FREQ=-1` and `SAVE_FREQ=-1`, disabling validation and
checkpoint saving. The earlier 8-card report attributes its step-25 OOM to
validation and memory fragmentation, not checkpoint saving.
The preparation script used 256 training rows as a placeholder validation
file; no validation was executed, so this is not an evaluation result.
Use the dataset's real test split for future evaluation.
No trained checkpoint was saved. This validates the training path only.
